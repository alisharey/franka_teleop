"""Frame-by-frame video writing through ffmpeg, off the ROS callback threads."""

import csv
import queue
import subprocess
import tempfile
import threading

# ROS image encoding -> (ffmpeg input pixel format, bytes per pixel).
PIXEL_FORMATS = {
    'rgb8': ('rgb24', 3),
    'bgr8': ('bgr24', 3),
    '16UC1': ('gray16le', 2),
    'mono16': ('gray16le', 2),
}


def color_command(path, width, height, fps, pixel_format, crf, preset):
    # Fragmented MP4: stays playable even if recording is cut off by a crash.
    return ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
            '-f', 'rawvideo', '-pix_fmt', pixel_format, '-s', '%dx%d' % (width, height),
            '-r', str(fps), '-i', '-',
            '-c:v', 'libx264', '-preset', preset, '-crf', str(crf), '-pix_fmt', 'yuv420p',
            '-g', str(fps), '-movflags', '+frag_keyframe+empty_moov+default_base_moof',
            str(path)]


def depth_command(path, width, height, fps, pixel_format):
    # FFV1 is lossless and keeps all 16 bits (millimetres from the D435i).
    return ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
            '-f', 'rawvideo', '-pix_fmt', pixel_format, '-s', '%dx%d' % (width, height),
            '-r', str(fps), '-i', '-',
            '-c:v', 'ffv1', '-level', '3', '-g', '1', '-slices', '4', '-slicecrc', '1',
            str(path)]


class VideoWriter:
    """Writes raw frames to an ffmpeg process and their timestamps to a CSV.

    put() never blocks: a full queue drops the frame and counts it, so a slow
    disk can never stall the ROS callbacks.
    """

    def __init__(self, command, frames_csv, max_queue=90):
        self.command = command
        self.frames = 0
        self.dropped = 0
        self.first_stamp_ns = None
        self.last_stamp_ns = None
        self.error = None
        self._queue = queue.Queue(maxsize=max_queue)
        self._csv_file = open(frames_csv, 'w', newline='')
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow(['frame', 'stamp_ns', 'receive_ns'])
        self._stderr = tempfile.TemporaryFile()  # a file, so ffmpeg can never block on it
        # Own session: a terminal Ctrl-C reaches only the recorder, which then
        # closes ffmpeg's input so it finishes the file properly.
        self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=self._stderr,
                                         start_new_session=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def put(self, data, stamp_ns, receive_ns):
        try:
            self._queue.put_nowait((data, stamp_ns, receive_ns))
        except queue.Full:
            self.dropped += 1

    def _run(self):
        while True:
            item = self._queue.get()
            if item is None:
                break
            data, stamp_ns, receive_ns = item
            if self.error is not None:
                continue
            try:
                self._process.stdin.write(data)
            except (BrokenPipeError, OSError) as exc:
                self.error = 'ffmpeg stopped: %s' % exc
                continue
            self._csv.writerow([self.frames, stamp_ns, receive_ns])
            if self.first_stamp_ns is None:
                self.first_stamp_ns = stamp_ns
            self.last_stamp_ns = stamp_ns
            self.frames += 1

    def close(self):
        """Flushes queued frames and waits for ffmpeg to finish the file."""
        self._queue.put(None)
        self._thread.join()
        self._csv_file.close()
        try:
            self._process.stdin.close()
        except OSError:
            pass
        code = self._process.wait()
        self._stderr.seek(0)
        stderr = self._stderr.read().decode(errors='replace').strip()
        self._stderr.close()
        if code != 0 and self.error is None:
            self.error = 'ffmpeg exited with %d: %s' % (code, stderr[-500:])
        return self.error


def make_side_by_side(inputs, output, height=480):
    """Replay video: all color videos next to each other at a common height."""
    inputs = [str(p) for p in inputs]
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y']
    for path in inputs:
        command += ['-i', path]
    if len(inputs) == 1:
        graph = '[0:v]scale=-2:%d[v]' % height
    else:
        scaled = ''.join('[%d:v]scale=-2:%d[s%d];' % (i, height, i) for i in range(len(inputs)))
        graph = scaled + ''.join('[s%d]' % i for i in range(len(inputs))) + \
            'hstack=inputs=%d:shortest=0[v]' % len(inputs)
    command += ['-filter_complex', graph, '-map', '[v]', '-c:v', 'libx264', '-preset', 'veryfast',
                '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)]
    result = subprocess.run(command, capture_output=True, text=True, start_new_session=True)
    if result.returncode != 0:
        raise RuntimeError('side-by-side video failed: %s' % result.stderr[-500:])
