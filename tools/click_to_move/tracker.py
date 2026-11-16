"""EdgeTAM (Hugging Face transformers, streaming video mode) tracking one object in one camera, started from a
clicked point. Runs in a background thread on the camera's newest frames; old frames and outputs are dropped so
memory stays flat on a live stream (the session keeps every frame otherwise).

Weights: yonigozlan/EdgeTAM-hf (56 MB), cached in data/models/hf (HF_HOME, set by click_to_move.sh).
"""
import threading
import time
import traceback

import torch

MODEL_ID = 'yonigozlan/EdgeTAM-hf'
KEEP_FRAMES = 16  # past frame outputs kept (EdgeTAM attends to the prompted frame + the last 7)


class Tracker:
    def __init__(self, frame_source, device='cuda'):
        """frame_source(cam) -> (color RGB uint8, depth uint16 mm, K, stamp) of the newest frame, or Nones."""
        self.frame_source = frame_source
        self.device = device
        self.model = self.processor = None
        self.error = None
        self.lock = threading.Lock()
        self.cam = None
        self.session = None
        self.pending = None   # (cam, (u, v)) to start from on the next frame
        self.result = None    # dict: cam, mask (bool HxW), color, depth, K, stamp, time
        self.ms = None        # EdgeTAM's time per frame
        self.fps = None       # tracked frames per second (also limited by the camera's rate)
        self.running = True
        threading.Thread(target=self._load, daemon=True).start()
        threading.Thread(target=self._loop, daemon=True).start()

    @property
    def ready(self):
        return self.model is not None

    def _load(self):
        try:
            from transformers import EdgeTamVideoModel, Sam2VideoProcessor
            self.processor = Sam2VideoProcessor.from_pretrained(MODEL_ID)
            self.model = EdgeTamVideoModel.from_pretrained(MODEL_ID).to(self.device).eval()
        except Exception as error:  # reported on the page
            self.error = 'EdgeTAM did not load: %s' % error

    def start(self, cam, u, v):
        if not self.ready:
            raise RuntimeError(self.error or 'EdgeTAM is still loading (a few seconds after start)')
        with self.lock:
            self.pending = (cam, (float(u), float(v)))
            self.result = None

    def clear(self):
        with self.lock:
            self.pending, self.cam, self.session, self.result = None, None, None, None
            self.ms = self.fps = None

    def latest(self):
        with self.lock:
            return self.result

    @staticmethod
    def _prune(s, frame_idx):
        for frames in s.output_dict_per_obj.values():
            old = [k for k in frames['non_cond_frame_outputs'] if k < frame_idx - KEEP_FRAMES]
            for k in old:
                del frames['non_cond_frame_outputs'][k]
        # old frames' images (12 MB each) are emptied, not deleted: the session numbers each new frame by how many
        # it holds, and deleting them made it hand back an old frame's mask (the object seemed not to move)
        for k in [k for k in s.processed_frames if 0 < k < frame_idx - 1]:
            if s.processed_frames[k].numel() > 1:
                s.processed_frames[k] = s.processed_frames[k].new_zeros(1)
        for tracked in s.frames_tracked_per_obj.values():
            for k in [k for k in tracked if 0 < k < frame_idx - KEEP_FRAMES]:
                del tracked[k]

    def _loop(self):
        last = [None, None]  # stamp, time of the last tracked frame
        while self.running:
            try:
                if not self._step(last):
                    time.sleep(0.005)
            except Exception:  # keep tracking after a bad frame (the thread dying stopped all tracking)
                traceback.print_exc()
                time.sleep(0.2)

    @torch.inference_mode()
    def _step(self, last):
        """Tracks the newest frame; False when there was nothing to do."""
        with self.lock:
            pending, cam = self.pending, self.cam
        if pending is not None:
            cam = pending[0]
        if cam is None or not self.ready:
            time.sleep(0.05)
            return False
        color, depth, K, stamp = self.frame_source(cam)
        if color is None or depth is None or K is None or (stamp == last[0] and pending is None):
            return False
        last[0] = stamp
        started = time.monotonic()
        inputs = self.processor(images=color, device=self.device, return_tensors='pt')
        with self.lock:
            if self.pending is not None:  # new object: a new session prompted on this frame
                self.cam, (u, v) = self.pending
                self.pending = None
                self.session = self.processor.init_video_session(inference_device=self.device, dtype=torch.float32)
                self.processor.add_inputs_to_inference_session(
                    inference_session=self.session, frame_idx=0, obj_ids=1, input_points=[[[[u, v]]]],
                    input_labels=[[[1]]], original_size=inputs.original_sizes[0])
                last[1] = None
            session = self.session
        if session is None:
            return False
        out = self.model(inference_session=session, frame=inputs.pixel_values[0])
        mask = self.processor.post_process_masks([out.pred_masks], original_sizes=inputs.original_sizes,
                                                 binarize=True)[0][0, 0].cpu().numpy().astype(bool)
        self._prune(session, out.frame_idx)
        now = time.monotonic()
        with self.lock:
            if self.session is not session:  # cleared or reselected meanwhile
                return True
            self.ms = (now - started) * 1000
            if last[1] is not None:
                rate = 1.0 / max(1e-3, now - last[1])
                self.fps = rate if self.fps is None else 0.9 * self.fps + 0.1 * rate
            last[1] = now
            self.result = {'cam': self.cam, 'mask': mask, 'color': color, 'depth': depth, 'K': K,
                           'stamp': stamp, 'time': now}
        return True
