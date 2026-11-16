"""Off-robot tests of the recorder's episode store and video writer."""
import datetime
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from franka_teleop_recorder.episodes import EpisodeStore, is_episode_id, slugify  # noqa: E402
from franka_teleop_recorder.video import (VideoWriter, color_command, depth_command,  # noqa: E402
                                          make_side_by_side)


class EpisodeStoreTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = EpisodeStore(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def _finish(self, episode_id):
        meta = self.store.read(episode_id)
        meta['status'] = 'ready'
        self.store.write(episode_id, meta)

    def test_default_names_and_ids(self):
        now = datetime.datetime(2026, 9, 28, 14, 5, 9)
        first, path, meta = self.store.create('', 'pick cube', now=now)
        self.assertEqual(first, '20260928_140509_task1_episode_001')
        self.assertEqual(meta['task'], 'pick cube')
        self.assertEqual(meta['status'], 'recording')
        self.assertEqual(self.store.create('', 'stack', now=now)[0], '20260928_140509_task2_episode_001')
        self.assertEqual(self.store.default_name('pick cube'), 'task1_episode_002')
        self.assertEqual(self.store.default_name('new task'), 'task3_episode_001')
        self.assertEqual(self.store.tasks(), {'pick cube': 1, 'stack': 2})  # previews assign no number
        self.assertEqual(self.store.create('', '', now=now)[0], '20260928_140509_episode_001')
        # after a deletion the freed number is used again
        self._finish(first)
        self.store.move_to_trash(first)
        self.assertEqual(self.store.default_name('pick cube'), 'task1_episode_001')
        later, _, _ = self.store.create('', 'pick cube', now=now)
        self._finish(later)
        self.assertEqual(self.store.default_name('pick cube'), 'task1_episode_002')

    def test_lost_task_list_is_rebuilt_from_the_episodes(self):
        now = datetime.datetime(2026, 9, 28, 14, 5, 9)
        for task in ('stack', 'slide', 'pick'):
            self._finish(self.store.create('', task, now=now)[0])
        (self.store.root / 'tasks.json').unlink()  # e.g. deleted with old data
        self.assertEqual(self.store.add_task('pick'), 3)  # its episodes are task3_*
        self.assertEqual(self.store.add_task('unstack'), 4)  # not 1 or 2: those are on disk
        self.assertEqual(self.store.default_name('slide'), 'task2_episode_002')

    def test_new_task_renumbers_default_names_only(self):
        now = datetime.datetime(2026, 9, 28, 14, 5, 9)
        auto, _, _ = self.store.create('', 'pick cube', now=now)
        named, _, _ = self.store.create('my try', 'pick cube', now=now)
        for episode_id in (auto, named):
            self._finish(episode_id)
        self.assertEqual(self.store.update(auto, task='stack'), '20260928_140509_task2_episode_001')
        self.assertEqual(self.store.update(named, task='stack'), named)

    def test_rename_moves_folder_and_keeps_timestamp(self):
        episode_id, _, _ = self.store.create('first try', '', now=datetime.datetime(2026, 1, 2, 3, 4, 5))
        with self.assertRaises(RuntimeError):  # still recording
            self.store.update(episode_id, name='x')
        self._finish(episode_id)
        new_id = self.store.update(episode_id, name='Red cube / bowl!', success=True)
        self.assertEqual(new_id, '20260102_030405_Red_cube_bowl')
        self.assertFalse((self.store.root / episode_id).exists())
        meta = self.store.read(new_id)
        self.assertEqual(meta['name'], 'Red cube / bowl!')
        self.assertTrue(meta['success'])
        self.store.update(new_id, clear_success=True)
        self.assertIsNone(self.store.read(new_id)['success'])

    def test_trash(self):
        episode_id, _, _ = self.store.create('a', '')
        self._finish(episode_id)
        self.store.move_to_trash(episode_id)
        self.assertEqual(self.store.list(), [])
        self.assertTrue((self.store.trash / episode_id / 'meta.json').is_file())

    def test_recover_interrupted(self):
        episode_id, _, _ = self.store.create('cut off', '')
        self.assertEqual(self.store.recover_interrupted(), [episode_id])
        meta = self.store.read(episode_id)
        self.assertEqual(meta['status'], 'interrupted')
        self.store.update(episode_id, success=False)  # editable again
        self.assertEqual(self.store.recover_interrupted(), [])

    def test_rejects_path_tricks(self):
        self.assertFalse(is_episode_id('../etc'))
        self.assertFalse(is_episode_id('20260101_000000/../x'))
        with self.assertRaises(KeyError):
            self.store.path('..')
        self.assertEqual(slugify('  a b/c  '), 'a_b_c')


class VideoWriterTest(unittest.TestCase):
    def test_color_depth_and_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            width, height = 64, 48
            color = VideoWriter(color_command(directory / 'c.mp4', width, height, 30, 'rgb24', 18, 'veryfast'),
                                directory / 'c.csv')
            depth = VideoWriter(depth_command(directory / 'd.mkv', width, height, 30, 'gray16le'),
                                directory / 'd.csv')
            depth_frames = []
            for index in range(10):
                color.put(bytes([index * 20]) * (width * height * 3), 1000 + index, 2000 + index)
                frame = bytes((value * 7 + index) % 256 for value in range(width * height * 2))
                depth_frames.append(frame)
                depth.put(frame, 1000 + index, 2000 + index)
            self.assertIsNone(color.close())
            self.assertIsNone(depth.close())
            self.assertEqual((color.frames, depth.frames), (10, 10))
            self.assertEqual(color.first_stamp_ns, 1000)
            self.assertEqual(color.last_stamp_ns, 1009)
            self.assertEqual(len((directory / 'c.csv').read_text().splitlines()), 11)
            # Depth must round-trip bit-exactly (lossless).
            decoded = subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', str(directory / 'd.mkv'),
                                      '-f', 'rawvideo', '-pix_fmt', 'gray16le', '-'],
                                     capture_output=True, check=True).stdout
            self.assertEqual(decoded, b''.join(depth_frames))
            make_side_by_side([directory / 'c.mp4', directory / 'c.mp4'], directory / 'r.mp4', height=48)
            probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                                    'stream=width,height', '-of', 'json', str(directory / 'r.mp4')],
                                   capture_output=True, text=True, check=True)
            stream = json.loads(probe.stdout)['streams'][0]
            self.assertEqual((stream['width'], stream['height']), (128, 48))


if __name__ == '__main__':
    unittest.main()
