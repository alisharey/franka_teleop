"""Live AprilTag pose estimation with a RealSense camera.

Detects tagStandard41h12 markers in the color stream, estimates each tag's pose
in the camera frame from the camera's factory intrinsics, and shows the image
with the outline, the tag's axes (x red, y green, z blue) and the pose.

Camera frame (optical): x right, y down, z forward out of the lens.
Tag frame (AprilTag convention): origin at the tag center, x right, y down,
z pointing into the tag (away from the camera when the tag faces it).

Tag size: the edge length of the tag's black square, the AprilTag convention.
For tagStandard41h12 the printed pattern is 9 cells wide and the black square
is the middle 5 cells: the default 0.0556 m is for a pattern printed 100 mm wide
(measured on this tag with the depth camera: 54-55 mm). For another print, pass
--tag-size <printed width * 5/9>. The depth check in the overlay tells you if the
size is wrong: the pose distance and the depth camera then disagree by a constant ratio.

Keys: q or Esc quits, s saves the current image and pose to snapshots/.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pupil_apriltags
import pyrealsense2 as rs


def euler_xyz_degrees(rotation):
    """Roll (x), pitch (y), yaw (z) for R = Rz(yaw) Ry(pitch) Rx(roll), in degrees."""
    pitch = np.arcsin(-np.clip(rotation[2, 0], -1.0, 1.0))
    if abs(rotation[2, 0]) < 0.9999:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:  # gimbal lock
        roll = np.arctan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0
    return np.degrees([roll, pitch, yaw])


def start_camera(serial, width, height, fps, use_depth):
    pipeline = rs.pipeline()
    config = rs.config()
    if serial:
        config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    if use_depth:
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    profile = pipeline.start(config)
    color = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    camera_matrix = np.array([[color.fx, 0, color.ppx], [0, color.fy, color.ppy], [0, 0, 1]])
    distortion = np.array(color.coeffs[:5])
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale() if use_depth else None
    device = profile.get_device()
    name = '%s %s' % (device.get_info(rs.camera_info.name), device.get_info(rs.camera_info.serial_number))
    return pipeline, camera_matrix, distortion, depth_scale, name


def depth_at(depth_image, depth_scale, center, radius=3):
    """Median depth (m) in a small window around a pixel; None if no valid depth."""
    x, y = int(round(center[0])), int(round(center[1]))
    window = depth_image[max(0, y - radius):y + radius + 1, max(0, x - radius):x + radius + 1]
    valid = window[window > 0]
    return float(np.median(valid)) * depth_scale if valid.size else None


def draw(image, detection, camera_matrix, distortion, tag_size, depth_m):
    corners = detection.corners.astype(int)
    cv2.polylines(image, [corners.reshape(-1, 1, 2)], True, (0, 255, 255), 2)
    rvec, _ = cv2.Rodrigues(detection.pose_R)
    tvec = detection.pose_t.reshape(3)
    cv2.drawFrameAxes(image, camera_matrix, distortion, rvec, tvec, tag_size * 0.75, 3)
    x, y, z = tvec * 1000
    roll, pitch, yaw = euler_xyz_degrees(detection.pose_R)
    lines = ['id %d  distance %.0f mm' % (detection.tag_id, np.linalg.norm(tvec) * 1000),
             'x %.1f  y %.1f  z %.1f mm' % (x, y, z),
             'roll %.1f  pitch %.1f  yaw %.1f deg' % (roll, pitch, yaw),
             'reprojection err %.2e' % detection.pose_err]
    if depth_m is not None:
        lines.append('depth cam z %.1f mm (ratio %.3f)' % (depth_m * 1000, depth_m / tvec[2]))
    origin = (int(detection.center[0]) + 12, int(detection.center[1]) - 12 * len(lines))
    for index, text in enumerate(lines):
        position = (origin[0], origin[1] + 22 * index)
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def pose_dict(detection, depth_m):
    transform = np.eye(4)
    transform[:3, :3] = detection.pose_R
    transform[:3, 3] = detection.pose_t.reshape(3)
    return {
        'tag_id': int(detection.tag_id),
        'T_camera_tag': transform.round(6).tolist(),
        'position_m': detection.pose_t.reshape(3).round(6).tolist(),
        'roll_pitch_yaw_deg': euler_xyz_degrees(detection.pose_R).round(3).tolist(),
        'pose_err': float(detection.pose_err),
        'depth_camera_z_m': depth_m,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tag-size', type=float, default=0.0556,
                        help='black-square edge in metres (default 0.0556: a 100 mm printed tagStandard41h12)')
    parser.add_argument('--tag-id', type=int, default=4, help='tag to track; -1 shows every tag (default 4)')
    parser.add_argument('--family', default='tagStandard41h12')
    parser.add_argument('--serial', default='', help='RealSense serial number (default: the first camera)')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--no-depth', action='store_true', help='skip the depth cross-check')
    args = parser.parse_args()

    pipeline, camera_matrix, distortion, depth_scale, name = start_camera(
        args.serial, args.width, args.height, args.fps, not args.no_depth)
    align = rs.align(rs.stream.color) if depth_scale else None
    detector = pupil_apriltags.Detector(families=args.family, nthreads=4, quad_decimate=1.0,
                                        refine_edges=1, decode_sharpening=0.25)
    fx, fy, cx, cy = camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2]
    print('%s, %dx%d, fx %.1f fy %.1f cx %.1f cy %.1f; looking for %s id %s, size %.1f mm. q quits, s saves.'
          % (name, args.width, args.height, fx, fy, cx, cy, args.family,
             'any' if args.tag_id < 0 else args.tag_id, args.tag_size * 1000))
    window = 'AprilTag pose - %s' % name
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    snapshots = Path(__file__).resolve().parent / 'snapshots'
    last_print = 0.0
    frame_times = []
    try:
        while True:
            frames = pipeline.wait_for_frames()
            if align:
                frames = align.process(frames)
            color = frames.get_color_frame()
            if not color:
                continue
            image = np.asanyarray(color.get_data()).copy()
            depth_image = np.asanyarray(frames.get_depth_frame().get_data()) if align else None
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detections = [d for d in detector.detect(gray, estimate_tag_pose=True,
                                                     camera_params=(fx, fy, cx, cy), tag_size=args.tag_size)
                          if args.tag_id < 0 or d.tag_id == args.tag_id]
            poses = []
            for detection in detections:
                depth_m = depth_at(depth_image, depth_scale, detection.center) if depth_image is not None else None
                draw(image, detection, camera_matrix, distortion, args.tag_size, depth_m)
                poses.append(pose_dict(detection, depth_m))

            now = time.time()
            frame_times = [t for t in frame_times if now - t < 1.0] + [now]
            status = '%d fps  %s' % (len(frame_times), 'tag found' if poses else 'no tag %s' % (
                'id %d' % args.tag_id if args.tag_id >= 0 else ''))
            cv2.putText(image, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(image, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 220, 0) if poses else (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow(window, image)

            if poses and now - last_print > 0.2:
                last_print = now
                for pose in poses:
                    x, y, z = (np.array(pose['position_m']) * 1000).round(1)
                    r, p, w = pose['roll_pitch_yaw_deg']
                    print('id %d: xyz [%7.1f %7.1f %7.1f] mm  rpy [%6.1f %6.1f %6.1f] deg' % (
                        pose['tag_id'], x, y, z, r, p, w), flush=True)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord('s'):
                snapshots.mkdir(exist_ok=True)
                stamp = time.strftime('%Y%m%d_%H%M%S')
                cv2.imwrite(str(snapshots / ('%s.png' % stamp)), image)
                (snapshots / ('%s.json' % stamp)).write_text(json.dumps({
                    'camera': name, 'camera_matrix': camera_matrix.tolist(), 'distortion': distortion.tolist(),
                    'tag_size_m': args.tag_size, 'tags': poses}, indent=2))
                print('saved snapshots/%s.png and .json' % stamp)
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
