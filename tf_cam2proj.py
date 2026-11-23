import cv2
import numpy as np

import argparse, glob

def main():
  parser = argparse.ArgumentParser(description='Get projector image coordinates from camera image coordinates and depth values')
  parser.add_argument('cam_x', type=int, help='camera pixel x coordinate')
  parser.add_argument('cam_y', type=int, help='camera pixel y coordinate')
  parser.add_argument('--cam_depth', type=float, help='depth value at camera pixel (must have the same units as the T vector)')
  parser.add_argument('--camera-index', type=int, default=0,
                          help='OpenCV camera index (default: 0)')
  parser.add_argument('calibration', type=str, default=str(), help='camera internal parameter json file')
  parser.add_argument('--proj2cam', action='store_true', default=False, help='Invert the extrinsic transformation to get camera coordinates from projector coordinates')

  args = parser.parse_args()
  
  cam_x = args.cam_x
  cam_y = args.cam_y

  # camera = cv2.VideoCapture(args.camera_index)
  # if not camera.isOpened():
  #     camera.release()
  #     parser.error(f'could not open camera index {args.camera_index}')
  #
  # camera.set(cv2.CAP_PROP_BUFFERSIZE, 1) #no repeated frames
  # ok, frame = camera.read()
  # if not ok:
  #   print('Could not read a frame from the camera.')
  #   return 1
  # Match the exact resolution used during your calibration
  first_img = sorted(glob.glob('./capture_*'))[0] + '/graycode_00.png'
  frame = cv2.imread(first_img, cv2.IMREAD_COLOR)
  if frame is None:
    raise ValueError("Could not read image to determine camera resolution")
  
  print("Cam resolution is", frame.shape[0], frame.shape[1])

  auto_depth = False
  if args.cam_depth is not None:
    cam_depth = args.cam_depth
  else:
    print("Trying to obtain depth automatically trough Realsense camera...")
    try:
      import depth
    except ImportError:
      print("Could not import depth utils!")
      exit(1)

    pipeline, align, config = depth.configureDepthCam(frame.shape[1], frame.shape[0], 30)
    auto_depth = True
    profile = pipeline.start(config)
    
    print("Looping until valid depth value")
    cam_depth = depth.getDepth(cam_x, cam_y, pipeline, align)
    while cam_depth is None:
      cam_depth = depth.getDepth(cam_x, cam_y, pipeline, align)

    if cam_depth is None:
      print("Error obtaining depth through camera")
      pipeline.stop()
      exit(1)

    print(f"Obtained depth: {cam_depth}m")

  print("Reading calibration...")

  fs = cv2.FileStorage(args.calibration, cv2.FILE_STORAGE_READ)
  if not fs.isOpened():
    parser.error(f"Could not open calibration file: {args.calibration}")

  R = fs.getNode('rotation').mat()
  T = np.asarray(fs.getNode('translation').mat(), dtype=np.float64)
  K_cam = fs.getNode('cam_int').mat()
  D_cam = fs.getNode('cam_dist').mat()
  K_proj = fs.getNode('proj_int').mat()
  D_proj = fs.getNode('proj_dist').mat()
  fs.release()

  if R is None:
    parser.error("Missing 'rotation' matrix in calibration file")
  
  if T is None:
    parser.error("Missing 'translation' vector in calibration file")
  
  # Set to True if you passed (Projector, Camera) to stereoCalibrate instead of (Camera, Projector)
  if args.proj2cam:
    R = R.T
    T = -R @ T

  print("R shape:", R.shape)
  print("T shape:", T.shape)
  # R = np.eye(3, dtype=np.float32)
  # T = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)

  # K_cam = np.array(
  #   [[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]],
  #   dtype=np.float32,
  # )
  # D_cam = np.zeros((5, 1), dtype=np.float32)
  # K_proj = np.array(
  #   [[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]],
  #   dtype=np.float32,
  # )
  # D_proj = np.zeros((5, 1), dtype=np.float32)

  print('  Camera intrinsic parameters :')
  printNumpyWithIndent(K_cam, '    ')
  print('  Camera distortion parameters :')
  printNumpyWithIndent(D_cam, '    ')
  print('  Projector intrinsic parameters :')
  printNumpyWithIndent(K_proj, '    ')
  print('  Projector distortion parameters :')
  printNumpyWithIndent(D_proj, '    ')
  print('  Rotation matrix / translation vector from camera to projector')
  print('  (they translate points from camera coord to projector coord) :')
  printNumpyWithIndent(R, '    ')
  printNumpyWithIndent(T, '    ')
  print()

  proj_x, proj_y = cam2proj(cam_x, cam_y, cam_depth, R, T, K_cam, D_cam, K_proj, D_proj)
  print(f'Camera pixel ({cam_x}, {cam_y}) with depth {cam_depth} maps to projector pixel ({proj_x}, {proj_y})')
  print(f"Inside projector's projection plane?: {0 <= proj_x < 1920 and 0 <= proj_y < 1080}")
  
  # Draw target on camera feed for visual feedback
  # camera = cv2.VideoCapture(args.camera_index)
  # if not camera.isOpened():
  #   camera.release()
  #   parser.error(f'could not open camera index {args.camera_index}')

  try:
    mapped_proj_x = round(proj_x/1920*frame.shape[1])
    mapped_proj_y = round(proj_y/1080*frame.shape[0])

    cv2.circle(frame, (cam_x, cam_y), 5, (0, 255, 0), -1)
    cv2.circle(frame, (mapped_proj_x, mapped_proj_y), 5, (0, 0, 255), -1)

    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty("Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Camera", frame)
    while True:
      key = cv2.waitKey(0) & 0xFF
      if key in (27, ord('q')):
          print('Capture stopped.')
          return 0
      if key in (32, 13):
          break

  finally:
    # if camera is not None:
    #   camera.release()
    if auto_depth:
      pipeline.stop()

    cv2.destroyAllWindows()

def printNumpyWithIndent(tar, indentchar):
  print(indentchar + str(tar).replace('\n', '\n' + indentchar))

def deproject_px_to_3d(pixel_u, pixel_v, depth_Z, K_cam, D_cam):
  """
  Deprojects a pixel coordinate (u, v) with a given depth value Z into 3D camera coordinates (X, Y, Z).
  
  Parameters:
  - pixel_u: The u-coordinate of the pixel.
  - pixel_v: The v-coordinate of the pixel.
  - depth_Z: The depth value at the pixel (in meters).
  - K_cam: The camera intrinsic matrix (3x3).
  - D_cam: The camera distortion coefficients (1xN).
  
  Returns:
  - A tuple (X_cam, Y_cam, Z_cam) representing the 3D coordinates in the camera frame.
  """
  if depth_Z <= 0:
    raise ValueError("Depth value must be positive and non-zero.")
  # cv2.undistortPoints expects an array of shape (N, 1, 2)
  pixel_coords = np.array([[[pixel_u, pixel_v]]], dtype=np.float64)

  # This function returns normalized image coordinates (x', y') where fx=fy=1 and cx=cy=0
  normalized_coords = cv2.undistortPoints(pixel_coords, K_cam, D_cam)

  # Extract the normalized coordinates
  x_norm, y_norm = normalized_coords[0, 0]

  # Since cv2.undistortPoints already factored out fx, fy, cx, cy,
  # the 3D coordinates are simply the normalized coordinates multiplied by depth.
  X_cam = x_norm * depth_Z
  Y_cam = y_norm * depth_Z
  Z_cam = depth_Z
  print("3D camera coordinates", X_cam, Y_cam, Z_cam)
  return (X_cam, Y_cam, Z_cam)

def project_3d_to_px(X, Y, Z, RCam2Proj, TCam2Proj, K_proj, D_proj):
  # if RCam2Proj.shape == (3, 3):
  #   rvec, _ = cv2.Rodrigues(RCam2Proj)
  # else:
  #   rvec = RCam2Proj
  
  point_3d = np.array([[[X, Y, Z]]], np.float64)
  proj_pts, _ = cv2.projectPoints(point_3d, RCam2Proj, TCam2Proj, K_proj, D_proj)
  x1, y1 = proj_pts[0][0]

  return (int(round(x1)), int(round(y1)))

def cam2proj(px_x, px_y, depth_Z, RCam2Proj, TCam2Proj, K_cam, D_cam, K_proj, D_proj):
  Xc, Yc, Zc = deproject_px_to_3d(px_x, px_y, depth_Z, K_cam, D_cam)
  return project_3d_to_px(Xc, Yc, Zc, RCam2Proj, TCam2Proj, K_proj, D_proj)

if __name__ == '__main__':
  main()
