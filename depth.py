import pyrealsense2 as rs
from typing import Union
# Having trouble with Pylance? Do these:
# pip install pybind11-stubgen
# pybind11-stubgen pyrealsense2 -o ./typings (I put this folder inside .vscode)
# You can now uninstall:
# pip uninstall pybind11-stubgen
# Add "python.analysis.stubPath": ".vscode/typings/pyrealsense2" to your .vscode/settings.json
def printRealsenseCameras():
  ctx = rs.context()
  devices = ctx.query_devices()
  print("Connected RealSense devices:", len(devices))
  for device in devices:
    print(device.get_info(rs.camera_info.name))

def configureDepthCam(width:int, height:int, fps:int) -> tuple[rs.pipeline, rs.align, rs.config]:
  """
  Returns:
  A camera `pipeline`, `align` (from rs.align(rs.stream.color)) and `config`
  """
  printRealsenseCameras()

  pipeline = rs.pipeline()
  config = rs.config()

  config.enable_stream(rs.stream.color, min(width, 1920), min(height, 1080), rs.format.bgr8, fps)
  config.enable_stream(rs.stream.depth, min(width, 640), min(height, 480), rs.format.z16, fps)

  # Align depth frames to the color stream coordinate space
  align = rs.align(rs.stream.color)

  return pipeline, align, config
# Functions onwards require a call to configureDepthCam() and may also need
# profile = pipeline.start(config) before requesting frames
# pipeline.stop() after program ends
def getIntrinsics(profile:rs.pipeline_profile):
  # Fetch factory intrinsics for the color stream
  color_stream = profile.get_stream(rs.stream.color)
  color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
  # Fetch factory intrinsics for the depth stream
  depth_stream = profile.get_stream(rs.stream.depth)
  depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
  return color_intrinsics, depth_intrinsics

def saveIntrinsicsJSON(profile:rs.pipeline_profile, jsonPath:str):
  color_i, _ = getIntrinsics(profile)
  import json
  from pathlib import Path
  outPath = Path(jsonPath)

  result = {
    "camera": {
      "P": [
        float(color_i.fx), 0.0, float(color_i.ppx),
        0.0, float(color_i.fy), float(color_i.ppy),
        0.0, 0.0, 1.0,
      ],
      "distortion": [float(coefficient) for coefficient in color_i.coeffs],
      "width": int(color_i.width),
      "height": int(color_i.height),
    }
  }

  try:
    outPath.write_text(
      json.dumps(result, indent=2, allow_nan=False) + "\n",
      encoding="utf-8",
    )
  except (OSError, ValueError) as error:
    print("Failed writing intrinsics to JSON", error)
    return
  print(f"Distortion model: {color_i.model}")
  print(f"Saved JSON to {outPath}")

def getFrames(pipeline:rs.pipeline, align:rs.align) -> tuple[rs.video_frame, rs.depth_frame]:
  frames = pipeline.wait_for_frames()
  aligned_frames = align.process(frames)

  color_frame = aligned_frames.get_color_frame()
  depth_frame = aligned_frames.get_depth_frame()
  return color_frame, depth_frame

def getColorImage(pipeline:rs.pipeline, align:rs.align):
  import numpy as np
  color_frame, _ = getFrames(pipeline, align)
  return np.asanyarray(color_frame.get_data())

def getDepth(cam_x:Union[int, float], cam_y:Union[int, float], pipeline:rs.pipeline, align:rs.align) -> Union[None, float]:
  _, depth_frame = getFrames(pipeline, align)
  if not depth_frame:
    return None

  depth_meters = depth_frame.get_distance(int(cam_x), int(cam_y))
  if depth_meters <= 0:
    return None
  else:
    return depth_meters
