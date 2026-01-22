import numpy as np
from pathlib import Path

from rtsimdataloader import add_mesh_to_scene
from rtcamera import Camera
from rtscene import Scene, RenderType
from rtmain import render_scene

import pyvale.dataset as dataset
import timeit

#################################################### INPUT #####################################################
# Choose output directory for the rendered images
base_dir = Path.cwd() / "pyvale-output"
if not base_dir.is_dir():
    base_dir.mkdir(parents=True, exist_ok=True)
# Output image dimensions
image_width = 400  # px
aspect_ratio = 16.0 / 9.0
image_height = int(image_width / aspect_ratio)  # px
# Assume single camera for now - but can be extended to multiple cameras later
camera_center = np.array([-0.5, 1.1, 1.1])
#camera_center = np.array([-1.0, 0.1, 0.5])
camera_target = np.array([0, 0, -1])
angle_vertical_view = 90  # degrees

number_of_samples = 1; # for anti-aliasing

################################################ SCENE BUILDER ###############################################
# Targets:
# Input: meshes for the objects + object coordinates in world coordinate system + cameraS(!) position (lights to be added)
# Functions: load meshes, specify materials (later), specify cameras and their params, create the scene
# Output: a scene dataclass containing all information to render the scene + an option to save the scene dataclass to file for later use

# Start with creating the scene to be rendered
scene = Scene()

# Create and add cameras to the scene
camera1 = Camera(image_width, image_height, camera_center, camera_target, angle_vertical_view) # Camera for tests
#camera0 = Camera(image_width, image_height) # Default camera (parameters i.e., at world origin, no funny angles) for tests
camera1.add_camera_to_scene(scene)
#camera0.add_camera_to_scene(scene)

# Load sample data files to test image rendering algorithm. Returns a file path to an exodus file
#data_path = dataset.render_simple_block_path() # Test mesh 1; deleted in Pyvale 2025.8.1
data_path = Path(Path().resolve().joinpath("simple_block.e")) # Temp to use the simple block simdata, since it was used in many benchmarks during the development
data_path2 = dataset.render_mechanical_3d_path() # Test mesh 2
print(type(data_path2))
add_mesh_to_scene(scene, data_path)
add_mesh_to_scene(scene, data_path2, world_position=np.array([-5.0, 0.0, -10.0]), scale=50)
#add_mesh_to_scene(scene, data_path, world_position=np.array([5.0, -3.5, -1.0]), scale=500)


# Lights - to be added later


render_scene(image_height, image_width, scene, number_of_samples, base_dir, RenderType.DYNAMIC)

# Timing tests
#no_repeats = 5
#time_results = timeit.repeat("render_scene(image_height, image_width, scene, number_of_samples, base_dir, RenderType.DYNAMIC)", globals=globals(), repeat=no_repeats, number=1)
#print(time_results)
#print(f"Min: {min(time_results)} max: {max(time_results)} average: {sum(time_results)/no_repeats}")



# Below is with connectivity and node coords, not expanded version, for rtbvh_stack and rtbvh_recursion
#time_results = timeit.repeat("cpp_render_scene(image_height, image_width, number_of_samples, scene.scene_connectivity, scene.scene_coords, scene.scene_face_colors, scene.scene_camera_center, scene.scene_pixel_00_center, scene.scene_matrix_pixel_spacing)", globals=globals(), repeat=no_repeats, number=1)
