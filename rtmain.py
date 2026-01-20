from ast import List
import numpy as np
from dataclasses import dataclass, field

import pyvale.dataset as dataset
import timeit
from pathlib import Path # Purely to use simple block mesh after it was removed in pyvale 2026.1.0 from dataset

from rtcamera import Camera
from rtsimdataloader import add_mesh_to_scene, get_mesh_data, simdata_to_mesh
from rtscene import Scene

# Tests for getting deformed timesteps
import pyvale.sensorsim.simtools as simtools

#################################################### INPUT #####################################################
# Output image dimensions
image_width = 400  # px
aspect_ratio = 16.0 / 9.0
image_height = int(image_width / aspect_ratio)  # px
#number_of_samples = 1; # for anti-aliasing; commented out to keep it below for now - more convenient for tests
# Assume single camera for now - but can be extended to multiple cameras later
camera_center = np.array([-0.5, 1.1, 1.1])
#camera_center = np.array([-1.0, 0.1, 0.5])
camera_target = np.array([0, 0, -1])
angle_vertical_view = 90  # degrees

################################################ SCENE BUILDER ###############################################
# Targets:
# Input: meshes for the objects + object coordinates in world coordinate system + cameraS(!) position (lights to be added)
# Functions: load meshes, specify materials (later), specify cameras and their params, create the scene
# Output: a scene dataclass containing all information to render the scene + an option to save the scene dataclass to file for later use

# Tests: For now just load simple sample data, assuming 1 of each element
scene = Scene()
# Create a camera
camera1 = Camera(image_width, image_height, camera_center, camera_target, angle_vertical_view) # Camera for tests
#camera1 = Camera(image_width, image_height) # Default camera (parameters i.e., at world origin, no funny angles) for tests
#cameras = list()
#cameras.append(repack_camera_data(camera1))
camera1.add_camera_to_scene(scene)

# Load sample data file with a simple rectangular block in 3D to test image rendering algorithm. Returns a file path to an exodus file
#data_path = dataset.render_simple_block_path() # Test mesh 1; deleted in Pyvale 2025.8.1
data_path = Path(Path().resolve().joinpath("simple_block.e")) # Temp to use the simple block simdata, since it was used in many benchmarks during the development
data_path2 = dataset.render_mechanical_3d_path() # Test mesh 2
add_mesh_to_scene(scene, data_path)
add_mesh_to_scene(scene, data_path2, world_position=np.array([-5.0, 0.0, -10.0]), scale=50)
add_mesh_to_scene(scene, data_path, world_position=np.array([5.0, -3.5, -1.0]), scale=500)

# Playing with getting values at different timesteps
#render_mesh = simdata_to_mesh(data_path,("disp_x","disp_y", "disp_z"), ("disp_y", "disp_x"), 100.0)
#timesteps = render_mesh.fields_render.shape[1]

#for timestep in range(timesteps):
#    print(f"Timestep {timestep}")
#    print("Node coords")
#    deformed_nodes = simtools.SimTools.get_deformed_nodes(timestep, render_mesh) # this works
#    print(deformed_nodes)
#    print(f"Deformed nodes size: {deformed_nodes.shape}")
#    print("x-displacement nodal values")
#    timestep_field_values = render_mesh.fields_render[:,timestep, 1] # Field displacement_x at given timestep for all nodes
#    print(timestep_field_values)
#    print(f"Timestep field values size: {timestep_field_values.shape}")
    

# Lights - to be added later

#print(scene.scene_face_colors[0].shape)
#print(scene.scene_face_colors)

#from rtmaincpp import cpp_render_scene
number_of_samples = 50; # for anti-aliasing

no_repeats = 5
#cpp_render_scene(image_height, image_width, number_of_samples, scene.scene_coords_expanded, scene.scene_face_colors, scene.scene_camera_center, scene.scene_pixel_00_center, scene.scene_matrix_pixel_spacing)
# Below is with connectivity and node coords, not expanded version, for rtbvh_stack and rtbvh_recursion
#time_results = timeit.repeat("cpp_render_scene(image_height, image_width, number_of_samples, scene.scene_connectivity, scene.scene_coords, scene.scene_face_colors, scene.scene_camera_center, scene.scene_pixel_00_center, scene.scene_matrix_pixel_spacing)", globals=globals(), repeat=no_repeats, number=1)
#time_results = timeit.repeat("cpp_render_scene(image_height, image_width, number_of_samples, scene.scene_coords_expanded, scene.scene_face_colors, scene.scene_camera_center, scene.scene_pixel_00_center, scene.scene_matrix_pixel_spacing)", globals=globals(), repeat=no_repeats, number=1)
#print(time_results)
#print(f"Min: {min(time_results)} max: {max(time_results)} average: {sum(time_results)/no_repeats}")

