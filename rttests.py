import numpy as np
from pathlib import Path

from rtsimdataloader import add_mesh_to_scene, get_mesh_data, simdata_to_mesh
from rtcamera import Camera
from rtscene import Scene, RenderType
from rtmain import render_scene

import pyvale.dataset as dataset
from pyvale.sensorsim.imagetools import ImageTools
import timeit

# Testing libigl
import igl
from stl import mesh
import scipy as sp
import pyvista
#import bpy
import pyvale.blender as blender
import pyvale.sensorsim as sens
from scipy.spatial.transform import Rotation
# Try xatlas
import xatlas

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

number_of_samples = 50; # for anti-aliasing

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

# Lights - to be added later


# Load sample data files to test image rendering algorithm. Returns a file path to an exodus file

#data_path = dataset.render_simple_block_path() # Test mesh 1; deleted in Pyvale 2025.8.1
data_path = Path(Path().resolve().joinpath("simple_block.e")) # Temp to use the simple block simdata, since it was used in many benchmarks during the development
data_path2 = dataset.render_mechanical_3d_path() # Test mesh 2
add_mesh_to_scene(scene, data_path)
add_mesh_to_scene(scene, data_path2, world_position=np.array([-5.0, 0.0, -10.0]), scale=50)
add_mesh_to_scene(scene, data_path, world_position=np.array([5.0, -3.5, -1.0]), scale=500)

############################################# DEBUG PLAYGROUND #############################################
# Texture tests

# Get datapaths to .tiff images that come with pyvale
ref_img = dataset.dic_plate_with_hole_ref() # Reference image with a speckle pattern. Comes with a hole so could potentiall be used with the mesh from data_path2?
ref_img_flat = dataset.dic_plate_rigid_ref() # Reference image without the hole. Might be useful for trying to wrap around mesh from data_path
# Numpy arrays with dimensions (width px, height px)
hole_img = ImageTools.load_image_greyscale(ref_img) # Get greyscale numpy array; this particular one is (1540,1040)
no_hole_img = ImageTools.load_image_greyscale(ref_img) # Get greyscale numpy array; this particular one is (1540,1040)
# Dictionaries with mesh data
hole_mesh = get_mesh_data(data_path2, world_position=np.array([-5.0, 0.0, -10.0]), scale=50)
block_mesh = get_mesh_data(data_path)
# RenderMesh objects
hole_mesh_rm = simdata_to_mesh(data_path2, field_components=("disp_x","disp_y", "disp_z"), fields_to_render = ("disp_y", "disp_x"), scale=50.0)
block_mesh_rm = simdata_to_mesh(data_path, field_components=("disp_x","disp_y", "disp_z"), fields_to_render = ("disp_y", "disp_x"), scale=100.0)
#print(block_mesh["coords"].shape)
#print(block_mesh["connectivity"].shape)

hole_mesh_faces = hole_mesh["connectivity"]
hole_mesh_vertices = hole_mesh["coords"]
block_mesh_faces = block_mesh["connectivity"]
block_mesh_vertices = block_mesh["coords"]

######################### Libigl and pyvista

# Use numpy-stl to export mesh to try if saving as stl and then reading in libigl works since directly using numpy arrays doesn't
test_block_exp = mesh.Mesh(np.zeros(block_mesh_faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(block_mesh_faces):
    for j in range(3):
        test_block_exp.vectors[i][j] = block_mesh_vertices[f[j],:]
#test_block_exp.save('test_block_exp.stl')

# Try creating an STL from node coords expanded directly since dimensionally this checks out
test_block_2 = mesh.Mesh(np.zeros(block_mesh_faces.shape[0], dtype=mesh.Mesh.dtype))
for i in range(44):
    test_block_2.vectors[i] = block_mesh["node_coords_expanded"][i]
#test_block_exp.save('test_block_exp_2.stl')

# Currently using pyvista which needs the texture to be (size_1, size_2, 3 rgb values), so copy values and add axis 
#hole_img_pv = np.tile(hole_img[...,None], (1,1,3)) # shape (1540, 1040, 3)
#tex_hole = pyvista.numpy_to_texture(hole_img_pv)
#mesh = pyvista.DataSet(node_coords_expanded)

#pl = pyvista.Plotter()
#pl.add_mesh(block_mesh["node_coords_expanded"], texture=tex_hole)
#pl.show()

#print(test.shape)

v, f = igl.read_triangle_mesh(Path.cwd() / "test_block_exp.stl")
#test = np.reshape(test_block_exp.vectors, (132,3))
#print(test == v)

#print(f"Node coords original shape: {block_mesh_vertices.shape}")
#print(f"Node coords after re-reading shape: {v.shape}")
#print(f"Connectivity original shape: {block_mesh_faces.shape}")
#print(f"Connectivity after re-reading shape: {f.shape}")

#print(f==block_mesh_faces)
#print(f"f: {f}")
#print(f"faces original: {block_mesh_faces}")

#print(f"v: {v}")

test_v = np.reshape(block_mesh["node_coords_expanded"], (132,3))
#print(test.shape)
#print(test_v == v)

#print(test_3 == v)

temp_f = np.ndarray(shape=(44,3))
count = 0
for i in range(44):
    for j in range(3):
        temp_f[i,j] = count
        count += 1
#print(temp)

## Find the open boundary - common for all methods
bnd = igl.boundary_loop(temp_f) # this works
#print(bnd)

#bnd_def = igl.boundary_loop(f)
#print(bnd_def)

## Map the boundary to a circle, preserving edge proportions
bnd_uv = igl.map_vertices_to_circle(test_v, bnd) # this returns segmentation fault with out data layout

## Harmonic parametrization for the internal vertices
#bnd_uv = igl.map_vertices_to_circle(v, bnd) # this returns segmentation fault with out data layout
uv = igl.harmonic(test_v, temp_f, bnd, bnd_uv, 1)
v_p = np.hstack([uv, np.zeros((uv.shape[0],1))])

# Now try with out data reshaped instead of reading stl

## Find the open boundary - common for all methods
#bnd = igl.boundary_loop(f) # this works

## Map the boundary to a circle, preserving edge proportions
#bnd_uv = igl.map_vertices_to_circle(v, bnd) # this returns segmentation fault with out data layout

## Harmonic parametrization for the internal vertices
#bnd_uv = igl.map_vertices_to_circle(v, bnd) # this returns segmentation fault with out data layout
#uv = igl.harmonic(v, f, bnd, bnd_uv, 1)
#v_p = np.hstack([uv, np.zeros((uv.shape[0],1))])

# As-rigid-as-possible parametrization
#bnd_uv = igl.map_vertices_to_circle(v, bnd) # this returns segmentation fault with out data layout
#uv = igl.harmonic(v, f, bnd, bnd_uv, 1)
#arap = igl.ARAP(v, f, 2, np.zeros(0)) # returns type error for MappingEnergyType
#uva = arap.solve(np.zeros((0, 0)), uv)

#print(f"Uv: type - {type(uv)}; shape - {uv.shape}")
#print(f"v_p: type - {type(v_p)}; shape - {v_p.shape}")

# LSCM parametrization
# Fix two points on the boundary
#b = np.array([2, 1])

#b[0] = bnd[0] # Returns 
#b[1] = bnd[int(bnd.size / 2)]

#bc = np.array([[0.0, 0.0], [1.0, 0.0]])

#_, uv = igl.lscm(block_mesh_vertices, block_mesh_faces, b, bc)
#print(f"Uv: type - {type(uv)}; shape - {uv.shape}")
#print(f"_: type - {type(_)}; shape - {_.shape}")


####################### Pyvale's blender module now

base_dir = Path.cwd() / "pyvale-output"
if not base_dir.is_dir():
    base_dir.mkdir(parents=True, exist_ok=True)
scene1 = blender.Scene()
part = scene1.add_part(hole_mesh_rm, sim_spat_dim=3)
# Set the part location
part_location = np.array([0, 0, 0])
blender.Tools.move_blender_obj(part=part, pos_world=part_location)
part_rotation = Rotation.from_euler("xyz", [0, 0, 0], degrees=True)
blender.Tools.rotate_blender_obj(part=part, rot_world=part_rotation)
cam_data = sens.CameraData(pixels_num=np.array([1540, 1040]),
                             pixels_size=np.array([0.00345, 0.00345]),
                             pos_world=(0, 0, 400),
                             rot_world=Rotation.from_euler("xyz", [0, 0, 0]),
                             roi_cent_world=(0, 0, 0),
                             focal_length=15.0)
camera = scene1.add_camera(cam_data)
camera.location = (0, 0, 410)
camera.rotation_euler = (0, 0, 0) # NOTE: The default is an XYZ Euler angle
# Need these to not render a black image, but otherwise mapping works without lights
light_data = blender.LightData(type=blender.LightType.POINT, pos_world=(0, 0, 400), rot_world=Rotation.from_euler("xyz",[0, 0, 0]), energy=1)
light = scene1.add_light(light_data)
light.location = (0, 0, 410)
light.rotation_euler = (0, 0, 0)
material_data = blender.MaterialData()
speckle_path = dataset.dic_pattern_5mpx_path()
mm_px_resolution = sens.CameraTools.calculate_mm_px_resolution(cam_data)
scene1.add_speckle(part=part, speckle_path=speckle_path, mat_data=material_data, mm_px_resolution=mm_px_resolution)
render_data = blender.RenderData(cam_data=cam_data, base_dir=base_dir, dir_name="blender-def", threads=8)
#scene1.render_deformed_images(hole_mesh_rm, sim_spat_dim=3, render_data=render_data, part=part, stage_image=False)


####################### xatlas (Python binding)
#print(f"Input mesh: vertices - {block_mesh_vertices.shape}; faces - {block_mesh_faces.shape}")
#vmapping, indices, uvs = xatlas.parametrize(block_mesh_vertices, block_mesh_faces)
#print(f"Vmapping: type - {type(vmapping)}; shape - {vmapping.shape}; values:")
# Vmapping - Original vertex index for each new vertex. Shape N - unrelated to the original mesh size whatsoever
#print(vmapping)
# Indices - Vertex indices of the new triangles. Shape (faces x nodes per element) = same as connectivity array
#print(f"Indices: type - {type(indices)}; shape - {indices.shape}; values:")
#print(indices)
# Uvs - Texture coordinates of the new vertices. Shape (N from vmapping, 2) - since we map to 2D coordinates.
#print(f"Indices: type - {type(uvs)}; shape - {uvs.shape}; values:") 
#print(uvs)




############################################# RENDERING #############################################


#render_scene(image_height, image_width, scene, number_of_samples, base_dir, RenderType.DYNAMIC)

# Timing tests
#no_repeats = 5
#time_results = timeit.repeat("render_scene(image_height, image_width, scene, number_of_samples, base_dir, RenderType.DYNAMIC)", globals=globals(), repeat=no_repeats, number=1)
#print(time_results)
#print(f"Min: {min(time_results)} max: {max(time_results)} average: {sum(time_results)/no_repeats}")



# Below is with connectivity and node coords, not expanded version, for rtbvh_stack and rtbvh_recursion
#time_results = timeit.repeat("cpp_render_scene(image_height, image_width, number_of_samples, scene.scene_connectivity, scene.scene_coords, scene.scene_face_colors, scene.scene_camera_center, scene.scene_pixel_00_center, scene.scene_matrix_pixel_spacing)", globals=globals(), repeat=no_repeats, number=1)
