# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

import pyvale.mooseherder as mh
import pyvale.sensorsim as sens
import pyvale.sensorsim.simtools as simtools

from enum import StrEnum, IntEnum
from dataclasses import dataclass, field


# Type of coloring that goes onto the mesh surface
class SurfType(StrEnum):
    FIELD_COLOR = "field_color"
    TEXTURE = "texture"

# Number of nodes per element
class ElementNodeCount(IntEnum):
    TRI3 = 3,
    QUAD4 = 4,
    TET4 = 4,
    TET10 = 10,
    TET14 = 14,
    HEX8 = 8,
    HEX20 = 20,
    HEX27 = 27

COORDS_PER_NODE = 3
RGB_VALS = 3

################################################ 1. Load mesh data into appropriate class ##############################

# Class used as a common interface between linear RenderMeshes and Mesh from curved element implementation
@dataclass
class RTMesh(): # Doesn't work if I set slots to True?
    node_coords: np.ndarray = field(default=None)
    connectivity: np.ndarray = field(default=None)
    node_coords_over_time: np.ndarray = field(default=None)
    node_coords_expanded_over_time: np.ndarray = field(default=None)
    face_colors_over_time: np.ndarray = field(default=None)
    uvs_over_time: np.ndarray = field(default=None)
    uvs: np.ndarray = field(default=None)
    texture: np.ndarray = field(default=None)
    # mesh_to_world_mat: np.ndarray = field(default=None)
    surface_type: SurfType = field(default=None)
    timestep_count: int = field(default=1)
    element_count: int = field(default=0)
    node_count: int = field(default=0)
    nodes_per_element: ElementNodeCount = field(default=ElementNodeCount.TRI3)

    def set_surface(self, surface_type: SurfType = SurfType.FIELD_COLOR, surface_fill: np.ndarray = None, uv_coords: np.ndarray = None):
        # surface_fill is either texture array or color, or field value-based colors
        # Reset everything if user is changing the surface type
        if self.surface_type is not None and surface_type != self.surface_type:
            self.face_colors_over_time = None
            self.uvs_over_time = None
            self.texture = None
            self.uvs = None

        self.surface_type = surface_type
        # Solid colors
        if surface_type == SurfType.FIELD_COLOR:
            if surface_fill is None:
                print("No colour data passed. Pre-filling automatically with grey.")
            elif surface_fill.shape == (RGB_VALS,):
                # Populate with passed solid color
                self.face_colors_over_time = np.ones((self.timestep_count, self.element_count, RGB_VALS)) * surface_fill
                return
            elif surface_fill.shape == (self.element_count, RGB_VALS):
                # One avg. RGB colour value per element, given only for one timestep
                self.face_colors_over_time =  np.broadcast_to(surface_fill[np.newaxis, ...], (self.timestep_count, self.element_count, RGB_VALS))
            elif surface_fill.shape == (self.timestep_count, self.element_count, RGB_VALS):
                # One avg. RGB colour value per element, given for each timestep
                self.face_colors_over_time = surface_fill
                return
            else:
                print("Surface fill must be of shape (3,) or (element_count, 3) or (timestep_count, element_count, 3).\nPre-filling automatically with grey.")
            # Create face colors over time of appropriate size and pre-populate with grey
            self.face_colors_over_time = np.ones((self.timestep_count, self.element_count, RGB_VALS), dtype=np.float64) * 0.5
        # Texture
        elif surface_type == SurfType.TEXTURE:
            if uv_coords is None:
                print("UV coordinates are required to append texture.")
                return
            # Uncomment/update this when I figure out how I want to store the uvs
            #if uv_coords.shape[1] != 2: # uvs must be of shape (new_vertex_count, 2)... or (element_count, 3, 2) if we index into them like with node_coords_expanded
            #    print(f"UV coordinates must be of shape (new_node_count, 2). Yours are {uv_coords.shape}.")
            #    return
            if surface_fill.ndim != 2:
                print("Wrong number of dimensions. The array containing the texture should be two-dimensional.")
                return
            self.uvs = uv_coords
            self.uvs_over_time = np.broadcast_to(uv_coords[np.newaxis, ...], (self.timestep_count, self.element_count, 3, 2)) # here 3 is nodes_per_elem and 2 is for 2 uv coords - update with constants once I'm happy this will be the case for curved elements also
            # Add check for shape of texture array
            self.texture = surface_fill


            """
            vmapping, indices, uvs = xatlas.parametrize(coords,
                                                        connectivity)  # Get UVs from xatlas. coords = vertices, connectivity = faces
            uvs_faces = uvs[indices]  # To get shape (element_count, 3, 2)
            # UVs shouldn't change unless we use procedural textures (not expected? so we can just broadcast these for now.
            # But since they don't change, might as well keep them without the timestep dimension to save memory. UPDATE LATER
            uvs_over_time = np.broadcast_to(uvs_faces[np.newaxis, ...], (timestep_count, element_count, 3,
                                                                         2)).copy()  # Shape (timestep_count, element_count, 3, 2)
            """
            pass

        # Linear meshes - use existing pyvale RenderMesh class and functions
def simdata_to_rtmesh(pypath: Path,
                    field_components: tuple = ("disp_x", "disp_y", "disp_z"),
                    fields_to_render: tuple = ("disp_y", "disp_x"),
                    scale: float = 100.0,
                    world_position: np.ndarray = None,
                    world_rotation: Rotation = None):
    # Convert the simulation output into a SimData object
    sim_data = mh.ExodusLoader(pypath).load_all_sim_data()  # Pyvale 2026.1.0
    # Scale the coordinates and displacement fields to mm
    sim_data = sens.scale_length_units(scale=scale, sim_data=sim_data, disp_keys=field_components)
    # Create RenderMesh object
    render_mesh = sens.create_render_mesh(sim_data, fields_to_render, sim_spat_dim=sens.EDim.THREED,
                                          field_disp_keys=field_components)
    # Set world position and rotation (where applicable)
    if world_position is not None:
        render_mesh.set_pos(world_position)
    if world_rotation is not None:
        render_mesh.set_rot(world_rotation)

    # Handle nodal coordinates
    coords_world = np.matmul(render_mesh.coords, render_mesh.mesh_to_world_mat.T)  # Convert to world coordinates
    render_mesh.coords = coords_world  # Replace nodal coordinates in RenderMesh with their world coordinate equivalents. We can do that since for deformed nodes, we just add values
    coords = np.ascontiguousarray(render_mesh.coords[:, :COORDS_PER_NODE])

    # Create RTMesh object and assign data appropriately
    rtmesh = RTMesh()
    try:
        rtmesh.nodes_per_element = ElementNodeCount(render_mesh.nodes_per_elem)
    except ValueError:
        print(f"Error: Invalid nodes_per_elem value: {render_mesh.nodes_per_elem}.")
    rtmesh.node_coords = coords
    connectivity = render_mesh.connectivity
    rtmesh.connectivity = connectivity
    timestep_count = render_mesh.fields_render.shape[1]
    rtmesh.timestep_count = timestep_count
    element_count = render_mesh.elem_count
    rtmesh.element_count = element_count

    node_count = render_mesh.node_count
    rtmesh.node_count = render_mesh.node_count

    # Nodal coordinates over time
    # Process data for the 0th element - always the same for deformable and static images
    coords = np.ascontiguousarray(render_mesh.coords[:, :COORDS_PER_NODE])
    coords_over_time = np.ndarray(shape=(timestep_count, node_count, COORDS_PER_NODE), dtype=np.float64)
    coords_over_time[0] = coords
    # This may stay, or may not. TBD
    node_coords_expanded_over_time = np.ndarray(shape=(timestep_count, element_count, rtmesh.nodes_per_element, COORDS_PER_NODE),
                                                dtype=np.float64)  # Store nodal coordinates over all timesteps
    node_coords_expanded_over_time[0] = coords[
        connectivity, :COORDS_PER_NODE]  # Expanded nodal coords, so we do not need the connectivity array

    # Get data over multiple timesteps
    if rtmesh.timestep_count != 1:
        for timestep in range(1, timestep_count):
            # Get deformed nodal coordinates and expand them
            node_coords = simtools.get_deformed_nodes(timestep, render_mesh)
            coords = np.ascontiguousarray(node_coords)
            coords_over_time[timestep] = coords
            node_coords_expanded_over_time[timestep] = coords[connectivity]  # Expand nodal coords,
            #node_coords_expanded_over_time[timestep] = coords[connectivity, :COORDS_PER_NODE]  # Expand nodal coords,
    rtmesh.node_coords_over_time = coords_over_time
    rtmesh.node_coords_expanded_over_time = node_coords_expanded_over_time
    return rtmesh

# Wiera's quadratic element functions
# Functions to load quadratic elements from .vol mesh file with uniform colour
class Mesh:
    def __init__(self):
        # self.elements contains indices; self.points contains coordinates
        self.points = np.empty((0, COORDS_PER_NODE))
        self.elements = np.empty((0, 10), dtype=int)
        self.elem_coords = np.empty((0, 10, COORDS_PER_NODE))

    def loadVolFile(self, filename):
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"Cannot open .vol file: {filename}")
            return

        # Use an iterator to move through lines linearly
        line_iter = iter(lines)

        for line in line_iter:
            # Load points
            if "points" in line:
                num_points = int(next(line_iter).strip())
                # Read next num_points lines and convert to float array
                point_data = []
                for _ in range(num_points):
                    point_data.append(list(map(np.float64, next(line_iter).split())))
                self.points = np.array(point_data)

            # Load volume elements
            elif "volumeelements" in line:
                num_elements = int(next(line_iter).strip())
                element_list = []
                for _ in range(num_elements):
                    parts = list(map(int, next(line_iter).split()))
                    # parts[0]: mat, parts[1]: np (num nodes)
                    # elements start from index 2 to the end
                    # Convert 1-based indexing to 0-based
                    nodes = [node - 1 for node in parts[2:]]
                    element_list.append(nodes)

                self.elements = np.array(element_list)

    def getElementCoords(self):
        """
        Map point coordinates to elements.
        """
        if self.points.size == 0 or self.elements.size == 0:
            print("Mesh data not loaded.")
            return

        self.elem_coords = self.points[self.elements]

def get_mesh_to_world_matrix(world_position: np.ndarray,
                             world_rotation: Rotation):
    mesh_to_world_mat = np.zeros((4,4),dtype=np.float64)
    mesh_to_world_mat[0:3, 0:3] = world_rotation.as_matrix()
    mesh_to_world_mat[-1, -1] = 1.0
    mesh_to_world_mat[0:3, -1] = world_position
    return mesh_to_world_mat

def orient_mesh_in_world(node_coords: np.ndarray,
                                world_position: np.ndarray,
                                world_rotation: Rotation):
    mesh_to_world_mat = get_mesh_to_world_matrix(world_position, world_rotation)
    node_count = node_coords.shape[0]
    coords_stack = np.column_stack([node_coords, np.ones(node_count, dtype=np.float64)])  #Stack so we have (node_count,4) matrix that can be multiplied by transformation matrix
    node_coords_world = np.matmul(coords_stack, mesh_to_world_mat.T)
    return np.ascontiguousarray(node_coords_world[:,:COORDS_PER_NODE])

def vol_mesh_to_rtmesh(pypath: Path,
                    scale: float = 100.0,
                    world_position: np.ndarray = None,
                    world_rotation: Rotation = None):
    '''Converts a .vol mesh to a RenderMesh object and returns it.'''
    mesh = Mesh()
    mesh.loadVolFile(pypath)
    mesh.getElementCoords()

    # Set world position and rotation (where applicable)
    if world_position is None:
        world_position = np.array((0.0, 0.0, 0.0), dtype=np.float64)
    if world_rotation is None:
        world_rotation = Rotation.from_euler("zyx", (0.0, 0.0, 0.0), degrees=True)

    # Handle nodal coordinates (scaling, positioning)
    coords_mesh = mesh.points * scale
    coords_mesh = orient_mesh_in_world(coords_mesh, world_position, world_rotation)

    # Create RTMesh object and assign data appropriately
    rtmesh = RTMesh()
    rtmesh.node_coords = coords_mesh
    rtmesh.connectivity = mesh.elements
    timestep_count = 1
    rtmesh.timestep_count = timestep_count  # Temporarily they only have data for static renders
    element_count = mesh.elements.shape[0]
    rtmesh.element_count = element_count
    rtmesh.nodes_per_element = ElementNodeCount(mesh.elements.shape[1]) # Is 10

    # Data "over time"
    node_count = mesh.points.shape[0]
    rtmesh.node_count = node_count
    coords_over_time = np.ndarray(shape=(timestep_count, node_count, COORDS_PER_NODE), dtype=np.float64)
    coords_over_time[0,:,:] = coords_mesh
    rtmesh.node_coords_over_time = coords_over_time
    #rtmesh.face_colors_over_time = np.ones((rtmesh.timestep_count, rtmesh.element_count, COORDS_PER_NODE)) * [1.0, 0.078, 0.57]

    # Node coords expanded. TBD if they stay
    mesh.getElementCoords() # Not need it anymore
    node_coords_expanded_over_time = np.ndarray(shape=(timestep_count, element_count, rtmesh.nodes_per_element, COORDS_PER_NODE),
                                                dtype=np.float64)  # Store nodal coordinates over all timesteps
    #face_colors_over_time = np.ndarray(shape=(timestep_count, element_count, RGB_VALS), dtype=np.float64)  # Store face colors over all timesteps
    #face_colors_over_time[:, :] = [1.0, 0.078, 0.57]
    node_coords_expanded_over_time[0, :, :, :] = coords_mesh[rtmesh.connectivity]
    rtmesh.node_coords_expanded_over_time = node_coords_expanded_over_time

    # For indexing tests
    #print(f"Elements as done by Wiera: {mesh.elem_coords}")
    #print(f"Elements as done by me: {mesh.points[rtmesh.connectivity]}")
    #print(f"Are the same?: {(mesh.elem_coords == mesh.points[rtmesh.connectivity]).all()}")

    # Get data over multiple timesteps - tbd when we actually render something non-static for these elements
    #if rtmesh.timestep_count != 1:
    #    for timestep in range(1, timestep_count):
    #        # Get deformed nodal coordinates and expand them
    #        node_coords = simtools.get_deformed_nodes(timestep, render_mesh)
    #        coords = np.ascontiguousarray(node_coords)
    #        coords_over_time[timestep] = coords
    #rtmesh.node_coords_over_time = coords_over_time
    return rtmesh

# Test to cheat a little bit with loading non-simdata meshes for now
def vtk_mesh_to_rtmesh(pypath: Path,
                    scale: float = 100.0,
                    world_position: np.ndarray = None,
                    world_rotation: Rotation = None):
    import pyvista as pv

    # Set world position and rotation (where applicable)
    if world_position is None:
        world_position = np.array((0.0, 0.0, 0.0), dtype=np.float64)
    if world_rotation is None:
        world_rotation = Rotation.from_euler("zyx", (0.0, 0.0, 0.0), degrees=True)

    mesh = pv.read(pypath)
    merged_grid = mesh.merge_points(tolerance=1e-5)  # Might change node/point ID, so need old->new mapping
    surf = merged_grid.extract_surface()

    # World positioning
    coords_world = np.array(surf.points)
    coords_world = np.hstack((coords_world, np.ones([coords_world.shape[0], 1])))
    # Handle nodal coordinates (scaling, positioning)
    coords_mesh = mesh.points * scale
    coords_mesh = orient_mesh_in_world(coords_mesh, world_position, world_rotation)

    # Connectivity
    faces = np.array(surf.faces)
    first_elem_nodes_per_face = faces[0]
    nodes_per_face_vec = faces[0::(first_elem_nodes_per_face + 1)]
    nodes_per_face = first_elem_nodes_per_face
    num_faces = int(faces.shape[0] / (nodes_per_face + 1))
    connectivity = np.reshape(faces, (num_faces, nodes_per_face + 1))
    # shape=(num_elems,nodes_per_elem), C format
    connectivity = np.ascontiguousarray(connectivity[:, 1:], dtype=np.uintp)

    # Create RTMesh object and assign data appropriately
    rtmesh = RTMesh()
    rtmesh.node_coords = coords_mesh
    rtmesh.connectivity = connectivity
    timestep_count = 1
    rtmesh.timestep_count = timestep_count  # Temporarily they only have data for static renders
    element_count = connectivity.shape[0]
    rtmesh.element_count = element_count
    rtmesh.nodes_per_element = ElementNodeCount(connectivity.shape[1])

    # Data "over time"
    node_count = coords_mesh.shape[0]
    rtmesh.node_count = node_count
    coords_over_time = np.ndarray(shape=(timestep_count, node_count, COORDS_PER_NODE), dtype=np.float64)
    coords_over_time[0, :, :] = coords_mesh
    rtmesh.node_coords_over_time = coords_over_time
    # rtmesh.face_colors_over_time = np.ones((rtmesh.timestep_count, rtmesh.element_count, COORDS_PER_NODE)) * [1.0, 0.078, 0.57]

    # Node coords expanded. TBD if they stay
    node_coords_expanded_over_time = np.ndarray(
        shape=(timestep_count, element_count, rtmesh.nodes_per_element, COORDS_PER_NODE),
        dtype=np.float64)  # Store nodal coordinates over all timesteps
    # face_colors_over_time = np.ndarray(shape=(timestep_count, element_count, RGB_VALS), dtype=np.float64)  # Store face colors over all timesteps
    # face_colors_over_time[:, :] = [1.0, 0.078, 0.57]
    node_coords_expanded_over_time[0, :, :, :] = coords_mesh[connectivity]
    rtmesh.node_coords_expanded_over_time = node_coords_expanded_over_time
    return rtmesh

"""
def test_vtk_mesh_loader():
    # Seems to pass for both 2d and 3d vtk meshes so keep it for now
    data_path = Path(Path().resolve().joinpath("cyl_gmsh_vtk_test_2d.vtk"))
    rtmesh = vtk_mesh_to_rtmesh(data_path, scale=500, world_position=np.array([1.0, -23, -1.0]))
    assert rtmesh.nodes_per_element == ElementNodeCount(3)
    assert rtmesh.timestep_count == 1
    assert rtmesh.node_coords.shape == (rtmesh.node_count, COORDS_PER_NODE)
    assert rtmesh.connectivity.shape == (rtmesh.element_count, rtmesh.nodes_per_element)
    assert rtmesh.node_coords_over_time.shape == (rtmesh.timestep_count, rtmesh.node_count, COORDS_PER_NODE)
    assert rtmesh.node_coords_expanded_over_time.shape == (rtmesh.timestep_count, rtmesh.element_count, rtmesh.nodes_per_element, COORDS_PER_NODE)

test_vtk_mesh_loader()

def compare_indexing():
    # Check if we need mesh.GetElementCoords() or if we can use the same indexing as for everything else - Yes, we can
    # Quadratic sphere
    data_path_sph = Path(Path().resolve().joinpath("sphere_1.vol"))
    rtmesh_sph = vol_mesh_to_rtmesh(data_path_sph, scale=500, world_position=np.array([1.0, -23, -1.0]))

    # Single curved tet
    #data_path_cur = Path(Path().resolve().joinpath("one_tet_1.vol"))
    #rtmesh_cur_tet = vol_mesh_to_rtmesh(data_path_cur, scale=500, world_position=np.array([1.0, -23, -1.0]))

compare_indexing()


def test_rtmesh_conversion():
    # Test if RTMesh is created correctly for both "regular" (simdata) meshes and volume meshes
    # Simdata example
    import pyvale.dataset as dataset
    data_path = dataset.render_mechanical_3d_path()  # Test mesh 2
    rtmesh_lin = simdata_to_rtmesh(data_path, scale=500, world_position=np.array([1.0, -23, -1.0]))
    # Assertions for known data
    print(rtmesh_lin.nodes_per_element)
    assert rtmesh_lin.timestep_count == 11
    assert rtmesh_lin.node_coords_over_time.shape == (11, rtmesh_lin.node_count, COORDS_PER_NODE)
    assert rtmesh_lin.element_count == rtmesh_lin.connectivity.shape[0]
    assert rtmesh_lin.connectivity.shape == (rtmesh_lin.element_count, rtmesh_lin.nodes_per_element)
    assert rtmesh_lin.node_coords_expanded_over_time.shape == (rtmesh_lin.timestep_count, rtmesh_lin.element_count, rtmesh_lin.nodes_per_element, COORDS_PER_NODE)

    # Single curved tet
    data_path_cur = Path(Path().resolve().joinpath(
        "one_tet_1.vol"))
    rtmesh_cur_tet = vol_mesh_to_rtmesh(data_path_cur, scale=500, world_position=np.array([1.0, -23, -1.0]))
    assert rtmesh_cur_tet.timestep_count == 1
    assert rtmesh_cur_tet.node_coords_over_time.shape == (1, rtmesh_cur_tet.node_count, COORDS_PER_NODE)
    assert rtmesh_cur_tet.element_count == 1 # Single tet, so we expect one element
    assert rtmesh_cur_tet.connectivity.shape == (rtmesh_cur_tet.element_count, rtmesh_cur_tet.nodes_per_element)
    assert rtmesh_cur_tet.node_coords_expanded_over_time.shape == (rtmesh_cur_tet.timestep_count, rtmesh_cur_tet.element_count, rtmesh_cur_tet.nodes_per_element, COORDS_PER_NODE)

    # Quadratic sphere
    data_path_sph = Path(Path().resolve().joinpath("sphere_1.vol"))
    rtmesh_sph = vol_mesh_to_rtmesh(data_path_sph, scale=500, world_position=np.array([1.0, -23, -1.0]))
    assert rtmesh_sph.timestep_count == 1
    assert rtmesh_sph.node_coords_over_time.shape == (1, rtmesh_sph.node_count, COORDS_PER_NODE)
    assert rtmesh_sph.connectivity.shape == (rtmesh_sph.element_count, rtmesh_sph.nodes_per_element)
    assert rtmesh_sph.node_coords_expanded_over_time.shape == (rtmesh_sph.timestep_count, rtmesh_sph.element_count, rtmesh_sph.nodes_per_element, COORDS_PER_NODE)

test_rtmesh_conversion() # All passed - sweet
"""

