import pyvale.mooseherder as mh
import pyvale.sensorsim as sens
import numpy as np
import rtscene

def simdata_to_mesh(pypath, field_components, fields_to_render, scale):
    # Convert the simulation output into a SimData object
    #sim_data = mh.ExodusReader(pypath).read_all_sim_data() # Pyvale 2025.8.1
    sim_data = mh.ExodusLoader(pypath).load_all_sim_data() # Pyvale 2026.1.0
    # Scale the coordinates and displ. fields to mm
    #sim_data = sens.scale_length_units(scale=scale,sim_data=sim_data,disp_comps=field_components) # Pyvale 2025.8.1
    sim_data = sens.scale_length_units(scale=scale,sim_data=sim_data,disp_keys=field_components) # Pyvale 2026.1.0
    #render_mesh = sens.create_render_mesh(sim_data, fields_to_render ,sim_spat_dim=3,field_disp_keys=field_components) # Pyvale 2025.8.1. Still works, but now we use enum for spatial dim, not a number
    render_mesh = sens.create_render_mesh(sim_data, fields_to_render ,sim_spat_dim=sens.EDim.THREED,field_disp_keys=field_components) # Pyvale 2026.1.0
    return render_mesh

def add_mesh_to_scene(scene, pypath, field_components=("disp_x","disp_y", "disp_z"), fields_to_render = ("disp_y", "disp_x"), world_position = None, scale = 100.0) -> None:
    '''Adds a mesh to the scene dataclass.'''
    render_mesh = simdata_to_mesh(pypath, field_components, fields_to_render, scale)
    if world_position is not None:
        render_mesh.set_pos(world_position)
    coords_world = np.matmul(render_mesh.coords, render_mesh.mesh_to_world_mat.T) # Convert to world coordinates
    coords = np.ascontiguousarray(coords_world[:,:3])
    #coords = np.ascontiguousarray(render_mesh.coords[:,:3])
    connectivity = render_mesh.connectivity
    node_coords_expanded = coords[connectivity,:3] # Expanded nodal coords, so we do not need the connectivity array. Comment out to test rtbvh_stack, rtbvh_recursion, or no BVH
    x_disp_node_vals = render_mesh.fields_render[:,1, 1] # Field displacement_x at timestep 1 for all nodes.
    x_disp_node_norm = (x_disp_node_vals - x_disp_node_vals.min())/(x_disp_node_vals.max()-x_disp_node_vals.min()) # Normalize displacement values, scaling them to range [0,1] so they can map to color intensities
    # Approach 2 - taking averages and stacking them together
    node_colors = np.column_stack((x_disp_node_norm, x_disp_node_norm, x_disp_node_norm))  # Convert each scalar to an RGB triplet
    face_colors = np.mean(node_colors[connectivity],axis=1)  # Compute each face's colour as the average of its 3 node colours
    # Approach 1 - using a colour map to assign an rgb value
    # cmap = plt.get_cmap('viridis')
    # face_colors = cmap(x_disp_node_norm)[:,:3]
    #scene.add_mesh(connectivity, coords, face_colors) # Uncomment to test rtbvh_stack, rtbvh_recursion, or no BVH
    scene.add_mesh(node_coords_expanded, face_colors)

# Function to get mesh data; partially deprecated due to the introduction of add_mesh_to_scene.
def get_mesh_data(pypath, field_components=("disp_x","disp_y", "disp_z"), fields_to_render = ("disp_y", "disp_x"), world_position = None, scale = 100.0) -> dict:
    '''Returns the mesh data as a numpy array.'''
    render_mesh = simdata_to_mesh(pypath, field_components, fields_to_render, scale)
    if world_position is not None:
        render_mesh.set_pos(world_position)
    coords_world = np.matmul(render_mesh.coords, render_mesh.mesh_to_world_mat.T) # Convert to world coordinates
    coords = np.ascontiguousarray(coords_world[:,:3])
    connectivity = render_mesh.connectivity
    node_coords_expanded = coords[connectivity,:3] # Expanded nodal coords, so we do not need the connectivity array.
    x_disp_node_vals = render_mesh.fields_render[:,1, 1] # Field displacement_x at timestep 1 for all nodes.
    x_disp_node_norm = (x_disp_node_vals - x_disp_node_vals.min())/(x_disp_node_vals.max()-x_disp_node_vals.min()) # Normalize displacement values, scaling them to range [0,1] so they can map to color intensities
    # Approach 2 - taking averages and stacking them together
    node_colors = np.column_stack((x_disp_node_norm, x_disp_node_norm, x_disp_node_norm))  # Convert each scalar to an RGB triplet
    face_colors = np.mean(node_colors[connectivity],axis=1)  # Compute each face's colour as the average of its 3 node colours
    # Approach 1 - using a colour map to assign an rgb value
    # cmap = plt.get_cmap('viridis')
    # face_colors = cmap(x_disp_node_norm)[:,:3]
    return {"node_coords_expanded": node_coords_expanded, "face_colors": face_colors}
    #return {"connectivity": connectivity, "coords": coords, "face_colors": face_colors}