from dataclasses import dataclass, field
import numpy as np

@dataclass(slots=True)
class Scene:
    '''WIP: Dataclass for storing camera, mesh, and light data in a format that should work best with C++
    while preserving user-friendly interface.'''
    #scene_connectivity: list[np.ndarray] = field(default_factory=list) # Uncomment to test rtbvh_stack, rtbvh_recursion, or no BVH
    #scene_coords: list[np.ndarray] = field(default_factory=list) # Uncomment to test rtbvh_stack, rtbvh_recursion, or no BVH
    scene_coords_expanded: list[np.ndarray] = field(default_factory=list)
    scene_face_colors: list[np.ndarray] = field(default_factory=list)
    scene_camera_center: list[np.ndarray] = field(default_factory=list)
    scene_pixel_00_center: list[np.ndarray] = field(default_factory=list)
    scene_matrix_pixel_spacing: list[np.ndarray] = field(default_factory=list)
    scene_timestep_count: int = 1 # Number of timesteps with the default value being 1 for static images
    
    # Uncomment to test rtbvh_stack, rtbvh_recursion, or no BVH    
    #def add_mesh(self, connectivity:np.ndarray, coords: np.ndarray, face_colors: np.ndarray) -> None:
    #    '''Adds a mesh to the scene.'''
    #    self.scene_connectivity.append(connectivity)
    #    self.scene_coords.append(coords)
    #    self.scene_face_colors.append(face_colors)

    def add_mesh(self, node_coords_expanded: np.ndarray, face_colors: np.ndarray, timestep_count: int) -> None:
        '''Adds a mesh to the scene.'''
        self.scene_coords_expanded.append(node_coords_expanded)
        self.scene_face_colors.append(face_colors)
        if timestep_count > self.scene_timestep_count: # Keep the highest timestep count (should be the same for all meshes, but you never know)
            self.scene_timestep_count = timestep_count

    def fill_empty_timesteps(self):
        '''Verifies that all meshes in the scene contain data for the defined number of timesteps. If there is missing data for some meshes,
         it fills the nodal coordinates with the repeats of the last known position, and the face colors with white by default. '''
        mesh_count = len(self.scene_coords_expanded)
        COORDS_PER_NODE = 3 # Number of coordinates per single node of mesh element

        for mesh in range(mesh_count):
            mesh_timesteps = self.scene_coords_expanded[mesh].shape[0]
            mesh_elements = self.scene_coords_expanded[mesh].shape[1]
            mesh_nodes_per_elem = self.scene_coords_expanded[mesh].shape[2]
            timestep_difference = self.scene_timestep_count - mesh_timesteps # Number of timesteps not accounted for
            if timestep_difference > 0:
                # Expand arrays to fill the missing data
                # Create an array that tells numpy.repeat to to keep all data the same, and only repeat the values
                # for the last known timestep. Add +1 to account for the existing row, i.e., get the total number of 
                # times this data appears in the array, not just how much we want to add.
                repeat_counts = [1] * (mesh_timesteps - 1) + [timestep_difference + 1] 
                self.scene_coords_expanded[mesh] = np.ascontiguousarray(np.repeat(self.scene_coords_expanded[mesh], repeat_counts, axis=0)) # Should be C-contiguous by default, but we need to be extra sure
                # For face colors, we just fill the blanks with 1s (RGB white)
                # TO DO: Give user choice if they want it white or filled with the last known values as well. Or if the mesh should just magically vanish once we run out of timesteps.
                filler_data = np.ones(shape=(timestep_difference, mesh_elements, COORDS_PER_NODE))
                self.scene_face_colors[mesh] = updated_color_data = np.ascontiguousarray(np.concatenate((self.scene_face_colors[mesh], filler_data), axis=0)) # Should be C-contiguous by default, but we need to be extra sure

    def add_camera (self, camera_center: np.ndarray, pixel_00_center: np.ndarray, matrix_pixel_spacing: np.ndarray) -> None:
        '''Adds a camera to the scene.'''
        self.scene_camera_center.append(camera_center)
        self.scene_pixel_00_center.append(pixel_00_center)
        self.scene_matrix_pixel_spacing.append(matrix_pixel_spacing)


 