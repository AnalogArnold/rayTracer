#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include "./Eigen/Dense"
#include <iostream>
#include <memory>

// nanobind header files
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include "access_test.h"
#include "rtbvh.h"

//constexpr int NODE_COORDINATES = 3; // number of coordinates per each mesh node. Used for some of flat indexing

void process_element_data_double_array(int mesh_number_of_elements,
    const double* mesh_node_coords_ptr,
    const int* mesh_connectivity_ptr,
    std::vector<std::array<double,3>>& mesh_triangle_centroids,
    std::vector<AABB>& mesh_triangle_aabbs,
    AABB& mesh_aabb){
    // Go over all triangles in a mesh and find their AABB and centroids, build mesh AABB, and store the data in vectors
    enum ElementNodeCount nodes_per_element = TRI3;
    const int coords_per_element = nodes_per_element * NODE_COORDINATES; // number of elements times 3 coordinates each

        // Iterate over ELEMENTS/TRIANGLES in this mesh
        for (int triangle_idx = 0; triangle_idx <mesh_number_of_elements; triangle_idx++) {
            // Use pointers - means we treat the 2D array as a flat 1D array and do the indexing manually by calculating the offset.
            // HAS to be contiguous in memory for this to work properly!
            std::array<double,9> triangle_node_coords;
            int node_0 = mesh_connectivity_ptr[triangle_idx * nodes_per_element + 0]; // Equivalent to indexing as connectivity[triangle_idx, 0]
            int node_1 = mesh_connectivity_ptr[triangle_idx * nodes_per_element + 1];
            int node_2 = mesh_connectivity_ptr[triangle_idx * nodes_per_element + 2];
            // Find centroid for this triangle
            for (int i = 0; i < 3; ++i){
                triangle_node_coords[i] = mesh_node_coords_ptr[node_0 * NODE_COORDINATES + i];
                triangle_node_coords[i+3] = mesh_node_coords_ptr[node_1 * NODE_COORDINATES + i];
                triangle_node_coords[i+6] = mesh_node_coords_ptr[node_2 * NODE_COORDINATES + i];   
            }

            std::array<double,3> triangle_centroid;
            compute_element_centroid_tri3(triangle_node_coords, triangle_centroid);
            mesh_triangle_centroids.push_back(triangle_centroid);

            // Create bounding volume for this triangle
            AABB triangle_aabb;
            triangle_aabb.build_for_tri3(triangle_node_coords);
            mesh_triangle_aabbs.push_back(triangle_aabb);
            mesh_aabb.expand_to_include_AABB(triangle_aabb);
        }
}

void copy_data_to_BLAS_node_soup(BLAS &mesh_bvh,
    std::vector<int>& mesh_element_indices,
    std::vector<int>& node_minimum_element_index,
    const double* mesh_node_coords_expanded_ptr,
    const int timestep){
    // Copies appropriate mesh data to store directly in BVH node, so it can be accessed easily upon intersection and be cache-friendly
    // This way we also avoid copying the mesh data when we move the node to the BVH tree vector as they're already there when we get to this part here.

    size_t bvh_node_count = mesh_bvh.tree_nodes.size();
    int mesh_element_count = mesh_element_indices.size();
   
    // Iterate over all BVH nodes
    for (int i = 0; i < bvh_node_count; ++i){
        BLAS_Node& Node = mesh_bvh.tree_nodes[i];
        
        // Get indices of the mesh elements assigned to the node for the for loop 
        const int node_min_element_idx = node_minimum_element_index[i];
        const int node_element_count = Node.element_count;
        const int node_max_element_idx = node_min_element_idx + Node.element_count;
        const int coords_per_element = Node.nodes_per_element * NODE_COORDINATES; // number of nodes per element times 3 coordinates each
        Node.node_coords.reserve(node_element_count * coords_per_element);
        
        const int timestep_coords_stride = timestep * mesh_element_count * coords_per_element;

        // Iterate over elements in the node
        for (int element_idx = node_min_element_idx; element_idx < node_max_element_idx; ++element_idx){
            // Get the index of the stored mesh element from the reshuffled vector of indices that was created in BLAS builder
            int original_element_idx = mesh_element_indices[element_idx];
            // Add element dimension stride to find min index of nodes comprising current mesh element
            int original_element_idx_at_t = timestep_coords_stride + original_element_idx * coords_per_element; 

            for (int j = 0; j < coords_per_element; ++j){
                Node.node_coords.push_back(mesh_node_coords_expanded_ptr[original_element_idx_at_t + j]);
            }
        }
    }
}

void copy_data_to_BLAS_node_double_array(BLAS &mesh_bvh,
    std::vector<int>& mesh_element_indices,
    std::vector<int>& node_minimum_element_index,
    const double* mesh_node_coords_ptr,
    const int* mesh_connectivity_ptr,
    const int timestep){
    // Copies appropriate mesh data to store directly in BVH node, so it can be accessed easily upon intersection and be cache-friendly
    // This way we also avoid copying the mesh data when we move the node to the BVH tree vector as they're already there when we get to this part here.

    //std::cout << "BLAS builder: Copying mesh data into leaf nodes..." << std::endl;
    size_t bvh_node_count = mesh_bvh.tree_nodes.size();
    int mesh_element_count = mesh_element_indices.size();
   
    // Iterate over all BVH nodes
    for (int i = 0; i < bvh_node_count; ++i){
        BLAS_Node& Node = mesh_bvh.tree_nodes[i];
        
        // Get indices of the mesh elements assigned to the node for the for loop 
        const int node_min_element_idx = node_minimum_element_index[i];
        const int node_element_count = Node.element_count;
        const int node_max_element_idx = node_min_element_idx + Node.element_count;
        const int coords_per_element = Node.nodes_per_element * NODE_COORDINATES; // number of nodes per element times 3 coordinates each
        Node.node_coords.reserve(node_element_count * coords_per_element);
        Node.face_color.reserve(node_element_count * NODE_COORDINATES);
        
        const int timestep_coords_stride = timestep * mesh_element_count * coords_per_element;

        // Iterate over elements in the node
        for (int element_idx = node_min_element_idx; element_idx < node_max_element_idx; ++element_idx){
            // Get the index of the stored mesh element from the reshuffled vector of indices that was created in BLAS builder
            int original_element_idx = mesh_element_indices[element_idx];
            // Add element dimension stride to find min index of nodes comprising current mesh element
            int original_element_idx_at_t = timestep_coords_stride + original_element_idx * coords_per_element;
            for (int j = 0; j < coords_per_element; ++j){
                Node.node_coords.push_back(mesh_node_coords_ptr[original_element_idx_at_t + j]);
            }
        }
    }
}

void compare_bvh_access_time_double_array(const std::vector <nanobind::ndarray<const int, nanobind::c_contig>>& scene_connectivity,
    const std::vector <nanobind::ndarray<const double, nanobind::c_contig>>& scene_coords){

    // Build BLASes - BVHs for respective meshes
    enum ElementNodeCount nodes_per_element = TRI3;
    size_t scene_mesh_count = scene_coords.size();
    int timestep = 0; // Hard-code for now since we only have one timestep in the test data, but will need to be changed when we start considering more than one timestep.
    // All containers to store the data in the scene
    std::vector<std::array<double,3>> scene_blas_centroids; // Stores centroids of the whole objectes (meshes) in this scene
    scene_blas_centroids.reserve(scene_mesh_count);
    std::vector<AABB> scene_blas_aabbs; // Store AABBs of the whole objects in this scene
    scene_blas_aabbs.reserve(scene_mesh_count);
    std::vector<BLAS> scene_blases; // Store mesh_bvhs - this will be used for TLAS
    scene_blases.reserve(scene_mesh_count);

    // Iterate over MESHES
    for (size_t mesh_idx = 0; mesh_idx < scene_mesh_count; ++mesh_idx) {
        // Access data from the scene for this particular mesh
		nanobind::ndarray<const double, nanobind::c_contig> mesh_node_coords = scene_coords[mesh_idx];
		nanobind::ndarray<const int, nanobind::c_contig> mesh_connectivity = scene_connectivity[mesh_idx];

        long long mesh_element_count = mesh_connectivity.shape(0); // number of triangles/faces, will give us indices for some bits
        double* mesh_node_coords_ptr = const_cast<double*>(mesh_node_coords.data());
        int* mesh_connectivity_ptr = const_cast<int*>(mesh_connectivity.data());

        // Containers for calculated data for this mesh
        std::vector<std::array<double,3>> mesh_element_centroids; // Store centroids for this mesh
        mesh_element_centroids.reserve(mesh_element_count);
        std::vector<AABB> mesh_element_aabbs; // Bounding volumes for the elements in this mesh
        mesh_element_aabbs.reserve(mesh_element_count);
        scene_blas_aabbs.emplace_back();
        AABB& mesh_aabb = scene_blas_aabbs[mesh_idx]; // AABB for the entire mesh
        
        // Iterate over ELEMENTS in this mesh (only triangles for now)
        process_element_data_double_array(mesh_element_count, mesh_node_coords_ptr, mesh_connectivity_ptr, mesh_element_centroids, mesh_element_aabbs, mesh_aabb);

        // Find centroid of the entire mesh
        scene_blas_centroids.emplace_back();
        std::array<double,3>& mesh_centroid = scene_blas_centroids[mesh_idx];
        compute_mesh_centroid(mesh_aabb, mesh_centroid);

        // Temporary vectors to reshuffle element indices as we build the BVH, then using this mapping
        // to append the mesh data in the nodes instead of needing to access it at the split time
        std::vector<int> mesh_element_indices;
        mesh_element_indices.resize(mesh_element_count);
        std::iota(mesh_element_indices.begin(), mesh_element_indices.end(), 0);
        std::vector<int> node_minimum_element_index; // Instead of wasting BLAS_Node struct space on storing this value
         
        //std::cout << "Generating BLAS for mesh " << mesh_idx << std::endl;
        scene_blases.emplace_back(); // Generate directly inside the vector to avoid copying data
        BLAS& mesh_bvh = scene_blases[mesh_idx]; // Get a reference to the BVH of the current mesh to pass it to the builder functions

        // BLAS BVH builder functions
        build_BLAS(mesh_bvh, mesh_element_centroids, mesh_element_aabbs, mesh_element_indices, node_minimum_element_index, mesh_element_count);
        copy_data_to_BLAS_node_double_array(mesh_bvh, mesh_element_indices, node_minimum_element_index, mesh_node_coords_ptr, mesh_connectivity_ptr, timestep);
    } //MESHES

}

void compare_bvh_access_time_soup(const std::vector <nanobind::ndarray<const double,nanobind::c_contig>>& scene_coords_expanded,
    const std::vector<size_t>& mesh_element_counts){
// Handles building all acceleration structures in the scene - bottom and top level

    size_t scene_mesh_count = scene_coords_expanded.size(); 
    int timestep = 0; // Hard-code for now since we only have one timestep in the test data, but will need to be changed when we start considering more than one timestep.
    // All containers to store the data in the scene
    std::vector<std::array<double,3>> scene_blas_centroids; // Stores centroids of the whole objectes (meshes) in this scene
    scene_blas_centroids.reserve(scene_mesh_count);
    std::vector<AABB> scene_blas_aabbs; // Store AABBs of the whole objects in this scene
    scene_blas_aabbs.reserve(scene_mesh_count);
    std::vector<BLAS> scene_blases; // Store mesh_bvhs - this will be used for TLAS
    scene_blases.reserve(scene_mesh_count);

    // Iterate over MESHES to build BLASes - BVHs for respective meshes
    for (size_t mesh_idx = 0; mesh_idx < scene_mesh_count; ++mesh_idx) {
        
        // Access data from Python buffer for this particular mesh (i.e., scene->object)
        enum ElementNodeCount nodes_per_element = TRI3; // Hard-code for now since we only have triangless.
		nanobind::ndarray<const double, nanobind::c_contig> mesh_node_coords = scene_coords_expanded[mesh_idx];

        size_t mesh_element_count = mesh_element_counts[mesh_idx]; // number of elements comprising the mesh
        double* mesh_node_coords_ptr = const_cast<double*>(mesh_node_coords.data());

        // Containers for calculated data for this mesh
        std::vector<std::array<double,3>> mesh_element_centroids; // Store centroids for this mesh
        mesh_element_centroids.reserve(mesh_element_count);
        std::vector<AABB> mesh_element_aabbs; // Bounding volumes for the elements in this mesh
        mesh_element_aabbs.reserve(mesh_element_count);
        scene_blas_aabbs.emplace_back();
        AABB& mesh_aabb = scene_blas_aabbs[mesh_idx]; // AABB for the entire mesh

        // Iterate over ELEMENTS in this mesh (only triangles for now)
        process_element_data_tri3(mesh_element_count, mesh_node_coords_ptr, mesh_element_centroids, mesh_element_aabbs, mesh_aabb, timestep);
     
        // Find centroid of the entire mesh
        scene_blas_centroids.emplace_back();
        std::array<double,3>& mesh_centroid = scene_blas_centroids[mesh_idx];
        compute_mesh_centroid(mesh_aabb, mesh_centroid);

        // Temporary vectors to reshuffle element indices as we build the BVH, then using this mapping
        // to append the mesh data in the nodes instead of needing to access it at the split time
        std::vector<int> mesh_element_indices;
        mesh_element_indices.resize(mesh_element_count);
        std::iota(mesh_element_indices.begin(), mesh_element_indices.end(), 0);
        std::vector<int> node_minimum_element_index; // Instead of wasting BLAS_Node struct space on storing this value

         //std::cout << "Generating BLAS for mesh " << mesh_idx << std::endl;
        scene_blases.emplace_back(); // Generate directly inside the vector to avoid copying data
        BLAS& mesh_bvh = scene_blases[mesh_idx]; // Get a reference to the BVH of the current mesh to pass it to the builder functions

        // BLAS BVH builder functions
        build_BLAS(mesh_bvh, mesh_element_centroids, mesh_element_aabbs, mesh_element_indices, node_minimum_element_index, mesh_element_count);
        copy_data_to_BLAS_node_soup(mesh_bvh, mesh_element_indices, node_minimum_element_index, mesh_node_coords_ptr, timestep);
    } //MESHES

}