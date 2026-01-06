/*
// STD header files
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include "./Eigen/Dense"
#include <iostream>
#include <memory>

#include "rtbvh.h"
#include "rtrayintersection.h"
#include "rthitrecord.h"
#include "ndarray.h"

inline void compute_triangle_centroid(int node_0,
    int node_1,
    int node_2,
    const double* mesh_node_coords_ptr,
    std::array<double,3> &triangle_centroid) {
    // Find the centroid of a triangle.
    // Update the value of the passed array, so we don't have to fiddle with structs etc. to return a value.
    for (int i = 0; i < 3; ++i){
        triangle_centroid[i] = (mesh_node_coords_ptr[node_0 * 3 + i] + mesh_node_coords_ptr[node_1 * 3 + i] + mesh_node_coords_ptr[node_2 * 3 + i]) / 3.0;
    }
}

bool intersect_AABB (const Ray& ray, const AABB& AABB) {
    // Slab method for ray-AABB_r intersection
    double t_axis[6]; // t values for each axis, so [0,1] are for x, [2,3] for y, and [4,5] for z
    EiVector3d inverse_direction = 1/(ray.direction.array()); // Divide first to use cheaper multiplication later

    // Find ray intersections with planes defining the AABB_r in X, Y, Z
    for (int i = 0; i < 3; ++i) {
        t_axis[2*i] = (AABB.corner_min[i] - ray.origin(i)) * inverse_direction(i);
        t_axis[2*i + 1] = (AABB.corner_max[i] - ray.origin(i)) * inverse_direction(i);
    }

    //Overlap test
    // Find the minimum t for each axis (x, y, z), then find maximum of these for (x,y,z)
    double t_min = std::max(std::max(std::min(t_axis[0], t_axis[1]), std::min(t_axis[2], t_axis[3])), std::min(t_axis[4], t_axis[5]));
    // Find the maximum t for each axis (x, y, z), then find minimum of these for (x,y,z)
    double t_max = std::min(std::min(std::max(t_axis[0], t_axis[1]), std::max(t_axis[2], t_axis[3])), std::max(t_axis[4], t_axis[5]));

    // t_min < t_max - Ray which just touches a corner, edge, or face of the AABB will be considered non-intersecting
    // t_min <= t_max - Rays which touch the box boundary are considered intersecting. A bit of a degenerate case, but decided to include it here, hence more relaxed inequality.
    // t_min < ray.t_max - Clip to ray segment
    return t_min <= t_max && t_max > 0.0 && t_min < ray.t_max; // False => No overlap => Ray does not intersect the AABB.
}


// Go over all triangles in a mesh and find their AABB and centroids, build mesh AABB, and store the data in vectors
inline void process_element_data_tri3(int mesh_number_of_triangles,
    const int* mesh_connectivity_ptr,
    const double* mesh_node_coords_ptr,
    std::vector<std::array<double,3>>& mesh_element_centroids,
    std::vector<AABB>& mesh_triangle_aabbs,
    AABB& mesh_aabb){
        enum ElementNodeQuantity nodes_per_element = TRI3;
        //int nodes_per_element = 3; // function specirfically for tri3, so we can define it here
        // Iterate over triangles comprising a mesh
        for (int triangle_idx = 0; triangle_idx < mesh_number_of_triangles; triangle_idx++) {
            // Use pointers - means we treat the 2D array as a flat 1D array and do the indexing manually by calculating the offset.
            // HAS to be contiguous in memory for this to work properly! c_contig flag in nanobind ensures that
            int node_0 = mesh_connectivity_ptr[triangle_idx * nodes_per_element + 0]; // Equivalent to indexing as connectivity[triangle_idx, 0]
            int node_1 = mesh_connectivity_ptr[triangle_idx * nodes_per_element + 1];
            int node_2 = mesh_connectivity_ptr[triangle_idx * nodes_per_element + 2];
            // Find centroid for this triangle
            std::array<double,3> triangle_centroid;
            compute_triangle_centroid(node_0, node_1, node_2, mesh_node_coords_ptr, triangle_centroid);
            mesh_element_centroids.push_back(triangle_centroid);

            // Create bounding volume for this triangle
            AABB triangle_aabb;
            triangle_aabb.expand_to_include_node(node_0, mesh_node_coords_ptr);
            triangle_aabb.expand_to_include_node(node_1, mesh_node_coords_ptr);
            triangle_aabb.expand_to_include_node(node_2, mesh_node_coords_ptr);
            mesh_triangle_aabbs.push_back(triangle_aabb);

            // Include triangle AABB in mesh AABB to get the bounding box for the whole thing
            mesh_aabb.expand_to_include_AABB(triangle_aabb);
        } // ELEMENTS/TRIANGLES
    }

// Handles building all acceleration structures in the scene - bottom and top level
// Might not need to pass scene_face_colors. Not sure yet.
void build_acceleration_structures(const std::vector <nanobind::ndarray<const int, nanobind::c_contig>>& scene_connectivity,
    const std::vector <nanobind::ndarray<const double, nanobind::c_contig>>& scene_coords,
    const std::vector<nanobind::ndarray<const double, nanobind::c_contig>>& scene_face_colors){

    // Build BLASes - BVHs for respective meshes
    enum ElementNodeQuantity nodes_per_element = TRI3; // Hard-code for now since we only have triangless.
    size_t num_meshes = scene_coords.size();
    // Create vectors to store centroid and AABB_r data for the scene; might not need these, but have them for now
    std::vector<std::vector<std::array<double,3>>> scene_centroids; // Stores centroids for all meshes in the scene
    //scene_centroids.reserve(num_meshes); // Might not be worth reserving since we'll need num_meshes * num_elements_mesh * 3 * 8 bytes (double) for the whole vector
    std::vector<std::vector<AABB>> scene_aabbs; // Stores AABBs for all elements comprising meshes in this scene
     //scene_aabbs.reserve(num_meshes); // // Might not be worth reserving since we'll need num_meshes * num_elements_mesh * 48 bytes (AABB) for the whole vector
    std::vector<AABB> scene_obj_aabbs; // Stores AABBs of the whole objectes (meshes) in this scene
    scene_obj_aabbs.reserve(num_meshes * sizeof(AABB)); // Can reliably reserve this size and not expect it to change

    // Iterate over MESHES
    for (size_t mesh_idx = 0; mesh_idx < num_meshes; ++mesh_idx) {
        // Access data from the scene for this particular mesh
		nanobind::ndarray<const double, nanobind::c_contig> mesh_node_coords = scene_coords[mesh_idx];
		nanobind::ndarray<const int, nanobind::c_contig> mesh_connectivity = scene_connectivity[mesh_idx];
        nanobind::ndarray<const double, nanobind::c_contig> mesh_face_colors = scene_face_colors[mesh_idx];
        size_t mesh_number_of_elements = mesh_connectivity.shape(0); // number of triangles/faces, will give us indices for some bits
        double* mesh_node_coords_ptr = const_cast<double*>(mesh_node_coords.data());
        int* mesh_connectivity_ptr = const_cast<int*>(mesh_connectivity.data());

        // Containers for calculated data
        std::vector<std::array<double,3>> mesh_element_centroids; // Store centroids for this mesh
        mesh_element_centroids.reserve(mesh_number_of_elements * 3 * sizeof(double));
        std::vector<AABB> mesh_element_aabbs; // Bounding volumes for the elements in this mesh
        mesh_element_aabbs.reserve(mesh_number_of_elements * sizeof(AABB));
        AABB mesh_aabb; // AABB_r for the entire mesh

        // Iterate over ELEMENTS/TRIANGLES in this mesh
        process_element_data_tri3(mesh_number_of_elements, mesh_connectivity_ptr, mesh_node_coords_ptr, mesh_element_centroids, mesh_element_aabbs, mesh_aabb);

        // NDArray for connectivity
        size_t dims[] = {mesh_number_of_elements, static_cast<size_t>(nodes_per_element)};
        size_t n_elems = mesh_number_of_elements * static_cast<size_t>(nodes_per_element);
        size_t n_dims = 2;
        NDArray(int) ndarray_mesh_connectivity;
        ndarray_init(int, &ndarray_mesh_connectivity, mesh_connectivity_ptr, n_elems, &dims[0], n_dims); // Copy data directly from NumPy array buffer
        ndarray_print(int, &ndarray_mesh_connectivity);
        //size_t indices[] = {2,1};
        //int test_index;
        //ndarray_get(int, &ndarray_mesh_connectivity, &indices[0], n_dims, &test_index);
        //std::cout << "index" << test_index << std::endl;
        ndarray_deinit(int, &ndarray_mesh_connectivity);
        

        scene_centroids.push_back(mesh_element_centroids);
        scene_aabbs.push_back(mesh_element_aabbs);
        scene_obj_aabbs.push_back(mesh_aabb);


        // Ok so
        // We need to organise mesh_triangle_aabbs in the trasversal order (i.e., how nodes are organised) and keep it as a part of the BVH
        // Same needs to happen to connectivity array for this mesh then - we want a copy of that, then organise it based on the triangles we have in each node
        // Which needs we also need to do the same for face_colors since they follow the order from the connectivity array? 



    } // MESHES

    // Build TLAS - structure of BLASes. Target is BVH in itself, but use a vector to just contain them for now?

 } // SCENE (end of function)

 */