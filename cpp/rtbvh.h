#pragma once

#include <array>
#include <vector>
#include <memory>
#include <limits>
#include <string>
#include <iostream>

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include "rteigentypes.h"
#include "rtrender.h"
#include "rtray.h"

// Enum storing the number of nodes per element, so we can update it later nicely when we add different types? At least that's the idea.
enum ElementNodeQuantity {
    TRI3 = 3,
    QUAD4 = 4,
    TET4 = 4,
    TET10 = 10,
    TET14 = 14,
    HEX8 = 8,
    HEX20 = 20,
    HEX27 = 27
};

inline void shuffle_flat_connectivity(std::vector<int>& connectivity,
                                      int old_element_index,
                                      int new_element_index,
                                      int number_of_nodes);

inline void compute_triangle_centroid(int node_0,
    int node_1,
    int node_2,
    const double* mesh_node_coords_ptr,
    std::array<double,3> &triangle_centroid);
    

// Bounding volume structure - axis-aligned bounding boxes (AABB)
struct AABB {
    double corner_min[3]{};
    double corner_max[3]{};

    AABB() {
        corner_min[0] = corner_min[1] = corner_min[2] = std::numeric_limits<double>::infinity();
        corner_max[0] = corner_max[1] = corner_max[2] = -std::numeric_limits<double>::infinity();
    }
     // Used for building AABBs for all mesh triangles
    void expand_to_include_node(const int& node_id,
        const double* mesh_node_coords_ptr){
        for (int i = 0; i < 3; ++i){
             double nodal_coordinate = mesh_node_coords_ptr[node_id * 3 + i];
             if (nodal_coordinate < corner_min[i]) corner_min[i] = nodal_coordinate;
             if (nodal_coordinate > corner_max[i]) corner_max[i] = nodal_coordinate;
        }
    }
     // Used for SAH splitting
     void expand_to_include_point(const std::array<double,3>& point){
        for (int i = 0; i < 3; ++i){
            double point_coordinate = point[i];
            if (point_coordinate < corner_min[i]) corner_min[i] = point_coordinate;
            if (point_coordinate > corner_max[i]) corner_max[i] = point_coordinate;
        }
    }
     // Used for creating child node AABBs
    void expand_to_include_AABB(const AABB& other) {
        for (int i = 0; i < 3; ++i){
            if (other.corner_min[i] < corner_min[i]) corner_min[i] = other.corner_min[i];
            if (other.corner_max[i] > corner_max[i]) corner_max[i] = other.corner_max[i];
        }
    }

    inline double find_axis_extent(int axis) const {
        return corner_max[axis] - corner_min[axis];
    }
    double find_surface_area() const {
        double height = find_axis_extent(2);
        double width = find_axis_extent(1);
        double depth = find_axis_extent(0);
        return 2 * (height * width + width * depth + height * depth);
    }
};

inline void compute_mesh_centroid_it(AABB mesh_aabb,
    std::array<double,3>& mesh_centroid);

// BVH node structure - naive implementation with pointers for now. Replace with indices once functional to save a few bytes
struct BVH_Node {
    AABB bounding_box {};
    // Unique pointers to prevent memory leaks with raw pointers and new BVH_Node use
    std::unique_ptr<BVH_Node> left_child;
    std::unique_ptr<BVH_Node> right_child;
    //BVH_Node* left_child {nullptr}; // Nullptr if leaf.
    //BVH_Node* right_child {nullptr};
    int min_triangle_idx;
    int triangle_count;
};

AABB create_node_AABB_it(const std::vector<AABB>& mesh_triangle_abbs,
    const std::vector<int>& mesh_triangle_indices,
    const int node_min_triangle_idx,
    const int node_triangle_count);


struct BVH {
    //std::vector<BVH_Node> nodes;
    std::unique_ptr<BVH_Node> root;
    std::vector<int> triangle_indices; // Triangle indices that will be swapped in splitting to avoid modifying the data passed from Python
    double* mesh_node_coords_ptr; // pointer to contiguous array of mesh node coordinates
    int* mesh_connectivity_ptr; // pointer to contiguous array of mesh connectivity
    double* mesh_face_colors_ptr; // pointer to contiguous array of mesh face colors
};


struct Bin {
    // Bin for binning SAH
    AABB bounding_box {};
    int element_count {0};
};


// Binned Surface Area Heuristic (SAH) split
bool binned_sah_split(BVH_Node& Node,
    const std::vector<std::array<double,3>>& mesh_triangle_centroids,
    const std::vector<AABB>& mesh_triangle_aabbs,
    const std::vector<int>& mesh_triangle_indices,
    unsigned int& out_split_axis,
    double& out_split_position);


 void build_bvh(BVH_Node& Node,
    const std::vector<std::array<double,3>>& mesh_triangle_centroids,
    const std::vector<AABB>& mesh_triangle_aabbs,
    std::vector<int>& mesh_triangle_indices);


// Handles building all acceleration structures in the scene - bottom and top level
// Might not need to pass scene_face_colors. Not sure yet.
void build_acceleration_structures(const std::vector <nanobind::ndarray<const int, nanobind::c_contig>>& scene_connectivity,
    const std::vector <nanobind::ndarray<const double, nanobind::c_contig>>& scene_coords,
    const std::vector<nanobind::ndarray<const double, nanobind::c_contig>>& scene_face_colors);