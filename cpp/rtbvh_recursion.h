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


inline void compute_triangle_centroid_r(int node_0,
    int node_1,
    int node_2,
    const double* mesh_node_coords_ptr,
    std::array<double,3> &triangle_centroid);

// Bounding volume structure - axis-aligned bounding boxes (AABB_r)
struct AABB_r {
    double corner_min[3]{};
    double corner_max[3]{};

    AABB_r() {
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
    void expand_to_include_AABB(const AABB_r& other) {
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

// BVH node structure - naive implementation with pointers for now. Replace with indices once functional to save a few bytes
struct BVH_Node_r {
    int min_triangle_idx;
    int triangle_count;
    AABB_r bounding_box {};
    // Unique pointers to prevent memory leaks with raw pointers and new BVH_Node_r use
    std::unique_ptr<BVH_Node_r> left_child;
    std::unique_ptr<BVH_Node_r> right_child;
    //BVH_Node_r* left_child {nullptr}; // Nullptr if leaf.
    //BVH_Node_r* right_child {nullptr};
};

AABB_r create_node_AABB_r(const std::vector<AABB_r>& mesh_triangle_abbs,
    const std::vector<int>& mesh_triangle_indices,
    const int node_min_triangle_idx,
    const int node_triangle_count);

/*
struct BVH {
    std::vector<BVH_Node_r> nodes;
    std::vector<unsigned int> triangle_indices; // Triangle indices that will be swapped in splitting to avoid modifying the data passed from Python
    std::unique_ptr<BVH_Node_r> root;
};
*/



struct Bin_r {
    // Bin_r for binning SAH
    AABB_r bounding_box {};
    int element_count {0};
};


// Binned Surface Area Heuristic (SAH) split
bool binned_sah_split_r(BVH_Node_r& Node,
    const std::vector<std::array<double,3>>& mesh_triangle_centroids,
    const std::vector<AABB_r>& mesh_triangle_aabbs,
    const std::vector<int>& mesh_triangle_indices,
    unsigned int& out_split_axis,
    double& out_split_position);

 void build_bvh_r(BVH_Node_r& Node,
    const std::vector<std::array<double,3>>& mesh_triangle_centroids,
    const std::vector<AABB_r>& mesh_triangle_aabbs,
    std::vector<int>& mesh_triangle_indices);


// Handles building all acceleration structures in the scene - bottom and top level
// Might not need to pass scene_face_colors. Not sure yet.
void build_acceleration_structures_r(const std::vector <nanobind::ndarray<const int, nanobind::c_contig>>& scene_connectivity,
    const std::vector <nanobind::ndarray<const double, nanobind::c_contig>>& scene_coords,
    const std::vector<nanobind::ndarray<const double, nanobind::c_contig>>& scene_face_colors);