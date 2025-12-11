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

inline void compute_triangle_centroid(int node_0,
    int node_1,
    int node_2,
    const double* mesh_node_coords_ptr,
    std::array<double,3> &triangle_centroid);

// Bounding volume structure - axis-aligned bounding boxes (AABB_r)
struct AABB {
    double corner_min[3]{};
    double corner_max[3]{};

    AABB() {
        corner_min[0] = corner_min[1] = corner_min[2] = std::numeric_limits<double>::infinity();
        corner_max[0] = corner_max[1] = corner_max[2] = -std::numeric_limits<double>::infinity();
    }

    // Note on comparisons in the expand functions:
    // If any of the numbers is NaN, <, >, etc. will return false -> Box won't be overwritten with bad data
    // Hence no checks like std::isnan or if in the for loops to avoid the function call and branching overhead

     // Used for building AABBs for all mesh triangles
    void expand_to_include_node(const int& node_id,
        const double* mesh_node_coords_ptr){
        if (!mesh_node_coords_ptr) return;
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
        return std::max(0.0, corner_max[axis] - corner_min[axis]); // In case we somehow get a negative extent
    }

    double find_surface_area() const {
        double box_dims[3]; // height, width, depth of AABB
        for (int i = 0; i < 3; ++i){
            box_dims[i] = find_axis_extent(i);
        return 2 * (box_dims[0] * box_dims[1] + box_dims[1] * box_dims[2] + box_dims[0] * box_dims[2]);
    }

    inline bool is_valid() const {
        // Safety function in case it is needed (likely if there is all-NaN input data only)
        for (int i = 0; i < 3; ++i){
            if (std::isnan(corner_min[i]) || std::isnan(corner_max[i])) return false; // Check for NaN values
            if (!std::isfinite(corner_min[i]) || !std::isfinite(corner_max[i])) return false; // Check for infinity
            if (corner_min[i] > corner_max[i]) return false;  // Degenerate or empty
        }
        return true;
    }
};
