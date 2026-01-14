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
enum ElementNodeCount {
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

/*
inline void compute_triangle_centroid(int node_0,
    int node_1,
    int node_2,
    const double* mesh_node_coords_ptr,
    std::array<double,3> &triangle_centroid);
*/

// Bounding volume structure - axis-aligned bounding boxes (AABB)
struct AABB {
    double corner_min[3]{};
    double corner_max[3]{};

    AABB() {
        corner_min[0] = corner_min[1] = corner_min[2] = std::numeric_limits<double>::infinity();
        corner_max[0] = corner_max[1] = corner_max[2] = -std::numeric_limits<double>::infinity();
    }

    // Used for building AABBs for all mesh triangles
    inline void build_for_tri3(const std::array<double,9> &triangle_node_coords){
        for (int node = 0; node < 3; ++node) {
        const int offset = node * 3;
            for (int i = 0; i < 3; ++i) {
                const double nodal_coordinate = triangle_node_coords[offset + i];
                corner_min[i] = std::min(corner_min[i], nodal_coordinate);
                corner_max[i] = std::max(corner_max[i], nodal_coordinate);
            }
        }
    }   

     // Used for SAH splitting
    inline void expand_to_include_point(const std::array<double,3>& point){
        for (int i = 0; i < 3; ++i){
            double point_coordinate = point[i];
            corner_min[i] = std::min(corner_min[i], point_coordinate);
            corner_max[i] = std::max(corner_max[i], point_coordinate);
        }
    }
     // Used for creating child node AABBs
    inline void expand_to_include_AABB(const AABB& other) {
        for (int i = 0; i < 3; ++i){
            corner_min[i] = std::min(corner_min[i], other.corner_min[i]);
            corner_max[i] = std::max(corner_max[i], other.corner_max[i]);
        }
    }

    inline double find_axis_extent(int axis) const {
        return corner_max[axis] - corner_min[axis];
    }
    inline double find_surface_area() const {
        double height = find_axis_extent(2);
        double width = find_axis_extent(1);
        double depth = find_axis_extent(0);
        return 2 * (height * width + width * depth + height * depth);
    }
};

inline void compute_mesh_centroid_it(AABB mesh_aabb,
    std::array<double,3>& mesh_centroid);

struct Bin {
    // Bin for binning SAH
    AABB bounding_box {};
    int element_count {0};
};

// Struct used as a placeholder in BLAS and TLAS builder
struct BuildTask {
    size_t element_count;      // number of elements
    int node_idx;
    int min_element_idx;      // first triangle index in tri_indices
};

struct BVH_Node {
    // Coordinates of nodes comprising the triangles stored in the node, if applicable
    std::vector<double> node_coords; // Stored as x,y,z. Node coords of contained elements in this node
    std::vector<double> face_color; // , as we access those per element in this order
    AABB bounding_box {};
    size_t element_count {0}; // If not zero, this is the leaf
    //int min_elem_idx {-1};
    enum ElementNodeCount nodes_per_element {TRI3}; // Assign 3 by default for now since we only do triangles
    int left_child_idx {-1};
   // int right_child_idx {-1}; // Remove this later as it's left + 1, but keep while developing as it makes it easier to see obvious issues with the code
    
};

// Keeping BVH as a struct for now since we may have to store the global texture data here. TBDetermined.
struct BVH {
    std::vector<BVH_Node> tree_nodes;
    AABB bounding_box;
    int root_idx {-1};

    BVH() = default;
};

struct TLAS_Node {
    AABB bounding_box {};
    int blas_count {0}; // Number of BLASes in this node (consecutive in the array)
    int left_child_idx {-1};
    int min_blas_idx {-1}; // Store this instead of data as we expect a few meshes in the scene tops, so indexing into BLAS vector shouldn't be too awful

    TLAS_Node() = default;
    TLAS_Node(AABB aabb, int count, int left_idx, int min_blas_idx):
        bounding_box(aabb),
        blas_count(count),
        left_child_idx(left_idx),
        min_blas_idx(min_blas_idx)
        {};
};

struct TLAS {
    std::vector<BVH> blases;
    std::vector<TLAS_Node> tlas_nodes;
};

// Binned Surface Area Heuristic (SAH) split
bool binned_sah_split(BVH_Node& Node,
    const std::vector<std::array<double,3>>& mesh_triangle_centroids,
    const std::vector<AABB>& mesh_triangle_aabbs,
    const std::vector<int>& mesh_triangle_indices,
    unsigned int& out_split_axis,
    double& out_split_position);

/**
 void build_bvh(BVH_Node& Node,
    const std::vector<std::array<double,3>>& mesh_triangle_centroids,
    const std::vector<AABB>& mesh_triangle_aabbs,
    std::vector<int>& mesh_triangle_indices);
*/

TLAS build_acceleration_structures(const std::vector <nanobind::ndarray<const double, nanobind::c_contig>>& scene_coords_expanded,
    const std::vector<nanobind::ndarray<const double, nanobind::c_contig>>& scene_face_colors);

// Handles building all acceleration structures in the scene - bottom and top level
// Might not need to pass scene_face_colors. Not sure yet.
/*
void build_acceleration_structures(const std::vector <nanobind::ndarray<const int, nanobind::c_contig>>& scene_connectivity,
    const std::vector <nanobind::ndarray<const double, nanobind::c_contig>>& scene_coords,
    const std::vector<nanobind::ndarray<const double, nanobind::c_contig>>& scene_face_colors);*/