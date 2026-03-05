#ifndef ACCESS_TEST_H
#define ACCESS_TEST_H

#include <vector>
#include <array>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include "rtbvh.h"

constexpr int NODE_COORDINATES = 3;

void process_element_data_double_array(int mesh_number_of_elements,
    const double* mesh_node_coords_ptr,
    const int* mesh_connectivity_ptr,
    std::vector<std::array<double,3>>& mesh_triangle_centroids,
    std::vector<AABB>& mesh_triangle_aabbs,
    AABB& mesh_aabb);

void copy_data_to_BLAS_node_soup(BLAS &mesh_bvh,
    std::vector<int>& mesh_element_indices,
    std::vector<int>& node_minimum_element_index,
    const double* mesh_node_coords_expanded_ptr,
    const int timestep);

void copy_data_to_BLAS_node_double_array(BLAS &mesh_bvh,
    std::vector<int>& mesh_element_indices,
    std::vector<int>& node_minimum_element_index,
    const double* mesh_node_coords_ptr,
    const int* mesh_connectivity_ptr,
    const int timestep);

void compare_bvh_access_time_double_array(const std::vector <nanobind::ndarray<const int, nanobind::c_contig>>& scene_connectivity,
    const std::vector <nanobind::ndarray<const double, nanobind::c_contig>>& scene_coords);

void compare_bvh_access_time_soup(const std::vector <nanobind::ndarray<const double,nanobind::c_contig>>& scene_coords_expanded,
    const std::vector<size_t>& mesh_element_counts);

#endif // ACCESS_TEST_H
