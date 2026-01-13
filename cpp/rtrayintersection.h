#pragma once

// STD header files
#include <array>

// ray tracer header files
#include "rteigentypes.h"
#include "rtray.h"

struct IntersectionOutput {
    Eigen::ArrayXXd barycentric_coordinates;
    EiVectorD3d plane_normals;
    Eigen::Array<double, Eigen::Dynamic, 1> t_values;
};

EiVectorD3d cross_rowwise(const EiVectorD3d& mat1, const EiVectorD3d& mat2);

IntersectionOutput intersect_triangle(const Ray& ray,
    const int* connectivity_ptr,
    const double* node_coords_ptr,
    const long long number_of_elements);

IntersectionOutput intersect_bvh_triangles(const Ray& ray,
    const std::vector<double>& node_coords,
    const unsigned int bvh_node_triangle_count);

/*
IntersectionOutput intersect_bvh_triangles(const Ray& ray,
    const int* connectivity_ptr,
    const double* node_coords_ptr,
    const unsigned int bvh_node_triangle_count,
    const std::vector<unsigned int>& bvh_node_triangle_indices);
*/