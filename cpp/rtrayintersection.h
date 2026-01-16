#pragma once

// STD header files
#include <array>

// ray tracer header files
#include "rteigentypes.h"
#include "rtray.h"
#include "rthitrecord.h"
#include "rtbvh.h"

inline EiVector3d get_face_color(Eigen::Index minRowIndex,
    std::vector<double>& face_color);

struct IntersectionOutput {
    Eigen::ArrayXXd barycentric_coordinates;
    EiVectorD3d plane_normals;
    Eigen::Array<double, Eigen::Dynamic, 1> t_values;
};

EiVectorD3d cross_rowwise(const EiVectorD3d& mat1, const EiVectorD3d& mat2);

IntersectionOutput intersect_bvh_triangles(const Ray& ray,
    const std::vector<double>& node_coords,
    const unsigned int bvh_node_triangle_count);

void intersect_BLAS(const Ray& ray,
    const BLAS& mesh_bvh,
    IntersectionOutput &out_intersection,
    HitRecord &intersection_record);

void intersect_TLAS(const Ray& ray,
    const TLAS& scene_TLAS,
    IntersectionOutput &out_intersection,
    HitRecord& out_intersection_record);

/* Version with pointers. Keeping it for potential rtbvh_stack and rtbvh_recursion tests
// PRE-BVH
IntersectionOutput intersect_triangle(const Ray& ray,
    const int* connectivity_ptr,
    const double* node_coords_ptr,
    const long long number_of_elements);

// For rtbvh_stack and rtbvh_recursion tests
IntersectionOutput intersect_bvh_triangles(const Ray& ray,
    const int* connectivity_ptr,
    const double* node_coords_ptr,
    const unsigned int bvh_node_triangle_count,
    const std::vector<unsigned int>& bvh_node_triangle_indices);
*/
