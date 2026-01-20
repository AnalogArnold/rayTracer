#pragma once
// STD header files
#include <array>
#include <string>
#include <vector>

// nanobind header files
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

// raytracer header files
#include "rteigentypes.h"
#include "rtray.h"
#include "rtbvh.h"

EiVector3d return_ray_color(const Ray& ray,
    const TLAS& TLAS);

void render_ppm_image(const EiVector3d &camera_center,
    const EiVector3d &pixel_00_center,
    const Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor> &matrix_pixel_spacing,
    const TLAS& TLAS,
    const int image_height,
    const int image_width,
    const int number_of_samples,
    const std::string filename);

/* Version with pointers for no BVH, rtbvh_stack, and rtbvh_recursion
inline EiVector3d get_color(Eigen::Index minRowIndex,
    const double* face_color_ptr);

EiVector3d return_ray_color(const Ray& ray,
    const std::vector < nanobind::ndarray<const int, nanobind::c_contig>>& scene_connectivity,
    const std::vector < nanobind::ndarray<const double, nanobind::c_contig>>& scene_coords,
    const std::vector<nanobind::ndarray<const double, nanobind::c_contig>>& scene_face_colors);


void render_ppm_image(const EiVector3d &camera_center,
    const EiVector3d &pixel_00_center,
    const Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor> &matrix_pixel_spacing,
    const std::vector < nanobind::ndarray<const int, nanobind::c_contig>>& scene_connectivity,
    const std::vector < nanobind::ndarray<const double, nanobind::c_contig>>& scene_coords,
    const std::vector<nanobind::ndarray<const double, nanobind::c_contig>>& scene_face_colors,
    const int image_height,
    const int image_width,
    const int number_of_samples);
    */