// STD header files
#include <cmath>
#include "./Eigen/Dense"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <array>
#include <vector>

// nanobind header files
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

// raytracer header files
#include "rteigentypes.h"
#include "rtrender.h"
#include "rtbvh.h"
#include "rtbvh_recursion.h"

namespace nb = nanobind;

void render_scene(const int image_height,
    const int image_width,
    const int number_of_samples,
    const std::vector<nb::ndarray<const int, nb::c_contig>>& scene_connectivity,
    const std::vector<nb::ndarray<const double, nb::c_contig>>& scene_coords,
    const std::vector<nb::ndarray<const double, nb::c_contig>>& scene_face_colors,
    const std::vector<nb::DRef<EiVector3d>> camera_centers,
    const std::vector<nb::DRef<EiVector3d>> pixel_00_centers,
    const std::vector<nb::DRef<Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>>> matrix_pixel_spacings) {

    size_t num_cameras = camera_centers.size();

    build_acceleration_structures_r(scene_connectivity, scene_coords, scene_face_colors);
    //build_acceleration_structures(scene_connectivity, scene_coords, scene_face_colors);

   
    /* // Comment out for preliminary testing of building BVHs
    // Iterate over all cameras and render an image for each
    for (size_t camera_idx = 0; camera_idx < num_cameras; ++camera_idx) {
        EiVector3d camera_center = camera_centers[camera_idx];
        EiVector3d pixel_00_center = pixel_00_centers[camera_idx];
        Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor> matrix_pixel_spacing = matrix_pixel_spacings[camera_idx];

        render_ppm_image(camera_center, pixel_00_center, matrix_pixel_spacing, scene_connectivity, scene_coords, scene_face_colors, image_height, image_width, number_of_samples);
    }
        */
}

NB_MODULE(rtmaincpp, a) {
    a.def("cpp_render_scene", &render_scene);
}
