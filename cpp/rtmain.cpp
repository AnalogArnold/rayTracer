// STD header files
#include <cmath>
#include "./Eigen/Dense"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <array>
#include <vector>
#include <chrono>
#include <valgrind/callgrind.h>

// nanobind header files
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

// raytracer header files
#include "rteigentypes.h"
#include "rtbvh.h"
#include "rtrender.h"
#include "rtbvh_recursion.h"
#include "rtbvh_stack.h"

namespace nb = nanobind;

void render_scene(const int image_height,
    const int image_width,
    const int number_of_samples,
    const int timestep,
    const std::vector<nb::ndarray<const double, nb::c_contig>>& scene_coords_expanded,
    const std::vector<nb::ndarray<const double, nb::c_contig>>& scene_face_colors,
    const std::vector<nb::DRef<EiVector3d>> camera_centers,
    const std::vector<nb::DRef<EiVector3d>> pixel_00_centers,
    const std::vector<nb::DRef<Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>>> matrix_pixel_spacings) {


    CALLGRIND_START_INSTRUMENTATION;
    size_t num_cameras = camera_centers.size();
    

    //std::chrono::time_point t1_d = std::chrono::high_resolution_clock::now();
    TLAS test_TLAS = build_acceleration_structures(scene_coords_expanded, scene_face_colors); // target stack-based DoD implementation
    //std::chrono::time_point t2_d = std::chrono::high_resolution_clock::now();
    //std::chrono::duration t_d = std::chrono::duration_cast<std::chrono::nanoseconds>(t2_d - t1_d);
    //std::cout << "Iterative, DoD approach duration: " << t_d.count() << "ns \n";

    std::string filename; // Output image file
    // Iterate over all cameras and render an image for each
    for (size_t camera_idx = 0; camera_idx < num_cameras; ++camera_idx) {
        EiVector3d camera_center = camera_centers[camera_idx];
        EiVector3d pixel_00_center = pixel_00_centers[camera_idx];
        Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor> matrix_pixel_spacing = matrix_pixel_spacings[camera_idx];

        filename = "rtimage_" + std::to_string(timestep) + "_cam" + std::to_string(camera_idx) + ".ppm"; // Output images in format rtimage_1_cam1 etc.
        render_ppm_image(camera_center, pixel_00_center, matrix_pixel_spacing, test_TLAS, image_height, image_width, number_of_samples, filename);
    }
        
    CALLGRIND_STOP_INSTRUMENTATION;
}

/* Commented out - using connectivity and nodal coordinates, not expanded. Keeping it for tests with rtbvh_recursion and rtbvh_stack 
void render_scene(const int image_height,
    const int image_width,
    const int number_of_samples,
    const std::vector<nb::ndarray<const int, nb::c_contig>>& scene_connectivity,
    const std::vector<nb::ndarray<const double, nb::c_contig>>& scene_coords,
    const std::vector<nb::ndarray<const double, nb::c_contig>>& scene_face_colors,
    const std::vector<nb::DRef<EiVector3d>> camera_centers,
    const std::vector<nb::DRef<EiVector3d>> pixel_00_centers,
    const std::vector<nb::DRef<Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>>> matrix_pixel_spacings) {

    //CALLGRIND_START_INSTRUMENTATION;
    size_t num_cameras = camera_centers.size();

    //std::chrono::time_point t1_r = std::chrono::high_resolution_clock::now();
    //build_acceleration_structures_r(scene_connectivity, scene_coords, scene_face_colors); // recursive implementation with pointers
    //std::chrono::time_point t2_r = std::chrono::high_resolution_clock::now();

    //std::chrono::time_point t1_i = std::chrono::high_resolution_clock::now();
    //build_acceleration_structures_it(scene_connectivity, scene_coords, scene_face_colors); // stack-based implementation with pointers
    //std::chrono::time_point t2_i = std::chrono::high_resolution_clock::now();

    //std::chrono::duration t_r = std::chrono::duration_cast<std::chrono::nanoseconds>(t2_r - t1_r);
    //std::chrono::duration t_i = std::chrono::duration_cast<std::chrono::nanoseconds>(t2_i - t1_i);

    //std::cout << "Recursive, pointer approach duration: " << t_r.count() << "ns \n";
    //std::cout << "Iterative, pointer approach duration: " << t_i.count() << "ns \n";


     // Comment out for preliminary testing of building BVHs
    // Iterate over all cameras and render an image for each
    for (size_t camera_idx = 0; camera_idx < num_cameras; ++camera_idx) {
        EiVector3d camera_center = camera_centers[camera_idx];
        EiVector3d pixel_00_center = pixel_00_centers[camera_idx];
        Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor> matrix_pixel_spacing = matrix_pixel_spacings[camera_idx];

        //render_ppm_image(camera_center, pixel_00_center, matrix_pixel_spacing, scene_connectivity, scene_coords, scene_face_colors, image_height, image_width, number_of_samples);
    }
        
    //CALLGRIND_STOP_INSTRUMENTATION;
}
*/

NB_MODULE(rtmaincpp, a) {
    a.def("cpp_render_scene", &render_scene);
}
