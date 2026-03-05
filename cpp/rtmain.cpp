// STD header files
#include <cmath>
#include "./Eigen/Dense"
#include <filesystem>
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
#include <nanobind/stl/filesystem.h>

// raytracer header files
#include "rteigentypes.h"
//#include "rtbvh.h"
#include "rtrender.h"
//#include "rtbvh_recursion.h"
//#include "rtbvh_stack.h"
#include "access_test.h"

namespace nb = nanobind;

/*
void render_scene(const int image_height,
    const int image_width,
    const int number_of_samples,
    const std::filesystem::path output_directory,
    const int timestep_count,
    const std::vector<nb::ndarray<const double, nb::c_contig>>& scene_coords_expanded,
    const std::vector<nb::ndarray<const double, nb::c_contig>>& scene_face_colors,
    const std::vector<nb::DRef<EiVector3d>> camera_centers,
    const std::vector<nb::DRef<EiVector3d>> pixel_00_centers,
    const std::vector<nb::DRef<Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>>> matrix_pixel_spacings) {


    CALLGRIND_START_INSTRUMENTATION;
    size_t num_cameras = camera_centers.size();
    // Use std::filesystem so it always constructs the path properly for the running OS
    std::filesystem::path output_filepath;
    std::string filename; // Output image file
    

    //std::chrono::time_point t1_d = std::chrono::high_resolution_clock::now();
    for (int timestep = 0; timestep < timestep_count; ++timestep){
        TLAS test_TLAS = build_acceleration_structures(scene_coords_expanded, scene_face_colors, timestep, timestep_count); // target stack-based DoD implementation
        
        // Iterate over all cameras and render an image for each
        for (size_t camera_idx = 0; camera_idx < num_cameras; ++camera_idx) {
            EiVector3d camera_center = camera_centers[camera_idx];
            EiVector3d pixel_00_center = pixel_00_centers[camera_idx];
            Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor> matrix_pixel_spacing = matrix_pixel_spacings[camera_idx];
            // Create the filepath for the rendered images
            filename = "rtimage_" + std::to_string(timestep) + "_cam" + std::to_string(camera_idx) + ".ppm"; // Output images in format rtimage_1_cam1 etc.
            output_filepath = output_directory;
            output_filepath.append(filename);
            std::cout << "Rendering frame " << (timestep+1) << "/" << timestep_count << std::endl;
            render_ppm_image(camera_center, pixel_00_center, matrix_pixel_spacing, test_TLAS, image_height, image_width, number_of_samples, output_filepath);
        }
    }

    //std::chrono::time_point t2_d = std::chrono::high_resolution_clock::now();
    //std::chrono::duration t_d = std::chrono::duration_cast<std::chrono::nanoseconds>(t2_d - t1_d);
    //std::cout << "Iterative, DoD approach duration: " << t_d.count() << "ns \n";

    
        
    CALLGRIND_STOP_INSTRUMENTATION;
}
*/

void compare_data_access(
    std::vector<nb::ndarray<const double, nb::c_contig>>& scene_coords_expanded,
    std::vector<nb::ndarray<const int, nb::c_contig>>& scene_connectivity,
    std::vector<nb::ndarray<const double, nb::c_contig>>& scene_coords) {
 // Purpose: Compare if it's better to have connectivity and node coordinate arrays separately, and index into them, or use "soup", which is the original approach with single array.
 // Test will be conducted in "operational" conditions, i.e., the test functions follow the same pattern as BVH (BLAS) builder - because at later stages, we copy the data anyway, so it doesn't matter.

   // Temp to get number of elements for second test
    size_t num_meshes = scene_coords.size();
    std::vector<size_t> mesh_element_counts(num_meshes);
    for (size_t mesh_idx = 0; mesh_idx < num_meshes; ++mesh_idx) {
        // Access data from the scene for this particular mesh
		nanobind::ndarray<const double, nanobind::c_contig> mesh_node_coords = scene_coords[mesh_idx];
		nanobind::ndarray<const int, nanobind::c_contig> mesh_connectivity = scene_connectivity[mesh_idx];

        size_t mesh_number_of_elements = mesh_connectivity.shape(0); // number of triangles/faces, will give us indices for some bits
        mesh_element_counts[mesh_idx] = mesh_number_of_elements;
    }

    CALLGRIND_START_INSTRUMENTATION;
    std::chrono::time_point t1_r = std::chrono::high_resolution_clock::now();
    compare_bvh_access_time_double_array(scene_connectivity, scene_coords);
    std::chrono::time_point t2_r = std::chrono::high_resolution_clock::now();

    std::chrono::time_point t1_s = std::chrono::high_resolution_clock::now();
    compare_bvh_access_time_soup(scene_coords_expanded, mesh_element_counts);
    std::chrono::time_point t2_s = std::chrono::high_resolution_clock::now();

    std::cout << "Access time for double array approach: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2_r - t1_r).count() << " ns \n";
    std::cout << "Access time for soup approach: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2_s - t1_s).count() << " ns \n";
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
    a.def("compare_data_access", &compare_data_access);
}

/*
NB_MODULE(rtmaincpp, a) {
    a.def("cpp_render_scene", &render_scene);
}*/