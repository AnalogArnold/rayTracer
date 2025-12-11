// STD header files
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include "./Eigen/Dense"
#include <iostream>
#include <memory>

#include "rtbvh_recursion.h"
#include "rtrayintersection.h"
#include "rthitrecord.h"
#include "ndarray.h"


/*
// nanobind header files
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

// raytracer header files
#include "rteigentypes.h"
#include "rtrender.h"
#include "rtray.h"
*/

// TO-DO'S:
// Update main to pass data and construct bvh; then rtrender and rtrayintersection to reflect these changes
// AABBs - Tested. Updated adding the point, since triangle node data isn't passed as EiVector3d.
// compute_triangle_centroid - Tested. Updated to reflect how triangle data is passed.
// Bin_r - Don't touch

// intersect_AABB - 1. Run test cases separately to make sure results are okay. 2. Bullet-proof it based on https://tavianator.com/2022/ray_box_boundary.html
// BVHNode - Decide on the final data layout
// intersect_bvh_r - Update with iterating over triangles (depends on node layout, though)
// find_SAH_cost functions - For now keep the simplified version.
// binned_sah_split, split_node  - Same as intersect_bvh_r. Update arguments. Test for corner cases and maybe add fail-safe option if SAH splitting fails.
// build_bvh - Write it (depends on everything above, though)

 inline void compute_triangle_centroid_r(int node_0,
    int node_1,
    int node_2,
    const double* mesh_node_coords_ptr,
    std::array<double,3> &triangle_centroid) {
    // Find the centroid of a triangle.
    // Update the value of the passed array, so we don't have to fiddle with structs etc. to return a value.
    for (int i = 0; i < 3; ++i){
        triangle_centroid[i] = (mesh_node_coords_ptr[node_0 * 3 + i] + mesh_node_coords_ptr[node_1 * 3 + i] + mesh_node_coords_ptr[node_2 * 3 + i]) / 3.0;
    }
}

AABB_r create_node_AABB_r(const std::vector<AABB_r>& mesh_triangle_abbs,
    const std::vector<int>& mesh_triangle_indices,
    const int node_min_triangle_idx,
    const int node_triangle_count) {
    // Iterates over all triangles assigned to the node to find its bounding box
    int node_max_triangle_idx = node_min_triangle_idx + node_triangle_count;
    AABB_r node_AABB;

    for (int i = node_min_triangle_idx; i < node_max_triangle_idx; ++i) {
        int triangle_idx = mesh_triangle_indices[i];
        node_AABB.expand_to_include_AABB(mesh_triangle_abbs[triangle_idx]);
    }
    return node_AABB;
}

// For performance improvement and corner cases, later have a look at: https://tavianator.com/2022/ray_box_boundary.html
bool intersect_AABB_r (const Ray& ray, const AABB_r& AABB_r) {
    // Slab method for ray-AABB_r intersection
    double t_min[3];
    double t_max[3];
    EiVector3d inverse_direction = 1/(ray.direction.array()); // Divide first to use cheaper multiplication later

    // Find ray intersections with planes defining the AABB_r in X, Y, Z
    for (int i = 0; i < 3; ++i) {
        t_min[i] = (AABB_r.corner_min[i] - ray.origin(i)) * inverse_direction(i);
        t_max[i] = (AABB_r.corner_max[i] - ray.origin(i)) * inverse_direction(i);
        if (t_min[i] > t_max[i]) std::swap(t_min[i], t_max[i]);
    }

    //Overlap test
    // Find min and max values out of all t-values found
    double t_close = t_min[0];
    double t_far = t_max[0];
    for (int i = 1; i < 3; ++i) {
        if (t_min[i] > t_close) t_close = t_min[i];
        if (t_max[i] < t_far) t_far = t_max[i];
    }

    return t_close < t_far; // False => No overlap => Ray does not intersect the AABB_r
}

/*
struct BVH {
    std::vector<BVH_Node_r> nodes;
    std::vector<unsigned int> triangle_indices; // Triangle indices that will be swapped in splitting to avoid modifying the data passed from Python
    std::unique_ptr<BVH_Node_r> root;
};
*/

// Auxiliary functions for splitting and binning
double find_SAH_cost_bin_r(unsigned int left_element_count, unsigned int right_element_count, const AABB_r& left_bounds, const AABB_r& right_bounds) {
    // Calculate the Surface Area Heuristic (SAH) cost of a node. Simplified equation for initial implementation.
   return (double)left_element_count * left_bounds.find_surface_area() + (double)right_element_count * right_bounds.find_surface_area(); // Static casts complained so leave C-style casts for now
}
/*
struct Bin_r {
    // Bin_r for binning SAH
    AABB_r bounding_box {};
    int element_count {0};
};

*/

// Binned Surface Area Heuristic (SAH) split
bool binned_sah_split_r(BVH_Node_r& Node,
    const std::vector<std::array<double,3>>& mesh_triangle_centroids,
    const std::vector<AABB_r>& mesh_triangle_aabbs,
    const std::vector<int>& mesh_triangle_indices,
    unsigned int& out_split_axis,
    double& out_split_position) {

    if (Node.triangle_count <= 2) return false; // Too small to split
    unsigned int node_max_triangle_idx = Node.min_triangle_idx + Node.triangle_count;

    // Compute centroid bounds for the node
    // We use existing AABB_r since it nicely implements everything we need, BUT it is not to be confused with the actual bounding box of the node
    // node_centroid_bounds - Only used to determine splitting
    // bounding_box - Actual bounding box of the node used for ray intersections
    AABB_r node_centroid_bounds{};
    for (int i = Node.min_triangle_idx; i < node_max_triangle_idx; ++i) {
        // Retrieve triangle and its centroid on the split axis
        unsigned int triangle_idx = mesh_triangle_indices[i];
        std::array<double,3> triangle_centroid = mesh_triangle_centroids[triangle_idx];
        node_centroid_bounds.expand_to_include_point(triangle_centroid);
    }

    // Pick the longest axis for splitting
    int best_axis = 0;
    double axis_extent = node_centroid_bounds.find_axis_extent(best_axis);
    for (int i = 1; i < 3; ++i) {
        double temp_extent = node_centroid_bounds.find_axis_extent(i);
        if (temp_extent > axis_extent){
            best_axis = i;
            axis_extent = temp_extent;
        }
    }
    if (axis_extent == 0) return false; // All centroids coincident along the chosen axis => No useful split
    // Might implement fallback split here (median et.c) here later rather than in the main build_bvh function?
    out_split_axis = best_axis;

    // Create bins
    constexpr int NUM_BINS = 8;
    Bin_r bins[NUM_BINS];

    const double inverse_extent = 1.0/axis_extent;
    for (unsigned int i = Node.min_triangle_idx; i < node_max_triangle_idx; ++i){
        unsigned int triangle_idx = mesh_triangle_indices[i];
        // Find the Bin_r containing the triangle centroid
        double t = (mesh_triangle_centroids[triangle_idx][best_axis] - node_centroid_bounds.corner_min[best_axis]) * inverse_extent;
        int bin_id = static_cast<int>(t * NUM_BINS);
        if (bin_id == NUM_BINS) bin_id = NUM_BINS - 1; // Round up to the last Bin_r
        bins[bin_id].element_count++;
        bins[bin_id].bounding_box.expand_to_include_AABB(mesh_triangle_aabbs[triangle_idx]);
    }

    // Pre-compute left/right bounds for all possible splits (so we don't have to recompute them from scratch to analyse every possible split)
    unsigned int left_count[NUM_BINS], right_count[NUM_BINS];
    AABB_r left_bounds[NUM_BINS], right_bounds[NUM_BINS];

    // Left-to-right
    AABB_r possible_left_box;
    unsigned int possible_left_count = 0;
    for (int i = 0; i < NUM_BINS; ++i) {
        if (bins[i].element_count > 0) {
            possible_left_box.expand_to_include_AABB(bins[i].bounding_box);
        }
        possible_left_count += bins[i].element_count;
        left_bounds[i] = possible_left_box;
        left_count[i] = possible_left_count;
    }
    // Right-to-left
    AABB_r possible_right_box;
    unsigned int possible_right_count = 0;
    for (int i = NUM_BINS - 1; i >= 0; --i) {
        if (bins[i].element_count > 0) {
            possible_right_box.expand_to_include_AABB(bins[i].bounding_box);
        }
        possible_right_count += bins[i].element_count;
        right_bounds[i] = possible_right_box;
        right_count[i] = possible_right_count;
        if (i == 0) break; // Safety
    }

    // Evaluate SAH at each Bin_r boundary and pick the best one (i.e., the one which minimizes the cost function)
    double best_cost = std::numeric_limits<double>::infinity();
    int best_split_bin = -1;

    for (int i = 0; i < NUM_BINS - 1; ++i) {
        unsigned int left_size = left_count[i];
        unsigned int right_size = right_count[i+1];
        if (left_size == 0 || right_size == 0) continue; // invalid split

        double cost = find_SAH_cost_bin_r(left_size, right_size, left_bounds[i], right_bounds[i+1]);
        if (cost < best_cost) {
            best_cost = cost;
            best_split_bin = i;
        }
    }
    if (best_split_bin == -1) return false; // No useful split found

    // Convert Bin_r index to world-space split position
    double bin_width = axis_extent / NUM_BINS;
    out_split_position = node_centroid_bounds.corner_min[best_axis] + bin_width * (best_split_bin + 1); // Boundary between best_split_bin and best_split_bin + 1
    return true;
}

 void build_bvh(BVH_Node_r& Node,
    const std::vector<std::array<double,3>>& mesh_triangle_centroids,
    const std::vector<AABB_r>& mesh_triangle_aabbs,
    std::vector<int>& mesh_triangle_indices){

    static constexpr int MAX_ELEMENTS_PER_LEAF = 4;

    // Check if we should terminate and make a leaf node
    if (Node.triangle_count <= MAX_ELEMENTS_PER_LEAF) {
        // For now leaf node means that both children are nullptr, so while these should be default values, set them again just to be sure
        Node.left_child = nullptr;
        Node.right_child = nullptr;
        return;
    }
    // Otherwise, split elements into child nodes
    // Run binned SAH
    unsigned int split_axis = 0;
    double split_position = 0.0;
    bool found_split = binned_sah_split_r(Node, mesh_triangle_centroids, mesh_triangle_aabbs, mesh_triangle_indices, split_axis, split_position);
    if (!found_split) {
        Node.left_child = nullptr;
        Node.right_child = nullptr;
        return;
        // Potential to implement some back-up splitting metod here, like median split or whatever. Need to look up what would be fail-safe where binning SAH might not perform well.
    }

    // Partition of indices by centroid[axis] < split_pos
    unsigned int begin = Node.min_triangle_idx;
    unsigned int end = begin + Node.triangle_count;
    unsigned int mid = begin;

    while (mid < end) {
        unsigned int triangle_idx = mesh_triangle_indices[mid];
        double triangle_centroid_split = mesh_triangle_centroids[triangle_idx][split_axis];
        // Compare triangle centroid position on the axis versus the splitting point
        if (triangle_centroid_split < split_position) { // Triangle on the left
            ++mid; // move mid to the right
        } else {
            --end; // Move end to left
            std::swap(mesh_triangle_indices[mid], mesh_triangle_indices[end]);
        }
    }
    // How many triangles are on the left and on the right
    unsigned int left_count = mid - begin;
    unsigned int right_count = Node.triangle_count - left_count;

    // Abort split if one side is empty
    if (left_count == 0 || right_count == 0) {
        Node.left_child = nullptr;
        Node.right_child = nullptr;
        return;
    }

    // Create children
    //Node.left_child = new BVH_Node_r{}; // raw pointer syntax
    //Node.right_child = new BVH_Node_r{};
    Node.left_child = std::make_unique<BVH_Node_r>();
    Node.right_child = std::make_unique<BVH_Node_r>();

    // Assign triangle ranges
    // Left child indices: [begin, begin+left_count)
    Node.left_child->triangle_count = left_count;
    Node.left_child->min_triangle_idx = begin;
    // Right child indices: [begin+left_count, begin+left_count+right_count)
    Node.right_child->triangle_count = right_count;
    Node.right_child->min_triangle_idx = begin + left_count;

    // Recompute child bounds from the indices
    Node.left_child->bounding_box = create_node_AABB_r(mesh_triangle_aabbs, mesh_triangle_indices, Node.left_child->min_triangle_idx, Node.left_child->triangle_count);
    Node.right_child->bounding_box = create_node_AABB_r(mesh_triangle_aabbs, mesh_triangle_indices, Node.right_child->min_triangle_idx, Node.right_child->triangle_count);
    
    Node.triangle_count = 0; // Split => Internal node containing no triangles, so update the count

    // Build BVH recursively
    build_bvh(*Node.left_child, mesh_triangle_centroids, mesh_triangle_aabbs, mesh_triangle_indices);
    build_bvh(*Node.right_child, mesh_triangle_centroids, mesh_triangle_aabbs, mesh_triangle_indices);
}

// Sanity-test check function
void print_bvh_r(const BVH_Node_r* node,
    int depth = 0) {
    if (!node) return;
    std::cout << std::string(depth*2, ' ') << "Node triangles: " << node->triangle_count << "\n";
    Ray test_ray;
    test_ray.origin = EiVector3d(0.0, 0.0, 0.0);
    test_ray.direction = EiVector3d(1.0, 0.0, 0.0);
    std::cout << std::string(depth*2, ' ') << "AABB_r intersection: " << intersect_AABB_r(test_ray, node->bounding_box) << "\n";
    if (node->left_child) print_bvh_r(node->left_child.get(), depth + 1);
    if (node->right_child) print_bvh_r(node->right_child.get(), depth + 1);
}

void intersect_bvh_r(const Ray& ray,
    const BVH_Node_r& node,
    const std::vector<int>& mesh_triangle_indices,
    const int* mesh_connectivity_ptr,
    const double* mesh_node_coords_ptr) {

     HitRecord intersection_record; // Only here for tests. Doesn't make sense to have it here later - it's one per ray, so it'll be in the renderer as usual.
     std::cout << "Starting BVH intersection test" << std::endl;

    if (!intersect_AABB_r(ray, node.bounding_box)) return; // Early exit if ray does not intersect the AABB_r of the node
    if (node.left_child == nullptr && node.right_child == nullptr) {
        // No children => Leaf node => Intersect triangles
        std::cout << "Leaf node reached with " << node.triangle_count << " triangles." << std::endl;
        int node_max_triangle_idx = node.min_triangle_idx + node.triangle_count;
        // Get indices of the triangles within the node and store in vector to pass to the intersection function
        std::vector<unsigned int> node_triangle_indices;
        node_triangle_indices.resize(node.triangle_count);
        for (int i = node.min_triangle_idx; i < node_max_triangle_idx; ++i) {
            int triangle_idx = mesh_triangle_indices[i];
            node_triangle_indices[i - node.min_triangle_idx] = triangle_idx;
            std::cout << "Node triangle index: " << triangle_idx << std::endl;
        }
        IntersectionOutput intersection = intersect_bvh_triangles(ray, mesh_connectivity_ptr, mesh_node_coords_ptr, node.triangle_count, node_triangle_indices);
        Eigen::Index minRowIndex, minColIndex;
        std::cout << "Number of t_values: " << intersection.t_values.size() << std::endl;

        intersection.t_values.minCoeff(&minRowIndex, &minColIndex); // Find indices of the smallest t_value

        double closest_t = intersection.t_values(minRowIndex, minColIndex);
        if (closest_t < intersection_record.t) {
            intersection_record.t = closest_t;
            intersection_record.barycentric_coordinates = intersection.barycentric_coordinates.row(minRowIndex);
            intersection_record.point_intersection = ray_at_t(closest_t, ray);
            intersection_record.normal_surface = intersection.plane_normals.row(minRowIndex);
            // Get a pointer to the array storing face colors for the mesh if intersected
            //double* face_colors_ptr = const_cast<double*>((scene_face_colors[mesh_idx]).data());
            //intersection_record.face_color = get_face_color(minRowIndex, face_colors_ptr);
        }
        if (intersection_record.t != std::numeric_limits<double>::infinity()) { // Instead of keeping a bool hit_anything, check if t value has changed from the default
            std::cout << "Intersection found" << std::endl;
        }
    }
    else { // Not a leaf node => Test children nodes for intersections
        intersect_bvh_r(ray, *node.left_child, mesh_triangle_indices, mesh_connectivity_ptr, mesh_node_coords_ptr);
        intersect_bvh_r(ray, *node.right_child, mesh_triangle_indices, mesh_connectivity_ptr, mesh_node_coords_ptr);
    }
}

// Handles building all acceleration structures in the scene - bottom and top level
// Might not need to pass scene_face_colors. Not sure yet.
void build_acceleration_structures_r(const std::vector <nanobind::ndarray<const int, nanobind::c_contig>>& scene_connectivity,
    const std::vector <nanobind::ndarray<const double, nanobind::c_contig>>& scene_coords,
    const std::vector<nanobind::ndarray<const double, nanobind::c_contig>>& scene_face_colors){

    // Build BLASes - BVHs for respective meshes
    int nodes_per_element = 3; // For readibility purposes. Will have to be changed when we start considering more than just triangles.
    size_t num_meshes = scene_coords.size();
    // Create vectors to store centroid and AABB_r data for the scene; might not need these, but have them for now
    std::vector<std::vector<std::array<double,3>>> scene_centroids; // Stores centroids for all meshes in the scene
    std::vector<std::vector<AABB_r>> scene_aabbs; // Stores AABBs for all meshes in this scene
    scene_centroids.reserve(num_meshes);
    scene_aabbs.reserve(num_meshes);

    // Iterate over MESHES
    for (size_t mesh_idx = 0; mesh_idx < num_meshes; ++mesh_idx) {
        // Access data from the scene for this particular mesh
		nanobind::ndarray<const double, nanobind::c_contig> mesh_node_coords = scene_coords[mesh_idx];
		nanobind::ndarray<const int, nanobind::c_contig> mesh_connectivity = scene_connectivity[mesh_idx];
        nanobind::ndarray<const double, nanobind::c_contig> mesh_face_colors = scene_face_colors[mesh_idx];

        long long mesh_number_of_elements = mesh_connectivity.shape(0); // number of triangles/faces, will give us indices for some bits
        double* mesh_node_coords_ptr = const_cast<double*>(mesh_node_coords.data());
        int* mesh_connectivity_ptr = const_cast<int*>(mesh_connectivity.data());

        // DEBUG LINES
        //std::cout << "number of elements " << mesh_number_of_elements << std::endl; // Test if there isn't issue passing the data that would cause errors further down the line
        //std::cout << "first nodal coordinate " << mesh_node_coords_ptr[0] << std::endl;

        // Containers for calculated data
        std::vector<std::array<double,3>> mesh_triangle_centroids; // Store centroids for this mesh
        mesh_triangle_centroids.reserve(mesh_number_of_elements);
        std::vector<AABB_r> mesh_triangle_aabbs; // Bounding volumes for the elements in this mesh
        mesh_triangle_aabbs.reserve(mesh_number_of_elements);
        AABB_r mesh_aabb; // AABB_r for the entire mesh

        // Iterate over ELEMENTS/TRIANGLES in this mesh
        for (int triangle_idx = 0; triangle_idx <mesh_number_of_elements; triangle_idx++) {
            // Use pointers - means we treat the 2D array as a flat 1D array and do the indexing manually by calculating the offset.
            // HAS to be contiguous in memory for this to work properly!
            int node_0 = mesh_connectivity_ptr[triangle_idx * nodes_per_element + 0]; // Equivalent to indexing as connectivity[triangle_idx, 0]
            int node_1 = mesh_connectivity_ptr[triangle_idx * nodes_per_element + 1];
            int node_2 = mesh_connectivity_ptr[triangle_idx * nodes_per_element + 2];
            // Find centroid for this triangle
            std::array<double,3> triangle_centroid;
            compute_triangle_centroid_r(node_0, node_1, node_2, mesh_node_coords_ptr, triangle_centroid);
            mesh_triangle_centroids.push_back(triangle_centroid);

            // Create bounding volume for this triangle
            AABB_r triangle_aabb;
            triangle_aabb.expand_to_include_node(node_0, mesh_node_coords_ptr);
            triangle_aabb.expand_to_include_node(node_1, mesh_node_coords_ptr);
            triangle_aabb.expand_to_include_node(node_2, mesh_node_coords_ptr);
            mesh_triangle_aabbs.push_back(triangle_aabb);

            // Include triangle AABB_r in mesh AABB_r to get the bounding box for the whole thing
            mesh_aabb.expand_to_include_AABB(triangle_aabb);
        }
        // DEBUG LINES
        //std::cout << "mesh_triangle_centroids[4][0] " << mesh_triangle_centroids[4][0] << std::endl;
        //std::cout << "mesh_triangle_aabbs[4].corner_max[0] " << mesh_triangle_aabbs[4].corner_max[0] << std::endl;

        // Actually build the BVH
        /*
        // With BVH struct
        BVH mesh_bvh;
        mesh_bvh.triangle_indices.resize(mesh_number_of_elements);
        for (unsigned int i = 0; i < N; ++i){
            mesh_bvh.triangle_indices[i] = i;
        }
        mesh_bvh.root = std::make_unique<BVH_Node_r>();
        mesh_bvh.root->bounding_box = mesh_aabb;
        mesh_bvh.root->min_triangle_idx = 0;
        mesh_bvh.root->triangle_count = mesh_number_of_elements;
        // Alternatively use iota, but we wanted minimal subset of C++ functions?
        //std::iota(mesh_bvh.triangle_indices.begin(), mesh_bvh.triangle_indices.end(), 0); // Fill the vector with triangle indices
    */

        std::vector<int> mesh_triangle_indices;
        mesh_triangle_indices.resize(mesh_number_of_elements);
        std::iota(mesh_triangle_indices.begin(), mesh_triangle_indices.end(), 0);
        // DEBUG LINES
        //std::cout << "size of mesh_triangle_indices " << mesh_triangle_indices.size() << std::endl;
        //std::cout << "mesh_triangle_indices[0] " << mesh_triangle_indices[0] << std::endl;
        //std::cout << "mesh_triangle_indices[30] " << mesh_triangle_indices[30] << std::endl;

        //BVH_Node_r* root = nullptr; // syntax for raw pointers
        //root = new BVH_Node_r();

        std::unique_ptr<BVH_Node_r> root = std::make_unique<BVH_Node_r>();
        root->bounding_box = mesh_aabb;
        root->min_triangle_idx = 0;
        root->triangle_count = mesh_number_of_elements;
        //mesh_bvh.root = std::move(root);

        build_bvh(*root, mesh_triangle_centroids, mesh_triangle_aabbs, mesh_triangle_indices);
        //print_bvh_r(root.get());
        Ray test_ray;
        test_ray.origin = EiVector3d(0.0, 0.0, 0.0);
        test_ray.direction = EiVector3d(1.0, 0.0, 0.0);
        intersect_bvh_r(test_ray, *root, mesh_triangle_indices, mesh_connectivity_ptr, mesh_node_coords_ptr);

        double data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        size_t dims[] = {2, 3};
        size_t nelems = 6;
        size_t ndims = 2;
        //NDArray_f64 arr;
        NDArray(double) arr;
        ndarray_init(double, &arr, data, nelems, dims, ndims);
        ndarray_print(double, &arr);
        ndarray_deinit(double, &arr);
        //ndarray_init_f64(&arr, data, nelems, dims, ndims);
        //ndarray_print_f64(&arr);
        //ndarray_deinit_f64(&arr);


        // Without struct bvh all data should be accessible via root pointer after building
    }

    // Build TLAS - structure of BLASes. Target is BVH in itself, but use a vector to just contain them for now?

 }


/*
double find_SAH_cost_node(const BVH_Node_r& node) {
    // Calculate the Surface Area Heuristic (SAH) cost of a node.
    double cost_traversal = 1.0;
    double cost_intersection = 1.0; // Might have to vary this with the element type when we start using more than just triangles.
    double area_parent = node.bounding_box.find_surface_area();
    double area_left_child = node.left_child->bounding_box.find_surface_area();
    double area_right_child = node.right_child->bounding_box.find_surface_area();
    return cost_traversal + cost_intersection * (area_left_child/area_parent * node.left_child->triangle_count + area_right_child/area_parent * node.right_child->triangle_count);
}
*/