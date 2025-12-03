// STD header files
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include "./Eigen/Dense"

// nanobind header files
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

// raytracer header files
#include "rteigentypes.h"
#include "rtrender.h"
#include "rtray.h"

// TO-DO'S:
// Update main to pass data and construct bvh; then rtrender and rtrayintersection to reflect these changes
// AABBs - Tested. Updated adding the point, since triangle node data isn't passed as EiVector3d.
// compute_triangle_centroid - Tested. Updated to reflect how triangle data is passed.
// Bin - Don't touch

// intersect_AABB - 1. Run test cases separately to make sure results are okay. 2. Bullet-proof it based on https://tavianator.com/2022/ray_box_boundary.html
// BVHNode - Decide on the final data layout
// intersect_bvh - Update with iterating over triangles (depends on node layout, though)
// find_SAH_cost functions - For now keep the simplified version.
// binned_sah_split, split_node  - Same as intersect_bvh. Update arguments. Test for corner cases and maybe add fail-safe option if SAH splitting fails.
// build_bvh - Write it (depends on everything above, though)


 inline void compute_triangle_centroid(int node_0,
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

// Bounding volume structure - axis-aligned bounding boxes (AABB)
struct AABB {
    double corner_min[3]{};
    double corner_max[3]{};

    AABB() {
        corner_min[0] = corner_min[1] = corner_min[2] = std::numeric_limits<double>::infinity();
        corner_max[0] = corner_max[1] = corner_max[2] = -std::numeric_limits<double>::infinity();
    }
    void expand_to_include_node(const int& node_id,
        const double* mesh_node_coords_ptr){
        for (int i = 0; i < 3; ++i){
             double nodal_coordinate = mesh_node_coords_ptr[node_id * 3 + i];
             if (nodal_coordinate < corner_min[i]) corner_min[i] = nodal_coordinate;
             if (nodal_coordinate > corner_max[i]) corner_max[i] = nodal_coordinate;
        }
    }

    void expand_to_include_AABB(const AABB& other) {
        for (int i = 0; i < 3; ++i){
            if (other.corner_min[i] < corner_min[i]) corner_min[i] = other.corner_min[i];
            if (other.corner_max[i] > corner_max[i]) corner_max[i] = other.corner_max[i];
        }
    }
    inline double find_axis_extent(int axis) const {
        return corner_max[axis] - corner_min[axis];
    }
    double find_surface_area() const {
        double height = find_axis_extent(2);
        double width = find_axis_extent(1);
        double depth = find_axis_extent(0);
        return 2 * (height * width + width * depth + height * depth);
    }
};

// BVH node structure - naive implementation with pointers for now. Replace with indices once functional to save a few bytes
struct BVH_Node {
    AABB bounding_box;
    int min_triangle_idx;
    int max_triangle_idx;
    int triangle_count;
    BVH_Node* left_child {nullptr}; // Nullptr if leaf.
    BVH_Node* right_child {nullptr};
};

// Auxiliary functions for splitting and binning
double find_SAH_cost_bin(unsigned int left_element_count, unsigned int right_element_count, const AABB& left_bounds, const AABB& right_bounds) {
    // Calculate the Surface Area Heuristic (SAH) cost of a node. Simplified equation for initial implementation.
   return (double)left_element_count * left_bounds.find_surface_area() + (double)right_element_count * right_bounds.find_surface_area(); // Static casts complained so leave C-style casts for now
}

struct Bin {
    // Bin for binning SAH
    AABB bounding_box;
    int element_count {0};
};

// Binned Surface Area Heuristic (SAH) split
bool binned_sah_split(BVH_Node& Node,
    const std::vector<AABB>& triangle_AABBs,
    const std::vector<std::array<double,3>>& triangle_centroids,
    unsigned int split_axis,
    double split_position) {
    // Find AABBs for the node
    AABB node_AABB;
    for (int i = 0; i < Node.triangle_count; ++i) {
        // find min anx max of all centroids of triangles contained within the node
        // set node_AABB values to the values found
    }

    // Pick the longest axis for splitting
    int best_axis = 0;
    double axis_extent = Node.bounding_box.find_axis_extent(best_axis);
    for (int i = 1; i < 3; ++i) {
        double temp_extent = Node.bounding_box.find_axis_extent(i);
        if (temp_extent > axis_extent){
            best_axis = i;
            axis_extent = temp_extent;
        }
    }
    if (axis_extent == 0) return false; // All centroids on a plane => No useful split
    split_axis = best_axis;

    // Create bins
    constexpr int NUM_BINS = 8;
    Bin bins[NUM_BINS];
    const double inverse_extent = 1.0/axis_extent[best_axis];
    for (unsigned int i = 0; i < Node.triangle_count; ++i) {
        // get triangle
        // get its centroid
        // Find the bin it belongs to
        //double t = (Node.bounding_box.corner_min[best_axis] - triangle_centroids[triangle_idx][best_axis]) * inverse_extent;
        int bin_id = static_cast<int>(t * NUM_BINS);
        if (bin_id == NUM_BINS) bin_id = NUM_BINS - 1; // Round up to the last bin
        bins[bin_id].element_count++;
        //bins[bin_id].bounding_box.expand_to_include_AABB(triangle_AABBs[triangle_idx]);
    }

    // Pre-compute left/right bounds for all possible splits (so we don't have to recompute them from scratch to analyse every possible split)
    unsigned int left_count[NUM_BINS], right_count[NUM_BINS];
    AABB left_bounds[NUM_BINS], right_bounds[NUM_BINS];

    // Left-to-right
    AABB possible_left_box;
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
    AABB possible_right_box;
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

    // Evaluate SAH at each bin boundary and pick the best one (i.e., the one which minimizes the cost function)
    double best_cost = std::numeric_limits<double>::infinity();
    int best_split_bin = -1;

    for (int i = 0; i < BINS; ++i) {
        unsigned int left_size = left_count[i];
        unsigned int right_size = right_count[i+1];
        if (left_size == 0 || right_size == 0) continue; // invalid split

        double cost = find_SAH_cost_bin(left_size, right_size, left_bounds[i], right_bounds[i+1]);
        if (cost < best_cost) {
            best_cost = cost;
            best_split_bin = i;
        }
    }
    if (best_split_bin == -1) return false; // No useful split found

    // Convert bin index to world-space split position
    double bin_width = axis_extent / NUM_BINS;
    split_position = Node.bounding_box.corner_min[best_axis] + bin_width * (best_split_bin + 1);
    return true;
}

void split_node(BVH_Node& node,
    const std::vector<std::array<double,3>>& triangle_centroids,
    double split_axis_position) {

    unsigned int begin = node.min_triangle_idx;
    unsigned int end = node.max_triangle_idx;
    unsigned int mid = begin;

    while (mid < end) {
        // get triangle
        // get its centroid and its value at the split_axis_position
        // if centroid[split_axis_position] < split_axis_position  (on the left)
        //   move mid to the right ++mid;
        // else
        //   move mid to the left --end; std::swap(triangle_indices[mid], triangle_indices[end]) or similar if triangles stored in bvh structure by their indices
    }
    unsigned int left_count = mid - begin;
    unsigned int right_count = node.triangle_count - left_count;

    if (left_count == 0 || right_count == 0) // Invalid split => No triangles left in either child node
    {
        return;
        // Potential to implement some back-up splitting metod here, like median split or whatever. Need to look up what would be fail-safe where binning SAH might not perform well.
    }
}

 void build_bvh(BVH_Node& Node,
    const int* mesh_connectivity_ptr,
    const double* mesh_node_coords_ptr,
    const std::vector<std::array<double,3>>& triangle_centroids,
    const std::vector<AABB>& triangle_abbs){
    
    static constexpr int MAX_ELEMENTS_PER_LEAF = 4;

    // Check if we should terminate and make a leaf node
    if (Node.triangle_count <= MAX_ELEMENTS_PER_LEAF) {
        // make this a leaf node (set node values appropriately depending on the final data layout)
        // For now leaf node means that both children are nullptr, so while these should be default values, set them again just to be sure
        Node.left_child = nullptr;
        Node.right_child = nullptr;
        return;
    }
    // Otherwise, split elements into child nodes
    // Run binned SAH
    int split_axis = 0;
    double split_position = 0.0;
    bool found_split = binned_sah_split(Node, triangle_abbs, triangle_centroids, split_axis, split_position);


}

// Handles building all acceleration structures in the scene - bottom and top level
// Might not need to pass scene_face_colors. Not sure yet.
void build_acceleration_structures(const std::vector <nanobind::ndarray<const int, nanobind::c_contig>>& scene_connectivity,
    const std::vector <nanobind::ndarray<const double, nanobind::c_contig>>& scene_coords,
    const std::vector<nanobind::ndarray<const double, nanobind::c_contig>>& scene_face_colors){

    // Build BLASes - BVHs for respective meshes
    int nodes_per_element = 3; // For readibility purposes. Will have to be changed when we start considering more than just triangles.
    size_t num_meshes = scene_coords.size();
    // Create vectors to store centroid and AABB data for the scene; might not need these, but have them for now
    std::vector<std::vector<std::array<double,3>>> scene_centroids; // Stores centroids for all meshes in the scene
    std::vector<std::vector<AABB>> scene_aabbs; // Stores AABBs for all meshes in this scene
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
        
        // Containers for calculated data
        std::vector<std::array<double,3>> mesh_triangle_centroids; // Store centroids for this mesh
        mesh_triangle_centroids.reserve(mesh_number_of_elements);
        std::vector<AABB> mesh_triangle_aabbs; // Bounding volumes for the elements in this mesh
        mesh_triangle_aabbs.reserve(mesh_number_of_elements);
        AABB mesh_aabb; // AABB for the entire mesh
        
        // Iterate over ELEMENTS/TRIANGLES in this mesh
        for (int triangle_idx = 0; triangle_idx <mesh_number_of_elements; triangle_idx++) {
            // Use pointers - means we treat the 2D array as a flat 1D array and do the indexing manually by calculating the offset.
            // HAS to be contiguous in memory for this to work properly!
            int node_0 = mesh_connectivity_ptr[triangle_idx * nodes_per_element + 0]; // Equivalent to indexing as connectivity[triangle_idx, 0]
            int node_1 = mesh_connectivity_ptr[triangle_idx * nodes_per_element + 1];
            int node_2 = mesh_connectivity_ptr[triangle_idx * nodes_per_element + 2];
            // Find centroid for this triangle
            std::array<double,3> triangle_centroid;
            compute_triangle_centroid(node_0, node_1, node_2, mesh_node_coords_ptr, triangle_centroid);
            mesh_triangle_centroids.push_back(triangle_centroid);

            // Create bounding volume for this triangle
            AABB triangle_aabb; 
            triangle_aabb.expand_to_include_node(node_0, mesh_node_coords_ptr);
            triangle_aabb.expand_to_include_node(node_1, mesh_node_coords_ptr);
            triangle_aabb.expand_to_include_node(node_2, mesh_node_coords_ptr);
            mesh_triangle_aabbs.push_back(triangle_aabb);

            // Include triangle AABB in mesh AABB to get the bounding box for the whole thing
            mesh_aabb.expand_to_include_AABB(triangle_aabb);
        }

        // Actually build the BVH
        std::vector<int> mesh_triangle_indices;
        mesh_triangle_indices.resize(mesh_number_of_elements); // Triangle indices that will be swapped in splitting to avoid modifying the data passed from Python
        std::iota(mesh_triangle_indices.begin(), mesh_triangle_indices.end(), 0); // Fill the vector with triangle indices

        BVH_Node* root = nullptr;
        root = new BVH_Node();
        root->bounding_box = mesh_aabb;
        root->min_triangle_idx = 0;
        root->max_triangle_idx = mesh_number_of_elements - 1;
        root->triangle_count = mesh_number_of_elements;
        
        build_bvh(root, mesh_connectivity_ptr, mesh_node_coords_ptr, mesh_triangle_centroids, mesh_triangle_aabbs, mesh_triangle_indices);

    }
    
    // Build TLAS - structure of BLASes. Target is BVH in itself, but use a vector to just contain them for now?

 }




 /*
// For performance improvement and corner cases, later have a look at: https://tavianator.com/2022/ray_box_boundary.html
bool intersect_AABB (const Ray& ray, const AABB& aabb) {
    // Slab method for ray-AABB intersection
    double t_min[3];
    double t_max[3];
    EiVector3d inverse_direction = 1/(ray.direction.array()); // Divide first to use cheaper multiplication later

    // Find ray intersections with planes defining the AABB in X, Y, Z
    for (int i = 0; i < 3; ++i) {
        t_min[i] = (aabb.corner_min[i] - ray.origin(i)) * inverse_direction(i);
        t_max[i] = (aabb.corner_max[i] - ray.origin(i)) * inverse_direction(i);
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

    return t_close < t_far; // False => No overlap => Ray does not intersect the AABB
}
*/
/*
void intersect_bvh(const Ray& ray, const BVH_Node& node) {
    if (!intersect_AABB(ray, node.bounding_box)) return; // Early exit if ray does not intersect the AABB of the node
    if (node.left_child == nullptr && node.right_child == nullptr) { // No children => Leaf node => Intersect triangles
        // Intersect all triangles contained in the node

    }
    else { // Not a leaf node => Test children nodes for intersections
        intersect_bvh(ray, *node.left_child);
        intersect_bvh(ray, *node.right_child);
    }
}
*/

/*
double find_SAH_cost_node(const BVH_Node& node) {
    // Calculate the Surface Area Heuristic (SAH) cost of a node.
    double cost_traversal = 1.0;
    double cost_intersection = 1.0; // Might have to vary this with the element type when we start using more than just triangles.
    double area_parent = node.bounding_box.find_surface_area();
    double area_left_child = node.left_child->bounding_box.find_surface_area();
    double area_right_child = node.right_child->bounding_box.find_surface_area();
    return cost_traversal + cost_intersection * (area_left_child/area_parent * node.left_child->triangle_count + area_right_child/area_parent * node.right_child->triangle_count);
}
*/