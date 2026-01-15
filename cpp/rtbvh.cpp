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

#include "rtbvh.h"
#include "rtrayintersection.h"
#include "rthitrecord.h"
#include "ndarray.h"

constexpr int NODE_COORDINATES = 3; // number of coordinates per each mesh node. Used for some of flat indexing

inline void compute_mesh_centroid(AABB& mesh_aabb, std::array<double,3>& mesh_centroid) {
    // Compute centroid of the mesh AABB
    for (int i = 0; i < 3; ++i){
        mesh_centroid[i] = (mesh_aabb.corner_min[i] + mesh_aabb.corner_max[i]) / 2.0;
    }
}

AABB create_node_AABB(const std::vector<AABB>& mesh_element_abbs,
    const std::vector<int>& mesh_element_indices,
    const int node_min_element_idx,
    const int node_element_count) {
    // Iterates over all elements assigned to the node to find its bounding box
    int node_max_element_idx = node_min_element_idx + node_element_count;
    AABB node_AABB;

    for (int i = node_min_element_idx; i < node_max_element_idx; ++i) {
        int element_idx = mesh_element_indices[i];
        node_AABB.expand_to_include_AABB(mesh_element_abbs[element_idx]);
    }
    return node_AABB;
}


bool intersect_AABB (const Ray& ray, const AABB& AABB) {
    // Slab method for ray-AABB_r intersection
    double t_axis[6]; // t values for each axis, so [0,1] are for x, [2,3] for y, and [4,5] for z
    EiVector3d inverse_direction = 1/(ray.direction.array()); // Divide first to use cheaper multiplication later

    // Find ray intersections with planes defining the AABB_r in X, Y, Z
    for (int i = 0; i < 3; ++i) {
        t_axis[2*i] = (AABB.corner_min[i] - ray.origin(i)) * inverse_direction(i);
        t_axis[2*i + 1] = (AABB.corner_max[i] - ray.origin(i)) * inverse_direction(i);
    }

    //Overlap test
    // Find the minimum t for each axis (x, y, z), then find maximum of these for (x,y,z)
    double t_min = std::max(std::max(std::min(t_axis[0], t_axis[1]), std::min(t_axis[2], t_axis[3])), std::min(t_axis[4], t_axis[5]));
    // Find the maximum t for each axis (x, y, z), then find minimum of these for (x,y,z)
    double t_max = std::min(std::min(std::max(t_axis[0], t_axis[1]), std::max(t_axis[2], t_axis[3])), std::max(t_axis[4], t_axis[5]));

    // t_min < t_max - Ray which just touches a corner, edge, or face of the AABB will be considered non-intersecting
    // t_min <= t_max - Rays which touch the box boundary are considered intersecting. A bit of a degenerate case, but decided to include it here, hence more relaxed inequality.
    // t_min < ray.t_max - Clip to ray segment
    return t_min <= t_max && t_max > 0.0 && t_min < ray.t_max; // False => No overlap => Ray does not intersect the AABB.
}


inline void compute_triangle_centroid(const std::array<double,9> &triangle_node_coords,
    std::array<double,3> &triangle_centroid) {
    // Find the centroid of a triangle.
    // Update the value of the passed array, so we don't have to fiddle with structs etc. to return a value.
    for (int i=0; i < 3; ++i){
            triangle_centroid[i] = (triangle_node_coords[i] + triangle_node_coords[i+3] + triangle_node_coords[i+6]) / 3.0;
    }
}

// Go over all triangles in a mesh and find their AABB and centroids, build mesh AABB, and store the data in vectors
inline void process_element_data_tri3(int mesh_number_of_triangles,
    const double* mesh_node_coords_ptr,
    std::vector<std::array<double,3>>& mesh_element_centroids,
    std::vector<AABB>& mesh_triangle_aabbs,
    AABB& mesh_aabb){
        enum ElementNodeCount nodes_per_element = TRI3;
        const int coords_per_element = nodes_per_element * NODE_COORDINATES; // number of elements times 3 coordinates each
        // Iterate over triangles comprising a mesh
        for (int triangle_idx = 0; triangle_idx < mesh_number_of_triangles; triangle_idx++) {
            // Use pointers - means we treat the 2D array as a flat 1D array and do the indexing manually by calculating the offset.
            // HAS to be contiguous in memory for this to work properly! c_contig flag in nanobind ensures that
            std::array<double,9> triangle_node_coords;
            int triangle_min_index = triangle_idx * coords_per_element;
            //std::cout << "Triangle " << triangle_idx << " nodes: ";
            for (int i = 0; i < coords_per_element; ++i){
                triangle_node_coords[i] = mesh_node_coords_ptr[triangle_min_index + i];
                //std::cout << triangle_node_coords[i] << " , ";
            }
            //std::cout<<std::endl;
            
            // Find centroid for this triangle
            std::array<double,3> triangle_centroid;
            compute_triangle_centroid(triangle_node_coords, triangle_centroid);
            
            mesh_element_centroids.push_back(triangle_centroid);
            //std::cout << "Centroid " << triangle_centroid[0] << " " << triangle_centroid[1] << " " << triangle_centroid[2] << std::endl;

            // Create bounding volume for this triangle
            AABB triangle_aabb;
            triangle_aabb.build_for_tri3(triangle_node_coords);
            mesh_triangle_aabbs.push_back(triangle_aabb);
            //std::cout << "AABB max " << triangle_aabb.corner_max[0] << " " << triangle_aabb.corner_max[1] << " " << triangle_aabb.corner_max[2] << std::endl;
            //std::cout << "AABB min " << triangle_aabb.corner_min[0] << " " << triangle_aabb.corner_min[1] << " " << triangle_aabb.corner_min[2] << std::endl;
            // Include triangle AABB in mesh AABB to get the bounding box for the whole thing
            mesh_aabb.expand_to_include_AABB(triangle_aabb);
        } // ELEMENTS/TRIANGLES
    }

// Auxiliary functions for splitting and binning
double find_SAH_cost_bin(unsigned int left_element_count, unsigned int right_element_count, const AABB& left_bounds, const AABB& right_bounds) {
    // Calculate the Surface Area Heuristic (SAH) cost of a node. Simplified equation for initial implementation.
   return (double)left_element_count * left_bounds.find_surface_area() + (double)right_element_count * right_bounds.find_surface_area(); // Static casts complained so leave C-style casts for now
}

inline void midpoint_split(AABB& node_centroid_bounds,
    double axis_extent,
    unsigned int& out_split_axis,
    double& out_split_position){
    // Fallback splitting if SAH fails: midpoint
    
    std::cout << "SAH splitting failed. Trying midpoint instead." << std::endl;
    // Find median index (median object)
    out_split_position = node_centroid_bounds.corner_min[out_split_axis] + axis_extent * 0.5;
}


bool binned_sah_split(BuildTask& Node,
    const std::vector<std::array<double,3>>& mesh_element_centroids,
    const std::vector<AABB>& mesh_element_aabbs,
    const std::vector<int>& mesh_element_indices,
    unsigned int& out_split_axis,
    double& out_split_position) {
    // Binned Surface Area Heuristic (SAH) split

    if (Node.element_count <= 2) return false; // Too small to split
    unsigned int node_max_element_idx = Node.min_element_idx + Node.element_count;

    // Compute centroid bounds for the node
    // We use existing AABB since it nicely implements everything we need, BUT it is not to be confused with the actual bounding box of the node
    // node_centroid_bounds - Only used to determine splitting
    // bounding_box - Actual bounding box of the node used for ray intersections
    AABB node_centroid_bounds{};
    for (int i = Node.min_element_idx; i < node_max_element_idx; ++i) {
        // Retrieve triangle and its centroid on the split axis
        unsigned int element_idx = mesh_element_indices[i];
        std::array<double,3> element_centroid = mesh_element_centroids[element_idx];
        node_centroid_bounds.expand_to_include_point(element_centroid);
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
    out_split_axis = best_axis;

    // All centroids coincident along the chosen axis => No useful split
    if (axis_extent == 0){
        //return false;
        midpoint_split(node_centroid_bounds, axis_extent, out_split_axis, out_split_position);
        return true;
    }

    // Create bins
    constexpr int NUM_BINS = 8;
    Bin bins[NUM_BINS];

    const double inverse_extent = 1.0/axis_extent;
    for (unsigned int i = Node.min_element_idx; i < node_max_element_idx; ++i){
        unsigned int element_idx = mesh_element_indices[i];
        // Find the Bin containing the triangle centroid
        double t = (mesh_element_centroids[element_idx][best_axis] - node_centroid_bounds.corner_min[best_axis]) * inverse_extent;
        int bin_id = static_cast<int>(t * NUM_BINS);
        if (bin_id == NUM_BINS) bin_id = NUM_BINS - 1; // Round up to the last Bin
        bins[bin_id].element_count++;
        bins[bin_id].bounding_box.expand_to_include_AABB(mesh_element_aabbs[element_idx]);
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

    // Evaluate SAH at each Bin boundary and pick the best one (i.e., the one which minimizes the cost function)
    double best_cost = std::numeric_limits<double>::infinity();
    int best_split_bin = -1;

    for (int i = 0; i < NUM_BINS - 1; ++i) {
        unsigned int left_size = left_count[i];
        unsigned int right_size = right_count[i+1];
        if (left_size == 0 || right_size == 0) continue; // invalid split

        double cost = find_SAH_cost_bin(left_size, right_size, left_bounds[i], right_bounds[i+1]);
        if (cost < best_cost) {
            best_cost = cost;
            best_split_bin = i;
        }
    }
    if (best_split_bin == -1){ // No useful split found
        //return false; 
        midpoint_split(node_centroid_bounds, axis_extent, out_split_axis, out_split_position);
        return true;
    } 

    // Convert Bin index to world-space split position
    double bin_width = axis_extent / NUM_BINS;
    out_split_position = node_centroid_bounds.corner_min[best_axis] + bin_width * (best_split_bin + 1); // Boundary between best_split_bin and best_split_bin + 1
    return true;
}

void build_bvh(BVH &mesh_bvh,
    const std::vector<std::array<double,3>>& mesh_element_centroids,
    const std::vector<AABB>& mesh_element_aabbs,
    std::vector<int>& mesh_element_indices,
    std::vector<int>& node_minimum_element_index,
    size_t mesh_element_count){

    static constexpr int MAX_ELEMENTS_PER_LEAF = 4;
    // DFS implementation so LIFO; need to think if queue with BFS wouldn't work better since we don't care THAT much about the memory
    mesh_bvh.tree_nodes.clear();
    mesh_bvh.tree_nodes.reserve(mesh_element_indices.size() * 2); // crude upper bound

    //std::cout << "BLAS builder: Splitting into nodes..." << std::endl;
  
    // Create root
    BVH_Node root;
    root.element_count = mesh_element_count;
    root.bounding_box = create_node_AABB(mesh_element_aabbs, mesh_element_indices, 0, mesh_element_count);
    //root.min_elem_idx = 0;
    mesh_bvh.tree_nodes.push_back(root);
    node_minimum_element_index.push_back(0);
    mesh_bvh.root_idx = 0;

    //std::cout << "Initializing building BVH" << std::endl;
    // Stack-based builder
    std::vector<BuildTask> stack;
    stack.push_back({root.element_count, mesh_bvh.root_idx, 0}); // push root onto the stack
   
    while(!stack.empty()){
        //std::cout << "Inside loop for building BVH" << std::endl;
        BuildTask task = stack.back(); // Get address to the last element on the stack
        stack.pop_back(); // Remove the last element from the stack
        int node_idx = task.node_idx;
        int min_element_idx = task.min_element_idx;
        int element_count = task.element_count;
        BVH_Node& Node = mesh_bvh.tree_nodes[node_idx];

         // Check if we should terminate and make a leaf node
        if (element_count <= MAX_ELEMENTS_PER_LEAF) {
            // Leaf node means that both children indices are -1, so while these should be default values, set them again just to be sure
            Node.element_count = element_count;
            Node.bounding_box = create_node_AABB(mesh_element_aabbs, mesh_element_indices, min_element_idx, element_count);
            Node.left_child_idx = -1;
            continue;
        }

        // Otherwise, split elements into child nodes
        // Run binned SAH
        unsigned int split_axis = 0;
        double split_position = 0.0;
        bool found_split = binned_sah_split(task, mesh_element_centroids, mesh_element_aabbs, mesh_element_indices, split_axis, split_position);
        if (!found_split) {
            // Fallback splitting implemented, so if SAH returns false, it ought to be too small to split => Mark as leaf node.
            Node.left_child_idx = -1;
            continue;
        }
            
        // Partition of indices by centroid[axis] < split_pos. A bit like QuickSort partitioning, where we get the pivot from our splitting function
        unsigned int begin = min_element_idx;
        unsigned int end = begin + element_count;
        unsigned int mid = begin;

        while (mid < end) {
            unsigned int element_idx = mesh_element_indices[mid];
            double element_centroid_split = mesh_element_centroids[element_idx][split_axis];
            // Compare triangle centroid position on the axis versus the splitting point
            if (element_centroid_split < split_position) { // Triangle on the left
                ++mid; // move mid to the right
            } else {
                --end; // Move end to left
                std::swap(mesh_element_indices[mid], mesh_element_indices[end]);
            }
        }
        // How many triangles are on the left and on the right
        size_t left_count = mid - begin;
        size_t right_count = element_count - left_count;
        
        // Abort split if one side is empty
        // NOTE: this could be improved by going through the midpoint split again, but choosing a different axis.
        // That being said, with both SAH and midpoint splitting, this is very unlikely
        if (left_count == 0 || right_count == 0) {
            Node.element_count = element_count;
            Node.bounding_box = create_node_AABB(mesh_element_aabbs, mesh_element_indices, min_element_idx, element_count);
            Node.left_child_idx = -1;
            continue;
        }
    
        // Create children
        int left_child_idx = mesh_bvh.tree_nodes.size();
        int right_child_idx = left_child_idx + 1;
        // Assign element ranges
        // Left child indices: [begin, begin+left_count)
        int left_min_element_idx = begin;
        // Right child indices: [begin+left_count, begin+left_count+right_count)
        int right_min_element_idx = begin + left_count;
        node_minimum_element_index.push_back(left_min_element_idx);
        node_minimum_element_index.push_back(right_min_element_idx);
         
        // Create left child directly in BVH
        mesh_bvh.tree_nodes.emplace_back(create_node_AABB(mesh_element_aabbs, mesh_element_indices, left_min_element_idx, left_count),
            left_count,
            -1);

         // Create right child directly in BVH
        mesh_bvh.tree_nodes.emplace_back(create_node_AABB(mesh_element_aabbs, mesh_element_indices, right_min_element_idx, right_count),
            right_count,
            -1);

         // Set parent data
        // This way instead of using references, as if the vector resizes when we add children, the references might become invalid and produce nonsensical results
        mesh_bvh.tree_nodes[node_idx].left_child_idx = left_child_idx;
        mesh_bvh.tree_nodes[node_idx].element_count = 0; // It is now an internal node
        
        // Push children to stack. LIFO -> Left child gets processed first
        stack.push_back({right_count, right_child_idx, right_min_element_idx});
        stack.push_back({left_count, left_child_idx, left_min_element_idx});
    }
}

void build_TLAS(std::vector<TLAS_Node>& TLAS,
    const std::vector<std::array<double,3>>& scene_blas_centroids,
    const std::vector<AABB>& scene_blas_aabbs,
    std::vector<int>& scene_blas_indices,
    size_t scene_mesh_count){

    static constexpr int MAX_ELEMENTS_PER_LEAF = 2;
    // DFS implementation so LIFO; need to think if queue with BFS wouldn't work better since we don't care THAT much about the memory

    //std::cout << "TLAS builder: Splitting into nodes..." << std::endl;

    // Create root
    TLAS_Node root;
    root.blas_count = scene_mesh_count;
    root.min_blas_idx = 0;
    root.bounding_box = create_node_AABB(scene_blas_aabbs, scene_blas_indices, 0, root.blas_count);
    TLAS.push_back(root);

    // Stack-based builder
    std::vector<BuildTask> stack;
    stack.push_back({scene_mesh_count, 0, 0});
   
    while(!stack.empty()){
        //std::cout << "Inside loop for building BVH" << std::endl;
        BuildTask task = stack.back(); // Get address to the last element on the stack
        stack.pop_back(); // Remove the last element from the stack
        int node_idx = task.node_idx;
        int min_blas_idx = task.min_element_idx;
        int element_count = task.element_count;
        TLAS_Node& Node = TLAS[node_idx];

         // Check if we should terminate and make a leaf node
        if (element_count <= MAX_ELEMENTS_PER_LEAF) {
            // Leaf node means that both children indices are -1, so while these should be default values, set them again just to be sure
            Node.min_blas_idx = min_blas_idx;
            Node.blas_count = element_count;
            Node.bounding_box = create_node_AABB(scene_blas_aabbs, scene_blas_indices, min_blas_idx, element_count);
            Node.left_child_idx = -1;
            continue;
        }

        // Otherwise, split elements into child nodes
        // Run binned SAH
        unsigned int split_axis = 0;
        double split_position = 0.0;
        bool found_split = binned_sah_split(task, scene_blas_centroids, scene_blas_aabbs, scene_blas_indices, split_axis, split_position);
        if (!found_split) {
            // Fallback splitting implemented, so if SAH returns false, it ought to be too small to split => Mark as leaf node.
            Node.left_child_idx = -1;
            continue;
        }
            
        // Partition of indices by centroid[axis] < split_pos
        unsigned int begin = min_blas_idx;
        unsigned int end = begin + element_count;
        unsigned int mid = begin;

        while (mid < end) {
            unsigned int element_idx = scene_blas_indices[mid];
            double element_centroid_split = scene_blas_centroids[element_idx][split_axis];
            // Compare triangle centroid position on the axis versus the splitting point
            if (element_centroid_split < split_position) { // Triangle on the left
                ++mid; // Move mid to the right
            } else {
                --end; // Move end to left
                std::swap(scene_blas_indices[mid], scene_blas_indices[end]);
            }
        }
        // How many triangles are on the left and on the right
        size_t left_count = mid - begin;
        size_t right_count = element_count - left_count;
        
        // Abort split if one side is empty
        // NOTE: this could be improved by going through the midpoint split again, but choosing a different axis.
        // That being said, with both SAH and midpoint splitting, this is very unlikely
        if (left_count == 0 || right_count == 0) {
            Node.min_blas_idx = min_blas_idx;
            Node.blas_count = element_count;
            Node.bounding_box = create_node_AABB(scene_blas_aabbs, scene_blas_indices, min_blas_idx, element_count);
            Node.left_child_idx = -1;
            continue;
        }
    
        // Create children
        int left_child_idx = TLAS.size();
        int right_child_idx = left_child_idx + 1;
         // Assign element ranges
        // Left child indices: [begin, begin+left_count)
        int left_min_element_idx = begin;
        // Right child indices: [begin+left_count, begin+left_count+right_count)
        int right_min_element_idx = begin + left_count;

        // Create left child directly in TLAS
        TLAS.emplace_back(create_node_AABB(scene_blas_aabbs, scene_blas_indices, left_min_element_idx, left_count),
            left_count,
            -1,
            left_min_element_idx);

        // Create right child directly in TLAS
            TLAS.emplace_back(create_node_AABB(scene_blas_aabbs, scene_blas_indices, right_min_element_idx, right_count),
            right_count,
            -1,
            right_min_element_idx);

         // Set parent data
        // This way instead of using references, as if the vector resizes when we add children, the references might become invalid and produce nonsensical results
        TLAS[node_idx].left_child_idx = left_child_idx;
        TLAS[node_idx].blas_count = 0; // It is now an internal node
        
        // Push children to stack. LIFO -> Left child gets processed first
        stack.push_back({right_count, right_child_idx, right_min_element_idx});
        stack.push_back({left_count, left_child_idx, left_min_element_idx});
    }
}

void copy_data_to_bvh_node(BVH &mesh_bvh,
    std::vector<int>& mesh_element_indices,
    std::vector<int>& node_minimum_element_index,
    const double* mesh_node_coords_expanded_ptr,
    const double* mesh_face_color_ptr){
    // Copies appropriate mesh data to store directly in BVH node, so it can be accessed easily upon intersection.
    // This way we also avoid copying the mesh data when we move the node to the BVH tree vector... they're already there when we get to this part here.

    std::cout << "BLAS builder: Copying mesh data into leaf nodes..." << std::endl;
    size_t bvh_node_count = mesh_bvh.tree_nodes.size();
    for (int i = 0; i < bvh_node_count; ++i){
        BVH_Node& Node = mesh_bvh.tree_nodes[i];
        // Iterate over elements in the node
        int node_min_element_idx = node_minimum_element_index[i];
        int node_element_count = Node.element_count;
        //std::cout << "BVH node id: " << i << " with element count: " << node_element_count << std::endl;
        //std::cout << "Min element id from vector: " << node_min_element_idx << std::endl;
        //std::cout << "Min element id from node: " << Node.min_elem_idx << std::endl;;
        int node_max_element_idx = node_min_element_idx + Node.element_count;
        Node.node_coords.resize(node_element_count * Node.nodes_per_element * NODE_COORDINATES);
        Node.face_color.resize(node_element_count * NODE_COORDINATES);
        const int coords_per_element = Node.nodes_per_element * NODE_COORDINATES; // number of nodes per element times 3 coordinates each
        for (int element_idx = node_min_element_idx; element_idx < node_max_element_idx; ++element_idx){
            //std::cout << "Element id " << element_idx << " with coords: " << std::endl;
            int element_min_index = element_idx * coords_per_element;
            //std::cout << "Min element idx: " << element_min_index << std::endl;
            for (int j = 0; j < coords_per_element; ++j){
                //std:: cout << mesh_node_coords_expanded_ptr[element_min_index + j];
                Node.node_coords.push_back(mesh_node_coords_expanded_ptr[element_min_index + j]);
            }
            //std::cout << std::endl;
            Node.face_color.push_back(mesh_face_color_ptr[element_idx * NODE_COORDINATES]);
            Node.face_color.push_back(mesh_face_color_ptr[element_idx * NODE_COORDINATES + 1]);
            Node.face_color.push_back(mesh_face_color_ptr[element_idx * NODE_COORDINATES + 2]);
        }
    }
}

void copy_data_to_TLAS(TLAS &tlas,
    std::vector<BVH>& scene_BLASes,
    const std::vector<int>& scene_blas_indices){

    const size_t tlas_node_count = tlas.tlas_nodes.size();
    std::vector<BVH>& blases_ordered = tlas.blases; // Copy BLASes so they're stored in the traversal order determined by the builder, so data for each node is more local

    for(size_t i = 0; i < tlas_node_count; ++i){
        TLAS_Node& Node = tlas.tlas_nodes[i];
        // Iterate over all BLASes stored in TLAS nodes to copy them
        int node_max_index = Node.min_blas_idx + Node.blas_count;
        for(int j = Node.min_blas_idx; j < node_max_index; ++j){
            int blas_idx = scene_blas_indices[j];
            blases_ordered.push_back(scene_BLASes[blas_idx]);
        }
    }
 }

void intersect_bvh(const Ray& ray,
    HitRecord &intersection_record,
    const BVH& mesh_bvh) {

     std::cout << "  BLAS: Starting BVH intersection test" << std::endl;
     //const BVH_Node& root = mesh_bvh.tree_nodes[mesh_bvh.root_idx];

     std::vector<int> stack; // Store node indices on the stack
     stack.push_back(mesh_bvh.root_idx);

     while(!stack.empty()){
        const BVH_Node& Node = mesh_bvh.tree_nodes[stack.back()];
        stack.pop_back();

        if (!intersect_AABB(ray, Node.bounding_box)) continue; // Early exit if ray does not intersect the AABB_it of the node
        if (Node.left_child_idx == -1) {
            // No children => Leaf node => Intersect triangles
            std::cout << "   BLAS: Leaf node reached with " << Node.element_count << " elements." << std::endl;
           
            IntersectionOutput intersection = intersect_bvh_triangles(ray, Node.node_coords, Node.element_count);
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
                //intersection_itecord.face_color = get_face_color(minRowIndex, face_colors_ptr);
            }
            if (intersection_record.t != std::numeric_limits<double>::infinity()) { // Instead of keeping a bool hit_anything, check if t value has changed from the default
                std::cout << "Intersection found" << std::endl;
            }
        }
        else { // Not a leaf node => Test children nodes for intersections
            // DFS order
            int left = Node.left_child_idx;
            int right = left + 1;
            if (right != 0) stack.push_back(right);
            if(left != -1) stack.push_back(left);
            // Potential improvement: testing node distance vs. ray to push the farther one first, so we trasverse closer child first.
        }
            
     }
}

void intersect_tlas(const Ray& ray,
    const TLAS& scene_TLAS){

    HitRecord intersection_record;
    std::cout << "TLAS: Starting BVH intersection test" << std::endl;

     std::vector<int> stack; // Store node indices on the stack
     stack.push_back(0); // Push root index

     while(!stack.empty()){
        const TLAS_Node& Node = scene_TLAS.tlas_nodes[stack.back()];
        stack.pop_back();

        if (!intersect_AABB(ray, Node.bounding_box)) continue; // Early exit if ray does not intersect the AABB_it of the node
        if (Node.left_child_idx == -1) {
            // No children => Leaf node => Intersect triangles
            std::cout << "TLAS: Leaf node reached with " << Node.blas_count << " BLASes." << std::endl;
            int node_max_index = Node.min_blas_idx + Node.blas_count;
            for (int i = Node.min_blas_idx; i < node_max_index; ++i){
                std::cout << " TLAS: Intersected BLAS index: " << i << std::endl;
                intersect_bvh(ray, intersection_record, scene_TLAS.blases[i]);
            }
        }
        else { // Not a leaf node => Test children nodes for intersections
            // DFS order
            int left = Node.left_child_idx;
            int right = left + 1;
            if (right != 0) stack.push_back(right);
            if(left != -1) stack.push_back(left);
            // Potential improvement: testing node distance vs. ray to push the farther one first, so we trasverse closer child first.
        }
     }
}

  inline void print_BLAS_data(BVH& mesh_bvh){
     std::cout << "     BVH has " << mesh_bvh.tree_nodes.size() << " nodes." << std::endl;
     for (int i = 0; i < mesh_bvh.tree_nodes.size(); ++i){
            std::cout << "          BVH Node ID: " << i << std::endl;
            BVH_Node& Node = mesh_bvh.tree_nodes[i];
            std::cout << "              Node coords vector size [elements]: " << Node.node_coords.size() << std::endl;
            std::cout << "              Node face colors vector size [elements]: " << Node.face_color.size() << std::endl;
            std::cout << "              Node struct size total [bytes]: " << sizeof(Node) << std::endl;
 }
}

 inline void print_TLAS(TLAS &scene_TLAS){
    for (int i = 0; i < scene_TLAS.tlas_nodes.size(); ++i){
        std::cout << "TLAS Node ID: " << i << std::endl;
        TLAS_Node& Node = scene_TLAS.tlas_nodes[i];
        std::cout << "  Node BLAS count: " << Node.blas_count << std::endl;
        std::cout << "  Node min index: " << Node.min_blas_idx << std::endl;
        std::cout << "  Printing contained BLASes..." << std::endl;
        for(int j = Node.min_blas_idx; j < Node.blas_count; ++j){
            BVH& mesh_bvh = scene_TLAS.blases[j];
            print_BLAS_data(mesh_bvh);
        }
    }
 }

// Handles building all acceleration structures in the scene - bottom and top level
// Might not need to pass scene_face_colors. Not sure yet.
TLAS build_acceleration_structures(const std::vector <nanobind::ndarray<const double, nanobind::c_contig>>& scene_coords_expanded,
    const std::vector<nanobind::ndarray<const double, nanobind::c_contig>>& scene_face_colors){

    size_t scene_mesh_count = scene_coords_expanded.size(); // niu
   
    std::vector<std::array<double,3>> scene_blas_centroids; // Stores centroids of the whole objectes (meshes) in this scene
    scene_blas_centroids.reserve(scene_mesh_count);
    std::vector<AABB> scene_blas_aabbs; // Store AABBs of the whole objects in this scene
    scene_blas_aabbs.reserve(scene_mesh_count);
    std::vector<BVH> scene_blases; // Store mesh_bvhs - this will be used for TLAS
    scene_blases.reserve(scene_mesh_count);

    // Iterate over MESHES to build BLASes - BVHs for respective meshes
    for (size_t mesh_idx = 0; mesh_idx < scene_mesh_count; ++mesh_idx) {
        
        // Access data from Python buffer for this particular mesh (i.e., scene->object)
        enum ElementNodeCount nodes_per_element = TRI3; // Hard-code for now since we only have triangless.
		nanobind::ndarray<const double, nanobind::c_contig> mesh_node_coords = scene_coords_expanded[mesh_idx];
        nanobind::ndarray<const double, nanobind::c_contig> mesh_face_colors = scene_face_colors[mesh_idx];
        size_t mesh_element_count = mesh_face_colors.shape(0); // number of elements comprising the mesh
        double* mesh_node_coords_ptr = const_cast<double*>(mesh_node_coords.data());
        double* mesh_face_colors_ptr = const_cast<double*>(mesh_face_colors.data());

        // Containers for calculated data for this mesh
        std::vector<std::array<double,3>> mesh_element_centroids; // Store centroids for this mesh
        mesh_element_centroids.reserve(mesh_element_count);
        std::vector<AABB> mesh_element_aabbs; // Bounding volumes for the elements in this mesh
        mesh_element_aabbs.reserve(mesh_element_count);
        scene_blas_aabbs.emplace_back();
        AABB& mesh_aabb = scene_blas_aabbs[mesh_idx]; // AABB for the entire mesh

        // Iterate over ELEMENTS in this mesh (only triangles for now)
        process_element_data_tri3(mesh_element_count, mesh_node_coords_ptr, mesh_element_centroids, mesh_element_aabbs, mesh_aabb);
     
        // Find centroid of the entire mesh
        scene_blas_centroids.emplace_back();
        std::array<double,3>& mesh_centroid = scene_blas_centroids[mesh_idx];
        compute_mesh_centroid(mesh_aabb, mesh_centroid);

        // Temporary vectors to reshuffle element indices as we build the BVH, instead of having to access the mesh data all the time to append it in nodes right away as we do so
        std::vector<int> mesh_element_indices;
        mesh_element_indices.resize(mesh_element_count);
        std::iota(mesh_element_indices.begin(), mesh_element_indices.end(), 0);
        std::vector<int> node_minimum_element_index;

        //std::cout << "Generating BLAS for mesh " << mesh_idx << std::endl;
        scene_blases.emplace_back(); // Generate directly inside the vector to avoid copying data
        BVH& mesh_bvh = scene_blases[mesh_idx]; // Get a reference to the BVH of the current mesh to pass it to the builder functions

        // BVH builder functions
        build_bvh(mesh_bvh, mesh_element_centroids, mesh_element_aabbs, mesh_element_indices, node_minimum_element_index, mesh_element_count);
        copy_data_to_bvh_node(mesh_bvh, mesh_element_indices, node_minimum_element_index, mesh_node_coords_ptr, mesh_face_colors_ptr);
        //std::cout << "BLAS successfully built." << std::endl;
        //std::cout << "BVH has " << mesh_bvh.tree_nodes.size() << " nodes." << std::endl;

        // DEBUG LINES
        /*
        Ray test_ray;
        test_ray.origin = EiVector3d(0.0, 0.0, 0.0);
        test_ray.direction = EiVector3d(1.0, 0.0, 0.0);
        HitRecord intersection_record; 
        intersect_bvh(test_ray, intersection_record, mesh_bvh);
        */
    
        // Without struct bvh all data should be accessible via root pointer after building
    } //MESHES

    // BUILD TLAS - structure of BLASes
    TLAS scene_TLAS;
    scene_TLAS.tlas_nodes.reserve(scene_mesh_count);
    scene_TLAS.blases.reserve(scene_mesh_count*2);
    // Temporary vector to reshuffle element indices as we build the BVH, instead of having to access the mesh data all the time to append it in nodes right away as we do so
    std::vector<int> scene_blas_indices;
    scene_blas_indices.resize(scene_mesh_count);
    std::iota(scene_blas_indices.begin(), scene_blas_indices.end(), 0);

    build_TLAS(scene_TLAS.tlas_nodes, scene_blas_centroids, scene_blas_aabbs, scene_blas_indices, scene_mesh_count);
    copy_data_to_TLAS(scene_TLAS, scene_blases, scene_blas_indices);
    //std::cout << "TLAS successfully built." << std::endl;
    Ray test_ray;
    test_ray.origin = EiVector3d(0.0, 0.0, 0.0);
    test_ray.direction = EiVector3d(1.0, 0.0, 0.0);
    //intersect_tlas(test_ray, scene_TLAS);
    print_TLAS(scene_TLAS);

    return scene_TLAS;
 } // SCENE (end of function)





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