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


/*non expanded coords
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
*/

/*
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
*/

inline void compute_mesh_centroid(AABB mesh_aabb, std::array<double,3>& mesh_centroid) {
    // Compute centroid of the mesh AABB
    for (int i = 0; i < 3; ++i){
        mesh_centroid[i] = (mesh_aabb.corner_min[i] + mesh_aabb.corner_max[i]) / 2.0;
    }
}


AABB create_node_AABB(const std::vector<AABB>& mesh_triangle_abbs,
    const std::vector<int>& mesh_triangle_indices,
    const int node_min_triangle_idx,
    const int node_triangle_count) {
    // Iterates over all triangles assigned to the node to find its bounding box
    int node_max_triangle_idx = node_min_triangle_idx + node_triangle_count;
    AABB node_AABB;

    for (int i = node_min_triangle_idx; i < node_max_triangle_idx; ++i) {
        int triangle_idx = mesh_triangle_indices[i];
        node_AABB.expand_to_include_AABB(mesh_triangle_abbs[triangle_idx]);
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
        const int coords_per_element = nodes_per_element * 3; // number of elements times 3 coordinates each
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

struct BuildTask {
    int node_idx;
    int min_element_idx;      // first triangle index in tri_indices
    int element_count;      // number of elements
};

// Binned Surface Area Heuristic (SAH) split
bool binned_sah_split(BuildTask& Node,
    const std::vector<std::array<double,3>>& mesh_element_centroids,
    const std::vector<AABB>& mesh_element_aabbs,
    const std::vector<int>& mesh_element_indices,
    unsigned int& out_split_axis,
    double& out_split_position) {

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
    if (axis_extent == 0) return false; // All centroids coincident along the chosen axis => No useful split
    // Might implement fallback split here (median et.c) here later rather than in the main build_bvh function?
    out_split_axis = best_axis;

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
    if (best_split_bin == -1) return false; // No useful split found

    // Convert Bin index to world-space split position
    double bin_width = axis_extent / NUM_BINS;
    out_split_position = node_centroid_bounds.corner_min[best_axis] + bin_width * (best_split_bin + 1); // Boundary between best_split_bin and best_split_bin + 1
    return true;
}


void build_bvh(BVH &mesh_bvh,
    const std::vector<std::array<double,3>>& mesh_element_centroids,
    const std::vector<AABB>& mesh_element_aabbs,
    std::vector<int>& mesh_element_indices,
    std::vector<int>& node_minimum_element_index){

    static constexpr int MAX_ELEMENTS_PER_LEAF = 4;
    // DFS implementation so LIFO; need to think if queue with BFS wouldn't work better since we don't care THAT much about the memory
    mesh_bvh.tree_nodes.clear();
   //mesh_bvh.tree_nodes.reserve(mesh_triangle_indices.size() * 2); // crude upper bound
  
   // Create root
   BVH_Node root;
   root.element_count = mesh_element_indices.size();
   root.bounding_box = create_node_AABB(mesh_element_aabbs, mesh_element_indices, 0, root.element_count);
   //root.min_elem_idx = 0;
   mesh_bvh.tree_nodes.push_back(root);
   node_minimum_element_index.push_back(0);
   mesh_bvh.root_idx = 0;

   //std::cout << "Initializing building BVH" << std::endl;
   // Stack-based builder
   std::vector<BuildTask> stack;
   stack.push_back({mesh_bvh.root_idx, 0, root.element_count}); // push root onto the stack
   
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
            //BVH_Node& Node = mesh_bvh.tree_nodes[node_idx];
            //Node.min_elem_idx = min_element_idx;
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
            Node.left_child_idx = -1;
            continue;
            // Potential to implement some back-up splitting metod here, like median split or whatever. Need to look up what would be fail-safe where binning SAH might not perform well.
        }
            
        // Partition of indices by centroid[axis] < split_pos
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
        int left_count = mid - begin;
        int right_count = element_count - left_count;
        
        // Abort split if one side is empty
        if (left_count == 0 || right_count == 0) {
            Node.left_child_idx = -1;
            continue;
        }
    
        // Create children
        int left_child_idx = mesh_bvh.tree_nodes.size();
        mesh_bvh.tree_nodes.push_back(BVH_Node());
        int right_child_idx = left_child_idx + 1;
        mesh_bvh.tree_nodes.push_back(BVH_Node());

         // Set parent data
        // This way instead of using references, as if the vector resizes when we add children, the references might become invalid and produce nonsensical results
        mesh_bvh.tree_nodes[node_idx].left_child_idx = left_child_idx;
        mesh_bvh.tree_nodes[node_idx].element_count = 0; // It is now an internal node

        // Initialize children metadata
        BVH_Node& left_child = mesh_bvh.tree_nodes[left_child_idx];
        BVH_Node& right_child = mesh_bvh.tree_nodes[right_child_idx];        
 
        // Assign element ranges
        // Left child indices: [begin, begin+left_count)
        left_child.element_count = left_count;
        int left_min_element_idx = begin;
        //left_child.min_elem_idx = begin;
        node_minimum_element_index.push_back(left_min_element_idx);
        // Right child indices: [begin+left_count, begin+left_count+right_count)
        int right_min_element_idx = begin + left_count;
        right_child.element_count = right_count;
        //right_child.min_elem_idx = begin + left_count;
        node_minimum_element_index.push_back(right_min_element_idx);

        // Recompute child bounds from the indices
        left_child.bounding_box = create_node_AABB(mesh_element_aabbs, mesh_element_indices, left_min_element_idx, left_count);
        right_child.bounding_box = create_node_AABB(mesh_element_aabbs, mesh_element_indices, right_min_element_idx, right_count);
        
        // Push children to stack instead of recursing. LIFO -> Left child gets processed first
        stack.push_back({right_child_idx, right_min_element_idx, right_count});
        stack.push_back({left_child_idx, left_min_element_idx, left_count});
    }
}


void copy_data_to_bvh_node(BVH &mesh_bvh,
    std::vector<int>& mesh_element_indices,
    std::vector<int>& node_minimum_element_index,
    const double* mesh_node_coords_expanded_ptr,
    const double* mesh_face_color_ptr){
    // Copies appropriate mesh data to store directly in BVH node, so it can be accessed easily upon intersection

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
        Node.node_coords.resize(node_element_count * Node.nodes_per_element * 3);
        Node.face_color.resize(node_element_count * 3);
        const int coords_per_element = Node.nodes_per_element * 3; // number of nodes per element times 3 coordinates each
        for (int element_idx = node_min_element_idx; element_idx < node_max_element_idx; ++element_idx){
            //std::cout << "Element id " << element_idx << " with coords: " << std::endl;
            int element_min_index = element_idx * coords_per_element;
            //std::cout << "Min element idx: " << element_min_index << std::endl;
            for (int j = 0; j < coords_per_element; ++j){
                //std:: cout << mesh_node_coords_expanded_ptr[element_min_index + j];
                Node.node_coords.push_back(mesh_node_coords_expanded_ptr[element_min_index + j]);
            }
            //std::cout << std::endl;
            Node.face_color.push_back(mesh_face_color_ptr[element_idx * 3]);
            Node.face_color.push_back(mesh_face_color_ptr[element_idx * 3 + 1]);
            Node.face_color.push_back(mesh_face_color_ptr[element_idx * 3 + 2]);
        }
    }
}

// Handles building all acceleration structures in the scene - bottom and top level
// Might not need to pass scene_face_colors. Not sure yet.
void build_acceleration_structures(const std::vector <nanobind::ndarray<const double, nanobind::c_contig>>& scene_coords_expanded,
    const std::vector<nanobind::ndarray<const double, nanobind::c_contig>>& scene_face_colors){

    // Build BLASes - BVHs for respective meshes
    size_t num_meshes = scene_coords_expanded.size();
    // Create vectors to store centroid and AABB_r data for the scene; might not need these, but have them for now
    std::vector<std::vector<std::array<double,3>>> scene_centroids; // Stores centroids for all meshes in the scene
    //scene_centroids.reserve(num_meshes); // Might not be worth reserving since we'll need num_meshes * num_elements_mesh * 3 * 8 bytes (double) for the whole vector
    std::vector<std::vector<AABB>> scene_aabbs; // Stores AABBs for all elements comprising meshes in this scene
     //scene_aabbs.reserve(num_meshes); // // Might not be worth reserving since we'll need num_meshes * num_elements_mesh * 48 bytes (AABB) for the whole vector
    std::vector<AABB> scene_obj_aabbs; // Stores AABBs of the whole objectes (meshes) in this scene
    //scene_obj_aabbs.reserve(num_meshes); // Can reliably reserve this size and not expect it to change
    
    std::vector<std::array<double,3>> scene_obj_centroids; // Stores centroids of the whole objectes (meshes) in this scene
    scene_obj_centroids.reserve(num_meshes); // Can reliably reserve this size and not expect it to change
    //std::vector<BVH> scene_mesh_bvhs; // Store BVHs for all meshes in the scene
    //scene_mesh_bvhs.reserve(num_meshes);
    std::vector<std::unique_ptr<BVH>> scene_mesh_bvhs;

    // Iterate over MESHES
    for (size_t mesh_idx = 0; mesh_idx < num_meshes; ++mesh_idx) {
           // Access data from the scene for this particular mesh
        enum ElementNodeCount nodes_per_element = TRI3; // Hard-code for now since we only have triangless.
		nanobind::ndarray<const double, nanobind::c_contig> mesh_node_coords = scene_coords_expanded[mesh_idx];
        nanobind::ndarray<const double, nanobind::c_contig> mesh_face_colors = scene_face_colors[mesh_idx];
        //size_t mesh_number_of_elements = mesh_connectivity.shape(0); // number of triangles/faces, will give us indices for some bits
        size_t mesh_number_of_elements = mesh_face_colors.shape(0); // number of triangles/faces, will give us indices for some bits
        double* mesh_node_coords_ptr = const_cast<double*>(mesh_node_coords.data());
        //int* mesh_connectivity_ptr = const_cast<int*>(mesh_connectivity.data());
        double* mesh_face_colors_ptr = const_cast<double*>(mesh_face_colors.data());

        // Containers for calculated data
        std::vector<std::array<double,3>> mesh_element_centroids; // Store centroids for this mesh
        mesh_element_centroids.reserve(mesh_number_of_elements * 3 * sizeof(double));
        std::vector<AABB> mesh_element_aabbs; // Bounding volumes for the elements in this mesh
        mesh_element_aabbs.reserve(mesh_number_of_elements * sizeof(AABB));
        AABB mesh_aabb; // AABB_r for the entire mesh
        std::vector<int> mesh_flat_connectivity;
        // Iterate over ELEMENTS/TRIANGLES in this mesh
        process_element_data_tri3(mesh_number_of_elements, mesh_node_coords_ptr, mesh_element_centroids, mesh_element_aabbs, mesh_aabb);
        // DEBUG: Test copying the connectivity array and flat shuffle
        /*
        for (int i = 0; i < mesh_number_of_elements; i++){
            std::cout << "element " << i << ": " << mesh_flat_connectivity[i] << " " << mesh_flat_connectivity[i+1] << " " << mesh_flat_connectivity[i+2] << std::endl;
        }
        shuffle_flat_connectivity(mesh_flat_connectivity, 0, 5, nodes_per_element);
for (int i = 0; i < mesh_number_of_elements; i++){
            std::cout << "element " << i << ": " << mesh_flat_connectivity[i] << " " << mesh_flat_connectivity[i+1] << " " << mesh_flat_connectivity[i+2] << std::endl;
        }
        //scene_centroids.push_back(mesh_element_centroids);
        */
        scene_obj_aabbs.push_back(mesh_aabb); // Store the AABB of this mesh in the scene vector
        std::array<double,3> mesh_centroid;
        compute_mesh_centroid(mesh_aabb, mesh_centroid);
        scene_obj_centroids.push_back(mesh_centroid); // Store the centroid of this mesh in the scene vector

        // Temp to reshuffle indices as we build the BVH, instead of having to access the mesh data all the time to append it in nodes
        std::vector<int> mesh_triangle_indices;
        mesh_triangle_indices.resize(mesh_number_of_elements);
        std::iota(mesh_triangle_indices.begin(), mesh_triangle_indices.end(), 0);
        std::vector<int> node_minimum_element_index;

        std::cout << "Before building BVH" << std::endl;
        BVH mesh_bvh;

        
        build_bvh(mesh_bvh, mesh_element_centroids, mesh_element_aabbs, mesh_triangle_indices, node_minimum_element_index);
        copy_data_to_bvh_node(mesh_bvh, mesh_triangle_indices, node_minimum_element_index, mesh_node_coords_ptr, mesh_face_colors_ptr);


        std::cout << "BVH has " << mesh_bvh.tree_nodes.size() << " nodes." << std::endl;

        for (int i = 0; i < mesh_bvh.tree_nodes.size(); ++i){
            std::cout << "BVH Node ID: " << i << std::endl;
            BVH_Node& Node = mesh_bvh.tree_nodes[i];
            std::cout << "Node coords vector size [elements]: " << Node.node_coords.size() << std::endl;
            std::cout << "Node face colors vector size [elements]: " << Node.face_color.size() << std::endl;
            std::cout << "Node struct size total [bytes]: " << sizeof(Node) << std::endl;
        }
        std::cout << "After building BVH" << std::endl;
        
        //scene_mesh_bvhs.push_back(std::make_unique<BVH>(std::move(mesh_bvh)));

        // Add to the vector storing scene BVHs
        //scene_mesh_bvhs.emplace_back(std::move(mesh_bvh)); // emplace_back to avoid unnecessary copies

        // DEBUG LINES
        Ray test_ray;
        test_ray.origin = EiVector3d(0.0, 0.0, 0.0);
        test_ray.direction = EiVector3d(1.0, 0.0, 0.0);
        HitRecord intersection_record; 
        //intersect_bvh(test_ray, intersection_record, mesh_bvh);
        //intersect_bvh(test_ray, intersection_record, *scene_mesh_bvhs.back());
        //intersect_bvh(test_ray, intersection_record, *root, mesh_triangle_indices, mesh_connectivity_ptr, mesh_node_coords_ptr);
    
        // Without struct bvh all data should be accessible via root pointer after building
    } //MESHES

    
 } // SCENE (end of function)
 


/* Commented out - using connectivity and nodal coordinates, not expanded
// Handles building all acceleration structures in the scene - bottom and top level
// Might not need to pass scene_face_colors. Not sure yet.
void build_acceleration_structures(const std::vector <nanobind::ndarray<const int, nanobind::c_contig>>& scene_connectivity,
    const std::vector <nanobind::ndarray<const double, nanobind::c_contig>>& scene_coords,
    const std::vector<nanobind::ndarray<const double, nanobind::c_contig>>& scene_face_colors){

    // Build BLASes - BVHs for respective meshes
    size_t num_meshes = scene_coords.size();
    // Create vectors to store centroid and AABB_r data for the scene; might not need these, but have them for now
    std::vector<std::vector<std::array<double,3>>> scene_centroids; // Stores centroids for all meshes in the scene
    //scene_centroids.reserve(num_meshes); // Might not be worth reserving since we'll need num_meshes * num_elements_mesh * 3 * 8 bytes (double) for the whole vector
    std::vector<std::vector<AABB>> scene_aabbs; // Stores AABBs for all elements comprising meshes in this scene
     //scene_aabbs.reserve(num_meshes); // // Might not be worth reserving since we'll need num_meshes * num_elements_mesh * 48 bytes (AABB) for the whole vector
    std::vector<AABB> scene_obj_aabbs; // Stores AABBs of the whole objectes (meshes) in this scene
    //scene_obj_aabbs.reserve(num_meshes); // Can reliably reserve this size and not expect it to change
    
    std::vector<std::array<double,3>> scene_obj_centroids; // Stores centroids of the whole objectes (meshes) in this scene
    scene_obj_centroids.reserve(num_meshes); // Can reliably reserve this size and not expect it to change
    //std::vector<BVH> scene_mesh_bvhs; // Store BVHs for all meshes in the scene
    //scene_mesh_bvhs.reserve(num_meshes);
    std::vector<std::unique_ptr<BVH>> scene_mesh_bvhs;

    // Iterate over MESHES
    for (size_t mesh_idx = 0; mesh_idx < num_meshes; ++mesh_idx) {
           // Access data from the scene for this particular mesh
        enum ElementNodeQuantity nodes_per_element = TRI3; // Hard-code for now since we only have triangless.
		nanobind::ndarray<const double, nanobind::c_contig> mesh_node_coords = scene_coords[mesh_idx];
		nanobind::ndarray<const int, nanobind::c_contig> mesh_connectivity = scene_connectivity[mesh_idx];
        nanobind::ndarray<const double, nanobind::c_contig> mesh_face_colors = scene_face_colors[mesh_idx];
        size_t mesh_number_of_elements = mesh_connectivity.shape(0); // number of triangles/faces, will give us indices for some bits
        double* mesh_node_coords_ptr = const_cast<double*>(mesh_node_coords.data());
        int* mesh_connectivity_ptr = const_cast<int*>(mesh_connectivity.data());

        // Containers for calculated data
        std::vector<std::array<double,3>> mesh_element_centroids; // Store centroids for this mesh
        mesh_element_centroids.reserve(mesh_number_of_elements * 3 * sizeof(double));
        std::vector<AABB> mesh_element_aabbs; // Bounding volumes for the elements in this mesh
        mesh_element_aabbs.reserve(mesh_number_of_elements * sizeof(AABB));
        AABB mesh_aabb; // AABB_r for the entire mesh
        std::vector<int> mesh_flat_connectivity;
        // Iterate over ELEMENTS/TRIANGLES in this mesh
        process_element_data_tri3(mesh_number_of_elements, mesh_connectivity_ptr, mesh_node_coords_ptr, mesh_element_centroids, mesh_element_aabbs, mesh_aabb, mesh_flat_connectivity);
        // DEBUG: Test copying the connectivity array and flat shuffle
        /*
        for (int i = 0; i < mesh_number_of_elements; i++){
            std::cout << "element " << i << ": " << mesh_flat_connectivity[i] << " " << mesh_flat_connectivity[i+1] << " " << mesh_flat_connectivity[i+2] << std::endl;
        }
        shuffle_flat_connectivity(mesh_flat_connectivity, 0, 5, nodes_per_element);
for (int i = 0; i < mesh_number_of_elements; i++){
            std::cout << "element " << i << ": " << mesh_flat_connectivity[i] << " " << mesh_flat_connectivity[i+1] << " " << mesh_flat_connectivity[i+2] << std::endl;
        }
        //scene_centroids.push_back(mesh_element_centroids);
        scene_obj_aabbs.push_back(mesh_aabb); // Store the AABB of this mesh in the scene vector
        std::vector<int> mesh_triangle_indices;
        mesh_triangle_indices.resize(mesh_number_of_elements);
        std::iota(mesh_triangle_indices.begin(), mesh_triangle_indices.end(), 0);
        std::array<double,3> mesh_centroid;
        compute_mesh_centroid(mesh_aabb, mesh_centroid);
        scene_obj_centroids.push_back(mesh_centroid); // Store the centroid of this mesh in the scene vector

        // DEBUG LINES
        //std::cout << "size of mesh_triangle_indices " << mesh_triangle_indices.size() << std::endl;
        //std::cout << "mesh_triangle_indices[0] " << mesh_triangle_indices[0] << std::endl;
        //std::cout << "mesh_triangle_indices[30] " << mesh_triangle_indices[30] << std::endl;

        std::unique_ptr<BVH_Node> root = std::make_unique<BVH_Node>();
        root->bounding_box = mesh_aabb;
        root->min_triangle_idx = 0;
        root->triangle_count = mesh_number_of_elements;
        //mesh_bvh.root = std::move(root);


        build_bvh(*root, mesh_element_centroids, mesh_element_aabbs, mesh_triangle_indices);
        // Build BVH struct
        BVH mesh_bvh;
        mesh_bvh.root = std::move(root);
        mesh_bvh.triangle_indices = std::move(mesh_triangle_indices); 
        mesh_bvh.mesh_node_coords_ptr = mesh_node_coords_ptr;
        mesh_bvh.mesh_connectivity_ptr = mesh_connectivity_ptr;
        double* mesh_face_colors_ptr = const_cast<double*>(mesh_face_colors.data());
        mesh_bvh.mesh_face_colors_ptr = mesh_face_colors_ptr;
        // Add to the vector storing scene BVHs
        //scene_mesh_bvhs.emplace_back(std::move(mesh_bvh)); // emplace_back to avoid unnecessary copies

        std::cout << "Before building BVH" << std::endl;
        BVH mesh_bvh;
        build_bvh(mesh_bvh, mesh_element_centroids, mesh_element_aabbs, mesh_triangle_indices);
        // If BVH has multiple nodes => Root must store 0 triangles => Update the triangle count
        if (mesh_bvh.tree_nodes.size() >=2){
            mesh_bvh.tree_nodes[0].triangle_count = 0;
        }
        
        for (int i = 0; i < mesh_bvh.tree_nodes.size(); i++){
            std::cout << "Node " << i << ": min_triangle_idx " << mesh_bvh.tree_nodes[i].min_triangle_idx << ", triangle_count " << mesh_bvh.tree_nodes[i].triangle_count << ", left_child_idx " << mesh_bvh.tree_nodes[i].left_child_idx << ", right_child_idx " << mesh_bvh.tree_nodes[i].right_child_idx << std::endl;
        }
        std::cout << "BVH has " << mesh_bvh.tree_nodes.size() << " nodes." << std::endl;
        std::cout << "After building BVH" << std::endl;
        
        mesh_bvh.mesh_node_coords_ptr = mesh_node_coords_ptr;
        mesh_bvh.mesh_connectivity_ptr = mesh_connectivity_ptr;
        double* mesh_face_colors_ptr = const_cast<double*>(mesh_face_colors.data());
        mesh_bvh.mesh_face_colors_ptr = mesh_face_colors_ptr;
        mesh_bvh.triangle_indices = std::move(mesh_triangle_indices); 
        //scene_mesh_bvhs.push_back(std::make_unique<BVH>(std::move(mesh_bvh)));



        // DEBUG LINES
        Ray test_ray;
        test_ray.origin = EiVector3d(0.0, 0.0, 0.0);
        test_ray.direction = EiVector3d(1.0, 0.0, 0.0);
        HitRecord intersection_record; 
        intersect_bvh(test_ray, intersection_record, mesh_bvh);
        //intersect_bvh(test_ray, intersection_record, *scene_mesh_bvhs.back());
        //intersect_bvh(test_ray, intersection_record, *root, mesh_triangle_indices, mesh_connectivity_ptr, mesh_node_coords_ptr);
    
        // Without struct bvh all data should be accessible via root pointer after building
    } //MESHES

    
 } // SCENE (end of function)
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