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

// Go over all triangles in a mesh and find their AABB and centroids, build mesh AABB, and store the data in vectors
inline void process_element_data_tri3(int mesh_number_of_triangles,
    const int* mesh_connectivity_ptr,
    const double* mesh_node_coords_ptr,
    std::vector<std::array<double,3>>& mesh_element_centroids,
    std::vector<AABB>& mesh_triangle_aabbs,
    AABB& mesh_aabb,
    std::vector<int>& mesh_flat_connectivity){
        enum ElementNodeQuantity nodes_per_element = TRI3;
        mesh_flat_connectivity.resize(mesh_number_of_triangles * nodes_per_element);
        // Iterate over triangles comprising a mesh
        for (int triangle_idx = 0; triangle_idx < mesh_number_of_triangles; triangle_idx++) {
            // Use pointers - means we treat the 2D array as a flat 1D array and do the indexing manually by calculating the offset.
            // HAS to be contiguous in memory for this to work properly! c_contig flag in nanobind ensures that
            int node_0 = mesh_connectivity_ptr[triangle_idx * nodes_per_element + 0]; // Equivalent to indexing as connectivity[triangle_idx, 0]
            int node_1 = mesh_connectivity_ptr[triangle_idx * nodes_per_element + 1];
            int node_2 = mesh_connectivity_ptr[triangle_idx * nodes_per_element + 2];
            // Copy the connectivity data into a vector since we are already processing it anyway
            mesh_flat_connectivity[triangle_idx * nodes_per_element + 0] = node_0;
            mesh_flat_connectivity[triangle_idx * nodes_per_element + 1] = node_1;
            mesh_flat_connectivity[triangle_idx * nodes_per_element + 2] = node_2;
            // Find centroid for this triangle
            std::array<double,3> triangle_centroid;
            compute_triangle_centroid(node_0, node_1, node_2, mesh_node_coords_ptr, triangle_centroid);
            mesh_element_centroids.push_back(triangle_centroid);

            // Create bounding volume for this triangle
            AABB triangle_aabb;
            triangle_aabb.expand_to_include_node(node_0, mesh_node_coords_ptr);
            triangle_aabb.expand_to_include_node(node_1, mesh_node_coords_ptr);
            triangle_aabb.expand_to_include_node(node_2, mesh_node_coords_ptr);
            mesh_triangle_aabbs.push_back(triangle_aabb);

            // Include triangle AABB in mesh AABB to get the bounding box for the whole thing
            mesh_aabb.expand_to_include_AABB(triangle_aabb);
        } // ELEMENTS/TRIANGLES
    }

// Auxiliary functions for splitting and binning
double find_SAH_cost_bin(unsigned int left_element_count, unsigned int right_element_count, const AABB& left_bounds, const AABB& right_bounds) {
    // Calculate the Surface Area Heuristic (SAH) cost of a node. Simplified equation for initial implementation.
   return (double)left_element_count * left_bounds.find_surface_area() + (double)right_element_count * right_bounds.find_surface_area(); // Static casts complained so leave C-style casts for now
}

// Binned Surface Area Heuristic (SAH) split
bool binned_sah_split(BVH_Node& Node,
    const std::vector<std::array<double,3>>& mesh_triangle_centroids,
    const std::vector<AABB>& mesh_triangle_aabbs,
    const std::vector<int>& mesh_triangle_indices,
    unsigned int& out_split_axis,
    double& out_split_position) {

    if (Node.triangle_count <= 2) return false; // Too small to split
    unsigned int node_max_triangle_idx = Node.min_triangle_idx + Node.triangle_count;

    // Compute centroid bounds for the node
    // We use existing AABB since it nicely implements everything we need, BUT it is not to be confused with the actual bounding box of the node
    // node_centroid_bounds - Only used to determine splitting
    // bounding_box - Actual bounding box of the node used for ray intersections
    AABB node_centroid_bounds{};
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
    Bin bins[NUM_BINS];

    const double inverse_extent = 1.0/axis_extent;
    for (unsigned int i = Node.min_triangle_idx; i < node_max_triangle_idx; ++i){
        unsigned int triangle_idx = mesh_triangle_indices[i];
        // Find the Bin containing the triangle centroid
        double t = (mesh_triangle_centroids[triangle_idx][best_axis] - node_centroid_bounds.corner_min[best_axis]) * inverse_extent;
        int bin_id = static_cast<int>(t * NUM_BINS);
        if (bin_id == NUM_BINS) bin_id = NUM_BINS - 1; // Round up to the last Bin
        bins[bin_id].element_count++;
        bins[bin_id].bounding_box.expand_to_include_AABB(mesh_triangle_aabbs[triangle_idx]);
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

 void build_bvh(BVH_Node& Root,
    const std::vector<std::array<double,3>>& mesh_triangle_centroids,
    const std::vector<AABB>& mesh_triangle_aabbs,
    std::vector<int>& mesh_triangle_indices){

    static constexpr int MAX_ELEMENTS_PER_LEAF = 4;
    // DFS implementation so LIFO; need to think if queue with BFS wouldn't work better since we don't care THAT much about the memory
    std::vector<BVH_Node*> stack;
    stack.reserve(128); // reduce reallocs; might make a better guess for that btw
    stack.push_back(&Root); // push root onto the stack

    while(!stack.empty()){
        BVH_Node* Node = stack.back(); // Get address to the last element on the stack
        stack.pop_back(); // Remove the last element from the stack

        // Check if we should terminate and make a leaf node
        if (Node->triangle_count <= MAX_ELEMENTS_PER_LEAF) {
        // For now leaf node means that both children are nullptr, so while these should be default values, set them again just to be sure
        Node->left_child = nullptr;
        Node->right_child = nullptr;
        continue;
        }

        // Otherwise, split elements into child nodes
        // Run binned SAH
        unsigned int split_axis = 0;
        double split_position = 0.0;
        bool found_split = binned_sah_split(*Node, mesh_triangle_centroids, mesh_triangle_aabbs, mesh_triangle_indices, split_axis, split_position);
        if (!found_split) {
            Node->left_child = nullptr;
            Node->right_child = nullptr;
            continue;
            // Potential to implement some back-up splitting metod here, like median split or whatever. Need to look up what would be fail-safe where binning SAH might not perform well.
        }

        // Partition of indices by centroid[axis] < split_pos
        unsigned int begin = Node->min_triangle_idx;
        unsigned int end = begin + Node->triangle_count;
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
        unsigned int right_count = Node->triangle_count - left_count;

        // Abort split if one side is empty
        if (left_count == 0 || right_count == 0) {
            Node->left_child = nullptr;
            Node->right_child = nullptr;
            continue;
        }

        // Create children
        Node->left_child = std::make_unique<BVH_Node>();
        Node->right_child = std::make_unique<BVH_Node>();

        // Assign triangle ranges
        // Left child indices: [begin, begin+left_count)
        Node->left_child->triangle_count = left_count;
        Node->left_child->min_triangle_idx = begin;
        // Right child indices: [begin+left_count, begin+left_count+right_count)
        Node->right_child->triangle_count = right_count;
        Node->right_child->min_triangle_idx = begin + left_count;

        // Recompute child bounds from the indices
        Node->left_child->bounding_box = create_node_AABB(mesh_triangle_aabbs, mesh_triangle_indices, Node->left_child->min_triangle_idx, Node->left_child->triangle_count);
        Node->right_child->bounding_box = create_node_AABB(mesh_triangle_aabbs, mesh_triangle_indices, Node->right_child->min_triangle_idx, Node->right_child->triangle_count);
        
        Node->triangle_count = 0; // Split => Internal node containing no triangles, so update the count

        // Push children to stack instead of recursing. LIFO -> Left child gets processed first
        stack.push_back(Node->right_child.get());
        stack.push_back(Node->left_child.get());
    }
}

// Rationale: Basically, instead of swapping triangle/element indices to reflect the BVH ordering, we swap the connectivity array to have one less memory access. So, for example:
// triangle_indices = [1,2,3] has corresponding connectivity = [0,1,2, 2,1,3, 0,3,4] and we swap first and last element
// triangle_indices = [3,2,1]. But instead, we directly swap in connectivity = [0,3,4, 2,1,3, 0,1,2]
// Also saves having to think about how many elements to swap, while keeping the vector flat
inline void shuffle_flat_connectivity(std::vector<int>& connectivity,
    int old_element_index,
    int new_element_index,
    int number_of_nodes){
    
    int temp;
    for(int i = 0; i < number_of_nodes; i++){
        temp = connectivity[old_element_index + i];
        connectivity[old_element_index + i] = connectivity[new_element_index + i];
        connectivity[new_element_index + i] = temp;
    }
}

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
    */
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

        // DEBUG LINES
        Ray test_ray;
        test_ray.origin = EiVector3d(0.0, 0.0, 0.0);
        test_ray.direction = EiVector3d(1.0, 0.0, 0.0);
        HitRecord intersection_record; //
        //intersect_bvh(test_ray, intersection_record, *root, mesh_triangle_indices, mesh_connectivity_ptr, mesh_node_coords_ptr);
    
        // Without struct bvh all data should be accessible via root pointer after building
    } //MESHES

    
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