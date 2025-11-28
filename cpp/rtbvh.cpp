// STD header files
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
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

// Steps:
// AABB structure - done
// BVH node structure - done-ish, some considerations to ponder about
// BVH struct
// Find intersection of ray with AABB - done
// SAH binning split
// Actually building the BVH

// Bounding volume structure - axis-aligned bounding boxes (AABB)
struct AABB {
    double minX, minY, minZ, maxX, maxY, maxZ; // Might change doubles to EiVector3d or just arrays of doubles. Think about it
    // Default constructor. Boundaries of the AABB in x, y, z. Cover entire space by default.
    AABB() {
        double minX { std::numeric_limits<double>::max() }, 
            minY { std::numeric_limits<double>::max() }, 
            minZ { std::numeric_limits<double>::max() }, 
            maxX { std::numeric_limits<double>::lowest() }, 
            maxY { std::numeric_limits<double>::lowest() }, 
            maxZ { std::numeric_limits<double>::lowest() };
    }

    void expand_to_include_point(const EiVector3d& point) {
        if (point(0) < minX) minX = point(0);
        if (point(1) < minY) minY = point(1);
        if (point(2) < minZ) minZ = point(2);
        if (point(0) > maxX) maxX = point(0);
        if (point(1) > maxY) maxY = point(1);
        if (point(2) > maxZ) maxZ = point(2);
    }
    void expand_to_include_AABB(const AABB& other) { // Do I need this if I only will build top-down, so split into smaller AABBs?
        if (other.minX < minX) minX = other.minX;
        if (other.minY < minY) minY = other.minY;
        if (other.minZ < minZ) minZ = other.minZ;
        if (other.maxX > maxX) maxX = other.maxX;
        if (other.maxY > maxY) maxY = other.maxY;
        if (other.maxZ > maxZ) maxZ = other.maxZ;
    }
}

bool intersect_AABB(const Ray& ray, const AABB& box) {
    // Slab method for ray-AABB intersection
    // X-plane slabs
    double tx_min = (box.minX - ray.origin(0)) / ray.direction(0);
    double tx_max = (box.maxX - ray.origin(0)) / ray.direction(0);
    if (tx_min > tx_max) std::swap(tx_min, tx_max);
    
    // Y-plane slabs
    double ty_min = (box.minY - ray.origin(1)) / ray.direction(1);
    double ty_max = (box.maxY - ray.origin(1)) / ray.direction(1);
    if (ty_min > ty_max) std::swap(ty_min, ty_max);
    // Overlap test - if no overlap, no intersection of box bounded by all planes
    if ((tx_min > ty_max) || (ty_min > tx_max))
        return false;
    if (ty_min > tx_min)
        tx_min = ty_min;
    if (ty_max < tx_max)
        tx_max = ty_max;

    // Z-plane slabs
    double tz_min = (box.minZ - ray.origin(2)) / ray.direction(2);
    double tz_max = (box.maxZ - ray.origin(2)) / ray.direction(2);
    if (tz_min > tz_max) std::swap(tz_min, tz_max);
    // Overlap test - if no overlap, no intersection of box bounded by all planes
    if ((tx_min > tz_max) || (tz_min > tx_max))
        return false;
    return true;
}

// BVH node structure
struct BVHNode {
    AABB bounds; // Bounding box enclosing all primitives in its subtree
    int leftChild {-1};  // Index of left child node (-1 if leaf). Or convert this to a pointer, then nullptr if leaf?
    int rightChild {-1}; // Index of right child node (-1 if leaf)
    // Also need to store the triangles (or their indices) somehow if leaf. Either min/max indices or std::vector if can't use contagious ranges
};

struct BVH {
    std::vector<BVHNode> nodes;
    // Also need to store the triangles (or their indices) somehow if leaf. Either min/max indices or std::vector if can't use contagious ranges
    //std::vector<int> triangleIndices; // Indices of triangles in the original mesh
};

BVH build_BVH(const int* connectivity_ptr,
    const double* node_coords_ptr,
    const long long number_of_elements) {

    // Compute AABBs for all triangles
    std::vector<AABB> primitiveAABBs(number_of_elements);
    for (long long i = 0; i < number_of_elements; ++i) {
        AABB aabb;
        int node_0 = connectivity_ptr[i * 3 + 0];
        int node_1 = connectivity_ptr[i * 3 + 1];
        int node_2 = connectivity_ptr[i * 3 + 2];
        EiVector3d v0(node_coords_ptr[node_0 * 3 + 0], node_coords_ptr[node_0 * 3 + 1], node_coords_ptr[node_0 * 3 + 2]);
        EiVector3d v1(node_coords_ptr[node_1 * 3 + 0], node_coords_ptr[node_1 * 3 + 1], node_coords_ptr[node_1 * 3 + 2]);
        EiVector3d v2(node_coords_ptr[node_2 * 3 + 0], node_coords_ptr[node_2 * 3 + 1], node_coords_ptr[node_2 * 3 + 2]);
        aabb.expand_to_include_point(v0);
        aabb.expand_to_include_point(v1);
        aabb.expand_to_include_point(v2);
        primitiveAABBs[i] = aabb;
    }

    // Recursively build the BVH
    BVH bvh;
    bvh.nodes.clear();
    BVHNode rootNode;
    build_BVH(rootNode, primitiveAABBs);
    bvh.nodes.push_back(rootNode);
    return bvh;
}

 // Function to compute the centroid of a triangle given its vertex indices and coordinates, for splitting
 EiVector3d compute_triangle_centroid(
    const int* connectivity_ptr,
    const double* node_coords_ptr,
    int triangle_index) {

    int node_0 = connectivity_ptr[triangle_index * 3 + 0];
    int node_1 = connectivity_ptr[triangle_index * 3 + 1];
    int node_2 = connectivity_ptr[triangle_index * 3 + 2];
    double centroid_x = (node_coords_ptr[node_0 * 3 + 0] + node_coords_ptr[node_1 * 3 + 0] + node_coords_ptr[node_2 * 3 + 0]) / 3.0;
    double centroid_y = (node_coords_ptr[node_0 * 3 + 1] + node_coords_ptr[node_1 * 3 + 1] + node_coords_ptr[node_2 * 3 + 1]) / 3.0;
    double centroid_z = (node_coords_ptr[node_0 * 3 + 2] + node_coords_ptr[node_1 * 3 + 2] + node_coords_ptr[node_2 * 3 + 2]) / 3.0;
    return EiVector3d(centroid_x, centroid_y, centroid_z);
 }


 /* WIP to build using binning SAH splitting


 void build_BVH(BVHNode& node, 
    std::vector<AABB>& primitives){
    static constexpr int max_primitives_per_leaf = 2; // Maximum number of primitives per leaf node 
        // Need to compuite bounds for the primitives first

    // Check if we should terminate and make a leaf node
    if (primitives.size() <= max_primitives_per_leaf) {
        // Make this a leaf node
        node.firstPrimitive = 0; // Assuming primitives are stored in a separate array
        node.leftChild = node.rightChild = -1;
        // And store the primitives?
        return;
    }

    // Otherwise, split primitives into child nodes
   

    
    */