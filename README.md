# Overview

`rayTracer` is a work-in-progress C++ ray tracing module intended to be integrated into a larger open-source engineering simulation toolkit - [Pyvale](https://computer-aided-validation-laboratory.github.io/pyvale/).

This repository is currently a rough development space rather than a polished standalone application — although it should, in principle, be able to render images in its current state, provided the meshes are given in Pyvale-compatible `simdata` format.

> ⚠️ **Status:** Early-stage, experimental, with minimal documentation, and subject to frequent breaking changes.

---

## Project goals
At this stage, the emphasis is on getting the core MVP done to be able to perform preliminary research analysis.
- Provide a flexible, CPU-based ray tracing module with the potential expansion to GPU parallelization later on.
- Focus on engineering and scientific rendering use cases, primarily simulating [Digital Image Correlation](https://www.zeiss.co.uk/metrology/explore/topics/digital-image-correlation.html)
    - This means using higher-order elements, not just triangles, to accurately capture complex geometries.
    - Avoiding approximations that would speed up the rendering time at the expense of scientific accuracy.
    - Rasterizing cannot handle complex set-ups like refractive index changes due to curvature or material changes very well, which is why ray tracing is needed for experimental simulation.
---
# Current features
## General
* Python interface (`rtmain.py`) that works with pyvale, allowing the user to customise the camera location and orientation, anti-aliasing, and import mesh data.
  * Currently WIP, a better API for users is yet to come.
* Passing Python data to C++ rendering engine using [nanobind](https://nanobind.readthedocs.io/en/latest/index.html) for minimal overhead. It was found that nanobind consistently outperformed [pybind11](https://github.com/pybind/pybind11).

## Rendering images
* **Multiple mesh and camera support**
* Supports only **triangular meshes** for now, but the goal is to expand to quads and more complex elements.
* **Anti-aliasing** - Sampling around the pixel rather than only considering its centre to achieve smoother results. 
  
| No anti-aliasing  | Anti-aliasing with 50 samples |
| ------------- | ------------- |
| ![Example render without anti-aliasing](/images/ex_render_1.jpg) | ![Example render with anti-aliasing](/images/ex_render_2.jpg) |

* **Output format**: Currently only .ppm images, with plans to expand to return a buffer and potentially different formats, as not all software can open .ppm images.
* **Colouring:** Mesh colours are currently based on the x-displacement field values. Grayscale is used for speed gains, as colours are not necessary for DIC.

## Acceleration structures - bounding volume hierarchies (BVH)
* Implements both **bottom-level (BLAS)** and **top-level (TLAS)** acceleration structures (BVHs) to avoid testing every ray against every mesh element, which is the most computationally expensive part of ray tracing.
    * One BLAS is built per mesh.
    * A global TLAS contains all BLASes to skip entire meshes that cannot be intersected by a given ray.
    * This reduced render time by roughly 93% in representative test cases.
### BVH implementations
* `rtbvh_recursion` - Initial, naive implementation using pointers and recursion. Analogous to building a simple binary tree.
* `rtbvh_stack` - Improved version using a stack for building and traversal to avoid the function call overhead from recursion.
* `rtbvh` - Final implementation following the **data-oriented** programming principles.
    * Replaces pointer-heavy structures with an **array-of-structs-of-arrays (AoSoA)** layout to keep **data contiguous and local**, avoiding pointer-chasing and improving cache friendliness.
    * This design is more memory-intensive because mesh data is copied into BVH nodes from the Python buffer, but the performance benefits were deemed worth it.
    * Most functions were updated to operate on arrays rather than individual elements to better exploit cache mechanisms and vectorisation where applicable.
### BVH building strategy
BVH construction occurs in two stages:
1. **Node splitting**:
    * Nodes are split according to the [Surface Area Heuristic](https://www.pbr-book.org/3ed-2018/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies), which is a standard approach for BVH construction in ray tracing.
    * Bins are employed to **reduce the build time and computational complexity**. Standard SAH is usually $O(N \log N)$ for $N$ primitives (mesh elements), whereas binned SAH reduces this to about $O(N+B)$ for $B$ bins.
    * Midpoint splitting is implemented as a fallback and reuses the same data as SAH, avoiding extra computation.
    * An `std::vector` of mesh element indices is reordered according to the traversal order to avoid random access to mesh data during splitting.
    * *To do:* Implement an additional fallback when splitting along the longest axis fails, trying alternative axes.
2. **Data assignment to nodes:**
    * **BLAS**: Mesh data (nodal coordinates and colours) is copied from the Python buffers into the respective BVH nodes.
    * **TLAS**: Sn assumption was made that the scenes would not feature too many meshes since engineering simulations do not require any extras such as background objects, etc. Therefore, a vector of ordered element indices is used to refer to elements (BLASes) assigned to each TLAS node. This keeps TLAS nodes small enough to fit into L1 cache, improving traversal performance.

Currently, BVHs are built separately from scratch for every frame.
Ideally, later on, TLAS would be updated incrementally for small deformations and only rebuilt for larger ones (or every few frames for smaller ones).

## Extras
* `ndarray.h` - C implementation of an n-dimensional array that stores data in a flat buffer while exposing ndarray-like indexing. It relies on C macros to specify data types, which limits compile-time checks (which, in hindsight, is not the best idea due to lack of compile-time checks); it is currently unused and kept mainly for experimentation and reference.
---

# Requirements
 * Python 3.11
 * Pyvale 2026.1.0 and all dependencies

A more detailed build and usage guide (including how this module is invoked through Pyvale) will be added once the API stabilises.
