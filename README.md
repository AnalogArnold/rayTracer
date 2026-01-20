# Overview

`rayTracer` is a work-in-progress C++ ray tracing module intended to be integrated into a larger open-source engineering simulation toolkit - [Pyvale](https://computer-aided-validation-laboratory.github.io/pyvale/). This repository is currently a rough development space rather than a polished standalone application — although, it should, in principle, be able to render images in its current state, provided the meshes are given in Pyvale-compatible simdata format.

> ⚠️ **Status:** Early-stage, experimental, with minimal documentation, and subject to frequent breaking changes.

---

## Project goals

- Provide a flexible, CPU-based ray tracing **module** with the potential expansion to GPU parallelization later on.
- Focus on engineering and scientific rendering use cases, primarily simulating [Digital Image Correlation](https://www.zeiss.co.uk/metrology/explore/topics/digital-image-correlation.html)
    - This means using higher-order elements, not just triangles, to accurately capture complex geometries.
    - Avoiding approximations that would speed up the rendering time at the expense of scientific accuracy.
    - 
At this stage, the emphasis is on getting the core MVP done to be able to perform preliminary research analysis.

---
# Current features
WIP
