#include <stddef.h> // size_t
#include <stdbool.h> // bool
#include <stdio.h>
#include <stdlib.h> // malloc, free

// C++ equivalents
//#include <cstddef>
//#include <cstdio>
//#include <cstdlib>

// Enum to define errors in NDArray
typedef enum {
    NDARRAY_OK = 0,
    NDARRAY_ELEMS_WRONG_LEN_FOR_DIMS,
    NDARRAY_INDEX_OUT_OF_BOUNDS,
    NDARRAY_DIM_OUT_OF_BOUNDS,
    NDARRAY_INDICES_WRONG_LEN_FOR_DIMS,
    NDARRAY_DIM_MISMATCH
} NDArrayError;

// C structs:
// struct name {data}; - Requires using the keyword every time it is used, e.g., struct MyStruct struct;
// typedef struct {data} name; - Doesn't require the use of keyword every time

typedef struct {
    double *elems; // external storage, not owned by NDArray. Need double explicitly in C
    size_t *dims; // Heap-owned copy of dimension sizes
    size_t *strides; // Heap-owned strides
    size_t ndims; // Number of dimensions
    size_t nelems; // Total number of elements
} NDArray;

// Initialization
NDArrayError
ndarray_init(NDArray *arr,
    double *elems,
    size_t nelems,
    const size_t *dims,
    size_t ndims) {
    if (ndims == 0) return NDARRAY_ELEMS_WRONG_LEN_FOR_DIMS; // No dimensions

    // Calculate the product of all dimensions
    size_t dim_prod = dims[0];
    for (size_t i = 1; i < ndims; ++i) {
        dim_prod *= dims[i];
    }
    if (nelems != dim_prod) return NDARRAY_ELEMS_WRONG_LEN_FOR_DIMS; // Number of elements does not match product of dimensions

    // Allocate the memory for the heap copies
    // arr->dims = malloc(ndims * sizeof(size_t)); // Omit explicit cast, but the result is converted to size_t* under the hood due to how malloc works (returns void*, which gets converted to any pointer type)
    arr->dims = (size_t *)malloc(ndims * sizeof(size_t)); // Explicit cast; required in C++, but not C. Same behaviour, just shows the conversion in the source. Might result in undefined behaviour if I forget to include <stdlib.h>
    if (!arr->dims) return NDARRAY_ELEMS_WRONG_LEN_FOR_DIMS; // Error with assigning

    arr->strides = (size_t *)malloc(ndims * sizeof(size_t));
    if (!arr->strides) {
        free(arr->dims); // Free the dimensions array if we failed to allocate the strides array
        return NDARRAY_ELEMS_WRONG_LEN_FOR_DIMS;
    }

    for (size_t i = 0; i < ndims; ++i) {
        arr->dims[i] = dims[i];
    }

    arr->elems = elems;
    arr->ndims = ndims;
    arr->nelems = nelems;

    // Compute row-major (C-style) strides (calculate flat stride)
    for (size_t dim = 0; dim < ndims; ++dim) {
        size_t stride = 1;
        for (size_t j = dim + 1; j < ndims; ++j) {
            stride *= arr->dims[j];
        }
        arr->strides[dim] = stride;
    }
    return NDARRAY_OK;
}

// Deinitialization
void
ndarray_deinit(NDArray *arr) {
    // Free dims and strides (elems is owned by the caller, so no need to)
    free(arr->dims);
    free(arr->strides);
    arr->dims = NULL;
    arr->strides = NULL;
}

// Index computation
NDArrayError
ndarray_get_flat_index(const NDArray *arr,
    const size_t *indices,
    size_t indices_len,
    size_t *out_index) {
// Convert ND index to its flat-array equivalent
    if (indices_len != arr->ndims) return NDARRAY_INDICES_WRONG_LEN_FOR_DIMS; // Incorrect number of indices

    size_t flat = 0; // Essentially the standard row-major offset
    for (size_t dim = 0; dim < arr->ndims; ++dim) {
        size_t ind = indices[dim];
        if (ind >= arr->dims[dim]) return NDARRAY_INDEX_OUT_OF_BOUNDS; // Index out of bounds
        flat += ind * arr->strides[dim];
    }
    *out_index = flat; // Output index that we return (passed as input argument that we modify since we return status here)
    return NDARRAY_OK;
}

NDArrayError
ndarray_set_index(NDArray *arr,
    const size_t *indices,
    size_t indices_len,
    double value) {
    // Set value at an index
    size_t flat;
    NDArrayError err = ndarray_get_flat_index(arr, indices, indices_len, &flat);
    if (err != NDARRAY_OK) return err;
    arr->elems[flat] = value;
    return NDARRAY_OK;
}

NDArrayError
ndarray_get(const NDArray *arr,
    const size_t *indices,
    size_t indices_len,
    double *out_value) {
    // Get value at an index
    size_t flat;
    NDArrayError err = ndarray_get_flat_index(arr, indices, indices_len, &flat);
    if (err != NDARRAY_OK) return err;
    *out_value = arr->elems[flat];
    return NDARRAY_OK;
}

void ndarray_fill(NDArray *arr,
    double value) {
    // Fill the array with a given value
    for (size_t i = 0; i < arr->nelems; ++i) {
        arr->elems[i] = value;
    }
}

void ndarray_print(const NDArray *arr) {
    // Print the array contents
    for (size_t i = 0; i < arr->nelems; ++i) {
        printf("%f ", arr->elems[i]);
    }
    printf("\n");
}

bool ndarray_match_dimensions(const NDArray *array1,
    const NDArray *array2) {
    // Check if two arrays have the same dimensions
    if (array1->ndims != array2->ndims) return false; // Different number of dimensions
    // Compare size for  each constituing dimension
    for (size_t i = 0; i < array1->ndims; ++i) {
        if (array1->dims[i] != array2->dims[i]) {
            return false;
        }
    }
    return true;
}
NDArrayError
ndarray_add_in_place(NDArray *self, const NDArray *other) {
    // Add two arrays together in-place
    if (!ndarray_match_dimensions(self, other)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < self->nelems; ++i) {
        self->elems[i] += other->elems[i];
    }
    return NDARRAY_OK;
}

NDArrayError
ndarray_subtract_in_place(NDArray *self, const NDArray *other) {
    // Subtract two arrays together in-place
    if (!ndarray_match_dimensions(self, other)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < self->nelems; ++i) {
        self->elems[i] -= other->elems[i];
    }
    return NDARRAY_OK;
}

NDArrayError
ndarray_multiply_in_place(NDArray *self, const NDArray *other) {
    // Multiply two arrays together in-place (element-wise)
    if (!ndarray_match_dimensions(self, other)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < self->nelems; ++i) {
        self->elems[i] *= other->elems[i];
    }
    return NDARRAY_OK;
}

NDArrayError
ndarray_divide_in_place(NDArray *self, const NDArray *other) {
    // Divide two arrays together in-place (element-wise)
    if (!ndarray_match_dimensions(self, other)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < self->nelems; ++i) {
        self->elems[i] /= other->elems[i];
    }
    return NDARRAY_OK;
}

void
ndarray_scalar_multiply(NDArray *self,
    double scalar) {
    // Multiply an array by a scalar
    for (size_t i = 0; i < self->nelems; ++i) {
        self->elems[i] *= scalar;
    }
}

NDArrayError
ndarray_get_slice(const NDArray *arr,
    size_t *fixed_indices, // these will be modified
    size_t fixed_length,
    size_t slice_fixed_dim,
    double **out_ptr,
    size_t *out_length) {
    // Get a slice in 1D
    // C can't return a length-tracked slice directly, so we return a pointer and the output length
    if (fixed_length != arr->ndims) return NDARRAY_INDICES_WRONG_LEN_FOR_DIMS;
    if (slice_fixed_dim +1 >= arr->ndims) return NDARRAY_DIM_OUT_OF_BOUNDS;

    // Zero off all other dimensions to ensure slice starts at the correct location
    for (size_t i = slice_fixed_dim + 1; i < fixed_length; ++i) {
        fixed_indices[i] = 0;
    }
    size_t start_index;
    NDArrayError err_start = ndarray_get_flat_index(arr, fixed_indices, fixed_length, &start_index);
    if (err_start != NDARRAY_OK) return err_start; // Error getting start index

    size_t slice_length = arr->dims[slice_fixed_dim];
    *out_ptr = arr->elems + start_index;
    *out_length = slice_length;
    return NDARRAY_OK;
}






NDArrayError
ndarray_add(const NDArray *array1, const NDArray *array2, NDArray *out_array) {
    // Add two arrays together and return a third array
    if (!ndarray_match_dimensions(array1, array2) || !ndarray_match_dimensions(array1, out_array)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    //ndarray_fill(out_array, 0);
    // Could also fill out_array with 0s and run ndarray_add_in_place twice with array1 and array2, but then we have to basically go into 3 for loops, so not the best option
    for (size_t i = 0; i < array1->nelems; ++i) {
        out_array->elems[i] = array1->elems[i] + array2->elems[i];
    }
    return NDARRAY_OK;
}
NDArrayError
ndarray_subtract(const NDArray *array1, const NDArray *array2, NDArray *out_array) {
    // Subtract two arrays together and return a third array
    if (!ndarray_match_dimensions(array1, array2) || !ndarray_match_dimensions(array1, out_array)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < array1->nelems; ++i) {
        out_array->elems[i] = array1->elems[i] - array2->elems[i];
    }
    return NDARRAY_OK;
}

NDArrayError
ndarray_multiply_elemWise(const NDArray *array1, const NDArray *array2, NDArray *out_array) {
    // Multiply two arrays together and return a third array
    if (!ndarray_match_dimensions(array1, array2) || !ndarray_match_dimensions(array1, out_array)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < array1->nelems; ++i) {
        out_array->elems[i] = array1->elems[i] * array2->elems[i];
    }
    return NDARRAY_OK;
}

NDArrayError
ndarray_divide_elemWise(const NDArray *array1, const NDArray *array2, NDArray *out_array) {
    // Divide two arrays together and return a third array
    if (!ndarray_match_dimensions(array1, array2) || !ndarray_match_dimensions(array1, out_array)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < array1->nelems; ++i) {
        out_array->elems[i] = array1->elems[i] / array2->elems[i];
    }
    return NDARRAY_OK;
}


