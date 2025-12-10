#include <stddef.h> // size_t
#include <stdbool.h> // bool
#include <stdio.h>
#include <stdlib.h> // malloc, free

// C++ equivalents
//#include <cstddef>
//#include <cstdio>
//#include <cstdlib>

// Written for double (_double), int (_int), unsigned int (_uint), and long long (_llong).
// Unit tests at the very bottom.

// Functions:
// ndarray_init_
//ndaray_deinit_
// ndarray_get_flat_index_
// ndarray_set_index_
// ndarray_get_
// ndaray_fill_
// ndarray_print_
// ndarray_match_dimensions_
// ndarray_add_in_place_
// ndarray_subtract_in_place_
// ndarray_multiply_in_place_
// ndarray_divide_in_place_
// ndarra_scalar_multiply_
// ndarray_get_slice_
// ndarray_add_
// ndarray_subtract_
// ndarray_multiply_elemWise_
// ndarray_divide_elemWise_


// Enum to define errors in NDArray_double
typedef enum {
    NDARRAY_OK = 0,
    NDARRAY_ELEMS_WRONG_LEN_FOR_DIMS,
    NDARRAY_INDEX_OUT_OF_BOUNDS,
    NDARRAY_DIM_OUT_OF_BOUNDS,
    NDARRAY_INDICES_WRONG_LEN_FOR_DIMS,
    NDARRAY_DIM_MISMATCH
} NDArrayError;

//////////////////////////////////////////////////////// DOUBLE //////////////////////////////////////////////////////

// C structs:
// struct name {data}; - Requires using the keyword every time it is used, e.g., struct MyStruct struct;
// typedef struct {data} name; - Doesn't require the use of keyword every time

typedef struct {
    double *elems; // external storage, not owned by NDArray_double. Need double explicitly in C
    size_t *dims; // Heap-owned copy of dimension sizes
    size_t *strides; // Heap-owned strides
    size_t ndims; // Number of dimensions
    size_t nelems; // Total number of elements
} NDArray_double;

// Initialization
NDArrayError
ndarray_init_double(NDArray_double *arr,
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
ndarray_deinit_double(NDArray_double *arr) {
    // Free dims and strides (elems is owned by the caller, so no need to)
    free(arr->dims);
    free(arr->strides);
    arr->dims = NULL;
    arr->strides = NULL;
}

// Index computation
NDArrayError
ndarray_get_flat_index_double(const NDArray_double *arr,
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
ndarray_set_index_double(NDArray_double *arr,
    const size_t *indices,
    size_t indices_len,
    double value) {
    // Set value at an index
    size_t flat;
    NDArrayError err = ndarray_get_flat_index_double(arr, indices, indices_len, &flat);
    if (err != NDARRAY_OK) return err;
    arr->elems[flat] = value;
    return NDARRAY_OK;
}

NDArrayError
ndarray_get_double(const NDArray_double *arr,
    const size_t *indices,
    size_t indices_len,
    double *out_value) {
    // Get value at an index
    size_t flat;
    NDArrayError err = ndarray_get_flat_index_double(arr, indices, indices_len, &flat);
    if (err != NDARRAY_OK) return err;
    *out_value = arr->elems[flat];
    return NDARRAY_OK;
}

void ndarray_fill_double(NDArray_double *arr,
    double value) {
    // Fill the array with a given value
    for (size_t i = 0; i < arr->nelems; ++i) {
        arr->elems[i] = value;
    }
}

void ndarray_print_double(const NDArray_double *arr) {
    // Print the array contents
    for (size_t i = 0; i < arr->nelems; ++i) {
        printf("%f ", arr->elems[i]);
    }
    printf("\n");
}

bool ndarray_match_dimensions_double(const NDArray_double *array1,
    const NDArray_double *array2) {
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
ndarray_add_in_place_double(NDArray_double *self, const NDArray_double *other) {
    // Add two arrays together in-place
    if (!ndarray_match_dimensions_double(self, other)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < self->nelems; ++i) {
        self->elems[i] += other->elems[i];
    }
    return NDARRAY_OK;
}

NDArrayError
ndarray_subtract_in_place_double(NDArray_double *self, const NDArray_double *other) {
    // Subtract two arrays together in-place
    if (!ndarray_match_dimensions_double(self, other)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < self->nelems; ++i) {
        self->elems[i] -= other->elems[i];
    }
    return NDARRAY_OK;
}

NDArrayError
ndarray_multiply_in_place_double(NDArray_double *self, const NDArray_double *other) {
    // Multiply two arrays together in-place (element-wise)
    if (!ndarray_match_dimensions_double(self, other)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < self->nelems; ++i) {
        self->elems[i] *= other->elems[i];
    }
    return NDARRAY_OK;
}

NDArrayError
ndarray_divide_in_place_double(NDArray_double *self, const NDArray_double *other) {
    // Divide two arrays together in-place (element-wise)
    if (!ndarray_match_dimensions_double(self, other)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < self->nelems; ++i) {
        self->elems[i] /= other->elems[i];
    }
    return NDARRAY_OK;
}

void
ndarray_scalar_multiply_double(NDArray_double *self,
    double scalar) {
    // Multiply an array by a scalar
    for (size_t i = 0; i < self->nelems; ++i) {
        self->elems[i] *= scalar;
    }
}

NDArrayError
ndarray_get_slice_double(const NDArray_double *arr,
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
    NDArrayError err_start = ndarray_get_flat_index_double(arr, fixed_indices, fixed_length, &start_index);
    if (err_start != NDARRAY_OK) return err_start; // Error getting start index

    size_t slice_length = arr->dims[slice_fixed_dim];
    *out_ptr = arr->elems + start_index;
    *out_length = slice_length;
    return NDARRAY_OK;
}

NDArrayError
ndarray_add_double(const NDArray_double *array1, const NDArray_double *array2, NDArray_double *out_array) {
    // Add two arrays together and return a third array
    if (!ndarray_match_dimensions_double(array1, array2) || !ndarray_match_dimensions_double(array1, out_array)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    //ndarray_fill(out_array, 0);
    // Could also fill out_array with 0s and run ndarray_add_in_place twice with array1 and array2, but then we have to basically go into 3 for loops, so not the best option
    for (size_t i = 0; i < array1->nelems; ++i) {
        out_array->elems[i] = array1->elems[i] + array2->elems[i];
    }
    return NDARRAY_OK;
}
NDArrayError
ndarray_subtract_double(const NDArray_double *array1, const NDArray_double *array2, NDArray_double *out_array) {
    // Subtract two arrays together and return a third array
    if (!ndarray_match_dimensions_double(array1, array2) || !ndarray_match_dimensions_double(array1, out_array)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < array1->nelems; ++i) {
        out_array->elems[i] = array1->elems[i] - array2->elems[i];
    }
    return NDARRAY_OK;
}

NDArrayError
ndarray_multiply_elemWise_double(const NDArray_double *array1, const NDArray_double *array2, NDArray_double *out_array) {
    // Multiply two arrays together and return a third array
    if (!ndarray_match_dimensions_double(array1, array2) || !ndarray_match_dimensions_double(array1, out_array)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < array1->nelems; ++i) {
        out_array->elems[i] = array1->elems[i] * array2->elems[i];
    }
    return NDARRAY_OK;
}

NDArrayError
ndarray_divide_elemWise_double(const NDArray_double *array1, const NDArray_double *array2, NDArray_double *out_array) {
    // Divide two arrays together and return a third array
    if (!ndarray_match_dimensions_double(array1, array2) || !ndarray_match_dimensions_double(array1, out_array)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < array1->nelems; ++i) {
        out_array->elems[i] = array1->elems[i] / array2->elems[i];
    }
    return NDARRAY_OK;
}


///////////////////////////////////////////////////////// INT ///////////////////////////////////////////////////////


typedef struct {
    int *elems; // external storage, not owned by NDArray_int. Need int explicitly in C
    size_t *dims; // Heap-owned copy of dimension sizes
    size_t *strides; // Heap-owned strides
    size_t ndims; // Number of dimensions
    size_t nelems; // Total number of elements
} NDArray_int;

// Initialization
NDArrayError
ndarray_init_int(NDArray_int *arr,
    int *elems,
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
ndarray_deinit_int(NDArray_int *arr) {
    // Free dims and strides (elems is owned by the caller, so no need to)
    free(arr->dims);
    free(arr->strides);
    arr->dims = NULL;
    arr->strides = NULL;
}

// Index computation
NDArrayError
ndarray_get_flat_index_int(const NDArray_int *arr,
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
ndarray_set_index_int(NDArray_int *arr,
    const size_t *indices,
    size_t indices_len,
    int value) {
    // Set value at an index
    size_t flat;
    NDArrayError err = ndarray_get_flat_index_int(arr, indices, indices_len, &flat);
    if (err != NDARRAY_OK) return err;
    arr->elems[flat] = value;
    return NDARRAY_OK;
}

NDArrayError
ndarray_get_int(const NDArray_int *arr,
    const size_t *indices,
    size_t indices_len,
    int *out_value) {
    // Get value at an index
    size_t flat;
    NDArrayError err = ndarray_get_flat_index_int(arr, indices, indices_len, &flat);
    if (err != NDARRAY_OK) return err;
    *out_value = arr->elems[flat];
    return NDARRAY_OK;
}

void ndarray_fill_int(NDArray_int *arr,
    int value) {
    // Fill the array with a given value
    for (size_t i = 0; i < arr->nelems; ++i) {
        arr->elems[i] = value;
    }
}

void ndarray_print_int(const NDArray_int *arr) {
    // Print the array contents
    for (size_t i = 0; i < arr->nelems; ++i) {
        printf("%f ", arr->elems[i]);
    }
    printf("\n");
}

bool ndarray_match_dimensions_int(const NDArray_int *array1,
    const NDArray_int *array2) {
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
ndarray_add_in_place_int(NDArray_int *self, const NDArray_int *other) {
    // Add two arrays together in-place
    if (!ndarray_match_dimensions_int(self, other)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < self->nelems; ++i) {
        self->elems[i] += other->elems[i];
    }
    return NDARRAY_OK;
}

NDArrayError
ndarray_subtract_in_place_int(NDArray_int *self, const NDArray_int *other) {
    // Subtract two arrays together in-place
    if (!ndarray_match_dimensions_int(self, other)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < self->nelems; ++i) {
        self->elems[i] -= other->elems[i];
    }
    return NDARRAY_OK;
}

NDArrayError
ndarray_multiply_in_place_int(NDArray_int *self, const NDArray_int *other) {
    // Multiply two arrays together in-place (element-wise)
    if (!ndarray_match_dimensions_int(self, other)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < self->nelems; ++i) {
        self->elems[i] *= other->elems[i];
    }
    return NDARRAY_OK;
}

NDArrayError
ndarray_divide_in_place_int(NDArray_int *self, const NDArray_int *other) {
    // Divide two arrays together in-place (element-wise)
    if (!ndarray_match_dimensions_int(self, other)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < self->nelems; ++i) {
        self->elems[i] /= other->elems[i];
    }
    return NDARRAY_OK;
}

void
ndarray_scalar_multiply_int(NDArray_int *self,
    int scalar) {
    // Multiply an array by a scalar
    for (size_t i = 0; i < self->nelems; ++i) {
        self->elems[i] *= scalar;
    }
}

NDArrayError
ndarray_get_slice_int(const NDArray_int *arr,
    size_t *fixed_indices, // these will be modified
    size_t fixed_length,
    size_t slice_fixed_dim,
    int **out_ptr,
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
    NDArrayError err_start = ndarray_get_flat_index_int(arr, fixed_indices, fixed_length, &start_index);
    if (err_start != NDARRAY_OK) return err_start; // Error getting start index

    size_t slice_length = arr->dims[slice_fixed_dim];
    *out_ptr = arr->elems + start_index;
    *out_length = slice_length;
    return NDARRAY_OK;
}

NDArrayError
ndarray_add_int(const NDArray_int *array1, const NDArray_int *array2, NDArray_int *out_array) {
    // Add two arrays together and return a third array
    if (!ndarray_match_dimensions_int(array1, array2) || !ndarray_match_dimensions_int(array1, out_array)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    //ndarray_fill(out_array, 0);
    // Could also fill out_array with 0s and run ndarray_add_in_place twice with array1 and array2, but then we have to basically go into 3 for loops, so not the best option
    for (size_t i = 0; i < array1->nelems; ++i) {
        out_array->elems[i] = array1->elems[i] + array2->elems[i];
    }
    return NDARRAY_OK;
}
NDArrayError
ndarray_subtract_int(const NDArray_int *array1, const NDArray_int *array2, NDArray_int *out_array) {
    // Subtract two arrays together and return a third array
    if (!ndarray_match_dimensions_int(array1, array2) || !ndarray_match_dimensions_int(array1, out_array)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < array1->nelems; ++i) {
        out_array->elems[i] = array1->elems[i] - array2->elems[i];
    }
    return NDARRAY_OK;
}

NDArrayError
ndarray_multiply_elemWise_int(const NDArray_int *array1, const NDArray_int *array2, NDArray_int *out_array) {
    // Multiply two arrays together and return a third array
    if (!ndarray_match_dimensions_int(array1, array2) || !ndarray_match_dimensions_int(array1, out_array)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < array1->nelems; ++i) {
        out_array->elems[i] = array1->elems[i] * array2->elems[i];
    }
    return NDARRAY_OK;
}

NDArrayError
ndarray_divide_elemWise_int(const NDArray_int *array1, const NDArray_int *array2, NDArray_int *out_array) {
    // Divide two arrays together and return a third array
    if (!ndarray_match_dimensions_int(array1, array2) || !ndarray_match_dimensions_int(array1, out_array)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < array1->nelems; ++i) {
        out_array->elems[i] = array1->elems[i] / array2->elems[i];
    }
    return NDARRAY_OK;
}

////////////////////////////////////////////////////// UNSIGNED INT ////////////////////////////////////////////////////


typedef struct {
    unsigned int *elems; // external storage, not owned by NDArray_uint. Need unsigned int explicitly in C
    size_t *dims; // Heap-owned copy of dimension sizes
    size_t *strides; // Heap-owned strides
    size_t ndims; // Number of dimensions
    size_t nelems; // Total number of elements
} NDArray_uint;

// Initialization
NDArrayError
ndarray_init_uint(NDArray_uint *arr,
    unsigned int *elems,
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
ndarray_deinit_uint(NDArray_uint *arr) {
    // Free dims and strides (elems is owned by the caller, so no need to)
    free(arr->dims);
    free(arr->strides);
    arr->dims = NULL;
    arr->strides = NULL;
}

// Index computation
NDArrayError
ndarray_get_flat_index_uint(const NDArray_uint *arr,
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
ndarray_set_index_uint(NDArray_uint *arr,
    const size_t *indices,
    size_t indices_len,
    unsigned int value) {
    // Set value at an index
    size_t flat;
    NDArrayError err = ndarray_get_flat_index_uint(arr, indices, indices_len, &flat);
    if (err != NDARRAY_OK) return err;
    arr->elems[flat] = value;
    return NDARRAY_OK;
}

NDArrayError
ndarray_get_uint(const NDArray_uint *arr,
    const size_t *indices,
    size_t indices_len,
    unsigned int *out_value) {
    // Get value at an index
    size_t flat;
    NDArrayError err = ndarray_get_flat_index_uint(arr, indices, indices_len, &flat);
    if (err != NDARRAY_OK) return err;
    *out_value = arr->elems[flat];
    return NDARRAY_OK;
}

void ndarray_fill_uint(NDArray_uint *arr,
    unsigned int value) {
    // Fill the array with a given value
    for (size_t i = 0; i < arr->nelems; ++i) {
        arr->elems[i] = value;
    }
}

void ndarray_print_uint(const NDArray_uint *arr) {
    // Print the array contents
    for (size_t i = 0; i < arr->nelems; ++i) {
        printf("%f ", arr->elems[i]);
    }
    printf("\n");
}

bool ndarray_match_dimensions_uint(const NDArray_uint *array1,
    const NDArray_uint *array2) {
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
ndarray_add_in_place_uint(NDArray_uint *self, const NDArray_uint *other) {
    // Add two arrays together in-place
    if (!ndarray_match_dimensions_uint(self, other)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < self->nelems; ++i) {
        self->elems[i] += other->elems[i];
    }
    return NDARRAY_OK;
}

NDArrayError
ndarray_subtract_in_place_uint(NDArray_uint *self, const NDArray_uint *other) {
    // Subtract two arrays together in-place
    if (!ndarray_match_dimensions_uint(self, other)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < self->nelems; ++i) {
        self->elems[i] -= other->elems[i];
    }
    return NDARRAY_OK;
}

NDArrayError
ndarray_multiply_in_place_uint(NDArray_uint *self, const NDArray_uint *other) {
    // Multiply two arrays together in-place (element-wise)
    if (!ndarray_match_dimensions_uint(self, other)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < self->nelems; ++i) {
        self->elems[i] *= other->elems[i];
    }
    return NDARRAY_OK;
}

NDArrayError
ndarray_divide_in_place_uint(NDArray_uint *self, const NDArray_uint *other) {
    // Divide two arrays together in-place (element-wise)
    if (!ndarray_match_dimensions_uint(self, other)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < self->nelems; ++i) {
        self->elems[i] /= other->elems[i];
    }
    return NDARRAY_OK;
}

void
ndarray_scalar_multiply_uint(NDArray_uint *self,
    unsigned int scalar) {
    // Multiply an array by a scalar
    for (size_t i = 0; i < self->nelems; ++i) {
        self->elems[i] *= scalar;
    }
}

NDArrayError
ndarray_get_slice_uint(const NDArray_uint *arr,
    size_t *fixed_indices, // these will be modified
    size_t fixed_length,
    size_t slice_fixed_dim,
    unsigned int **out_ptr,
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
    NDArrayError err_start = ndarray_get_flat_index_uint(arr, fixed_indices, fixed_length, &start_index);
    if (err_start != NDARRAY_OK) return err_start; // Error getting start index

    size_t slice_length = arr->dims[slice_fixed_dim];
    *out_ptr = arr->elems + start_index;
    *out_length = slice_length;
    return NDARRAY_OK;
}

NDArrayError
ndarray_add_uint(const NDArray_uint *array1, const NDArray_uint *array2, NDArray_uint *out_array) {
    // Add two arrays together and return a third array
    if (!ndarray_match_dimensions_uint(array1, array2) || !ndarray_match_dimensions_uint(array1, out_array)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    //ndarray_fill(out_array, 0);
    // Could also fill out_array with 0s and run ndarray_add_in_place twice with array1 and array2, but then we have to basically go into 3 for loops, so not the best option
    for (size_t i = 0; i < array1->nelems; ++i) {
        out_array->elems[i] = array1->elems[i] + array2->elems[i];
    }
    return NDARRAY_OK;
}
NDArrayError
ndarray_subtract_uint(const NDArray_uint *array1, const NDArray_uint *array2, NDArray_uint *out_array) {
    // Subtract two arrays together and return a third array
    if (!ndarray_match_dimensions_uint(array1, array2) || !ndarray_match_dimensions_uint(array1, out_array)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < array1->nelems; ++i) {
        out_array->elems[i] = array1->elems[i] - array2->elems[i];
    }
    return NDARRAY_OK;
}

NDArrayError
ndarray_multiply_elemWise_uint(const NDArray_uint *array1, const NDArray_uint *array2, NDArray_uint *out_array) {
    // Multiply two arrays together and return a third array
    if (!ndarray_match_dimensions_uint(array1, array2) || !ndarray_match_dimensions_uint(array1, out_array)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < array1->nelems; ++i) {
        out_array->elems[i] = array1->elems[i] * array2->elems[i];
    }
    return NDARRAY_OK;
}

NDArrayError
ndarray_divide_elemWise_uint(const NDArray_uint *array1, const NDArray_uint *array2, NDArray_uint *out_array) {
    // Divide two arrays together and return a third array
    if (!ndarray_match_dimensions_uint(array1, array2) || !ndarray_match_dimensions_uint(array1, out_array)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < array1->nelems; ++i) {
        out_array->elems[i] = array1->elems[i] / array2->elems[i];
    }
    return NDARRAY_OK;
}

/////////////////////////////////////////////////////// LONG LONG /////////////////////////////////////////////////////

typedef struct {
    long long *elems; // external storage, not owned by NDArray_llong. Need long long explicitly in C
    size_t *dims; // Heap-owned copy of dimension sizes
    size_t *strides; // Heap-owned strides
    size_t ndims; // Number of dimensions
    size_t nelems; // Total number of elements
} NDArray_llong;

// Initialization
NDArrayError
ndarray_init_llong(NDArray_llong *arr,
    long long *elems,
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
ndarray_deinit_llong(NDArray_llong *arr) {
    // Free dims and strides (elems is owned by the caller, so no need to)
    free(arr->dims);
    free(arr->strides);
    arr->dims = NULL;
    arr->strides = NULL;
}

// Index computation
NDArrayError
ndarray_get_flat_index_llong(const NDArray_llong *arr,
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
ndarray_set_index_llong(NDArray_llong *arr,
    const size_t *indices,
    size_t indices_len,
    long long value) {
    // Set value at an index
    size_t flat;
    NDArrayError err = ndarray_get_flat_index_llong(arr, indices, indices_len, &flat);
    if (err != NDARRAY_OK) return err;
    arr->elems[flat] = value;
    return NDARRAY_OK;
}

NDArrayError
ndarray_get_llong(const NDArray_llong *arr,
    const size_t *indices,
    size_t indices_len,
    long long *out_value) {
    // Get value at an index
    size_t flat;
    NDArrayError err = ndarray_get_flat_index_llong(arr, indices, indices_len, &flat);
    if (err != NDARRAY_OK) return err;
    *out_value = arr->elems[flat];
    return NDARRAY_OK;
}

void ndarray_fill_llong(NDArray_llong *arr,
    long long value) {
    // Fill the array with a given value
    for (size_t i = 0; i < arr->nelems; ++i) {
        arr->elems[i] = value;
    }
}

void ndarray_print_llong(const NDArray_llong *arr) {
    // Print the array contents
    for (size_t i = 0; i < arr->nelems; ++i) {
        printf("%f ", arr->elems[i]);
    }
    printf("\n");
}

bool ndarray_match_dimensions_llong(const NDArray_llong *array1,
    const NDArray_llong *array2) {
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
ndarray_add_in_place_llong(NDArray_llong *self, const NDArray_llong *other) {
    // Add two arrays together in-place
    if (!ndarray_match_dimensions_llong(self, other)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < self->nelems; ++i) {
        self->elems[i] += other->elems[i];
    }
    return NDARRAY_OK;
}

NDArrayError
ndarray_subtract_in_place_llong(NDArray_llong *self, const NDArray_llong *other) {
    // Subtract two arrays together in-place
    if (!ndarray_match_dimensions_llong(self, other)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < self->nelems; ++i) {
        self->elems[i] -= other->elems[i];
    }
    return NDARRAY_OK;
}

NDArrayError
ndarray_multiply_in_place_llong(NDArray_llong *self, const NDArray_llong *other) {
    // Multiply two arrays together in-place (element-wise)
    if (!ndarray_match_dimensions_llong(self, other)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < self->nelems; ++i) {
        self->elems[i] *= other->elems[i];
    }
    return NDARRAY_OK;
}

NDArrayError
ndarray_divide_in_place_llong(NDArray_llong *self, const NDArray_llong *other) {
    // Divide two arrays together in-place (element-wise)
    if (!ndarray_match_dimensions_llong(self, other)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < self->nelems; ++i) {
        self->elems[i] /= other->elems[i];
    }
    return NDARRAY_OK;
}

void
ndarray_scalar_multiply_llong(NDArray_llong *self,
    long long scalar) {
    // Multiply an array by a scalar
    for (size_t i = 0; i < self->nelems; ++i) {
        self->elems[i] *= scalar;
    }
}

NDArrayError
ndarray_get_slice_llong(const NDArray_llong *arr,
    size_t *fixed_indices, // these will be modified
    size_t fixed_length,
    size_t slice_fixed_dim,
    long long **out_ptr,
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
    NDArrayError err_start = ndarray_get_flat_index_llong(arr, fixed_indices, fixed_length, &start_index);
    if (err_start != NDARRAY_OK) return err_start; // Error getting start index

    size_t slice_length = arr->dims[slice_fixed_dim];
    *out_ptr = arr->elems + start_index;
    *out_length = slice_length;
    return NDARRAY_OK;
}

NDArrayError
ndarray_add_llong(const NDArray_llong *array1, const NDArray_llong *array2, NDArray_llong *out_array) {
    // Add two arrays together and return a third array
    if (!ndarray_match_dimensions_llong(array1, array2) || !ndarray_match_dimensions_llong(array1, out_array)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    //ndarray_fill(out_array, 0);
    // Could also fill out_array with 0s and run ndarray_add_in_place twice with array1 and array2, but then we have to basically go into 3 for loops, so not the best option
    for (size_t i = 0; i < array1->nelems; ++i) {
        out_array->elems[i] = array1->elems[i] + array2->elems[i];
    }
    return NDARRAY_OK;
}
NDArrayError
ndarray_subtract_llong(const NDArray_llong *array1, const NDArray_llong *array2, NDArray_llong *out_array) {
    // Subtract two arrays together and return a third array
    if (!ndarray_match_dimensions_llong(array1, array2) || !ndarray_match_dimensions_llong(array1, out_array)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < array1->nelems; ++i) {
        out_array->elems[i] = array1->elems[i] - array2->elems[i];
    }
    return NDARRAY_OK;
}

NDArrayError
ndarray_multiply_elemWise_llong(const NDArray_llong *array1, const NDArray_llong *array2, NDArray_llong *out_array) {
    // Multiply two arrays together and return a third array
    if (!ndarray_match_dimensions_llong(array1, array2) || !ndarray_match_dimensions_llong(array1, out_array)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < array1->nelems; ++i) {
        out_array->elems[i] = array1->elems[i] * array2->elems[i];
    }
    return NDARRAY_OK;
}

NDArrayError
ndarray_divide_elemWise_llong(const NDArray_llong *array1, const NDArray_llong *array2, NDArray_llong *out_array) {
    // Divide two arrays together and return a third array
    if (!ndarray_match_dimensions_llong(array1, array2) || !ndarray_match_dimensions_llong(array1, out_array)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < array1->nelems; ++i) {
        out_array->elems[i] = array1->elems[i] / array2->elems[i];
    }
    return NDARRAY_OK;
}

/////////////////////////////////////////////////////// UNIT TESTS /////////////////////////////////////////////////////
// Tests (for doubles, which these functions were initially written for)
/*
#define TEST_PASS (0)
#define TEST_FAIL (1)

int total_tests = 0;
int passed_tests = 0;

void print_test_result(const char* name, int result) {
    total_tests++;
    if (result == TEST_PASS) {
        passed_tests++;
        printf("PASS: %s\n", name);
    } else {
        printf("FAIL: %s\n", name);
    }
}

int test_init_deinit() {
    // 1. Valid initialization (2x3 array)
    double data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    size_t dims[] = {2, 3};
    size_t nelems = 6;
    size_t ndims = 2;
    NDArray_double arr;

    if (ndarray_init_double(&arr, data, nelems, dims, ndims) != NDARRAY_OK) {
        return TEST_FAIL;
    }
    // Check core properties
    if (arr.ndims != ndims || arr.nelems != nelems) {
        ndarray_deinit_double(&arr);
        return TEST_FAIL;
    }
    // Check calculated strides (C-style/row-major: stride[0]=3, stride[1]=1)
    if (arr.strides[0] != 3 || arr.strides[1] != 1) {
        ndarray_deinit_double(&arr);
        return TEST_FAIL;
    }

    // 2. Invalid initialization (nelems mismatch)
    size_t wrong_nelems = 5;
    NDArray_double arr_err;
    if (ndarray_init_double(&arr_err, data, wrong_nelems, dims, ndims) != NDARRAY_ELEMS_WRONG_LEN_FOR_DIMS) {
        ndarray_deinit_double(&arr); // Deinit the good one
        return TEST_FAIL;
    }

    // 3. Deinitialization (manual check for no crash)
    ndarray_deinit_double(&arr);

    return TEST_PASS;
}

int test_indexing() {
    double data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    size_t dims[] = {2, 2, 3}; // A 2x2x3 array
    size_t nelems = 12;
    size_t ndims = 3;
    NDArray_double arr;
    ndarray_init_double(&arr, data, nelems, dims, ndims);

    // 1. Flat Index check: Index (1, 0, 2)
    // Formula: 1*stride[0] + 0*stride[1] + 2*stride[2]
    // Strides should be (6, 3, 1)
    // Flat index: 1*6 + 0*3 + 2*1 = 8
    size_t indices_102[] = {1, 0, 2};
    size_t flat_index;
    if (ndarray_get_flat_index_double(&arr, indices_102, ndims, &flat_index) != NDARRAY_OK || flat_index != 8) {
        ndarray_deinit_double(&arr);
        return TEST_FAIL;
    }

    // 2. Set/Get check: Set (0, 1, 1) to 99.9
    // Index (0, 1, 1): 0*6 + 1*3 + 1*1 = 4. Original value is 5.0
    size_t indices_011[] = {0, 1, 1};
    double new_value = 99.9;
    if (ndarray_set_index_double(&arr, indices_011, ndims, new_value) != NDARRAY_OK) {
        ndarray_deinit_double(&arr);
        return TEST_FAIL;
    }

    double retrieved_value;
    if (ndarray_get_double(&arr, indices_011, ndims, &retrieved_value) != NDARRAY_OK || retrieved_value != new_value) {
        ndarray_deinit_double(&arr);
        return TEST_FAIL;
    }

    // 3. Out-of-bounds check (index 2 for dim 0, which is size 2)
    size_t indices_oob[] = {2, 0, 0};
    if (ndarray_get_flat_index_double(&arr, indices_oob, ndims, &flat_index) != NDARRAY_INDEX_OUT_OF_BOUNDS) {
        ndarray_deinit_double(&arr);
        return TEST_FAIL;
    }

    ndarray_deinit_double(&arr);
    return TEST_PASS;
}

int test_array_utilities() {
    double data1[] = {0.0, 0.0, 0.0, 0.0};
    size_t dims1[] = {2, 2};
    NDArray_double arr1;
    ndarray_init_double(&arr1, data1, 4, dims1, 2);

    // 1. Fill check
    ndarray_fill_double(&arr1, 5.5);
    for (size_t i = 0; i < arr1.nelems; ++i) {
        if (arr1.elems[i] != 5.5) {
            ndarray_deinit_double(&arr1);
            return TEST_FAIL;
        }
    }

    // 2. Match dimensions check (same dims)
    double data2[] = {1.0, 2.0, 3.0, 4.0};
    size_t dims2[] = {2, 2};
    NDArray_double arr2;
    ndarray_init_double(&arr2, data2, 4, dims2, 2);

    if (!ndarray_match_dimensions_double(&arr1, &arr2)) {
        ndarray_deinit_double(&arr1);
        ndarray_deinit_double(&arr2);
        return TEST_FAIL;
    }

    // 3. Match dimensions check (different dims)
    double data3[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    size_t dims3[] = {3, 2}; // 3x2 vs 2x2
    NDArray_double arr3;
    ndarray_init_double(&arr3, data3, 6, dims3, 2);

    if (ndarray_match_dimensions_double(&arr1, &arr3)) {
        ndarray_deinit_double(&arr1);
        ndarray_deinit_double(&arr2);
        ndarray_deinit_double(&arr3);
        return TEST_FAIL;
    }

    ndarray_deinit_double(&arr1);
    ndarray_deinit_double(&arr2);
    ndarray_deinit_double(&arr3);
    return TEST_PASS;
}

int test_arithmetic() {
    size_t dims[] = {2, 2};
    size_t nelems = 4;
    size_t ndims = 2;

    // Data for array 1: {{1, 2}, {3, 4}}
    double d1[] = {1.0, 2.0, 3.0, 4.0};
    // Data for array 2: {{10, 20}, {30, 40}}
    double d2[] = {10.0, 20.0, 30.0, 40.0};

    NDArray_double arr1, arr2;
    ndarray_init_double(&arr1, d1, nelems, dims, ndims);
    ndarray_init_double(&arr2, d2, nelems, dims, ndims);

    // Temp storage for in-place modification
    double d1_copy[] = {1.0, 2.0, 3.0, 4.0};
    NDArray_double arr1_copy;
    ndarray_init_double(&arr1_copy, d1_copy, nelems, dims, ndims);

    // 1. In-place Add check (arr1_copy += arr2)
    // Expected: {11, 22, 33, 44}
    if (ndarray_add_in_place_double(&arr1_copy, &arr2) != NDARRAY_OK) return TEST_FAIL;
    if (d1_copy[0] != 11.0 || d1_copy[3] != 44.0) {
        return TEST_FAIL;
    }

    // 2. Out-of-place Multiply check (arr_out = arr1 * arr2)
    // Expected: {10, 40, 90, 160}
    double d_out[] = {0.0, 0.0, 0.0, 0.0};
    NDArray_double arr_out;
    ndarray_init_double(&arr_out, d_out, nelems, dims, ndims);

    if (ndarray_multiply_elemWise_double(&arr1, &arr2, &arr_out) != NDARRAY_OK) return TEST_FAIL;
    if (d_out[1] != 40.0 || d_out[2] != 90.0) {
        return TEST_FAIL;
    }

    // 3. Scalar Multiply check (arr1 *= 2.0)
    // Expected: {2, 4, 6, 8}
    ndarray_scalar_multiply_double(&arr1, 2.0);
    if (d1[0] != 2.0 || d1[3] != 8.0) {
        return TEST_FAIL;
    }

    // 4. Dimension Mismatch check (Subtract)
    double wrong_dims_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    size_t wrong_dims[] = {3, 2}; // Mismatch with 2x2
    NDArray_double arr_wrong;
    ndarray_init_double(&arr_wrong, wrong_dims_data, 6, wrong_dims, 2);

    if (ndarray_subtract_in_place_double(&arr1, &arr_wrong) != NDARRAY_DIM_MISMATCH) {
        ndarray_deinit_double(&arr_wrong);
        return TEST_FAIL;
    }

    ndarray_deinit_double(&arr1);
    ndarray_deinit_double(&arr2);
    ndarray_deinit_double(&arr1_copy);
    ndarray_deinit_double(&arr_out);
    ndarray_deinit_double(&arr_wrong);
    return TEST_PASS;
}

int test_slicing() {
    // 3x2x2 array
    // Layer 0: {{1, 2}, {3, 4}}
    // Layer 1: {{5, 6}, {7, 8}}
    // Layer 2: {{9, 10}, {11, 12}}
    double data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    size_t dims[] = {3, 2, 2};
    size_t nelems = 12;
    size_t ndims = 3;
    NDArray_double arr;
    ndarray_init_double(&arr, data, nelems, dims, ndims);

    double *slice_ptr;
    size_t slice_len;

    // 1. Slice along dim 1 (the middle dimension, size 2)
    // Fixed indices: (1, 1, _) -> should be {7.0, 8.0}
    size_t fixed_2[] = {1, 1, 0};
    size_t slice_fixed_dim_2 = 1; // Will fail if set to 2 since slicer doesn't work for the last dimension. As expected.
    // The logic: slice_fixed_dim is the dimension *to be* sliced.
    if (ndarray_get_slice_double(&arr, fixed_2, ndims, slice_fixed_dim_2, &slice_ptr, &slice_len) != NDARRAY_OK) {
        printf("Getting slice failed\n");
        printf("%d", ndarray_get_slice_double(&arr, fixed_2, ndims, slice_fixed_dim_2, &slice_ptr, &slice_len));
        return TEST_FAIL;
    }
    // Check slice length and content
    if (slice_len != 2) return TEST_FAIL;
    if (slice_ptr[0] != 7.0 || slice_ptr[1] != 8.0) {
        printf("Bad slice\n");
        return TEST_FAIL;
    }
    ndarray_deinit_double(&arr);
    return TEST_PASS;
}

int main() {
    printf("--- Running NDArray Unit Tests ---\n");

    print_test_result("Test Init/Deinit", test_init_deinit());
    print_test_result("Test Indexing (Flat, Get, Set)", test_indexing());
    print_test_result("Test Array Utilities (Fill, Match Dims)", test_array_utilities());
    print_test_result("Test Arithmetic (In-place & Out-of-place)", test_arithmetic());
    print_test_result("Test Slicing", test_slicing());

    printf("\n--- Test Summary ---\n");
    printf("Total Tests: %d\n", total_tests);
    printf("Passed: %d\n", passed_tests);
    printf("Failed: %d\n", total_tests - passed_tests);

    // Return 0 if all tests passed, 1 otherwise
    return total_tests == passed_tests ? 0 : 1;
}
*/