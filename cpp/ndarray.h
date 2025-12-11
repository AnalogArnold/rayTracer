// C++ equivalents
//#include <cstddef>
//#include <cstdio>
//#include <cstdlib>

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h> // size_t
#include <stdbool.h> // bool
#include <stdio.h>
#include <stdlib.h> // malloc, free

//////////////////////////////////////////////////////// USAGE //////////////////////////////////////////////////////
// Written for double, int, unsigned int, and long long.
// Unit tests at the very bottom.

// Syntax: function_name(variableype, &array1, &array2), etc.

// Function names and arguments:
// ndarray_init(type, *array, *elems, nelems, *dims, ndims)
// ndarray_deinit(type,*array)
// ndarray_get_flat_index(type,*array, *indices, indices_len, *out_index)
// ndarray_set_index(type,*array, *indices, indices_len, value)
// ndarray_get(type,*array, *indices, indices_len, *out_value)
// ndarray_fill(type,*array, value)
// ndarray_print(type,*array)
// ndarray_match_dimensions(type,*array1, *array2)
// ndarray_add_in_place(type,*array1, *array2)
// ndarray_subtract_in_place(type,*array1, *array2)
// ndarray_multiply_in_place(type,*array1, *array2)
// ndarray_divide_in_place(type,*array1, *array2)
// ndarray_scalar_multiply(type,*array, scalar) Nb4 scalar is the same type as the array elements
// ndarray_get_slice(type,*array, *fixed_indices, fixed_length, slice_fixed_dim, **out_pointer, *out_length)
// ndarray_add(type,*in_array1, *in_array2, *out_array)
// ndarray_subtract(type,*in_array1, *in_array2, *out_array)
// ndarray_multiply_elemWise(type,*in_array1, *in_array2, *out_array)
// ndarray_divide_elemWise(type,*in_array1, *in_array2, *out_array)

///////////////////////////////////////////////////// TYPE DEFS ///////////////////////////////////////////////////

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
    double *elems; // external storage, not owned by NDArray_f64. Need double explicitly in C
    size_t *dims; // Heap-owned copy of dimension size_ts
    size_t *strides; // Heap-owned strides
    size_t ndims; // Number of dimensions
    size_t nelems; // Total number of elements
} NDArray_f64;

typedef struct {
    int *elems; // external storage, not owned by NDArray_f64. Need double explicitly in C
    size_t *dims; // Heap-owned copy of dimension size_ts
    size_t *strides; // Heap-owned strides
    size_t ndims; // Number of dimensions
    size_t nelems; // Total number of elements
} NDArray_i32;

typedef struct {
    unsigned int *elems; // external storage, not owned by NDArray_f64. Need double explicitly in C
    size_t *dims; // Heap-owned copy of dimension size_ts
    size_t *strides; // Heap-owned strides
    size_t ndims; // Number of dimensions
    size_t nelems; // Total number of elements
} NDArray_u32;

typedef struct {
    long long *elems; // external storage, not owned by NDArray_f64. Need double explicitly in C
    size_t *dims; // Heap-owned copy of dimension size_ts
    size_t *strides; // Heap-owned strides
    size_t ndims; // Number of dimensions
    size_t nelems; // Total number of elements
} NDArray_i64;


////////////////////////////////////////////////////////// MACROS ////////////////////////////////////////////////////////
// Purpose: Save the user having to write out full function names, since each type needs its own separate thing.
// So for example, the syntax simplifies to:
// NDArray_f64 arr;  -->  NDArray(double);
// ndarray_init_f64(&arr, data, nelems, dims, ndims);  -->  ndarray_init(double, &arr, data, nelems, dims, ndims);
// ndarray_print_f64(&arr);  -->  ndarray_print(double, &arr);
// ndarray_deinit_f64(&arr);  -->  ndarray_deinit(double, &arr);

// Map standard C type names to type tags
#define ND_double f64
#define ND_i32 i32
#define ND_unsigned_i32 u32
#define ND_long_long i64

// Paste macro for name-mangling
// Ensure arguments are fully expanded before concantenating
#define ND_PASTE2(a,b) a##b // ## is the concatenation operator
#define ND_PASTE(a,b) ND_PASTE2(a,b)
// Given a functionName and a typeTag, build symbol 'ndarray_functionNameypeTag`
#define ND_FUNC(functionName, typeTag) ND_PASTE(ndarray_##functionName##_, typeTag) // expand to create a specific function name, e.g., ndarray_add_f64
// Map C type token to a tag
#define NDAG(T) ND_##T // Take a C-type token (e.g., int) and use the ND tag to retrieve the type tag (e.g., i32)

// Top-level macros to create a "generic" interface. Take a C type token and pick the right function
// Need to be updated for each additional function created...
#define NDArray(T) ND_PASTE(NDArray_, NDAG(T))
#define ndarray_init(T, array, data, nelems, dims, ndims) ND_FUNC(init, NDAG(T))(array, data, nelems, dims, ndims)
#define ndarray_deinit(T, array) ND_FUNC(deinit, NDAG(T))(array)
#define ndarray_get_flat_index(T, array, indices, indices_len, out_index) ND_FUNC(get_flat_index, NDAG(T))(array, indices, indices_len, out_index)
#define ndarray_set_index(T, array, indices, indices_len, value) ND_FUNC(set_index, NDAG(T))(array, indices, indices_len, value)
#define ndarray_get(T, array, indices, indices_len, out_value) ND_FUNC(get, NDAG(T))(array, indices, indices_len, out_value)
#define ndarray_fill(T, array, value) ND_FUNC(fill, NDAG(T))(array, value)
#define ndarray_print(T, array) ND_FUNC(print, NDAG(T))(array)
#define ndarray_match_dimensions(T, array1, array2) ND_FUNC(match_dimensions, NDAG(T))(array1, array2)
#define ndarray_add_in_place(T, array1, array2) ND_FUNC(add_in_place, NDAG(T))(array1, array2)
#define ndarray_subtract_in_place(T, array1, array2) ND_FUNC(subtract_in_place, NDAG(T))(array1, array2)
#define ndarray_multiply_in_place(T, array1, array2) ND_FUNC(multiply_in_place, NDAG(T))(array1, array2)
#define ndarray_divide_in_place(T, array1, array2) ND_FUNC(divide_in_place, NDA
#define ndarray_scalar_multiply(T, array, scalar) ND_FUNC(scalar_multiply, NDAG(T))(array, scalar)
#define ndarray_get_slice(T, array, fixed_indices, fixed_length, slice_fixed_dim, out_pointer, out_length) ND_FUNC(get_slice, NDAG(T))(array, fixed_indices, fixed_length, slice_fixed_dim, out_pointer, out_length)
#define ndarray_add(T, array1, array2, out_array) ND_FUNC(add, NDAG(T))(array1, array2, out_array)
#define ndarray_subtract(T, array1, array2, out_array) ND_FUNC(subtract, NDAG(T))(array1, array2, out_array)
#define ndarray_multiply_elemWise(T, array1, array2, out_array) ND_FUNC(multiply_elemWise, NDAG(T))(array1, array2, out_array)
#define ndarray_divide_elemWise(T, array1, array2, out_array) ND_FUNC(divide_elemWise, NDAG(T))(array1, array2, out_array)

//////////////////////////////////////////////////////// DOUBLE //////////////////////////////////////////////////////

// Initialization
NDArrayError
ndarray_init_f64(NDArray_f64 *arr,
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
ndarray_deinit_f64(NDArray_f64 *arr) {
    // Free dims and strides (elems is owned by the caller, so no need to)
    free(arr->dims);
    free(arr->strides);
    arr->dims = NULL;
    arr->strides = NULL;
}

// Index computation
NDArrayError
ndarray_get_flat_index_f64(const NDArray_f64 *arr,
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
ndarray_set_index_f64(NDArray_f64 *arr,
    const size_t *indices,
    size_t indices_len,
    double value) {
    // Set value at an index
    size_t flat;
    NDArrayError err = ndarray_get_flat_index_f64(arr, indices, indices_len, &flat);
    if (err != NDARRAY_OK) return err;
    arr->elems[flat] = value;
    return NDARRAY_OK;
}

NDArrayError
ndarray_get_f64(const NDArray_f64 *arr,
    const size_t *indices,
    size_t indices_len,
    double *out_value) {
    // Get value at an index
    size_t flat;
    NDArrayError err = ndarray_get_flat_index_f64(arr, indices, indices_len, &flat);
    if (err != NDARRAY_OK) return err;
    *out_value = arr->elems[flat];
    return NDARRAY_OK;
}

void ndarray_fill_f64(NDArray_f64 *arr,
    double value) {
    // Fill the array with a given value
    for (size_t i = 0; i < arr->nelems; ++i) {
        arr->elems[i] = value;
    }
}

void ndarray_print_f64(const NDArray_f64 *arr) {
    // Print the array contents
    for (size_t i = 0; i < arr->nelems; ++i) {
        printf("%f ", arr->elems[i]);
    }
    printf("\n");
}

bool ndarray_match_dimensions_f64(const NDArray_f64 *array1,
    const NDArray_f64 *array2) {
    // Check if two arrays have the same dimensions
    if (array1->ndims != array2->ndims) return false; // Different number of dimensions
    // Compare size_t for  each constituing dimension
    for (size_t i = 0; i < array1->ndims; ++i) {
        if (array1->dims[i] != array2->dims[i]) {
            return false;
        }
    }
    return true;
}
NDArrayError
ndarray_add_in_place_f64(NDArray_f64 *self, const NDArray_f64 *other) {
    // Add two arrays together in-place
    if (!ndarray_match_dimensions_f64(self, other)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < self->nelems; ++i) {
        self->elems[i] += other->elems[i];
    }
    return NDARRAY_OK;
}

NDArrayError
ndarray_subtract_in_place_f64(NDArray_f64 *self, const NDArray_f64 *other) {
    // Subtract two arrays together in-place
    if (!ndarray_match_dimensions_f64(self, other)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < self->nelems; ++i) {
        self->elems[i] -= other->elems[i];
    }
    return NDARRAY_OK;
}

NDArrayError
ndarray_multiply_in_place_f64(NDArray_f64 *self, const NDArray_f64 *other) {
    // Multiply two arrays together in-place (element-wise)
    if (!ndarray_match_dimensions_f64(self, other)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < self->nelems; ++i) {
        self->elems[i] *= other->elems[i];
    }
    return NDARRAY_OK;
}

NDArrayError
ndarray_divide_in_place_f64(NDArray_f64 *self, const NDArray_f64 *other) {
    // Divide two arrays together in-place (element-wise)
    if (!ndarray_match_dimensions_f64(self, other)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < self->nelems; ++i) {
        self->elems[i] /= other->elems[i];
    }
    return NDARRAY_OK;
}

void
ndarray_scalar_multiply_f64(NDArray_f64 *self,
    double scalar) {
    // Multiply an array by a scalar
    for (size_t i = 0; i < self->nelems; ++i) {
        self->elems[i] *= scalar;
    }
}

NDArrayError
ndarray_get_slice_f64(const NDArray_f64 *arr,
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
    NDArrayError err_start = ndarray_get_flat_index_f64(arr, fixed_indices, fixed_length, &start_index);
    if (err_start != NDARRAY_OK) return err_start; // Error getting start index

    size_t slice_length = arr->dims[slice_fixed_dim];
    *out_ptr = arr->elems + start_index;
    *out_length = slice_length;
    return NDARRAY_OK;
}

NDArrayError
ndarray_add_f64(const NDArray_f64 *array1, const NDArray_f64 *array2, NDArray_f64 *out_array) {
    // Add two arrays together and return a third array
    if (!ndarray_match_dimensions_f64(array1, array2) || !ndarray_match_dimensions_f64(array1, out_array)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    //ndarray_fill(out_array, 0);
    // Could also fill out_array with 0s and run ndarray_add_in_place twice with array1 and array2, but then we have to basically go into 3 for loops, so not the best option
    for (size_t i = 0; i < array1->nelems; ++i) {
        out_array->elems[i] = array1->elems[i] + array2->elems[i];
    }
    return NDARRAY_OK;
}
NDArrayError
ndarray_subtract_f64(const NDArray_f64 *array1, const NDArray_f64 *array2, NDArray_f64 *out_array) {
    // Subtract two arrays together and return a third array
    if (!ndarray_match_dimensions_f64(array1, array2) || !ndarray_match_dimensions_f64(array1, out_array)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < array1->nelems; ++i) {
        out_array->elems[i] = array1->elems[i] - array2->elems[i];
    }
    return NDARRAY_OK;
}

NDArrayError
ndarray_multiply_elemWise_f64(const NDArray_f64 *array1, const NDArray_f64 *array2, NDArray_f64 *out_array) {
    // Multiply two arrays together and return a third array
    if (!ndarray_match_dimensions_f64(array1, array2) || !ndarray_match_dimensions_f64(array1, out_array)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < array1->nelems; ++i) {
        out_array->elems[i] = array1->elems[i] * array2->elems[i];
    }
    return NDARRAY_OK;
}

NDArrayError
ndarray_divide_elemWise_f64(const NDArray_f64 *array1, const NDArray_f64 *array2, NDArray_f64 *out_array) {
    // Divide two arrays together and return a third array
    if (!ndarray_match_dimensions_f64(array1, array2) || !ndarray_match_dimensions_f64(array1, out_array)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < array1->nelems; ++i) {
        out_array->elems[i] = array1->elems[i] / array2->elems[i];
    }
    return NDARRAY_OK;
}

///////////////////////////////////////////////////////// INT ///////////////////////////////////////////////////////

// Initialization
NDArrayError
ndarray_init_i32(NDArray_i32 *arr,
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
ndarray_deinit_i32(NDArray_i32 *arr) {
    // Free dims and strides (elems is owned by the caller, so no need to)
    free(arr->dims);
    free(arr->strides);
    arr->dims = NULL;
    arr->strides = NULL;
}

// Index computation
NDArrayError
ndarray_get_flat_index_i32(const NDArray_i32 *arr,
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
ndarray_set_index_i32(NDArray_i32 *arr,
    const size_t *indices,
    size_t indices_len,
    int value) {
    // Set value at an index
    size_t flat;
    NDArrayError err = ndarray_get_flat_index_i32(arr, indices, indices_len, &flat);
    if (err != NDARRAY_OK) return err;
    arr->elems[flat] = value;
    return NDARRAY_OK;
}

NDArrayError
ndarray_get_i32(const NDArray_i32 *arr,
    const size_t *indices,
    size_t indices_len,
    int *out_value) {
    // Get value at an index
    size_t flat;
    NDArrayError err = ndarray_get_flat_index_i32(arr, indices, indices_len, &flat);
    if (err != NDARRAY_OK) return err;
    *out_value = arr->elems[flat];
    return NDARRAY_OK;
}

void ndarray_fill_i32(NDArray_i32 *arr,
    int value) {
    // Fill the array with a given value
    for (size_t i = 0; i < arr->nelems; ++i) {
        arr->elems[i] = value;
    }
}

void ndarray_print_i32(const NDArray_i32 *arr) {
    // Print the array contents
    for (size_t i = 0; i < arr->nelems; ++i) {
        printf("%i ", arr->elems[i]);
    }
    printf("\n");
}

bool ndarray_match_dimensions_i32(const NDArray_i32 *array1,
    const NDArray_i32 *array2) {
    // Check if two arrays have the same dimensions
    if (array1->ndims != array2->ndims) return false; // Different number of dimensions
    // Compare size_t for  each constituing dimension
    for (size_t i = 0; i < array1->ndims; ++i) {
        if (array1->dims[i] != array2->dims[i]) {
            return false;
        }
    }
    return true;
}
NDArrayError
ndarray_add_in_place_i32(NDArray_i32 *self, const NDArray_i32 *other) {
    // Add two arrays together in-place
    if (!ndarray_match_dimensions_i32(self, other)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < self->nelems; ++i) {
        self->elems[i] += other->elems[i];
    }
    return NDARRAY_OK;
}

NDArrayError
ndarray_subtract_in_place_i32(NDArray_i32 *self, const NDArray_i32 *other) {
    // Subtract two arrays together in-place
    if (!ndarray_match_dimensions_i32(self, other)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < self->nelems; ++i) {
        self->elems[i] -= other->elems[i];
    }
    return NDARRAY_OK;
}

NDArrayError
ndarray_multiply_in_place_i32(NDArray_i32 *self, const NDArray_i32 *other) {
    // Multiply two arrays together in-place (element-wise)
    if (!ndarray_match_dimensions_i32(self, other)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < self->nelems; ++i) {
        self->elems[i] *= other->elems[i];
    }
    return NDARRAY_OK;
}

NDArrayError
ndarray_divide_in_place_i32(NDArray_i32 *self, const NDArray_i32 *other) {
    // Divide two arrays together in-place (element-wise)
    if (!ndarray_match_dimensions_i32(self, other)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < self->nelems; ++i) {
        self->elems[i] /= other->elems[i];
    }
    return NDARRAY_OK;
}

void
ndarray_scalar_multiply_i32(NDArray_i32 *self,
    int scalar) {
    // Multiply an array by a scalar
    for (size_t i = 0; i < self->nelems; ++i) {
        self->elems[i] *= scalar;
    }
}

NDArrayError
ndarray_get_slice_i32(const NDArray_i32 *arr,
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
    NDArrayError err_start = ndarray_get_flat_index_i32(arr, fixed_indices, fixed_length, &start_index);
    if (err_start != NDARRAY_OK) return err_start; // Error getting start index

    size_t slice_length = arr->dims[slice_fixed_dim];
    *out_ptr = arr->elems + start_index;
    *out_length = slice_length;
    return NDARRAY_OK;
}

NDArrayError
ndarray_add_i32(const NDArray_i32 *array1, const NDArray_i32 *array2, NDArray_i32 *out_array) {
    // Add two arrays together and return a third array
    if (!ndarray_match_dimensions_i32(array1, array2) || !ndarray_match_dimensions_i32(array1, out_array)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    //ndarray_fill(out_array, 0);
    // Could also fill out_array with 0s and run ndarray_add_in_place twice with array1 and array2, but then we have to basically go into 3 for loops, so not the best option
    for (size_t i = 0; i < array1->nelems; ++i) {
        out_array->elems[i] = array1->elems[i] + array2->elems[i];
    }
    return NDARRAY_OK;
}
NDArrayError
ndarray_subtract_i32(const NDArray_i32 *array1, const NDArray_i32 *array2, NDArray_i32 *out_array) {
    // Subtract two arrays together and return a third array
    if (!ndarray_match_dimensions_i32(array1, array2) || !ndarray_match_dimensions_i32(array1, out_array)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < array1->nelems; ++i) {
        out_array->elems[i] = array1->elems[i] - array2->elems[i];
    }
    return NDARRAY_OK;
}

NDArrayError
ndarray_multiply_elemWise_i32(const NDArray_i32 *array1, const NDArray_i32 *array2, NDArray_i32 *out_array) {
    // Multiply two arrays together and return a third array
    if (!ndarray_match_dimensions_i32(array1, array2) || !ndarray_match_dimensions_i32(array1, out_array)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < array1->nelems; ++i) {
        out_array->elems[i] = array1->elems[i] * array2->elems[i];
    }
    return NDARRAY_OK;
}

NDArrayError
ndarray_divide_elemWise_i32(const NDArray_i32 *array1, const NDArray_i32 *array2, NDArray_i32 *out_array) {
    // Divide two arrays together and return a third array
    if (!ndarray_match_dimensions_i32(array1, array2) || !ndarray_match_dimensions_i32(array1, out_array)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < array1->nelems; ++i) {
        out_array->elems[i] = array1->elems[i] / array2->elems[i];
    }
    return NDARRAY_OK;
}

////////////////////////////////////////////////////// UNSIGNED INT ////////////////////////////////////////////////////

// Initialization
NDArrayError
ndarray_init_u32(NDArray_u32 *arr,
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
ndarray_deinit_u32(NDArray_u32 *arr) {
    // Free dims and strides (elems is owned by the caller, so no need to)
    free(arr->dims);
    free(arr->strides);
    arr->dims = NULL;
    arr->strides = NULL;
}

// Index computation
NDArrayError
ndarray_get_flat_index_u32(const NDArray_u32 *arr,
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
ndarray_set_index_u32(NDArray_u32 *arr,
    const size_t *indices,
    size_t indices_len,
    unsigned int value) {
    // Set value at an index
    size_t flat;
    NDArrayError err = ndarray_get_flat_index_u32(arr, indices, indices_len, &flat);
    if (err != NDARRAY_OK) return err;
    arr->elems[flat] = value;
    return NDARRAY_OK;
}

NDArrayError
ndarray_get_u32(const NDArray_u32 *arr,
    const size_t *indices,
    size_t indices_len,
    unsigned int *out_value) {
    // Get value at an index
    size_t flat;
    NDArrayError err = ndarray_get_flat_index_u32(arr, indices, indices_len, &flat);
    if (err != NDARRAY_OK) return err;
    *out_value = arr->elems[flat];
    return NDARRAY_OK;
}

void ndarray_fill_u32(NDArray_u32 *arr,
    unsigned int value) {
    // Fill the array with a given value
    for (size_t i = 0; i < arr->nelems; ++i) {
        arr->elems[i] = value;
    }
}

void ndarray_print_u32(const NDArray_u32 *arr) {
    // Print the array contents
    for (size_t i = 0; i < arr->nelems; ++i) {
        printf("%u ", arr->elems[i]);
    }
    printf("\n");
}

bool ndarray_match_dimensions_u32(const NDArray_u32 *array1,
    const NDArray_u32 *array2) {
    // Check if two arrays have the same dimensions
    if (array1->ndims != array2->ndims) return false; // Different number of dimensions
    // Compare size_t for  each constituing dimension
    for (size_t i = 0; i < array1->ndims; ++i) {
        if (array1->dims[i] != array2->dims[i]) {
            return false;
        }
    }
    return true;
}
NDArrayError
ndarray_add_in_place_u32(NDArray_u32 *self, const NDArray_u32 *other) {
    // Add two arrays together in-place
    if (!ndarray_match_dimensions_u32(self, other)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < self->nelems; ++i) {
        self->elems[i] += other->elems[i];
    }
    return NDARRAY_OK;
}

NDArrayError
ndarray_subtract_in_place_u32(NDArray_u32 *self, const NDArray_u32 *other) {
    // Subtract two arrays together in-place
    if (!ndarray_match_dimensions_u32(self, other)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < self->nelems; ++i) {
        self->elems[i] -= other->elems[i];
    }
    return NDARRAY_OK;
}

NDArrayError
ndarray_multiply_in_place_u32(NDArray_u32 *self, const NDArray_u32 *other) {
    // Multiply two arrays together in-place (element-wise)
    if (!ndarray_match_dimensions_u32(self, other)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < self->nelems; ++i) {
        self->elems[i] *= other->elems[i];
    }
    return NDARRAY_OK;
}

NDArrayError
ndarray_divide_in_place_u32(NDArray_u32 *self, const NDArray_u32 *other) {
    // Divide two arrays together in-place (element-wise)
    if (!ndarray_match_dimensions_u32(self, other)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < self->nelems; ++i) {
        self->elems[i] /= other->elems[i];
    }
    return NDARRAY_OK;
}

void
ndarray_scalar_multiply_u32(NDArray_u32 *self,
    unsigned int scalar) {
    // Multiply an array by a scalar
    for (size_t i = 0; i < self->nelems; ++i) {
        self->elems[i] *= scalar;
    }
}

NDArrayError
ndarray_get_slice_u32(const NDArray_u32 *arr,
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
    NDArrayError err_start = ndarray_get_flat_index_u32(arr, fixed_indices, fixed_length, &start_index);
    if (err_start != NDARRAY_OK) return err_start; // Error getting start index

    size_t slice_length = arr->dims[slice_fixed_dim];
    *out_ptr = arr->elems + start_index;
    *out_length = slice_length;
    return NDARRAY_OK;
}

NDArrayError
ndarray_add_u32(const NDArray_u32 *array1, const NDArray_u32 *array2, NDArray_u32 *out_array) {
    // Add two arrays together and return a third array
    if (!ndarray_match_dimensions_u32(array1, array2) || !ndarray_match_dimensions_u32(array1, out_array)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    //ndarray_fill(out_array, 0);
    // Could also fill out_array with 0s and run ndarray_add_in_place twice with array1 and array2, but then we have to basically go into 3 for loops, so not the best option
    for (size_t i = 0; i < array1->nelems; ++i) {
        out_array->elems[i] = array1->elems[i] + array2->elems[i];
    }
    return NDARRAY_OK;
}
NDArrayError
ndarray_subtract_u32(const NDArray_u32 *array1, const NDArray_u32 *array2, NDArray_u32 *out_array) {
    // Subtract two arrays together and return a third array
    if (!ndarray_match_dimensions_u32(array1, array2) || !ndarray_match_dimensions_u32(array1, out_array)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < array1->nelems; ++i) {
        out_array->elems[i] = array1->elems[i] - array2->elems[i];
    }
    return NDARRAY_OK;
}

NDArrayError
ndarray_multiply_elemWise_u32(const NDArray_u32 *array1, const NDArray_u32 *array2, NDArray_u32 *out_array) {
    // Multiply two arrays together and return a third array
    if (!ndarray_match_dimensions_u32(array1, array2) || !ndarray_match_dimensions_u32(array1, out_array)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < array1->nelems; ++i) {
        out_array->elems[i] = array1->elems[i] * array2->elems[i];
    }
    return NDARRAY_OK;
}

NDArrayError
ndarray_divide_elemWise_u32(const NDArray_u32 *array1, const NDArray_u32 *array2, NDArray_u32 *out_array) {
    // Divide two arrays together and return a third array
    if (!ndarray_match_dimensions_u32(array1, array2) || !ndarray_match_dimensions_u32(array1, out_array)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < array1->nelems; ++i) {
        out_array->elems[i] = array1->elems[i] / array2->elems[i];
    }
    return NDARRAY_OK;
}

/////////////////////////////////////////////////////// LONG LONG /////////////////////////////////////////////////////

// Initialization
NDArrayError
ndarray_init_i64(NDArray_i64 *arr,
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
ndarray_deinit_i64(NDArray_i64 *arr) {
    // Free dims and strides (elems is owned by the caller, so no need to)
    free(arr->dims);
    free(arr->strides);
    arr->dims = NULL;
    arr->strides = NULL;
}

// Index computation
NDArrayError
ndarray_get_flat_index_i64(const NDArray_i64 *arr,
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
ndarray_set_index_i64(NDArray_i64 *arr,
    const size_t *indices,
    size_t indices_len,
    long long value) {
    // Set value at an index
    size_t flat;
    NDArrayError err = ndarray_get_flat_index_i64(arr, indices, indices_len, &flat);
    if (err != NDARRAY_OK) return err;
    arr->elems[flat] = value;
    return NDARRAY_OK;
}

NDArrayError
ndarray_get_i64(const NDArray_i64 *arr,
    const size_t *indices,
    size_t indices_len,
    long long *out_value) {
    // Get value at an index
    size_t flat;
    NDArrayError err = ndarray_get_flat_index_i64(arr, indices, indices_len, &flat);
    if (err != NDARRAY_OK) return err;
    *out_value = arr->elems[flat];
    return NDARRAY_OK;
}

void ndarray_fill_i64(NDArray_i64 *arr,
    long long value) {
    // Fill the array with a given value
    for (size_t i = 0; i < arr->nelems; ++i) {
        arr->elems[i] = value;
    }
}

void ndarray_print_i64(const NDArray_i64 *arr) {
    // Print the array contents
    for (size_t i = 0; i < arr->nelems; ++i) {
        printf("%lld ", arr->elems[i]);
    }
    printf("\n");
}

bool ndarray_match_dimensions_i64(const NDArray_i64 *array1,
    const NDArray_i64 *array2) {
    // Check if two arrays have the same dimensions
    if (array1->ndims != array2->ndims) return false; // Different number of dimensions
    // Compare size_t for  each constituing dimension
    for (size_t i = 0; i < array1->ndims; ++i) {
        if (array1->dims[i] != array2->dims[i]) {
            return false;
        }
    }
    return true;
}
NDArrayError
ndarray_add_in_place_i64(NDArray_i64 *self, const NDArray_i64 *other) {
    // Add two arrays together in-place
    if (!ndarray_match_dimensions_i64(self, other)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < self->nelems; ++i) {
        self->elems[i] += other->elems[i];
    }
    return NDARRAY_OK;
}

NDArrayError
ndarray_subtract_in_place_i64(NDArray_i64 *self, const NDArray_i64 *other) {
    // Subtract two arrays together in-place
    if (!ndarray_match_dimensions_i64(self, other)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < self->nelems; ++i) {
        self->elems[i] -= other->elems[i];
    }
    return NDARRAY_OK;
}

NDArrayError
ndarray_multiply_in_place_i64(NDArray_i64 *self, const NDArray_i64 *other) {
    // Multiply two arrays together in-place (element-wise)
    if (!ndarray_match_dimensions_i64(self, other)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < self->nelems; ++i) {
        self->elems[i] *= other->elems[i];
    }
    return NDARRAY_OK;
}

NDArrayError
ndarray_divide_in_place_i64(NDArray_i64 *self, const NDArray_i64 *other) {
    // Divide two arrays together in-place (element-wise)
    if (!ndarray_match_dimensions_i64(self, other)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < self->nelems; ++i) {
        self->elems[i] /= other->elems[i];
    }
    return NDARRAY_OK;
}

void
ndarray_scalar_multiply_i64(NDArray_i64 *self,
    long long scalar) {
    // Multiply an array by a scalar
    for (size_t i = 0; i < self->nelems; ++i) {
        self->elems[i] *= scalar;
    }
}

NDArrayError
ndarray_get_slice_i64(const NDArray_i64 *arr,
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
    NDArrayError err_start = ndarray_get_flat_index_i64(arr, fixed_indices, fixed_length, &start_index);
    if (err_start != NDARRAY_OK) return err_start; // Error getting start index

    size_t slice_length = arr->dims[slice_fixed_dim];
    *out_ptr = arr->elems + start_index;
    *out_length = slice_length;
    return NDARRAY_OK;
}

NDArrayError
ndarray_add_i64(const NDArray_i64 *array1, const NDArray_i64 *array2, NDArray_i64 *out_array) {
    // Add two arrays together and return a third array
    if (!ndarray_match_dimensions_i64(array1, array2) || !ndarray_match_dimensions_i64(array1, out_array)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    //ndarray_fill(out_array, 0);
    // Could also fill out_array with 0s and run ndarray_add_in_place twice with array1 and array2, but then we have to basically go into 3 for loops, so not the best option
    for (size_t i = 0; i < array1->nelems; ++i) {
        out_array->elems[i] = array1->elems[i] + array2->elems[i];
    }
    return NDARRAY_OK;
}
NDArrayError
ndarray_subtract_i64(const NDArray_i64 *array1, const NDArray_i64 *array2, NDArray_i64 *out_array) {
    // Subtract two arrays together and return a third array
    if (!ndarray_match_dimensions_i64(array1, array2) || !ndarray_match_dimensions_i64(array1, out_array)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < array1->nelems; ++i) {
        out_array->elems[i] = array1->elems[i] - array2->elems[i];
    }
    return NDARRAY_OK;
}

NDArrayError
ndarray_multiply_elemWise_i64(const NDArray_i64 *array1, const NDArray_i64 *array2, NDArray_i64 *out_array) {
    // Multiply two arrays together and return a third array
    if (!ndarray_match_dimensions_i64(array1, array2) || !ndarray_match_dimensions_i64(array1, out_array)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
    for (size_t i = 0; i < array1->nelems; ++i) {
        out_array->elems[i] = array1->elems[i] * array2->elems[i];
    }
    return NDARRAY_OK;
}

NDArrayError
ndarray_divide_elemWise_i64(const NDArray_i64 *array1, const NDArray_i64 *array2, NDArray_i64 *out_array) {
    // Divide two arrays together and return a third array
    if (!ndarray_match_dimensions_i64(array1, array2) || !ndarray_match_dimensions_i64(array1, out_array)) return NDARRAY_DIM_MISMATCH; // Dimensions do not match, can't add
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

int totalests = 0;
int passedests = 0;

void printest_result(const char* name, int result) {
    totalests++;
    if (result == TEST_PASS) {
        passedests++;
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
    NDArray_f64 arr;

    if (ndarray_init_f64(&arr, data, nelems, dims, ndims) != NDARRAY_OK) {
        return TEST_FAIL;
    }
    // Check core properties
    if (arr.ndims != ndims || arr.nelems != nelems) {
        ndarray_deinit_f64(&arr);
        return TEST_FAIL;
    }
    // Check calculated strides (C-style/row-major: stride[0]=3, stride[1]=1)
    if (arr.strides[0] != 3 || arr.strides[1] != 1) {
        ndarray_deinit_f64(&arr);
        return TEST_FAIL;
    }

    // 2. Invalid initialization (nelems mismatch)
    size_t wrong_nelems = 5;
    NDArray_f64 arr_err;
    if (ndarray_init_f64(&arr_err, data, wrong_nelems, dims, ndims) != NDARRAY_ELEMS_WRONG_LEN_FOR_DIMS) {
        ndarray_deinit_f64(&arr); // Deinit the good one
        return TEST_FAIL;
    }

    // 3. Deinitialization (manual check for no crash)
    ndarray_deinit_f64(&arr);

    return TEST_PASS;
}

int test_indexing() {
    double data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    size_t dims[] = {2, 2, 3}; // A 2x2x3 array
    size_t nelems = 12;
    size_t ndims = 3;
    NDArray_f64 arr;
    ndarray_init_f64(&arr, data, nelems, dims, ndims);

    // 1. Flat Index check: Index (1, 0, 2)
    // Formula: 1*stride[0] + 0*stride[1] + 2*stride[2]
    // Strides should be (6, 3, 1)
    // Flat index: 1*6 + 0*3 + 2*1 = 8
    size_t indices_102[] = {1, 0, 2};
    size_t flat_index;
    if (ndarray_get_flat_index_f64(&arr, indices_102, ndims, &flat_index) != NDARRAY_OK || flat_index != 8) {
        ndarray_deinit_f64(&arr);
        return TEST_FAIL;
    }

    // 2. Set/Get check: Set (0, 1, 1) to 99.9
    // Index (0, 1, 1): 0*6 + 1*3 + 1*1 = 4. Original value is 5.0
    size_t indices_011[] = {0, 1, 1};
    double new_value = 99.9;
    if (ndarray_set_index_f64(&arr, indices_011, ndims, new_value) != NDARRAY_OK) {
        ndarray_deinit_f64(&arr);
        return TEST_FAIL;
    }

    double retrieved_value;
    if (ndarray_get_f64(&arr, indices_011, ndims, &retrieved_value) != NDARRAY_OK || retrieved_value != new_value) {
        ndarray_deinit_f64(&arr);
        return TEST_FAIL;
    }

    // 3. Out-of-bounds check (index 2 for dim 0, which is size_t 2)
    size_t indices_oob[] = {2, 0, 0};
    if (ndarray_get_flat_index_f64(&arr, indices_oob, ndims, &flat_index) != NDARRAY_INDEX_OUT_OF_BOUNDS) {
        ndarray_deinit_f64(&arr);
        return TEST_FAIL;
    }

    ndarray_deinit_f64(&arr);
    return TEST_PASS;
}

int test_array_utilities() {
    double data1[] = {0.0, 0.0, 0.0, 0.0};
    size_t dims1[] = {2, 2};
    NDArray_f64 arr1;
    ndarray_init_f64(&arr1, data1, 4, dims1, 2);

    // 1. Fill check
    ndarray_fill_f64(&arr1, 5.5);
    for (size_t i = 0; i < arr1.nelems; ++i) {
        if (arr1.elems[i] != 5.5) {
            ndarray_deinit_f64(&arr1);
            return TEST_FAIL;
        }
    }

    // 2. Match dimensions check (same dims)
    double data2[] = {1.0, 2.0, 3.0, 4.0};
    size_t dims2[] = {2, 2};
    NDArray_f64 arr2;
    ndarray_init_f64(&arr2, data2, 4, dims2, 2);

    if (!ndarray_match_dimensions_f64(&arr1, &arr2)) {
        ndarray_deinit_f64(&arr1);
        ndarray_deinit_f64(&arr2);
        return TEST_FAIL;
    }

    // 3. Match dimensions check (different dims)
    double data3[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    size_t dims3[] = {3, 2}; // 3x2 vs 2x2
    NDArray_f64 arr3;
    ndarray_init_f64(&arr3, data3, 6, dims3, 2);

    if (ndarray_match_dimensions_f64(&arr1, &arr3)) {
        ndarray_deinit_f64(&arr1);
        ndarray_deinit_f64(&arr2);
        ndarray_deinit_f64(&arr3);
        return TEST_FAIL;
    }

    ndarray_deinit_f64(&arr1);
    ndarray_deinit_f64(&arr2);
    ndarray_deinit_f64(&arr3);
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

    NDArray_f64 arr1, arr2;
    ndarray_init_f64(&arr1, d1, nelems, dims, ndims);
    ndarray_init_f64(&arr2, d2, nelems, dims, ndims);

    // Temp storage for in-place modification
    double d1_copy[] = {1.0, 2.0, 3.0, 4.0};
    NDArray_f64 arr1_copy;
    ndarray_init_f64(&arr1_copy, d1_copy, nelems, dims, ndims);

    // 1. In-place Add check (arr1_copy += arr2)
    // Expected: {11, 22, 33, 44}
    if (ndarray_add_in_place_f64(&arr1_copy, &arr2) != NDARRAY_OK) return TEST_FAIL;
    if (d1_copy[0] != 11.0 || d1_copy[3] != 44.0) {
        return TEST_FAIL;
    }

    // 2. Out-of-place Multiply check (arr_out = arr1 * arr2)
    // Expected: {10, 40, 90, 160}
    double d_out[] = {0.0, 0.0, 0.0, 0.0};
    NDArray_f64 arr_out;
    ndarray_init_f64(&arr_out, d_out, nelems, dims, ndims);

    if (ndarray_multiply_elemWise_f64(&arr1, &arr2, &arr_out) != NDARRAY_OK) return TEST_FAIL;
    if (d_out[1] != 40.0 || d_out[2] != 90.0) {
        return TEST_FAIL;
    }

    // 3. Scalar Multiply check (arr1 *= 2.0)
    // Expected: {2, 4, 6, 8}
    ndarray_scalar_multiply_f64(&arr1, 2.0);
    if (d1[0] != 2.0 || d1[3] != 8.0) {
        return TEST_FAIL;
    }

    // 4. Dimension Mismatch check (Subtract)
    double wrong_dims_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    size_t wrong_dims[] = {3, 2}; // Mismatch with 2x2
    NDArray_f64 arr_wrong;
    ndarray_init_f64(&arr_wrong, wrong_dims_data, 6, wrong_dims, 2);

    if (ndarray_subtract_in_place_f64(&arr1, &arr_wrong) != NDARRAY_DIM_MISMATCH) {
        ndarray_deinit_f64(&arr_wrong);
        return TEST_FAIL;
    }

    ndarray_deinit_f64(&arr1);
    ndarray_deinit_f64(&arr2);
    ndarray_deinit_f64(&arr1_copy);
    ndarray_deinit_f64(&arr_out);
    ndarray_deinit_f64(&arr_wrong);
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
    NDArray_f64 arr;
    ndarray_init_f64(&arr, data, nelems, dims, ndims);

    double *slice_ptr;
    size_t slice_len;

    // 1. Slice along dim 1 (the middle dimension, size_t 2)
    // Fixed indices: (1, 1, _) -> should be {7.0, 8.0}
    size_t fixed_2[] = {1, 1, 0};
    size_t slice_fixed_dim_2 = 1; // Will fail if set to 2 since slicer doesn't work for the last dimension. As expected.
    // The logic: slice_fixed_dim is the dimension *to be* sliced.
    if (ndarray_get_slice_f64(&arr, fixed_2, ndims, slice_fixed_dim_2, &slice_ptr, &slice_len) != NDARRAY_OK) {
        printf("Getting slice failed\n");
        printf("%d", ndarray_get_slice_f64(&arr, fixed_2, ndims, slice_fixed_dim_2, &slice_ptr, &slice_len));
        return TEST_FAIL;
    }
    // Check slice length and content
    if (slice_len != 2) return TEST_FAIL;
    if (slice_ptr[0] != 7.0 || slice_ptr[1] != 8.0) {
        printf("Bad slice\n");
        return TEST_FAIL;
    }
    ndarray_deinit_f64(&arr);
    return TEST_PASS;
}

int test_macros() {
    // Test if C macros for generics work
    size_t dims[] = {2, 2};
    size_t nelems = 4;
    double data[] = {0.0, 1.0, 0.0, 4.0};
    size_t ndims = 2;
    NDArray_f64 arr;
    //ndarray_init_f64(&arr, data, nelems, dims, ndims);
    if (ndarray_init(double, &arr, data, nelems, dims, ndims) == NDARRAY_OK) {
        ndarray_print(double, &arr);
        size_t indices_011[] = {0, 1};
        double new_value = 99.9;
        ndarray_set_index(double, &arr, indices_011, ndims, new_value);
        // Check if we successfully changed the values and if we can retrieve them
        if (ndarray_get(double, &arr, indices_011, ndims, &new_value) != NDARRAY_OK || new_value != 99.9) {
            ndarray_deinit_f64(&arr); // if fail, assume that this macro might not work either and use function explicitly
            return TEST_FAIL;
        }

        ndarray_deinit(double, &arr);
        return TEST_PASS;
    } else {
        printf("error %d", ndarray_init(double, &arr, data, nelems, dims, ndims));
        ndarray_deinit_f64(&arr);
        return TEST_FAIL;
    }
}

int main() {
    printf("--- Running NDArray Unit Tests ---\n");

    printest_result("Test init/deinit", test_init_deinit());
    printest_result("Test indexing (flat, get, set)", test_indexing());
    printest_result("Test array utilities (fill, match dims)", test_array_utilities());
    printest_result("Test arithmetic (in-place & out-of-place)", test_arithmetic());
    printest_result("Test slicing", test_slicing());
    printest_result("Test macros", test_macros());

    printf("\n--- Test summary ---\n");
    printf("Total tests: %d\n", totalests);
    printf("Passed: %d\n", passedests);
    printf("Failed: %d\n", totalests - passedests);

    // Return 0 if all tests passed, 1 otherwise
    return totalests == passedests ? 0 : 1;
}
*/

#ifdef __cplusplus
}
#endif