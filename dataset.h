/**
 * @file dataset.h
 * @brief Header-only column-oriented dataset library for ML
 *
 * A lightweight library for converting row-oriented CSV data into column-oriented
 * data structures optimized for machine learning workflows.
 *
 * Features:
 * - Automatic numeric detection for features
 * - String label encoding with value mapping
 * - One-vs-All binary label expansion
 * - Support for 0/1 and -1/+1 binary labels (AdaBoost style)
 *
 * Data Layout:
 * @code
 * Row-oriented CSV:
 *   sepal_length,sepal_width,species
 *   5.1,3.5,setosa
 *   4.9,3.0,versicolor
 *
 * Column-oriented Dataset:
 *   features[0].name = "sepal_length"
 *   features[0].data = [5.1, 4.9, ...]
 *   labels[0].name = "species"
 *   labels[0].labels = [0, 1, ...]     // integer encoding
 *   labels[0].value_map = ["setosa", "versicolor", "virginica"]
 * @endcode
 *
 * Usage:
 * @code
 * csv_t* csv = csv_load("data/iris.csv");
 * dataset* ds = csv_to_dataset(csv, (const char*[]){"species"}, 1);
 *
 * // Access features: ds->features[i].data[j] = feature i, sample j
 * // Access labels: ds->labels[i].labels[j] = label i, sample j
 *
 * // Expand for binary classification:
 * dataset_expand_binary_labels(ds, 0, BINARY_SIGNED);
 *
 * free_dataset(ds);
 * free(ds);
 * @endcode
 *
 * @license MIT
 */

#ifndef DATASET_H
#define DATASET_H

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Include CSV types from csv.h */
#ifndef CSV_H_INCLUDED
#include "csv.h"
#endif

/* ============================================================================
 * DATA STRUCTURES
 * ============================================================================ */

/**
 * @brief Numerical feature column
 *
 * Stores a single feature (column) of numerical data. All values are stored
 * as doubles for uniformity, regardless of whether they were integers in CSV.
 *
 * @note Use features[i].data[j] to access feature i, sample j
 */
typedef struct {
    const char* name;   /**< Feature name (from CSV header) */
    double* data;       /**< Array of feature values [n samples] */
    size_t n;           /**< Number of samples in this column */
} num_column;

/**
 * @brief Multi-class label column
 *
 * Stores a categorical label column with integer encoding and value mapping.
 * Integer labels allow efficient processing for multi-class classification.
 *
 * @par Example (Iris dataset):
 * @code
 * labels[0].name = "species"
 * labels[0].labels = [0, 0, ..., 1, 1, ..., 2, 2, ...]
 *                    // setosa=0, versicolor=1, virginica=2
 * labels[0].value_map = ["setosa", "versicolor", "virginica"]
 * labels[0].classes = 3
 * @endcode
 */
typedef struct {
    const char* name;       /**< Label column name (from CSV header) */
    int* labels;            /**< Integer labels [n samples], each 0 to classes-1 */
    char** value_map;       /**< Maps index -> original string value */
    size_t n;               /**< Number of samples */
    size_t classes;         /**< Number of unique classes */
} label_column;

/**
 * @brief Binary label encoding modes
 *
 * Specifies how binary labels should be encoded for one-vs-all classification.
 *
 * @var BINARY_01
 * Standard binary encoding: 0 = negative class, 1 = positive class
 * Suitable for: Logistic Regression, SVM, Neural Networks
 *
 * @var BINARY_SIGNED
 * Signed encoding: -1 = negative class, +1 = positive class
 * Suitable for: AdaBoost, SVM with margin-based algorithms
 */
typedef enum {
    BINARY_01,      /**< 0 = negative, 1 = positive */
    BINARY_SIGNED   /**< -1 = negative, +1 = positive (AdaBoost style) */
} binary_mode_t;

/**
 * @brief Binary label column for one-vs-all classification
 *
 * Expanded from a multi-class label column. Each binary column represents
 * one class vs. all other classes (One-vs-All / One-vs-Rest).
 *
 * @par Example:
 * @code
 * // Original: species = {setosa, versicolor, virginica}
 * // Expanded (BINARY_01 mode):
 * binary_labels[0].name = "species_setosa"
 * binary_labels[0].positive_class = "setosa"
 * binary_labels[0].labels = [1, 1, ..., 0, 0, ..., 0, 0, ...]
 *                          // setosa=1, others=0
 *
 * binary_labels[1].name = "species_versicolor"
 * binary_labels[1].labels = [0, 0, ..., 1, 1, ..., 0, 0, ...]
 * @endcode
 */
typedef struct {
    const char* name;           /**< Column name (e.g., "species_setosa") */
    const char* positive_class;/**< The class this column identifies */
    int* labels;               /**< Binary labels [n samples] */
    size_t n;                  /**< Number of samples */
    binary_mode_t mode;        /**< Encoding mode (BINARY_01 or BINARY_SIGNED) */
} binary_label_column;

/**
 * @brief Column-oriented dataset structure
 *
 * The main data structure for ML workflows. Stores features and labels
 * in column-oriented format for efficient access patterns during training.
 *
 * Memory Layout:
 * @code
 * samples:     0    1    2    ...  n-1
 *            +----+----+----+---+-----+
 * features[0]| f00| f01| f02|...|f0n-1|
 * features[1]| f10| f11| f12|...|f1n-1|
 * ...        |....|....|....|...|.....|
 * features[m]| fm0| fm1| fm2|...|fmn-1|
 *            +----+----+----+---+-----+
 * labels[0]  | l00| l01| l02|...|l0n-1|
 * @endcode
 *
 * Access patterns:
 * - Get all feature values for sample i: features[j].data[i] for all j
 * - Get all label values for sample i: labels[j].labels[i] for all j
 */
typedef struct {
    num_column* features;         /**< Numerical feature columns */
    label_column* labels;          /**< Multi-class label columns */
    binary_label_column* binary_labels; /**< Expanded binary label columns */
    size_t rows;                   /**< Number of data samples (rows) */
    size_t num_features;           /**< Number of feature columns */
    size_t num_labels;            /**< Number of multi-class label columns */
    size_t num_binary_labels;      /**< Total number of binary label columns */
} dataset;

/* ============================================================================
 * HELPER FUNCTIONS
 * ============================================================================ */

/**
 * @brief Check if a string represents a numeric value
 *
 * Uses strtod() to check if the entire string is a valid number.
 * Empty strings or strings with trailing non-numeric characters return false.
 *
 * @param[in] str String to check
 *
 * @return true if string is a valid number
 * @return false otherwise
 *
 * @par Examples:
 * @code
 * is_numeric("123.45")  -> true
 * is_numeric("123")     -> true
 * is_numeric("-3.14e-2") -> true
 * is_numeric("")        -> false
 * is_numeric("abc")     -> false
 * is_numeric("12.34abc") -> false
 * @endcode
 */
static bool is_numeric(const char* str) {
    if (!str || *str == '\0')
        return false;
    char* end;
    strtod(str, &end);
    return *end == '\0';
}

/**
 * @brief Build label mapping from string values
 *
 * Converts an array of string label values into:
 * 1. Integer labels (0, 1, 2, ...) assigned in order of first appearance
 * 2. Value map that stores the original string for each integer label
 *
 * Algorithm: Two-pass approach
 * - Pass 1: Count unique values by comparing each value with all previous values
 * - Pass 2: Assign integer indices, reusing existing indices for duplicates
 *
 * @param[out] col   Pointer to label_column to populate
 * @param[in]  values Array of string values [n samples]
 * @param[in]  n      Number of samples
 *
 * @return 0 on success, -1 on memory allocation failure
 *
 * @par Example:
 * @code
 * // Input: ["cat", "dog", "cat", "bird", "dog", "cat"]
 * // Output:
 * //   labels = [0, 1, 0, 2, 1, 0]
 * //   value_map = ["cat", "dog", "bird"]
 * //   classes = 3
 * @endcode
 */
static int build_label_map(label_column* col, char** values, size_t n) {
    /* First pass: count unique values */
    col->classes = 0;
    for (size_t i = 0; i < n; i++) {
        bool found = false;
        for (size_t j = 0; j < i; j++) {
            if (strcmp(values[i], values[j]) == 0) {
                found = true;
                break;
            }
        }
        if (!found)
            col->classes++;
    }

    /* Allocate value_map array */
    col->value_map = (char**)malloc(sizeof(char*) * col->classes);
    if (!col->value_map)
        return -1;

    /* Allocate labels array */
    col->labels = (int*)malloc(sizeof(int) * n);
    if (!col->labels) {
        free(col->value_map);
        return -1;
    }

    /* Second pass: assign indices */
    size_t idx = 0;
    for (size_t i = 0; i < n; i++) {
        bool found = false;
        for (size_t j = 0; j < i; j++) {
            if (strcmp(values[i], values[j]) == 0) {
                col->labels[i] = col->labels[j];
                found = true;
                break;
            }
        }
        if (!found) {
            col->value_map[idx] = (char*)malloc(strlen(values[i]) + 1);
            if (!col->value_map[idx]) { free(col->labels); free(col->value_map); return -1; }
            strcpy(col->value_map[idx], values[i]);
            col->labels[i] = (int)idx;
            idx++;
        }
    }

    return 0;
}

/* ============================================================================
 * DATASET CONVERSION
 * ============================================================================ */

/**
 * @brief Convert row-oriented CSV to column-oriented dataset
 *
 * Main function to transform CSV data into the dataset structure.
 * Automatically identifies numeric columns as features and specified columns
 * as labels.
 *
 * Column Classification:
 * 1. If column name matches one in label_column_names -> treated as label
 * 2. If all values in column are numeric -> treated as numeric feature
 * 3. Otherwise -> treated as feature but stored as 0.0 (placeholder)
 *
 * @param[in] csv                Pointer to parsed CSV data
 * @param[in] label_column_names Array of column names to treat as labels
 * @param[in] num_labels         Number of label column names
 *
 * @return Newly allocated dataset, or NULL on error
 *
 * @par Memory Allocation:
 * - Features: num_columns * sizeof(num_column) + num_rows * sizeof(double) each
 * - Labels: num_label_columns * sizeof(label_column) + allocations for labels/value_map
 *
 * @par Example:
 * @code
 * csv_t* csv = csv_load("data/iris.csv");
 * const char* labels[] = {"species"};
 * dataset* ds = csv_to_dataset(csv, labels, 1);
 * // ds->num_features = 4 (sepal_length, sepal_width, petal_length, petal_width)
 * // ds->num_labels = 1 (species)
 * @endcode
 */
static dataset* csv_to_dataset(const csv_t* csv, const char** label_column_names, size_t num_labels) {
    if (!csv || csv->size == 0)
        return NULL;

    /* Get header row and dimensions */
    csv_row* header = &csv->rows[0];
    size_t num_cols = header->size;
    size_t num_rows = csv->size - 1; /* Exclude header row */

    /* Identify which columns are labels */
    bool* is_label = (bool*)calloc(num_cols, sizeof(bool));
    if (!is_label)
        return NULL;

    for (size_t li = 0; li < num_labels; li++) {
        for (size_t ci = 0; ci < num_cols; ci++) {
            if (strcmp(header->fields[ci], label_column_names[li]) == 0) {
                is_label[ci] = true;
                break;
            }
        }
    }

    /* Count features and label columns */
    size_t num_features = 0, num_label_cols = 0;
    for (size_t i = 0; i < num_cols; i++) {
        if (is_label[i])
            num_label_cols++;
        else
            num_features++;
    }

    /* Allocate dataset structure */
    dataset* ds = (dataset*)malloc(sizeof(dataset));
    if (!ds) {
        free(is_label);
        return NULL;
    }

    /* Initialize dataset fields */
    ds->rows = num_rows;
    ds->num_features = num_features;
    ds->num_labels = num_label_cols;
    ds->num_binary_labels = 0;
    ds->binary_labels = NULL;

    /* Allocate column arrays */
    ds->features = (num_column*)malloc(sizeof(num_column) * num_features);
    ds->labels = (label_column*)malloc(sizeof(label_column) * num_label_cols);

    if (!ds->features || !ds->labels) {
        free(ds->features);
        free(ds->labels);
        free(ds);
        free(is_label);
        return NULL;
    }

    /* Process each column */
    size_t feat_idx = 0;
    size_t label_idx = 0;

    for (size_t ci = 0; ci < num_cols; ci++) {
        if (is_label[ci]) {
            /* --- LABEL COLUMN --- */
            /* Collect string values for this column */
            char** values = (char**)malloc(sizeof(char*) * num_rows);
            if (!values) {
                free(is_label);
                return NULL;
            }
            for (size_t ri = 0; ri < num_rows; ri++) {
                values[ri] = csv->rows[ri + 1].fields[ci];
            }

            /* Build label mapping */
            ds->labels[label_idx].name = header->fields[ci];
            ds->labels[label_idx].n = num_rows;
            if (build_label_map(&ds->labels[label_idx], values, num_rows) < 0) {
                free(values);
                free(is_label);
                return NULL;
            }
            free(values);
            label_idx++;

        } else {
            /* --- FEATURE COLUMN --- */
            /* Check if all values are numeric */
            bool numeric = true;
            for (size_t ri = 0; ri < num_rows; ri++) {
                if (!is_numeric(csv->rows[ri + 1].fields[ci])) {
                    numeric = false;
                    break;
                }
            }

            /* Initialize feature column */
            ds->features[feat_idx].name = header->fields[ci];
            ds->features[feat_idx].n = num_rows;
            ds->features[feat_idx].data = (double*)malloc(sizeof(double) * num_rows);

            if (!ds->features[feat_idx].data) {
                free(is_label);
                return NULL;
            }

            /* Convert string values to doubles */
            if (numeric) {
                for (size_t ri = 0; ri < num_rows; ri++) {
                    ds->features[feat_idx].data[ri] = atof(csv->rows[ri + 1].fields[ci]);
                }
            } else {
                /* Non-numeric features stored as 0.0 (placeholder) */
                for (size_t ri = 0; ri < num_rows; ri++) {
                    ds->features[feat_idx].data[ri] = 0.0;
                }
            }
            feat_idx++;
        }
    }

    free(is_label);
    return ds;
}

/* ============================================================================
 * LABEL UTILITIES
 * ============================================================================ */

/**
 * @brief Get the integer index for a label string value
 *
 * Looks up a string value in the label column's value map and returns
 * its corresponding integer index.
 *
 * @param[in] col   Pointer to label column
 * @param[in] value String value to look up
 *
 * @return Integer index (0 to classes-1) if found
 * @return -1 if not found
 *
 * @par Example:
 * @code
 * // labels[0].value_map = ["setosa", "versicolor", "virginica"]
 * get_label_index(&labels[0], "versicolor")  -> 1
 * get_label_index(&labels[0], "unknown")    -> -1
 * @endcode
 */
static int get_label_index(const label_column* col, const char* value) {
    for (size_t i = 0; i < col->classes; i++) {
        if (strcmp(col->value_map[i], value) == 0) {
            return (int)i;
        }
    }
    return -1;
}

/**
 * @brief Expand a multi-class label into multiple binary labels (One-vs-All)
 *
 * Creates binary_label_column entries for each class in the source label column.
 * Each binary column contains 1 (or +1) for samples belonging to that class
 * and 0 (or -1) for all other samples.
 *
 * @param[in] col  Source multi-class label column
 * @param[in] mode Binary encoding mode (BINARY_01 or BINARY_SIGNED)
 *
 * @return Newly allocated array of binary_label_column (caller must free each)
 * @return NULL on error
 *
 * @par Binary Encoding:
 * @code
 * BINARY_01 mode:
 *   positive_val = 1, negative_val = 0
 *
 * BINARY_SIGNED mode (AdaBoost):
 *   positive_val = 1, negative_val = -1
 * @endcode
 *
 * @par Example:
 * @code
 * // col with 3 classes: ["setosa", "versicolor", "virginica"]
 * binary_label_column* expanded = expand_label_to_binary(col, BINARY_SIGNED);
 * // expanded[0].name = "species_setosa"
 * // expanded[0].positive_class = "setosa"
 * // expanded[0].labels = [1, 1, ..., -1, -1, ..., -1, -1, ...]
 * // expanded[1].name = "species_versicolor"
 * // ...
 * @endcode
 */
static binary_label_column* expand_label_to_binary(const label_column* col, binary_mode_t mode) {
    if (!col || !col->value_map)
        return NULL;

    /* Allocate array of binary columns (one per class) */
    binary_label_column* result = (binary_label_column*)malloc(
        sizeof(binary_label_column) * col->classes);
    if (!result)
        return NULL;

    for (size_t c = 0; c < col->classes; c++) {
        /* Build name: "columnName_className" */
        size_t name_len = strlen(col->name) + 1 + strlen(col->value_map[c]) + 1;
        result[c].name = (char*)malloc(name_len);
        if (!result[c].name) {
            /* Cleanup on failure */
            for (size_t k = 0; k < c; k++) {
                free((char*)result[k].name);
                free(result[k].labels);
            }
            free(result);
            return NULL;
        }
        snprintf((char*)result[c].name, name_len, "%s_%s",
                 col->name, col->value_map[c]);

        result[c].positive_class = col->value_map[c];
        result[c].n = col->n;
        result[c].mode = mode;

        /* Allocate labels array */
        result[c].labels = (int*)malloc(sizeof(int) * col->n);
        if (!result[c].labels) {
            free((char*)result[c].name);
            for (size_t k = 0; k < c; k++) {
                free((char*)result[k].name);
                free(result[k].labels);
            }
            free(result);
            return NULL;
        }

        /* Set binary values based on mode */
        int positive_val = (mode == BINARY_SIGNED) ? 1 : 1;
        int negative_val = (mode == BINARY_SIGNED) ? -1 : 0;

        /* Fill binary labels: 1 if matches this class, 0/-1 otherwise */
        for (size_t i = 0; i < col->n; i++) {
            result[c].labels[i] = (col->labels[i] == (int)c) ?
                                  positive_val : negative_val;
        }
    }

    return result;
}

/**
 * @brief Free a single binary label column
 *
 * Releases memory for a binary_label_column structure including
 * its name string and labels array.
 *
 * @param[in,out] bin_col Pointer to binary column to free
 *
 * @note Safe to call with NULL pointer
 */
static void free_binary_label(binary_label_column* bin_col) {
    if (!bin_col)
        return;
    free((char*)bin_col->name);
    free(bin_col->labels);
    bin_col->name = NULL;
    bin_col->labels = NULL;
}

/**
 * @brief Expand a dataset label column into binary columns
 *
 * Transforms one multi-class label column into multiple binary label columns
 * using One-vs-All strategy. The expanded columns are appended to the
 * dataset's binary_labels array.
 *
 * @param[in,out] ds        Pointer to dataset
 * @param[in]     label_idx Which label column to expand (0 to num_labels-1)
 * @param[in]     mode      Binary encoding mode
 *
 * @return Number of binary columns created (等于原始类别的数量)
 * @return -1 on error
 *
 * @par Example:
 * @code
 * // ds->labels[0] has 3 classes
 * int created = dataset_expand_binary_labels(ds, 0, BINARY_SIGNED);
 * // created = 3
 * // ds->num_binary_labels increased by 3
 * @endcode
 */
static int dataset_expand_binary_labels(dataset* ds, size_t label_idx, binary_mode_t mode) {
    if (!ds || label_idx >= ds->num_labels)
        return -1;

    label_column* col = &ds->labels[label_idx];
    binary_label_column* expanded = expand_label_to_binary(col, mode);
    if (!expanded)
        return -1;

    /* Reallocate binary_labels array */
    size_t new_count = ds->num_binary_labels + col->classes;
    binary_label_column* new_binary = (binary_label_column*)realloc(
        ds->binary_labels, sizeof(binary_label_column) * new_count);
    if (!new_binary) {
        for (size_t i = 0; i < col->classes; i++) {
            free_binary_label(&expanded[i]);
        }
        free(expanded);
        return -1;
    }

    /* Copy expanded columns into dataset */
    for (size_t i = 0; i < col->classes; i++) {
        new_binary[ds->num_binary_labels + i] = expanded[i];
    }

    free(expanded);
    ds->binary_labels = new_binary;
    ds->num_binary_labels = new_count;

    return (int)col->classes;
}

/* ============================================================================
 * MEMORY MANAGEMENT
 * ============================================================================ */

/**
 * @brief Free all memory associated with a dataset
 *
 * Releases all memory for features, labels, and binary labels.
 * Sets all pointers to NULL and counts to 0.
 *
 * @param[in,out] ds Pointer to dataset to free
 *
 * @note Safe to call with NULL pointer
 * @note After calling, caller should also free(ds) if it was malloc'd
 *
 * @par Example:
 * @code
 * dataset* ds = csv_to_dataset(csv, labels, 1);
 * // ... use dataset ...
 * free_dataset(ds);   // Frees all internal arrays
 * free(ds);          // Frees the dataset structure itself
 * @endcode
 */
static void free_dataset(dataset* ds) {
    if (!ds)
        return;

    /* Free feature columns */
    if (ds->features) {
        for (size_t i = 0; i < ds->num_features; i++) {
            free((double*)ds->features[i].data);
            ds->features[i].data = NULL;
        }
        free(ds->features);
        ds->features = NULL;
    }

    /* Free label columns */
    if (ds->labels) {
        for (size_t i = 0; i < ds->num_labels; i++) {
            free(ds->labels[i].labels);
            free(ds->labels[i].value_map);
            ds->labels[i].labels = NULL;
            ds->labels[i].value_map = NULL;
        }
        free(ds->labels);
        ds->labels = NULL;
    }

    /* Free binary label columns */
    if (ds->binary_labels) {
        for (size_t i = 0; i < ds->num_binary_labels; i++) {
            free_binary_label(&ds->binary_labels[i]);
        }
        free(ds->binary_labels);
        ds->binary_labels = NULL;
    }

    /* Reset counters */
    ds->rows = 0;
    ds->num_features = 0;
    ds->num_labels = 0;
    ds->num_binary_labels = 0;
}

#endif /* DATASET_H */
