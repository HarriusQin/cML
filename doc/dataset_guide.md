# Dataset Library Guide

A header-only column-oriented dataset library for machine learning, converting row-oriented CSV data into optimized columnar format.

## Quick Start

```c
#define CSV_IMPLEMENTATION
#include "csv.h"
#define DATASET_IMPLEMENTATION
#include "dataset.h"

int main(void) {
    // Load CSV and convert to dataset
    csv_t* csv = csv_load("iris.csv");
    const char* labels[] = {"species"};
    dataset* ds = csv_to_dataset(csv, labels, 1);

    // Access features
    for (size_t i = 0; i < ds->num_features; i++) {
        printf("%s: %.1f, %.1f, ...\n",
               ds->features[i].name,
               ds->features[i].data[0],
               ds->features[i].data[1]);
    }

    // Cleanup
    free_dataset(ds);
    free(ds);
    free_csv_data(csv);
    free(csv);
    return 0;
}
```

## Data Structures

### `num_column`

Numerical feature column.

```c
typedef struct {
    const char* name;   // Feature name (from CSV header)
    double* data;       // Array of feature values [n samples]
    size_t n;           // Number of samples
} num_column;
```

**Access:** `features[i].data[j]` = feature i, sample j

---

### `label_column`

Multi-class label column with integer encoding.

```c
typedef struct {
    const char* name;       // Label column name
    int* labels;            // Integer labels [n samples]
    char** value_map;       // Maps index -> original string
    size_t n;               // Number of samples
    size_t classes;         // Number of unique classes
} label_column;
```

**Access:**
- `labels[i].labels[j]` = integer label for sample j
- `labels[i].value_map[k]` = original string for class k

---

### `binary_label_column`

Binary label column for one-vs-all classification.

```c
typedef enum {
    BINARY_01,      // 0 = negative, 1 = positive
    BINARY_SIGNED   // -1 = negative, +1 = positive
} binary_mode_t;

typedef struct {
    const char* name;           // e.g., "species_setosa"
    const char* positive_class;// e.g., "setosa"
    int* labels;               // Binary labels [n samples]
    size_t n;                  // Number of samples
    binary_mode_t mode;        // Encoding mode
} binary_label_column;
```

---

### `dataset`

Complete dataset structure.

```c
typedef struct {
    num_column* features;         // Numerical feature columns
    label_column* labels;         // Multi-class label columns
    binary_label_column* binary_labels; // Expanded binary labels
    size_t rows;                 // Number of samples
    size_t num_features;         // Number of feature columns
    size_t num_labels;           // Number of label columns
    size_t num_binary_labels;    // Number of binary label columns
} dataset;
```

## Memory Layout

```
              Sample 0   Sample 1   Sample 2   ...  Sample n-1
            +----------+----------+----------+---+----------+
features[0] |  f00     |  f01     |  f02     |...|  f0n-1  |
features[1] |  f10     |  f11     |  f12     |...|  f1n-1  |
...         |  ...    |  ...    |  ...    |...|  ...    |
features[m] |  fm0    |  fm1    |  fm2     |...|  fmn-1  |
            +----------+----------+----------+---+----------+
labels[0]   |  l00     |  l01     |  l02     |...|  l0n-1  |
            +----------+----------+----------+---+----------+
```

**Access patterns:**
- All features for sample i: `features[j].data[i]` for j = 0..num_features-1
- All labels for sample i: `labels[j].labels[i]` for j = 0..num_labels-1

## API Reference

### `csv_to_dataset()`

Convert CSV to column-oriented dataset.

```c
dataset* csv_to_dataset(const csv_t* csv,
                        const char** label_column_names,
                        size_t num_labels);
```

**Parameters:**
- `csv` - Parsed CSV data (first row = header)
- `label_column_names` - Array of column names to treat as labels
- `num_labels` - Number of label columns

**Returns:** Pointer to `dataset`, or `NULL` on error.

**Example:**
```c
// Treat "species" as label column
const char* labels[] = {"species"};
dataset* ds = csv_to_dataset(csv, labels, 1);
// Features: sepal_length, sepal_width, petal_length, petal_width
// Labels: species (integer encoded: 0=setosa, 1=versicolor, 2=virginica)
```

---

### `get_label_index()`

Get integer index for a label string value.

```c
int get_label_index(const label_column* col, const char* value);
```

**Returns:** Integer index (0 to classes-1), or -1 if not found.

**Example:**
```c
int idx = get_label_index(&ds->labels[0], "versicolor");
// idx = 1
```

---

### `dataset_expand_binary_labels()`

Expand multi-class labels to binary one-vs-all format.

```c
int dataset_expand_binary_labels(dataset* ds,
                                 size_t label_idx,
                                 binary_mode_t mode);
```

**Parameters:**
- `ds` - Dataset to modify
- `label_idx` - Which label column to expand (0 to num_labels-1)
- `mode` - `BINARY_01` or `BINARY_SIGNED`

**Returns:** Number of binary columns created, or -1 on error.

**Example:**
```c
// Expand "species" to 3 binary columns
dataset_expand_binary_labels(ds, 0, BINARY_01);
// Creates: species_setosa, species_versicolor, species_virginica

dataset_expand_binary_labels(ds, 0, BINARY_SIGNED);
// Creates: species_setosa, species_versicolor, species_virginica (-1/+1)
```

---

### `free_dataset()`

Free all memory in a dataset.

```c
void free_dataset(dataset* ds);
```

**Note:** Only frees internal arrays. The `dataset` pointer itself must be freed by caller.

## Binary Label Encoding

### BINARY_01 (Standard)

```c
// For 3-class classification: setosa, versicolor, virginica
dataset_expand_binary_labels(ds, 0, BINARY_01);

// species_setosa:     [1, 1, ..., 0, 0, ..., 0, 0, ...]
// species_versicolor: [0, 0, ..., 1, 1, ..., 0, 0, ...]
// species_virginica:  [0, 0, ..., 0, 0, ..., 1, 1, ...]
```

### BINARY_SIGNED (AdaBoost)

```c
dataset_expand_binary_labels(ds, 0, BINARY_SIGNED);

// species_setosa:     [ 1,  1, ..., -1, -1, ..., -1, -1, ...]
// species_versicolor: [-1, -1, ...,  1,  1, ..., -1, -1, ...]
// species_virginica:  [-1, -1, ..., -1, -1, ...,  1,  1, ...]
```

## Complete Example

```c
#include <stdio.h>
#include <stdlib.h>
#define CSV_IMPLEMENTATION
#include "csv.h"
#define DATASET_IMPLEMENTATION
#include "dataset.h"

int main(void) {
    // Load and convert
    csv_t* csv = csv_load("iris.csv");
    if (!csv) {
        fprintf(stderr, "Failed to load CSV\n");
        return 1;
    }

    const char* labels[] = {"species"};
    dataset* ds = csv_to_dataset(csv, labels, 1);
    if (!ds) {
        fprintf(stderr, "Failed to convert to dataset\n");
        free_csv_data(csv);
        free(csv);
        return 1;
    }

    // Print dataset info
    printf("Dataset: %zu samples, %zu features, %zu labels\n",
           ds->rows, ds->num_features, ds->num_labels);

    // Print features
    for (size_t i = 0; i < ds->num_features; i++) {
        printf("  Feature[%zu] %s: ", i, ds->features[i].name);
        for (size_t j = 0; j < 3; j++) {
            printf("%.1f ", ds->features[i].data[j]);
        }
        printf("...\n");
    }

    // Print label info
    printf("  Label[0] %s: %zu classes\n",
           ds->labels[0].name, ds->labels[0].classes);
    for (size_t i = 0; i < ds->labels[0].classes; i++) {
        printf("    Class %zu: %s\n", i, ds->labels[0].value_map[i]);
    }

    // Expand to binary labels
    dataset_expand_binary_labels(ds, 0, BINARY_SIGNED);
    printf("  Expanded to %zu binary columns\n", ds->num_binary_labels);

    // Cleanup
    free_dataset(ds);
    free(ds);
    free_csv_data(csv);
    free(csv);
    return 0;
}
```

## Memory Management

| Function | Frees |
|----------|-------|
| `free_dataset()` | All features, labels, binary_labels arrays |
| Caller's responsibility | The `dataset` and `csv_t` structures themselves |

Always pair `csv_to_dataset()` with `free_dataset()` + `free()`, and `csv_load()` with `free_csv_data()` + `free()`.

## Error Handling

All allocation functions return `NULL` on failure. Check return values:

```c
csv_t* csv = csv_load("data.csv");
if (!csv) {
    perror("csv_load failed");
    return 1;
}

dataset* ds = csv_to_dataset(csv, labels, 1);
if (!ds) {
    fprintf(stderr, "csv_to_dataset failed\n");
    free_csv_data(csv);
    free(csv);
    return 1;
}
```
