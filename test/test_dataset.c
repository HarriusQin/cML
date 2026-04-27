#include <stdio.h>
#include <stdlib.h>
#define CSV_IMPLEMENTATION
#include "csv.h"
#define DATASET_IMPLEMENTATION
#include "dataset.h"

int main(void) {
    printf("=== Dataset Test ===\n\n");

    // Test 1: Load CSV
    printf("Test 1: Loading CSV from file...\n");
    csv_t* csv = csv_load("data/iris.csv");
    if (!csv) {
        fprintf(stderr, "  FAILED: Could not load data/iris.csv\n");
        return 1;
    }
    printf("  OK: Loaded %zu rows\n", csv->size);

    // Test 2: Convert to dataset
    printf("\nTest 2: Converting CSV to dataset...\n");
    const char* label_cols[] = {"species"};
    dataset* ds = csv_to_dataset(csv, label_cols, 1);
    if (!ds) {
        fprintf(stderr, "  FAILED: Could not convert to dataset\n");
        free_csv_data(csv);
        free(csv);
        return 1;
    }
    printf("  OK: Dataset has %zu samples, %zu features, %zu labels\n",
           ds->rows, ds->num_features, ds->num_labels);

    // Test 3: Check features
    printf("\nTest 3: Checking features...\n");
    for (size_t i = 0; i < ds->num_features; i++) {
        printf("  [%zu] %s: ", i, ds->features[i].name);
        for (size_t j = 0; j < 3 && j < ds->features[i].n; j++) {
            printf("%.1f ", ds->features[i].data[j]);
        }
        printf("...\n");
    }

    // Test 4: Check labels
    printf("\nTest 4: Checking labels...\n");
    for (size_t i = 0; i < ds->num_labels; i++) {
        printf("  [%zu] %s:\n", i, ds->labels[i].name);
        printf("    Classes (%zu): ", ds->labels[i].classes);
        for (size_t j = 0; j < ds->labels[i].classes; j++) {
            printf("%s", ds->labels[i].value_map[j]);
            if (j < ds->labels[i].classes - 1) printf(", ");
        }
        printf("\n");
    }

    // Test 5: Label lookup
    printf("\nTest 5: Label index lookup...\n");
    int idx = get_label_index(&ds->labels[0], "versicolor");
    printf("  Index of 'versicolor': %d\n", idx);
    if (idx != 1) {
        fprintf(stderr, "  FAILED: Expected 1, got %d\n", idx);
        return 1;
    }
    printf("  OK\n");

    // Test 6: Expand to binary labels (0/1 mode)
    printf("\nTest 6: Expanding to binary labels (0/1 mode)...\n");
    int expanded = dataset_expand_binary_labels(ds, 0, BINARY_01);
    if (expanded < 0) {
        fprintf(stderr, "  FAILED: Could not expand labels\n");
        return 1;
    }
    printf("  Expanded into %d binary columns\n", expanded);
    printf("  First 5 values of each binary column:\n");
    for (size_t i = 0; i < ds->num_binary_labels; i++) {
        printf("    [%zu] %s: ", i, ds->binary_labels[i].name);
        for (size_t j = 0; j < 5; j++) {
            printf("%d ", ds->binary_labels[i].labels[j]);
        }
        printf("...\n");
    }

    // Test 7: Verify setosa is correctly marked
    printf("\nTest 7: Verifying setosa samples (first 50)...\n");
    int errors = 0;
    for (size_t i = 0; i < 50; i++) {
        if (ds->binary_labels[0].labels[i] != 1) {
            printf("  ERROR: Expected 1 at index %zu, got %d\n", i, ds->binary_labels[0].labels[i]);
            errors++;
        }
    }
    if (errors == 0) {
        printf("  OK: All first 50 samples correctly labeled as setosa\n");
    }

    // Test 8: Expand to binary labels (signed mode)
    printf("\nTest 8: Expanding to binary labels (signed -1/+1 mode)...\n");
    size_t before = ds->num_binary_labels;
    expanded = dataset_expand_binary_labels(ds, 0, BINARY_SIGNED);
    if (expanded < 0) {
        fprintf(stderr, "  FAILED: Could not expand labels\n");
        return 1;
    }
    printf("  Expanded into %d signed binary columns\n", expanded);
    printf("  First 5 values of first signed binary column:\n");
    printf("    [%zu] %s: ", before, ds->binary_labels[before].name);
    for (size_t j = 0; j < 5; j++) {
        printf("%d ", ds->binary_labels[before].labels[j]);
    }
    printf("...\n");

    // Cleanup
    printf("\n=== All Dataset Tests Passed ===\n");
    free_dataset(ds);
    free(ds);
    free_csv_data(csv);
    free(csv);

    return 0;
}
