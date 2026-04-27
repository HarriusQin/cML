#include <stdio.h>
#include <stdlib.h>
#define CSV_IMPLEMENTATION
#include "csv.h"

int main(void) {
    printf("=== CSV Parser Test ===\n\n");

    // Test 1: Load CSV from file
    printf("Test 1: Loading CSV from file...\n");
    csv_t* csv = csv_load("data/iris.csv");
    if (!csv) {
        fprintf(stderr, "FAILED: Could not load data/iris.csv\n");
        return 1;
    }
    printf("  OK: Loaded %zu rows\n", csv->size);

    // Test 2: Check header row
    printf("\nTest 2: Checking header row...\n");
    if (csv->size < 1) {
        fprintf(stderr, "  FAILED: No rows in CSV\n");
        return 1;
    }
    csv_row* header = &csv->rows[0];
    printf("  Header columns (%zu): ", header->size);
    for (size_t i = 0; i < header->size; i++) {
        printf("%s", header->fields[i]);
        if (i < header->size - 1) printf(", ");
    }
    printf("\n");

    // Test 3: Check data rows
    printf("\nTest 3: Checking data rows...\n");
    if (csv->size < 2) {
        fprintf(stderr, "  FAILED: No data rows\n");
        return 1;
    }
    printf("  First data row: ");
    for (size_t i = 0; i < csv->rows[1].size; i++) {
        printf("[%s]", csv->rows[1].fields[i]);
        if (i < csv->rows[1].size - 1) printf(", ");
    }
    printf("\n");
    printf("  Last data row: ");
    csv_row* last = &csv->rows[csv->size - 1];
    for (size_t i = 0; i < last->size; i++) {
        printf("[%s]", last->fields[i]);
        if (i < last->size - 1) printf(", ");
    }
    printf("\n");

    // Test 4: Parse simple inline CSV
    printf("\nTest 4: Parsing inline CSV...\n");
    const char* inline_csv = "name,age,city\nAlice,30,NYC\nBob,25,LA\n\"Charlie, Jr.\",35,Chicago";
    csv_t* inline_data = parse_csv(inline_csv);
    if (!inline_data) {
        fprintf(stderr, "  FAILED: Could not parse inline CSV\n");
        return 1;
    }
    printf("  Parsed %zu rows\n", inline_data->size);
    for (size_t i = 0; i < inline_data->size; i++) {
        printf("  Row %zu: ", i);
        for (size_t j = 0; j < inline_data->rows[i].size; j++) {
            printf("[%s]", inline_data->rows[i].fields[j]);
            if (j < inline_data->rows[i].size - 1) printf(", ");
        }
        printf("\n");
    }
    free_csv_data(inline_data);
    free(inline_data);

    // Cleanup
    printf("\n=== All CSV Tests Passed ===\n");
    free_csv_data(csv);
    free(csv);

    return 0;
}
