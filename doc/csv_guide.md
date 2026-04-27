# CSV Parser Library Guide

A header-only CSV parsing library for C.

## Quick Start

```c
#define CSV_IMPLEMENTATION
#include "csv.h"

int main(void) {
    csv_t* csv = csv_load("data.csv");

    for (size_t i = 0; i < csv->size; i++) {
        for (size_t j = 0; j < csv->rows[i].size; j++) {
            printf("%s ", csv->rows[i].fields[j]);
        }
    }

    free_csv_data(csv);
    free(csv);
    return 0;
}
```

## Data Structures

### `csv_t`

Main CSV structure containing all rows.

```c
typedef struct csv_t {
    csv_row* rows;    // Array of rows
    size_t size;      // Number of rows
    size_t capacity;  // Allocated capacity
} csv_t;
```

### `csv_row`

A single row containing field strings.

```c
typedef struct csv_row {
    char** fields;    // Array of field strings
    size_t size;      // Number of fields
    size_t capacity;  // Allocated capacity
} csv_row;
```

## API Reference

### `csv_load()`

Load and parse a CSV file.

```c
csv_t* csv_load(const char* filepath);
```

**Parameters:**
- `filepath` - Path to the CSV file

**Returns:** Pointer to `csv_t`, or `NULL` on error.

**Example:**
```c
csv_t* csv = csv_load("iris.csv");
if (!csv) {
    perror("Failed to load CSV");
    return 1;
}
```

---

### `parse_csv()`

Parse CSV data from a string.

```c
csv_t* parse_csv(const char* input);
```

**Parameters:**
- `input` - Null-terminated CSV string

**Returns:** Pointer to `csv_t`, or `NULL` on error.

**Example:**
```c
const char* csv_string = "name,age\nAlice,30\nBob,25";
csv_t* csv = parse_csv(csv_string);
```

---

### `free_csv_data()`

Free all rows and fields within a CSV structure.

```c
void free_csv_data(csv_t* csv_data);
```

**Note:** Only frees internal arrays. The `csv_t` pointer itself must be freed by caller.

**Example:**
```c
free_csv_data(csv);
free(csv);
```

---

## Parsing Rules

### Quoted Fields

Fields enclosed in double quotes can contain commas:

```
"John Doe, Jr.",35,NYC
```

Result: `["John Doe, Jr.", "35", "NYC"]`

### Escaped Quotes

Use double quotes inside quoted fields by doubling them:

```
"say ""hello""",42
```

Result: `["say \"hello\"", "42"]`

### Whitespace Trimming

Leading and trailing whitespace is automatically trimmed from unquoted fields:

```
  Alice  ,  30
```

Result: `["Alice", "30"]`

### Line Endings

Supports both LF (`\n`) and CRLF (`\r\n`) line endings.

## Complete Example

```c
#include <stdio.h>
#include <stdlib.h>
#define CSV_IMPLEMENTATION
#include "csv.h"

int main(void) {
    // Load CSV file
    csv_t* csv = csv_load("data.csv");
    if (!csv) {
        fprintf(stderr, "Error loading CSV\n");
        return 1;
    }

    printf("Loaded %zu rows:\n", csv->size);

    // Print header row
    if (csv->size > 0) {
        printf("Header: ");
        for (size_t j = 0; j < csv->rows[0].size; j++) {
            printf("[%s]", csv->rows[0].fields[j]);
        }
        printf("\n");
    }

    // Print first few data rows
    for (size_t i = 1; i < csv->size && i <= 5; i++) {
        printf("Row %zu: ", i);
        for (size_t j = 0; j < csv->rows[i].size; j++) {
            printf("[%s]", csv->rows[i].fields[j]);
        }
        printf("\n");
    }

    // Cleanup
    free_csv_data(csv);
    free(csv);
    return 0;
}
```

## Memory Management

| Function | Frees |
|----------|-------|
| `free_csv_data()` | All rows and fields (internal arrays) |
| Caller's responsibility | The `csv_t` structure itself |

Always pair `csv_load()` / `parse_csv()` with `free_csv_data()` + `free()`.
