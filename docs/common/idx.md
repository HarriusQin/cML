# IDX File Format

Reader for IDX (Indexed Binary) format used in MNIST and similar datasets.

## Overview

IDX is a simple binary format developed for the MNIST database of handwritten digits. It supports various numeric data types and arbitrary dimensionality.

## File Format

```
[magic bytes] [dimensions] [data...]
```

### Magic Number

```
+------++----------++------------+
| type || n_dims  ||  zeros     |
+------++----------++------------+
  1 byte    1 byte    1 byte
```

- **type**: Data type code
  - `0x08`: unsigned byte (uint8)
  - `0x09`: signed byte (int8)
  - `0x0B`: short (int16)
  - `0x0C`: int (int32)
  - `0x0D`: float (float32)
  - `0x0E`: double (float64)

- **n_dims**: Number of dimensions (0-255)

- **zeros**: Reserved bytes (typically 0)

### Dimension Codes

Dimensions are stored as 32-bit big-endian unsigned integers.

Example for MNIST images (60000 × 28 × 28):
```
Magic:     0x00000803  (type=UBYTE, n_dims=3)
Dim 0:     60000       (number of images)
Dim 1:     28          (rows)
Dim 2:     28          (columns)
```

## Data Structure

```c
typedef enum {
    IDX_UBYTE = 0x08,   // uint8
    IDX_BYTE  = 0x09,   // int8
    IDX_SHORT = 0x0B,   // int16
    IDX_INT   = 0x0C,    // int32
    IDX_FLOAT = 0x0D,    // float32
    IDX_DOUBLE = 0x0E     // float64
} IDX_TYPE;

typedef struct cIDX {
    IDX_TYPE type;        // Data type
    uint8_t n_dims;       // Number of dimensions
    uint32_t* dims;       // Dimension sizes
    void* idx_data;       // Raw data pointer
    size_t size;          // Total bytes
} cIDX;
```

## Functions

```c
// Load IDX file
static cIDX* idx_load(const char* path);

// Free IDX structure
static void free_idx(cIDX* idx);

// Get element size in bytes
static size_t idx_elem_size(IDX_TYPE type);

// Check endianness
static int is_little_endian(void);
```

## Endianness Handling

IDX format uses big-endian byte order. Automatic byte-swapping is performed on little-endian systems:

```c
static void byte_swap(void* data, size_t elem_size, size_t count) {
    uint8_t* p = (uint8_t*)data;
    for (size_t i = 0; i < count; i++) {
        // Reverse bytes within each element
        for (size_t j = 0; j < elem_size / 2; j++) {
            uint8_t tmp = p[i * elem_size + j];
            p[i * elem_size + j] = p[i * elem_size + elem_size - 1 - j];
            p[i * elem_size + elem_size - 1 - j] = tmp;
        }
    }
}
```

## MNIST Dataset

The MNIST database consists of four IDX files:

| File | Type | Shape | Description |
|------|------|-------|-------------|
| train-images-idx3-ubyte | UBYTE | [60000, 28, 28] | Training images |
| train-labels-idx1-ubyte | UBYTE | [60000] | Training labels |
| t10k-images-idx3-ubyte | UBYTE | [10000, 28, 28] | Test images |
| t10k-labels-idx1-ubyte | UBYTE | [10000] | Test labels |

## Example

```c
#define IDX_IMPLEMENTATION
#include "idx.h"

// Load MNIST images
cIDX* train_images = idx_load("data/train-images-idx3-ubyte");
// Returns: type=IDX_UBYTE, n_dims=3
// dims=[60000, 28, 28]
// Access pixel at image 0, row 10, col 15:
// uint8_t pixel = ((uint8_t*)train_images->idx_data)[0 * 28 * 28 + 10 * 28 + 15];

cIDX* train_labels = idx_load("data/train-labels-idx1-ubyte");
// Returns: type=IDX_UBYTE, n_dims=1
// dims=[60000]
// Access label 0:
// uint8_t label = ((uint8_t*)train_labels->idx_data)[0];

// Use data...
printf("Loaded %u images of size %ux%u\n",
       train_images->dims[0],
       train_images->dims[1],
       train_images->dims[2]);

// Free
free_idx(train_images);
free_idx(train_labels);
```

## Element Sizes

| Type | Code | Size (bytes) |
|------|------|--------------|
| IDX_UBYTE | 0x08 | 1 |
| IDX_BYTE | 0x09 | 1 |
| IDX_SHORT | 0x0B | 2 |
| IDX_INT | 0x0C | 4 |
| IDX_FLOAT | 0x0D | 4 |
| IDX_DOUBLE | 0x0E | 8 |

## Notes

- IDX files are always big-endian (MSB first)
- Automatic conversion for little-endian systems
- Data is stored in row-major order
- For 3D MNIST data: [images, rows, cols] or [features, samples]
- Labels are typically stored as scalar values (1D array)
- Pixel values normalized to [0, 255] for UBYTE type
