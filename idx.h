#ifndef __C_IDX_H__
#define __C_IDX_H__

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

typedef enum {
    IDX_UBYTE = 0x08,
    IDX_BYTE = 0x09,
    IDX_SHORT = 0x0B,
    IDX_INT = 0x0C,
    IDX_FLOAT = 0x0D,
    IDX_DOUBLE = 0x0E
} IDX_TYPE;

typedef struct cIDX {
    IDX_TYPE type;
    uint8_t n_dims;
    uint32_t* dims;
    void* idx_data;
    size_t size;
} cIDX;

static int is_little_endian(void) {
    uint16_t test = 1;
    return *(uint8_t*)&test == 1;
}

static void byte_swap(void* data, size_t elem_size, size_t count) {
    uint8_t* p = (uint8_t*)data;
    for (size_t i = 0; i < count; i++) {
        for (size_t j = 0; j < elem_size / 2; j++) {
            uint8_t tmp = p[i * elem_size + j];
            p[i * elem_size + j] = p[i * elem_size + elem_size - 1 - j];
            p[i * elem_size + elem_size - 1 - j] = tmp;
        }
    }
}

static size_t idx_elem_size(IDX_TYPE type) {
    switch (type) {
        case IDX_UBYTE:
        case IDX_BYTE:  return 1;
        case IDX_SHORT: return 2;
        case IDX_INT:
        case IDX_FLOAT: return 4;
        case IDX_DOUBLE: return 8;
        default: return 0;
    }
}

static uint32_t read_u32_be(FILE* fp) {
    uint32_t val;
    fread(&val, sizeof(uint32_t), 1, fp);
    if (is_little_endian()) {
        uint8_t* p = (uint8_t*)&val;
        uint8_t tmp = p[0]; p[0] = p[3]; p[3] = tmp;
        tmp = p[1]; p[1] = p[2]; p[2] = tmp;
    }
    return val;
}

static cIDX* idx_load(const char* path) {
    FILE* fp = fopen(path, "rb");
    if (!fp) return NULL;

    uint32_t magic = read_u32_be(fp);
    IDX_TYPE type = (IDX_TYPE)((magic >> 8) & 0xFF);
    uint8_t ndims = magic & 0xFF;

    cIDX* idx = (cIDX*)malloc(sizeof(cIDX));
    idx->type = type;
    idx->n_dims = ndims;
    idx->dims = (uint32_t*)malloc(sizeof(uint32_t) * ndims);

    size_t total = 1;
    for (uint8_t d = 0; d < ndims; d++) {
        idx->dims[d] = read_u32_be(fp);
        total *= idx->dims[d];
    }

    size_t esize = idx_elem_size(type);
    idx->size = total * esize;
    idx->idx_data = malloc(idx->size);

    fread(idx->idx_data, esize, total, fp);
    if (is_little_endian())
        byte_swap(idx->idx_data, esize, total);

    fclose(fp);
    return idx;
}

static void free_idx(cIDX* idx) {
    if (!idx) return;
    free(idx->idx_data);
    free(idx->dims);
    free(idx);
}

#endif
