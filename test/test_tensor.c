/**
 * @file test_tensor.c
 * @brief Comprehensive test for tensor operations
 */

#define TENSOR_IMPLEMENTATION
#include "tensor.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int nearly_equal(float a, float b, float eps) {
    return fabsf(a - b) < eps;
}

static int test_tensor_create(void) {
    printf("=== test_tensor_create ===\n");
    size_t shape[] = {2, 3, 4};
    Tensor* t = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 3);

    if (t->ndim != 3) { printf("FAIL: ndim\n"); return 1; }
    if (t->shape[0] != 2 || t->shape[1] != 3 || t->shape[2] != 4) { printf("FAIL: shape\n"); return 1; }
    if (t->size != 24) { printf("FAIL: size\n"); return 1; }
    if (t->dtype != TENSOR_DTYPE_F32) { printf("FAIL: dtype\n"); return 1; }

    tensor_free(t);
    printf("PASS\n");
    return 0;
}

static int test_tensor_fill_and_access(void) {
    printf("=== test_tensor_fill_and_access ===\n");
    size_t shape[] = {2, 3, 4};
    Tensor* t = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 3);

    tensor_fill_f32(t, 0.0f);

    size_t idx[] = {1, 2, 3};
    tensor_set_f32(t, idx, 3.14f);
    if (!nearly_equal(tensor_get_f32(t, idx), 3.14f, 1e-6f)) {
        printf("FAIL: get/set\n"); return 1;
    }

    tensor_free(t);
    printf("PASS\n");
    return 0;
}

static int test_tensor_clone(void) {
    printf("=== test_tensor_clone ===\n");
    size_t shape[] = {2, 3};
    Tensor* t = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 2);
    tensor_fill_f32(t, 2.5f);

    Tensor* c = tensor_clone(t);
    if (c->size != t->size) { printf("FAIL: size\n"); return 1; }
    if (!nearly_equal(tensor_get_f32(c, (size_t[]){0,0}), 2.5f, 1e-6f)) {
        printf("FAIL: value\n"); return 1;
    }

    tensor_free(t);
    tensor_free(c);
    printf("PASS\n");
    return 0;
}

static int test_tensor_add(void) {
    printf("=== test_tensor_add ===\n");
    size_t shape[] = {2, 3};
    Tensor* a = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 2);
    Tensor* b = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 2);
    tensor_fill_f32(a, 1.0f);
    tensor_fill_f32(b, 2.0f);

    Tensor* c = tensor_add(a, b);
    if (!nearly_equal(tensor_get_f32(c, (size_t[]){0,0}), 3.0f, 1e-6f)) {
        printf("FAIL: 1+2 != 3\n"); return 1;
    }
    if (c->size != a->size) { printf("FAIL: size\n"); return 1; }

    tensor_free(a); tensor_free(b); tensor_free(c);
    printf("PASS\n");
    return 0;
}

static int test_tensor_mul(void) {
    printf("=== test_tensor_mul ===\n");
    size_t shape[] = {2, 3};
    Tensor* a = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 2);
    Tensor* b = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 2);
    tensor_fill_f32(a, 3.0f);
    tensor_fill_f32(b, 4.0f);

    Tensor* c = tensor_mul(a, b);
    if (!nearly_equal(tensor_get_f32(c, (size_t[]){0,0}), 12.0f, 1e-5f)) {
        printf("FAIL: 3*4 != 12\n"); return 1;
    }

    tensor_free(a); tensor_free(b); tensor_free(c);
    printf("PASS\n");
    return 0;
}

static int test_tensor_scale(void) {
    printf("=== test_tensor_scale ===\n");
    size_t shape[] = {10};
    Tensor* t = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 1);
    for (size_t i = 0; i < 10; i++) tensor_set_f32(t, (size_t[]){i}, (float)i);

    tensor_scale(t, 2.0f);
    if (!nearly_equal(tensor_get_f32(t, (size_t[]){5}), 10.0f, 1e-6f)) {
        printf("FAIL: scale 2*5 != 10\n"); return 1;
    }

    tensor_free(t);
    printf("PASS\n");
    return 0;
}

static int test_tensor_gemm(void) {
    printf("=== test_tensor_gemm ===\n");
    // A[2x3] * B[3x2] = C[2x2]
    // A = [[1,2,3],[4,5,6]]
    // B = [[7,8],[9,10],[11,12]]
    // C = [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
    //   = [[7+18+33, 8+20+36], [28+45+66, 32+50+72]]
    //   = [[58, 64], [139, 154]]

    float A[] = {1,2,3, 4,5,6};
    float B[] = {7,8, 9,10, 11,12};
    float C[4] = {0};

    tensor_gemm(C, A, B, 2, 2, 3, 1.0f, 0.0f);

    if (!nearly_equal(C[0], 58.0f, 1e-5f)) { printf("FAIL: C[0]=%f\n", C[0]); return 1; }
    if (!nearly_equal(C[1], 64.0f, 1e-5f)) { printf("FAIL: C[1]=%f\n", C[1]); return 1; }
    if (!nearly_equal(C[2], 139.0f, 1e-5f)) { printf("FAIL: C[2]=%f\n", C[2]); return 1; }
    if (!nearly_equal(C[3], 154.0f, 1e-5f)) { printf("FAIL: C[3]=%f\n", C[3]); return 1; }

    printf("PASS\n");
    return 0;
}

static int test_tensor_matmul(void) {
    printf("=== test_tensor_matmul ===\n");
    size_t shape_a[] = {2, 3};
    size_t shape_b[] = {3, 2};
    Tensor* a = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape_a, 2);
    Tensor* b = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape_b, 2);

    float* a_data = (float*)a->data;
    float* b_data = (float*)b->data;
    a_data[0]=1; a_data[1]=2; a_data[2]=3;
    a_data[3]=4; a_data[4]=5; a_data[5]=6;
    b_data[0]=7; b_data[1]=8;
    b_data[2]=9; b_data[3]=10;
    b_data[4]=11; b_data[5]=12;

    Tensor* c = tensor_matmul(a, b);
    if (c->shape[0] != 2 || c->shape[1] != 2) { printf("FAIL: shape\n"); return 1; }

    float* c_data = (float*)c->data;
    if (!nearly_equal(c_data[0], 58.0f, 1e-5f)) { printf("FAIL: C[0]\n"); return 1; }
    if (!nearly_equal(c_data[1], 64.0f, 1e-5f)) { printf("FAIL: C[1]\n"); return 1; }
    if (!nearly_equal(c_data[2], 139.0f, 1e-5f)) { printf("FAIL: C[2]\n"); return 1; }
    if (!nearly_equal(c_data[3], 154.0f, 1e-5f)) { printf("FAIL: C[3]\n"); return 1; }

    tensor_free(a); tensor_free(b); tensor_free(c);
    printf("PASS\n");
    return 0;
}

static int test_tensor_sum(void) {
    printf("=== test_tensor_sum ===\n");
    size_t shape[] = {2, 3};
    Tensor* t = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 2);

    float* data = (float*)t->data;
    for (size_t i = 0; i < 6; i++) data[i] = (float)(i + 1);  // [1,2,3,4,5,6]

    // sum axis=1 -> [6, 15]
    Tensor* s = tensor_sum(t, 1);
    float* s_data = (float*)s->data;
    if (!nearly_equal(s_data[0], 6.0f, 1e-5f)) { printf("FAIL: sum[0]=%f\n", s_data[0]); return 1; }
    if (!nearly_equal(s_data[1], 15.0f, 1e-5f)) { printf("FAIL: sum[1]=%f\n", s_data[1]); return 1; }

    tensor_free(t); tensor_free(s);
    printf("PASS\n");
    return 0;
}

static int test_tensor_mean(void) {
    printf("=== test_tensor_mean ===\n");
    size_t shape[] = {2, 2};
    Tensor* t = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 2);

    float* data = (float*)t->data;
    data[0]=1; data[1]=2; data[2]=3; data[3]=4;

    Tensor* m = tensor_mean(t, 1);  // mean along axis=1
    float* m_data = (float*)m->data;
    if (!nearly_equal(m_data[0], 1.5f, 1e-5f)) { printf("FAIL: mean[0]=%f\n", m_data[0]); return 1; }
    if (!nearly_equal(m_data[1], 3.5f, 1e-5f)) { printf("FAIL: mean[1]=%f\n", m_data[1]); return 1; }

    tensor_free(t); tensor_free(m);
    printf("PASS\n");
    return 0;
}

static int test_tensor_max(void) {
    printf("=== test_tensor_max ===\n");
    size_t shape[] = {3, 3};
    Tensor* t = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 2);

    float* data = (float*)t->data;
    for (size_t i = 0; i < 9; i++) data[i] = (float)(i + 1);

    Tensor* m = tensor_max(t, 0);  // max along axis=0
    float* m_data = (float*)m->data;
    if (!nearly_equal(m_data[0], 7.0f, 1e-5f)) { printf("FAIL: max[0]=%f\n", m_data[0]); return 1; }
    if (!nearly_equal(m_data[1], 8.0f, 1e-5f)) { printf("FAIL: max[1]=%f\n", m_data[1]); return 1; }
    if (!nearly_equal(m_data[2], 9.0f, 1e-5f)) { printf("FAIL: max[2]=%f\n", m_data[2]); return 1; }

    tensor_free(t); tensor_free(m);
    printf("PASS\n");
    return 0;
}

static int test_tensor_relu(void) {
    printf("=== test_tensor_relu ===\n");
    size_t shape[] = {4};
    Tensor* t = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 1);
    float* data = (float*)t->data;
    data[0] = -1.0f; data[1] = 2.0f; data[2] = -3.0f; data[3] = 4.0f;

    tensor_relu(t);
    if (!nearly_equal(data[0], 0.0f, 1e-6f)) { printf("FAIL: relu(-1)\n"); return 1; }
    if (!nearly_equal(data[1], 2.0f, 1e-6f)) { printf("FAIL: relu(2)\n"); return 1; }
    if (!nearly_equal(data[2], 0.0f, 1e-6f)) { printf("FAIL: relu(-3)\n"); return 1; }
    if (!nearly_equal(data[3], 4.0f, 1e-6f)) { printf("FAIL: relu(4)\n"); return 1; }

    tensor_free(t);
    printf("PASS\n");
    return 0;
}

static int test_tensor_sigmoid(void) {
    printf("=== test_tensor_sigmoid ===\n");
    size_t shape[] = {3};
    Tensor* t = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 1);
    float* data = (float*)t->data;
    data[0] = 0.0f; data[1] = 1.0f; data[2] = -1.0f;

    tensor_sigmoid(t);
    if (!nearly_equal(data[0], 0.5f, 1e-6f)) { printf("FAIL: sigmoid(0)\n"); return 1; }
    if (!nearly_equal(data[1], 0.7310586f, 1e-5f)) { printf("FAIL: sigmoid(1)\n"); return 1; }
    if (!nearly_equal(data[2], 0.2689414f, 1e-5f)) { printf("FAIL: sigmoid(-1)\n"); return 1; }

    tensor_free(t);
    printf("PASS\n");
    return 0;
}

static int test_tensor_softmax(void) {
    printf("=== test_tensor_softmax ===\n");
    size_t shape[] = {1, 3};  // [batch=1, 3]
    Tensor* t = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 2);
    float* data = (float*)t->data;
    data[0] = 1.0f; data[1] = 2.0f; data[2] = 3.0f;

    tensor_softmax(t, 1);

    float sum = 0.0f;
    for (size_t i = 0; i < 3; i++) sum += data[i];
    if (!nearly_equal(sum, 1.0f, 1e-6f)) { printf("FAIL: sum != 1\n"); return 1; }
    if (!nearly_equal(data[2], 0.66524096f, 1e-5f)) { printf("FAIL: softmax[2]=%f\n", data[2]); return 1; }

    tensor_free(t);
    printf("PASS\n");
    return 0;
}

static int test_tensor_maxpool2d(void) {
    printf("=== test_tensor_maxpool2d ===\n");
    // Input [1, 1, 4, 4]
    size_t shape[] = {1, 1, 4, 4};
    Tensor* t = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 4);
    float* data = (float*)t->data;
    for (size_t i = 0; i < 16; i++) data[i] = (float)i;  // 0..15

    // 2x2 maxpool with stride 2 -> [1, 1, 2, 2]
    Tensor* p = tensor_maxpool2d(t, 2, 2, 2, 2);
    float* p_data = (float*)p->data;
    // [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]] maxpool 2x2 stride 2
    // [[5,7],[13,15]]
    if (!nearly_equal(p_data[0], 5.0f, 1e-5f)) { printf("FAIL: p[0]=%f\n", p_data[0]); return 1; }
    if (!nearly_equal(p_data[1], 7.0f, 1e-5f)) { printf("FAIL: p[1]=%f\n", p_data[1]); return 1; }
    if (!nearly_equal(p_data[2], 13.0f, 1e-5f)) { printf("FAIL: p[2]=%f\n", p_data[2]); return 1; }
    if (!nearly_equal(p_data[3], 15.0f, 1e-5f)) { printf("FAIL: p[3]=%f\n", p_data[3]); return 1; }

    tensor_free(t); tensor_free(p);
    printf("PASS\n");
    return 0;
}

static int test_tensor_conv2d(void) {
    printf("=== test_tensor_conv2d ===\n");
    // Input [1, 1, 3, 3], no padding, stride 1
    size_t in_shape[] = {1, 1, 3, 3};
    Tensor* input = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, in_shape, 4);
    float* in_data = (float*)input->data;
    // [[1,2,3],[4,5,6],[7,8,9]]
    for (size_t i = 0; i < 9; i++) in_data[i] = (float)(i + 1);

    // Weight [1, 1, 2, 2] (one 2x2 filter)
    size_t w_shape[] = {1, 1, 2, 2};
    Tensor* weight = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, w_shape, 4);
    float* w_data = (float*)weight->data;
    w_data[0]=1; w_data[1]=0; w_data[2]=0; w_data[3]=1;  // identity-like

    Conv2DParams params = {1, 1, 0, 0, 1, 1};  // stride=1, pad=0

    // Output [1, 1, 2, 2]
    // [1*1+2*0+4*0+5*1=6, ...]
    Tensor* output = tensor_conv2d(input, weight, &params);
    float* out_data = (float*)output->data;

    // Without padding: (3-2+0)/1+1 = 2
    // Conv with [[1,0],[0,1]] on [[1,2,3],[4,5,6],[7,8,9]]:
    // out[0,0] = 1*1 + 2*0 + 4*0 + 5*1 = 6
    // out[0,1] = 2*1 + 3*0 + 5*0 + 6*1 = 8
    // out[1,0] = 4*1 + 5*0 + 7*0 + 8*1 = 12
    // out[1,1] = 5*1 + 6*0 + 8*0 + 9*1 = 14
    if (!nearly_equal(out_data[0], 6.0f, 1e-5f)) { printf("FAIL: out[0]=%f\n", out_data[0]); return 1; }
    if (!nearly_equal(out_data[1], 8.0f, 1e-5f)) { printf("FAIL: out[1]=%f\n", out_data[1]); return 1; }
    if (!nearly_equal(out_data[2], 12.0f, 1e-5f)) { printf("FAIL: out[2]=%f\n", out_data[2]); return 1; }
    if (!nearly_equal(out_data[3], 14.0f, 1e-5f)) { printf("FAIL: out[3]=%f\n", out_data[3]); return 1; }

    tensor_free(input); tensor_free(weight); tensor_free(output);
    printf("PASS\n");
    return 0;
}

static int test_tensor_quantize_dequantize(void) {
    printf("=== test_tensor_quantize_dequantize ===\n");
    size_t shape[] = {10};
    Tensor* t = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 1);
    float* data = (float*)t->data;
    for (size_t i = 0; i < 10; i++) data[i] = (float)i * 10.0f;  // 0, 10, 20, ..., 90

    Tensor* q = tensor_quantize_affine(t, TENSOR_DTYPE_UINT8);
    if (q->quant_type != QUANT_AFFINE) { printf("FAIL: quant_type\n"); return 1; }

    Tensor* restored = tensor_dequantize(q);
    float* r_data = (float*)restored->data;

    for (size_t i = 0; i < 10; i++) {
        if (!nearly_equal(r_data[i], data[i], 1.0f)) {  // tolerance for quantization error
            printf("FAIL: [%zu] %f vs %f\n", i, r_data[i], data[i]); return 1;
        }
    }

    tensor_free(t); tensor_free(q); tensor_free(restored);
    printf("PASS\n");
    return 0;
}

static int test_tensor_pad(void) {
    printf("=== test_tensor_pad ===\n");
    size_t shape[] = {2, 2};
    Tensor* t = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 2);
    float* data = (float*)t->data;
    data[0]=1; data[1]=2; data[2]=3; data[3]=4;

    Tensor* p = tensor_pad(t, 1, 1, 0.0f);  // pad 1 on all sides
    // [1,2] -> [[0,0,0],[0,1,2,0],[0,3,4,0],[0,0,0,0]]
    if (p->shape[0] != 4 || p->shape[1] != 4) { printf("FAIL: shape\n"); return 1; }

    float* p_data = (float*)p->data;
    if (!nearly_equal(p_data[5], 1.0f, 1e-6f)) { printf("FAIL: p[5]=%f\n", p_data[5]); return 1; }
    if (!nearly_equal(p_data[6], 2.0f, 1e-6f)) { printf("FAIL: p[6]=%f\n", p_data[6]); return 1; }
    if (!nearly_equal(p_data[9], 3.0f, 1e-6f)) { printf("FAIL: p[9]=%f\n", p_data[9]); return 1; }
    if (!nearly_equal(p_data[10], 4.0f, 1e-6f)) { printf("FAIL: p[10]=%f\n", p_data[10]); return 1; }

    tensor_free(t); tensor_free(p);
    printf("PASS\n");
    return 0;
}

static int test_tensor_clip(void) {
    printf("=== test_tensor_clip ===\n");
    size_t shape[] = {5};
    Tensor* t = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 1);
    float* data = (float*)t->data;
    data[0] = -5.0f; data[1] = 0.0f; data[2] = 3.0f; data[3] = 7.0f; data[4] = 10.0f;

    tensor_clip(t, 0.0f, 5.0f);

    if (!nearly_equal(data[0], 0.0f, 1e-6f)) { printf("FAIL: clip(-5)\n"); return 1; }
    if (!nearly_equal(data[3], 5.0f, 1e-6f)) { printf("FAIL: clip(7)\n"); return 1; }
    if (!nearly_equal(data[4], 5.0f, 1e-6f)) { printf("FAIL: clip(10)\n"); return 1; }

    tensor_free(t);
    printf("PASS\n");
    return 0;
}

static int test_tensor_randn(void) {
    printf("=== test_tensor_randn ===\n");
    size_t shape[] = {1000};
    Tensor* t = tensor_create(TENSOR_DTYPE_F32, TENSOR_LAYOUT_NCHW, shape, 1);
    tensor_fill_randn(t, 0.0f, 1.0f);

    float* data = (float*)t->data;
    float mean = 0.0f;
    for (size_t i = 0; i < 1000; i++) mean += data[i];
    mean /= 1000.0f;

    // Mean should be close to 0 (within 0.1)
    if (fabsf(mean) > 0.2f) { printf("FAIL: mean=%f\n", mean); return 1; }

    tensor_free(t);
    printf("PASS\n");
    return 0;
}

int main(void) {
    printf("=== Tensor Comprehensive Tests ===\n\n");

    int failures = 0;
    failures += test_tensor_create();
    failures += test_tensor_fill_and_access();
    failures += test_tensor_clone();
    failures += test_tensor_add();
    failures += test_tensor_mul();
    failures += test_tensor_scale();
    failures += test_tensor_gemm();
    failures += test_tensor_matmul();
    failures += test_tensor_sum();
    failures += test_tensor_mean();
    failures += test_tensor_max();
    failures += test_tensor_relu();
    failures += test_tensor_sigmoid();
    failures += test_tensor_softmax();
    failures += test_tensor_maxpool2d();
    failures += test_tensor_conv2d();
    failures += test_tensor_quantize_dequantize();
    failures += test_tensor_pad();
    failures += test_tensor_clip();
    failures += test_tensor_randn();

    printf("\n=== Summary: %d failures ===\n", failures);
    return failures ? 1 : 0;
}
