#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>

typedef struct Matrix
{
    double *mat;
    size_t rows;
    size_t cols;
} Matrix;

int matrix_init(Matrix *mat, size_t n_rows, size_t n_cols) {
    if (mat->mat) free(mat->mat);

    double *matrix = calloc(n_rows * n_cols, sizeof(double));
    if (matrix == NULL) return 1;

    mat->rows = n_rows;
    mat->cols = n_cols;
    mat->mat = matrix;
    return 0;
}

static inline double* matrix_at(Matrix* self, size_t row, size_t col) {
    return &self->mat[row * self->cols + col];
}

static inline double* matrix_transpose_at(Matrix* self, size_t row, size_t col) {
    return &self->mat[col * self->rows + row];
}

int matrix_multiply(const Matrix *restrict left, const Matrix *restrict right, Matrix *restrict product) {
    if (left->cols != right->rows) return 1;
    if (matrix_init(product, left->rows, right->cols)) return 1;

    const size_t m = left->rows;
    const size_t n = right->cols;
    const size_t k = left->cols;

    // 分块乘法：block_size 设为 64，适合 L1 cache
    const size_t block = 64;
    double *restrict a = left->mat;
    double *restrict b = right->mat;
    double *restrict c = product->mat;

    for (size_t ii = 0; ii < m; ii += block) {
        for (size_t jj = 0; jj < n; jj += block) {
            for (size_t kk = 0; kk < k; kk += block) {
                size_t i_max = (ii + block < m) ? ii + block : m;
                size_t j_max = (jj + block < n) ? jj + block : n;
                size_t k_max = (kk + block < k) ? kk + block : k;
                for (size_t i = ii; i < i_max; i++) {
                    size_t a_row = i * k;
                    size_t c_row = i * n;
                    for (size_t kk_off = kk; kk_off < k_max; kk_off++) {
                        double a_ik = a[a_row + kk_off];
                        size_t b_row = kk_off * n;
                        for (size_t j = jj; j < j_max; j++) {
                            c[c_row + j] += a_ik * b[b_row + j];
                        }
                    }
                }
            }
        }
    }
    return 0;
}

// 原地转置（仅限方阵）
int matrix_transpose_inplace(Matrix *mat) {
    if (mat->rows != mat->cols) return 1;
    for (size_t i = 0; i < mat->rows; i++)
        for (size_t j = i + 1; j < mat->cols; j++) {
            double tmp = mat->mat[i * mat->cols + j];
            mat->mat[i * mat->cols + j] = mat->mat[j * mat->cols + i];
            mat->mat[j * mat->cols + i] = tmp;
        }
    return 0;
}

// 非方阵转置（需要新矩阵）
int matrix_transpose(const Matrix *restrict mat, Matrix *restrict result) {
    if (matrix_init(result, mat->cols, mat->rows)) return 1;
    for (size_t i = 0; i < mat->rows; i++)
        for (size_t j = 0; j < mat->cols; j++)
            result->mat[j * result->cols + i] = mat->mat[i * mat->cols + j];
    return 0;
}

double matrix_det(Matrix *mat) {
    if (mat->cols != mat->rows) return 1;
    if (mat->cols == 1) return mat->mat[0];
    if (mat->cols == 2)
        return mat->mat[0] * mat->mat[3] - mat->mat[1] * mat->mat[2];

    Matrix copy = {0};
    matrix_init(&copy, mat->rows, mat->cols);
    memcpy(copy.mat, mat->mat, mat->rows * mat->cols * sizeof(double));

    double *tmp_row = malloc(copy.cols * sizeof(double));
    if (!tmp_row) { free(copy.mat); return 1; }

    double det = 1.0;
    int sign = 1;

    // LU 分解：只做下三角消元，不需要右侧单位阵
    for (size_t k = 0; k < copy.rows; k++) {
        size_t row_k_offset = k * copy.cols;

        // 选主元
        size_t max_row = k;
        double max_val = fabs(copy.mat[row_k_offset + k]);
        for (size_t i = k + 1; i < copy.rows; i++) {
            double val = fabs(copy.mat[i * copy.cols + k]);
            if (val > max_val) { max_val = val; max_row = i; }
        }

        if (max_val < 1e-12) { det = 0; goto cleanup; }

        if (max_row != k) {
            double *row_k = &copy.mat[row_k_offset];
            double *row_max = &copy.mat[max_row * copy.cols];
            memcpy(tmp_row, row_k, copy.cols * sizeof(double));
            memcpy(row_k, row_max, copy.cols * sizeof(double));
            memcpy(row_max, tmp_row, copy.cols * sizeof(double));
            sign = -sign;
        }

        // 归一化 pivot
        double pivot_inv = 1.0 / copy.mat[row_k_offset + k];
        double *pivot_row = &copy.mat[row_k_offset];

        // 只消元 j>k 列（LU 分解只需下三角部分）
        for (size_t i = k + 1; i < copy.rows; i++) {
            double *row_i = &copy.mat[i * copy.cols];
            double factor = row_i[k] * pivot_inv;
            row_i[k] = factor;  // 保存 L 的元素（可选）
            for (size_t j = k + 1; j < copy.cols; j++)
                row_i[j] -= factor * pivot_row[j];
        }
    }

    // det = sign × Π(U[i][i])，U 对角线即为 copy.mat[i*cols+i]
    for (size_t i = 0; i < copy.rows; i++)
        det *= copy.mat[i * copy.cols + i];
    det *= sign;

cleanup:
    free(tmp_row);
    free(copy.mat);
    return det;
}

void matrix_swap_rows_memcpy(Matrix *mat, size_t r1, size_t r2) {
    if (r1 == r2) return;
    static double *tmp_buf = NULL;
    static size_t tmp_size = 0;
    size_t n = mat->cols * sizeof(double);
    double *row1 = &mat->mat[r1 * mat->cols];
    double *row2 = &mat->mat[r2 * mat->cols];
    if (tmp_size < n) { free(tmp_buf); tmp_buf = malloc(n); tmp_size = n; }
    if (!tmp_buf) { double *t = malloc(n); memcpy(t, row1, n); memcpy(row1, row2, n); memcpy(row2, t, n); free(t); return; }
    memcpy(tmp_buf, row1, n);
    memcpy(row1, row2, n);
    memcpy(row2, tmp_buf, n);
}

void matrix_swap_rows_xor(Matrix *mat, size_t r1, size_t r2) {
    if (r1 == r2) return;
    double *row1 = &mat->mat[r1 * mat->cols];
    double *row2 = &mat->mat[r2 * mat->cols];
    size_t n = mat->cols * sizeof(double) / sizeof(uint64_t);
    uint64_t *p1 = (uint64_t *)row1;
    uint64_t *p2 = (uint64_t *)row2;
    for (size_t j = 0; j < n; j++) {
        p1[j] ^= p2[j];
        p2[j] ^= p1[j];
        p1[j] ^= p2[j];
    }
}

void matrix_scale_row(Matrix *mat, size_t r, double scalar) {
    double *restrict row = &mat->mat[r * mat->cols];
    for (size_t j = 0; j < mat->cols; j++)
        row[j] *= scalar;
}

void matrix_add_row(Matrix *mat, size_t dest, size_t src, double scalar) {
    double *restrict d = &mat->mat[dest * mat->cols];
    double *restrict s = &mat->mat[src * mat->cols];
    for (size_t j = 0; j < mat->cols; j++)
        d[j] += scalar * s[j];
}

int matrix_inverse(Matrix *mat, Matrix *result) {
    if (mat->cols != mat->rows) return 1;

    const size_t n = mat->cols;
    double *aug = calloc(n * 2 * n, sizeof(double));
    if (!aug) return 1;

    // 预分配栈上临时行缓冲区（避免每次 swap 都要 malloc）
    double *tmp_row = malloc(2 * n * sizeof(double));
    if (!tmp_row) { free(aug); return 1; }

    // 增广矩阵 [A | I]
    for (size_t i = 0; i < n; i++) {
        memcpy(&aug[i * 2 * n], &mat->mat[i * n], n * sizeof(double));
        aug[i * 2 * n + n + i] = 1.0;
    }

    const size_t stride = 2 * n;

    // Gauss-Jordan 消元
    for (size_t k = 0; k < n; k++) {
        // 选主元
        size_t max_row = k;
        double max_val = fabs(aug[k * stride + k]);
        for (size_t i = k + 1; i < n; i++) {
            double val = fabs(aug[i * stride + k]);
            if (val > max_val) { max_val = val; max_row = i; }
        }

        if (max_val < 1e-12) { free(aug); free(tmp_row); return 1; }

        // 行交换（复用 tmp_row）
        if (max_row != k) {
            double *row_k = &aug[k * stride];
            double *row_max = &aug[max_row * stride];
            memcpy(tmp_row, row_k, stride * sizeof(double));
            memcpy(row_k, row_max, stride * sizeof(double));
            memcpy(row_max, tmp_row, stride * sizeof(double));
        }

        // 归一化：乘以 1/pivot 而非每次除以 pivot
        double pivot_inv = 1.0 / aug[k * stride + k];
        double *row_k_ptr = &aug[k * stride];
        for (size_t j = 0; j < stride; j++)
            row_k_ptr[j] *= pivot_inv;

        // 消去第 k 列
        for (size_t i = 0; i < n; i++) {
            if (i == k) continue;
            double factor = aug[i * stride + k];
            if (fabs(factor) < 1e-15) continue;
            double *row_i = &aug[i * stride];
            for (size_t j = 0; j < stride; j++)
                row_i[j] -= factor * row_k_ptr[j];
        }
    }

    // 提取右侧 A⁻¹
    if (matrix_init(result, n, n)) { free(aug); free(tmp_row); return 1; }
    for (size_t i = 0; i < n; i++)
        memcpy(&result->mat[i * n], &aug[i * stride + n], n * sizeof(double));

    free(aug);
    free(tmp_row);
    return 0;
}

// ========== 基准测试 ==========

static volatile double sink_d;
static volatile void *sink_p;

void benchmark_one(const char *name, void (*func)(void), int iterations) {
    // warmup
    for (int i = 0; i < 3; i++) func();
    // timed
    clock_t start = clock();
    for (int i = 0; i < iterations; i++) func();
    clock_t end = clock();
    double t = (double)(end - start) / CLOCKS_PER_SEC;
    printf("%-24s %8.4f s\n", name, t);
}

void bench_multiply(void) {
    static Matrix a = {0}, b = {0}, c = {0};
    static int init = 0;
    if (!init) { matrix_init(&a, 500, 500); matrix_init(&b, 500, 500); matrix_init(&c, 500, 500); srand(42); for (size_t i = 0; i < 500*500; i++) { a.mat[i] = (double)rand()/RAND_MAX; b.mat[i] = (double)rand()/RAND_MAX; } init = 1; }
    matrix_multiply(&a, &b, &c);
    sink_p = c.mat;
}

void bench_transpose(void) {
    static Matrix m = {0};
    static Matrix r = {0};
    static int init = 0;
    if (!init) { matrix_init(&m, 5000, 5000); srand(42); for (size_t i = 0; i < 5000*5000; i++) m.mat[i] = (double)rand()/RAND_MAX; init = 1; }
    matrix_transpose(&m, &r);
    sink_p = r.mat;
}

void bench_transpose_inplace(void) {
    static Matrix m = {0};
    static int init = 0;
    if (!init) { matrix_init(&m, 5000, 5000); srand(42); for (size_t i = 0; i < 5000*5000; i++) m.mat[i] = (double)rand()/RAND_MAX; init = 1; }
    matrix_transpose_inplace(&m);
    sink_p = m.mat;
}

void bench_det_small(void) {
    static Matrix m = {0};
    static int init = 0;
    if (!init) { matrix_init(&m, 100, 100); srand(42); for (size_t i = 0; i < 100*100; i++) m.mat[i] = (double)rand()/RAND_MAX; init = 1; }
    sink_d = matrix_det(&m);
}

void bench_det_large(void) {
    static Matrix m = {0};
    static int init = 0;
    if (!init) { matrix_init(&m, 500, 500); srand(42); for (size_t i = 0; i < 500*500; i++) m.mat[i] = (double)rand()/RAND_MAX; init = 1; }
    sink_d = matrix_det(&m);
}

void bench_inverse_small(void) {
    static Matrix m = {0}, r = {0};
    static int init = 0;
    if (!init) { matrix_init(&m, 100, 100); srand(42); for (size_t i = 0; i < 100*100; i++) m.mat[i] = (double)rand()/RAND_MAX; init = 1; }
    matrix_inverse(&m, &r);
    sink_p = r.mat;
}

void bench_inverse_large(void) {
    static Matrix m = {0}, r = {0};
    static int init = 0;
    if (!init) { matrix_init(&m, 300, 300); srand(42); for (size_t i = 0; i < 300*300; i++) m.mat[i] = (double)rand()/RAND_MAX; init = 1; }
    matrix_inverse(&m, &r);
    sink_p = r.mat;
}

void bench_swap_memcpy(void) {
    static Matrix m = {0};
    static int init = 0;
    if (!init) { matrix_init(&m, 5000, 5000); srand(42); for (size_t i = 0; i < 5000*5000; i++) m.mat[i] = (double)rand()/RAND_MAX; init = 1; }
    matrix_swap_rows_memcpy(&m, 0, 2500);
    matrix_swap_rows_memcpy(&m, 2500, 0);
    sink_p = m.mat;
}

void bench_swap_xor(void) {
    static Matrix m = {0};
    static int init = 0;
    if (!init) { matrix_init(&m, 5000, 5000); srand(42); for (size_t i = 0; i < 5000*5000; i++) m.mat[i] = (double)rand()/RAND_MAX; init = 1; }
    matrix_swap_rows_xor(&m, 0, 2500);
    matrix_swap_rows_xor(&m, 2500, 0);
    sink_p = m.mat;
}

void bench_scale_row(void) {
    static Matrix m = {0};
    static int init = 0;
    if (!init) { matrix_init(&m, 5000, 5000); srand(42); for (size_t i = 0; i < 5000*5000; i++) m.mat[i] = (double)rand()/RAND_MAX; init = 1; }
    matrix_scale_row(&m, 0, 2.5);
    matrix_scale_row(&m, 2500, 0.5);
    sink_p = m.mat;
}

void bench_add_row(void) {
    static Matrix m = {0};
    static int init = 0;
    if (!init) { matrix_init(&m, 5000, 5000); srand(42); for (size_t i = 0; i < 5000*5000; i++) m.mat[i] = (double)rand()/RAND_MAX; init = 1; }
    matrix_add_row(&m, 1, 0, 1.5);
    matrix_add_row(&m, 0, 1, -1.5);
    sink_p = m.mat;
}

int main(void) {
    printf("=== Matrix Algorithm Benchmarks ===\n\n");
    printf("%-24s %s\n", "Operation", "Time");
    printf("%-24s %s\n", "---------", "----");

    benchmark_one("multiply (500x500)",         bench_multiply,          10);
    benchmark_one("transpose (5000x5000)",       bench_transpose,         10);
    benchmark_one("transpose_inplace(5000x5000)",bench_transpose_inplace, 10);
    benchmark_one("det (100x100)",               bench_det_small,        100);
    benchmark_one("det (500x500)",               bench_det_large,         10);
    benchmark_one("inverse (100x100)",           bench_inverse_small,     10);
    benchmark_one("inverse (300x300)",            bench_inverse_large,     10);
    benchmark_one("swap_memcpy (5000x5000)",     bench_swap_memcpy,      100);
    benchmark_one("swap_xor (5000x5000)",         bench_swap_xor,         100);
    benchmark_one("scale_row (5000x5000)",       bench_scale_row,        100);
    benchmark_one("add_row (5000x5000)",          bench_add_row,          100);

    return 0;
}