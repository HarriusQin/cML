#ifndef __C_LINEAR_ALGEBRA_H__
#define __C_LINEAR_ALGEBRA_H__

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
#include <math.h>

typedef struct Matrix
{
    double *mat;
    size_t rows;
    size_t cols;
} Matrix;

static int matrix_init(Matrix *mat, size_t n_rows, size_t n_cols);
static inline double* matrix_at(Matrix* self, size_t row, size_t col);
static inline double* matrix_transpose_at(Matrix* self, size_t row, size_t col);
static int matrix_multiply(const Matrix *restrict left, const Matrix *restrict right, Matrix *restrict product);
static int matrix_transpose_inplace(Matrix *mat);
static int matrix_transpose(const Matrix *restrict mat, Matrix *restrict result);
static double matrix_det(Matrix *mat);
static void matrix_swap_rows(Matrix *mat, size_t r1, size_t r2);
static void matrix_swap_rows_memcpy(Matrix *mat, size_t r1, size_t r2);
static void matrix_swap_rows_xor(Matrix *mat, size_t r1, size_t r2);
static void matrix_scale_row(Matrix *mat, size_t r, double scalar);
static void matrix_add_row(Matrix *mat, size_t dest, size_t src, double scalar);
static int matrix_inverse(Matrix *mat, Matrix *result);
static int matrix_add(const Matrix *restrict a, const Matrix *restrict b, Matrix *restrict result);
static int matrix_subtract(const Matrix *restrict a, const Matrix *restrict b, Matrix *restrict result);
static void free_matrix_data(Matrix* mat);
static void free_matrix(Matrix* mat);

// SVD: A = U * SIGMA * V^T
// U is m×m orthogonal, SIGMA is m×n diagonal (singular values), V is n×n orthogonal
// caller must free U, singular_values, V
typedef struct {
    Matrix *U;          // m×m orthogonal matrix
    double *singular_values;  // min(m,n) singular values
    Matrix *V;          // n×n orthogonal matrix
    size_t m, n;        // original matrix dimensions
} SVD_Result;

static int matrix_svd(const Matrix *mat, SVD_Result *result);
static void free_svd_result(SVD_Result *result);

// Pseudoinverse using SVD: A^+ = V * SIGMA^+ * U^T
// SIGMA^+: reciprocal of non-zero singular values, transposed
static int matrix_pseudoinverse(const Matrix *mat, Matrix *result);
static int matrix_solve_svd(const Matrix *A, const Matrix *b, Matrix *x);

// Normal equation solver using SVD (more stable for ill-conditioned matrices)
// theta = (X^T X)^{-1} X^T y  using pseudoinverse: theta = X^+ y = V * SIGMA^+ * U^T * y
static int normal_equation_svd(const Matrix *X, const Matrix *y, Matrix *theta);

#ifdef LINEAR_ALGEBRA_IMPLEMENTATION

static int matrix_init(Matrix *mat, size_t n_rows, size_t n_cols) {
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

static int matrix_multiply(const Matrix *restrict left, const Matrix *restrict right, Matrix *restrict product) {
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

static int matrix_transpose_inplace(Matrix *mat) {
    if (mat->rows != mat->cols) return 1;
    for (size_t i = 0; i < mat->rows; i++)
        for (size_t j = i + 1; j < mat->cols; j++) {
            double tmp = mat->mat[i * mat->cols + j];
            mat->mat[i * mat->cols + j] = mat->mat[j * mat->cols + i];
            mat->mat[j * mat->cols + i] = tmp;
        }
    return 0;
}

static int matrix_transpose(const Matrix *restrict mat, Matrix *restrict result) {
    if (matrix_init(result, mat->cols, mat->rows)) return 1;
    for (size_t i = 0; i < mat->rows; i++)
        for (size_t j = 0; j < mat->cols; j++)
            result->mat[j * result->cols + i] = mat->mat[i * mat->cols + j];
    return 0;
}

static double matrix_det(Matrix *mat) {
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

static void matrix_swap_rows(Matrix *mat, size_t r1, size_t r2) {
    if (r1 == r2) return;
    double *row1 = &mat->mat[r1 * mat->cols];
    double *row2 = &mat->mat[r2 * mat->cols];
    for (size_t j = 0; j < mat->cols; j++) {
        double tmp = row1[j];
        row1[j] = row2[j];
        row2[j] = tmp;
    }
}

static void matrix_swap_rows_memcpy(Matrix *mat, size_t r1, size_t r2) {
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

static void matrix_swap_rows_xor(Matrix *mat, size_t r1, size_t r2) {
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

static void matrix_scale_row(Matrix *mat, size_t r, double scalar) {
    double *restrict row = &mat->mat[r * mat->cols];
    for (size_t j = 0; j < mat->cols; j++)
        row[j] *= scalar;
}

static void matrix_add_row(Matrix *mat, size_t dest, size_t src, double scalar) {
    double *restrict d = &mat->mat[dest * mat->cols];
    double *restrict s = &mat->mat[src * mat->cols];
    for (size_t j = 0; j < mat->cols; j++)
        d[j] += scalar * s[j];
}

static int matrix_inverse(Matrix *mat, Matrix *result) {
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

// Frees only the matrix data (mat->mat), NOT the Matrix struct itself.
// The Matrix struct is stack-allocated in temporary variables and auto-freed.
// For heap-allocated Matrix* (via malloc), call this then free(mat).
static void free_matrix_data(Matrix* mat) {
    if (!mat) return;
    if (mat->mat)
        free(mat->mat);
    mat->mat = NULL;
    mat->rows = mat->cols = 0;
}

// Convenience: frees matrix data AND the Matrix struct itself (for heap-allocated Matrix*)
static void free_matrix(Matrix* mat) {
    free_matrix_data(mat);
    free(mat);
}

static int matrix_add(const Matrix *restrict a, const Matrix *restrict b, Matrix *restrict result) {
    if (a->rows != b->rows || a->cols != b->cols) return 1;
    if (matrix_init(result, a->rows, a->cols)) return 1;
    double *restrict r = result->mat;
    double *restrict A = a->mat;
    double *restrict B = b->mat;
    size_t n = a->rows * a->cols;
    for (size_t i = 0; i < n; i++)
        r[i] = A[i] + B[i];
    return 0;
}

static int matrix_subtract(const Matrix *restrict a, const Matrix *restrict b, Matrix *restrict result) {
    if (a->rows != b->rows || a->cols != b->cols) return 1;
    if (matrix_init(result, a->rows, a->cols)) return 1;
    double *restrict r = result->mat;
    double *restrict A = a->mat;
    double *restrict B = b->mat;
    size_t n = a->rows * a->cols;
    for (size_t i = 0; i < n; i++)
        r[i] = A[i] - B[i];
    return 0;
}

// Compute Householder reflection: v = x - beta*e1, where Q = I - beta*v*v^T
static void householder_vector(double *x, size_t n, double *beta, double *v) {
    double sigma = 0.0;
    for (size_t i = 1; i < n; i++)
        sigma += x[i] * x[i];

    v[0] = 1.0;
    for (size_t i = 1; i < n; i++)
        v[i] = x[i];

    if (sigma < 1e-15) {
        *beta = 0.0;
    } else {
        double alpha = sqrt(x[0] * x[0] + sigma);
        if (x[0] <= 0)
            v[0] = x[0] - alpha;
        else
            v[0] = -sigma / (x[0] + alpha);

        *beta = 2.0 / (v[0] * v[0] + sigma);
    }
}

// Apply Householder reflection from left: A = (I - beta*v*v^T) * A
static void apply_householder_left(double *v, double beta, double *A, size_t m, size_t n, size_t col_offset) {
    for (size_t j = 0; j < n; j++) {
        double sum = 0.0;
        for (size_t i = 0; i < m; i++)
            sum += v[i] * A[i * n + j];
        sum *= beta;
        for (size_t i = 0; i < m; i++)
            A[i * n + j] -= sum * v[i];
    }
    (void)col_offset;
}

// Apply Householder reflection from right: A = A * (I - beta*v*v^T)
static void apply_householder_right(double *A, double *v, double beta, size_t m, size_t n, size_t row_offset) {
    for (size_t i = 0; i < m; i++) {
        double sum = 0.0;
        for (size_t j = 0; j < n; j++)
            sum += A[i * n + j] * v[j];
        sum *= beta;
        for (size_t j = 0; j < n; j++)
            A[i * n + j] -= sum * v[j];
    }
    (void)row_offset;
}

// One-sided Jacobi SVD for m>=n. Computes A = U * SIGMA * V^T
// U overwrites A (left singular vectors stored as columns)
// SIGMA stored as singular values
// V stored as right singular vectors
static int jacobi_svd(double *A, size_t m, size_t n, double *singular_values, double *V) {
    const size_t max_iter = 30;
    const double tol = 1e-14;

    // Initialize V to identity
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++)
            V[i * n + j] = (i == j) ? 1.0 : 0.0;
    }

    // Copy A to work matrix
    double *B = malloc(m * n * sizeof(double));
    if (!B) return 1;
    memcpy(B, A, m * n * sizeof(double));

    for (size_t iter = 0; iter < max_iter; iter++) {
        double off_norm = 0.0;
        for (size_t i = 0; i < n; i++) {
            for (size_t j = i + 1; j < n; j++) {
                // Compute 2x2 SVD of columns i and j of B
                double sigma_ii = 0.0, sigma_ij = 0.0, sigma_jj = 0.0;
                for (size_t k = 0; k < m; k++) {
                    sigma_ii += B[k * n + i] * B[k * n + i];
                    sigma_ij += B[k * n + i] * B[k * n + j];
                    sigma_jj += B[k * n + j] * B[k * n + j];
                }

                if (fabs(sigma_ij) < tol * sqrt(sigma_ii * sigma_jj))
                    continue;

                off_norm += sigma_ij * sigma_ij;

                // Compute rotation angles for 2x2 symmetric matrix [sigma_ii, sigma_ij; sigma_ij, sigma_jj]
                double tau = (sigma_jj - sigma_ii) / (2.0 * sigma_ij);
                double t;
                if (tau >= 0)
                    t = 1.0 / (tau + sqrt(1.0 + tau * tau));
                else
                    t = 1.0 / (tau - sqrt(1.0 + tau * tau));
                double c = 1.0 / sqrt(1.0 + t * t);
                double s = c * t;

                // Apply rotation to B (columns i and j): B(:, [i j]) = B(:, [i j]) * R
                for (size_t k = 0; k < m; k++) {
                    double bi = B[k * n + i];
                    double bj = B[k * n + j];
                    B[k * n + i] = c * bi - s * bj;
                    B[k * n + j] = s * bi + c * bj;
                }

                // Apply rotation to V (right singular vectors): V(:, [i j]) = V(:, [i j]) * R
                for (size_t k = 0; k < n; k++) {
                    double vi = V[k * n + i];
                    double vj = V[k * n + j];
                    V[k * n + i] = c * vi - s * vj;
                    V[k * n + j] = s * vi + c * vj;
                }
            }
        }

        if (off_norm < tol * tol)
            break;
    }

    // Extract singular values (norms of columns of B) and normalize columns of B to get U
    for (size_t i = 0; i < n; i++) {
        double norm = 0.0;
        for (size_t k = 0; k < m; k++)
            norm += B[k * n + i] * B[k * n + i];
        singular_values[i] = sqrt(norm);
        if (singular_values[i] > 1e-15) {
            for (size_t k = 0; k < m; k++)
                A[k * n + i] = B[k * n + i] / singular_values[i];
        } else {
            for (size_t k = 0; k < m; k++)
                A[k * n + i] = 0.0;
        }
    }

    free(B);
    return 0;
}

static int matrix_svd(const Matrix *mat, SVD_Result *result) {
    if (!mat || !result) return 1;

    size_t m = mat->rows;
    size_t n = mat->cols;

    result->m = m;
    result->n = n;

    // Allocate U (m×n, not m×m, for economy SVD)
    result->U = calloc(1, sizeof(Matrix));
    if (matrix_init(result->U, m, n) != 0) return 1;

    // Copy input to U (will be overwritten with left singular vectors)
    memcpy(result->U->mat, mat->mat, m * n * sizeof(double));

    // Allocate singular values
    size_t k = (m < n) ? m : n;
    result->singular_values = calloc(k, sizeof(double));

    // Allocate V (n×n)
    result->V = calloc(1, sizeof(Matrix));
    if (matrix_init(result->V, n, n) != 0) {
        free(result->singular_values);
        return 1;
    }

    // For m < n case, we compute SVD of A^T instead and transpose
    if (m < n) {
        // Compute SVD of A^T (which is n×m)
        double *At = malloc(n * m * sizeof(double));
        for (size_t i = 0; i < m; i++)
            for (size_t j = 0; j < n; j++)
                At[j * m + i] = mat->mat[i * n + j];

        // SVD of At gives: At = U_t * SIGMA_t * V_t^T
        // So A = U_t * SIGMA_t * V_t^T (where U_t becomes V, V_t becomes U)
        double *V_work = malloc(m * m * sizeof(double));
        double *singular_work = malloc(m * sizeof(double));

        // Actually compute SVD of A^T
        // A^T is n×m, we want U_n×m, SIGMA_m×m, V_m×m
        // But for one-sided Jacobi we need m <= n, so work with transpose

        // For A^T (n×m with n>m), we'll compute using a modified approach
        // Copy At to U_work area
        double *U_work = malloc(n * m * sizeof(double));
        memcpy(U_work, At, n * m * sizeof(double));

        // Compute SVD of U_work (n×m), m <= n
        // This gives U_work = U_t * SIGMA_t * V_t^T where U_t is n×n, SIGMA_t is m×m diagonal, V_t is m×m
        // But we want A^T = U * SIGMA * V^T with U (n×m), SIGMA (m×m), V (m×m)

        // Actually simpler: just use the standard approach
        // Compute A^T A = V SIGMA^2 V^T using symmetric Jacobi
        double *ATA = calloc(m * m, sizeof(double));
        for (size_t i = 0; i < m; i++)
            for (size_t j = 0; j < m; j++)
                for (size_t k = 0; k < n; k++)
                    ATA[i * m + j] += At[k * m + i] * At[k * m + j];

        // Compute eigenvalues/eigenvectors of ATA using Jacobi
        // This gives V and SIGMA^2
        for (size_t i = 0; i < m; i++)
            for (size_t j = 0; j < m; j++)
                V_work[i * m + j] = ATA[i * m + j];

        // Jacobi eigenvalue algorithm for symmetric matrix
        for (size_t iter = 0; iter < 30; iter++) {
            double off = 0.0;
            for (size_t p = 0; p < m; p++)
                for (size_t q = p + 1; q < m; q++)
                    off += V_work[p * m + q] * V_work[p * m + q];

            if (off < 1e-15) break;

            for (size_t p = 0; p < m; p++) {
                for (size_t q = p + 1; q < m; q++) {
                    double apq = V_work[p * m + q];
                    double app = V_work[p * m + p];
                    double aqq = V_work[q * m + q];

                    double tau = (aqq - app) / (2.0 * apq);
                    double t;
                    if (tau >= 0)
                        t = 1.0 / (tau + sqrt(1.0 + tau * tau));
                    else
                        t = 1.0 / (tau - sqrt(1.0 + tau * tau));
                    double c = 1.0 / sqrt(1.0 + t * t);
                    double s = c * t;

                    double app_new = c * c * app + s * s * aqq + 2.0 * c * s * apq;
                    double aqq_new = s * s * app + c * c * aqq - 2.0 * c * s * apq;
                    V_work[p * m + p] = app_new;
                    V_work[q * m + q] = aqq_new;
                    V_work[p * m + q] = 0.0;
                    V_work[q * m + p] = 0.0;

                    for (size_t r = 0; r < m; r++) {
                        if (r == p || r == q) continue;
                        double apr = V_work[p * m + r];
                        double aqr = V_work[q * m + r];
                        V_work[p * m + r] = c * apr - s * aqr;
                        V_work[q * m + r] = s * apr + c * aqr;
                        V_work[r * m + p] = V_work[p * m + r];
                        V_work[r * m + q] = V_work[q * m + r];
                    }
                }
            }
        }

        // Extract singular values (sqrt of eigenvalues)
        for (size_t i = 0; i < m; i++)
            singular_work[i] = sqrt(fabs(V_work[i * m + i]));

        // V is eigenvectors of A^T A (stored in V_work), but we need n×n V
        // For m < n, V = [V_work; 0] not applicable - instead we compute V directly
        // Let's just use the simpler approach: compute V from eigenvectors of A^T A

        // Actually, we need to re-think. Let me use a simpler implementation.

        // For A (m×n) with m < n:
        // SVD of A gives U (m×m), SIGMA (m×n), V (n×n)
        // We can compute V as eigenvectors of A^T A

        free(ATA);
        free(U_work);
        free(V_work);
        free(singular_work);
        free(At);

        // Fall back to simple implementation
        result->singular_values = calloc(m, sizeof(double));
        result->V = calloc(1, sizeof(Matrix));
        if (matrix_init(result->V, m, m) != 0) return 1;

        // For now, let's compute U and SIGMA properly
        // Compute A^T A
        double *ATA2 = calloc(m * m, sizeof(double));
        for (size_t i = 0; i < m; i++)
            for (size_t j = 0; j < m; j++)
                for (size_t k = 0; k < n; k++)
                    ATA2[i * m + j] += mat->mat[k * n + i] * mat->mat[k * n + j];

        // Jacobi eigenvalues on ATA2 -> eigenvectors become V (first m columns)
        double *V_temp = malloc(m * m * sizeof(double));
        for (size_t i = 0; i < m; i++)
            for (size_t j = 0; j < m; j++)
                V_temp[i * m + j] = ATA2[i * m + j];

        for (size_t iter = 0; iter < 30; iter++) {
            double off = 0.0;
            for (size_t p = 0; p < m; p++)
                for (size_t q = p + 1; q < m; q++)
                    off += V_temp[p * m + q] * V_temp[p * m + q];
            if (off < 1e-15) break;

            for (size_t p = 0; p < m; p++) {
                for (size_t q = p + 1; q < m; q++) {
                    double apq = V_temp[p * m + q];
                    double app = V_temp[p * m + p];
                    double aqq = V_temp[q * m + q];
                    double tau = (aqq - app) / (2.0 * apq);
                    double t;
                    if (tau >= 0) t = 1.0 / (tau + sqrt(1.0 + tau * tau));
                    else t = 1.0 / (tau - sqrt(1.0 + tau * t));
                    double c = 1.0 / sqrt(1.0 + t * t);
                    double s = c * t;

                    double app_new = c * c * app + s * s * aqq + 2.0 * c * s * apq;
                    V_temp[p * m + p] = app_new;
                    V_temp[q * m + q] = s * s * app + c * c * aqq - 2.0 * c * s * apq;
                    V_temp[p * m + q] = 0.0;
                    V_temp[q * m + p] = 0.0;

                    for (size_t r = 0; r < m; r++) {
                        if (r == p || r == q) continue;
                        double apr = V_temp[p * m + r];
                        double aqr = V_temp[q * m + r];
                        V_temp[p * m + r] = c * apr - s * aqr;
                        V_temp[q * m + r] = s * apr + c * aqr;
                        V_temp[r * m + p] = V_temp[p * m + r];
                        V_temp[r * m + q] = V_temp[q * m + r];
                    }
                }
            }
        }

        for (size_t i = 0; i < m; i++)
            result->singular_values[i] = sqrt(fabs(V_temp[i * m + i]));

        // Copy V_temp to result->V (which is m×m for this case)
        for (size_t i = 0; i < m; i++)
            for (size_t j = 0; j < m; j++)
                result->V->mat[i * m + j] = V_temp[i * m + j];

        // U = A * V * SIGMA^+ (for m < n, SIGMA is m×m diagonal)
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < m; j++) {
                double sum = 0.0;
                for (size_t k = 0; k < n; k++)
                    sum += mat->mat[k * n + j] * V_temp[k * m + i];
                if (result->singular_values[i] > 1e-15)
                    result->U->mat[i * m + j] = sum / result->singular_values[i];
                else
                    result->U->mat[i * m + j] = 0.0;
            }
        }

        free(ATA2);
        free(V_temp);
        return 0;
    }

    // m >= n case: use one-sided Jacobi
    if (jacobi_svd(result->U->mat, m, n, result->singular_values, result->V->mat) != 0)
        return 1;

    return 0;
}

static void free_svd_result(SVD_Result *result) {
    if (!result) return;
    if (result->U) { free_matrix_data(result->U); free(result->U); }
    if (result->singular_values) free(result->singular_values);
    if (result->V) { free_matrix_data(result->V); free(result->V); }
}

static int matrix_pseudoinverse(const Matrix *mat, Matrix *result) {
    if (!mat || !result) return 1;

    SVD_Result svd = {0};
    if (matrix_svd(mat, &svd) != 0) return 1;

    size_t m = mat->rows;
    size_t n = mat->cols;
    size_t k = (m < n) ? m : n;

    // Compute pseudoinverse: A^+ = V * SIGMA^+ * U^T
    // SIGMA^+ is n×m: diagonal entries are 1/sigma_i for i < rank, 0 otherwise

    // U^T is n×m, SIGMA^+ is n×m (only diagonal), V is n×n
    // A^+ = V * (SIGMA^+ * U^T) = n×m result

    if (matrix_init(result, n, m) != 0) {
        free_svd_result(&svd);
        return 1;
    }

    // Compute SIGMA^+ * U^T
    double *SIGMA_plus_UT = calloc(n * m, sizeof(double));
    if (!SIGMA_plus_UT) {
        free_svd_result(&svd);
        free_matrix_data(result);
        return 1;
    }

    for (size_t i = 0; i < k; i++) {
        if (svd.singular_values[i] > 1e-12) {
            double sigma_inv = 1.0 / svd.singular_values[i];
            // SIGMA^+ * U^T: row i = sigma_i^-1 * row i of U^T = sigma_i^-1 * column i of U
            for (size_t j = 0; j < m; j++)
                SIGMA_plus_UT[i * m + j] = sigma_inv * svd.U->mat[j * k + i];
        }
    }

    // A^+ = V * SIGMA_plus_UT (n×n * n×m = n×m)
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            double sum = 0.0;
            for (size_t p = 0; p < n; p++)
                sum += svd.V->mat[i * n + p] * SIGMA_plus_UT[p * m + j];
            result->mat[i * m + j] = sum;
        }
    }

    free(SIGMA_plus_UT);
    free_svd_result(&svd);
    return 0;
}

// Solve Ax = b using SVD (for least squares when A is not full rank or ill-conditioned)
static int matrix_solve_svd(const Matrix *A, const Matrix *b, Matrix *x) {
    if (!A || !b || !x) return 1;
    if (b->cols != 1) return 1;

    SVD_Result svd = {0};
    if (matrix_svd(A, &svd) != 0) return 1;

    size_t m = A->rows;
    size_t n = A->cols;
    size_t k = (m < n) ? m : n;

    // x = V * SIGMA^+ * U^T * b
    // First compute c = U^T * b (k×1)
    double *c = calloc(k, sizeof(double));
    if (!c) {
        free_svd_result(&svd);
        return 1;
    }

    for (size_t i = 0; i < k; i++) {
        if (svd.singular_values[i] > 1e-12) {
            for (size_t j = 0; j < m; j++)
                c[i] += svd.U->mat[j * k + i] * b->mat[j];
            c[i] /= svd.singular_values[i];
        }
    }

    // x = V * c (n×k * k×1 = n×1)
    if (matrix_init(x, n, 1) != 0) {
        free(c);
        free_svd_result(&svd);
        return 1;
    }

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < k; j++)
            x->mat[i] += svd.V->mat[i * n + j] * c[j];
    }

    free(c);
    free_svd_result(&svd);
    return 0;
}

// Solve normal equation using SVD: theta = (X^T X)^{-1} X^T y = X^+ y
// This is more stable than direct inverse when X^T X is ill-conditioned or singular
static int normal_equation_svd(const Matrix *X, const Matrix *y, Matrix *theta) {
    if (!X || !y || !theta) return 1;
    if (X->rows != y->rows || y->cols != 1) return 1;

    // For overdetermined system (m > n), use pseudoinverse of X (n×m)
    // theta = X^+ y where X^+ is n×m, y is m×1, result is n×1
    Matrix X_pinv = {0};
    if (matrix_pseudoinverse(X, &X_pinv) != 0) return 1;

    // theta = X_pinv * y
    if (matrix_init(theta, X->cols, 1) != 0) {
        free_matrix_data(&X_pinv);
        return 1;
    }

    for (size_t i = 0; i < X->cols; i++) {
        theta->mat[i] = 0.0;
        for (size_t j = 0; j < X->rows; j++)
            theta->mat[i] += X_pinv.mat[i * X->rows + j] * y->mat[j];
    }

    free_matrix_data(&X_pinv);
    return 0;
}

#endif /* LINEAR_ALGEBRA_IMPLEMENTATION */

#endif /* __C_LINEAR_ALGEBRA_H__ */
