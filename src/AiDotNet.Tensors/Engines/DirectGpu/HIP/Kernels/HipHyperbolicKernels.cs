// Copyright (c) AiDotNet. All rights reserved.
// HIP kernels for hyperbolic (Poincare ball) neural network operations.

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

/// <summary>
/// HIP kernels for hyperbolic geometry operations used in hyperbolic neural networks.
/// Implements Poincare ball model operations with full forward and backward support.
/// </summary>
internal static class HipHyperbolicKernels
{
    public static string GetSource()
    {
        return @"
#include <hip/hip_runtime.h>
#include <math.h>

// Epsilon for division safety (preventing divide-by-zero)
#define EPSILON_DIV 1e-10f
// Epsilon for boundary clamping (float precision: 1.0f - 1e-6f != 1.0f)
#define EPSILON_BOUNDARY 1e-6f
#define MAX_NORM_FACTOR 0.99999f
#define MAX_DIM 128

// Helper to clamp dimension to prevent buffer overflow
__device__ __forceinline__ int safe_dim(int dim) {
    return (dim > MAX_DIM) ? MAX_DIM : dim;
}

// ===========================================================================
// HYPERBOLIC GEOMETRY HELPER FUNCTIONS
// ===========================================================================

__device__ float compute_norm_sq(const float* v, int dim) {
    int d = safe_dim(dim);
    float sum = 0.0f;
    for (int i = 0; i < d; i++) {
        sum += v[i] * v[i];
    }
    return sum;
}

__device__ float compute_norm(const float* v, int dim) {
    return sqrtf(compute_norm_sq(v, dim));
}

__device__ float dot_product(const float* a, const float* b, int dim) {
    int d = safe_dim(dim);
    float sum = 0.0f;
    for (int i = 0; i < d; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

__device__ float safe_arctanh(float x) {
    x = fmaxf(-1.0f + EPSILON_BOUNDARY, fminf(1.0f - EPSILON_BOUNDARY, x));
    return 0.5f * logf((1.0f + x) / (1.0f - x));
}

__device__ void project_to_ball(float* point, int dim, float c) {
    int d = safe_dim(dim);
    float maxNorm = MAX_NORM_FACTOR / sqrtf(c);
    float norm = compute_norm(point, d);
    if (norm > maxNorm) {
        float scale = maxNorm / norm;
        for (int i = 0; i < d; i++) {
            point[i] *= scale;
        }
    }
}

__device__ float conformal_factor(const float* x, int dim, float c) {
    float normSq = compute_norm_sq(x, dim);
    return 2.0f / fmaxf(1.0f - c * normSq, EPSILON_DIV);
}

__device__ void mobius_add(const float* x, const float* y, float* result, int dim, float c) {
    int d = safe_dim(dim);
    float xNormSq = compute_norm_sq(x, d);
    float yNormSq = compute_norm_sq(y, d);
    float xyDot = dot_product(x, y, d);

    float denom = 1.0f + 2.0f * c * xyDot + c * c * xNormSq * yNormSq;
    denom = fmaxf(fabsf(denom), EPSILON_DIV);

    float coeff1 = 1.0f + 2.0f * c * xyDot + c * yNormSq;
    float coeff2 = 1.0f - c * xNormSq;

    for (int i = 0; i < d; i++) {
        result[i] = (coeff1 * x[i] + coeff2 * y[i]) / denom;
    }
}

__device__ float poincare_distance(const float* x, const float* y, int dim, float c) {
    int d = safe_dim(dim);
    float negX[MAX_DIM];
    for (int i = 0; i < d; i++) {
        negX[i] = -x[i];
    }

    float diff[MAX_DIM];
    mobius_add(negX, y, diff, d, c);

    float diffNorm = compute_norm(diff, d);
    float sqrtC = sqrtf(c);
    float arg = sqrtC * diffNorm;
    arg = fminf(arg, 1.0f - EPSILON_BOUNDARY);

    return (2.0f / sqrtC) * safe_arctanh(arg);
}

__device__ void poincare_exp_map(const float* basePoint, const float* tangentVec, float* result, int dim, float c) {
    int d = safe_dim(dim);
    float vNorm = compute_norm(tangentVec, d);
    if (vNorm < EPSILON_DIV) {
        for (int i = 0; i < d; i++) {
            result[i] = basePoint[i];
        }
        return;
    }

    float xNormSq = compute_norm_sq(basePoint, d);
    float lambdaX = 2.0f / fmaxf(1.0f - c * xNormSq, EPSILON_DIV);

    float sqrtC = sqrtf(c);
    float scaledNorm = sqrtC * lambdaX * vNorm / 2.0f;
    scaledNorm = fminf(scaledNorm, 20.0f);
    float tanhVal = tanhf(scaledNorm);
    float coeff = tanhVal / (sqrtC * vNorm + EPSILON_DIV);

    float scaledV[MAX_DIM];
    for (int i = 0; i < d; i++) {
        scaledV[i] = coeff * tangentVec[i];
    }

    mobius_add(basePoint, scaledV, result, d, c);
    project_to_ball(result, d, c);
}

__device__ void poincare_log_map(const float* x, const float* y, float* result, int dim, float c) {
    int d = safe_dim(dim);
    float negX[MAX_DIM];
    for (int i = 0; i < d; i++) {
        negX[i] = -x[i];
    }

    float diff[MAX_DIM];
    mobius_add(negX, y, diff, d, c);

    float diffNorm = compute_norm(diff, d);
    if (diffNorm < EPSILON_DIV) {
        for (int i = 0; i < d; i++) {
            result[i] = 0.0f;
        }
        return;
    }

    float xNormSq = compute_norm_sq(x, d);
    float lambdaX = 2.0f / fmaxf(1.0f - c * xNormSq, EPSILON_DIV);

    float sqrtC = sqrtf(c);
    float arg = sqrtC * diffNorm;
    arg = fminf(arg, 1.0f - EPSILON_BOUNDARY);
    float arctanhVal = safe_arctanh(arg);

    float coeff = (2.0f / (lambdaX * sqrtC)) * arctanhVal / diffNorm;

    for (int i = 0; i < d; i++) {
        result[i] = coeff * diff[i];
    }
}

// ===========================================================================
// HYPERBOLIC LINEAR FORWARD KERNEL
// ===========================================================================

extern ""C"" __global__ void hyperbolic_linear_forward(
    const float* input,
    const float* weights,
    const float* biases,
    float* output,
    int batch, int inputFeatures, int outputFeatures, float curvature)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int b = gid / outputFeatures;
    int o = gid % outputFeatures;

    if (b >= batch || o >= outputFeatures) return;

    float c = fabsf(curvature);
    if (c < EPSILON_DIV) c = 1.0f;

    // Clamp inputFeatures to MAX_DIM to prevent buffer overflow
    int safeDim = safe_dim(inputFeatures);

    float projectedInput[MAX_DIM];
    for (int i = 0; i < safeDim; i++) {
        projectedInput[i] = input[b * inputFeatures + i];
    }
    project_to_ball(projectedInput, safeDim, c);

    float weightVec[MAX_DIM];
    float biasVec[MAX_DIM];
    for (int i = 0; i < safeDim; i++) {
        weightVec[i] = weights[o * inputFeatures + i];
        biasVec[i] = biases[o * inputFeatures + i];
    }

    float origin[MAX_DIM];
    for (int i = 0; i < safeDim; i++) {
        origin[i] = 0.0f;
    }

    float weightPoint[MAX_DIM];
    poincare_exp_map(origin, weightVec, weightPoint, safeDim, c);

    float transformed[MAX_DIM];
    mobius_add(projectedInput, weightPoint, transformed, safeDim, c);

    project_to_ball(biasVec, safeDim, c);

    float withBias[MAX_DIM];
    mobius_add(transformed, biasVec, withBias, safeDim, c);

    float distance = poincare_distance(origin, withBias, safeDim, c);
    output[b * outputFeatures + o] = distance;
}

// ===========================================================================
// HYPERBOLIC LINEAR BACKWARD KERNELS
// ===========================================================================

extern ""C"" __global__ void hyperbolic_linear_backward_input(
    const float* gradOutput,
    const float* input,
    const float* weights,
    float* gradInput,
    int batch, int inputFeatures, int outputFeatures, float curvature)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int b = gid / inputFeatures;
    int i = gid % inputFeatures;

    if (b >= batch || i >= inputFeatures) return;

    // Guard against buffer overflow when inputFeatures > MAX_DIM
    // Skip gradient computation for dimensions beyond MAX_DIM
    if (i >= MAX_DIM) {
        gradInput[b * inputFeatures + i] = 0.0f;
        return;
    }

    float c = fabsf(curvature);
    if (c < EPSILON_DIV) c = 1.0f;

    // Load and project input for this batch element (clamped to MAX_DIM)
    int safeDim = safe_dim(inputFeatures);
    float projectedInput[MAX_DIM];
    for (int j = 0; j < safeDim; j++) {
        projectedInput[j] = input[b * inputFeatures + j];
    }
    project_to_ball(projectedInput, safeDim, c);

    float squaredNorm = compute_norm_sq(projectedInput, safeDim);
    float cNormSquared = c * squaredNorm;
    float oneMinusCNorm = 1.0f - cNormSquared;
    float conformalFactor = (oneMinusCNorm * oneMinusCNorm) / 4.0f;

    float gradSum = 0.0f;
    for (int o = 0; o < outputFeatures; o++) {
        float gradOut = gradOutput[b * outputFeatures + o];
        float riemannianGrad = gradOut * conformalFactor;
        float weight = weights[o * inputFeatures + i];
        gradSum += riemannianGrad * weight;
    }

    gradInput[b * inputFeatures + i] = gradSum;
}

// Parallelized weight gradient kernel - each thread handles one (batch, output, input) element
// and uses atomicAdd for accumulation. Caller must zero gradWeights before invoking.
extern ""C"" __global__ void hyperbolic_linear_backward_weights(
    const float* gradOutput,
    const float* input,
    float* gradWeights,
    int batch, int inputFeatures, int outputFeatures, float curvature)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batch * outputFeatures * inputFeatures;

    if (gid >= totalElements) return;

    // Decompose gid into (b, o, i)
    int b = gid / (outputFeatures * inputFeatures);
    int remainder = gid % (outputFeatures * inputFeatures);
    int o = remainder / inputFeatures;
    int i = remainder % inputFeatures;

    // Guard against buffer overflow when inputFeatures > MAX_DIM
    // projectedInput is a fixed-size local array of MAX_DIM elements
    if (i >= MAX_DIM) return;

    float c = fabsf(curvature);
    if (c < EPSILON_DIV) c = 1.0f;

    // Load and project input for this batch element (clamped to MAX_DIM)
    int safeDim = safe_dim(inputFeatures);
    float projectedInput[MAX_DIM];
    for (int j = 0; j < safeDim; j++) {
        projectedInput[j] = input[b * inputFeatures + j];
    }
    project_to_ball(projectedInput, safeDim, c);

    float squaredNorm = compute_norm_sq(projectedInput, safeDim);
    float cNormSquared = c * squaredNorm;
    float oneMinusCNorm = 1.0f - cNormSquared;
    float conformalFactor = (oneMinusCNorm * oneMinusCNorm) / 4.0f;

    float gradOut = gradOutput[b * outputFeatures + o];
    float riemannianGrad = gradOut * conformalFactor;

    // Use atomic add to accumulate gradient from all batch elements
    // Note: i is already guarded to be < MAX_DIM above
    atomicAdd(&gradWeights[o * inputFeatures + i], riemannianGrad * projectedInput[i]);
}

// Parallelized bias gradient kernel - each thread handles one (batch, output, input) element
// and uses atomicAdd for accumulation. Caller must zero gradBiases before invoking.
extern ""C"" __global__ void hyperbolic_linear_backward_biases(
    const float* gradOutput,
    const float* input,
    float* gradBiases,
    int batch, int inputFeatures, int outputFeatures, float curvature)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batch * outputFeatures * inputFeatures;

    if (gid >= totalElements) return;

    // Decompose gid into (b, o, i)
    int b = gid / (outputFeatures * inputFeatures);
    int remainder = gid % (outputFeatures * inputFeatures);
    int o = remainder / inputFeatures;
    int i = remainder % inputFeatures;

    // Guard against buffer overflow when inputFeatures > MAX_DIM
    if (i >= MAX_DIM) return;

    float c = fabsf(curvature);
    if (c < EPSILON_DIV) c = 1.0f;

    // Load and project input for this batch element (clamped to MAX_DIM)
    int safeDim = safe_dim(inputFeatures);
    float projectedInput[MAX_DIM];
    for (int j = 0; j < safeDim; j++) {
        projectedInput[j] = input[b * inputFeatures + j];
    }
    project_to_ball(projectedInput, safeDim, c);

    float squaredNorm = compute_norm_sq(projectedInput, safeDim);
    float cNormSquared = c * squaredNorm;
    float oneMinusCNorm = 1.0f - cNormSquared;
    float conformalFactor = (oneMinusCNorm * oneMinusCNorm) / 4.0f;

    float gradOut = gradOutput[b * outputFeatures + o];
    float riemannianGrad = gradOut * conformalFactor;

    // Use atomic add to accumulate gradient from all batch elements
    // Bias gradient is distributed across input features
    atomicAdd(&gradBiases[o * inputFeatures + i], riemannianGrad / (float)inputFeatures);
}

extern ""C"" __global__ void hyperbolic_mobius_add_backward(
    const float* gradOutput,
    const float* x,
    const float* y,
    float* gradX,
    float* gradY,
    int size, int dim, float curvature)
{
    // Each thread computes gradient for one batch element, one dimension
    // But must accumulate contributions from ALL output dimensions (full Jacobian)
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = gid / dim;
    int i = gid % dim;  // Input dimension we're computing gradient for

    int safeDim = safe_dim(dim);
    if (batch >= size / dim || i >= safeDim) return;

    float c = fabsf(curvature);
    if (c < EPSILON_DIV) c = 1.0f;

    int baseIdx = batch * dim;

    // Load local copies of x and y
    float xLocal[MAX_DIM], yLocal[MAX_DIM], gradOutLocal[MAX_DIM];
    for (int k = 0; k < safeDim; k++) {
        xLocal[k] = x[baseIdx + k];
        yLocal[k] = y[baseIdx + k];
        gradOutLocal[k] = gradOutput[baseIdx + k];
    }

    // Precompute scalar values
    float xNormSq = compute_norm_sq(xLocal, safeDim);
    float yNormSq = compute_norm_sq(yLocal, safeDim);
    float xyDot = dot_product(xLocal, yLocal, safeDim);

    float denom = 1.0f + 2.0f * c * xyDot + c * c * xNormSq * yNormSq;
    denom = fmaxf(fabsf(denom), EPSILON_DIV);
    float denomSq = denom * denom;

    float coeff1 = 1.0f + 2.0f * c * xyDot + c * yNormSq;
    float coeff2 = 1.0f - c * xNormSq;

    // Compute full Jacobian-vector product: gradX[i] = sum_j(gradOut[j] * d(result[j])/d(x[i]))
    // result[j] = (coeff1 * x[j] + coeff2 * y[j]) / denom
    // d(result[j])/d(x[i]) involves:
    //   - Direct term when j==i: d(coeff1*x[i])/d(x[i]) = coeff1 + x[i]*d(coeff1)/d(x[i])
    //   - Cross terms from d(coeff1)/d(x[i]), d(coeff2)/d(x[i]), d(denom)/d(x[i])

    // Derivatives of scalar terms w.r.t. x[i]:
    float dXNormSq_dxi = 2.0f * xLocal[i];
    float dXYDot_dxi = yLocal[i];
    float dCoeff1_dxi = 2.0f * c * dXYDot_dxi;  // d(1 + 2c*xyDot + c*yNormSq)/d(x[i])
    float dCoeff2_dxi = -c * dXNormSq_dxi;       // d(1 - c*xNormSq)/d(x[i])
    float dDenom_dxi = 2.0f * c * dXYDot_dxi + c * c * dXNormSq_dxi * yNormSq;

    float gradXi = 0.0f;
    for (int j = 0; j < safeDim; j++) {
        // result[j] = (coeff1 * x[j] + coeff2 * y[j]) / denom
        float numerator_j = coeff1 * xLocal[j] + coeff2 * yLocal[j];

        // d(numerator_j)/d(x[i]):
        float dNumerator_dxi = dCoeff1_dxi * xLocal[j] + dCoeff2_dxi * yLocal[j];
        if (j == i) {
            dNumerator_dxi += coeff1;  // Direct term: d(coeff1 * x[j])/d(x[i]) when j==i
        }

        // d(result[j])/d(x[i]) using quotient rule
        float dResult_dxi = (dNumerator_dxi * denom - numerator_j * dDenom_dxi) / denomSq;
        gradXi += gradOutLocal[j] * dResult_dxi;
    }
    gradX[baseIdx + i] = gradXi;

    // Similar for gradY[i] = sum_j(gradOut[j] * d(result[j])/d(y[i]))
    float dYNormSq_dyi = 2.0f * yLocal[i];
    float dXYDot_dyi = xLocal[i];
    float dCoeff1_dyi = 2.0f * c * dXYDot_dyi + c * dYNormSq_dyi;  // d(1 + 2c*xyDot + c*yNormSq)/d(y[i])
    float dDenom_dyi = 2.0f * c * dXYDot_dyi + c * c * xNormSq * dYNormSq_dyi;

    float gradYi = 0.0f;
    for (int j = 0; j < safeDim; j++) {
        float numerator_j = coeff1 * xLocal[j] + coeff2 * yLocal[j];

        // d(numerator_j)/d(y[i]): coeff2 doesn't depend on y, but coeff1 does
        float dNumerator_dyi = dCoeff1_dyi * xLocal[j];
        if (j == i) {
            dNumerator_dyi += coeff2;  // Direct term: d(coeff2 * y[j])/d(y[i]) when j==i
        }

        float dResult_dyi = (dNumerator_dyi * denom - numerator_j * dDenom_dyi) / denomSq;
        gradYi += gradOutLocal[j] * dResult_dyi;
    }
    gradY[baseIdx + i] = gradYi;
}

extern ""C"" __global__ void hyperbolic_exp_map_backward(
    const float* gradOutput,
    const float* basePoint,
    const float* tangentVec,
    float* gradTangent,
    int size, int dim, float curvature)
{
    // Each thread computes gradient for one input dimension 'i'
    // Must compute full Jacobian-vector product over all output dimensions
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = gid / dim;
    int i = gid % dim;  // Input dimension we're computing gradient for

    int safeDim = safe_dim(dim);
    if (batch >= size / dim || i >= safeDim) return;

    float c = fabsf(curvature);
    if (c < EPSILON_DIV) c = 1.0f;

    int baseIdx = batch * dim;

    // Load local copies
    float vLocal[MAX_DIM], xLocal[MAX_DIM], gradOutLocal[MAX_DIM];
    for (int k = 0; k < safeDim; k++) {
        vLocal[k] = tangentVec[baseIdx + k];
        xLocal[k] = basePoint[baseIdx + k];
        gradOutLocal[k] = gradOutput[baseIdx + k];
    }

    float vNorm = compute_norm(vLocal, safeDim);
    if (vNorm < EPSILON_DIV) {
        // When tangent is zero, exp_map returns basePoint, gradient is identity
        gradTangent[baseIdx + i] = gradOutLocal[i];
        return;
    }

    float xNormSq = compute_norm_sq(xLocal, safeDim);
    float lambdaX = 2.0f / fmaxf(1.0f - c * xNormSq, EPSILON_DIV);
    float sqrtC = sqrtf(c);

    float scaledNorm = sqrtC * lambdaX * vNorm / 2.0f;
    scaledNorm = fminf(scaledNorm, 20.0f);

    float tanhVal = tanhf(scaledNorm);
    float sech2 = 1.0f - tanhVal * tanhVal;
    float coeff = tanhVal / (sqrtC * vNorm + EPSILON_DIV);

    // Compute scaledV = coeff * tangentVec
    float scaledV[MAX_DIM];
    for (int k = 0; k < safeDim; k++) {
        scaledV[k] = coeff * vLocal[k];
    }

    // d(vNorm)/d(v[i]) = v[i] / vNorm
    float dVNorm_dvi = vLocal[i] / (vNorm + EPSILON_DIV);
    float dScaledNorm_dvi = sqrtC * lambdaX * dVNorm_dvi / 2.0f;

    // d(coeff)/d(v[i]) using quotient rule
    float dCoeff_dvi = (sech2 * dScaledNorm_dvi * sqrtC * vNorm - tanhVal * sqrtC * dVNorm_dvi)
                       / (sqrtC * sqrtC * vNorm * vNorm + EPSILON_DIV);

    // d(scaledV[k])/d(v[i]) = d(coeff)/d(v[i]) * v[k] + coeff * delta_{ki}
    float dScaledV_dvi[MAX_DIM];
    for (int k = 0; k < safeDim; k++) {
        dScaledV_dvi[k] = dCoeff_dvi * vLocal[k];
        if (k == i) dScaledV_dvi[k] += coeff;
    }

    // Now compute Jacobian of mobius_add(basePoint, scaledV) w.r.t. scaledV
    // result = mobius_add(x, y) where x=basePoint (fixed), y=scaledV (variable)
    // We need d(result[j])/d(y[k]) for all j, k

    float yNormSq = compute_norm_sq(scaledV, safeDim);
    float xyDot = dot_product(xLocal, scaledV, safeDim);

    float denom = 1.0f + 2.0f * c * xyDot + c * c * xNormSq * yNormSq;
    denom = fmaxf(fabsf(denom), EPSILON_DIV);
    float denomSq = denom * denom;

    float coeff1 = 1.0f + 2.0f * c * xyDot + c * yNormSq;
    float coeff2 = 1.0f - c * xNormSq;

    // Compute full gradient: gradTangent[i] = sum_j(gradOut[j] * d(result[j])/d(v[i]))
    // = sum_j( gradOut[j] * sum_k( d(result[j])/d(scaledV[k]) * d(scaledV[k])/d(v[i]) ) )
    float gradTi = 0.0f;

    for (int j = 0; j < safeDim; j++) {
        float numerator_j = coeff1 * xLocal[j] + coeff2 * scaledV[j];

        // Accumulate contribution from all scaledV dimensions
        for (int k = 0; k < safeDim; k++) {
            // Derivatives w.r.t. scaledV[k] (y in mobius_add formula)
            float dYNormSq_dyk = 2.0f * scaledV[k];
            float dXYDot_dyk = xLocal[k];
            float dCoeff1_dyk = 2.0f * c * dXYDot_dyk + c * dYNormSq_dyk;
            float dDenom_dyk = 2.0f * c * dXYDot_dyk + c * c * xNormSq * dYNormSq_dyk;

            float dNumerator_dyk = dCoeff1_dyk * xLocal[j];
            if (j == k) {
                dNumerator_dyk += coeff2;  // Direct term from coeff2 * y[j]
            }

            float dResult_dyk = (dNumerator_dyk * denom - numerator_j * dDenom_dyk) / denomSq;

            gradTi += gradOutLocal[j] * dResult_dyk * dScaledV_dvi[k];
        }
    }

    gradTangent[baseIdx + i] = gradTi;
}

extern ""C"" __global__ void hyperbolic_log_map_backward(
    const float* gradOutput,
    const float* x,
    const float* y,
    float* gradX,
    float* gradY,
    int size, int dim, float curvature)
{
    // Each thread computes gradient for one input dimension 'i'
    // Must compute full Jacobian-vector product over all output dimensions
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = gid / dim;
    int i = gid % dim;  // Input dimension we're computing gradient for

    int safeDim = safe_dim(dim);
    if (batch >= size / dim || i >= safeDim) return;

    float c = fabsf(curvature);
    if (c < EPSILON_DIV) c = 1.0f;

    int baseIdx = batch * dim;

    // Load local copies
    float xLocal[MAX_DIM], yLocal[MAX_DIM], gradOutLocal[MAX_DIM];
    for (int k = 0; k < safeDim; k++) {
        xLocal[k] = x[baseIdx + k];
        yLocal[k] = y[baseIdx + k];
        gradOutLocal[k] = gradOutput[baseIdx + k];
    }

    float negX[MAX_DIM];
    for (int k = 0; k < safeDim; k++) {
        negX[k] = -xLocal[k];
    }

    // diff = mobius_add(-x, y)
    float diff[MAX_DIM];
    mobius_add(negX, yLocal, diff, safeDim, c);

    float diffNorm = compute_norm(diff, safeDim);
    if (diffNorm < EPSILON_DIV) {
        // When diff is zero, log_map returns 0, gradient w.r.t. y is identity, x is -identity
        gradX[baseIdx + i] = -gradOutLocal[i];
        gradY[baseIdx + i] = gradOutLocal[i];
        return;
    }

    float xNormSq = compute_norm_sq(xLocal, safeDim);
    float lambdaX = 2.0f / fmaxf(1.0f - c * xNormSq, EPSILON_DIV);
    float sqrtC = sqrtf(c);

    float arg = sqrtC * diffNorm;
    arg = fminf(arg, 1.0f - EPSILON_BOUNDARY);
    float arctanhVal = safe_arctanh(arg);

    // result[j] = coeff * diff[j]
    // coeff = (2 / (lambdaX * sqrtC)) * arctanh(sqrtC * diffNorm) / diffNorm
    float baseCoeff = 2.0f / (lambdaX * sqrtC);
    float coeff = baseCoeff * arctanhVal / diffNorm;

    // For gradY[i]: need d(result[j])/d(y[i]) for all j
    // result[j] = coeff * diff[j]
    // diff depends on y through mobius_add(-x, y)
    // coeff depends on diff through diffNorm

    // Step 1: Compute mobius_add(-x, y) Jacobian w.r.t. y
    // Using same formula as mobius_add_backward for the 'y' gradient
    float negXNormSq = xNormSq;  // ||-x||^2 = ||x||^2
    float yNormSq = compute_norm_sq(yLocal, safeDim);
    float negXYDot = dot_product(negX, yLocal, safeDim);

    float denom = 1.0f + 2.0f * c * negXYDot + c * c * negXNormSq * yNormSq;
    denom = fmaxf(fabsf(denom), EPSILON_DIV);
    float denomSq = denom * denom;

    float mCoeff1 = 1.0f + 2.0f * c * negXYDot + c * yNormSq;
    float mCoeff2 = 1.0f - c * negXNormSq;

    // Step 2: Compute d(result[j])/d(y[i])
    // result[j] = coeff * diff[j]
    // d(result[j])/d(y[i]) = d(coeff)/d(y[i]) * diff[j] + coeff * d(diff[j])/d(y[i])

    // d(coeff)/d(diffNorm) = baseCoeff * (d(arctanh)/d(diffNorm) * diffNorm - arctanh) / diffNorm^2
    // d(arctanh(u))/du = 1/(1-u^2) for u = sqrtC * diffNorm
    float u = arg;
    float dArctanh_du = 1.0f / fmaxf(1.0f - u * u, EPSILON_DIV);
    float dArctanh_dDiffNorm = dArctanh_du * sqrtC;

    float dCoeff_dDiffNorm = baseCoeff * (dArctanh_dDiffNorm * diffNorm - arctanhVal) / (diffNorm * diffNorm + EPSILON_DIV);

    // d(diffNorm)/d(diff[k]) = diff[k] / diffNorm
    // d(diffNorm)/d(y[i]) = sum_k( d(diffNorm)/d(diff[k]) * d(diff[k])/d(y[i]) )

    float gradYi = 0.0f;
    float gradXi = 0.0f;

    for (int j = 0; j < safeDim; j++) {
        // Compute d(result[j])/d(y[i]) by chain rule through diff
        float dResultj_dyi = 0.0f;

        for (int k = 0; k < safeDim; k++) {
            // d(diff[k])/d(y[i]) from mobius_add Jacobian
            float dYNormSq_dyi = 2.0f * yLocal[i];
            float dNegXYDot_dyi = negX[i];
            float dMCoeff1_dyi = 2.0f * c * dNegXYDot_dyi + c * dYNormSq_dyi;
            float dDenom_dyi = 2.0f * c * dNegXYDot_dyi + c * c * negXNormSq * dYNormSq_dyi;

            float numerator_k = mCoeff1 * negX[k] + mCoeff2 * yLocal[k];
            float dNumerator_dyi = dMCoeff1_dyi * negX[k];
            if (k == i) {
                dNumerator_dyi += mCoeff2;  // Direct term
            }

            float dDiff_dyi_k = (dNumerator_dyi * denom - numerator_k * dDenom_dyi) / denomSq;

            // Contribution to d(result[j])/d(y[i]):
            // From coeff term: d(coeff)/d(diff[k]) * d(diff[k])/d(y[i]) * diff[j]
            float dDiffNorm_dDiffk = diff[k] / (diffNorm + EPSILON_DIV);
            float dCoeff_dDiffk = dCoeff_dDiffNorm * dDiffNorm_dDiffk;
            dResultj_dyi += dCoeff_dDiffk * dDiff_dyi_k * diff[j];

            // From direct term: coeff * d(diff[j])/d(y[i]) (only when k==j)
            if (k == j) {
                dResultj_dyi += coeff * dDiff_dyi_k;
            }
        }

        gradYi += gradOutLocal[j] * dResultj_dyi;
    }

    // For gradX[i]: similar but through -x in mobius_add(-x, y)
    // d(diff)/d(x[i]) = d(diff)/d(negX[i]) * d(negX[i])/d(x[i]) = -d(diff)/d(negX[i])
    for (int j = 0; j < safeDim; j++) {
        float dResultj_dxi = 0.0f;

        for (int k = 0; k < safeDim; k++) {
            // d(diff[k])/d(negX[i]) from mobius_add Jacobian (x is negX here)
            float dNegXNormSq_dnegXi = 2.0f * negX[i];
            float dNegXYDot_dnegXi = yLocal[i];
            float dMCoeff1_dnegXi = 2.0f * c * dNegXYDot_dnegXi;
            float dMCoeff2_dnegXi = -c * dNegXNormSq_dnegXi;
            float dDenom_dnegXi = 2.0f * c * dNegXYDot_dnegXi + c * c * dNegXNormSq_dnegXi * yNormSq;

            float numerator_k = mCoeff1 * negX[k] + mCoeff2 * yLocal[k];
            float dNumerator_dnegXi = dMCoeff1_dnegXi * negX[k] + dMCoeff2_dnegXi * yLocal[k];
            if (k == i) {
                dNumerator_dnegXi += mCoeff1;  // Direct term
            }

            float dDiff_dnegXi_k = (dNumerator_dnegXi * denom - numerator_k * dDenom_dnegXi) / denomSq;

            // d(diff[k])/d(x[i]) = -d(diff[k])/d(negX[i])
            float dDiff_dxi_k = -dDiff_dnegXi_k;

            // Also need to account for lambdaX depending on x (for gradX only)
            // lambdaX = 2 / (1 - c*||x||^2)
            // d(lambdaX)/d(x[i]) = 2 * c * 2*x[i] / (1 - c*||x||^2)^2 = lambdaX^2 * c * x[i]
            // d(coeff)/d(lambdaX) = -baseCoeff/lambdaX^2 * arctanhVal/diffNorm = -coeff/lambdaX

            float dDiffNorm_dDiffk = diff[k] / (diffNorm + EPSILON_DIV);
            float dCoeff_dDiffk = dCoeff_dDiffNorm * dDiffNorm_dDiffk;
            dResultj_dxi += dCoeff_dDiffk * dDiff_dxi_k * diff[j];

            if (k == j) {
                dResultj_dxi += coeff * dDiff_dxi_k;
            }
        }

        // Add contribution from lambdaX depending on x
        float dLambdaX_dxi = lambdaX * lambdaX * c * xLocal[i];
        float dCoeff_dLambdaX = -coeff / lambdaX;
        // Note: The direct term (j == i ? 0.0f : 0.0f) evaluates to 0 always, so it's omitted.
        // The lambdaX contribution is captured via lambdaXContrib below.

        // Actually the lambdaX term affects ALL result[j] through coeff:
        // d(result[j])/d(x[i]) += diff[j] * d(coeff)/d(lambdaX) * d(lambdaX)/d(x[i])
        // This is independent of the loop over k, so add it once per j
        float lambdaXContrib = diff[j] * dCoeff_dLambdaX * dLambdaX_dxi;

        gradXi += gradOutLocal[j] * (dResultj_dxi + lambdaXContrib);
    }

    gradY[baseIdx + i] = gradYi;
    gradX[baseIdx + i] = gradXi;
}
";
    }

    public static string[] GetKernelNames()
    {
        return new[]
        {
            "hyperbolic_linear_forward",
            "hyperbolic_linear_backward_input",
            "hyperbolic_linear_backward_weights",
            "hyperbolic_linear_backward_biases",
            "hyperbolic_mobius_add_backward",
            "hyperbolic_exp_map_backward",
            "hyperbolic_log_map_backward"
        };
    }
}
