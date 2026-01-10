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

// ===========================================================================
// HYPERBOLIC GEOMETRY HELPER FUNCTIONS
// ===========================================================================

__device__ float compute_norm_sq(const float* v, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        sum += v[i] * v[i];
    }
    return sum;
}

__device__ float compute_norm(const float* v, int dim) {
    return sqrtf(compute_norm_sq(v, dim));
}

__device__ float dot_product(const float* a, const float* b, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

__device__ float safe_arctanh(float x) {
    x = fmaxf(-1.0f + EPSILON_BOUNDARY, fminf(1.0f - EPSILON_BOUNDARY, x));
    return 0.5f * logf((1.0f + x) / (1.0f - x));
}

__device__ void project_to_ball(float* point, int dim, float c) {
    float maxNorm = MAX_NORM_FACTOR / sqrtf(c);
    float norm = compute_norm(point, dim);
    if (norm > maxNorm) {
        float scale = maxNorm / norm;
        for (int i = 0; i < dim; i++) {
            point[i] *= scale;
        }
    }
}

__device__ float conformal_factor(const float* x, int dim, float c) {
    float normSq = compute_norm_sq(x, dim);
    return 2.0f / fmaxf(1.0f - c * normSq, EPSILON_DIV);
}

__device__ void mobius_add(const float* x, const float* y, float* result, int dim, float c) {
    float xNormSq = compute_norm_sq(x, dim);
    float yNormSq = compute_norm_sq(y, dim);
    float xyDot = dot_product(x, y, dim);

    float denom = 1.0f + 2.0f * c * xyDot + c * c * xNormSq * yNormSq;
    denom = fmaxf(fabsf(denom), EPSILON_DIV);

    float coeff1 = 1.0f + 2.0f * c * xyDot + c * yNormSq;
    float coeff2 = 1.0f - c * xNormSq;

    for (int i = 0; i < dim; i++) {
        result[i] = (coeff1 * x[i] + coeff2 * y[i]) / denom;
    }
}

__device__ float poincare_distance(const float* x, const float* y, int dim, float c) {
    float negX[MAX_DIM];
    for (int i = 0; i < dim; i++) {
        negX[i] = -x[i];
    }

    float diff[MAX_DIM];
    mobius_add(negX, y, diff, dim, c);

    float diffNorm = compute_norm(diff, dim);
    float sqrtC = sqrtf(c);
    float arg = sqrtC * diffNorm;
    arg = fminf(arg, 1.0f - EPSILON_BOUNDARY);

    return (2.0f / sqrtC) * safe_arctanh(arg);
}

__device__ void poincare_exp_map(const float* basePoint, const float* tangentVec, float* result, int dim, float c) {
    float vNorm = compute_norm(tangentVec, dim);
    if (vNorm < EPSILON_DIV) {
        for (int i = 0; i < dim; i++) {
            result[i] = basePoint[i];
        }
        return;
    }

    float xNormSq = compute_norm_sq(basePoint, dim);
    float lambdaX = 2.0f / fmaxf(1.0f - c * xNormSq, EPSILON_DIV);

    float sqrtC = sqrtf(c);
    float scaledNorm = sqrtC * lambdaX * vNorm / 2.0f;
    scaledNorm = fminf(scaledNorm, 20.0f);
    float tanhVal = tanhf(scaledNorm);
    float coeff = tanhVal / (sqrtC * vNorm + EPSILON_DIV);

    float scaledV[MAX_DIM];
    for (int i = 0; i < dim; i++) {
        scaledV[i] = coeff * tangentVec[i];
    }

    mobius_add(basePoint, scaledV, result, dim, c);
    project_to_ball(result, dim, c);
}

__device__ void poincare_log_map(const float* x, const float* y, float* result, int dim, float c) {
    float negX[MAX_DIM];
    for (int i = 0; i < dim; i++) {
        negX[i] = -x[i];
    }

    float diff[MAX_DIM];
    mobius_add(negX, y, diff, dim, c);

    float diffNorm = compute_norm(diff, dim);
    if (diffNorm < EPSILON_DIV) {
        for (int i = 0; i < dim; i++) {
            result[i] = 0.0f;
        }
        return;
    }

    float xNormSq = compute_norm_sq(x, dim);
    float lambdaX = 2.0f / fmaxf(1.0f - c * xNormSq, EPSILON_DIV);

    float sqrtC = sqrtf(c);
    float arg = sqrtC * diffNorm;
    arg = fminf(arg, 1.0f - EPSILON_BOUNDARY);
    float arctanhVal = safe_arctanh(arg);

    float coeff = (2.0f / (lambdaX * sqrtC)) * arctanhVal / diffNorm;

    for (int i = 0; i < dim; i++) {
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

    float projectedInput[MAX_DIM];
    for (int i = 0; i < inputFeatures && i < MAX_DIM; i++) {
        projectedInput[i] = input[b * inputFeatures + i];
    }
    project_to_ball(projectedInput, inputFeatures, c);

    float weightVec[MAX_DIM];
    float biasVec[MAX_DIM];
    for (int i = 0; i < inputFeatures && i < MAX_DIM; i++) {
        weightVec[i] = weights[o * inputFeatures + i];
        biasVec[i] = biases[o * inputFeatures + i];
    }

    float origin[MAX_DIM];
    for (int i = 0; i < inputFeatures && i < MAX_DIM; i++) {
        origin[i] = 0.0f;
    }

    float weightPoint[MAX_DIM];
    poincare_exp_map(origin, weightVec, weightPoint, inputFeatures, c);

    float transformed[MAX_DIM];
    mobius_add(projectedInput, weightPoint, transformed, inputFeatures, c);

    project_to_ball(biasVec, inputFeatures, c);

    float withBias[MAX_DIM];
    mobius_add(transformed, biasVec, withBias, inputFeatures, c);

    float distance = poincare_distance(origin, withBias, inputFeatures, c);
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

    float c = fabsf(curvature);
    if (c < EPSILON_DIV) c = 1.0f;

    float projectedInput[MAX_DIM];
    for (int j = 0; j < inputFeatures && j < MAX_DIM; j++) {
        projectedInput[j] = input[b * inputFeatures + j];
    }
    project_to_ball(projectedInput, inputFeatures, c);

    float squaredNorm = compute_norm_sq(projectedInput, inputFeatures);
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

    float c = fabsf(curvature);
    if (c < EPSILON_DIV) c = 1.0f;

    // Load and project input for this batch element
    float projectedInput[MAX_DIM];
    for (int j = 0; j < inputFeatures && j < MAX_DIM; j++) {
        projectedInput[j] = input[b * inputFeatures + j];
    }
    project_to_ball(projectedInput, inputFeatures, c);

    float squaredNorm = compute_norm_sq(projectedInput, inputFeatures);
    float cNormSquared = c * squaredNorm;
    float oneMinusCNorm = 1.0f - cNormSquared;
    float conformalFactor = (oneMinusCNorm * oneMinusCNorm) / 4.0f;

    float gradOut = gradOutput[b * outputFeatures + o];
    float riemannianGrad = gradOut * conformalFactor;

    // Use atomic add to accumulate gradient from all batch elements
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

    float c = fabsf(curvature);
    if (c < EPSILON_DIV) c = 1.0f;

    // Load and project input for this batch element
    float projectedInput[MAX_DIM];
    for (int j = 0; j < inputFeatures && j < MAX_DIM; j++) {
        projectedInput[j] = input[b * inputFeatures + j];
    }
    project_to_ball(projectedInput, inputFeatures, c);

    float squaredNorm = compute_norm_sq(projectedInput, inputFeatures);
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
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = gid / dim;
    int d = gid % dim;

    if (batch >= size / dim || d >= dim) return;

    float c = fabsf(curvature);
    if (c < EPSILON_DIV) c = 1.0f;

    int baseIdx = batch * dim;

    float xLocal[MAX_DIM], yLocal[MAX_DIM];
    for (int i = 0; i < dim && i < MAX_DIM; i++) {
        xLocal[i] = x[baseIdx + i];
        yLocal[i] = y[baseIdx + i];
    }

    float xNormSq = compute_norm_sq(xLocal, dim);
    float yNormSq = compute_norm_sq(yLocal, dim);
    float xyDot = dot_product(xLocal, yLocal, dim);

    float denom = 1.0f + 2.0f * c * xyDot + c * c * xNormSq * yNormSq;
    denom = fmaxf(fabsf(denom), EPSILON_DIV);
    float denomSq = denom * denom;

    float coeff1 = 1.0f + 2.0f * c * xyDot + c * yNormSq;
    float coeff2 = 1.0f - c * xNormSq;

    float gradOutD = gradOutput[baseIdx + d];

    float dCoeff1_dxd = 2.0f * c * yLocal[d];
    float dCoeff2_dxd = -2.0f * c * xLocal[d];
    float dDenom_dxd = 2.0f * c * yLocal[d] + 2.0f * c * c * xLocal[d] * yNormSq;

    float numerator_x = coeff1 * xLocal[d] + coeff2 * yLocal[d];
    float dNumerator_dxd = dCoeff1_dxd * xLocal[d] + coeff1 + dCoeff2_dxd * yLocal[d];

    float gradXd = (dNumerator_dxd * denom - numerator_x * dDenom_dxd) / denomSq;
    gradX[baseIdx + d] = gradOutD * gradXd;

    float dCoeff1_dyd = 2.0f * c * xLocal[d] + 2.0f * c * yLocal[d];
    float dDenom_dyd = 2.0f * c * xLocal[d] + 2.0f * c * c * xNormSq * yLocal[d];

    float dNumerator_dyd = dCoeff1_dyd * xLocal[d] + coeff2;

    float gradYd = (dNumerator_dyd * denom - numerator_x * dDenom_dyd) / denomSq;
    gradY[baseIdx + d] = gradOutD * gradYd;
}

extern ""C"" __global__ void hyperbolic_exp_map_backward(
    const float* gradOutput,
    const float* basePoint,
    const float* tangentVec,
    float* gradTangent,
    int size, int dim, float curvature)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = gid / dim;
    int d = gid % dim;

    if (batch >= size / dim || d >= dim) return;

    float c = fabsf(curvature);
    if (c < EPSILON_DIV) c = 1.0f;

    int baseIdx = batch * dim;

    float vLocal[MAX_DIM];
    float xLocal[MAX_DIM];
    for (int i = 0; i < dim && i < MAX_DIM; i++) {
        vLocal[i] = tangentVec[baseIdx + i];
        xLocal[i] = basePoint[baseIdx + i];
    }

    float vNorm = compute_norm(vLocal, dim);
    if (vNorm < EPSILON_DIV) {
        gradTangent[baseIdx + d] = gradOutput[baseIdx + d];
        return;
    }

    float xNormSq = compute_norm_sq(xLocal, dim);
    float lambdaX = 2.0f / fmaxf(1.0f - c * xNormSq, EPSILON_DIV);
    float sqrtC = sqrtf(c);

    float scaledNorm = sqrtC * lambdaX * vNorm / 2.0f;
    scaledNorm = fminf(scaledNorm, 20.0f);

    float tanhVal = tanhf(scaledNorm);
    float sech2 = 1.0f - tanhVal * tanhVal;

    float coeff = tanhVal / (sqrtC * vNorm + EPSILON_DIV);

    float dVNorm_dvd = vLocal[d] / (vNorm + EPSILON_DIV);
    float dScaledNorm_dvd = sqrtC * lambdaX * dVNorm_dvd / 2.0f;

    float dCoeff_dvd = (sech2 * dScaledNorm_dvd * sqrtC * vNorm - tanhVal * sqrtC * dVNorm_dvd)
                       / (sqrtC * sqrtC * vNorm * vNorm + EPSILON_DIV);

    float gradOutD = gradOutput[baseIdx + d];
    gradTangent[baseIdx + d] = gradOutD * (coeff + vLocal[d] * dCoeff_dvd);
}

extern ""C"" __global__ void hyperbolic_log_map_backward(
    const float* gradOutput,
    const float* x,
    const float* y,
    float* gradX,
    float* gradY,
    int size, int dim, float curvature)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = gid / dim;
    int d = gid % dim;

    if (batch >= size / dim || d >= dim) return;

    float c = fabsf(curvature);
    if (c < EPSILON_DIV) c = 1.0f;

    int baseIdx = batch * dim;

    float xLocal[MAX_DIM], yLocal[MAX_DIM];
    for (int i = 0; i < dim && i < MAX_DIM; i++) {
        xLocal[i] = x[baseIdx + i];
        yLocal[i] = y[baseIdx + i];
    }

    float negX[MAX_DIM];
    for (int i = 0; i < dim && i < MAX_DIM; i++) {
        negX[i] = -xLocal[i];
    }

    float diff[MAX_DIM];
    mobius_add(negX, yLocal, diff, dim, c);

    float diffNorm = compute_norm(diff, dim);
    if (diffNorm < EPSILON_DIV) {
        gradX[baseIdx + d] = 0.0f;
        gradY[baseIdx + d] = gradOutput[baseIdx + d];
        return;
    }

    float xNormSq = compute_norm_sq(xLocal, dim);
    float lambdaX = 2.0f / fmaxf(1.0f - c * xNormSq, EPSILON_DIV);
    float sqrtC = sqrtf(c);

    float arg = sqrtC * diffNorm;
    arg = fminf(arg, 1.0f - EPSILON_BOUNDARY);
    float arctanhVal = safe_arctanh(arg);

    float coeff = (2.0f / (lambdaX * sqrtC)) * arctanhVal / diffNorm;

    float gradOutD = gradOutput[baseIdx + d];

    gradY[baseIdx + d] = gradOutD * coeff;
    gradX[baseIdx + d] = -gradOutD * coeff;
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
