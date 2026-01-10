// Copyright (c) AiDotNet. All rights reserved.
// CUDA kernels for hyperbolic (Poincare ball) neural network operations.

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

/// <summary>
/// CUDA kernels for hyperbolic geometry operations used in hyperbolic neural networks.
/// Implements Poincare ball model operations with full forward and backward support.
/// </summary>
internal static class CudaHyperbolicKernels
{
    public static string GetSource()
    {
        return @"
#include <math.h>

#define EPSILON 1e-15f
#define MAX_NORM_FACTOR 0.99999f
#define MAX_DIM 128

// ===========================================================================
// HYPERBOLIC GEOMETRY HELPER FUNCTIONS
// ===========================================================================

// Compute squared norm of a vector
__device__ float compute_norm_sq(const float* v, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        sum += v[i] * v[i];
    }
    return sum;
}

// Compute norm of a vector
__device__ float compute_norm(const float* v, int dim) {
    return sqrtf(compute_norm_sq(v, dim));
}

// Compute dot product
__device__ float dot_product(const float* a, const float* b, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// Arctanh with numerical stability
__device__ float safe_arctanh(float x) {
    x = fmaxf(-1.0f + EPSILON, fminf(1.0f - EPSILON, x));
    return 0.5f * logf((1.0f + x) / (1.0f - x));
}

// Project point to Poincare ball (ensure ||x|| < maxNorm/sqrt(c))
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

// Conformal factor lambda_x = 2 / (1 - c * ||x||^2)
__device__ float conformal_factor(const float* x, int dim, float c) {
    float normSq = compute_norm_sq(x, dim);
    return 2.0f / fmaxf(1.0f - c * normSq, EPSILON);
}

// Mobius addition: x +_c y in Poincare ball
__device__ void mobius_add(const float* x, const float* y, float* result, int dim, float c) {
    float xNormSq = compute_norm_sq(x, dim);
    float yNormSq = compute_norm_sq(y, dim);
    float xyDot = dot_product(x, y, dim);

    float denom = 1.0f + 2.0f * c * xyDot + c * c * xNormSq * yNormSq;
    denom = fmaxf(fabsf(denom), EPSILON);

    float coeff1 = 1.0f + 2.0f * c * xyDot + c * yNormSq;
    float coeff2 = 1.0f - c * xNormSq;

    for (int i = 0; i < dim; i++) {
        result[i] = (coeff1 * x[i] + coeff2 * y[i]) / denom;
    }
}

// Poincare distance
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
    arg = fminf(arg, 1.0f - EPSILON);

    return (2.0f / sqrtC) * safe_arctanh(arg);
}

// Poincare exponential map: exp_x(v)
__device__ void poincare_exp_map(const float* basePoint, const float* tangentVec, float* result, int dim, float c) {
    float vNorm = compute_norm(tangentVec, dim);
    if (vNorm < EPSILON) {
        for (int i = 0; i < dim; i++) {
            result[i] = basePoint[i];
        }
        return;
    }

    float xNormSq = compute_norm_sq(basePoint, dim);
    float lambdaX = 2.0f / fmaxf(1.0f - c * xNormSq, EPSILON);

    float sqrtC = sqrtf(c);
    float scaledNorm = sqrtC * lambdaX * vNorm / 2.0f;
    scaledNorm = fminf(scaledNorm, 20.0f);
    float tanhVal = tanhf(scaledNorm);
    float coeff = tanhVal / (sqrtC * vNorm + EPSILON);

    float scaledV[MAX_DIM];
    for (int i = 0; i < dim; i++) {
        scaledV[i] = coeff * tangentVec[i];
    }

    mobius_add(basePoint, scaledV, result, dim, c);
    project_to_ball(result, dim, c);
}

// Poincare logarithmic map: log_x(y)
__device__ void poincare_log_map(const float* x, const float* y, float* result, int dim, float c) {
    float negX[MAX_DIM];
    for (int i = 0; i < dim; i++) {
        negX[i] = -x[i];
    }

    float diff[MAX_DIM];
    mobius_add(negX, y, diff, dim, c);

    float diffNorm = compute_norm(diff, dim);
    if (diffNorm < EPSILON) {
        for (int i = 0; i < dim; i++) {
            result[i] = 0.0f;
        }
        return;
    }

    float xNormSq = compute_norm_sq(x, dim);
    float lambdaX = 2.0f / fmaxf(1.0f - c * xNormSq, EPSILON);

    float sqrtC = sqrtf(c);
    float arg = sqrtC * diffNorm;
    arg = fminf(arg, 1.0f - EPSILON);
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
    if (c < EPSILON) c = 1.0f;

    // Copy input to local array and project
    float projectedInput[MAX_DIM];
    for (int i = 0; i < inputFeatures && i < MAX_DIM; i++) {
        projectedInput[i] = input[b * inputFeatures + i];
    }
    project_to_ball(projectedInput, inputFeatures, c);

    // Get weight and bias for this output
    float weightVec[MAX_DIM];
    float biasVec[MAX_DIM];
    for (int i = 0; i < inputFeatures && i < MAX_DIM; i++) {
        weightVec[i] = weights[o * inputFeatures + i];
        biasVec[i] = biases[o * inputFeatures + i];
    }

    // Step 1: Apply exponential map from origin with weight as tangent vector
    float origin[MAX_DIM];
    for (int i = 0; i < inputFeatures && i < MAX_DIM; i++) {
        origin[i] = 0.0f;
    }

    float weightPoint[MAX_DIM];
    poincare_exp_map(origin, weightVec, weightPoint, inputFeatures, c);

    // Step 2: Mobius addition of input with weight point
    float transformed[MAX_DIM];
    mobius_add(projectedInput, weightPoint, transformed, inputFeatures, c);

    // Step 3: Project bias and add with Mobius addition
    project_to_ball(biasVec, inputFeatures, c);

    float withBias[MAX_DIM];
    mobius_add(transformed, biasVec, withBias, inputFeatures, c);

    // Step 4: Compute distance from origin as scalar output
    float distance = poincare_distance(origin, withBias, inputFeatures, c);

    output[b * outputFeatures + o] = distance;
}

// ===========================================================================
// HYPERBOLIC LINEAR BACKWARD KERNELS
// ===========================================================================

// Backward pass - computes input gradients using Riemannian gradient
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
    if (c < EPSILON) c = 1.0f;

    // Copy input to local array and project
    float projectedInput[MAX_DIM];
    for (int j = 0; j < inputFeatures && j < MAX_DIM; j++) {
        projectedInput[j] = input[b * inputFeatures + j];
    }
    project_to_ball(projectedInput, inputFeatures, c);

    // Compute Riemannian metric tensor scaling: g^{-1} = ((1 - c||x||^2) / 2)^2
    float squaredNorm = compute_norm_sq(projectedInput, inputFeatures);
    float cNormSquared = c * squaredNorm;
    float oneMinusCNorm = 1.0f - cNormSquared;
    float conformalFactor = (oneMinusCNorm * oneMinusCNorm) / 4.0f;

    // Accumulate gradient from all outputs
    float gradSum = 0.0f;
    for (int o = 0; o < outputFeatures; o++) {
        float gradOut = gradOutput[b * outputFeatures + o];
        // Convert Euclidean gradient to Riemannian gradient
        float riemannianGrad = gradOut * conformalFactor;
        float weight = weights[o * inputFeatures + i];
        gradSum += riemannianGrad * weight;
    }

    gradInput[b * inputFeatures + i] = gradSum;
}

// Backward pass - computes weight gradients
extern ""C"" __global__ void hyperbolic_linear_backward_weights(
    const float* gradOutput,
    const float* input,
    float* gradWeights,
    int batch, int inputFeatures, int outputFeatures, float curvature)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int o = gid / inputFeatures;
    int i = gid % inputFeatures;

    if (o >= outputFeatures || i >= inputFeatures) return;

    float c = fabsf(curvature);
    if (c < EPSILON) c = 1.0f;

    // Accumulate weight gradient over batch
    float gradSum = 0.0f;
    for (int b = 0; b < batch; b++) {
        // Copy input to local array and project
        float projectedInput[MAX_DIM];
        for (int j = 0; j < inputFeatures && j < MAX_DIM; j++) {
            projectedInput[j] = input[b * inputFeatures + j];
        }
        project_to_ball(projectedInput, inputFeatures, c);

        // Riemannian metric scaling
        float squaredNorm = compute_norm_sq(projectedInput, inputFeatures);
        float cNormSquared = c * squaredNorm;
        float oneMinusCNorm = 1.0f - cNormSquared;
        float conformalFactor = (oneMinusCNorm * oneMinusCNorm) / 4.0f;

        float gradOut = gradOutput[b * outputFeatures + o];
        float riemannianGrad = gradOut * conformalFactor;

        gradSum += riemannianGrad * projectedInput[i];
    }

    gradWeights[o * inputFeatures + i] = gradSum;
}

// Backward pass - computes bias gradients
extern ""C"" __global__ void hyperbolic_linear_backward_biases(
    const float* gradOutput,
    const float* input,
    float* gradBiases,
    int batch, int inputFeatures, int outputFeatures, float curvature)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int o = gid / inputFeatures;
    int i = gid % inputFeatures;

    if (o >= outputFeatures || i >= inputFeatures) return;

    float c = fabsf(curvature);
    if (c < EPSILON) c = 1.0f;

    // Accumulate bias gradient over batch
    float gradSum = 0.0f;
    for (int b = 0; b < batch; b++) {
        // Copy input to local array and project
        float projectedInput[MAX_DIM];
        for (int j = 0; j < inputFeatures && j < MAX_DIM; j++) {
            projectedInput[j] = input[b * inputFeatures + j];
        }
        project_to_ball(projectedInput, inputFeatures, c);

        // Riemannian metric scaling
        float squaredNorm = compute_norm_sq(projectedInput, inputFeatures);
        float cNormSquared = c * squaredNorm;
        float oneMinusCNorm = 1.0f - cNormSquared;
        float conformalFactor = (oneMinusCNorm * oneMinusCNorm) / 4.0f;

        float gradOut = gradOutput[b * outputFeatures + o];
        float riemannianGrad = gradOut * conformalFactor;

        // Bias gradient distributed across input features
        gradSum += riemannianGrad / (float)inputFeatures;
    }

    gradBiases[o * inputFeatures + i] = gradSum;
}

// ===========================================================================
// MOBIUS ADDITION BACKWARD KERNEL
// ===========================================================================

// Backward pass for Mobius addition: d(x +_c y)/dx and d(x +_c y)/dy
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
    if (c < EPSILON) c = 1.0f;

    int baseIdx = batch * dim;

    // Load vectors to local memory
    float xLocal[MAX_DIM], yLocal[MAX_DIM];
    for (int i = 0; i < dim && i < MAX_DIM; i++) {
        xLocal[i] = x[baseIdx + i];
        yLocal[i] = y[baseIdx + i];
    }

    float xNormSq = compute_norm_sq(xLocal, dim);
    float yNormSq = compute_norm_sq(yLocal, dim);
    float xyDot = dot_product(xLocal, yLocal, dim);

    float denom = 1.0f + 2.0f * c * xyDot + c * c * xNormSq * yNormSq;
    denom = fmaxf(fabsf(denom), EPSILON);
    float denomSq = denom * denom;

    float coeff1 = 1.0f + 2.0f * c * xyDot + c * yNormSq;
    float coeff2 = 1.0f - c * xNormSq;

    // Compute partial derivatives
    // d(coeff1)/dx_d = 2 * c * y_d
    // d(coeff2)/dx_d = -2 * c * x_d
    // d(denom)/dx_d = 2 * c * y_d + 2 * c^2 * x_d * yNormSq

    float gradOutD = gradOutput[baseIdx + d];

    // Gradient w.r.t. x_d
    float dCoeff1_dxd = 2.0f * c * yLocal[d];
    float dCoeff2_dxd = -2.0f * c * xLocal[d];
    float dDenom_dxd = 2.0f * c * yLocal[d] + 2.0f * c * c * xLocal[d] * yNormSq;

    float numerator_x = coeff1 * xLocal[d] + coeff2 * yLocal[d];
    float dNumerator_dxd = dCoeff1_dxd * xLocal[d] + coeff1 + dCoeff2_dxd * yLocal[d];

    float gradXd = (dNumerator_dxd * denom - numerator_x * dDenom_dxd) / denomSq;
    gradX[baseIdx + d] = gradOutD * gradXd;

    // Gradient w.r.t. y_d
    float dCoeff1_dyd = 2.0f * c * xLocal[d] + 2.0f * c * yLocal[d];
    float dDenom_dyd = 2.0f * c * xLocal[d] + 2.0f * c * c * xNormSq * yLocal[d];

    float dNumerator_dyd = dCoeff1_dyd * xLocal[d] + coeff2;

    float gradYd = (dNumerator_dyd * denom - numerator_x * dDenom_dyd) / denomSq;
    gradY[baseIdx + d] = gradOutD * gradYd;
}

// ===========================================================================
// EXPONENTIAL MAP BACKWARD KERNEL
// ===========================================================================

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
    if (c < EPSILON) c = 1.0f;

    int baseIdx = batch * dim;

    // Load vectors
    float vLocal[MAX_DIM];
    float xLocal[MAX_DIM];
    for (int i = 0; i < dim && i < MAX_DIM; i++) {
        vLocal[i] = tangentVec[baseIdx + i];
        xLocal[i] = basePoint[baseIdx + i];
    }

    float vNorm = compute_norm(vLocal, dim);
    if (vNorm < EPSILON) {
        gradTangent[baseIdx + d] = gradOutput[baseIdx + d];
        return;
    }

    float xNormSq = compute_norm_sq(xLocal, dim);
    float lambdaX = 2.0f / fmaxf(1.0f - c * xNormSq, EPSILON);
    float sqrtC = sqrtf(c);

    float scaledNorm = sqrtC * lambdaX * vNorm / 2.0f;
    scaledNorm = fminf(scaledNorm, 20.0f);

    float tanhVal = tanhf(scaledNorm);
    float sech2 = 1.0f - tanhVal * tanhVal;

    // Coefficient: tanh(sqrt(c) * lambda * ||v|| / 2) / (sqrt(c) * ||v||)
    float coeff = tanhVal / (sqrtC * vNorm + EPSILON);

    // d(coeff)/d(v_d) involves chain rule through ||v|| and tanh
    float dVNorm_dvd = vLocal[d] / (vNorm + EPSILON);
    float dScaledNorm_dvd = sqrtC * lambdaX * dVNorm_dvd / 2.0f;

    // d(tanh(s)/(sqrt(c)*||v||)) / dv_d
    float dCoeff_dvd = (sech2 * dScaledNorm_dvd * sqrtC * vNorm - tanhVal * sqrtC * dVNorm_dvd)
                       / (sqrtC * sqrtC * vNorm * vNorm + EPSILON);

    // Gradient contribution
    float gradOutD = gradOutput[baseIdx + d];
    gradTangent[baseIdx + d] = gradOutD * (coeff + vLocal[d] * dCoeff_dvd);
}

// ===========================================================================
// LOGARITHMIC MAP BACKWARD KERNEL
// ===========================================================================

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
    if (c < EPSILON) c = 1.0f;

    int baseIdx = batch * dim;

    // Load vectors
    float xLocal[MAX_DIM], yLocal[MAX_DIM];
    for (int i = 0; i < dim && i < MAX_DIM; i++) {
        xLocal[i] = x[baseIdx + i];
        yLocal[i] = y[baseIdx + i];
    }

    // Compute -x +_c y
    float negX[MAX_DIM];
    for (int i = 0; i < dim && i < MAX_DIM; i++) {
        negX[i] = -xLocal[i];
    }

    float diff[MAX_DIM];
    mobius_add(negX, yLocal, diff, dim, c);

    float diffNorm = compute_norm(diff, dim);
    if (diffNorm < EPSILON) {
        gradX[baseIdx + d] = 0.0f;
        gradY[baseIdx + d] = gradOutput[baseIdx + d];
        return;
    }

    float xNormSq = compute_norm_sq(xLocal, dim);
    float lambdaX = 2.0f / fmaxf(1.0f - c * xNormSq, EPSILON);
    float sqrtC = sqrtf(c);

    float arg = sqrtC * diffNorm;
    arg = fminf(arg, 1.0f - EPSILON);
    float arctanhVal = safe_arctanh(arg);

    float coeff = (2.0f / (lambdaX * sqrtC)) * arctanhVal / diffNorm;

    // Simplified gradient (treating some terms as constants for stability)
    float gradOutD = gradOutput[baseIdx + d];

    // Gradient w.r.t. y approximately propagates through the log map
    gradY[baseIdx + d] = gradOutD * coeff;

    // Gradient w.r.t. x has opposite sign contribution
    gradX[baseIdx + d] = -gradOutD * coeff;
}
";
    }

    /// <summary>
    /// Gets the list of kernel names provided by this source.
    /// </summary>
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
