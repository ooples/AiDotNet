// Copyright (c) AiDotNet. All rights reserved.
// OpenCL kernels for hyperbolic (Poincare ball) neural network operations.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;

/// <summary>
/// OpenCL kernels for hyperbolic geometry operations used in hyperbolic neural networks.
/// Implements Poincare ball model operations.
/// </summary>
internal static class HyperbolicKernels
{
    public static string GetSource()
    {
        return @"
// ===========================================================================
// HYPERBOLIC GEOMETRY HELPER FUNCTIONS
// ===========================================================================

#define EPSILON 1e-15f
#define MAX_NORM_FACTOR 0.99999f
#define MAX_DIM 128

// Compute squared norm of a vector
inline float compute_norm_sq(__global const float* v, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        sum += v[i] * v[i];
    }
    return sum;
}

// Compute squared norm of a local vector
inline float compute_norm_sq_local(float* v, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        sum += v[i] * v[i];
    }
    return sum;
}

// Compute norm of a local vector
inline float compute_norm_local(float* v, int dim) {
    return sqrt(compute_norm_sq_local(v, dim));
}

// Compute dot product
inline float dot_product_global(__global const float* a, __global const float* b, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// Compute dot product of local vectors
inline float dot_product_local(float* a, float* b, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// Arctanh with numerical stability
inline float safe_arctanh(float x) {
    x = fmax(-1.0f + EPSILON, fmin(1.0f - EPSILON, x));
    return 0.5f * log((1.0f + x) / (1.0f - x));
}

// Project point to Poincare ball (ensure ||x|| < maxNorm/sqrt(c))
inline void project_to_ball_local(float* point, int dim, float c) {
    float maxNorm = MAX_NORM_FACTOR / sqrt(c);
    float norm = compute_norm_local(point, dim);
    if (norm > maxNorm) {
        float scale = maxNorm / norm;
        for (int i = 0; i < dim; i++) {
            point[i] *= scale;
        }
    }
}

// Mobius addition: x + y in Poincare ball (local arrays)
inline void mobius_add_local(float* x, float* y, float* result, int dim, float c) {
    float xNormSq = compute_norm_sq_local(x, dim);
    float yNormSq = compute_norm_sq_local(y, dim);
    float xyDot = dot_product_local(x, y, dim);

    float denom = 1.0f + 2.0f * c * xyDot + c * c * xNormSq * yNormSq;
    denom = fmax(fabs(denom), EPSILON);

    float coeff1 = 1.0f + 2.0f * c * xyDot + c * yNormSq;
    float coeff2 = 1.0f - c * xNormSq;

    for (int i = 0; i < dim; i++) {
        result[i] = (coeff1 * x[i] + coeff2 * y[i]) / denom;
    }
}

// Poincare distance using local arrays
inline float poincare_distance_local(float* x, float* y, int dim, float c) {
    float negX[MAX_DIM];
    for (int i = 0; i < dim; i++) {
        negX[i] = -x[i];
    }

    float diff[MAX_DIM];
    mobius_add_local(negX, y, diff, dim, c);

    float diffNorm = compute_norm_local(diff, dim);
    float sqrtC = sqrt(c);
    float arg = sqrtC * diffNorm;
    arg = fmin(arg, 1.0f - EPSILON);

    return (2.0f / sqrtC) * safe_arctanh(arg);
}

// Poincare exponential map (local arrays)
inline void poincare_exp_map_local(float* basePoint, float* tangentVec, float* result, int dim, float c) {
    float vNorm = compute_norm_local(tangentVec, dim);
    if (vNorm < EPSILON) {
        for (int i = 0; i < dim; i++) {
            result[i] = basePoint[i];
        }
        return;
    }

    // Compute lambda_x = 2 / (1 - c * ||x||^2)
    float xNormSq = compute_norm_sq_local(basePoint, dim);
    float lambdaX = 2.0f / fmax(1.0f - c * xNormSq, EPSILON);

    // Compute tanh(sqrt(c) * lambda_x * ||v|| / 2) / (sqrt(c) * ||v||)
    float sqrtC = sqrt(c);
    float scaledNorm = sqrtC * lambdaX * vNorm / 2.0f;
    scaledNorm = fmin(scaledNorm, 20.0f); // Prevent overflow
    float tanhVal = tanh(scaledNorm);
    float coeff = tanhVal / (sqrtC * vNorm + EPSILON);

    // Compute scaled tangent vector
    float scaledV[MAX_DIM];
    for (int i = 0; i < dim; i++) {
        scaledV[i] = coeff * tangentVec[i];
    }

    // Result = basePoint + scaledV (Mobius addition)
    mobius_add_local(basePoint, scaledV, result, dim, c);

    // Project to ensure we stay inside the ball
    project_to_ball_local(result, dim, c);
}

// ===========================================================================
// HYPERBOLIC LINEAR FORWARD KERNEL
// ===========================================================================

__kernel void hyperbolic_linear_forward(
    __global const float* input,
    __global const float* weights,
    __global const float* biases,
    __global float* output,
    const int batch, const int inputFeatures, const int outputFeatures, const float curvature)
{
    int gid = get_global_id(0);
    int b = gid / outputFeatures;
    int o = gid % outputFeatures;

    if (b >= batch || o >= outputFeatures) return;

    float c = fabs(curvature);
    if (c < EPSILON) c = 1.0f;

    // Copy input to local array and project
    float projectedInput[MAX_DIM];
    for (int i = 0; i < inputFeatures && i < MAX_DIM; i++) {
        projectedInput[i] = input[b * inputFeatures + i];
    }
    project_to_ball_local(projectedInput, inputFeatures, c);

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
    poincare_exp_map_local(origin, weightVec, weightPoint, inputFeatures, c);

    // Step 2: Mobius addition of input with weight point
    float transformed[MAX_DIM];
    mobius_add_local(projectedInput, weightPoint, transformed, inputFeatures, c);

    // Step 3: Project bias and add with Mobius addition
    project_to_ball_local(biasVec, inputFeatures, c);

    float withBias[MAX_DIM];
    mobius_add_local(transformed, biasVec, withBias, inputFeatures, c);

    // Step 4: Compute distance from origin as scalar output
    float distance = poincare_distance_local(origin, withBias, inputFeatures, c);

    output[b * outputFeatures + o] = distance;
}

// ===========================================================================
// HYPERBOLIC LINEAR BACKWARD KERNELS
// ===========================================================================

// Hyperbolic linear backward pass - computes input gradients
__kernel void hyperbolic_linear_backward_input(
    __global const float* gradOutput,
    __global const float* input,
    __global const float* weights,
    __global float* gradInput,
    const int batch, const int inputFeatures, const int outputFeatures, const float curvature)
{
    int gid = get_global_id(0);
    int b = gid / inputFeatures;
    int i = gid % inputFeatures;

    if (b >= batch || i >= inputFeatures) return;

    float c = fabs(curvature);
    if (c < EPSILON) c = 1.0f;

    // Copy input to local array and project
    float projectedInput[MAX_DIM];
    for (int j = 0; j < inputFeatures && j < MAX_DIM; j++) {
        projectedInput[j] = input[b * inputFeatures + j];
    }
    project_to_ball_local(projectedInput, inputFeatures, c);

    // Compute squared norm and conformal factor
    float squaredNorm = compute_norm_sq_local(projectedInput, inputFeatures);
    float cNormSquared = c * squaredNorm;
    float oneMinusCNorm = 1.0f - cNormSquared;
    float conformalFactor = (oneMinusCNorm * oneMinusCNorm) / 4.0f;

    // Accumulate gradient from all outputs
    float gradSum = 0.0f;
    for (int o = 0; o < outputFeatures; o++) {
        float gradOut = gradOutput[b * outputFeatures + o];
        float riemannianGrad = gradOut * conformalFactor;
        float weight = weights[o * inputFeatures + i];
        gradSum += riemannianGrad * weight;
    }

    gradInput[b * inputFeatures + i] = gradSum;
}

// Hyperbolic linear backward pass - computes weight gradients
__kernel void hyperbolic_linear_backward_weights(
    __global const float* gradOutput,
    __global const float* input,
    __global float* gradWeights,
    const int batch, const int inputFeatures, const int outputFeatures, const float curvature)
{
    int gid = get_global_id(0);
    int o = gid / inputFeatures;
    int i = gid % inputFeatures;

    if (o >= outputFeatures || i >= inputFeatures) return;

    float c = fabs(curvature);
    if (c < EPSILON) c = 1.0f;

    // Accumulate weight gradient over batch
    float gradSum = 0.0f;
    for (int b = 0; b < batch; b++) {
        // Copy input to local array and project
        float projectedInput[MAX_DIM];
        for (int j = 0; j < inputFeatures && j < MAX_DIM; j++) {
            projectedInput[j] = input[b * inputFeatures + j];
        }
        project_to_ball_local(projectedInput, inputFeatures, c);

        float squaredNorm = compute_norm_sq_local(projectedInput, inputFeatures);
        float cNormSquared = c * squaredNorm;
        float oneMinusCNorm = 1.0f - cNormSquared;
        float conformalFactor = (oneMinusCNorm * oneMinusCNorm) / 4.0f;

        float gradOut = gradOutput[b * outputFeatures + o];
        float riemannianGrad = gradOut * conformalFactor;

        gradSum += riemannianGrad * projectedInput[i];
    }

    gradWeights[o * inputFeatures + i] = gradSum;
}

// Hyperbolic linear backward pass - computes bias gradients
__kernel void hyperbolic_linear_backward_biases(
    __global const float* gradOutput,
    __global const float* input,
    __global float* gradBiases,
    const int batch, const int inputFeatures, const int outputFeatures, const float curvature)
{
    int gid = get_global_id(0);
    int o = gid / inputFeatures;
    int i = gid % inputFeatures;

    if (o >= outputFeatures || i >= inputFeatures) return;

    float c = fabs(curvature);
    if (c < EPSILON) c = 1.0f;

    // Accumulate bias gradient over batch
    float gradSum = 0.0f;
    for (int b = 0; b < batch; b++) {
        // Copy input to local array and project
        float projectedInput[MAX_DIM];
        for (int j = 0; j < inputFeatures && j < MAX_DIM; j++) {
            projectedInput[j] = input[b * inputFeatures + j];
        }
        project_to_ball_local(projectedInput, inputFeatures, c);

        float squaredNorm = compute_norm_sq_local(projectedInput, inputFeatures);
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
            "hyperbolic_linear_backward_biases"
        };
    }
}
