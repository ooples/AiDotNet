// Copyright (c) AiDotNet. All rights reserved.
// HIP kernels for octonion neural network operations.
// Implements 8-component non-associative algebra with full forward and backward support.

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

/// <summary>
/// HIP kernels for octonion algebra operations used in octonion neural networks.
/// Implements the Cayley-Dickson construction for octonion multiplication.
/// </summary>
internal static class HipOctonionKernels
{
    public static string GetSource()
    {
        return @"
#include <hip/hip_runtime.h>
#include <math.h>

// Epsilon for division safety - must be appropriate for float32 precision
// Float32 machine epsilon is ~1.19e-7, so use 1e-7f for safe divisions
#define EPSILON_DIV 1e-7f
// Epsilon for norm^2 comparisons (allows smaller values before protection kicks in)
#define EPSILON_NORM_SQ 1e-14f

// ===========================================================================
// OCTONION MULTIPLICATION TABLE
// ===========================================================================
// Octonion basis: {e0, e1, e2, e3, e4, e5, e6, e7}
// e0 = 1 (real unit)
// e1, e2, e3 = i, j, k (quaternion imaginaries)
// e4, e5, e6, e7 = l, il, jl, kl (octonion extensions)
//
// Multiplication rules follow Cayley-Dickson construction:
// ei * ej = -delta_ij * e0 + epsilon_ijk * ek (for i,j > 0)
// where epsilon_ijk is the structure constant

// ===========================================================================
// OCTONION HELPER FUNCTIONS
// ===========================================================================

// Octonion multiplication: r = a * b
// Uses explicit Cayley-Dickson multiplication table
__device__ void octonion_multiply(const float* a, const float* b, float* r) {
    // Real part: e0
    r[0] = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3]
         - a[4]*b[4] - a[5]*b[5] - a[6]*b[6] - a[7]*b[7];

    // e1 component
    r[1] = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2]
         + a[4]*b[5] - a[5]*b[4] - a[6]*b[7] + a[7]*b[6];

    // e2 component
    r[2] = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1]
         + a[4]*b[6] + a[5]*b[7] - a[6]*b[4] - a[7]*b[5];

    // e3 component
    r[3] = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0]
         + a[4]*b[7] - a[5]*b[6] + a[6]*b[5] - a[7]*b[4];

    // e4 component
    r[4] = a[0]*b[4] - a[1]*b[5] - a[2]*b[6] - a[3]*b[7]
         + a[4]*b[0] + a[5]*b[1] + a[6]*b[2] + a[7]*b[3];

    // e5 component
    r[5] = a[0]*b[5] + a[1]*b[4] - a[2]*b[7] + a[3]*b[6]
         - a[4]*b[1] + a[5]*b[0] - a[6]*b[3] + a[7]*b[2];

    // e6 component
    r[6] = a[0]*b[6] + a[1]*b[7] + a[2]*b[4] - a[3]*b[5]
         - a[4]*b[2] + a[5]*b[3] + a[6]*b[0] - a[7]*b[1];

    // e7 component
    r[7] = a[0]*b[7] - a[1]*b[6] + a[2]*b[5] + a[3]*b[4]
         - a[4]*b[3] - a[5]*b[2] + a[6]*b[1] + a[7]*b[0];
}

// Octonion conjugate: conj(a) = a0 - a1*e1 - ... - a7*e7
__device__ void octonion_conjugate(const float* a, float* r) {
    r[0] = a[0];
    for (int i = 1; i < 8; i++) {
        r[i] = -a[i];
    }
}

// Octonion norm squared: ||a||^2 = a * conj(a) = sum(ai^2)
__device__ float octonion_norm_sq(const float* a) {
    float sum = 0.0f;
    for (int i = 0; i < 8; i++) {
        sum += a[i] * a[i];
    }
    return sum;
}

// Octonion inverse: a^(-1) = conj(a) / ||a||^2
__device__ void octonion_inverse(const float* a, float* r) {
    float normSq = octonion_norm_sq(a);
    normSq = fmaxf(normSq, EPSILON_NORM_SQ);

    r[0] = a[0] / normSq;
    for (int i = 1; i < 8; i++) {
        r[i] = -a[i] / normSq;
    }
}

// Octonion addition
__device__ void octonion_add(const float* a, const float* b, float* r) {
    for (int i = 0; i < 8; i++) {
        r[i] = a[i] + b[i];
    }
}

// Scale octonion by scalar
__device__ void octonion_scale(const float* a, float s, float* r) {
    for (int i = 0; i < 8; i++) {
        r[i] = a[i] * s;
    }
}

// ===========================================================================
// OCTONION MULTIPLICATION JACOBIANS FOR BACKWARD PASS
// ===========================================================================

// Compute dr/da for r = a * b (left multiplication Jacobian)
// This is an 8x8 matrix stored row-major
__device__ void octonion_multiply_jacobian_a(const float* b, float* jac) {
    // dr[0]/da[i]: r0 = a0*b0 - a1*b1 - a2*b2 - a3*b3 - a4*b4 - a5*b5 - a6*b6 - a7*b7
    jac[0*8+0] = b[0];  jac[0*8+1] = -b[1]; jac[0*8+2] = -b[2]; jac[0*8+3] = -b[3];
    jac[0*8+4] = -b[4]; jac[0*8+5] = -b[5]; jac[0*8+6] = -b[6]; jac[0*8+7] = -b[7];

    // dr[1]/da[i]: r1 = a0*b1 + a1*b0 + a2*b3 - a3*b2 + a4*b5 - a5*b4 - a6*b7 + a7*b6
    jac[1*8+0] = b[1];  jac[1*8+1] = b[0];  jac[1*8+2] = b[3];  jac[1*8+3] = -b[2];
    jac[1*8+4] = b[5];  jac[1*8+5] = -b[4]; jac[1*8+6] = -b[7]; jac[1*8+7] = b[6];

    // dr[2]/da[i]: r2 = a0*b2 - a1*b3 + a2*b0 + a3*b1 + a4*b6 + a5*b7 - a6*b4 - a7*b5
    jac[2*8+0] = b[2];  jac[2*8+1] = -b[3]; jac[2*8+2] = b[0];  jac[2*8+3] = b[1];
    jac[2*8+4] = b[6];  jac[2*8+5] = b[7];  jac[2*8+6] = -b[4]; jac[2*8+7] = -b[5];

    // dr[3]/da[i]: r3 = a0*b3 + a1*b2 - a2*b1 + a3*b0 + a4*b7 - a5*b6 + a6*b5 - a7*b4
    jac[3*8+0] = b[3];  jac[3*8+1] = b[2];  jac[3*8+2] = -b[1]; jac[3*8+3] = b[0];
    jac[3*8+4] = b[7];  jac[3*8+5] = -b[6]; jac[3*8+6] = b[5];  jac[3*8+7] = -b[4];

    // dr[4]/da[i]: r4 = a0*b4 - a1*b5 - a2*b6 - a3*b7 + a4*b0 + a5*b1 + a6*b2 + a7*b3
    jac[4*8+0] = b[4];  jac[4*8+1] = -b[5]; jac[4*8+2] = -b[6]; jac[4*8+3] = -b[7];
    jac[4*8+4] = b[0];  jac[4*8+5] = b[1];  jac[4*8+6] = b[2];  jac[4*8+7] = b[3];

    // dr[5]/da[i]: r5 = a0*b5 + a1*b4 - a2*b7 + a3*b6 - a4*b1 + a5*b0 - a6*b3 + a7*b2
    jac[5*8+0] = b[5];  jac[5*8+1] = b[4];  jac[5*8+2] = -b[7]; jac[5*8+3] = b[6];
    jac[5*8+4] = -b[1]; jac[5*8+5] = b[0];  jac[5*8+6] = -b[3]; jac[5*8+7] = b[2];

    // dr[6]/da[i]: r6 = a0*b6 + a1*b7 + a2*b4 - a3*b5 - a4*b2 + a5*b3 + a6*b0 - a7*b1
    jac[6*8+0] = b[6];  jac[6*8+1] = b[7];  jac[6*8+2] = b[4];  jac[6*8+3] = -b[5];
    jac[6*8+4] = -b[2]; jac[6*8+5] = b[3];  jac[6*8+6] = b[0];  jac[6*8+7] = -b[1];

    // dr[7]/da[i]: r7 = a0*b7 - a1*b6 + a2*b5 + a3*b4 - a4*b3 - a5*b2 + a6*b1 + a7*b0
    jac[7*8+0] = b[7];  jac[7*8+1] = -b[6]; jac[7*8+2] = b[5];  jac[7*8+3] = b[4];
    jac[7*8+4] = -b[3]; jac[7*8+5] = -b[2]; jac[7*8+6] = b[1];  jac[7*8+7] = b[0];
}

// Compute dr/db for r = a * b (right multiplication Jacobian)
// This is an 8x8 matrix stored row-major
__device__ void octonion_multiply_jacobian_b(const float* a, float* jac) {
    // dr[0]/db[i]: r0 = a0*b0 - a1*b1 - a2*b2 - a3*b3 - a4*b4 - a5*b5 - a6*b6 - a7*b7
    jac[0*8+0] = a[0];  jac[0*8+1] = -a[1]; jac[0*8+2] = -a[2]; jac[0*8+3] = -a[3];
    jac[0*8+4] = -a[4]; jac[0*8+5] = -a[5]; jac[0*8+6] = -a[6]; jac[0*8+7] = -a[7];

    // dr[1]/db[i]: r1 = a0*b1 + a1*b0 + a2*b3 - a3*b2 + a4*b5 - a5*b4 - a6*b7 + a7*b6
    jac[1*8+0] = a[1];  jac[1*8+1] = a[0];  jac[1*8+2] = -a[3]; jac[1*8+3] = a[2];
    jac[1*8+4] = -a[5]; jac[1*8+5] = a[4];  jac[1*8+6] = a[7];  jac[1*8+7] = -a[6];

    // dr[2]/db[i]: r2 = a0*b2 - a1*b3 + a2*b0 + a3*b1 + a4*b6 + a5*b7 - a6*b4 - a7*b5
    jac[2*8+0] = a[2];  jac[2*8+1] = a[3];  jac[2*8+2] = a[0];  jac[2*8+3] = -a[1];
    jac[2*8+4] = -a[6]; jac[2*8+5] = -a[7]; jac[2*8+6] = a[4];  jac[2*8+7] = a[5];

    // dr[3]/db[i]: r3 = a0*b3 + a1*b2 - a2*b1 + a3*b0 + a4*b7 - a5*b6 + a6*b5 - a7*b4
    jac[3*8+0] = a[3];  jac[3*8+1] = -a[2]; jac[3*8+2] = a[1];  jac[3*8+3] = a[0];
    jac[3*8+4] = -a[7]; jac[3*8+5] = a[6];  jac[3*8+6] = -a[5]; jac[3*8+7] = a[4];

    // dr[4]/db[i]: r4 = a0*b4 - a1*b5 - a2*b6 - a3*b7 + a4*b0 + a5*b1 + a6*b2 + a7*b3
    jac[4*8+0] = a[4];  jac[4*8+1] = a[5];  jac[4*8+2] = a[6];  jac[4*8+3] = a[7];
    jac[4*8+4] = a[0];  jac[4*8+5] = -a[1]; jac[4*8+6] = -a[2]; jac[4*8+7] = -a[3];

    // dr[5]/db[i]: r5 = a0*b5 + a1*b4 - a2*b7 + a3*b6 - a4*b1 + a5*b0 - a6*b3 + a7*b2
    jac[5*8+0] = a[5];  jac[5*8+1] = -a[4]; jac[5*8+2] = a[7];  jac[5*8+3] = -a[6];
    jac[5*8+4] = a[1];  jac[5*8+5] = a[0];  jac[5*8+6] = a[3];  jac[5*8+7] = -a[2];

    // dr[6]/db[i]: r6 = a0*b6 + a1*b7 + a2*b4 - a3*b5 - a4*b2 + a5*b3 + a6*b0 - a7*b1
    jac[6*8+0] = a[6];  jac[6*8+1] = -a[7]; jac[6*8+2] = -a[4]; jac[6*8+3] = a[5];
    jac[6*8+4] = a[2];  jac[6*8+5] = -a[3]; jac[6*8+6] = a[0];  jac[6*8+7] = a[1];

    // dr[7]/db[i]: r7 = a0*b7 - a1*b6 + a2*b5 + a3*b4 - a4*b3 - a5*b2 + a6*b1 + a7*b0
    jac[7*8+0] = a[7];  jac[7*8+1] = a[6];  jac[7*8+2] = -a[5]; jac[7*8+3] = -a[4];
    jac[7*8+4] = a[3];  jac[7*8+5] = a[2];  jac[7*8+6] = -a[1]; jac[7*8+7] = a[0];
}

// ===========================================================================
// OCTONION LINEAR FORWARD KERNEL
// ===========================================================================

// Forward pass: output = input * weight + bias (all octonion-valued)
// input: [batch, inputFeatures, 8] - batch of octonion vectors
// weights: [outputFeatures, inputFeatures, 8] - octonion weight matrix
// biases: [outputFeatures, 8] - octonion bias vector
// output: [batch, outputFeatures, 8] - output octonion vectors
extern ""C"" __global__ void octonion_linear_forward(
    const float* input,
    const float* weights,
    const float* biases,
    float* output,
    int batch, int inputFeatures, int outputFeatures)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalOutputs = batch * outputFeatures;

    if (gid >= totalOutputs) return;

    int b = gid / outputFeatures;
    int o = gid % outputFeatures;

    // Initialize output with bias
    float result[8];
    for (int c = 0; c < 8; c++) {
        result[c] = biases[o * 8 + c];
    }

    // Accumulate: result += sum_i(input[b,i] * weight[o,i])
    for (int i = 0; i < inputFeatures; i++) {
        float inputOct[8], weightOct[8], product[8];

        // Load input octonion
        for (int c = 0; c < 8; c++) {
            inputOct[c] = input[(b * inputFeatures + i) * 8 + c];
        }

        // Load weight octonion
        for (int c = 0; c < 8; c++) {
            weightOct[c] = weights[(o * inputFeatures + i) * 8 + c];
        }

        // Octonion multiplication
        octonion_multiply(inputOct, weightOct, product);

        // Accumulate
        for (int c = 0; c < 8; c++) {
            result[c] += product[c];
        }
    }

    // Store result
    for (int c = 0; c < 8; c++) {
        output[(b * outputFeatures + o) * 8 + c] = result[c];
    }
}

// ===========================================================================
// OCTONION LINEAR BACKWARD KERNELS
// ===========================================================================

// Backward pass - computes input gradients
// gradOutput: [batch, outputFeatures, 8]
// weights: [outputFeatures, inputFeatures, 8]
// gradInput: [batch, inputFeatures, 8]
extern ""C"" __global__ void octonion_linear_backward_input(
    const float* gradOutput,
    const float* input,
    const float* weights,
    float* gradInput,
    int batch, int inputFeatures, int outputFeatures)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalInputs = batch * inputFeatures;

    if (gid >= totalInputs) return;

    int b = gid / inputFeatures;
    int i = gid % inputFeatures;

    // Initialize gradient
    float gradSum[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // For octonion linear: output[o] = sum_i(input[i] * weight[o,i]) + bias[o]
    // d(output[o])/d(input[i]) uses the Jacobian of octonion multiplication
    // grad_input[i] = sum_o(grad_output[o] * d(input[i] * weight[o,i])/d(input[i]))

    for (int o = 0; o < outputFeatures; o++) {
        float gradOut[8], weightOct[8];

        // Load gradient output
        for (int c = 0; c < 8; c++) {
            gradOut[c] = gradOutput[(b * outputFeatures + o) * 8 + c];
        }

        // Load weight
        for (int c = 0; c < 8; c++) {
            weightOct[c] = weights[(o * inputFeatures + i) * 8 + c];
        }

        // Compute Jacobian d(input * weight)/d(input) and apply to gradOutput
        // Using the transpose of the right multiplication Jacobian
        float jacA[64];
        octonion_multiply_jacobian_a(weightOct, jacA);

        // grad_input += jacA^T * gradOut (matrix-vector multiply)
        for (int row = 0; row < 8; row++) {
            for (int col = 0; col < 8; col++) {
                // jacA^T[row, col] = jacA[col, row]
                gradSum[row] += jacA[col * 8 + row] * gradOut[col];
            }
        }
    }

    // Store gradient
    for (int c = 0; c < 8; c++) {
        gradInput[(b * inputFeatures + i) * 8 + c] = gradSum[c];
    }
}

// Backward pass - computes weight gradients
// gradOutput: [batch, outputFeatures, 8]
// input: [batch, inputFeatures, 8]
// gradWeights: [outputFeatures, inputFeatures, 8]
extern ""C"" __global__ void octonion_linear_backward_weights(
    const float* gradOutput,
    const float* input,
    float* gradWeights,
    int batch, int inputFeatures, int outputFeatures)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalWeights = outputFeatures * inputFeatures;

    if (gid >= totalWeights) return;

    int o = gid / inputFeatures;
    int i = gid % inputFeatures;

    // Initialize gradient
    float gradSum[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // Accumulate weight gradient over batch
    // d(output[o])/d(weight[o,i]) uses the Jacobian of octonion multiplication
    for (int b = 0; b < batch; b++) {
        float gradOut[8], inputOct[8];

        // Load gradient output
        for (int c = 0; c < 8; c++) {
            gradOut[c] = gradOutput[(b * outputFeatures + o) * 8 + c];
        }

        // Load input
        for (int c = 0; c < 8; c++) {
            inputOct[c] = input[(b * inputFeatures + i) * 8 + c];
        }

        // Compute Jacobian d(input * weight)/d(weight) and apply to gradOutput
        // Using the transpose of the left multiplication Jacobian evaluated at input
        float jacB[64];
        octonion_multiply_jacobian_b(inputOct, jacB);

        // grad_weight += jacB^T * gradOut (matrix-vector multiply)
        for (int row = 0; row < 8; row++) {
            for (int col = 0; col < 8; col++) {
                // jacB^T[row, col] = jacB[col, row]
                gradSum[row] += jacB[col * 8 + row] * gradOut[col];
            }
        }
    }

    // Store gradient
    for (int c = 0; c < 8; c++) {
        gradWeights[(o * inputFeatures + i) * 8 + c] = gradSum[c];
    }
}

// Backward pass - computes bias gradients
// gradOutput: [batch, outputFeatures, 8]
// gradBiases: [outputFeatures, 8]
extern ""C"" __global__ void octonion_linear_backward_biases(
    const float* gradOutput,
    float* gradBiases,
    int batch, int outputFeatures)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid >= outputFeatures) return;

    int o = gid;

    // Bias gradient is just sum of output gradients over batch
    float gradSum[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < 8; c++) {
            gradSum[c] += gradOutput[(b * outputFeatures + o) * 8 + c];
        }
    }

    // Store gradient
    for (int c = 0; c < 8; c++) {
        gradBiases[o * 8 + c] = gradSum[c];
    }
}

// ===========================================================================
// OCTONION MULTIPLICATION BACKWARD KERNEL
// ===========================================================================

// Backward pass for standalone octonion multiplication: r = a * b
// Computes gradients for both inputs
extern ""C"" __global__ void octonion_multiply_backward(
    const float* gradOutput,
    const float* a,
    const float* b,
    float* gradA,
    float* gradB,
    int count)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid >= count) return;

    int baseIdx = gid * 8;

    // Load operands
    float aLocal[8], bLocal[8], gradOut[8];
    for (int c = 0; c < 8; c++) {
        aLocal[c] = a[baseIdx + c];
        bLocal[c] = b[baseIdx + c];
        gradOut[c] = gradOutput[baseIdx + c];
    }

    // Compute Jacobians
    float jacA[64], jacB[64];
    octonion_multiply_jacobian_a(bLocal, jacA);
    octonion_multiply_jacobian_b(aLocal, jacB);

    // grad_a = jacA^T * gradOut
    float gradALocal[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 8; col++) {
            gradALocal[row] += jacA[col * 8 + row] * gradOut[col];
        }
    }

    // grad_b = jacB^T * gradOut
    float gradBLocal[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 8; col++) {
            gradBLocal[row] += jacB[col * 8 + row] * gradOut[col];
        }
    }

    // Store gradients
    for (int c = 0; c < 8; c++) {
        gradA[baseIdx + c] = gradALocal[c];
        gradB[baseIdx + c] = gradBLocal[c];
    }
}

// ===========================================================================
// OCTONION CONJUGATE BACKWARD KERNEL
// ===========================================================================

extern ""C"" __global__ void octonion_conjugate_backward(
    const float* gradOutput,
    float* gradInput,
    int count)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid >= count) return;

    int baseIdx = gid * 8;

    // Conjugate gradient: d(conj(a))/da = diag(1, -1, -1, -1, -1, -1, -1, -1)
    gradInput[baseIdx + 0] = gradOutput[baseIdx + 0];
    for (int c = 1; c < 8; c++) {
        gradInput[baseIdx + c] = -gradOutput[baseIdx + c];
    }
}

// ===========================================================================
// OCTONION NORM BACKWARD KERNEL
// ===========================================================================

extern ""C"" __global__ void octonion_norm_backward(
    const float* gradOutput,
    const float* input,
    float* gradInput,
    int count)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid >= count) return;

    int baseIdx = gid * 8;

    // Load input
    float inputLocal[8];
    for (int c = 0; c < 8; c++) {
        inputLocal[c] = input[baseIdx + c];
    }

    // Compute norm with proper float32 protection
    float normSq = octonion_norm_sq(inputLocal);
    float norm = sqrtf(fmaxf(normSq, EPSILON_NORM_SQ));
    // Ensure norm is large enough for safe division
    norm = fmaxf(norm, EPSILON_DIV);

    // d(||a||)/da = a / ||a||
    float gradOutScalar = gradOutput[gid]; // Note: norm output is scalar
    for (int c = 0; c < 8; c++) {
        gradInput[baseIdx + c] = gradOutScalar * inputLocal[c] / norm;
    }
}

// ===========================================================================
// OCTONION ACTIVATION FORWARD/BACKWARD KERNELS
// ===========================================================================

// Split activation: apply real-valued activation to each component
extern ""C"" __global__ void octonion_split_relu_forward(
    const float* input,
    float* output,
    int count)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = count * 8;

    if (gid >= totalElements) return;

    float val = input[gid];
    output[gid] = fmaxf(0.0f, val);
}

extern ""C"" __global__ void octonion_split_relu_backward(
    const float* gradOutput,
    const float* input,
    float* gradInput,
    int count)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = count * 8;

    if (gid >= totalElements) return;

    float val = input[gid];
    gradInput[gid] = val > 0.0f ? gradOutput[gid] : 0.0f;
}

// Modulus activation: scale octonion by activation of its norm with threshold
// output = relu(||input|| - threshold) * input / ||input||
// When norm <= threshold, output is zero (dead zone)
// When norm > threshold, output preserves direction but scales magnitude
extern ""C"" __global__ void octonion_modulus_relu_forward(
    const float* input,
    float* output,
    int count,
    float threshold)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid >= count) return;

    int baseIdx = gid * 8;

    // Load input
    float inputLocal[8];
    for (int c = 0; c < 8; c++) {
        inputLocal[c] = input[baseIdx + c];
    }

    // Compute norm with proper float32 protection
    float normSq = octonion_norm_sq(inputLocal);
    float norm = sqrtf(fmaxf(normSq, EPSILON_NORM_SQ));
    // Ensure norm is large enough for safe division
    norm = fmaxf(norm, EPSILON_DIV);

    // Apply ReLU to (norm - threshold)
    float activatedNorm = fmaxf(0.0f, norm - threshold);

    // Scale octonion: output = input * relu(norm - threshold) / norm
    float scale = activatedNorm / norm;
    for (int c = 0; c < 8; c++) {
        output[baseIdx + c] = inputLocal[c] * scale;
    }
}

extern ""C"" __global__ void octonion_modulus_relu_backward(
    const float* gradOutput,
    const float* input,
    float* gradInput,
    int count,
    float threshold)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid >= count) return;

    int baseIdx = gid * 8;

    // Load input and gradient
    float inputLocal[8], gradOut[8];
    for (int c = 0; c < 8; c++) {
        inputLocal[c] = input[baseIdx + c];
        gradOut[c] = gradOutput[baseIdx + c];
    }

    // Compute norm with proper float32 protection
    float normSq = octonion_norm_sq(inputLocal);
    float norm = sqrtf(fmaxf(normSq, EPSILON_NORM_SQ));
    // Ensure norm is large enough for safe division (especially for norm^3)
    norm = fmaxf(norm, EPSILON_DIV);

    // Check if in dead zone (norm <= threshold)
    if (norm <= threshold) {
        // ReLU is zero, gradient is zero
        for (int c = 0; c < 8; c++) {
            gradInput[baseIdx + c] = 0.0f;
        }
        return;
    }

    // For modulus ReLU with threshold:
    // output = input * (norm - threshold) / norm
    // d(output)/d(input) involves both scaling and direction terms
    // Simplified: when norm > threshold, gradient scales by (norm - threshold) / norm
    // plus contribution from norm gradient
    float activatedNorm = norm - threshold;
    float scale = activatedNorm / norm;

    // Compute dot product of gradOut and inputLocal for the radial component
    float dotProduct = 0.0f;
    for (int c = 0; c < 8; c++) {
        dotProduct += gradOut[c] * inputLocal[c];
    }

    // Gradient: scale * gradOut + (threshold / norm^3) * dotProduct * input
    // For numerical stability, compute norm^3 as norm * normSq to avoid cubing small numbers
    float normCubed = norm * normSq;
    // Additional protection: ensure normCubed is not too small
    normCubed = fmaxf(normCubed, EPSILON_DIV * EPSILON_DIV * EPSILON_DIV);
    float radialFactor = threshold * dotProduct / normCubed;
    for (int c = 0; c < 8; c++) {
        gradInput[baseIdx + c] = scale * gradOut[c] + radialFactor * inputLocal[c];
    }
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
            "octonion_linear_forward",
            "octonion_linear_backward_input",
            "octonion_linear_backward_weights",
            "octonion_linear_backward_biases",
            "octonion_multiply_backward",
            "octonion_conjugate_backward",
            "octonion_norm_backward",
            "octonion_split_relu_forward",
            "octonion_split_relu_backward",
            "octonion_modulus_relu_forward",
            "octonion_modulus_relu_backward"
        };
    }
}
