// Copyright (c) AiDotNet. All rights reserved.
// HIP kernels for GRU (Gated Recurrent Unit) neural network operations.
// Implements sequence-level forward and backward passes for efficient BPTT on AMD GPUs.

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

/// <summary>
/// HIP kernels for sequence-level GRU operations on AMD GPUs.
/// Implements full forward and backward passes for GRU layers processing entire sequences.
/// </summary>
internal static class HipGruKernels
{
    public static string GetSource()
    {
        return @"
#include <hip/hip_runtime.h>
#include <math.h>

#define EPSILON 1e-15f
#define WARP_SIZE 64

// ===========================================================================
// ACTIVATION FUNCTIONS
// ===========================================================================

__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float sigmoid_derivative(float sigmoid_output) {
    return sigmoid_output * (1.0f - sigmoid_output);
}

__device__ __forceinline__ float tanh_derivative(float tanh_output) {
    return 1.0f - tanh_output * tanh_output;
}

// ===========================================================================
// GRU CELL FORWARD KERNEL (Single Timestep)
// ===========================================================================

extern ""C"" __global__ void gru_cell_forward(
    const float* input,
    const float* prevH,
    const float* Wz, const float* Wr, const float* Wh,
    const float* Uz, const float* Ur, const float* Uh,
    const float* bz, const float* br, const float* bh,
    float* newH,
    float* gateZ,
    float* gateR,
    float* gateHCandidate,
    int batch,
    int inputSize,
    int hiddenSize)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batch * hiddenSize;

    if (gid >= totalElements) return;

    int b = gid / hiddenSize;
    int h = gid % hiddenSize;

    float sumZ = bz[h];
    float sumR = br[h];

    for (int i = 0; i < inputSize; i++) {
        float x_val = input[b * inputSize + i];
        sumZ += Wz[h * inputSize + i] * x_val;
        sumR += Wr[h * inputSize + i] * x_val;
    }

    for (int j = 0; j < hiddenSize; j++) {
        float h_val = prevH[b * hiddenSize + j];
        sumZ += Uz[h * hiddenSize + j] * h_val;
        sumR += Ur[h * hiddenSize + j] * h_val;
    }

    float z = sigmoid(sumZ);
    float r = sigmoid(sumR);

    float sumH = bh[h];

    for (int i = 0; i < inputSize; i++) {
        float x_val = input[b * inputSize + i];
        sumH += Wh[h * inputSize + i] * x_val;
    }

    for (int j = 0; j < hiddenSize; j++) {
        float h_val = prevH[b * hiddenSize + j];
        sumH += Uh[h * hiddenSize + j] * r * h_val;
    }

    float h_candidate = tanhf(sumH);

    float prevHVal = prevH[gid];
    float newHVal = (1.0f - z) * prevHVal + z * h_candidate;

    newH[gid] = newHVal;
    gateZ[gid] = z;
    gateR[gid] = r;
    gateHCandidate[gid] = h_candidate;
}

// ===========================================================================
// GRU SEQUENCE FORWARD KERNEL
// ===========================================================================

extern ""C"" __global__ void gru_forward_sequence(
    const float* input,
    const float* h_init,
    const float* Wz, const float* Wr, const float* Wh,
    const float* Uz, const float* Ur, const float* Uh,
    const float* bz, const float* br, const float* bh,
    float* output,
    float* h_states,
    float* gates,
    int batch,
    int timeSteps,
    int inputSize,
    int hiddenSize)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batch * hiddenSize;

    if (gid >= totalElements) return;

    int b = gid / hiddenSize;
    int h_idx = gid % hiddenSize;

    float h_val = h_init[gid];

    for (int t = 0; t < timeSteps; t++) {
        float sumZ = bz[h_idx];
        float sumR = br[h_idx];

        int inputOffset = (b * timeSteps + t) * inputSize;
        for (int i = 0; i < inputSize; i++) {
            float x_val = input[inputOffset + i];
            sumZ += Wz[h_idx * inputSize + i] * x_val;
            sumR += Wr[h_idx * inputSize + i] * x_val;
        }

        for (int j = 0; j < hiddenSize; j++) {
            float hj;
            if (t == 0) {
                hj = h_init[b * hiddenSize + j];
            } else {
                hj = h_states[(t - 1) * batch * hiddenSize + b * hiddenSize + j];
            }
            sumZ += Uz[h_idx * hiddenSize + j] * hj;
            sumR += Ur[h_idx * hiddenSize + j] * hj;
        }

        float z = sigmoid(sumZ);
        float r = sigmoid(sumR);

        __syncthreads();

        float sumH = bh[h_idx];

        for (int i = 0; i < inputSize; i++) {
            float x_val = input[inputOffset + i];
            sumH += Wh[h_idx * inputSize + i] * x_val;
        }

        for (int j = 0; j < hiddenSize; j++) {
            float hj;
            if (t == 0) {
                hj = h_init[b * hiddenSize + j];
            } else {
                hj = h_states[(t - 1) * batch * hiddenSize + b * hiddenSize + j];
            }
            float rj = r;
            sumH += Uh[h_idx * hiddenSize + j] * rj * hj;
        }

        float h_candidate = tanhf(sumH);

        float h_prev;
        if (t == 0) {
            h_prev = h_init[gid];
        } else {
            h_prev = h_states[(t - 1) * batch * hiddenSize + gid];
        }

        h_val = (1.0f - z) * h_prev + z * h_candidate;

        int stateOffset = t * batch * hiddenSize + gid;
        h_states[stateOffset] = h_val;

        output[(b * timeSteps + t) * hiddenSize + h_idx] = h_val;

        int gateOffset = t * batch * 3 * hiddenSize + b * 3 * hiddenSize;
        gates[gateOffset + h_idx] = z;
        gates[gateOffset + hiddenSize + h_idx] = r;
        gates[gateOffset + 2 * hiddenSize + h_idx] = h_candidate;

        __syncthreads();
    }
}

// ===========================================================================
// GRU BACKWARD KERNELS
// ===========================================================================

extern ""C"" __global__ void gru_backward_input(
    const float* dGates,
    const float* Wz, const float* Wr, const float* Wh,
    float* dInput,
    int batch,
    int inputSize,
    int hiddenSize)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batch * inputSize;

    if (gid >= totalElements) return;

    int b = gid / inputSize;
    int i = gid % inputSize;

    float grad = 0.0f;

    for (int h = 0; h < hiddenSize; h++) {
        float dZ = dGates[b * 3 * hiddenSize + h];
        float dR = dGates[b * 3 * hiddenSize + hiddenSize + h];
        float dHCand = dGates[b * 3 * hiddenSize + 2 * hiddenSize + h];

        grad += dZ * Wz[h * inputSize + i];
        grad += dR * Wr[h * inputSize + i];
        grad += dHCand * Wh[h * inputSize + i];
    }

    dInput[gid] = grad;
}

extern ""C"" __global__ void gru_backward_prevh(
    const float* dH,
    const float* dGates,
    const float* gateZ,
    const float* gateR,
    const float* Uz, const float* Ur, const float* Uh,
    float* dPrevH,
    int batch,
    int hiddenSize)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batch * hiddenSize;

    if (gid >= totalElements) return;

    int b = gid / hiddenSize;
    int j = gid % hiddenSize;

    float z = gateZ[gid];
    float r = gateR[gid];

    float grad = dH[gid] * (1.0f - z);

    for (int h = 0; h < hiddenSize; h++) {
        float dZ = dGates[b * 3 * hiddenSize + h];
        float dR = dGates[b * 3 * hiddenSize + hiddenSize + h];
        float dHCand = dGates[b * 3 * hiddenSize + 2 * hiddenSize + h];

        grad += dZ * Uz[h * hiddenSize + j];
        grad += dR * Ur[h * hiddenSize + j];
        grad += dHCand * Uh[h * hiddenSize + j] * r;
    }

    dPrevH[gid] = grad;
}

extern ""C"" __global__ void gru_compute_gate_gradients(
    const float* dH,
    const float* gateZ,
    const float* gateR,
    const float* gateHCand,
    const float* prevH,
    const float* Uh,
    float* dGates,
    int batch,
    int hiddenSize)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batch * hiddenSize;

    if (gid >= totalElements) return;

    int b = gid / hiddenSize;
    int h = gid % hiddenSize;

    float z = gateZ[gid];
    float r = gateR[gid];
    float h_cand = gateHCand[gid];
    float h_prev = prevH[gid];

    float dh = dH[gid];

    float dHCand = dh * z * tanh_derivative(h_cand);
    float dZ = dh * (h_cand - h_prev) * sigmoid_derivative(z);

    float sumUhH = 0.0f;
    for (int j = 0; j < hiddenSize; j++) {
        sumUhH += Uh[h * hiddenSize + j] * prevH[b * hiddenSize + j];
    }
    float dR = dHCand * sumUhH * sigmoid_derivative(r);

    int gateOffset = b * 3 * hiddenSize;
    dGates[gateOffset + h] = dZ;
    dGates[gateOffset + hiddenSize + h] = dR;
    dGates[gateOffset + 2 * hiddenSize + h] = dHCand;
}

extern ""C"" __global__ void gru_accumulate_weight_gradients(
    const float* input,
    const float* prevH,
    const float* gateR,
    const float* dGates,
    float* dWz, float* dWr, float* dWh,
    float* dUz, float* dUr, float* dUh,
    float* dbz, float* dbr, float* dbh,
    int batch,
    int inputSize,
    int hiddenSize)
{
    int gateIdx = blockIdx.x;
    int colIdx = blockIdx.y * blockDim.x + threadIdx.x;

    if (gateIdx >= 3 * hiddenSize) return;

    int gateType = gateIdx / hiddenSize;
    int h = gateIdx % hiddenSize;

    if (colIdx < inputSize) {
        float grad = 0.0f;
        for (int b = 0; b < batch; b++) {
            float dGate = dGates[b * 3 * hiddenSize + gateIdx];
            float x_val = input[b * inputSize + colIdx];
            grad += dGate * x_val;
        }

        if (gateType == 0) atomicAdd(&dWz[h * inputSize + colIdx], grad);
        else if (gateType == 1) atomicAdd(&dWr[h * inputSize + colIdx], grad);
        else atomicAdd(&dWh[h * inputSize + colIdx], grad);
    }

    if (colIdx < hiddenSize) {
        float grad = 0.0f;
        for (int b = 0; b < batch; b++) {
            float dGate = dGates[b * 3 * hiddenSize + gateIdx];
            float h_val = prevH[b * hiddenSize + colIdx];

            if (gateType == 2) {
                float r = gateR[b * hiddenSize + h];
                h_val *= r;
            }

            grad += dGate * h_val;
        }

        if (gateType == 0) atomicAdd(&dUz[h * hiddenSize + colIdx], grad);
        else if (gateType == 1) atomicAdd(&dUr[h * hiddenSize + colIdx], grad);
        else atomicAdd(&dUh[h * hiddenSize + colIdx], grad);
    }

    if (colIdx == 0) {
        float grad = 0.0f;
        for (int b = 0; b < batch; b++) {
            grad += dGates[b * 3 * hiddenSize + gateIdx];
        }

        if (gateType == 0) atomicAdd(&dbz[h], grad);
        else if (gateType == 1) atomicAdd(&dbr[h], grad);
        else atomicAdd(&dbh[h], grad);
    }
}

extern ""C"" __global__ void gru_backward_sequence(
    const float* gradOutput,
    const float* h_states,
    const float* gates,
    const float* h_init,
    const float* input,
    const float* Wz, const float* Wr, const float* Wh,
    const float* Uz, const float* Ur, const float* Uh,
    float* gradInput,
    float* dWz, float* dWr, float* dWh,
    float* dUz, float* dUr, float* dUh,
    float* dbz, float* dbr, float* dbh,
    float* dH_init,
    int batch,
    int timeSteps,
    int inputSize,
    int hiddenSize)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batch * hiddenSize;

    if (gid >= totalElements) return;

    int b = gid / hiddenSize;
    int h_idx = gid % hiddenSize;

    float dH = 0.0f;

    for (int t = timeSteps - 1; t >= 0; t--) {
        dH += gradOutput[(b * timeSteps + t) * hiddenSize + h_idx];

        int gateOffset = t * batch * 3 * hiddenSize + b * 3 * hiddenSize;
        float z = gates[gateOffset + h_idx];
        float r = gates[gateOffset + hiddenSize + h_idx];
        float h_cand = gates[gateOffset + 2 * hiddenSize + h_idx];

        float h_prev;
        if (t == 0) {
            h_prev = h_init[gid];
        } else {
            h_prev = h_states[(t - 1) * batch * hiddenSize + gid];
        }

        float dHCand = dH * z * tanh_derivative(h_cand);
        float dZ = dH * (h_cand - h_prev) * sigmoid_derivative(z);

        float sumUhH = 0.0f;
        for (int j = 0; j < hiddenSize; j++) {
            float hj;
            if (t == 0) {
                hj = h_init[b * hiddenSize + j];
            } else {
                hj = h_states[(t - 1) * batch * hiddenSize + b * hiddenSize + j];
            }
            sumUhH += Uh[h_idx * hiddenSize + j] * hj;
        }
        float dR = dHCand * sumUhH * sigmoid_derivative(r);

        int inputOffset = (b * timeSteps + t) * inputSize;
        for (int i = 0; i < inputSize; i++) {
            float x_val = input[inputOffset + i];
            atomicAdd(&dWz[h_idx * inputSize + i], dZ * x_val);
            atomicAdd(&dWr[h_idx * inputSize + i], dR * x_val);
            atomicAdd(&dWh[h_idx * inputSize + i], dHCand * x_val);
        }

        for (int j = 0; j < hiddenSize; j++) {
            float hj;
            if (t == 0) {
                hj = h_init[b * hiddenSize + j];
            } else {
                hj = h_states[(t - 1) * batch * hiddenSize + b * hiddenSize + j];
            }
            atomicAdd(&dUz[h_idx * hiddenSize + j], dZ * hj);
            atomicAdd(&dUr[h_idx * hiddenSize + j], dR * hj);
            atomicAdd(&dUh[h_idx * hiddenSize + j], dHCand * r * hj);
        }

        atomicAdd(&dbz[h_idx], dZ);
        atomicAdd(&dbr[h_idx], dR);
        atomicAdd(&dbh[h_idx], dHCand);

        int gradInputOffset = (b * timeSteps + t) * inputSize;
        for (int i = 0; i < inputSize; i++) {
            float grad_i = 0.0f;
            grad_i += dZ * Wz[h_idx * inputSize + i];
            grad_i += dR * Wr[h_idx * inputSize + i];
            grad_i += dHCand * Wh[h_idx * inputSize + i];
            atomicAdd(&gradInput[gradInputOffset + i], grad_i);
        }

        float dH_prev = dH * (1.0f - z);

        for (int j = 0; j < hiddenSize; j++) {
            if (j == h_idx) {
                dH_prev += dZ * Uz[h_idx * hiddenSize + j];
                dH_prev += dR * Ur[h_idx * hiddenSize + j];
                dH_prev += dHCand * Uh[h_idx * hiddenSize + j] * r;
            }
        }

        dH = dH_prev;

        __syncthreads();
    }

    dH_init[gid] = dH;
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
            "gru_cell_forward",
            "gru_forward_sequence",
            "gru_backward_input",
            "gru_backward_prevh",
            "gru_compute_gate_gradients",
            "gru_accumulate_weight_gradients",
            "gru_backward_sequence"
        };
    }
}
