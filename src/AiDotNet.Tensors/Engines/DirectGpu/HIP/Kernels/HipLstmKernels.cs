// Copyright (c) AiDotNet. All rights reserved.
// HIP kernels for standard LSTM (Long Short-Term Memory) neural network operations.
// Implements sequence-level forward and backward passes for efficient BPTT on AMD GPUs.

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

/// <summary>
/// HIP kernels for sequence-level LSTM operations on AMD GPUs.
/// Implements full forward and backward passes for LSTM layers processing entire sequences.
/// </summary>
internal static class HipLstmKernels
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
// LSTM CELL FORWARD KERNEL (Single Timestep)
// ===========================================================================

extern ""C"" __global__ void lstm_cell_forward(
    const float* input,
    const float* prevH,
    const float* prevC,
    const float* Wi,
    const float* Wh,
    const float* bias,
    float* newH,
    float* newC,
    float* gateF,
    float* gateI,
    float* gateC,
    float* gateO,
    int batch,
    int inputSize,
    int hiddenSize)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batch * hiddenSize;

    if (gid >= totalElements) return;

    int b = gid / hiddenSize;
    int h = gid % hiddenSize;

    float sumF = bias[h];
    float sumI = bias[hiddenSize + h];
    float sumC = bias[2 * hiddenSize + h];
    float sumO = bias[3 * hiddenSize + h];

    for (int i = 0; i < inputSize; i++) {
        float x_val = input[b * inputSize + i];
        sumF += Wi[h * inputSize + i] * x_val;
        sumI += Wi[(hiddenSize + h) * inputSize + i] * x_val;
        sumC += Wi[(2 * hiddenSize + h) * inputSize + i] * x_val;
        sumO += Wi[(3 * hiddenSize + h) * inputSize + i] * x_val;
    }

    for (int j = 0; j < hiddenSize; j++) {
        float h_val = prevH[b * hiddenSize + j];
        sumF += Wh[h * hiddenSize + j] * h_val;
        sumI += Wh[(hiddenSize + h) * hiddenSize + j] * h_val;
        sumC += Wh[(2 * hiddenSize + h) * hiddenSize + j] * h_val;
        sumO += Wh[(3 * hiddenSize + h) * hiddenSize + j] * h_val;
    }

    float f = sigmoid(sumF);
    float i_gate = sigmoid(sumI);
    float c_candidate = tanhf(sumC);
    float o = sigmoid(sumO);

    float prevCVal = prevC[gid];
    float newCVal = f * prevCVal + i_gate * c_candidate;
    float newHVal = o * tanhf(newCVal);

    newC[gid] = newCVal;
    newH[gid] = newHVal;
    gateF[gid] = f;
    gateI[gid] = i_gate;
    gateC[gid] = c_candidate;
    gateO[gid] = o;
}

// ===========================================================================
// LSTM SEQUENCE FORWARD KERNEL
// ===========================================================================

extern ""C"" __global__ void lstm_forward_sequence(
    const float* input,
    const float* h_init,
    const float* c_init,
    const float* Wi,
    const float* Wh,
    const float* bias,
    float* output,
    float* h_states,
    float* c_states,
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
    float c_val = c_init[gid];

    for (int t = 0; t < timeSteps; t++) {
        float sumF = bias[h_idx];
        float sumI = bias[hiddenSize + h_idx];
        float sumC = bias[2 * hiddenSize + h_idx];
        float sumO = bias[3 * hiddenSize + h_idx];

        int inputOffset = (b * timeSteps + t) * inputSize;
        for (int i = 0; i < inputSize; i++) {
            float x_val = input[inputOffset + i];
            sumF += Wi[h_idx * inputSize + i] * x_val;
            sumI += Wi[(hiddenSize + h_idx) * inputSize + i] * x_val;
            sumC += Wi[(2 * hiddenSize + h_idx) * inputSize + i] * x_val;
            sumO += Wi[(3 * hiddenSize + h_idx) * inputSize + i] * x_val;
        }

        for (int j = 0; j < hiddenSize; j++) {
            float hj;
            if (t == 0) {
                hj = h_init[b * hiddenSize + j];
            } else {
                hj = h_states[(t - 1) * batch * hiddenSize + b * hiddenSize + j];
            }
            sumF += Wh[h_idx * hiddenSize + j] * hj;
            sumI += Wh[(hiddenSize + h_idx) * hiddenSize + j] * hj;
            sumC += Wh[(2 * hiddenSize + h_idx) * hiddenSize + j] * hj;
            sumO += Wh[(3 * hiddenSize + h_idx) * hiddenSize + j] * hj;
        }

        float f = sigmoid(sumF);
        float i_gate = sigmoid(sumI);
        float c_candidate = tanhf(sumC);
        float o = sigmoid(sumO);

        float prev_c;
        if (t == 0) {
            prev_c = c_init[gid];
        } else {
            prev_c = c_states[(t - 1) * batch * hiddenSize + gid];
        }

        c_val = f * prev_c + i_gate * c_candidate;
        h_val = o * tanhf(c_val);

        int stateOffset = t * batch * hiddenSize + gid;
        h_states[stateOffset] = h_val;
        c_states[stateOffset] = c_val;

        output[(b * timeSteps + t) * hiddenSize + h_idx] = h_val;

        int gateOffset = t * batch * 4 * hiddenSize + b * 4 * hiddenSize;
        gates[gateOffset + h_idx] = f;
        gates[gateOffset + hiddenSize + h_idx] = i_gate;
        gates[gateOffset + 2 * hiddenSize + h_idx] = c_candidate;
        gates[gateOffset + 3 * hiddenSize + h_idx] = o;

        __syncthreads();
    }
}

// ===========================================================================
// LSTM BACKWARD KERNELS
// ===========================================================================

extern ""C"" __global__ void lstm_backward_input(
    const float* dGates,
    const float* Wi,
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
        float dF = dGates[b * 4 * hiddenSize + h];
        float dI = dGates[b * 4 * hiddenSize + hiddenSize + h];
        float dC = dGates[b * 4 * hiddenSize + 2 * hiddenSize + h];
        float dO = dGates[b * 4 * hiddenSize + 3 * hiddenSize + h];

        grad += dF * Wi[h * inputSize + i];
        grad += dI * Wi[(hiddenSize + h) * inputSize + i];
        grad += dC * Wi[(2 * hiddenSize + h) * inputSize + i];
        grad += dO * Wi[(3 * hiddenSize + h) * inputSize + i];
    }

    dInput[gid] = grad;
}

extern ""C"" __global__ void lstm_backward_prevh(
    const float* dGates,
    const float* Wh,
    float* dPrevH,
    int batch,
    int hiddenSize)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batch * hiddenSize;

    if (gid >= totalElements) return;

    int b = gid / hiddenSize;
    int j = gid % hiddenSize;

    float grad = 0.0f;

    for (int h = 0; h < hiddenSize; h++) {
        float dF = dGates[b * 4 * hiddenSize + h];
        float dI = dGates[b * 4 * hiddenSize + hiddenSize + h];
        float dC = dGates[b * 4 * hiddenSize + 2 * hiddenSize + h];
        float dO = dGates[b * 4 * hiddenSize + 3 * hiddenSize + h];

        grad += dF * Wh[h * hiddenSize + j];
        grad += dI * Wh[(hiddenSize + h) * hiddenSize + j];
        grad += dC * Wh[(2 * hiddenSize + h) * hiddenSize + j];
        grad += dO * Wh[(3 * hiddenSize + h) * hiddenSize + j];
    }

    dPrevH[gid] = grad;
}

extern ""C"" __global__ void lstm_compute_gate_gradients(
    const float* dH,
    const float* dC_next,
    const float* gateF,
    const float* gateI,
    const float* gateC,
    const float* gateO,
    const float* prevC,
    const float* newC,
    float* dGates,
    float* dPrevC,
    int batch,
    int hiddenSize)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batch * hiddenSize;

    if (gid >= totalElements) return;

    int b = gid / hiddenSize;
    int h = gid % hiddenSize;

    float f = gateF[gid];
    float i_gate = gateI[gid];
    float c_candidate = gateC[gid];
    float o = gateO[gid];
    float prevCVal = prevC[gid];
    float newCVal = newC[gid];

    float dh = dH[gid];
    float tanh_c = tanhf(newCVal);

    float dO = dh * tanh_c * sigmoid_derivative(o);
    float dC_from_H = dh * o * tanh_derivative(tanh_c);
    float dC = dC_next[gid] + dC_from_H;

    float dF = dC * prevCVal * sigmoid_derivative(f);
    float dI = dC * c_candidate * sigmoid_derivative(i_gate);
    float dCCandidate = dC * i_gate * tanh_derivative(c_candidate);

    int gateOffset = b * 4 * hiddenSize;
    dGates[gateOffset + h] = dF;
    dGates[gateOffset + hiddenSize + h] = dI;
    dGates[gateOffset + 2 * hiddenSize + h] = dCCandidate;
    dGates[gateOffset + 3 * hiddenSize + h] = dO;

    dPrevC[gid] = dC * f;
}

extern ""C"" __global__ void lstm_accumulate_weight_gradients(
    const float* input,
    const float* prevH,
    const float* dGates,
    float* dWi,
    float* dWh,
    float* dBias,
    int batch,
    int inputSize,
    int hiddenSize)
{
    int gateIdx = blockIdx.x;
    int colIdx = blockIdx.y * blockDim.x + threadIdx.x;

    if (gateIdx >= 4 * hiddenSize) return;

    if (colIdx < inputSize) {
        float grad = 0.0f;
        for (int b = 0; b < batch; b++) {
            float dGate = dGates[b * 4 * hiddenSize + gateIdx];
            float x_val = input[b * inputSize + colIdx];
            grad += dGate * x_val;
        }
        atomicAdd(&dWi[gateIdx * inputSize + colIdx], grad);
    }

    if (colIdx < hiddenSize) {
        float grad = 0.0f;
        for (int b = 0; b < batch; b++) {
            float dGate = dGates[b * 4 * hiddenSize + gateIdx];
            float h_val = prevH[b * hiddenSize + colIdx];
            grad += dGate * h_val;
        }
        atomicAdd(&dWh[gateIdx * hiddenSize + colIdx], grad);
    }

    if (colIdx == 0) {
        float grad = 0.0f;
        for (int b = 0; b < batch; b++) {
            grad += dGates[b * 4 * hiddenSize + gateIdx];
        }
        atomicAdd(&dBias[gateIdx], grad);
    }
}

extern ""C"" __global__ void lstm_backward_sequence(
    const float* gradOutput,
    const float* h_states,
    const float* c_states,
    const float* gates,
    const float* c_init,
    const float* h_init,
    const float* input,
    const float* Wi,
    const float* Wh,
    float* gradInput,
    float* dWi,
    float* dWh,
    float* dBias,
    float* dH_init,
    float* dC_init,
    int batch,
    int timeSteps,
    int inputSize,
    int hiddenSize)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batch * hiddenSize;
    int b = gid / hiddenSize;
    int h_idx = gid % hiddenSize;

    // Initialize dH_init buffer for use as intermediate storage during BPTT
    // (all threads participate for sync safety)
    if (gid < totalElements) {
        dH_init[gid] = 0.0f;
    }
    __syncthreads();

    if (gid >= totalElements) return;

    float dH = 0.0f;
    float dC = 0.0f;

    for (int t = timeSteps - 1; t >= 0; t--) {
        // Read accumulated recurrent gradient from previous iteration (if any)
        if (t < timeSteps - 1) {
            dH = dH_init[gid];
            dH_init[gid] = 0.0f;  // Clear for next accumulation
        }
        __syncthreads();

        dH += gradOutput[(b * timeSteps + t) * hiddenSize + h_idx];

        int gateOffset = t * batch * 4 * hiddenSize + b * 4 * hiddenSize;
        float f = gates[gateOffset + h_idx];
        float i_gate = gates[gateOffset + hiddenSize + h_idx];
        float c_candidate = gates[gateOffset + 2 * hiddenSize + h_idx];
        float o = gates[gateOffset + 3 * hiddenSize + h_idx];

        int stateOffset = t * batch * hiddenSize + gid;
        float c_t = c_states[stateOffset];
        float c_prev;
        if (t == 0) {
            c_prev = c_init[gid];
        } else {
            c_prev = c_states[(t - 1) * batch * hiddenSize + gid];
        }

        float tanh_c = tanhf(c_t);
        float dO = dH * tanh_c * sigmoid_derivative(o);
        float dC_from_H = dH * o * tanh_derivative(tanh_c);
        dC += dC_from_H;

        float dF = dC * c_prev * sigmoid_derivative(f);
        float dI = dC * c_candidate * sigmoid_derivative(i_gate);
        float dCCandidate = dC * i_gate * tanh_derivative(c_candidate);
        float dC_prev = dC * f;

        int inputOffset = (b * timeSteps + t) * inputSize;
        for (int i = 0; i < inputSize; i++) {
            float x_val = input[inputOffset + i];
            atomicAdd(&dWi[h_idx * inputSize + i], dF * x_val);
            atomicAdd(&dWi[(hiddenSize + h_idx) * inputSize + i], dI * x_val);
            atomicAdd(&dWi[(2 * hiddenSize + h_idx) * inputSize + i], dCCandidate * x_val);
            atomicAdd(&dWi[(3 * hiddenSize + h_idx) * inputSize + i], dO * x_val);
        }

        for (int j = 0; j < hiddenSize; j++) {
            float hj;
            if (t == 0) {
                hj = h_init[b * hiddenSize + j];
            } else {
                hj = h_states[(t - 1) * batch * hiddenSize + b * hiddenSize + j];
            }
            atomicAdd(&dWh[h_idx * hiddenSize + j], dF * hj);
            atomicAdd(&dWh[(hiddenSize + h_idx) * hiddenSize + j], dI * hj);
            atomicAdd(&dWh[(2 * hiddenSize + h_idx) * hiddenSize + j], dCCandidate * hj);
            atomicAdd(&dWh[(3 * hiddenSize + h_idx) * hiddenSize + j], dO * hj);
        }

        atomicAdd(&dBias[h_idx], dF);
        atomicAdd(&dBias[hiddenSize + h_idx], dI);
        atomicAdd(&dBias[2 * hiddenSize + h_idx], dCCandidate);
        atomicAdd(&dBias[3 * hiddenSize + h_idx], dO);

        int gradInputOffset = (b * timeSteps + t) * inputSize;
        for (int i = 0; i < inputSize; i++) {
            float grad_i = 0.0f;
            grad_i += dF * Wi[h_idx * inputSize + i];
            grad_i += dI * Wi[(hiddenSize + h_idx) * inputSize + i];
            grad_i += dCCandidate * Wi[(2 * hiddenSize + h_idx) * inputSize + i];
            grad_i += dO * Wi[(3 * hiddenSize + h_idx) * inputSize + i];
            atomicAdd(&gradInput[gradInputOffset + i], grad_i);
        }

        // Gradient to previous hidden state for BPTT
        // dH_prev[j] = sum_k (dGate[k] * Wh[k, j]) for all four gates
        // This is a matrix-vector product: dH_prev = Wh^T @ dGates
        // Accumulate to dH_init buffer (used as temp storage during loop, final output at t=0)
        for (int j = 0; j < hiddenSize; j++) {
            // Contribution from gate derivatives at position h_idx to hidden unit j
            // Wh layout: [4*hiddenSize, hiddenSize], so Wh[k, j] = Wh[k * hiddenSize + j]
            float contrib = dF * Wh[h_idx * hiddenSize + j];
            contrib += dI * Wh[(hiddenSize + h_idx) * hiddenSize + j];
            contrib += dCCandidate * Wh[(2 * hiddenSize + h_idx) * hiddenSize + j];
            contrib += dO * Wh[(3 * hiddenSize + h_idx) * hiddenSize + j];
            atomicAdd(&dH_init[b * hiddenSize + j], contrib);
        }

        // Update dC for next iteration
        dC = dC_prev;

        __syncthreads();
    }

    // Store initial cell state gradient
    // dH_init already contains the accumulated gradient for h_init from the t=0 iteration
    dC_init[gid] = dC;
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
            "lstm_cell_forward",
            "lstm_forward_sequence",
            "lstm_backward_input",
            "lstm_backward_prevh",
            "lstm_compute_gate_gradients",
            "lstm_accumulate_weight_gradients",
            "lstm_backward_sequence"
        };
    }
}
