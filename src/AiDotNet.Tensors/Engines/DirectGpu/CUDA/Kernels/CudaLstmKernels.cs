// Copyright (c) AiDotNet. All rights reserved.
// CUDA kernels for standard LSTM (Long Short-Term Memory) neural network operations.
// Implements sequence-level forward and backward passes for efficient BPTT.

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

/// <summary>
/// CUDA kernels for sequence-level LSTM operations.
/// Implements full forward and backward passes for LSTM layers processing entire sequences.
/// </summary>
/// <remarks>
/// LSTM equations:
/// f_t = sigmoid(W_f * x_t + U_f * h_{t-1} + b_f)  // forget gate
/// i_t = sigmoid(W_i * x_t + U_i * h_{t-1} + b_i)  // input gate
/// c̃_t = tanh(W_c * x_t + U_c * h_{t-1} + b_c)     // candidate cell
/// o_t = sigmoid(W_o * x_t + U_o * h_{t-1} + b_o)  // output gate
/// c_t = f_t * c_{t-1} + i_t * c̃_t                 // cell state
/// h_t = o_t * tanh(c_t)                           // hidden state
/// </remarks>
internal static class CudaLstmKernels
{
    public static string GetSource()
    {
        return @"
#include <math.h>

#define EPSILON 1e-15f
#define WARP_SIZE 32

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

// LSTM cell forward pass for a single time step
// input: [batch, inputSize]
// prevH: [batch, hiddenSize]
// prevC: [batch, hiddenSize]
// Wi: [4 * hiddenSize, inputSize] (Wf, Wi, Wc, Wo stacked)
// Wh: [4 * hiddenSize, hiddenSize] (Uf, Ui, Uc, Uo stacked)
// bias: [4 * hiddenSize]
// output newH, newC: [batch, hiddenSize]
// gateF, gateI, gateC, gateO: [batch, hiddenSize] (cached for backward)
extern ""C"" __global__ void lstm_cell_forward(
    const float* input,
    const float* prevH,
    const float* prevC,
    const float* Wi,      // [4*hidden, input]
    const float* Wh,      // [4*hidden, hidden]
    const float* bias,    // [4*hidden]
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

    // Compute gate pre-activations via dot products
    float sumF = bias[h];
    float sumI = bias[hiddenSize + h];
    float sumC = bias[2 * hiddenSize + h];
    float sumO = bias[3 * hiddenSize + h];

    // Input contribution: Wi * x
    for (int i = 0; i < inputSize; i++) {
        float x_val = input[b * inputSize + i];
        sumF += Wi[h * inputSize + i] * x_val;
        sumI += Wi[(hiddenSize + h) * inputSize + i] * x_val;
        sumC += Wi[(2 * hiddenSize + h) * inputSize + i] * x_val;
        sumO += Wi[(3 * hiddenSize + h) * inputSize + i] * x_val;
    }

    // Hidden contribution: Wh * h_prev
    for (int j = 0; j < hiddenSize; j++) {
        float h_val = prevH[b * hiddenSize + j];
        sumF += Wh[h * hiddenSize + j] * h_val;
        sumI += Wh[(hiddenSize + h) * hiddenSize + j] * h_val;
        sumC += Wh[(2 * hiddenSize + h) * hiddenSize + j] * h_val;
        sumO += Wh[(3 * hiddenSize + h) * hiddenSize + j] * h_val;
    }

    // Apply activations
    float f = sigmoid(sumF);
    float i_gate = sigmoid(sumI);
    float c_candidate = tanhf(sumC);
    float o = sigmoid(sumO);

    // Compute new cell state: newC = f * prevC + i * c_candidate
    float prevCVal = prevC[gid];
    float newCVal = f * prevCVal + i_gate * c_candidate;

    // Compute new hidden state: newH = o * tanh(newC)
    float newHVal = o * tanhf(newCVal);

    // Store outputs
    newC[gid] = newCVal;
    newH[gid] = newHVal;

    // Store gate values for backward pass
    gateF[gid] = f;
    gateI[gid] = i_gate;
    gateC[gid] = c_candidate;
    gateO[gid] = o;
}

// ===========================================================================
// LSTM SEQUENCE FORWARD KERNEL
// ===========================================================================

// LSTM forward pass for entire sequence
// Processes all timesteps sequentially within the kernel
// input: [batch, timeSteps, inputSize]
// h_init: [batch, hiddenSize]
// c_init: [batch, hiddenSize]
// output: [batch, timeSteps, hiddenSize]
// h_states: [timeSteps, batch, hiddenSize] (cached for backward)
// c_states: [timeSteps, batch, hiddenSize] (cached for backward)
// gates: [timeSteps, batch, 4*hiddenSize] (F,I,C,O cached for backward)
extern ""C"" __global__ void lstm_forward_sequence(
    const float* input,
    const float* h_init,
    const float* c_init,
    const float* Wi,      // [4*hidden, input]
    const float* Wh,      // [4*hidden, hidden]
    const float* bias,    // [4*hidden]
    float* output,
    float* h_states,      // Cache: [timeSteps, batch, hidden]
    float* c_states,      // Cache: [timeSteps, batch, hidden]
    float* gates,         // Cache: [timeSteps, batch, 4*hidden]
    int batch,
    int timeSteps,
    int inputSize,
    int hiddenSize)
{
    // Each thread handles one (batch, hidden) element
    // Outer loop over timesteps is sequential
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batch * hiddenSize;

    // Use isValid flag instead of early return to avoid __syncthreads deadlock
    bool isValid = gid < totalElements;

    int b = isValid ? (gid / hiddenSize) : 0;
    int h_idx = isValid ? (gid % hiddenSize) : 0;

    // Initialize states (only valid threads)
    float h_val = isValid ? h_init[gid] : 0.0f;
    float c_val = isValid ? c_init[gid] : 0.0f;

    // Process each timestep
    for (int t = 0; t < timeSteps; t++) {
        // Compute gate pre-activations (only valid threads)
        float sumF = 0.0f, sumI = 0.0f, sumC = 0.0f, sumO = 0.0f;
        float f = 0.0f, i_gate = 0.0f, c_candidate = 0.0f, o = 0.0f;
        float prev_c = 0.0f;

        if (isValid) {
            sumF = bias[h_idx];
            sumI = bias[hiddenSize + h_idx];
            sumC = bias[2 * hiddenSize + h_idx];
            sumO = bias[3 * hiddenSize + h_idx];

            // Input contribution: Wi * x_t
            int inputOffset = (b * timeSteps + t) * inputSize;
            for (int i = 0; i < inputSize; i++) {
                float x_val = input[inputOffset + i];
                sumF += Wi[h_idx * inputSize + i] * x_val;
                sumI += Wi[(hiddenSize + h_idx) * inputSize + i] * x_val;
                sumC += Wi[(2 * hiddenSize + h_idx) * inputSize + i] * x_val;
                sumO += Wi[(3 * hiddenSize + h_idx) * inputSize + i] * x_val;
            }

            // Hidden contribution: Wh * h_prev
            // Need to read h_val from all hidden units - use shared memory for efficiency
            for (int j = 0; j < hiddenSize; j++) {
                // Read from prev timestep's stored h, or from h_val if same element
                float hj;
                if (t == 0) {
                    hj = h_init[b * hiddenSize + j];
                } else {
                    // Read from cached h_states for previous timestep
                    hj = h_states[(t - 1) * batch * hiddenSize + b * hiddenSize + j];
                }
                sumF += Wh[h_idx * hiddenSize + j] * hj;
                sumI += Wh[(hiddenSize + h_idx) * hiddenSize + j] * hj;
                sumC += Wh[(2 * hiddenSize + h_idx) * hiddenSize + j] * hj;
                sumO += Wh[(3 * hiddenSize + h_idx) * hiddenSize + j] * hj;
            }

            // Apply activations
            f = sigmoid(sumF);
            i_gate = sigmoid(sumI);
            c_candidate = tanhf(sumC);
            o = sigmoid(sumO);

            // Previous cell state
            if (t == 0) {
                prev_c = c_init[gid];
            } else {
                prev_c = c_states[(t - 1) * batch * hiddenSize + gid];
            }

            // Update cell state
            c_val = f * prev_c + i_gate * c_candidate;

            // Update hidden state
            h_val = o * tanhf(c_val);

            // Store states for output and caching
            int stateOffset = t * batch * hiddenSize + gid;
            h_states[stateOffset] = h_val;
            c_states[stateOffset] = c_val;

            // Store output
            output[(b * timeSteps + t) * hiddenSize + h_idx] = h_val;

            // Store gates for backward pass
            int gateOffset = t * batch * 4 * hiddenSize + b * 4 * hiddenSize;
            gates[gateOffset + h_idx] = f;
            gates[gateOffset + hiddenSize + h_idx] = i_gate;
            gates[gateOffset + 2 * hiddenSize + h_idx] = c_candidate;
            gates[gateOffset + 3 * hiddenSize + h_idx] = o;
        }

        // Sync to ensure all threads have written h_states before next iteration
        // All threads (valid and invalid) must reach this barrier
        __syncthreads();
    }
}

// ===========================================================================
// LSTM CELL BACKWARD KERNEL (Single Timestep)
// ===========================================================================

// Computes gradients for a single LSTM cell timestep
// dH: gradient from output [batch, hiddenSize]
// dC_next: gradient from next cell state [batch, hiddenSize]
// Returns: dInput, dPrevH, dPrevC, and accumulated weight/bias gradients
extern ""C"" __global__ void lstm_cell_backward(
    const float* dH,          // [batch, hidden]
    const float* dC_next,     // [batch, hidden] gradient from next timestep
    const float* gateF,       // [batch, hidden]
    const float* gateI,       // [batch, hidden]
    const float* gateC,       // [batch, hidden]
    const float* gateO,       // [batch, hidden]
    const float* prevC,       // [batch, hidden]
    const float* newC,        // [batch, hidden]
    const float* prevH,       // [batch, hidden]
    const float* input,       // [batch, input]
    const float* Wi,          // [4*hidden, input]
    const float* Wh,          // [4*hidden, hidden]
    float* dPrevH,            // [batch, hidden]
    float* dPrevC,            // [batch, hidden]
    float* dInput,            // [batch, input]
    float* dWi,               // [4*hidden, input] - atomic add
    float* dWh,               // [4*hidden, hidden] - atomic add
    float* dBias,             // [4*hidden] - atomic add
    int batch,
    int inputSize,
    int hiddenSize)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batch * hiddenSize;

    if (gid >= totalElements) return;

    int b = gid / hiddenSize;
    int h = gid % hiddenSize;

    // Get cached values
    float f = gateF[gid];
    float i_gate = gateI[gid];
    float c_candidate = gateC[gid];
    float o = gateO[gid];
    float prevCVal = prevC[gid];
    float newCVal = newC[gid];

    // Gradient from output hidden state
    float dh = dH[gid];

    // tanh(c_t)
    float tanh_c = tanhf(newCVal);

    // Gradient through output gate
    float dO = dh * tanh_c * sigmoid_derivative(o);

    // Gradient to cell state from hidden state
    float dC_from_H = dh * o * tanh_derivative(tanh_c);

    // Total cell state gradient (from next timestep + from output)
    float dC = dC_next[gid] + dC_from_H;

    // Gradient through cell state equation: c_t = f * c_{t-1} + i * c_candidate
    float dF = dC * prevCVal * sigmoid_derivative(f);
    float dI = dC * c_candidate * sigmoid_derivative(i_gate);
    float dCCandidate = dC * i_gate * tanh_derivative(c_candidate);

    // Gradient to previous cell state
    dPrevC[gid] = dC * f;

    // Gradient to previous hidden state (through all gates)
    // dPrevH[b,j] = sum_h (dF_h * Wh[h,j] + dI_h * Wh[H+h,j] + dCCandidate_h * Wh[2H+h,j] + dO_h * Wh[3H+h,j])
    // This thread handles output position h, so we contribute to each j using atomicAdd
    for (int j = 0; j < hiddenSize; j++) {
        float contrib = dF * Wh[h * hiddenSize + j];
        contrib += dI * Wh[(hiddenSize + h) * hiddenSize + j];
        contrib += dCCandidate * Wh[(2 * hiddenSize + h) * hiddenSize + j];
        contrib += dO * Wh[(3 * hiddenSize + h) * hiddenSize + j];
        atomicAdd(&dPrevH[b * hiddenSize + j], contrib);
    }

    // Accumulate weight gradients (using atomic operations)
    for (int i = 0; i < inputSize; i++) {
        float x_val = input[b * inputSize + i];
        atomicAdd(&dWi[h * inputSize + i], dF * x_val);
        atomicAdd(&dWi[(hiddenSize + h) * inputSize + i], dI * x_val);
        atomicAdd(&dWi[(2 * hiddenSize + h) * inputSize + i], dCCandidate * x_val);
        atomicAdd(&dWi[(3 * hiddenSize + h) * inputSize + i], dO * x_val);
    }

    // Accumulate bias gradients
    atomicAdd(&dBias[h], dF);
    atomicAdd(&dBias[hiddenSize + h], dI);
    atomicAdd(&dBias[2 * hiddenSize + h], dCCandidate);
    atomicAdd(&dBias[3 * hiddenSize + h], dO);
}

// ===========================================================================
// LSTM BACKWARD INPUT KERNEL
// ===========================================================================

// Computes gradient with respect to input
extern ""C"" __global__ void lstm_backward_input(
    const float* dGates,      // [batch, 4*hidden] - dF, dI, dC, dO concatenated
    const float* Wi,          // [4*hidden, input]
    float* dInput,            // [batch, input]
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

    // Accumulate gradients from all 4 gates
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

// ===========================================================================
// LSTM BACKWARD PREV HIDDEN KERNEL
// ===========================================================================

// Computes gradient with respect to previous hidden state
extern ""C"" __global__ void lstm_backward_prevh(
    const float* dGates,      // [batch, 4*hidden]
    const float* Wh,          // [4*hidden, hidden]
    float* dPrevH,            // [batch, hidden]
    int batch,
    int hiddenSize)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batch * hiddenSize;

    if (gid >= totalElements) return;

    int b = gid / hiddenSize;
    int j = gid % hiddenSize;

    float grad = 0.0f;

    // Accumulate gradients from all 4 gates
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

// ===========================================================================
// LSTM SEQUENCE BACKWARD KERNEL
// ===========================================================================

// LSTM backward pass for entire sequence (BPTT)
// Processes timesteps in reverse order
extern ""C"" __global__ void lstm_backward_sequence(
    const float* gradOutput,  // [batch, timeSteps, hidden]
    const float* h_states,    // [timeSteps, batch, hidden]
    const float* c_states,    // [timeSteps, batch, hidden]
    const float* gates,       // [timeSteps, batch, 4*hidden]
    const float* c_init,      // [batch, hidden]
    const float* h_init,      // [batch, hidden]
    const float* input,       // [batch, timeSteps, input]
    const float* Wi,          // [4*hidden, input]
    const float* Wh,          // [4*hidden, hidden]
    float* gradInput,         // [batch, timeSteps, input]
    float* dWi,               // [4*hidden, input]
    float* dWh,               // [4*hidden, hidden]
    float* dBias,             // [4*hidden]
    float* dH_init,           // [batch, hidden]
    float* dC_init,           // [batch, hidden]
    int batch,
    int timeSteps,
    int inputSize,
    int hiddenSize)
{
    // Each thread handles one (batch, hidden) element
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batch * hiddenSize;

    // Use isValid flag instead of early return to avoid __syncthreads deadlock
    bool isValid = gid < totalElements;

    int b = isValid ? (gid / hiddenSize) : 0;
    int h_idx = isValid ? (gid % hiddenSize) : 0;

    // Initialize gradients for recurrence
    float dH = 0.0f;
    float dC = 0.0f;

    // Clear dH_init buffer for use as intermediate storage during BPTT (only valid threads)
    if (isValid) {
        dH_init[gid] = 0.0f;
    }
    __syncthreads();

    // Process timesteps in reverse (BPTT)
    for (int t = timeSteps - 1; t >= 0; t--) {
        // Read accumulated recurrent gradient from previous iteration (if any)
        if (isValid && t < timeSteps - 1) {
            dH = dH_init[gid];
            dH_init[gid] = 0.0f;  // Clear for next accumulation
        }
        __syncthreads();

        if (isValid) {
            // Add gradient from output at this timestep
            dH += gradOutput[(b * timeSteps + t) * hiddenSize + h_idx];

            // Get cached gate values
            int gateOffset = t * batch * 4 * hiddenSize + b * 4 * hiddenSize;
            float f = gates[gateOffset + h_idx];
            float i_gate = gates[gateOffset + hiddenSize + h_idx];
            float c_candidate = gates[gateOffset + 2 * hiddenSize + h_idx];
            float o = gates[gateOffset + 3 * hiddenSize + h_idx];

            // Get cell states
            int stateOffset = t * batch * hiddenSize + gid;
            float c_t = c_states[stateOffset];
            float c_prev;
            if (t == 0) {
                c_prev = c_init[gid];
            } else {
                c_prev = c_states[(t - 1) * batch * hiddenSize + gid];
            }

            // tanh(c_t)
            float tanh_c = tanhf(c_t);

            // Gradient through output gate
            float dO = dH * tanh_c * sigmoid_derivative(o);

            // Gradient to cell state from hidden state
            float dC_from_H = dH * o * tanh_derivative(tanh_c);

            // Total cell state gradient
            dC += dC_from_H;

            // Gradient through cell state equation
            float dF = dC * c_prev * sigmoid_derivative(f);
            float dI = dC * c_candidate * sigmoid_derivative(i_gate);
            float dCCandidate = dC * i_gate * tanh_derivative(c_candidate);

            // Gradient to previous cell state for next iteration
            float dC_prev = dC * f;

            // Get previous hidden state for weight gradients
            float h_prev_val;
            if (t == 0) {
                h_prev_val = h_init[b * hiddenSize + h_idx];
            } else {
                h_prev_val = h_states[(t - 1) * batch * hiddenSize + gid];
            }

            // Accumulate weight gradients (atomic for multi-thread safety)
            int inputOffset = (b * timeSteps + t) * inputSize;
            for (int i = 0; i < inputSize; i++) {
                float x_val = input[inputOffset + i];
                atomicAdd(&dWi[h_idx * inputSize + i], dF * x_val);
                atomicAdd(&dWi[(hiddenSize + h_idx) * inputSize + i], dI * x_val);
                atomicAdd(&dWi[(2 * hiddenSize + h_idx) * inputSize + i], dCCandidate * x_val);
                atomicAdd(&dWi[(3 * hiddenSize + h_idx) * inputSize + i], dO * x_val);
            }

            // Hidden weight gradients - need all prev hidden values
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

            // Bias gradients
            atomicAdd(&dBias[h_idx], dF);
            atomicAdd(&dBias[hiddenSize + h_idx], dI);
            atomicAdd(&dBias[2 * hiddenSize + h_idx], dCCandidate);
            atomicAdd(&dBias[3 * hiddenSize + h_idx], dO);

            // Compute gradient to input at this timestep
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
            // Each thread k contributes: dGates[k] * Wh[k, j] for all j
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
        }

        // All threads must reach this barrier
        __syncthreads();
    }

    // Store initial cell state gradient (only valid threads)
    // dH_init already contains the accumulated gradient for h_init from the t=0 iteration
    if (isValid) {
        dC_init[gid] = dC;
    }
}

// ===========================================================================
// LSTM WEIGHT GRADIENT ACCUMULATION KERNEL
// ===========================================================================

// Accumulates weight gradients across batch dimension
extern ""C"" __global__ void lstm_accumulate_weight_gradients(
    const float* input,       // [batch, inputSize]
    const float* prevH,       // [batch, hiddenSize]
    const float* dGates,      // [batch, 4*hiddenSize]
    float* dWi,               // [4*hiddenSize, inputSize]
    float* dWh,               // [4*hiddenSize, hiddenSize]
    float* dBias,             // [4*hiddenSize]
    int batch,
    int inputSize,
    int hiddenSize)
{
    // Grid: (4*hidden, max(input, hidden))
    int gateIdx = blockIdx.x;  // Which gate row
    int colIdx = blockIdx.y * blockDim.x + threadIdx.x;

    if (gateIdx >= 4 * hiddenSize) return;

    // Accumulate Wi gradients
    if (colIdx < inputSize) {
        float grad = 0.0f;
        for (int b = 0; b < batch; b++) {
            float dGate = dGates[b * 4 * hiddenSize + gateIdx];
            float x_val = input[b * inputSize + colIdx];
            grad += dGate * x_val;
        }
        atomicAdd(&dWi[gateIdx * inputSize + colIdx], grad);
    }

    // Accumulate Wh gradients
    if (colIdx < hiddenSize) {
        float grad = 0.0f;
        for (int b = 0; b < batch; b++) {
            float dGate = dGates[b * 4 * hiddenSize + gateIdx];
            float h_val = prevH[b * hiddenSize + colIdx];
            grad += dGate * h_val;
        }
        atomicAdd(&dWh[gateIdx * hiddenSize + colIdx], grad);
    }

    // Accumulate bias gradients (only first column handles this)
    if (colIdx == 0) {
        float grad = 0.0f;
        for (int b = 0; b < batch; b++) {
            grad += dGates[b * 4 * hiddenSize + gateIdx];
        }
        atomicAdd(&dBias[gateIdx], grad);
    }
}

// ===========================================================================
// LSTM COMPUTE GATE GRADIENTS KERNEL
// ===========================================================================

// Computes gate gradients from hidden and cell gradients
extern ""C"" __global__ void lstm_compute_gate_gradients(
    const float* dH,          // [batch, hidden]
    const float* dC_next,     // [batch, hidden]
    const float* gateF,       // [batch, hidden]
    const float* gateI,       // [batch, hidden]
    const float* gateC,       // [batch, hidden]
    const float* gateO,       // [batch, hidden]
    const float* prevC,       // [batch, hidden]
    const float* newC,        // [batch, hidden]
    float* dGates,            // [batch, 4*hidden] - output
    float* dPrevC,            // [batch, hidden] - output
    int batch,
    int hiddenSize)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batch * hiddenSize;

    if (gid >= totalElements) return;

    int b = gid / hiddenSize;
    int h = gid % hiddenSize;

    // Get cached values
    float f = gateF[gid];
    float i_gate = gateI[gid];
    float c_candidate = gateC[gid];
    float o = gateO[gid];
    float prevCVal = prevC[gid];
    float newCVal = newC[gid];

    // Gradient from output
    float dh = dH[gid];

    // tanh(c_t)
    float tanh_c = tanhf(newCVal);

    // Gradient through output gate
    float dO = dh * tanh_c * sigmoid_derivative(o);

    // Gradient to cell state from hidden state
    float dC_from_H = dh * o * tanh_derivative(tanh_c);

    // Total cell state gradient
    float dC = dC_next[gid] + dC_from_H;

    // Gradient through cell state equation
    float dF = dC * prevCVal * sigmoid_derivative(f);
    float dI = dC * c_candidate * sigmoid_derivative(i_gate);
    float dCCandidate = dC * i_gate * tanh_derivative(c_candidate);

    // Store gate gradients
    int gateOffset = b * 4 * hiddenSize;
    dGates[gateOffset + h] = dF;
    dGates[gateOffset + hiddenSize + h] = dI;
    dGates[gateOffset + 2 * hiddenSize + h] = dCCandidate;
    dGates[gateOffset + 3 * hiddenSize + h] = dO;

    // Gradient to previous cell state
    dPrevC[gid] = dC * f;
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
            "lstm_cell_backward",
            "lstm_backward_input",
            "lstm_backward_prevh",
            "lstm_backward_sequence",
            "lstm_accumulate_weight_gradients",
            "lstm_compute_gate_gradients"
        };
    }
}
