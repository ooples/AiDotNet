// Copyright (c) AiDotNet. All rights reserved.
// OpenCL kernels for GRU (Gated Recurrent Unit) sequence neural network operations.
// Implements sequence-level forward and backward passes for efficient BPTT training.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;

/// <summary>
/// OpenCL kernels for GRU sequence operations used in recurrent neural networks.
/// Implements full forward and backward passes for GRU cells with 3 gates (reset, update, new).
/// These are sequence-level kernels that process all timesteps efficiently for BPTT.
/// </summary>
internal static class GruKernels
{
    public static string GetSource()
    {
        return @"
#define EPSILON 1e-15f

// ===========================================================================
// ACTIVATION FUNCTIONS
// ===========================================================================

inline float sigmoid_fn(float x) {
    return 1.0f / (1.0f + exp(-x));
}

inline float sigmoid_derivative(float sigmoid_output) {
    return sigmoid_output * (1.0f - sigmoid_output);
}

inline float tanh_derivative(float tanh_output) {
    return 1.0f - tanh_output * tanh_output;
}

// ===========================================================================
// GRU CELL FORWARD KERNEL (Single Timestep)
// ===========================================================================
// Processes one GRU cell computation for a single timestep.
// Each thread handles one (batch, hidden) element.

__kernel void gru_cell_forward(
    __global const float* input,       // [batch, input_size]
    __global const float* prevH,       // [batch, hidden_size]
    __global const float* weightsIh,   // [3 * hidden_size, input_size] - input to hidden weights
    __global const float* weightsHh,   // [3 * hidden_size, hidden_size] - hidden to hidden weights
    __global const float* biasIh,      // [3 * hidden_size] - input to hidden bias
    __global const float* biasHh,      // [3 * hidden_size] - hidden to hidden bias
    __global float* newH,              // [batch, hidden_size]
    __global float* gateR,             // [batch, hidden_size] - reset gate cache
    __global float* gateZ,             // [batch, hidden_size] - update gate cache
    __global float* gateN,             // [batch, hidden_size] - new gate (candidate) cache
    const int batch,
    const int inputSize,
    const int hiddenSize)
{
    int gid = get_global_id(0);
    int totalElements = batch * hiddenSize;

    if (gid >= totalElements) return;

    int b = gid / hiddenSize;
    int h = gid % hiddenSize;

    // Compute gate pre-activations
    // Gates order: reset(r), update(z), new(n)
    float sumR = biasIh[h] + biasHh[h];
    float sumZ = biasIh[hiddenSize + h] + biasHh[hiddenSize + h];
    float sumN_input = biasIh[2 * hiddenSize + h];
    float sumN_hidden = biasHh[2 * hiddenSize + h];

    // Input to hidden contribution
    for (int j = 0; j < inputSize; j++) {
        float inVal = input[b * inputSize + j];
        sumR += inVal * weightsIh[h * inputSize + j];
        sumZ += inVal * weightsIh[(hiddenSize + h) * inputSize + j];
        sumN_input += inVal * weightsIh[(2 * hiddenSize + h) * inputSize + j];
    }

    // Hidden to hidden contribution for r and z gates
    for (int j = 0; j < hiddenSize; j++) {
        float hVal = prevH[b * hiddenSize + j];
        sumR += hVal * weightsHh[h * hiddenSize + j];
        sumZ += hVal * weightsHh[(hiddenSize + h) * hiddenSize + j];
    }

    // Apply activations for r and z
    float r = sigmoid_fn(sumR);
    float z = sigmoid_fn(sumZ);

    // Hidden to hidden contribution for n gate (uses reset gate)
    for (int j = 0; j < hiddenSize; j++) {
        float hVal = prevH[b * hiddenSize + j];
        sumN_hidden += (r * hVal) * weightsHh[(2 * hiddenSize + h) * hiddenSize + j];
    }

    // New gate activation
    float n = tanh(sumN_input + sumN_hidden);

    // Hidden state update: h_new = (1 - z) * n + z * h_prev
    float prevHVal = prevH[gid];
    float newHVal = (1.0f - z) * n + z * prevHVal;

    // Store results
    newH[gid] = newHVal;
    gateR[gid] = r;
    gateZ[gid] = z;
    gateN[gid] = n;
}

// ===========================================================================
// GRU FORWARD SEQUENCE KERNEL
// ===========================================================================
// Processes the entire sequence in a single kernel launch.

__kernel void gru_forward_sequence(
    __global const float* input,       // [seqLen, batch, input_size]
    __global const float* hInit,       // [batch, hidden_size]
    __global const float* weightsIh,   // [3 * hidden_size, input_size]
    __global const float* weightsHh,   // [3 * hidden_size, hidden_size]
    __global const float* biasIh,      // [3 * hidden_size]
    __global const float* biasHh,      // [3 * hidden_size]
    __global float* output,            // [seqLen, batch, hidden_size]
    __global float* hFinal,            // [batch, hidden_size]
    __global float* allH,              // [seqLen + 1, batch, hidden_size] - all hidden states
    __global float* cacheGates,        // [seqLen, batch, hidden_size, 3] - gate values for backward
    const int seqLen,
    const int batch,
    const int inputSize,
    const int hiddenSize)
{
    int gid = get_global_id(0);
    int totalElements = batch * hiddenSize;
    int b = gid / hiddenSize;
    int h = gid % hiddenSize;
    int isValid = (gid < totalElements) ? 1 : 0;

    // Initialize from hInit (only valid threads)
    float hPrev = 0.0f;
    if (isValid) {
        hPrev = hInit[gid];
        // Store initial state
        allH[gid] = hPrev;
    }

    // Barrier to ensure all threads have written initial hidden state
    // Note: All threads must reach barrier for proper synchronization
    barrier(CLK_GLOBAL_MEM_FENCE);

    // Process each timestep
    for (int t = 0; t < seqLen; t++) {
        if (isValid) {
            // Compute gate pre-activations
            float sumR = biasIh[h] + biasHh[h];
            float sumZ = biasIh[hiddenSize + h] + biasHh[hiddenSize + h];
            float sumN_input = biasIh[2 * hiddenSize + h];
            float sumN_hidden = biasHh[2 * hiddenSize + h];

            // Input contribution at this timestep
            int inputOffset = t * batch * inputSize + b * inputSize;
            for (int j = 0; j < inputSize; j++) {
                float inVal = input[inputOffset + j];
                sumR += inVal * weightsIh[h * inputSize + j];
                sumZ += inVal * weightsIh[(hiddenSize + h) * inputSize + j];
                sumN_input += inVal * weightsIh[(2 * hiddenSize + h) * inputSize + j];
            }

            // Previous hidden state contribution for r and z
            for (int j = 0; j < hiddenSize; j++) {
                float hVal = (j == h) ? hPrev : allH[t * batch * hiddenSize + b * hiddenSize + j];
                sumR += hVal * weightsHh[h * hiddenSize + j];
                sumZ += hVal * weightsHh[(hiddenSize + h) * hiddenSize + j];
            }

            // Apply activations for r and z
            float r = sigmoid_fn(sumR);
            float z = sigmoid_fn(sumZ);

            // Hidden contribution for n gate (gated by reset)
            for (int j = 0; j < hiddenSize; j++) {
                float hVal = (j == h) ? hPrev : allH[t * batch * hiddenSize + b * hiddenSize + j];
                sumN_hidden += (r * hVal) * weightsHh[(2 * hiddenSize + h) * hiddenSize + j];
            }

            // New gate activation
            float n = tanh(sumN_input + sumN_hidden);

            // Hidden state update
            float newH = (1.0f - z) * n + z * hPrev;

            // Store output
            int outIdx = t * batch * hiddenSize + gid;
            output[outIdx] = newH;

            // Store all states for backward pass
            int stateIdx = (t + 1) * batch * hiddenSize + gid;
            allH[stateIdx] = newH;

            // Cache gate values for backward
            int gateIdx = (t * batch * hiddenSize + gid) * 3;
            cacheGates[gateIdx + 0] = r;
            cacheGates[gateIdx + 1] = z;
            cacheGates[gateIdx + 2] = n;

            // Update for next iteration
            hPrev = newH;
        }

        // Barrier to ensure all threads have written new hidden state before next iteration
        // All threads (valid and invalid) must reach this barrier
        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    // Store final state (only valid threads)
    if (isValid) {
        hFinal[gid] = hPrev;
    }
}

// ===========================================================================
// GRU CELL BACKWARD KERNEL (Single Timestep)
// ===========================================================================

__kernel void gru_cell_backward(
    __global const float* gradH,       // [batch, hidden_size] - gradient from next layer
    __global const float* gateR,       // [batch, hidden_size]
    __global const float* gateZ,       // [batch, hidden_size]
    __global const float* gateN,       // [batch, hidden_size]
    __global const float* prevH,       // [batch, hidden_size]
    __global const float* weightsHh,   // [3 * hidden_size, hidden_size] - recurrent weights
    __global float* gradPrevH,         // [batch, hidden_size] - gradient to previous hidden
    __global float* gradGateR,         // [batch, hidden_size] - gradient for reset gate
    __global float* gradGateZ,         // [batch, hidden_size] - gradient for update gate
    __global float* gradGateN,         // [batch, hidden_size] - gradient for new gate
    const int batch,
    const int hiddenSize)
{
    int gid = get_global_id(0);
    int totalElements = batch * hiddenSize;

    if (gid >= totalElements) return;

    int b = gid / hiddenSize;
    int h = gid % hiddenSize;

    float r = gateR[gid];
    float z = gateZ[gid];
    float n = gateN[gid];
    float hPrevLocal = prevH[gid];

    float dH = gradH[gid];

    // Gradient through hidden state update: h_new = (1 - z) * n + z * h_prev
    // dL/dz = dH * (-n + h_prev)
    float dZ = dH * (hPrevLocal - n) * sigmoid_derivative(z);

    // dL/dn = dH * (1 - z)
    float dN = dH * (1.0f - z) * tanh_derivative(n);

    // Compute Wn_hh @ h_prev for reset gate gradient
    // n = tanh(Wn_ih @ x + r * (Wn_hh @ h_prev) + bias)
    // dR = dN * (Wn_hh @ h_prev) * sigmoid_derivative(r)
    float Wn_h_prev_dot = 0.0f;
    for (int j = 0; j < hiddenSize; j++) {
        float hPrevJ = prevH[b * hiddenSize + j];
        // Wn_hh is at offset 2*hiddenSize in weightsHh
        Wn_h_prev_dot += hPrevJ * weightsHh[(2 * hiddenSize + h) * hiddenSize + j];
    }
    float dR = dN * Wn_h_prev_dot * sigmoid_derivative(r);

    // dL/dh_prev from z branch: dH * z
    float dHPrev = dH * z;

    // dL/dh_prev from n branch through reset gate: dN * r * Wn_hh[h, j]
    for (int hh = 0; hh < hiddenSize; hh++) {
        dHPrev += dN * r * weightsHh[(2 * hiddenSize + hh) * hiddenSize + h];
    }

    // dL/dh_prev from r gate: dR * Wr_hh[h, j]
    for (int hh = 0; hh < hiddenSize; hh++) {
        dHPrev += dR * weightsHh[hh * hiddenSize + h];
    }

    // dL/dh_prev from z gate: dZ * Wz_hh[h, j]
    for (int hh = 0; hh < hiddenSize; hh++) {
        dHPrev += dZ * weightsHh[(hiddenSize + hh) * hiddenSize + h];
    }

    // Store results
    gradPrevH[gid] = dHPrev;
    gradGateR[gid] = dR;
    gradGateZ[gid] = dZ;
    gradGateN[gid] = dN;
}

// ===========================================================================
// GRU BACKWARD INPUT GRADIENT KERNEL
// ===========================================================================

__kernel void gru_backward_input(
    __global const float* gradGateR,   // [batch, hidden_size]
    __global const float* gradGateZ,   // [batch, hidden_size]
    __global const float* gradGateN,   // [batch, hidden_size]
    __global const float* weightsIh,   // [3 * hidden_size, input_size]
    __global float* gradInput,         // [batch, input_size]
    const int batch,
    const int inputSize,
    const int hiddenSize)
{
    int gid = get_global_id(0);
    int totalElements = batch * inputSize;

    if (gid >= totalElements) return;

    int b = gid / inputSize;
    int j = gid % inputSize;

    float gradSum = 0.0f;

    for (int h = 0; h < hiddenSize; h++) {
        int batchHiddenIdx = b * hiddenSize + h;

        float dR = gradGateR[batchHiddenIdx];
        float dZ = gradGateZ[batchHiddenIdx];
        float dN = gradGateN[batchHiddenIdx];

        gradSum += dR * weightsIh[h * inputSize + j];
        gradSum += dZ * weightsIh[(hiddenSize + h) * inputSize + j];
        gradSum += dN * weightsIh[(2 * hiddenSize + h) * inputSize + j];
    }

    gradInput[gid] = gradSum;
}

// ===========================================================================
// GRU BACKWARD PREVIOUS HIDDEN GRADIENT KERNEL
// ===========================================================================

__kernel void gru_backward_prevh(
    __global const float* gradGateR,   // [batch, hidden_size]
    __global const float* gradGateZ,   // [batch, hidden_size]
    __global const float* gradGateN,   // [batch, hidden_size]
    __global const float* gradH,       // [batch, hidden_size] - gradient from output
    __global const float* gateR,       // [batch, hidden_size]
    __global const float* gateZ,       // [batch, hidden_size]
    __global const float* weightsHh,   // [3 * hidden_size, hidden_size]
    __global float* gradPrevH,         // [batch, hidden_size]
    const int batch,
    const int hiddenSize)
{
    int gid = get_global_id(0);
    int totalElements = batch * hiddenSize;

    if (gid >= totalElements) return;

    int b = gid / hiddenSize;
    int j = gid % hiddenSize;

    float z = gateZ[gid];
    float dH = gradH[gid];

    // Gradient through z path
    float gradSum = dH * z;

    // Gradient through gates
    for (int h = 0; h < hiddenSize; h++) {
        int batchHiddenIdx = b * hiddenSize + h;

        float dR = gradGateR[batchHiddenIdx];
        float dZ = gradGateZ[batchHiddenIdx];
        float dN = gradGateN[batchHiddenIdx];
        float r = gateR[batchHiddenIdx];

        gradSum += dR * weightsHh[h * hiddenSize + j];
        gradSum += dZ * weightsHh[(hiddenSize + h) * hiddenSize + j];
        gradSum += dN * r * weightsHh[(2 * hiddenSize + h) * hiddenSize + j];
    }

    gradPrevH[gid] = gradSum;
}

// ===========================================================================
// GRU BACKWARD SEQUENCE KERNEL
// ===========================================================================
// Processes backward pass through entire sequence.

__kernel void gru_backward_sequence(
    __global const float* gradOutput,  // [seqLen, batch, hidden_size]
    __global const float* allH,        // [seqLen + 1, batch, hidden_size]
    __global const float* cacheGates,  // [seqLen, batch, hidden_size, 3]
    __global const float* weightsIh,   // [3 * hidden_size, input_size]
    __global const float* weightsHh,   // [3 * hidden_size, hidden_size]
    __global float* gradInput,         // [seqLen, batch, input_size]
    __global float* gradHInit,         // [batch, hidden_size]
    __global float* gradWeightsIh,     // [3 * hidden_size, input_size] - accumulated
    __global float* gradWeightsHh,     // [3 * hidden_size, hidden_size] - accumulated
    __global float* gradBiasIh,        // [3 * hidden_size] - accumulated
    __global float* gradBiasHh,        // [3 * hidden_size] - accumulated
    __global const float* input,       // [seqLen, batch, input_size]
    const int seqLen,
    const int batch,
    const int inputSize,
    const int hiddenSize)
{
    int gid = get_global_id(0);
    int totalElements = batch * hiddenSize;

    if (gid >= totalElements) return;

    int b = gid / hiddenSize;
    int h = gid % hiddenSize;

    // Initialize gradients
    float dH = 0.0f;  // Gradient w.r.t. hidden state (accumulated from next timestep)

    // Process timesteps in reverse
    for (int t = seqLen - 1; t >= 0; t--) {
        // Add gradient from output at this timestep
        int outIdx = t * batch * hiddenSize + gid;
        dH += gradOutput[outIdx];

        // Load cached gate values
        int gateIdx = (t * batch * hiddenSize + gid) * 3;
        float r = cacheGates[gateIdx + 0];
        float z = cacheGates[gateIdx + 1];
        float n = cacheGates[gateIdx + 2];

        // Load hidden state
        int prevStateIdx = t * batch * hiddenSize + gid;
        float hPrev = allH[prevStateIdx];

        // Gradient through hidden state update: h_new = (1 - z) * n + z * h_prev
        float dZ = dH * (hPrev - n) * sigmoid_derivative(z);
        float dN = dH * (1.0f - z) * tanh_derivative(n);

        // Gradient to previous hidden state from z path
        float dHPrev = dH * z;

        // Compute dR: gradient through reset gate from n gate
        // n = tanh(Wn_ih @ x + r * (Wn_hh @ h_prev))
        // dn/dr = tanh_derivative(n) * (Wn_hh @ h_prev) - but dN already includes tanh_derivative
        // So dR = dN * (Wn_hh @ h_prev)[h] * sigmoid_derivative(r)
        float Wn_h_prev_dot = 0.0f;
        for (int j = 0; j < hiddenSize; j++) {
            // Get h_prev for position j
            float hPrevJ = allH[t * batch * hiddenSize + b * hiddenSize + j];
            // Wn_hh has shape [hidden, hidden] starting at offset 2*hiddenSize
            Wn_h_prev_dot += hPrevJ * weightsHh[(2 * hiddenSize + h) * hiddenSize + j];
        }
        float dR = dN * Wn_h_prev_dot * sigmoid_derivative(r);

        // Gradient through n gate to previous hidden (via reset gate)
        // dh_prev[j] += dN * r * Wn_hh[h, j] for each source hidden unit j
        for (int hh = 0; hh < hiddenSize; hh++) {
            dHPrev += dN * r * weightsHh[(2 * hiddenSize + hh) * hiddenSize + h];
        }

        // Gradient through r and z to previous hidden
        for (int hh = 0; hh < hiddenSize; hh++) {
            dHPrev += dR * weightsHh[hh * hiddenSize + h];
            dHPrev += dZ * weightsHh[(hiddenSize + hh) * hiddenSize + h];
        }

        // Update for next iteration
        dH = dHPrev;
    }

    // Store initial state gradient
    gradHInit[gid] = dH;
}

// ===========================================================================
// GRU WEIGHT GRADIENT ACCUMULATION KERNELS
// ===========================================================================

__kernel void gru_accumulate_weight_gradients_ih(
    __global const float* input,        // [seqLen, batch, input_size]
    __global const float* allH,         // [seqLen + 1, batch, hidden_size]
    __global const float* cacheGates,   // [seqLen, batch, hidden_size, 3]
    __global const float* gradOutput,   // [seqLen, batch, hidden_size]
    __global const float* weightsHh,    // [3 * hidden_size, hidden_size] - for proper dR computation
    __global float* gradWeightsIh,      // [3 * hidden_size, input_size]
    const int seqLen,
    const int batch,
    const int inputSize,
    const int hiddenSize)
{
    int gid = get_global_id(0);
    int totalWeights = 3 * hiddenSize * inputSize;

    if (gid >= totalWeights) return;

    int gateIdx = gid / (hiddenSize * inputSize);  // Which gate (0-2)
    int remainder = gid % (hiddenSize * inputSize);
    int h = remainder / inputSize;
    int j = remainder % inputSize;

    float gradSum = 0.0f;

    for (int t = 0; t < seqLen; t++) {
        for (int b = 0; b < batch; b++) {
            int batchHiddenIdx = b * hiddenSize + h;

            // Load cached values
            int cacheIdx = (t * batch * hiddenSize + batchHiddenIdx) * 3;
            float r = cacheGates[cacheIdx + 0];
            float z = cacheGates[cacheIdx + 1];
            float n = cacheGates[cacheIdx + 2];

            // Previous hidden state
            int prevStateIdx = t * batch * hiddenSize + batchHiddenIdx;
            float hPrev = allH[prevStateIdx];

            // Get output gradient
            float dH = gradOutput[t * batch * hiddenSize + batchHiddenIdx];

            // Compute gate gradients
            float dZ = dH * (hPrev - n) * sigmoid_derivative(z);
            float dN = dH * (1.0f - z) * tanh_derivative(n);

            // Compute proper dR using Wn_hh @ h_prev dot product
            float Wn_h_prev_dot = 0.0f;
            for (int jj = 0; jj < hiddenSize; jj++) {
                float hPrevJ = allH[t * batch * hiddenSize + b * hiddenSize + jj];
                Wn_h_prev_dot += hPrevJ * weightsHh[(2 * hiddenSize + h) * hiddenSize + jj];
            }
            float dR = dN * Wn_h_prev_dot * sigmoid_derivative(r);

            // Get input value
            float inputVal = input[t * batch * inputSize + b * inputSize + j];

            // Accumulate based on gate index
            if (gateIdx == 0) {
                gradSum += dR * inputVal;
            } else if (gateIdx == 1) {
                gradSum += dZ * inputVal;
            } else {
                gradSum += dN * inputVal;
            }
        }
    }

    gradWeightsIh[gid] = gradSum;
}

__kernel void gru_accumulate_weight_gradients_hh(
    __global const float* allH,         // [seqLen + 1, batch, hidden_size]
    __global const float* cacheGates,   // [seqLen, batch, hidden_size, 3]
    __global const float* gradOutput,   // [seqLen, batch, hidden_size]
    __global const float* weightsHh,    // [3 * hidden_size, hidden_size] - for proper dR computation
    __global float* gradWeightsHh,      // [3 * hidden_size, hidden_size]
    const int seqLen,
    const int batch,
    const int hiddenSize)
{
    int gid = get_global_id(0);
    int totalWeights = 3 * hiddenSize * hiddenSize;

    if (gid >= totalWeights) return;

    int gateIdx = gid / (hiddenSize * hiddenSize);  // Which gate (0-2)
    int remainder = gid % (hiddenSize * hiddenSize);
    int h = remainder / hiddenSize;
    int colIdx = remainder % hiddenSize;

    float gradSum = 0.0f;

    for (int t = 0; t < seqLen; t++) {
        for (int b = 0; b < batch; b++) {
            int batchHiddenIdx = b * hiddenSize + h;

            // Load cached values
            int cacheIdx = (t * batch * hiddenSize + batchHiddenIdx) * 3;
            float r = cacheGates[cacheIdx + 0];
            float z = cacheGates[cacheIdx + 1];
            float n = cacheGates[cacheIdx + 2];

            // Previous hidden state
            int prevStateIdx = t * batch * hiddenSize + batchHiddenIdx;
            float hPrev = allH[prevStateIdx];
            float hPrevColIdx = allH[t * batch * hiddenSize + b * hiddenSize + colIdx];

            // Get output gradient
            float dH = gradOutput[t * batch * hiddenSize + batchHiddenIdx];

            // Compute gate gradients
            float dZ = dH * (hPrev - n) * sigmoid_derivative(z);
            float dN = dH * (1.0f - z) * tanh_derivative(n);

            // Compute proper dR using Wn_hh @ h_prev dot product
            float Wn_h_prev_dot = 0.0f;
            for (int k = 0; k < hiddenSize; k++) {
                float hPrevK = allH[t * batch * hiddenSize + b * hiddenSize + k];
                Wn_h_prev_dot += hPrevK * weightsHh[(2 * hiddenSize + h) * hiddenSize + k];
            }
            float dR = dN * Wn_h_prev_dot * sigmoid_derivative(r);

            // Accumulate based on gate index
            if (gateIdx == 0) {
                gradSum += dR * hPrevColIdx;
            } else if (gateIdx == 1) {
                gradSum += dZ * hPrevColIdx;
            } else {
                // For n gate, the hidden contribution is gated by r
                gradSum += dN * (r * hPrevColIdx);
            }
        }
    }

    gradWeightsHh[gid] = gradSum;
}

__kernel void gru_accumulate_bias_gradients(
    __global const float* allH,         // [seqLen + 1, batch, hidden_size]
    __global const float* cacheGates,   // [seqLen, batch, hidden_size, 3]
    __global const float* gradOutput,   // [seqLen, batch, hidden_size]
    __global const float* weightsHh,    // [3 * hidden_size, hidden_size] - for proper dR computation
    __global float* gradBias,           // [3 * hidden_size]
    const int seqLen,
    const int batch,
    const int hiddenSize)
{
    int gid = get_global_id(0);
    int totalBiases = 3 * hiddenSize;

    if (gid >= totalBiases) return;

    int gateIdx = gid / hiddenSize;  // Which gate (0-2)
    int h = gid % hiddenSize;

    float gradSum = 0.0f;

    for (int t = 0; t < seqLen; t++) {
        for (int b = 0; b < batch; b++) {
            int batchHiddenIdx = b * hiddenSize + h;

            // Load cached values
            int cacheIdx = (t * batch * hiddenSize + batchHiddenIdx) * 3;
            float r = cacheGates[cacheIdx + 0];
            float z = cacheGates[cacheIdx + 1];
            float n = cacheGates[cacheIdx + 2];

            // Previous hidden state
            int prevStateIdx = t * batch * hiddenSize + batchHiddenIdx;
            float hPrev = allH[prevStateIdx];

            // Get output gradient
            float dH = gradOutput[t * batch * hiddenSize + batchHiddenIdx];

            // Compute gate gradients
            float dZ = dH * (hPrev - n) * sigmoid_derivative(z);
            float dN = dH * (1.0f - z) * tanh_derivative(n);

            // Compute proper dR using Wn_hh @ h_prev dot product
            float Wn_h_prev_dot = 0.0f;
            for (int k = 0; k < hiddenSize; k++) {
                float hPrevK = allH[t * batch * hiddenSize + b * hiddenSize + k];
                Wn_h_prev_dot += hPrevK * weightsHh[(2 * hiddenSize + h) * hiddenSize + k];
            }
            float dR = dN * Wn_h_prev_dot * sigmoid_derivative(r);

            // Accumulate based on gate index
            if (gateIdx == 0) {
                gradSum += dR;
            } else if (gateIdx == 1) {
                gradSum += dZ;
            } else {
                gradSum += dN;
            }
        }
    }

    gradBias[gid] = gradSum;
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
            "gru_cell_backward",
            "gru_backward_input",
            "gru_backward_prevh",
            "gru_backward_sequence",
            "gru_accumulate_weight_gradients_ih",
            "gru_accumulate_weight_gradients_hh",
            "gru_accumulate_bias_gradients"
        };
    }
}
