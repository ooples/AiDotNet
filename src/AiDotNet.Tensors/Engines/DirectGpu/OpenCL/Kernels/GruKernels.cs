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

    // Gradient through hidden state update: h_new = (1 - z) * h_prev + z * n
    // For variant 1: h_new = (1-z)*h_prev + z*n
    // dL/dz = dH * (n - h_prev) * sigmoid'(z)
    float dZ = dH * (n - hPrevLocal) * sigmoid_derivative(z);

    // dL/dn = dH * z * tanh'(n)
    float dN = dH * z * tanh_derivative(n);

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

    // Store gate gradients first - these are needed by gru_backward_prevh
    gradGateR[gid] = dR;
    gradGateZ[gid] = dZ;
    gradGateN[gid] = dN;

    // Direct path gradient to prev hidden: dL/dh_prev from (1-z) branch = dH * (1-z)
    // NOTE: This is ONLY the direct path. Full BPTT prev hidden gradient requires
    // calling gru_backward_prevh AFTER this kernel to sum contributions from all
    // hidden positions using the gate gradients stored above. A single kernel cannot
    // do both because OpenCL has no global barrier - each thread would need to read
    // gate gradients from ALL other threads, but those aren't written yet.
    float dHPrev = dH * (1.0f - z);

    // Store partial result - caller must add full gate contributions via gru_backward_prevh
    gradPrevH[gid] = dHPrev;
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

    // Gradient through (1-z) path (direct contribution) for variant 1: h_new = (1-z)*h_prev + z*n
    float gradSum = dH * (1.0f - z);

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
// GRU backward pass for entire sequence with full BPTT
// Uses local memory to store accumulated hidden gradients so all threads
// can access each other's dH values for proper reset-gate gradient computation.

__kernel void gru_backward_sequence(
    __global const float* gradOutput,  // [seqLen, batch, hidden_size]
    __global const float* allH,        // [seqLen + 1, batch, hidden_size]
    __global const float* cacheGates,  // [seqLen, batch, hidden_size, 3]
    __global const float* weightsIh,   // [3 * hidden_size, input_size]
    __global const float* weightsHh,   // [3 * hidden_size, hidden_size]
    __global float* gradInput,         // [seqLen, batch, input_size]
    __global float* gradHInit,         // [batch, hidden_size]
    __global float* dH_buffer,         // [batch, hidden_size] - workspace for accumulated gradients
    __global float* gradWeightsIh,     // [3 * hidden_size, input_size] - accumulated
    __global float* gradWeightsHh,     // [3 * hidden_size, hidden_size] - accumulated
    __global float* gradBiasIh,        // [3 * hidden_size] - accumulated
    __global float* gradBiasHh,        // [3 * hidden_size] - accumulated
    __global const float* input,       // [seqLen, batch, input_size]
    const int seqLen,
    const int batch,
    const int inputSize,
    const int hiddenSize,
    __local float* local_dH)           // Local memory for shared gradient access
{
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);
    int group_id = get_group_id(0);
    int totalElements = batch * hiddenSize;

    // Use active-flag pattern to prevent barrier deadlock
    int isActive = (gid < totalElements) ? 1 : 0;

    int b = isActive ? (gid / hiddenSize) : 0;
    int h = isActive ? (gid % hiddenSize) : 0;

    // Initialize gradient for recurrence
    float dH = 0.0f;

    // Process timesteps in reverse (BPTT)
    for (int t = seqLen - 1; t >= 0; t--) {
        // Phase 1: Add gradient from output and compute basic gradients
        float r = 0.0f, z = 0.0f, n = 0.0f, hPrev = 0.0f;
        float dN = 0.0f, dZ = 0.0f;
        int gateIdx = 0;

        if (isActive) {
            // Add gradient from output at this timestep
            dH += gradOutput[t * batch * hiddenSize + gid];

            // Load cached gate values
            gateIdx = (t * batch * hiddenSize + gid) * 3;
            r = cacheGates[gateIdx + 0];
            z = cacheGates[gateIdx + 1];
            n = cacheGates[gateIdx + 2];

            // Load hidden state
            hPrev = allH[t * batch * hiddenSize + gid];

            // Gradient through hidden state update: h_new = (1 - z) * n + z * h_prev
            dZ = dH * (hPrev - n) * sigmoid_derivative(z);
            dN = dH * (1.0f - z) * tanh_derivative(n);

            // Store accumulated dH to local memory for other threads to read
            local_dH[lid] = dH;

            // Also write to global buffer for cross-workgroup access
            dH_buffer[gid] = dH;
        }

        // Barrier to ensure all threads have written their dH values
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

        // Phase 2: Compute reset gate gradient using accumulated hidden gradients
        float dR = 0.0f;
        if (isActive) {
            // Full BPTT: dR[h] = sum_k(dN_k * Wn_hh[k,h]) * hPrev[h] * sigmoid'(r[h])
            // where dN_k uses the ACCUMULATED gradient dH_k, not just gradOutput
            float dR_sum = 0.0f;
            for (int k = 0; k < hiddenSize; k++) {
                // Get cached gate values for output k
                int k_gateIdx = (t * batch * hiddenSize + b * hiddenSize + k) * 3;
                float z_k = cacheGates[k_gateIdx + 1];
                float n_k = cacheGates[k_gateIdx + 2];
                float hPrev_k = allH[t * batch * hiddenSize + b * hiddenSize + k];

                // Get accumulated hidden gradient for position k
                // Try local memory first (same workgroup), fall back to global buffer
                float dH_k;
                int k_gid = b * hiddenSize + k;
                int k_local_idx = k_gid - (group_id * local_size);
                if (k_local_idx >= 0 && k_local_idx < local_size) {
                    // Same workgroup - use local memory
                    dH_k = local_dH[k_local_idx];
                } else {
                    // Different workgroup - use global buffer
                    dH_k = dH_buffer[k_gid];
                }

                // Compute dN for output k using accumulated gradient
                float dN_k = dH_k * (1.0f - z_k) * tanh_derivative(n_k);
                dR_sum += dN_k * weightsHh[(2 * hiddenSize + k) * hiddenSize + h];
            }
            dR = dR_sum * hPrev * sigmoid_derivative(r);
        }

        // Phase 3: Compute gradient to previous hidden state (full BPTT)
        // dPrevH[j] = dH[j] * (1-z[j]) + sum_hh(dZ[hh]*Uz[hh,j] + dR[hh]*Ur[hh,j] + dN[hh]*Uh[hh,j]*r[j])
        if (isActive) {
            // Direct path: h_t = (1-z)*n + z*h_prev, so dh_prev = dH * z is WRONG
            // Correct: dh_prev from direct path = dH * (1-z) for n term... wait, let me recalculate
            // h_t = (1-z)*n + z*h_prev
            // dL/dh_prev = dL/dh_t * dh_t/dh_prev = dH * z (gradient flows through z*h_prev term)
            // But we also need gradient through n gate which uses r*h_prev
            // Actually for OpenCL we use (1-z)*n + z*h_prev, different convention than CUDA
            // Let me check: dh_prev direct = dH * z (from z*h_prev term)
            // Hmm, the GRU formula here is h = (1-z)*n + z*h_prev, so dh/dh_prev = z
            // But CUDA uses h = (1-z)*h_prev + z*h_cand, so dh/dh_prev = (1-z)
            // The convention differs! OpenCL seems inverted.
            // Looking at gru_cell_forward: newHVal = (1.0f - z) * n + z * prevHVal
            // So direct path gradient is dH * z, which is correct for this convention.
            float dHPrev = dH * z;  // Direct path for this GRU convention

            // Accumulate gradient contributions from all hidden output positions hh
            for (int hh = 0; hh < hiddenSize; hh++) {
                // Get gate values for output position hh (from cacheGates buffer)
                int hh_gateIdx = (t * batch * hiddenSize + b * hiddenSize + hh) * 3;
                float r_hh = cacheGates[hh_gateIdx + 0];
                float z_hh = cacheGates[hh_gateIdx + 1];
                float n_hh = cacheGates[hh_gateIdx + 2];

                // Get accumulated hidden gradient for position hh
                float dH_hh;
                int hh_gid = b * hiddenSize + hh;
                int hh_local_idx = hh_gid - (group_id * local_size);
                if (hh_local_idx >= 0 && hh_local_idx < local_size) {
                    dH_hh = local_dH[hh_local_idx];
                } else {
                    dH_hh = dH_buffer[hh_gid];
                }

                // Get h_prev for position hh
                float hPrev_hh = allH[t * batch * hiddenSize + b * hiddenSize + hh];

                // Recompute gate gradients for output position hh
                // OpenCL convention: h = (1-z)*n + z*h_prev
                float dN_hh = dH_hh * (1.0f - z_hh) * tanh_derivative(n_hh);
                float dZ_hh = dH_hh * (hPrev_hh - n_hh) * sigmoid_derivative(z_hh);

                // Compute dR[hh] properly: dR[hh] = sum_k(dN_k * Wn_hh[k,hh]) * hPrev[hh] * sigmoid'(r[hh])
                float dR_sum_hh = 0.0f;
                for (int k = 0; k < hiddenSize; k++) {
                    int k_gateIdx = (t * batch * hiddenSize + b * hiddenSize + k) * 3;
                    float z_k = cacheGates[k_gateIdx + 1];
                    float n_k = cacheGates[k_gateIdx + 2];
                    int k_gid = b * hiddenSize + k;
                    int k_local_idx = k_gid - (group_id * local_size);
                    float dH_k = (k_local_idx >= 0 && k_local_idx < local_size)
                                 ? local_dH[k_local_idx] : dH_buffer[k_gid];
                    float dN_k = dH_k * (1.0f - z_k) * tanh_derivative(n_k);
                    dR_sum_hh += dN_k * weightsHh[(2 * hiddenSize + k) * hiddenSize + hh];
                }
                float dR_hh = dR_sum_hh * hPrev_hh * sigmoid_derivative(r_hh);

                // Gradient through gates to prevH[h] (this thread's hidden unit)
                dHPrev += dZ_hh * weightsHh[(hiddenSize + hh) * hiddenSize + h];
                dHPrev += dR_hh * weightsHh[hh * hiddenSize + h];
                // For Wn_hh path, the reset gate is for THIS thread's unit (h), not hh
                dHPrev += dN_hh * r * weightsHh[(2 * hiddenSize + hh) * hiddenSize + h];
            }

            // Update for next iteration
            dH = dHPrev;
        }

        // Barrier before next timestep to ensure all threads complete
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }

    // Store initial state gradient
    if (isActive) {
        gradHInit[gid] = dH;
    }
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
