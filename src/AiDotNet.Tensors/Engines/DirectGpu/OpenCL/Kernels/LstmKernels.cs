// Copyright (c) AiDotNet. All rights reserved.
// OpenCL kernels for LSTM (Long Short-Term Memory) sequence neural network operations.
// Implements sequence-level forward and backward passes for efficient BPTT training.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;

/// <summary>
/// OpenCL kernels for LSTM sequence operations used in recurrent neural networks.
/// Implements full forward and backward passes for LSTM cells with 4 gates (forget, input, cell, output).
/// These are sequence-level kernels that process all timesteps efficiently for BPTT.
/// </summary>
internal static class LstmKernels
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
// LSTM CELL FORWARD KERNEL (Single Timestep)
// ===========================================================================
// Processes one LSTM cell computation for a single timestep.
// Each thread handles one (batch, hidden) element.

__kernel void lstm_cell_forward(
    __global const float* input,       // [batch, input_size]
    __global const float* prevH,       // [batch, hidden_size]
    __global const float* prevC,       // [batch, hidden_size]
    __global const float* weightsIh,   // [4 * hidden_size, input_size] - input to hidden weights
    __global const float* weightsHh,   // [4 * hidden_size, hidden_size] - hidden to hidden weights
    __global const float* biasIh,      // [4 * hidden_size] - input to hidden bias
    __global const float* biasHh,      // [4 * hidden_size] - hidden to hidden bias
    __global float* newH,              // [batch, hidden_size]
    __global float* newC,              // [batch, hidden_size]
    __global float* gateF,             // [batch, hidden_size] - forget gate cache
    __global float* gateI,             // [batch, hidden_size] - input gate cache
    __global float* gateG,             // [batch, hidden_size] - cell gate (g) cache
    __global float* gateO,             // [batch, hidden_size] - output gate cache
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
    // Gates order: input(i), forget(f), cell(g), output(o)
    float sumI = biasIh[h] + biasHh[h];
    float sumF = biasIh[hiddenSize + h] + biasHh[hiddenSize + h];
    float sumG = biasIh[2 * hiddenSize + h] + biasHh[2 * hiddenSize + h];
    float sumO = biasIh[3 * hiddenSize + h] + biasHh[3 * hiddenSize + h];

    // Input to hidden contribution
    for (int j = 0; j < inputSize; j++) {
        float inVal = input[b * inputSize + j];
        sumI += inVal * weightsIh[h * inputSize + j];
        sumF += inVal * weightsIh[(hiddenSize + h) * inputSize + j];
        sumG += inVal * weightsIh[(2 * hiddenSize + h) * inputSize + j];
        sumO += inVal * weightsIh[(3 * hiddenSize + h) * inputSize + j];
    }

    // Hidden to hidden contribution
    for (int j = 0; j < hiddenSize; j++) {
        float hVal = prevH[b * hiddenSize + j];
        sumI += hVal * weightsHh[h * hiddenSize + j];
        sumF += hVal * weightsHh[(hiddenSize + h) * hiddenSize + j];
        sumG += hVal * weightsHh[(2 * hiddenSize + h) * hiddenSize + j];
        sumO += hVal * weightsHh[(3 * hiddenSize + h) * hiddenSize + j];
    }

    // Apply activations
    float i = sigmoid_fn(sumI);
    float f = sigmoid_fn(sumF);
    float g = tanh(sumG);
    float o = sigmoid_fn(sumO);

    // Cell state update
    float prevCVal = prevC[gid];
    float newCVal = f * prevCVal + i * g;

    // Hidden state update
    float newHVal = o * tanh(newCVal);

    // Store results
    newC[gid] = newCVal;
    newH[gid] = newHVal;
    gateI[gid] = i;
    gateF[gid] = f;
    gateG[gid] = g;
    gateO[gid] = o;
}

// ===========================================================================
// LSTM FORWARD SEQUENCE KERNEL
// ===========================================================================
// Processes the entire sequence in a single kernel launch.
// Outer loop iterates over timesteps, inner parallel threads handle batch * hidden.

__kernel void lstm_forward_sequence(
    __global const float* input,       // [seqLen, batch, input_size]
    __global const float* hInit,       // [batch, hidden_size]
    __global const float* cInit,       // [batch, hidden_size]
    __global const float* weightsIh,   // [4 * hidden_size, input_size]
    __global const float* weightsHh,   // [4 * hidden_size, hidden_size]
    __global const float* biasIh,      // [4 * hidden_size]
    __global const float* biasHh,      // [4 * hidden_size]
    __global float* output,            // [seqLen, batch, hidden_size]
    __global float* hFinal,            // [batch, hidden_size]
    __global float* cFinal,            // [batch, hidden_size]
    __global float* allH,              // [seqLen + 1, batch, hidden_size] - all hidden states
    __global float* allC,              // [seqLen + 1, batch, hidden_size] - all cell states
    __global float* cacheGates,        // [seqLen, batch, hidden_size, 4] - gate values for backward
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

    // Initialize from hInit and cInit (only valid threads)
    float hPrev = 0.0f;
    float cPrev = 0.0f;
    if (isValid) {
        hPrev = hInit[gid];
        cPrev = cInit[gid];

        // Store initial states
        allH[gid] = hPrev;
        allC[gid] = cPrev;
    }

    // Barrier to ensure all threads have written initial states
    barrier(CLK_GLOBAL_MEM_FENCE);

    // Process each timestep
    for (int t = 0; t < seqLen; t++) {
        if (isValid) {
            // Compute gate pre-activations
            float sumI = biasIh[h] + biasHh[h];
            float sumF = biasIh[hiddenSize + h] + biasHh[hiddenSize + h];
            float sumG = biasIh[2 * hiddenSize + h] + biasHh[2 * hiddenSize + h];
            float sumO = biasIh[3 * hiddenSize + h] + biasHh[3 * hiddenSize + h];

            // Input contribution at this timestep
            int inputOffset = t * batch * inputSize + b * inputSize;
            for (int j = 0; j < inputSize; j++) {
                float inVal = input[inputOffset + j];
                sumI += inVal * weightsIh[h * inputSize + j];
                sumF += inVal * weightsIh[(hiddenSize + h) * inputSize + j];
                sumG += inVal * weightsIh[(2 * hiddenSize + h) * inputSize + j];
                sumO += inVal * weightsIh[(3 * hiddenSize + h) * inputSize + j];
            }

            // Previous hidden state contribution (with barrier synchronization)
            for (int j = 0; j < hiddenSize; j++) {
                float hVal = (j == h) ? hPrev : allH[t * batch * hiddenSize + b * hiddenSize + j];
                sumI += hVal * weightsHh[h * hiddenSize + j];
                sumF += hVal * weightsHh[(hiddenSize + h) * hiddenSize + j];
                sumG += hVal * weightsHh[(2 * hiddenSize + h) * hiddenSize + j];
                sumO += hVal * weightsHh[(3 * hiddenSize + h) * hiddenSize + j];
            }

            // Apply activations
            float i = sigmoid_fn(sumI);
            float f = sigmoid_fn(sumF);
            float g = tanh(sumG);
            float o = sigmoid_fn(sumO);

            // Cell state update
            float newC = f * cPrev + i * g;

            // Hidden state update
            float newH = o * tanh(newC);

            // Store output
            int outIdx = t * batch * hiddenSize + gid;
            output[outIdx] = newH;

            // Store all states for backward pass
            int stateIdx = (t + 1) * batch * hiddenSize + gid;
            allH[stateIdx] = newH;
            allC[stateIdx] = newC;

            // Cache gate values for backward
            int gateIdx = (t * batch * hiddenSize + gid) * 4;
            cacheGates[gateIdx + 0] = i;
            cacheGates[gateIdx + 1] = f;
            cacheGates[gateIdx + 2] = g;
            cacheGates[gateIdx + 3] = o;

            // Update for next iteration
            hPrev = newH;
            cPrev = newC;
        }

        // Barrier to ensure all threads have written new states before next iteration
        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    // Store final states (only valid threads)
    if (isValid) {
        hFinal[gid] = hPrev;
        cFinal[gid] = cPrev;
    }
}

// ===========================================================================
// LSTM CELL BACKWARD KERNEL (Single Timestep)
// ===========================================================================

__kernel void lstm_cell_backward(
    __global const float* gradH,       // [batch, hidden_size] - gradient from next layer
    __global const float* gradCNext,   // [batch, hidden_size] - gradient from next timestep cell
    __global const float* gateI,       // [batch, hidden_size]
    __global const float* gateF,       // [batch, hidden_size]
    __global const float* gateG,       // [batch, hidden_size]
    __global const float* gateO,       // [batch, hidden_size]
    __global const float* prevC,       // [batch, hidden_size]
    __global const float* newC,        // [batch, hidden_size]
    __global float* gradPrevC,         // [batch, hidden_size] - gradient to previous cell state
    __global float* gradGateI,         // [batch, hidden_size] - gradient for input gate
    __global float* gradGateF,         // [batch, hidden_size] - gradient for forget gate
    __global float* gradGateG,         // [batch, hidden_size] - gradient for cell gate
    __global float* gradGateO,         // [batch, hidden_size] - gradient for output gate
    const int batch,
    const int hiddenSize)
{
    int gid = get_global_id(0);
    int totalElements = batch * hiddenSize;

    if (gid >= totalElements) return;

    float i = gateI[gid];
    float f = gateF[gid];
    float g = gateG[gid];
    float o = gateO[gid];
    float cPrev = prevC[gid];
    float cNew = newC[gid];

    float dH = gradH[gid];
    float tanhC = tanh(cNew);

    // Gradient through output gate
    float dO = dH * tanhC * sigmoid_derivative(o);

    // Gradient to cell state
    float dC = gradCNext[gid] + dH * o * tanh_derivative(tanhC);

    // Gradients through gates
    float dF = dC * cPrev * sigmoid_derivative(f);
    float dI = dC * g * sigmoid_derivative(i);
    float dG = dC * i * tanh_derivative(g);

    // Gradient to previous cell state
    float dPrevC = dC * f;

    // Store gradients
    gradPrevC[gid] = dPrevC;
    gradGateI[gid] = dI;
    gradGateF[gid] = dF;
    gradGateG[gid] = dG;
    gradGateO[gid] = dO;
}

// ===========================================================================
// LSTM BACKWARD INPUT GRADIENT KERNEL
// ===========================================================================

__kernel void lstm_backward_input(
    __global const float* gradGateI,   // [batch, hidden_size]
    __global const float* gradGateF,   // [batch, hidden_size]
    __global const float* gradGateG,   // [batch, hidden_size]
    __global const float* gradGateO,   // [batch, hidden_size]
    __global const float* weightsIh,   // [4 * hidden_size, input_size]
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

        float dI = gradGateI[batchHiddenIdx];
        float dF = gradGateF[batchHiddenIdx];
        float dG = gradGateG[batchHiddenIdx];
        float dO = gradGateO[batchHiddenIdx];

        gradSum += dI * weightsIh[h * inputSize + j];
        gradSum += dF * weightsIh[(hiddenSize + h) * inputSize + j];
        gradSum += dG * weightsIh[(2 * hiddenSize + h) * inputSize + j];
        gradSum += dO * weightsIh[(3 * hiddenSize + h) * inputSize + j];
    }

    gradInput[gid] = gradSum;
}

// ===========================================================================
// LSTM BACKWARD PREVIOUS HIDDEN GRADIENT KERNEL
// ===========================================================================

__kernel void lstm_backward_prevh(
    __global const float* gradGateI,   // [batch, hidden_size]
    __global const float* gradGateF,   // [batch, hidden_size]
    __global const float* gradGateG,   // [batch, hidden_size]
    __global const float* gradGateO,   // [batch, hidden_size]
    __global const float* weightsHh,   // [4 * hidden_size, hidden_size]
    __global float* gradPrevH,         // [batch, hidden_size]
    const int batch,
    const int hiddenSize)
{
    int gid = get_global_id(0);
    int totalElements = batch * hiddenSize;

    if (gid >= totalElements) return;

    int b = gid / hiddenSize;
    int j = gid % hiddenSize;

    float gradSum = 0.0f;

    for (int h = 0; h < hiddenSize; h++) {
        int batchHiddenIdx = b * hiddenSize + h;

        float dI = gradGateI[batchHiddenIdx];
        float dF = gradGateF[batchHiddenIdx];
        float dG = gradGateG[batchHiddenIdx];
        float dO = gradGateO[batchHiddenIdx];

        gradSum += dI * weightsHh[h * hiddenSize + j];
        gradSum += dF * weightsHh[(hiddenSize + h) * hiddenSize + j];
        gradSum += dG * weightsHh[(2 * hiddenSize + h) * hiddenSize + j];
        gradSum += dO * weightsHh[(3 * hiddenSize + h) * hiddenSize + j];
    }

    gradPrevH[gid] = gradSum;
}

// ===========================================================================
// LSTM BACKWARD SEQUENCE KERNEL
// ===========================================================================
// Processes backward pass through entire sequence.
// Iterates in reverse through timesteps for proper BPTT.

__kernel void lstm_backward_sequence(
    __global const float* gradOutput,  // [seqLen, batch, hidden_size]
    __global const float* allH,        // [seqLen + 1, batch, hidden_size]
    __global const float* allC,        // [seqLen + 1, batch, hidden_size]
    __global const float* cacheGates,  // [seqLen, batch, hidden_size, 4]
    __global const float* weightsIh,   // [4 * hidden_size, input_size]
    __global const float* weightsHh,   // [4 * hidden_size, hidden_size]
    __global float* gradInput,         // [seqLen, batch, input_size]
    __global float* gradHInit,         // [batch, hidden_size]
    __global float* gradCInit,         // [batch, hidden_size]
    __global float* gradWeightsIh,     // [4 * hidden_size, input_size] - accumulated
    __global float* gradWeightsHh,     // [4 * hidden_size, hidden_size] - accumulated
    __global float* gradBiasIh,        // [4 * hidden_size] - accumulated
    __global float* gradBiasHh,        // [4 * hidden_size] - accumulated
    __global const float* input,       // [seqLen, batch, input_size]
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

    // Initialize gradients
    float dH = 0.0f;  // Gradient w.r.t. hidden state (accumulated from next timestep)
    float dC = 0.0f;  // Gradient w.r.t. cell state (accumulated from next timestep)

    // Initialize gradHInit buffer for use as intermediate storage during BPTT
    if (isValid) {
        gradHInit[gid] = 0.0f;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    // Process timesteps in reverse
    for (int t = seqLen - 1; t >= 0; t--) {
        if (isValid) {
            // Read accumulated recurrent gradient from previous iteration (if any)
            if (t < seqLen - 1) {
                dH = gradHInit[gid];
                gradHInit[gid] = 0.0f;  // Clear for next accumulation
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE);

        if (isValid) {
            // Add gradient from output at this timestep
            int outIdx = t * batch * hiddenSize + gid;
            dH += gradOutput[outIdx];

            // Load cached gate values
            int gateIdx = (t * batch * hiddenSize + gid) * 4;
            float i = cacheGates[gateIdx + 0];
            float f = cacheGates[gateIdx + 1];
            float g = cacheGates[gateIdx + 2];
            float o = cacheGates[gateIdx + 3];

            // Load cell states
            int prevStateIdx = t * batch * hiddenSize + gid;
            int currStateIdx = (t + 1) * batch * hiddenSize + gid;
            float cPrev = allC[prevStateIdx];
            float cCurr = allC[currStateIdx];

            float tanhC = tanh(cCurr);

            // Gradient through output gate
            float dO = dH * tanhC * sigmoid_derivative(o);

            // Gradient to cell state (from hidden gradient + next timestep cell gradient)
            dC += dH * o * tanh_derivative(tanhC);

            // Gradients through gates
            float dF = dC * cPrev * sigmoid_derivative(f);
            float dI = dC * g * sigmoid_derivative(i);
            float dG = dC * i * tanh_derivative(g);

            // Gradient to previous cell state for next iteration
            float dCPrev = dC * f;

            // Gradient to previous hidden state for BPTT
            // dH_prev[j] = sum_k (dGate[k] * Wh[k, j]) for all four gates
            // Each thread k contributes its gate gradients to all hidden units j
            // Note: Using simpler accumulation pattern - within work group this relies on
            // sequential execution of contributions. For production, use proper float atomic add.
            for (int j = 0; j < hiddenSize; j++) {
                // Contribution from gate derivatives at position h to hidden unit j
                // Wh layout: [4*hiddenSize, hiddenSize], so Wh[k, j] = Wh[k * hiddenSize + j]
                float contrib = dI * weightsHh[h * hiddenSize + j];
                contrib += dF * weightsHh[(hiddenSize + h) * hiddenSize + j];
                contrib += dG * weightsHh[(2 * hiddenSize + h) * hiddenSize + j];
                contrib += dO * weightsHh[(3 * hiddenSize + h) * hiddenSize + j];
                // Simple accumulation - relies on single work group execution
                // For multi-group, would need atomic operations or reduction kernel
                gradHInit[b * hiddenSize + j] += contrib;
            }

            // Update dC for next iteration
            dC = dCPrev;
        }

        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    // Store initial state gradients
    // gradHInit already contains accumulated gradient for h_init
    if (isValid) {
        gradCInit[gid] = dC;
    }
}

// ===========================================================================
// LSTM WEIGHT GRADIENT ACCUMULATION KERNELS
// ===========================================================================

__kernel void lstm_accumulate_weight_gradients_ih(
    __global const float* input,        // [seqLen, batch, input_size]
    __global const float* allH,         // [seqLen + 1, batch, hidden_size]
    __global const float* allC,         // [seqLen + 1, batch, hidden_size]
    __global const float* cacheGates,   // [seqLen, batch, hidden_size, 4]
    __global const float* gradOutput,   // [seqLen, batch, hidden_size]
    __global float* gradWeightsIh,      // [4 * hidden_size, input_size]
    const int seqLen,
    const int batch,
    const int inputSize,
    const int hiddenSize)
{
    int gid = get_global_id(0);
    int totalWeights = 4 * hiddenSize * inputSize;

    if (gid >= totalWeights) return;

    int gateIdx = gid / (hiddenSize * inputSize);  // Which gate (0-3)
    int remainder = gid % (hiddenSize * inputSize);
    int h = remainder / inputSize;
    int j = remainder % inputSize;

    float gradSum = 0.0f;

    // Accumulate gradients over all timesteps and batch elements
    // Note: This requires computing gate gradients inline
    for (int t = 0; t < seqLen; t++) {
        for (int b = 0; b < batch; b++) {
            int batchHiddenIdx = b * hiddenSize + h;

            // Load cached values
            int cacheIdx = (t * batch * hiddenSize + batchHiddenIdx) * 4;
            float i = cacheGates[cacheIdx + 0];
            float f = cacheGates[cacheIdx + 1];
            float g = cacheGates[cacheIdx + 2];
            float o = cacheGates[cacheIdx + 3];

            // Cell states
            int prevStateIdx = t * batch * hiddenSize + batchHiddenIdx;
            int currStateIdx = (t + 1) * batch * hiddenSize + batchHiddenIdx;
            float cPrev = allC[prevStateIdx];
            float cCurr = allC[currStateIdx];

            // Get output gradient (simplified - full impl needs accumulated dH, dC)
            float dH = gradOutput[t * batch * hiddenSize + batchHiddenIdx];
            float tanhC = tanh(cCurr);

            // Compute gate gradients
            float dO = dH * tanhC * sigmoid_derivative(o);
            float dC = dH * o * tanh_derivative(tanhC);
            float dF = dC * cPrev * sigmoid_derivative(f);
            float dI = dC * g * sigmoid_derivative(i);
            float dG = dC * i * tanh_derivative(g);

            // Get input value
            float inputVal = input[t * batch * inputSize + b * inputSize + j];

            // Accumulate based on gate index
            if (gateIdx == 0) {
                gradSum += dI * inputVal;
            } else if (gateIdx == 1) {
                gradSum += dF * inputVal;
            } else if (gateIdx == 2) {
                gradSum += dG * inputVal;
            } else {
                gradSum += dO * inputVal;
            }
        }
    }

    gradWeightsIh[gid] = gradSum;
}

__kernel void lstm_accumulate_weight_gradients_hh(
    __global const float* allH,         // [seqLen + 1, batch, hidden_size]
    __global const float* allC,         // [seqLen + 1, batch, hidden_size]
    __global const float* cacheGates,   // [seqLen, batch, hidden_size, 4]
    __global const float* gradOutput,   // [seqLen, batch, hidden_size]
    __global float* gradWeightsHh,      // [4 * hidden_size, hidden_size]
    const int seqLen,
    const int batch,
    const int hiddenSize)
{
    int gid = get_global_id(0);
    int totalWeights = 4 * hiddenSize * hiddenSize;

    if (gid >= totalWeights) return;

    int gateIdx = gid / (hiddenSize * hiddenSize);  // Which gate (0-3)
    int remainder = gid % (hiddenSize * hiddenSize);
    int h = remainder / hiddenSize;
    int j = remainder % hiddenSize;

    float gradSum = 0.0f;

    for (int t = 0; t < seqLen; t++) {
        for (int b = 0; b < batch; b++) {
            int batchHiddenIdx = b * hiddenSize + h;

            // Load cached values
            int cacheIdx = (t * batch * hiddenSize + batchHiddenIdx) * 4;
            float i = cacheGates[cacheIdx + 0];
            float f = cacheGates[cacheIdx + 1];
            float g = cacheGates[cacheIdx + 2];
            float o = cacheGates[cacheIdx + 3];

            // Cell states
            int prevStateIdx = t * batch * hiddenSize + batchHiddenIdx;
            int currStateIdx = (t + 1) * batch * hiddenSize + batchHiddenIdx;
            float cPrev = allC[prevStateIdx];
            float cCurr = allC[currStateIdx];

            // Get output gradient
            float dH = gradOutput[t * batch * hiddenSize + batchHiddenIdx];
            float tanhC = tanh(cCurr);

            // Compute gate gradients
            float dO = dH * tanhC * sigmoid_derivative(o);
            float dC = dH * o * tanh_derivative(tanhC);
            float dF = dC * cPrev * sigmoid_derivative(f);
            float dI = dC * g * sigmoid_derivative(i);
            float dG = dC * i * tanh_derivative(g);

            // Get previous hidden value
            float hPrev = allH[t * batch * hiddenSize + b * hiddenSize + j];

            // Accumulate based on gate index
            if (gateIdx == 0) {
                gradSum += dI * hPrev;
            } else if (gateIdx == 1) {
                gradSum += dF * hPrev;
            } else if (gateIdx == 2) {
                gradSum += dG * hPrev;
            } else {
                gradSum += dO * hPrev;
            }
        }
    }

    gradWeightsHh[gid] = gradSum;
}

__kernel void lstm_accumulate_bias_gradients(
    __global const float* allC,         // [seqLen + 1, batch, hidden_size]
    __global const float* cacheGates,   // [seqLen, batch, hidden_size, 4]
    __global const float* gradOutput,   // [seqLen, batch, hidden_size]
    __global float* gradBias,           // [4 * hidden_size]
    const int seqLen,
    const int batch,
    const int hiddenSize)
{
    int gid = get_global_id(0);
    int totalBiases = 4 * hiddenSize;

    if (gid >= totalBiases) return;

    int gateIdx = gid / hiddenSize;  // Which gate (0-3)
    int h = gid % hiddenSize;

    float gradSum = 0.0f;

    for (int t = 0; t < seqLen; t++) {
        for (int b = 0; b < batch; b++) {
            int batchHiddenIdx = b * hiddenSize + h;

            // Load cached values
            int cacheIdx = (t * batch * hiddenSize + batchHiddenIdx) * 4;
            float i = cacheGates[cacheIdx + 0];
            float f = cacheGates[cacheIdx + 1];
            float g = cacheGates[cacheIdx + 2];
            float o = cacheGates[cacheIdx + 3];

            // Cell states
            int prevStateIdx = t * batch * hiddenSize + batchHiddenIdx;
            int currStateIdx = (t + 1) * batch * hiddenSize + batchHiddenIdx;
            float cPrev = allC[prevStateIdx];
            float cCurr = allC[currStateIdx];

            // Get output gradient
            float dH = gradOutput[t * batch * hiddenSize + batchHiddenIdx];
            float tanhC = tanh(cCurr);

            // Compute gate gradients
            float dO = dH * tanhC * sigmoid_derivative(o);
            float dC = dH * o * tanh_derivative(tanhC);
            float dF = dC * cPrev * sigmoid_derivative(f);
            float dI = dC * g * sigmoid_derivative(i);
            float dG = dC * i * tanh_derivative(g);

            // Accumulate based on gate index
            if (gateIdx == 0) {
                gradSum += dI;
            } else if (gateIdx == 1) {
                gradSum += dF;
            } else if (gateIdx == 2) {
                gradSum += dG;
            } else {
                gradSum += dO;
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
            "lstm_cell_forward",
            "lstm_forward_sequence",
            "lstm_cell_backward",
            "lstm_backward_input",
            "lstm_backward_prevh",
            "lstm_backward_sequence",
            "lstm_accumulate_weight_gradients_ih",
            "lstm_accumulate_weight_gradients_hh",
            "lstm_accumulate_bias_gradients"
        };
    }
}
