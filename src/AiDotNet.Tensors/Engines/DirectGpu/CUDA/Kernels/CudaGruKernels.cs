// Copyright (c) AiDotNet. All rights reserved.
// CUDA kernels for GRU (Gated Recurrent Unit) neural network operations.
// Implements sequence-level forward and backward passes for efficient BPTT.

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

/// <summary>
/// CUDA kernels for sequence-level GRU operations.
/// Implements full forward and backward passes for GRU layers processing entire sequences.
/// </summary>
/// <remarks>
/// GRU equations:
/// z_t = sigmoid(W_z * x_t + U_z * h_{t-1} + b_z)       // update gate
/// r_t = sigmoid(W_r * x_t + U_r * h_{t-1} + b_r)       // reset gate
/// h̃_t = tanh(W_h * x_t + U_h * (r_t ⊙ h_{t-1}) + b_h)  // candidate hidden
/// h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t               // hidden state
/// </remarks>
internal static class CudaGruKernels
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
// GRU CELL FORWARD KERNEL (Single Timestep)
// ===========================================================================

// GRU cell forward pass for a single time step
// input: [batch, inputSize]
// prevH: [batch, hiddenSize]
// Wz, Wr, Wh: [hiddenSize, inputSize]
// Uz, Ur, Uh: [hiddenSize, hiddenSize]
// bz, br, bh: [hiddenSize]
// output newH: [batch, hiddenSize]
// gateZ, gateR, gateH: [batch, hiddenSize] (cached for backward)
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
    int b = gid / hiddenSize;
    int h = gid % hiddenSize;
    int isValid = (gid < totalElements) ? 1 : 0;

    // Phase 1: Compute z and r gates for all threads
    float z = 0.0f;
    float r = 0.0f;

    if (isValid) {
        // Compute update gate z: sigmoid(Wz*x + Uz*h + bz)
        float sumZ = bz[h];
        float sumR = br[h];

        // Input contribution
        for (int i = 0; i < inputSize; i++) {
            float x_val = input[b * inputSize + i];
            sumZ += Wz[h * inputSize + i] * x_val;
            sumR += Wr[h * inputSize + i] * x_val;
        }

        // Hidden contribution for z and r gates
        for (int j = 0; j < hiddenSize; j++) {
            float h_val = prevH[b * hiddenSize + j];
            sumZ += Uz[h * hiddenSize + j] * h_val;
            sumR += Ur[h * hiddenSize + j] * h_val;
        }

        z = sigmoid(sumZ);
        r = sigmoid(sumR);

        // Store r to global buffer so other threads can read it
        gateR[gid] = r;
    }

    // Synchronize to ensure all r values are written before reading
    __syncthreads();

    // Phase 2: Compute candidate using per-element r_j values
    float h_candidate = 0.0f;

    if (isValid) {
        float sumH = bh[h];

        for (int i = 0; i < inputSize; i++) {
            float x_val = input[b * inputSize + i];
            sumH += Wh[h * inputSize + i] * x_val;
        }

        // Use per-element reset gate r_j for proper GRU computation
        // In standard GRU: candidate = tanh(Wh*x + Uh*(r ⊙ h_prev) + bh)
        for (int j = 0; j < hiddenSize; j++) {
            float h_val = prevH[b * hiddenSize + j];
            float r_j = gateR[b * hiddenSize + j];  // Read r for hidden unit j
            sumH += Uh[h * hiddenSize + j] * r_j * h_val;
        }

        h_candidate = tanhf(sumH);

        // Compute new hidden: (1-z)*h_prev + z*h_candidate
        float prevHVal = prevH[gid];
        float newHVal = (1.0f - z) * prevHVal + z * h_candidate;

        // Store outputs
        newH[gid] = newHVal;

        // Store gate values for backward pass
        gateZ[gid] = z;
        // gateR already stored above
        gateHCandidate[gid] = h_candidate;
    }
}

// ===========================================================================
// GRU SEQUENCE FORWARD KERNEL
// ===========================================================================

// GRU forward pass for entire sequence
// input: [batch, timeSteps, inputSize]
// h_init: [batch, hiddenSize]
// output: [batch, timeSteps, hiddenSize]
// h_states: [timeSteps, batch, hiddenSize] (cached for backward)
// gates: [timeSteps, batch, 3*hiddenSize] (Z,R,H cached for backward)
extern ""C"" __global__ void gru_forward_sequence(
    const float* input,
    const float* h_init,
    const float* Wz, const float* Wr, const float* Wh,
    const float* Uz, const float* Ur, const float* Uh,
    const float* bz, const float* br, const float* bh,
    float* output,
    float* h_states,      // Cache: [timeSteps, batch, hidden]
    float* gates,         // Cache: [timeSteps, batch, 3*hidden]
    int batch,
    int timeSteps,
    int inputSize,
    int hiddenSize)
{
    // Each thread handles one (batch, hidden) element
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batch * hiddenSize;

    // Use active flag instead of early return to avoid syncthreads deadlock
    bool active = (gid < totalElements);

    int b = 0, h_idx = 0;
    float h_val = 0.0f;

    if (active) {
        b = gid / hiddenSize;
        h_idx = gid % hiddenSize;
        // Initialize hidden state
        h_val = h_init[gid];
    }

    // Process each timestep
    for (int t = 0; t < timeSteps; t++) {
        float z = 0.0f, r = 0.0f;
        int gateOffset = 0;

        if (active) {
            // Compute gate pre-activations
            float sumZ = bz[h_idx];
            float sumR = br[h_idx];

            // Input contribution
            int inputOffset = (b * timeSteps + t) * inputSize;
            for (int i = 0; i < inputSize; i++) {
                float x_val = input[inputOffset + i];
                sumZ += Wz[h_idx * inputSize + i] * x_val;
                sumR += Wr[h_idx * inputSize + i] * x_val;
            }

            // Hidden contribution for z and r
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

            z = sigmoid(sumZ);
            r = sigmoid(sumR);

            // Store r to gates buffer immediately so other threads can read it
            gateOffset = t * batch * 3 * hiddenSize + b * 3 * hiddenSize;
            gates[gateOffset + hiddenSize + h_idx] = r;
        }

        // Sync to ensure all threads have stored their r values
        // All threads (active and inactive) must reach this point
        __syncthreads();

        if (active) {
            // Compute candidate hidden using per-element reset gate
            float sumH = bh[h_idx];
            int inputOffset = (b * timeSteps + t) * inputSize;

            for (int i = 0; i < inputSize; i++) {
                float x_val = input[inputOffset + i];
                sumH += Wh[h_idx * inputSize + i] * x_val;
            }

            // Use per-element reset gate r_j for proper GRU computation
            // In standard GRU: candidate = tanh(Wh*x + Uh*(r ⊙ h_prev) + bh)
            gateOffset = t * batch * 3 * hiddenSize + b * 3 * hiddenSize;
            for (int j = 0; j < hiddenSize; j++) {
                float hj;
                if (t == 0) {
                    hj = h_init[b * hiddenSize + j];
                } else {
                    hj = h_states[(t - 1) * batch * hiddenSize + b * hiddenSize + j];
                }
                // Read r_j for hidden unit j from gates buffer
                float rj = gates[gateOffset + hiddenSize + j];
                sumH += Uh[h_idx * hiddenSize + j] * rj * hj;
            }

            float h_candidate = tanhf(sumH);

            // Get previous hidden state
            float h_prev;
            if (t == 0) {
                h_prev = h_init[gid];
            } else {
                h_prev = h_states[(t - 1) * batch * hiddenSize + gid];
            }

            // Update hidden state
            h_val = (1.0f - z) * h_prev + z * h_candidate;

            // Store states
            int stateOffset = t * batch * hiddenSize + gid;
            h_states[stateOffset] = h_val;

            // Store output
            output[(b * timeSteps + t) * hiddenSize + h_idx] = h_val;

            // Store remaining gates for backward pass (r was stored earlier)
            gates[gateOffset + h_idx] = z;
            gates[gateOffset + 2 * hiddenSize + h_idx] = h_candidate;
        }

        // All threads must reach this syncthreads
        __syncthreads();
    }
}

// ===========================================================================
// GRU CELL BACKWARD KERNEL
// ===========================================================================

// Computes gradients for a single GRU cell timestep
// Reset gate gradient is computed per-element: for each input hidden unit j,
// we compute sum over outputs h of dHCand_h * Uh[h, j], then multiply by
// prevH[j] * sigmoid_derivative(r[j])
//
// The reset gate is applied element-wise in forward: candidate uses r[j] * prevH[j]
// So the backward must compute dR[j] as a reduction over all output positions h.
extern ""C"" __global__ void gru_cell_backward(
    const float* dH,          // [batch, hidden]
    const float* gateZ,       // [batch, hidden]
    const float* gateR,       // [batch, hidden]
    const float* gateHCand,   // [batch, hidden]
    const float* prevH,       // [batch, hidden]
    const float* input,       // [batch, input]
    const float* Wz, const float* Wr, const float* Wh,
    const float* Uz, const float* Ur, const float* Uh,
    float* dPrevH,            // [batch, hidden]
    float* dInput,            // [batch, input]
    float* dWz, float* dWr, float* dWh,
    float* dUz, float* dUr, float* dUh,
    float* dbz, float* dbr, float* dbh,
    int batch,
    int inputSize,
    int hiddenSize)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batch * hiddenSize;

    if (gid >= totalElements) return;

    int b = gid / hiddenSize;
    int h = gid % hiddenSize;

    // Get cached values for this output position h
    float z = gateZ[gid];
    float h_cand = gateHCand[gid];
    float h_prev_h = prevH[gid];

    // Gradient from output
    float dh = dH[gid];

    // Gradient through hidden state update: h_t = (1-z)*h_prev + z*h_cand
    // dh_cand = dh * z * tanh'(h_cand)
    float dHCand = dh * z * tanh_derivative(h_cand);

    // dz = dh * (h_cand - h_prev) * sigmoid'(z)
    float dZ = dh * (h_cand - h_prev_h) * sigmoid_derivative(z);

    // dh_prev from direct path = dh * (1-z)
    float dHPrev_direct = dh * (1.0f - z);

    // Add direct path contribution to dPrevH[h]
    atomicAdd(&dPrevH[gid], dHPrev_direct);

    // Gradient to previous hidden through reset gate path and reset gate gradient
    // Forward: candidate_h = tanh(... + sum_j(Uh[h,j] * r[j] * prevH[j]) + ...)
    // So for each input position j:
    //   d(candidate_h)/d(prevH[j]) = Uh[h,j] * r[j]
    //   d(candidate_h)/d(r[j]) = Uh[h,j] * prevH[j]
    //
    // dPrevH[j] from reset path = sum_h(dHCand_h * Uh[h,j] * r[j])
    // dR[j] (pre-activation) = sum_h(dHCand_h * Uh[h,j]) * prevH[j] * sigmoid'(r[j])
    //
    // This thread handles output h, so we contribute to each j via atomicAdd
    for (int j = 0; j < hiddenSize; j++) {
        float r_j = gateR[b * hiddenSize + j];
        float prevH_j = prevH[b * hiddenSize + j];

        // Contribution from output h to dPrevH[j] through reset path
        float dPrevH_contrib = dHCand * Uh[h * hiddenSize + j] * r_j;
        atomicAdd(&dPrevH[b * hiddenSize + j], dPrevH_contrib);

        // Contribution from output h to dR[j]
        // dR[j] = sum_h(dHCand_h * Uh[h,j]) * prevH[j] * sigmoid'(r[j])
        // This thread contributes: dHCand * Uh[h,j]
        // Then multiply by prevH[j] * sigmoid'(r[j]) to get contribution to pre-activation gradient
        float dR_contrib_h = dHCand * Uh[h * hiddenSize + j];
        float dR_j_contrib = dR_contrib_h * prevH_j * sigmoid_derivative(r_j);

        // Accumulate reset gate weight gradients: dUr[h,j] = dR[h] * prevH[j]
        // But reset weights connect output h to input j, and dR is per input position j
        // The gradient for Ur[h,j] comes from output h's contribution to the reset path
        // dL/dUr[h,j] = dL/d(preR[h]) * d(preR[h])/dUr[h,j] = dR[h] * prevH[j]
        // We need dR[h] which is the gradient at output position h, not input position j

        // Hidden weight gradient for reset gate: dUr[h,j] uses dR[h], not dR[j]
        // We compute dR[h] later for weight updates

        // Accumulate dUh: dUh[h,j] = dHCand * r[j] * prevH[j]
        atomicAdd(&dUh[h * hiddenSize + j], dHCand * r_j * prevH_j);

        // Accumulate bias gradient for reset gate position j
        // dbr[j] = sum_b sum_h dR_j_contrib (already has sigmoid' and prevH)
        atomicAdd(&dbr[j], dR_j_contrib);

        // Accumulate hidden weight gradient for reset: dUr[h,j] from this output h
        // The reset gate computation is: preR[h] = sum_j(Ur[h,j] * prevH[j]) + ...
        // So dUr[h,j] needs dR[h] * prevH[j]
        // We'll compute dR[h] separately below for correct weight gradient
    }

    // Compute dR[h] - the gradient of the reset gate at output position h
    // This is needed for Wr weight gradient: dWr[h,i] = dR[h] * x[i]
    // and Ur weight gradient: dUr[h,j] = dR[h] * prevH[j]
    //
    // dR[h] = sum_j(dL/d(r[j]) contribution from position h)
    // But actually for the reset gate weights, we need the gradient w.r.t. the
    // pre-activation at position h, not j.
    //
    // The forward is: r[h] = sigmoid(sum_i(Wr[h,i]*x[i]) + sum_j(Ur[h,j]*prevH[j]) + br[h])
    // The reset gate r[h] at position h is used to gate prevH[h] in the candidate for all outputs.
    //
    // So dL/dr[h] = sum_over_outputs_k(dHCand_k * Uh[k,h]) * prevH[h]
    // And dL/d(preR[h]) = dL/dr[h] * sigmoid'(r[h])
    float dR_h_sum = 0.0f;
    for (int k = 0; k < hiddenSize; k++) {
        float dHCand_k = dH[b * hiddenSize + k] * gateZ[b * hiddenSize + k] *
                         tanh_derivative(gateHCand[b * hiddenSize + k]);
        dR_h_sum += dHCand_k * Uh[k * hiddenSize + h];
    }
    float r_h = gateR[gid];
    float dR_h = dR_h_sum * prevH[gid] * sigmoid_derivative(r_h);

    // Accumulate input weight gradients
    for (int i = 0; i < inputSize; i++) {
        float x_val = input[b * inputSize + i];
        atomicAdd(&dWz[h * inputSize + i], dZ * x_val);
        atomicAdd(&dWr[h * inputSize + i], dR_h * x_val);
        atomicAdd(&dWh[h * inputSize + i], dHCand * x_val);
    }

    // Hidden weight gradients for Z gate and Ur
    for (int j = 0; j < hiddenSize; j++) {
        float h_val = prevH[b * hiddenSize + j];
        atomicAdd(&dUz[h * hiddenSize + j], dZ * h_val);
        atomicAdd(&dUr[h * hiddenSize + j], dR_h * h_val);
    }

    // Bias gradients for Z gate and candidate
    atomicAdd(&dbz[h], dZ);
    // Note: dbr was already accumulated per-element in the loop above
    atomicAdd(&dbh[h], dHCand);
}

// ===========================================================================
// GRU BACKWARD INPUT KERNEL
// ===========================================================================

// Computes gradient with respect to input
extern ""C"" __global__ void gru_backward_input(
    const float* dGates,      // [batch, 3*hidden] - dZ, dR, dH concatenated
    const float* Wz, const float* Wr, const float* Wh,
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

    // Accumulate gradients from all 3 gates
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

// ===========================================================================
// GRU BACKWARD PREV HIDDEN KERNEL
// ===========================================================================

// Computes gradient with respect to previous hidden state
extern ""C"" __global__ void gru_backward_prevh(
    const float* dH,          // [batch, hidden] - total gradient to h_t
    const float* dGates,      // [batch, 3*hidden]
    const float* gateZ,       // [batch, hidden]
    const float* gateR,       // [batch, hidden]
    const float* Uz, const float* Ur, const float* Uh,
    float* dPrevH,            // [batch, hidden]
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

    // Direct gradient from h_t = (1-z)*h_prev + z*h_cand
    float grad = dH[gid] * (1.0f - z);

    // Gradient through gates
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

// ===========================================================================
// GRU SEQUENCE BACKWARD KERNEL
// ===========================================================================

// GRU backward pass for entire sequence with full BPTT
// Uses shared memory to store accumulated hidden gradients so all threads
// can access each other's dH values for proper reset-gate gradient computation.
extern ""C"" __global__ void gru_backward_sequence(
    const float* gradOutput,  // [batch, timeSteps, hidden]
    const float* h_states,    // [timeSteps, batch, hidden]
    const float* gates,       // [timeSteps, batch, 3*hidden]
    const float* h_init,      // [batch, hidden]
    const float* input,       // [batch, timeSteps, input]
    const float* Wz, const float* Wr, const float* Wh,
    const float* Uz, const float* Ur, const float* Uh,
    float* gradInput,         // [batch, timeSteps, input]
    float* dWz, float* dWr, float* dWh,
    float* dUz, float* dUr, float* dUh,
    float* dbz, float* dbr, float* dbh,
    float* dH_init,           // [batch, hidden]
    float* dH_buffer,         // [batch, hidden] - workspace for accumulated gradients
    int batch,
    int timeSteps,
    int inputSize,
    int hiddenSize)
{
    // Shared memory for accumulated hidden gradients within this block
    // Each thread stores its accumulated dH so other threads can read it
    extern __shared__ float shared_dH[];

    // Each thread handles one (batch, hidden) element
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batch * hiddenSize;

    // Use active-flag pattern to prevent __syncthreads deadlock
    int isActive = (gid < totalElements) ? 1 : 0;

    int b = isActive ? (gid / hiddenSize) : 0;
    int h_idx = isActive ? (gid % hiddenSize) : 0;

    // Initialize gradient for recurrence
    float dH = 0.0f;

    // Process timesteps in reverse (BPTT)
    for (int t = timeSteps - 1; t >= 0; t--) {
        // Phase 1: Add gradient from output and compute basic gradients
        float z = 0.0f, r = 0.0f, h_cand = 0.0f, h_prev = 0.0f;
        float dHCand = 0.0f, dZ = 0.0f;
        int gateOffset = 0;

        if (isActive) {
            // Add gradient from output at this timestep
            dH += gradOutput[(b * timeSteps + t) * hiddenSize + h_idx];

            // Get cached gate values
            gateOffset = t * batch * 3 * hiddenSize + b * 3 * hiddenSize;
            z = gates[gateOffset + h_idx];
            r = gates[gateOffset + hiddenSize + h_idx];
            h_cand = gates[gateOffset + 2 * hiddenSize + h_idx];

            // Get previous hidden state
            if (t == 0) {
                h_prev = h_init[gid];
            } else {
                h_prev = h_states[(t - 1) * batch * hiddenSize + gid];
            }

            // Gradient through hidden state update: h_t = (1-z)*h_prev + z*h_cand
            dHCand = dH * z * tanh_derivative(h_cand);
            dZ = dH * (h_cand - h_prev) * sigmoid_derivative(z);

            // Store accumulated dH to shared memory for other threads to read
            shared_dH[threadIdx.x] = dH;

            // Also write to global buffer for cross-block access
            dH_buffer[gid] = dH;
        }

        // Sync to ensure all threads have written their dH values
        __syncthreads();

        // Phase 2: Compute reset gate gradient using accumulated hidden gradients
        float dR = 0.0f;
        if (isActive) {
            // Full BPTT: dR[h_idx] = sum_k(dHCand_k * Uh[k,h_idx]) * prevH[h_idx] * sigmoid'(r[h_idx])
            // where dHCand_k uses the ACCUMULATED gradient dH_k, not just gradOutput
            float dR_sum = 0.0f;
            for (int k = 0; k < hiddenSize; k++) {
                // Get cached gate values for output k
                float z_k = gates[gateOffset + k];
                float h_cand_k = gates[gateOffset + 2 * hiddenSize + k];

                // Get accumulated hidden gradient for position k
                // Try shared memory first (same block), fall back to global buffer
                float dH_k;
                int k_gid = b * hiddenSize + k;
                int k_local_idx = k_gid - (blockIdx.x * blockDim.x);
                if (k_local_idx >= 0 && k_local_idx < blockDim.x) {
                    // Same block - use shared memory
                    dH_k = shared_dH[k_local_idx];
                } else {
                    // Different block - use global buffer
                    dH_k = dH_buffer[k_gid];
                }

                // Compute dHCand for output k using accumulated gradient
                float dHCand_k = dH_k * z_k * tanh_derivative(h_cand_k);
                dR_sum += dHCand_k * Uh[k * hiddenSize + h_idx];
            }
            dR = dR_sum * h_prev * sigmoid_derivative(r);
        }

        // Phase 3: Accumulate weight gradients
        if (isActive) {
            int inputOffset = (b * timeSteps + t) * inputSize;
            for (int i = 0; i < inputSize; i++) {
                float x_val = input[inputOffset + i];
                atomicAdd(&dWz[h_idx * inputSize + i], dZ * x_val);
                atomicAdd(&dWr[h_idx * inputSize + i], dR * x_val);
                atomicAdd(&dWh[h_idx * inputSize + i], dHCand * x_val);
            }

            // Hidden weight gradients
            for (int j = 0; j < hiddenSize; j++) {
                float hj;
                if (t == 0) {
                    hj = h_init[b * hiddenSize + j];
                } else {
                    hj = h_states[(t - 1) * batch * hiddenSize + b * hiddenSize + j];
                }
                float r_j = gates[gateOffset + hiddenSize + j];
                atomicAdd(&dUz[h_idx * hiddenSize + j], dZ * hj);
                atomicAdd(&dUr[h_idx * hiddenSize + j], dR * hj);
                atomicAdd(&dUh[h_idx * hiddenSize + j], dHCand * r_j * hj);
            }

            // Bias gradients
            atomicAdd(&dbz[h_idx], dZ);
            atomicAdd(&dbr[h_idx], dR);
            atomicAdd(&dbh[h_idx], dHCand);

            // Compute gradient to input at this timestep
            int gradInputOffset = (b * timeSteps + t) * inputSize;
            for (int i = 0; i < inputSize; i++) {
                float grad_i = 0.0f;
                grad_i += dZ * Wz[h_idx * inputSize + i];
                grad_i += dR * Wr[h_idx * inputSize + i];
                grad_i += dHCand * Wh[h_idx * inputSize + i];
                atomicAdd(&gradInput[gradInputOffset + i], grad_i);
            }

            // Gradient to previous hidden state for next iteration (full BPTT)
            // dPrevH = dH * (1-z) + contributions from all output positions through gates
            float dH_prev = dH * (1.0f - z);  // Direct path

            // Gradient through Z gate to prevH
            dH_prev += dZ * Uz[h_idx * hiddenSize + h_idx];

            // Gradient through R gate to prevH
            dH_prev += dR * Ur[h_idx * hiddenSize + h_idx];

            // Gradient through candidate (Uh) path: dHCand * Uh[h_idx, h_idx] * r[h_idx]
            dH_prev += dHCand * Uh[h_idx * hiddenSize + h_idx] * r;

            dH = dH_prev;
        }

        // Sync before next timestep to ensure all threads complete
        __syncthreads();
    }

    // Store initial hidden gradient
    if (isActive) {
        dH_init[gid] = dH;
    }
}

// ===========================================================================
// GRU COMPUTE GATE GRADIENTS KERNEL
// ===========================================================================

// Computes gate gradients from hidden gradient
// dR[h] is computed correctly as: sum_k(dHCand_k * Uh[k,h]) * prevH[h] * sigmoid'(r[h])
// This is because r[h] is used to gate prevH[h] for all output positions k in the candidate computation
extern ""C"" __global__ void gru_compute_gate_gradients(
    const float* dH,          // [batch, hidden]
    const float* gateZ,       // [batch, hidden]
    const float* gateR,       // [batch, hidden]
    const float* gateHCand,   // [batch, hidden]
    const float* prevH,       // [batch, hidden]
    const float* Uh,          // [hidden, hidden]
    float* dGates,            // [batch, 3*hidden] - output
    int batch,
    int hiddenSize)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batch * hiddenSize;

    if (gid >= totalElements) return;

    int b = gid / hiddenSize;
    int h = gid % hiddenSize;

    // Get cached values
    float z = gateZ[gid];
    float r = gateR[gid];
    float h_cand = gateHCand[gid];
    float h_prev = prevH[gid];

    // Gradient from output
    float dh = dH[gid];

    // Gradient through candidate
    float dHCand = dh * z * tanh_derivative(h_cand);

    // Gradient through update gate
    float dZ = dh * (h_cand - h_prev) * sigmoid_derivative(z);

    // Gradient through reset gate
    // Forward: candidate_k = tanh(... + sum_j(Uh[k,j] * r[j] * prevH[j]) + ...)
    // So r[h] affects all candidate outputs k through the term Uh[k,h] * r[h] * prevH[h]
    // dL/dr[h] = sum_k(dHCand_k * Uh[k,h]) * prevH[h]
    // dL/d(preR[h]) = dL/dr[h] * sigmoid'(r[h])
    float dR_sum = 0.0f;
    for (int k = 0; k < hiddenSize; k++) {
        // Compute dHCand for output k
        float dHCand_k = dH[b * hiddenSize + k] * gateZ[b * hiddenSize + k] *
                         tanh_derivative(gateHCand[b * hiddenSize + k]);
        dR_sum += dHCand_k * Uh[k * hiddenSize + h];
    }
    float dR = dR_sum * h_prev * sigmoid_derivative(r);

    // Store gate gradients
    int gateOffset = b * 3 * hiddenSize;
    dGates[gateOffset + h] = dZ;
    dGates[gateOffset + hiddenSize + h] = dR;
    dGates[gateOffset + 2 * hiddenSize + h] = dHCand;
}

// ===========================================================================
// GRU WEIGHT GRADIENT ACCUMULATION KERNEL
// ===========================================================================

// Accumulates weight gradients across batch dimension
extern ""C"" __global__ void gru_accumulate_weight_gradients(
    const float* input,       // [batch, inputSize]
    const float* prevH,       // [batch, hiddenSize]
    const float* gateR,       // [batch, hiddenSize]
    const float* dGates,      // [batch, 3*hiddenSize]
    float* dWz, float* dWr, float* dWh,
    float* dUz, float* dUr, float* dUh,
    float* dbz, float* dbr, float* dbh,
    int batch,
    int inputSize,
    int hiddenSize)
{
    // Grid: (3*hidden, max(input, hidden))
    int gateIdx = blockIdx.x;  // Which gate row (0-hidden*3)
    int colIdx = blockIdx.y * blockDim.x + threadIdx.x;

    if (gateIdx >= 3 * hiddenSize) return;

    int gateType = gateIdx / hiddenSize;  // 0=Z, 1=R, 2=H
    int h = gateIdx % hiddenSize;

    // Accumulate input weight gradients
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

    // Accumulate hidden weight gradients
    if (colIdx < hiddenSize) {
        float grad = 0.0f;
        for (int b = 0; b < batch; b++) {
            float dGate = dGates[b * 3 * hiddenSize + gateIdx];
            float h_val = prevH[b * hiddenSize + colIdx];

            // For Uh (candidate gate), multiply by reset gate for the INPUT hidden unit (colIdx)
            // not the output unit (h), since GRU applies r element-wise to prevH before Uh multiplication
            if (gateType == 2) {
                float r = gateR[b * hiddenSize + colIdx];
                h_val *= r;
            }

            grad += dGate * h_val;
        }

        if (gateType == 0) atomicAdd(&dUz[h * hiddenSize + colIdx], grad);
        else if (gateType == 1) atomicAdd(&dUr[h * hiddenSize + colIdx], grad);
        else atomicAdd(&dUh[h * hiddenSize + colIdx], grad);
    }

    // Accumulate bias gradients (only first column handles this)
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
            "gru_compute_gate_gradients",
            "gru_accumulate_weight_gradients"
        };
    }
}
