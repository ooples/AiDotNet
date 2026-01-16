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

    // Use per-element reset gate r_j for proper GRU computation
    // In standard GRU: candidate = tanh(Wh*x + Uh*(r ⊙ h_prev) + bh)
    // We need to compute r_j for each hidden unit j, not use scalar r
    for (int j = 0; j < hiddenSize; j++) {
        float h_val = prevH[b * hiddenSize + j];
        // Compute r_j = sigmoid(br[j] + Wr[j,:] @ x + Ur[j,:] @ h_prev)
        float sumRj = br[j];
        for (int ii = 0; ii < inputSize; ii++) {
            sumRj += Wr[j * inputSize + ii] * input[b * inputSize + ii];
        }
        for (int jj = 0; jj < hiddenSize; jj++) {
            sumRj += Ur[j * hiddenSize + jj] * prevH[b * hiddenSize + jj];
        }
        float rj = sigmoid(sumRj);
        sumH += Uh[h * hiddenSize + j] * rj * h_val;
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
    int b = gid / hiddenSize;
    int h_idx = gid % hiddenSize;
    int isValid = (gid < totalElements) ? 1 : 0;

    float h_val = 0.0f;
    if (isValid) {
        h_val = h_init[gid];
    }

    for (int t = 0; t < timeSteps; t++) {
        float z = 0.0f;
        float r = 0.0f;
        int gateOffset = t * batch * 3 * hiddenSize + b * 3 * hiddenSize;

        if (isValid) {
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

            z = sigmoid(sumZ);
            r = sigmoid(sumR);

            // Store r to gates buffer immediately so other threads can read it
            gates[gateOffset + hiddenSize + h_idx] = r;
        }

        // Sync to ensure all threads have stored their r values
        __syncthreads();

        if (isValid) {
            float sumH = bh[h_idx];

            int inputOffset = (b * timeSteps + t) * inputSize;
            for (int i = 0; i < inputSize; i++) {
                float x_val = input[inputOffset + i];
                sumH += Wh[h_idx * inputSize + i] * x_val;
            }

            // Use per-element reset gate r_j for proper GRU computation
            // In standard GRU: candidate = tanh(Wh*x + Uh*(r ⊙ h_prev) + bh)
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

            // Store remaining gates for backward pass (r was stored earlier)
            gates[gateOffset + h_idx] = z;
            gates[gateOffset + 2 * hiddenSize + h_idx] = h_candidate;
        }

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

    // Compute sumUhH with correct index ordering
    // Uh is [hiddenSize, hiddenSize] in row-major, so Uh[row, col] = Uh[row * hiddenSize + col]
    // For the reset gate gradient, we need sum over input j: Uh[j, h] * prevH[j]
    // which is Uh[j * hiddenSize + h] in row-major
    float sumUhH = 0.0f;
    for (int j = 0; j < hiddenSize; j++) {
        sumUhH += Uh[j * hiddenSize + h] * prevH[b * hiddenSize + j];
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
            // dPrevH[j] = dH[j] * (1-z[j]) + sum_h(dZ[h]*Uz[h,j] + dR[h]*Ur[h,j] + dHCand[h]*Uh[h,j]*r[j])
            float dH_prev = dH * (1.0f - z);  // Direct path for this hidden unit

            // Accumulate gradient contributions from all hidden output positions h
            for (int h = 0; h < hiddenSize; h++) {
                // Get cached gate values for output position h
                float z_h = gates[gateOffset + h];
                float r_h = gates[gateOffset + hiddenSize + h];
                float h_cand_h = gates[gateOffset + 2 * hiddenSize + h];

                // Get previous hidden state for position h
                float h_prev_h;
                if (t == 0) {
                    h_prev_h = h_init[b * hiddenSize + h];
                } else {
                    h_prev_h = h_states[(t - 1) * batch * hiddenSize + b * hiddenSize + h];
                }

                // Get accumulated hidden gradient for position h
                float dH_h;
                int h_gid = b * hiddenSize + h;
                int h_local_idx = h_gid - (blockIdx.x * blockDim.x);
                if (h_local_idx >= 0 && h_local_idx < blockDim.x) {
                    dH_h = shared_dH[h_local_idx];
                } else {
                    dH_h = dH_buffer[h_gid];
                }

                // Compute gate gradients for output position h
                float dHCand_h = dH_h * z_h * tanh_derivative(h_cand_h);
                float dZ_h = dH_h * (h_cand_h - h_prev_h) * sigmoid_derivative(z_h);

                // Compute dR[h] with nested loop: dR[h] = sum_k(dHCand_k * Uh[k,h]) * prevH[h] * sigmoid'(r[h])
                float dR_sum_h = 0.0f;
                for (int k = 0; k < hiddenSize; k++) {
                    float z_k = gates[gateOffset + k];
                    float h_cand_k = gates[gateOffset + 2 * hiddenSize + k];
                    float dH_k;
                    int k_gid = b * hiddenSize + k;
                    int k_local_idx = k_gid - (blockIdx.x * blockDim.x);
                    if (k_local_idx >= 0 && k_local_idx < blockDim.x) {
                        dH_k = shared_dH[k_local_idx];
                    } else {
                        dH_k = dH_buffer[k_gid];
                    }
                    float dHCand_k = dH_k * z_k * tanh_derivative(h_cand_k);
                    dR_sum_h += dHCand_k * Uh[k * hiddenSize + h];
                }
                float dR_h = dR_sum_h * h_prev_h * sigmoid_derivative(r_h);

                // Accumulate contributions from all gates to prev hidden at position h_idx
                dH_prev += dZ_h * Uz[h * hiddenSize + h_idx];
                dH_prev += dR_h * Ur[h * hiddenSize + h_idx];
                dH_prev += dHCand_h * Uh[h * hiddenSize + h_idx] * r;
            }

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
// UNIFIED GRU CELL BACKWARD KERNEL
// ===========================================================================
// Matches the OpenCL interface for cross-backend compatibility.
// Computes gate gradients and partial prevH (direct path only).
// Must be followed by gru_backward_prevh_unified for full BPTT gradient.

extern ""C"" __global__ void gru_cell_backward_unified(
    const float* gradH,       // [batch, hiddenSize] - gradient from next layer
    const float* gateR,       // [batch, hiddenSize] - reset gate values
    const float* gateZ,       // [batch, hiddenSize] - update gate values
    const float* gateN,       // [batch, hiddenSize] - candidate values
    const float* prevH,       // [batch, hiddenSize] - previous hidden state
    const float* weightsHh,   // [3 * hiddenSize, hiddenSize] - recurrent weights (R, Z, N stacked)
    float* gradPrevH,         // [batch, hiddenSize] - output: partial gradient (direct path only)
    float* gradGateR,         // [batch, hiddenSize] - output: reset gate gradient
    float* gradGateZ,         // [batch, hiddenSize] - output: update gate gradient
    float* gradGateN,         // [batch, hiddenSize] - output: candidate gradient
    int batch,
    int hiddenSize)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
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
    // dR = dN * (Wn_hh @ h_prev) * sigmoid'(r)
    float Wn_h_prev_dot = 0.0f;
    for (int j = 0; j < hiddenSize; j++) {
        float hPrevJ = prevH[b * hiddenSize + j];
        // Wn_hh is at offset 2*hiddenSize in weightsHh (layout: R, Z, N)
        Wn_h_prev_dot += hPrevJ * weightsHh[(2 * hiddenSize + h) * hiddenSize + j];
    }
    float dR = dN * Wn_h_prev_dot * sigmoid_derivative(r);

    // Store gate gradients
    gradGateR[gid] = dR;
    gradGateZ[gid] = dZ;
    gradGateN[gid] = dN;

    // Direct path gradient to prev hidden: dL/dh_prev from (1-z) branch = dH * (1-z)
    // NOTE: This is ONLY the direct path. Full gradient requires gru_backward_prevh_unified.
    float dHPrev = dH * (1.0f - z);
    gradPrevH[gid] = dHPrev;
}

// ===========================================================================
// UNIFIED GRU BACKWARD PREVH KERNEL
// ===========================================================================
// Computes full gradient to previous hidden state by summing contributions
// from all hidden positions through the gate weight matrices.
// Must be called AFTER gru_cell_backward_unified which computes gate gradients.

extern ""C"" __global__ void gru_backward_prevh_unified(
    const float* gradGateR,   // [batch, hiddenSize] - from gru_cell_backward_unified
    const float* gradGateZ,   // [batch, hiddenSize] - from gru_cell_backward_unified
    const float* gradGateN,   // [batch, hiddenSize] - from gru_cell_backward_unified
    const float* gradH,       // [batch, hiddenSize] - gradient from output
    const float* gateR,       // [batch, hiddenSize] - reset gate values
    const float* gateZ,       // [batch, hiddenSize] - update gate values
    const float* weightsHh,   // [3 * hiddenSize, hiddenSize] - recurrent weights (R, Z, N stacked)
    float* gradPrevH,         // [batch, hiddenSize] - output: OVERWRITES with full gradient
    int batch,
    int hiddenSize)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batch * hiddenSize;

    if (gid >= totalElements) return;

    int b = gid / hiddenSize;
    int j = gid % hiddenSize;  // This thread computes gradPrevH[b, j]

    float z = gateZ[gid];
    float dH = gradH[gid];

    // Gradient through (1-z) path (direct contribution) for variant 1: h_new = (1-z)*h_prev + z*n
    float gradSum = dH * (1.0f - z);

    // Gradient through all gates from all hidden positions h
    for (int hh = 0; hh < hiddenSize; hh++) {
        int batchHiddenIdx = b * hiddenSize + hh;

        float dR = gradGateR[batchHiddenIdx];
        float dZ = gradGateZ[batchHiddenIdx];
        float dN = gradGateN[batchHiddenIdx];
        float r = gateR[batchHiddenIdx];

        // weightsHh layout: [R weights, Z weights, N weights] each [hiddenSize, hiddenSize]
        // R weights: weightsHh[hh * hiddenSize + j] for Ur[hh, j]
        // Z weights: weightsHh[(hiddenSize + hh) * hiddenSize + j] for Uz[hh, j]
        // N weights: weightsHh[(2 * hiddenSize + hh) * hiddenSize + j] for Uh[hh, j]
        gradSum += dR * weightsHh[hh * hiddenSize + j];
        gradSum += dZ * weightsHh[(hiddenSize + hh) * hiddenSize + j];
        gradSum += dN * r * weightsHh[(2 * hiddenSize + hh) * hiddenSize + j];
    }

    gradPrevH[gid] = gradSum;
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
            "gru_backward_sequence",
            "gru_cell_backward_unified",
            "gru_backward_prevh_unified"
        };
    }
}
