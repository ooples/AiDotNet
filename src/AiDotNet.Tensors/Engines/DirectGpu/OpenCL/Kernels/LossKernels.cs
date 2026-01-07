// Copyright (c) AiDotNet. All rights reserved.
// GPU kernels for loss functions.
// Provides GPU-resident loss computation and gradient calculation.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;

/// <summary>
/// GPU kernels for loss functions.
/// These kernels enable GPU-resident training by computing losses
/// and gradients entirely on the GPU.
/// </summary>
internal static class LossKernels
{
    /// <summary>
    /// Gets all loss function kernel sources.
    /// </summary>
    public static string GetSource()
    {
        return @"
// ===========================================================================
// LOSS FUNCTION KERNELS
// ===========================================================================

// ---------------------------------------------------------------------------
// Mean Squared Error (MSE) Loss
// Formula: loss = (predicted - actual)²
// Gradient: d_loss/d_predicted = 2 * (predicted - actual)
// ---------------------------------------------------------------------------
__kernel void mse_loss(
    __global const float* predicted,
    __global const float* actual,
    __global float* output,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;
    
    float diff = predicted[idx] - actual[idx];
    output[idx] = diff * diff;
}

__kernel void mse_gradient(
    __global const float* predicted,
    __global const float* actual,
    __global float* gradient,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;
    
    gradient[idx] = 2.0f * (predicted[idx] - actual[idx]);
}

// ---------------------------------------------------------------------------
// Binary Cross-Entropy Loss
// Formula: loss = -(actual * log(predicted) + (1-actual) * log(1-predicted))
// Gradient: d_loss/d_predicted = (predicted - actual) / (predicted * (1-predicted))
// ---------------------------------------------------------------------------
__kernel void bce_loss(
    __global const float* predicted,
    __global const float* actual,
    __global float* output,
    const float epsilon,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;
    
    float pred = fmax(fmin(predicted[idx], 1.0f - epsilon), epsilon);
    float target = actual[idx];
    output[idx] = -(target * log(pred) + (1.0f - target) * log(1.0f - pred));
}

__kernel void bce_gradient(
    __global const float* predicted,
    __global const float* actual,
    __global float* gradient,
    const float epsilon,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;
    
    float pred = fmax(fmin(predicted[idx], 1.0f - epsilon), epsilon);
    float target = actual[idx];
    gradient[idx] = (pred - target) / (pred * (1.0f - pred));
}

// ---------------------------------------------------------------------------
// Cross-Entropy Loss (for multi-class classification)
// Formula: loss = -Σ(actual * log(predicted))
// Gradient: d_loss/d_predicted = -actual / predicted
// ---------------------------------------------------------------------------
__kernel void cross_entropy_loss(
    __global const float* predicted,
    __global const float* actual,
    __global float* output,
    const float epsilon,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;
    
    float pred = fmax(predicted[idx], epsilon);
    output[idx] = -actual[idx] * log(pred);
}

__kernel void cross_entropy_gradient(
    __global const float* predicted,
    __global const float* actual,
    __global float* gradient,
    const float epsilon,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;
    
    float pred = fmax(predicted[idx], epsilon);
    gradient[idx] = -actual[idx] / pred;
}

// ---------------------------------------------------------------------------
// Huber Loss (robust regression)
// Formula: loss = 0.5 * diff² if |diff| <= delta, else delta * (|diff| - 0.5*delta)
// Gradient: d_loss/d_predicted = diff if |diff| <= delta, else delta * sign(diff)
// ---------------------------------------------------------------------------
__kernel void huber_loss(
    __global const float* predicted,
    __global const float* actual,
    __global float* output,
    const float delta,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;
    
    float diff = predicted[idx] - actual[idx];
    float abs_diff = fabs(diff);
    
    if (abs_diff <= delta) {
        output[idx] = 0.5f * diff * diff;
    } else {
        output[idx] = delta * (abs_diff - 0.5f * delta);
    }
}

__kernel void huber_gradient(
    __global const float* predicted,
    __global const float* actual,
    __global float* gradient,
    const float delta,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;
    
    float diff = predicted[idx] - actual[idx];
    float abs_diff = fabs(diff);
    
    if (abs_diff <= delta) {
        gradient[idx] = diff;
    } else {
        gradient[idx] = delta * sign(diff);
    }
}

// ---------------------------------------------------------------------------
// Focal Loss (for class imbalance)
// Formula: loss = -alpha * (1-pred)^gamma * actual * log(pred)
// Gradient: complex, see implementation
// ---------------------------------------------------------------------------
__kernel void focal_loss(
    __global const float* predicted,
    __global const float* actual,
    __global float* output,
    const float alpha,
    const float gamma,
    const float epsilon,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;
    
    float pred = fmax(fmin(predicted[idx], 1.0f - epsilon), epsilon);
    float target = actual[idx];
    
    float ce = -log(pred);
    float focal_weight = pow(1.0f - pred, gamma);
    output[idx] = alpha * focal_weight * target * ce;
}

__kernel void focal_gradient(
    __global const float* predicted,
    __global const float* actual,
    __global float* gradient,
    const float alpha,
    const float gamma,
    const float epsilon,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;
    
    float pred = fmax(fmin(predicted[idx], 1.0f - epsilon), epsilon);
    float target = actual[idx];
    
    float focal_weight = pow(1.0f - pred, gamma);
    float d_focal = gamma * pow(1.0f - pred, gamma - 1.0f);
    
    gradient[idx] = alpha * target * (
        d_focal * log(pred) - focal_weight / pred
    );
}

// ---------------------------------------------------------------------------
// Triplet Loss (for metric learning)
// Formula: loss = max(0, ||anchor-pos||² - ||anchor-neg||² + margin)
// ---------------------------------------------------------------------------
__kernel void triplet_loss(
    __global const float* anchor,
    __global const float* positive,
    __global const float* negative,
    __global float* output,
    const float margin,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;
    
    float pos_dist_sq = 0.0f;
    float neg_dist_sq = 0.0f;
    
    // Note: This is simplified - full implementation needs reduction
    float diff_pos = anchor[idx] - positive[idx];
    float diff_neg = anchor[idx] - negative[idx];
    
    pos_dist_sq += diff_pos * diff_pos;
    neg_dist_sq += diff_neg * diff_neg;
    
    output[idx] = fmax(0.0f, pos_dist_sq - neg_dist_sq + margin);
}

// ---------------------------------------------------------------------------
// Contrastive Loss (for siamese networks)
// Formula: loss = (1-label) * 0.5 * dist² + label * 0.5 * max(0, margin-dist)²
// ---------------------------------------------------------------------------
__kernel void contrastive_loss(
    __global const float* pred1,
    __global const float* pred2,
    __global const float* label,
    __global float* output,
    const float margin,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;
    
    float diff = pred1[idx] - pred2[idx];
    float dist_sq = diff * diff;
    float l = label[idx];
    
    // Similar pairs (label=0): minimize distance
    // Dissimilar pairs (label=1): push apart up to margin
    output[idx] = (1.0f - l) * 0.5f * dist_sq + 
                  l * 0.5f * pow(fmax(0.0f, margin - sqrt(dist_sq)), 2.0f);
}

// ---------------------------------------------------------------------------
// Reduction kernel - sum all elements
// Used to compute total loss from element-wise losses
// ---------------------------------------------------------------------------
__kernel void reduce_sum(
    __global const float* input,
    __global float* output,
    const int size,
    __local float* scratch)
{
    int local_id = get_local_id(0);
    int global_id = get_global_id(0);
    int local_size = get_local_size(0);
    
    // Load data into local memory
    scratch[local_id] = (global_id < size) ? input[global_id] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Perform reduction in local memory
    for (int offset = local_size / 2; offset > 0; offset >>= 1) {
        if (local_id < offset) {
            scratch[local_id] += scratch[local_id + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Write result
    if (local_id == 0) {
        output[get_group_id(0)] = scratch[0];
    }
}

// ---------------------------------------------------------------------------
// Reduction kernel - compute mean
// ---------------------------------------------------------------------------
__kernel void reduce_mean(
    __global const float* input,
    __global float* output,
    const int size,
    __local float* scratch)
{
    int local_id = get_local_id(0);
    int global_id = get_global_id(0);
    int local_size = get_local_size(0);
    
    // Load data into local memory
    scratch[local_id] = (global_id < size) ? input[global_id] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Perform reduction in local memory
    for (int offset = local_size / 2; offset > 0; offset >>= 1) {
        if (local_id < offset) {
            scratch[local_id] += scratch[local_id + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Write result (divide by size for mean)
    if (local_id == 0) {
        output[get_group_id(0)] = scratch[0] / (float)size;
    }
}
";
    }
}
