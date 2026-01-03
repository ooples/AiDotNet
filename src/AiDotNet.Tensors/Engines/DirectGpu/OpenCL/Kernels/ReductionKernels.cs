// Copyright (c) AiDotNet. All rights reserved.
// Reduction kernels for sum, mean, max operations.
// Works on ALL .NET versions including .NET Framework 4.6.2.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels
{
    /// <summary>
    /// GPU kernels for reduction operations (sum, max, mean, etc.).
    /// </summary>
    internal static class ReductionKernels
    {
        /// <summary>
        /// Gets all reduction kernel sources.
        /// </summary>
        public static string GetSource()
        {
            return @"
// ===========================================================================
// REDUCTION KERNELS
// ===========================================================================

#define REDUCTION_BLOCK_SIZE 256

// Parallel sum reduction
// Input is reduced to partial sums, one per workgroup
__kernel void reduce_sum(
    __global const float* input,
    __global float* output,
    __local float* scratch,
    const int size)
{
    const int globalIdx = get_global_id(0);
    const int localIdx = get_local_id(0);
    const int groupIdx = get_group_id(0);
    const int localSize = get_local_size(0);

    // Load into local memory
    scratch[localIdx] = (globalIdx < size) ? input[globalIdx] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Tree reduction in local memory
    for (int stride = localSize / 2; stride > 0; stride >>= 1) {
        if (localIdx < stride) {
            scratch[localIdx] += scratch[localIdx + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write partial sum
    if (localIdx == 0) {
        output[groupIdx] = scratch[0];
    }
}

// Parallel max reduction
__kernel void reduce_max(
    __global const float* input,
    __global float* output,
    __local float* scratch,
    const int size)
{
    const int globalIdx = get_global_id(0);
    const int localIdx = get_local_id(0);
    const int groupIdx = get_group_id(0);
    const int localSize = get_local_size(0);

    // Load into local memory
    scratch[localIdx] = (globalIdx < size) ? input[globalIdx] : -INFINITY;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Tree reduction in local memory
    for (int stride = localSize / 2; stride > 0; stride >>= 1) {
        if (localIdx < stride) {
            scratch[localIdx] = fmax(scratch[localIdx], scratch[localIdx + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write partial max
    if (localIdx == 0) {
        output[groupIdx] = scratch[0];
    }
}

// Parallel min reduction
__kernel void reduce_min(
    __global const float* input,
    __global float* output,
    __local float* scratch,
    const int size)
{
    const int globalIdx = get_global_id(0);
    const int localIdx = get_local_id(0);
    const int groupIdx = get_group_id(0);
    const int localSize = get_local_size(0);

    // Load into local memory
    scratch[localIdx] = (globalIdx < size) ? input[globalIdx] : INFINITY;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Tree reduction in local memory
    for (int stride = localSize / 2; stride > 0; stride >>= 1) {
        if (localIdx < stride) {
            scratch[localIdx] = fmin(scratch[localIdx], scratch[localIdx + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write partial min
    if (localIdx == 0) {
        output[groupIdx] = scratch[0];
    }
}

// Sum along axis (rows or columns)
// For a 2D tensor of shape [outerSize, reduceSize], sum along the reduce dimension
__kernel void sum_axis(
    __global const float* input,
    __global float* output,
    const int outerSize,
    const int reduceSize)
{
    const int outerIdx = get_global_id(0);
    if (outerIdx >= outerSize) return;

    float sum = 0.0f;
    for (int i = 0; i < reduceSize; i++) {
        sum += input[outerIdx * reduceSize + i];
    }
    output[outerIdx] = sum;
}

// Max along axis
__kernel void max_axis(
    __global const float* input,
    __global float* output,
    const int outerSize,
    const int reduceSize)
{
    const int outerIdx = get_global_id(0);
    if (outerIdx >= outerSize) return;

    float maxVal = -INFINITY;
    for (int i = 0; i < reduceSize; i++) {
        maxVal = fmax(maxVal, input[outerIdx * reduceSize + i]);
    }
    output[outerIdx] = maxVal;
}

// Argmax along axis
__kernel void argmax_axis(
    __global const float* input,
    __global int* output,
    const int outerSize,
    const int reduceSize)
{
    const int outerIdx = get_global_id(0);
    if (outerIdx >= outerSize) return;

    float maxVal = -INFINITY;
    int maxIdx = 0;
    for (int i = 0; i < reduceSize; i++) {
        float val = input[outerIdx * reduceSize + i];
        if (val > maxVal) {
            maxVal = val;
            maxIdx = i;
        }
    }
    output[outerIdx] = maxIdx;
}
";
        }

        /// <summary>
        /// Gets kernel names for compilation.
        /// </summary>
        public static string[] GetKernelNames()
        {
            return new string[]
            {
                "reduce_sum", "reduce_max", "reduce_min",
                "sum_axis", "max_axis", "argmax_axis"
            };
        }
    }
}
