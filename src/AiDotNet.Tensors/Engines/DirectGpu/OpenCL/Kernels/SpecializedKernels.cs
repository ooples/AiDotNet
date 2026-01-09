// Copyright (c) AiDotNet. All rights reserved.
// OpenCL kernels for specialized mathematical operations used in advanced layers:
// - Hyperbolic geometry (Poincare ball operations)
// - Octonion algebra (8D hypercomplex numbers)
// - Quantum computing operations
// - Measurement/probability operations

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;

internal static class SpecializedKernels
{
    public static string GetSource()
    {
        return @"
// ============================================================================
// HYPERBOLIC GEOMETRY KERNELS (Poincare Ball Model)
// ============================================================================

__kernel void poincare_project(
    __global const float* input,
    __global float* output,
    const int batchSize,
    const int dim,
    const float curvature,
    const float epsilon)
{
    int idx = get_global_id(0);
    if (idx >= batchSize) return;

    __global const float* x = input + idx * dim;
    __global float* y = output + idx * dim;

    float sqNorm = 0.0f;
    for (int i = 0; i < dim; i++) {
        sqNorm += x[i] * x[i];
    }

    float maxNorm = 1.0f / sqrt(curvature) - epsilon;
    float maxNormSq = maxNorm * maxNorm;

    if (sqNorm >= maxNormSq) {
        float scale = maxNorm / sqrt(sqNorm);
        for (int i = 0; i < dim; i++) {
            y[i] = x[i] * scale;
        }
    } else {
        for (int i = 0; i < dim; i++) {
            y[i] = x[i];
        }
    }
}

__kernel void mobius_add(
    __global const float* x,
    __global const float* y,
    __global float* output,
    const int batchSize,
    const int dim,
    const float curvature)
{
    int idx = get_global_id(0);
    if (idx >= batchSize) return;

    __global const float* xi = x + idx * dim;
    __global const float* yi = y + idx * dim;
    __global float* out = output + idx * dim;

    float c = curvature;
    float xNormSq = 0.0f, yNormSq = 0.0f, xyDot = 0.0f;

    for (int i = 0; i < dim; i++) {
        xNormSq += xi[i] * xi[i];
        yNormSq += yi[i] * yi[i];
        xyDot += xi[i] * yi[i];
    }

    float num_coef_x = 1.0f + 2.0f * c * xyDot + c * yNormSq;
    float num_coef_y = 1.0f - c * xNormSq;
    float denom = 1.0f + 2.0f * c * xyDot + c * c * xNormSq * yNormSq;

    if (fabs(denom) < 1e-10f) denom = 1e-10f;

    for (int i = 0; i < dim; i++) {
        out[i] = (num_coef_x * xi[i] + num_coef_y * yi[i]) / denom;
    }
}

__kernel void poincare_exp_map(
    __global const float* basePoint,
    __global const float* tangentVec,
    __global float* output,
    const int batchSize,
    const int dim,
    const float curvature)
{
    int idx = get_global_id(0);
    if (idx >= batchSize) return;

    __global const float* xp = basePoint + idx * dim;
    __global const float* vp = tangentVec + idx * dim;
    __global float* out = output + idx * dim;

    float c = curvature;
    float sqrtC = sqrt(c);

    float xNormSq = 0.0f, vNormSq = 0.0f;
    for (int i = 0; i < dim; i++) {
        xNormSq += xp[i] * xp[i];
        vNormSq += vp[i] * vp[i];
    }
    float vNorm = sqrt(vNormSq);

    if (vNorm < 1e-10f) {
        for (int i = 0; i < dim; i++) {
            out[i] = xp[i];
        }
        return;
    }

    float conformalFactor = 1.0f - c * xNormSq;
    float lambda = conformalFactor / 2.0f;
    float arg = sqrtC * vNorm * lambda;
    float scale = tanh(arg) / (sqrtC * vNorm);

    // Compute scaled tangent and Mobius addition inline
    float svNormSq = 0.0f, xsvDot = 0.0f;
    for (int i = 0; i < dim; i++) {
        float sv = vp[i] * scale;
        svNormSq += sv * sv;
        xsvDot += xp[i] * sv;
    }

    float num_coef_x = 1.0f + 2.0f * c * xsvDot + c * svNormSq;
    float num_coef_sv = 1.0f - c * xNormSq;
    float denom = 1.0f + 2.0f * c * xsvDot + c * c * xNormSq * svNormSq;
    if (fabs(denom) < 1e-10f) denom = 1e-10f;

    for (int i = 0; i < dim; i++) {
        float sv = vp[i] * scale;
        out[i] = (num_coef_x * xp[i] + num_coef_sv * sv) / denom;
    }
}

__kernel void poincare_distance(
    __global const float* x,
    __global const float* y,
    __global float* output,
    const int batchSize,
    const int dim,
    const float curvature)
{
    int idx = get_global_id(0);
    if (idx >= batchSize) return;

    __global const float* xi = x + idx * dim;
    __global const float* yi = y + idx * dim;

    float c = curvature;
    float sqrtC = sqrt(c);

    float xNormSq = 0.0f, yNormSq = 0.0f, negXyDot = 0.0f;
    for (int i = 0; i < dim; i++) {
        xNormSq += xi[i] * xi[i];
        yNormSq += yi[i] * yi[i];
        negXyDot += (-xi[i]) * yi[i];
    }

    float num_coef_negX = 1.0f + 2.0f * c * negXyDot + c * yNormSq;
    float num_coef_y = 1.0f - c * xNormSq;
    float denom = 1.0f + 2.0f * c * negXyDot + c * c * xNormSq * yNormSq;
    if (fabs(denom) < 1e-10f) denom = 1e-10f;

    float diffNormSq = 0.0f;
    for (int i = 0; i < dim; i++) {
        float diff = (num_coef_negX * (-xi[i]) + num_coef_y * yi[i]) / denom;
        diffNormSq += diff * diff;
    }

    float diffNorm = sqrt(diffNormSq);
    float arg = sqrtC * diffNorm;
    if (arg >= 1.0f) arg = 1.0f - 1e-6f;

    output[idx] = (2.0f / sqrtC) * atanh(arg);
}

__kernel void hyperbolic_linear_forward(
    __global const float* input,
    __global const float* weights,
    __global const float* biases,
    __global float* output,
    const int batchSize,
    const int inputFeatures,
    const int outputFeatures,
    const float curvature,
    const float epsilon)
{
    int tid = get_global_id(0);
    int b = tid / outputFeatures;
    int o = tid % outputFeatures;

    if (b >= batchSize || o >= outputFeatures) return;

    float c = curvature;
    float sqrtC = sqrt(c);
    float maxNorm = 1.0f / sqrtC - epsilon;
    float maxNormSq = maxNorm * maxNorm;

    __global const float* xi = input + b * inputFeatures;
    __global const float* wo = weights + o * inputFeatures;
    __global const float* bo = biases + o * inputFeatures;

    // Project input
    float inputNormSq = 0.0f;
    for (int i = 0; i < inputFeatures; i++) {
        inputNormSq += xi[i] * xi[i];
    }
    float inputScale = (inputNormSq >= maxNormSq) ? (maxNorm / sqrt(inputNormSq)) : 1.0f;

    // Compute exp_origin(weight)
    float weightNormSq = 0.0f;
    for (int i = 0; i < inputFeatures; i++) {
        weightNormSq += wo[i] * wo[i];
    }
    float weightNorm = sqrt(weightNormSq);
    float weightScale = (weightNorm > 1e-10f) ? (tanh(sqrtC * weightNorm / 2.0f) / (sqrtC * weightNorm)) : 0.0f;

    // Mobius add: projInput (+) weightPoint
    float wpNormSq = 0.0f, piWpDot = 0.0f, piNormSq = 0.0f;
    for (int i = 0; i < inputFeatures; i++) {
        float pi = xi[i] * inputScale;
        float wp = wo[i] * weightScale;
        wpNormSq += wp * wp;
        piNormSq += pi * pi;
        piWpDot += pi * wp;
    }

    float num1 = 1.0f + 2.0f * c * piWpDot + c * wpNormSq;
    float num2 = 1.0f - c * piNormSq;
    float den = 1.0f + 2.0f * c * piWpDot + c * c * piNormSq * wpNormSq;
    if (fabs(den) < 1e-10f) den = 1e-10f;

    // Compute transformed + bias
    float biasNormSq = 0.0f;
    for (int i = 0; i < inputFeatures; i++) {
        biasNormSq += bo[i] * bo[i];
    }
    float biasScale = (biasNormSq >= maxNormSq) ? (maxNorm / sqrt(biasNormSq)) : 1.0f;

    float pbNormSq = 0.0f, trPbDot = 0.0f, trNormSq = 0.0f;
    for (int i = 0; i < inputFeatures; i++) {
        float pi = xi[i] * inputScale;
        float wp = wo[i] * weightScale;
        float tr = (num1 * pi + num2 * wp) / den;
        float pb = bo[i] * biasScale;
        pbNormSq += pb * pb;
        trNormSq += tr * tr;
        trPbDot += tr * pb;
    }

    num1 = 1.0f + 2.0f * c * trPbDot + c * pbNormSq;
    num2 = 1.0f - c * trNormSq;
    den = 1.0f + 2.0f * c * trPbDot + c * c * trNormSq * pbNormSq;
    if (fabs(den) < 1e-10f) den = 1e-10f;

    float wbNormSq = 0.0f;
    for (int i = 0; i < inputFeatures; i++) {
        float pi = xi[i] * inputScale;
        float wp = wo[i] * weightScale;
        float tr = (1.0f + 2.0f * c * piWpDot + c * wpNormSq) * pi / den;
        tr += (1.0f - c * piNormSq) * wp / den;
        float pb = bo[i] * biasScale;
        float wb = (num1 * tr + num2 * pb) / den;
        wbNormSq += wb * wb;
    }

    float wbNorm = sqrt(wbNormSq);
    float arg = sqrtC * wbNorm;
    if (arg >= 1.0f) arg = 1.0f - 1e-6f;

    output[b * outputFeatures + o] = (2.0f / sqrtC) * atanh(arg);
}

// ============================================================================
// OCTONION ALGEBRA KERNELS
// ============================================================================

__kernel void octonion_multiply(
    __global const float* a,
    __global const float* b,
    __global float* output,
    const int count)
{
    int idx = get_global_id(0);
    if (idx >= count) return;

    __global const float* ai = a + idx * 8;
    __global const float* bi = b + idx * 8;
    __global float* out = output + idx * 8;

    float a0 = ai[0], a1 = ai[1], a2 = ai[2], a3 = ai[3];
    float a4 = ai[4], a5 = ai[5], a6 = ai[6], a7 = ai[7];
    float b0 = bi[0], b1 = bi[1], b2 = bi[2], b3 = bi[3];
    float b4 = bi[4], b5 = bi[5], b6 = bi[6], b7 = bi[7];

    out[0] = a0*b0 - a1*b1 - a2*b2 - a3*b3 - a4*b4 - a5*b5 - a6*b6 - a7*b7;
    out[1] = a0*b1 + a1*b0 + a2*b3 - a3*b2 + a4*b5 - a5*b4 - a6*b7 + a7*b6;
    out[2] = a0*b2 - a1*b3 + a2*b0 + a3*b1 + a4*b6 + a5*b7 - a6*b4 - a7*b5;
    out[3] = a0*b3 + a1*b2 - a2*b1 + a3*b0 + a4*b7 - a5*b6 + a6*b5 - a7*b4;
    out[4] = a0*b4 - a1*b5 - a2*b6 - a3*b7 + a4*b0 + a5*b1 + a6*b2 + a7*b3;
    out[5] = a0*b5 + a1*b4 - a2*b7 + a3*b6 - a4*b1 + a5*b0 - a6*b3 + a7*b2;
    out[6] = a0*b6 + a1*b7 + a2*b4 - a3*b5 - a4*b2 + a5*b3 + a6*b0 - a7*b1;
    out[7] = a0*b7 - a1*b6 + a2*b5 + a3*b4 - a4*b3 - a5*b2 + a6*b1 + a7*b0;
}

__kernel void octonion_add(
    __global const float* a,
    __global const float* b,
    __global float* output,
    const int count)
{
    int idx = get_global_id(0);
    if (idx >= count * 8) return;
    output[idx] = a[idx] + b[idx];
}

__kernel void octonion_linear_forward(
    __global const float* input,
    __global const float* weights,
    __global const float* biases,
    __global float* output,
    const int batchSize,
    const int inputFeatures,
    const int outputFeatures)
{
    int tid = get_global_id(0);
    int b = tid / outputFeatures;
    int o = tid % outputFeatures;

    if (b >= batchSize || o >= outputFeatures) return;

    float acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    float acc4 = 0, acc5 = 0, acc6 = 0, acc7 = 0;

    __global const float* bias = biases + o * 8;
    acc0 = bias[0]; acc1 = bias[1]; acc2 = bias[2]; acc3 = bias[3];
    acc4 = bias[4]; acc5 = bias[5]; acc6 = bias[6]; acc7 = bias[7];

    for (int i = 0; i < inputFeatures; i++) {
        __global const float* inp = input + b * inputFeatures * 8 + i * 8;
        __global const float* w = weights + (o * inputFeatures + i) * 8;

        float a0 = w[0], a1 = w[1], a2 = w[2], a3 = w[3];
        float a4 = w[4], a5 = w[5], a6 = w[6], a7 = w[7];
        float b0 = inp[0], b1 = inp[1], b2 = inp[2], b3 = inp[3];
        float b4 = inp[4], b5 = inp[5], b6 = inp[6], b7 = inp[7];

        acc0 += a0*b0 - a1*b1 - a2*b2 - a3*b3 - a4*b4 - a5*b5 - a6*b6 - a7*b7;
        acc1 += a0*b1 + a1*b0 + a2*b3 - a3*b2 + a4*b5 - a5*b4 - a6*b7 + a7*b6;
        acc2 += a0*b2 - a1*b3 + a2*b0 + a3*b1 + a4*b6 + a5*b7 - a6*b4 - a7*b5;
        acc3 += a0*b3 + a1*b2 - a2*b1 + a3*b0 + a4*b7 - a5*b6 + a6*b5 - a7*b4;
        acc4 += a0*b4 - a1*b5 - a2*b6 - a3*b7 + a4*b0 + a5*b1 + a6*b2 + a7*b3;
        acc5 += a0*b5 + a1*b4 - a2*b7 + a3*b6 - a4*b1 + a5*b0 - a6*b3 + a7*b2;
        acc6 += a0*b6 + a1*b7 + a2*b4 - a3*b5 - a4*b2 + a5*b3 + a6*b0 - a7*b1;
        acc7 += a0*b7 - a1*b6 + a2*b5 + a3*b4 - a4*b3 - a5*b2 + a6*b1 + a7*b0;
    }

    __global float* outPtr = output + b * outputFeatures * 8 + o * 8;
    outPtr[0] = acc0; outPtr[1] = acc1; outPtr[2] = acc2; outPtr[3] = acc3;
    outPtr[4] = acc4; outPtr[5] = acc5; outPtr[6] = acc6; outPtr[7] = acc7;
}

// ============================================================================
// QUANTUM COMPUTING KERNELS
// ============================================================================

__kernel void quantum_measurement(
    __global const float* realPart,
    __global const float* imagPart,
    __global float* probabilities,
    const int batchSize,
    const int stateSize)
{
    int tid = get_global_id(0);
    int b = tid / stateSize;
    int i = tid % stateSize;

    if (b >= batchSize) return;

    int idx = b * stateSize + i;
    float real = realPart[idx];
    float imag = imagPart[idx];
    probabilities[idx] = real * real + imag * imag;
}

__kernel void normalize_probabilities(
    __global float* probabilities,
    __local float* sdata,
    const int batchSize,
    const int stateSize)
{
    int b = get_group_id(0);
    if (b >= batchSize) return;

    int tid = get_local_id(0);
    int localSize = get_local_size(0);
    __global float* probs = probabilities + b * stateSize;

    float sum = 0.0f;
    for (int i = tid; i < stateSize; i += localSize) {
        sum += probs[i];
    }
    sdata[tid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = localSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float totalSum = sdata[0];
    if (totalSum < 1e-10f) totalSum = 1e-10f;

    for (int i = tid; i < stateSize; i += localSize) {
        probs[i] /= totalSum;
    }
}

__kernel void complex_matvec(
    __global const float* matReal,
    __global const float* matImag,
    __global const float* vecReal,
    __global const float* vecImag,
    __global float* outReal,
    __global float* outImag,
    const int batchSize,
    const int dim)
{
    int tid = get_global_id(0);
    int b = tid / dim;
    int row = tid % dim;

    if (b >= batchSize) return;

    __global const float* vr = vecReal + b * dim;
    __global const float* vi = vecImag + b * dim;

    float sumReal = 0.0f;
    float sumImag = 0.0f;

    for (int col = 0; col < dim; col++) {
        int matIdx = row * dim + col;
        float mr = matReal[matIdx];
        float mi = matImag[matIdx];
        float xr = vr[col];
        float xi = vi[col];

        sumReal += mr * xr - mi * xi;
        sumImag += mr * xi + mi * xr;
    }

    outReal[b * dim + row] = sumReal;
    outImag[b * dim + row] = sumImag;
}

__kernel void quantum_rotation(
    __global const float* stateReal,
    __global const float* stateImag,
    __global float* outReal,
    __global float* outImag,
    __global const float* angles,
    const int numQubits,
    const int batchSize)
{
    int b = get_group_id(0);
    if (b >= batchSize) return;

    int dim = 1 << numQubits;
    int tid = get_local_id(0);
    int localSize = get_local_size(0);

    __global const float* sr = stateReal + b * dim;
    __global const float* si = stateImag + b * dim;
    __global float* or_ = outReal + b * dim;
    __global float* oi = outImag + b * dim;

    for (int i = tid; i < dim; i += localSize) {
        or_[i] = sr[i];
        oi[i] = si[i];
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    for (int q = 0; q < numQubits; q++) {
        float theta = angles[q];
        float cosHalf = cos(theta / 2.0f);
        float sinHalf = sin(theta / 2.0f);

        int stride = 1 << q;

        for (int i = tid; i < dim / 2; i += localSize) {
            int block = i / stride;
            int within = i % stride;
            int idx0 = block * 2 * stride + within;
            int idx1 = idx0 + stride;

            float r0 = or_[idx0], i0 = oi[idx0];
            float r1 = or_[idx1], i1 = oi[idx1];

            or_[idx0] = cosHalf * r0 - sinHalf * r1;
            oi[idx0] = cosHalf * i0 - sinHalf * i1;
            or_[idx1] = sinHalf * r0 + cosHalf * r1;
            oi[idx1] = sinHalf * i0 + cosHalf * i1;
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}

__kernel void measurement_forward(
    __global const float* input,
    __global float* output,
    __local float* sdata,
    const int batchSize,
    const int stateSize)
{
    int b = get_group_id(0);
    if (b >= batchSize) return;

    int tid = get_local_id(0);
    int localSize = get_local_size(0);
    __global const float* inp = input + b * stateSize * 2;
    __global float* out = output + b * stateSize;

    float localSum = 0.0f;
    for (int i = tid; i < stateSize; i += localSize) {
        float real = inp[i * 2];
        float imag = inp[i * 2 + 1];
        float mag = real * real + imag * imag;
        out[i] = mag;
        localSum += mag;
    }

    sdata[tid] = localSum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = localSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float totalSum = sdata[0];
    if (totalSum < 1e-10f) totalSum = 1e-10f;

    for (int i = tid; i < stateSize; i += localSize) {
        out[i] /= totalSum;
    }
}
";
    }

    public static string[] GetKernelNames()
    {
        return
        [
            // Hyperbolic geometry
            "poincare_project",
            "mobius_add",
            "poincare_exp_map",
            "poincare_distance",
            "hyperbolic_linear_forward",
            // Octonion algebra
            "octonion_multiply",
            "octonion_add",
            "octonion_linear_forward",
            // Quantum computing
            "quantum_measurement",
            "normalize_probabilities",
            "complex_matvec",
            "quantum_rotation",
            "measurement_forward"
        ];
    }
}
