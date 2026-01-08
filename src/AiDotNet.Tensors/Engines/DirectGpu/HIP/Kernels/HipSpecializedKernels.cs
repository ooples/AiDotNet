// Copyright (c) AiDotNet. All rights reserved.
// HIP kernels for specialized mathematical operations used in advanced layers:
// - Hyperbolic geometry (Poincare ball operations)
// - Octonion algebra (8D hypercomplex numbers)
// - Quantum computing operations
// - Measurement/probability operations
// Note: HIP is source-compatible with CUDA device code.

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

internal static class HipSpecializedKernels
{
    public static string GetSource()
    {
        return @"
// HIP RTC Compatibility - no includes needed, device intrinsics are built-in
#ifndef INFINITY
#define INFINITY __builtin_huge_valf()
#endif

// ============================================================================
// HYPERBOLIC GEOMETRY KERNELS (Poincare Ball Model)
// ============================================================================

// Poincare ball projection: project point to be inside the ball
extern ""C"" __global__ void poincare_project(
    const float* input, float* output,
    int batchSize, int dim, float curvature, float epsilon)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batchSize) return;

    const float* x = input + idx * dim;
    float* y = output + idx * dim;

    float sqNorm = 0.0f;
    for (int i = 0; i < dim; i++) {
        sqNorm += x[i] * x[i];
    }

    float maxNorm = 1.0f / sqrtf(curvature) - epsilon;
    float maxNormSq = maxNorm * maxNorm;

    if (sqNorm >= maxNormSq) {
        float scale = maxNorm / sqrtf(sqNorm);
        for (int i = 0; i < dim; i++) {
            y[i] = x[i] * scale;
        }
    } else {
        for (int i = 0; i < dim; i++) {
            y[i] = x[i];
        }
    }
}

// Mobius addition: x (+) y in Poincare ball
extern ""C"" __global__ void mobius_add(
    const float* x, const float* y, float* output,
    int batchSize, int dim, float curvature)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batchSize) return;

    const float* xi = x + idx * dim;
    const float* yi = y + idx * dim;
    float* out = output + idx * dim;

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

    if (fabsf(denom) < 1e-10f) denom = 1e-10f;

    for (int i = 0; i < dim; i++) {
        out[i] = (num_coef_x * xi[i] + num_coef_y * yi[i]) / denom;
    }
}

// Poincare exponential map
extern ""C"" __global__ void poincare_exp_map(
    const float* basePoint, const float* tangentVec, float* output,
    int batchSize, int dim, float curvature)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batchSize) return;

    const float* x = basePoint + idx * dim;
    const float* v = tangentVec + idx * dim;
    float* out = output + idx * dim;

    float c = curvature;
    float sqrtC = sqrtf(c);

    float xNormSq = 0.0f, vNorm = 0.0f;
    for (int i = 0; i < dim; i++) {
        xNormSq += x[i] * x[i];
        vNorm += v[i] * v[i];
    }
    vNorm = sqrtf(vNorm);

    if (vNorm < 1e-10f) {
        for (int i = 0; i < dim; i++) {
            out[i] = x[i];
        }
        return;
    }

    float conformalFactor = 1.0f - c * xNormSq;
    float lambda = conformalFactor / 2.0f;
    float arg = sqrtC * vNorm * lambda;
    float scale = tanhf(arg) / (sqrtC * vNorm);

    float scaledV[32];
    for (int i = 0; i < dim && i < 32; i++) {
        scaledV[i] = v[i] * scale;
    }

    float svNormSq = 0.0f, xsvDot = 0.0f;
    for (int i = 0; i < dim && i < 32; i++) {
        svNormSq += scaledV[i] * scaledV[i];
        xsvDot += x[i] * scaledV[i];
    }

    float num_coef_x = 1.0f + 2.0f * c * xsvDot + c * svNormSq;
    float num_coef_sv = 1.0f - c * xNormSq;
    float denom = 1.0f + 2.0f * c * xsvDot + c * c * xNormSq * svNormSq;
    if (fabsf(denom) < 1e-10f) denom = 1e-10f;

    for (int i = 0; i < dim && i < 32; i++) {
        out[i] = (num_coef_x * x[i] + num_coef_sv * scaledV[i]) / denom;
    }
}

// Poincare distance
extern ""C"" __global__ void poincare_distance(
    const float* x, const float* y, float* output,
    int batchSize, int dim, float curvature)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batchSize) return;

    const float* xi = x + idx * dim;
    const float* yi = y + idx * dim;

    float c = curvature;
    float sqrtC = sqrtf(c);

    float negX[32];
    float xNormSq = 0.0f;
    for (int i = 0; i < dim && i < 32; i++) {
        negX[i] = -xi[i];
        xNormSq += xi[i] * xi[i];
    }

    float yNormSq = 0.0f, negXyDot = 0.0f;
    for (int i = 0; i < dim && i < 32; i++) {
        yNormSq += yi[i] * yi[i];
        negXyDot += negX[i] * yi[i];
    }

    float num_coef_negX = 1.0f + 2.0f * c * negXyDot + c * yNormSq;
    float num_coef_y = 1.0f - c * xNormSq;
    float denom = 1.0f + 2.0f * c * negXyDot + c * c * xNormSq * yNormSq;
    if (fabsf(denom) < 1e-10f) denom = 1e-10f;

    float diffNormSq = 0.0f;
    for (int i = 0; i < dim && i < 32; i++) {
        float diff = (num_coef_negX * negX[i] + num_coef_y * yi[i]) / denom;
        diffNormSq += diff * diff;
    }

    float diffNorm = sqrtf(diffNormSq);
    float arg = sqrtC * diffNorm;
    if (arg >= 1.0f) arg = 1.0f - 1e-6f;

    output[idx] = (2.0f / sqrtC) * atanhf(arg);
}

// Hyperbolic linear forward
extern ""C"" __global__ void hyperbolic_linear_forward(
    const float* input, const float* weights, const float* biases, float* output,
    int batchSize, int inputFeatures, int outputFeatures, float curvature, float epsilon)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int b = tid / outputFeatures;
    int o = tid % outputFeatures;

    if (b >= batchSize || o >= outputFeatures) return;

    float c = curvature;
    float sqrtC = sqrtf(c);

    const float* xi = input + b * inputFeatures;
    const float* wo = weights + o * inputFeatures;
    const float* bo = biases + o * inputFeatures;

    float projInput[32];
    float inputNormSq = 0.0f;
    for (int i = 0; i < inputFeatures && i < 32; i++) {
        inputNormSq += xi[i] * xi[i];
    }
    float maxNorm = 1.0f / sqrtC - epsilon;
    float maxNormSq = maxNorm * maxNorm;
    float inputScale = (inputNormSq >= maxNormSq) ? (maxNorm / sqrtf(inputNormSq)) : 1.0f;
    for (int i = 0; i < inputFeatures && i < 32; i++) {
        projInput[i] = xi[i] * inputScale;
    }

    float weightNorm = 0.0f;
    for (int i = 0; i < inputFeatures && i < 32; i++) {
        weightNorm += wo[i] * wo[i];
    }
    weightNorm = sqrtf(weightNorm);

    float weightPoint[32];
    if (weightNorm > 1e-10f) {
        float scale = tanhf(sqrtC * weightNorm / 2.0f) / (sqrtC * weightNorm);
        for (int i = 0; i < inputFeatures && i < 32; i++) {
            weightPoint[i] = wo[i] * scale;
        }
    } else {
        for (int i = 0; i < inputFeatures && i < 32; i++) {
            weightPoint[i] = 0.0f;
        }
    }

    float wpNormSq = 0.0f, piWpDot = 0.0f, piNormSq = 0.0f;
    for (int i = 0; i < inputFeatures && i < 32; i++) {
        wpNormSq += weightPoint[i] * weightPoint[i];
        piNormSq += projInput[i] * projInput[i];
        piWpDot += projInput[i] * weightPoint[i];
    }

    float transformed[32];
    float num1 = 1.0f + 2.0f * c * piWpDot + c * wpNormSq;
    float num2 = 1.0f - c * piNormSq;
    float den = 1.0f + 2.0f * c * piWpDot + c * c * piNormSq * wpNormSq;
    if (fabsf(den) < 1e-10f) den = 1e-10f;
    for (int i = 0; i < inputFeatures && i < 32; i++) {
        transformed[i] = (num1 * projInput[i] + num2 * weightPoint[i]) / den;
    }

    float projBias[32];
    float biasNormSq = 0.0f;
    for (int i = 0; i < inputFeatures && i < 32; i++) {
        biasNormSq += bo[i] * bo[i];
    }
    float biasScale = (biasNormSq >= maxNormSq) ? (maxNorm / sqrtf(biasNormSq)) : 1.0f;
    for (int i = 0; i < inputFeatures && i < 32; i++) {
        projBias[i] = bo[i] * biasScale;
    }

    float pbNormSq = 0.0f, trPbDot = 0.0f, trNormSq = 0.0f;
    for (int i = 0; i < inputFeatures && i < 32; i++) {
        pbNormSq += projBias[i] * projBias[i];
        trNormSq += transformed[i] * transformed[i];
        trPbDot += transformed[i] * projBias[i];
    }

    float withBias[32];
    num1 = 1.0f + 2.0f * c * trPbDot + c * pbNormSq;
    num2 = 1.0f - c * trNormSq;
    den = 1.0f + 2.0f * c * trPbDot + c * c * trNormSq * pbNormSq;
    if (fabsf(den) < 1e-10f) den = 1e-10f;
    for (int i = 0; i < inputFeatures && i < 32; i++) {
        withBias[i] = (num1 * transformed[i] + num2 * projBias[i]) / den;
    }

    float wbNormSq = 0.0f;
    for (int i = 0; i < inputFeatures && i < 32; i++) {
        wbNormSq += withBias[i] * withBias[i];
    }
    float wbNorm = sqrtf(wbNormSq);
    float arg = sqrtC * wbNorm;
    if (arg >= 1.0f) arg = 1.0f - 1e-6f;

    output[b * outputFeatures + o] = (2.0f / sqrtC) * atanhf(arg);
}

// ============================================================================
// OCTONION ALGEBRA KERNELS
// ============================================================================

extern ""C"" __global__ void octonion_multiply(
    const float* a, const float* b, float* output, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const float* ai = a + idx * 8;
    const float* bi = b + idx * 8;
    float* out = output + idx * 8;

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

extern ""C"" __global__ void octonion_add(
    const float* a, const float* b, float* output, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count * 8) return;
    output[idx] = a[idx] + b[idx];
}

extern ""C"" __global__ void octonion_linear_forward(
    const float* input, const float* weights, const float* biases, float* output,
    int batchSize, int inputFeatures, int outputFeatures)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int b = tid / outputFeatures;
    int o = tid % outputFeatures;

    if (b >= batchSize || o >= outputFeatures) return;

    float acc[8];
    const float* bias = biases + o * 8;
    for (int k = 0; k < 8; k++) {
        acc[k] = bias[k];
    }

    for (int i = 0; i < inputFeatures; i++) {
        const float* inp = input + b * inputFeatures * 8 + i * 8;
        const float* w = weights + (o * inputFeatures + i) * 8;

        float a0 = w[0], a1 = w[1], a2 = w[2], a3 = w[3];
        float a4 = w[4], a5 = w[5], a6 = w[6], a7 = w[7];
        float b0 = inp[0], b1 = inp[1], b2 = inp[2], b3 = inp[3];
        float b4 = inp[4], b5 = inp[5], b6 = inp[6], b7 = inp[7];

        acc[0] += a0*b0 - a1*b1 - a2*b2 - a3*b3 - a4*b4 - a5*b5 - a6*b6 - a7*b7;
        acc[1] += a0*b1 + a1*b0 + a2*b3 - a3*b2 + a4*b5 - a5*b4 - a6*b7 + a7*b6;
        acc[2] += a0*b2 - a1*b3 + a2*b0 + a3*b1 + a4*b6 + a5*b7 - a6*b4 - a7*b5;
        acc[3] += a0*b3 + a1*b2 - a2*b1 + a3*b0 + a4*b7 - a5*b6 + a6*b5 - a7*b4;
        acc[4] += a0*b4 - a1*b5 - a2*b6 - a3*b7 + a4*b0 + a5*b1 + a6*b2 + a7*b3;
        acc[5] += a0*b5 + a1*b4 - a2*b7 + a3*b6 - a4*b1 + a5*b0 - a6*b3 + a7*b2;
        acc[6] += a0*b6 + a1*b7 + a2*b4 - a3*b5 - a4*b2 + a5*b3 + a6*b0 - a7*b1;
        acc[7] += a0*b7 - a1*b6 + a2*b5 + a3*b4 - a4*b3 - a5*b2 + a6*b1 + a7*b0;
    }

    float* outPtr = output + b * outputFeatures * 8 + o * 8;
    for (int k = 0; k < 8; k++) {
        outPtr[k] = acc[k];
    }
}

// ============================================================================
// QUANTUM COMPUTING KERNELS
// ============================================================================

extern ""C"" __global__ void quantum_measurement(
    const float* realPart, const float* imagPart, float* probabilities,
    int batchSize, int stateSize)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int b = tid / stateSize;
    int i = tid % stateSize;

    if (b >= batchSize) return;

    int idx = b * stateSize + i;
    float real = realPart[idx];
    float imag = imagPart[idx];
    probabilities[idx] = real * real + imag * imag;
}

extern ""C"" __global__ void normalize_probabilities(
    float* probabilities, int batchSize, int stateSize)
{
    extern __shared__ float sdata[];

    int b = blockIdx.x;
    if (b >= batchSize) return;

    int tid = threadIdx.x;
    float* probs = probabilities + b * stateSize;

    float sum = 0.0f;
    for (int i = tid; i < stateSize; i += blockDim.x) {
        sum += probs[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    float totalSum = sdata[0];
    if (totalSum < 1e-10f) totalSum = 1e-10f;

    for (int i = tid; i < stateSize; i += blockDim.x) {
        probs[i] /= totalSum;
    }
}

extern ""C"" __global__ void complex_matvec(
    const float* matReal, const float* matImag,
    const float* vecReal, const float* vecImag,
    float* outReal, float* outImag,
    int batchSize, int dim)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int b = tid / dim;
    int row = tid % dim;

    if (b >= batchSize) return;

    const float* vr = vecReal + b * dim;
    const float* vi = vecImag + b * dim;

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

extern ""C"" __global__ void quantum_rotation(
    const float* stateReal, const float* stateImag,
    float* outReal, float* outImag,
    const float* angles, int numQubits, int batchSize)
{
    int b = blockIdx.x;
    if (b >= batchSize) return;

    int dim = 1 << numQubits;
    int tid = threadIdx.x;

    const float* sr = stateReal + b * dim;
    const float* si = stateImag + b * dim;
    float* or_ = outReal + b * dim;
    float* oi = outImag + b * dim;

    for (int i = tid; i < dim; i += blockDim.x) {
        or_[i] = sr[i];
        oi[i] = si[i];
    }
    __syncthreads();

    for (int q = 0; q < numQubits; q++) {
        float theta = angles[q];
        float cosHalf = cosf(theta / 2.0f);
        float sinHalf = sinf(theta / 2.0f);

        int stride = 1 << q;

        for (int i = tid; i < dim / 2; i += blockDim.x) {
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
        __syncthreads();
    }
}

extern ""C"" __global__ void measurement_forward(
    const float* input, float* output, int batchSize, int stateSize)
{
    extern __shared__ float sdata[];

    int b = blockIdx.x;
    if (b >= batchSize) return;

    int tid = threadIdx.x;
    const float* inp = input + b * stateSize * 2;
    float* out = output + b * stateSize;

    float localSum = 0.0f;
    for (int i = tid; i < stateSize; i += blockDim.x) {
        float real = inp[i * 2];
        float imag = inp[i * 2 + 1];
        float mag = real * real + imag * imag;
        out[i] = mag;
        localSum += mag;
    }

    sdata[tid] = localSum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    float totalSum = sdata[0];
    if (totalSum < 1e-10f) totalSum = 1e-10f;

    for (int i = tid; i < stateSize; i += blockDim.x) {
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
