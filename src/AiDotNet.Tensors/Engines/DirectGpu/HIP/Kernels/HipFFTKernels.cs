// Copyright (c) AiDotNet. All rights reserved.
// HIP FFT kernels for Fast Fourier Transform operations.
// HIP is source-compatible with CUDA device code.

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

internal static class HipFFTKernels
{
    public static string[] GetKernelNames() => new[]
    {
        "bit_reverse_permutation",
        "fft_butterfly",
        "rfft_postprocess",
        "irfft_preprocess",
        "complex_magnitude",
        "complex_phase",
        "polar_to_complex",
        "apply_mel_filterbank",
        "power_to_db",
        "db_to_power",
        "overlap_add",
        "window_sum_squares",
        "apply_window",
        "fft_rows_butterfly",
        "fft_cols_butterfly",
        "scale_inverse",
        "batched_bit_reverse",
        "batched_fft_butterfly"
    };

    public static string GetSource()
    {
        // Note: hiprtc provides device intrinsics built-in, no includes needed
        return @"
// HIP RTC Compatibility - no includes needed, device intrinsics are built-in
#ifndef INFINITY
#define INFINITY __builtin_huge_valf()
#endif

#define PI 3.14159265358979323846f

// Bit reversal permutation for in-place FFT
extern ""C"" __global__ void bit_reverse_permutation(
    float* real, float* imag, int n, int log2n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int reversed = 0;
    int temp = idx;
    for (int i = 0; i < log2n; i++) {
        reversed = (reversed << 1) | (temp & 1);
        temp >>= 1;
    }

    if (reversed > idx) {
        float tempReal = real[idx];
        float tempImag = imag[idx];
        real[idx] = real[reversed];
        imag[idx] = imag[reversed];
        real[reversed] = tempReal;
        imag[reversed] = tempImag;
    }
}

// Butterfly operation for Cooley-Tukey FFT
extern ""C"" __global__ void fft_butterfly(
    float* real, float* imag, int n, int stride, int inverse)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int halfStride = stride / 2;
    int numButterflies = n / stride;
    int butterflyId = idx / halfStride;
    int wingId = idx % halfStride;

    if (butterflyId >= numButterflies) return;

    int topIdx = butterflyId * stride + wingId;
    int botIdx = topIdx + halfStride;

    // Twiddle factor
    float angle = (inverse ? 2.0f : -2.0f) * PI * wingId / stride;
    float twiddleReal = cosf(angle);
    float twiddleImag = sinf(angle);

    // Load values
    float topReal = real[topIdx];
    float topImag = imag[topIdx];
    float botReal = real[botIdx];
    float botImag = imag[botIdx];

    // Multiply bottom by twiddle factor
    float twiddledReal = botReal * twiddleReal - botImag * twiddleImag;
    float twiddledImag = botReal * twiddleImag + botImag * twiddleReal;

    // Butterfly
    real[topIdx] = topReal + twiddledReal;
    imag[topIdx] = topImag + twiddledImag;
    real[botIdx] = topReal - twiddledReal;
    imag[botIdx] = topImag - twiddledImag;
}

// Post-process RFFT output to extract positive frequencies
extern ""C"" __global__ void rfft_postprocess(
    const float* fullReal, const float* fullImag,
    float* outReal, float* outImag, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int outLen = n / 2 + 1;
    if (idx >= outLen) return;

    outReal[idx] = fullReal[idx];
    outImag[idx] = fullImag[idx];
}

// Pre-process for IRFFT (reconstruct negative frequencies)
extern ""C"" __global__ void irfft_preprocess(
    const float* inReal, const float* inImag,
    float* fullReal, float* fullImag, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int inLen = n / 2 + 1;

    if (idx < inLen) {
        fullReal[idx] = inReal[idx];
        fullImag[idx] = inImag[idx];
    } else if (idx < n) {
        // Conjugate symmetry for negative frequencies
        int mirrorIdx = n - idx;
        fullReal[idx] = inReal[mirrorIdx];
        fullImag[idx] = -inImag[mirrorIdx];
    }
}

// Compute magnitude from complex numbers
extern ""C"" __global__ void complex_magnitude(
    const float* real, const float* imag, float* magnitude, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float r = real[idx];
    float i = imag[idx];
    magnitude[idx] = sqrtf(r * r + i * i);
}

// Compute phase from complex numbers
extern ""C"" __global__ void complex_phase(
    const float* real, const float* imag, float* phase, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    phase[idx] = atan2f(imag[idx], real[idx]);
}

// Convert polar to complex
extern ""C"" __global__ void polar_to_complex(
    const float* magnitude, const float* phase,
    float* real, float* imag, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float mag = magnitude[idx];
    float ph = phase[idx];
    real[idx] = mag * cosf(ph);
    imag[idx] = mag * sinf(ph);
}

// Apply Mel filterbank to power spectrum
extern ""C"" __global__ void apply_mel_filterbank(
    const float* powerSpec, const float* filterbank, float* melSpec,
    int numFrames, int numFreqs, int nMels)
{
    int frame = blockIdx.x;
    int mel = blockIdx.y * blockDim.x + threadIdx.x;

    if (frame >= numFrames || mel >= nMels) return;

    float sum = 0.0f;
    for (int f = 0; f < numFreqs; f++) {
        sum += powerSpec[frame * numFreqs + f] * filterbank[mel * numFreqs + f];
    }
    melSpec[frame * nMels + mel] = sum;
}

// Power to dB conversion
extern ""C"" __global__ void power_to_db(
    const float* power, float* db, int n, float refValue, float minDb)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float p = power[idx];
    float refSq = refValue * refValue;
    float dbVal = 10.0f * log10f(fmaxf(p, 1e-10f) / refSq);
    db[idx] = fmaxf(dbVal, minDb);
}

// dB to power conversion
extern ""C"" __global__ void db_to_power(
    const float* db, float* power, int n, float refValue)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float refSq = refValue * refValue;
    power[idx] = powf(10.0f, db[idx] / 10.0f) * refSq;
}

// Overlap-add for ISTFT
extern ""C"" __global__ void overlap_add(
    const float* frames, float* output, const float* window,
    int numFrames, int nFft, int hopLength, int outputLength)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outputLength) return;

    float sum = 0.0f;
    for (int frame = 0; frame < numFrames; frame++) {
        int frameStart = frame * hopLength;
        int localIdx = idx - frameStart;
        if (localIdx >= 0 && localIdx < nFft) {
            float w = window[localIdx];
            sum += frames[frame * nFft + localIdx] * w;
        }
    }
    output[idx] = sum;
}

// Window normalization for ISTFT
extern ""C"" __global__ void window_sum_squares(
    float* winSqSum, const float* window, int nFft, int hopLength, int outputLength)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outputLength) return;

    float sum = 0.0f;
    int numFrames = (outputLength - nFft) / hopLength + 1;
    for (int frame = 0; frame < numFrames; frame++) {
        int frameStart = frame * hopLength;
        int localIdx = idx - frameStart;
        if (localIdx >= 0 && localIdx < nFft) {
            float w = window[localIdx];
            sum += w * w;
        }
    }
    winSqSum[idx] = sum;
}

// Apply window function
extern ""C"" __global__ void apply_window(
    const float* input, const float* window, float* output, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx] = input[idx] * window[idx];
}

// 2D FFT row-wise butterfly
extern ""C"" __global__ void fft_rows_butterfly(
    float* real, float* imag, int height, int width, int stride, int inverse)
{
    int row = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int halfStride = stride / 2;
    int numButterflies = width / stride;
    int butterflyId = idx / halfStride;
    int wingId = idx % halfStride;

    if (row >= height || butterflyId >= numButterflies) return;

    int topIdx = row * width + butterflyId * stride + wingId;
    int botIdx = topIdx + halfStride;

    float angle = (inverse ? 2.0f : -2.0f) * PI * wingId / stride;
    float twiddleReal = cosf(angle);
    float twiddleImag = sinf(angle);

    float topReal = real[topIdx];
    float topImag = imag[topIdx];
    float botReal = real[botIdx];
    float botImag = imag[botIdx];

    float twiddledReal = botReal * twiddleReal - botImag * twiddleImag;
    float twiddledImag = botReal * twiddleImag + botImag * twiddleReal;

    real[topIdx] = topReal + twiddledReal;
    imag[topIdx] = topImag + twiddledImag;
    real[botIdx] = topReal - twiddledReal;
    imag[botIdx] = topImag - twiddledImag;
}

// 2D FFT column-wise butterfly
extern ""C"" __global__ void fft_cols_butterfly(
    float* real, float* imag, int height, int width, int stride, int inverse)
{
    int col = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int halfStride = stride / 2;
    int numButterflies = height / stride;
    int butterflyId = idx / halfStride;
    int wingId = idx % halfStride;

    if (col >= width || butterflyId >= numButterflies) return;

    int topIdx = (butterflyId * stride + wingId) * width + col;
    int botIdx = topIdx + halfStride * width;

    float angle = (inverse ? 2.0f : -2.0f) * PI * wingId / stride;
    float twiddleReal = cosf(angle);
    float twiddleImag = sinf(angle);

    float topReal = real[topIdx];
    float topImag = imag[topIdx];
    float botReal = real[botIdx];
    float botImag = imag[botIdx];

    float twiddledReal = botReal * twiddleReal - botImag * twiddleImag;
    float twiddledImag = botReal * twiddleImag + botImag * twiddleReal;

    real[topIdx] = topReal + twiddledReal;
    imag[topIdx] = topImag + twiddledImag;
    real[botIdx] = topReal - twiddledReal;
    imag[botIdx] = topImag - twiddledImag;
}

// Scale by 1/N for inverse FFT
extern ""C"" __global__ void scale_inverse(float* real, float* imag, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float scale = 1.0f / n;
    real[idx] *= scale;
    imag[idx] *= scale;
}

// Batched FFT bit reversal
extern ""C"" __global__ void batched_bit_reverse(
    float* real, float* imag, int batch, int n, int log2n)
{
    int b = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch || idx >= n) return;

    int reversed = 0;
    int temp = idx;
    for (int i = 0; i < log2n; i++) {
        reversed = (reversed << 1) | (temp & 1);
        temp >>= 1;
    }

    if (reversed > idx) {
        int baseOffset = b * n;
        float tempReal = real[baseOffset + idx];
        float tempImag = imag[baseOffset + idx];
        real[baseOffset + idx] = real[baseOffset + reversed];
        imag[baseOffset + idx] = imag[baseOffset + reversed];
        real[baseOffset + reversed] = tempReal;
        imag[baseOffset + reversed] = tempImag;
    }
}

// Batched FFT butterfly
extern ""C"" __global__ void batched_fft_butterfly(
    float* real, float* imag, int batch, int n, int stride, int inverse)
{
    int b = blockIdx.z;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int halfStride = stride / 2;
    int numButterflies = n / stride;
    int butterflyId = idx / halfStride;
    int wingId = idx % halfStride;

    if (b >= batch || butterflyId >= numButterflies) return;

    int baseOffset = b * n;
    int topIdx = baseOffset + butterflyId * stride + wingId;
    int botIdx = topIdx + halfStride;

    float angle = (inverse ? 2.0f : -2.0f) * PI * wingId / stride;
    float twiddleReal = cosf(angle);
    float twiddleImag = sinf(angle);

    float topReal = real[topIdx];
    float topImag = imag[topIdx];
    float botReal = real[botIdx];
    float botImag = imag[botIdx];

    float twiddledReal = botReal * twiddleReal - botImag * twiddleImag;
    float twiddledImag = botReal * twiddleImag + botImag * twiddleReal;

    real[topIdx] = topReal + twiddledReal;
    imag[topIdx] = topImag + twiddledImag;
    real[botIdx] = topReal - twiddledReal;
    imag[botIdx] = topImag - twiddledImag;
}
";
    }
}
