// OpenCL kernels for FFT and signal processing operations.
// Implements Cooley-Tukey radix-2 FFT algorithm for GPU execution.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;

/// <summary>
/// OpenCL kernel source for FFT operations.
/// Includes 1D FFT, 2D FFT, RFFT/IRFFT, and signal processing utilities.
/// </summary>
public static class FFTKernels
{
    /// <summary>
    /// Gets the OpenCL kernel source code for FFT operations.
    /// </summary>
    public static string GetSource() => @"
// Bit-reversal permutation for FFT
__kernel void bit_reverse_permutation(
    __global float* real,
    __global float* imag,
    const int n,
    const int log2n)
{
    int i = get_global_id(0);
    if (i >= n) return;

    // Compute bit-reversed index
    int j = 0;
    int temp = i;
    for (int k = 0; k < log2n; k++) {
        j = (j << 1) | (temp & 1);
        temp >>= 1;
    }

    // Only swap if i < j to avoid double-swapping
    if (i < j) {
        float tmpReal = real[i];
        float tmpImag = imag[i];
        real[i] = real[j];
        imag[i] = imag[j];
        real[j] = tmpReal;
        imag[j] = tmpImag;
    }
}

// Batched bit-reversal permutation
__kernel void batched_bit_reverse(
    __global float* real,
    __global float* imag,
    const int batch,
    const int n,
    const int log2n)
{
    int idx = get_global_id(0);
    int b = idx / n;
    int i = idx % n;

    if (b >= batch || i >= n) return;

    int j = 0;
    int temp = i;
    for (int k = 0; k < log2n; k++) {
        j = (j << 1) | (temp & 1);
        temp >>= 1;
    }

    if (i < j) {
        int baseIdx = b * n;
        float tmpReal = real[baseIdx + i];
        float tmpImag = imag[baseIdx + i];
        real[baseIdx + i] = real[baseIdx + j];
        imag[baseIdx + i] = imag[baseIdx + j];
        real[baseIdx + j] = tmpReal;
        imag[baseIdx + j] = tmpImag;
    }
}

// Cooley-Tukey FFT butterfly operation for a single stage
__kernel void fft_butterfly(
    __global float* real,
    __global float* imag,
    const int n,
    const int stage,
    const int inverse)
{
    int tid = get_global_id(0);
    int halfSize = 1 << stage;
    int fullSize = halfSize << 1;
    int numGroups = n / fullSize;

    int groupIdx = tid / halfSize;
    int pairIdx = tid % halfSize;

    if (groupIdx >= numGroups) return;

    int i = groupIdx * fullSize + pairIdx;
    int j = i + halfSize;

    // Twiddle factor: W_n^k = exp(-2*pi*i*k/fullSize) for forward, positive for inverse
    float angle = (inverse ? 1.0f : -1.0f) * 2.0f * 3.14159265358979323846f * pairIdx / fullSize;
    float wReal = cos(angle);
    float wImag = sin(angle);

    // t = W * x[j]
    float tReal = wReal * real[j] - wImag * imag[j];
    float tImag = wReal * imag[j] + wImag * real[j];

    // Butterfly: x[i] = x[i] + t, x[j] = x[i] - t
    float uReal = real[i];
    float uImag = imag[i];

    real[i] = uReal + tReal;
    imag[i] = uImag + tImag;
    real[j] = uReal - tReal;
    imag[j] = uImag - tImag;
}

// Batched FFT butterfly for multiple signals
__kernel void batched_fft_butterfly(
    __global float* real,
    __global float* imag,
    const int batch,
    const int n,
    const int stage,
    const int inverse)
{
    int tid = get_global_id(0);
    int totalPairs = batch * (n / 2);

    if (tid >= totalPairs) return;

    int batchIdx = tid / (n / 2);
    int localTid = tid % (n / 2);

    int halfSize = 1 << stage;
    int fullSize = halfSize << 1;
    int numGroups = n / fullSize;

    int groupIdx = localTid / halfSize;
    int pairIdx = localTid % halfSize;

    if (groupIdx >= numGroups) return;

    int baseIdx = batchIdx * n;
    int i = baseIdx + groupIdx * fullSize + pairIdx;
    int j = i + halfSize;

    float angle = (inverse ? 1.0f : -1.0f) * 2.0f * 3.14159265358979323846f * pairIdx / fullSize;
    float wReal = cos(angle);
    float wImag = sin(angle);

    float tReal = wReal * real[j] - wImag * imag[j];
    float tImag = wReal * imag[j] + wImag * real[j];

    float uReal = real[i];
    float uImag = imag[i];

    real[i] = uReal + tReal;
    imag[i] = uImag + tImag;
    real[j] = uReal - tReal;
    imag[j] = uImag - tImag;
}

// Scale by 1/n for inverse FFT normalization
__kernel void fft_scale(
    __global float* real,
    __global float* imag,
    const int n,
    const float scale)
{
    int i = get_global_id(0);
    if (i >= n) return;

    real[i] *= scale;
    imag[i] *= scale;
}

// Real-to-complex FFT post-processing
// Converts N-point real FFT to N/2+1 complex values
__kernel void rfft_postprocess(
    __global const float* fftReal,
    __global const float* fftImag,
    __global float* outReal,
    __global float* outImag,
    const int n)
{
    int k = get_global_id(0);
    int halfN = n / 2;

    if (k > halfN) return;

    if (k == 0) {
        // DC component
        outReal[0] = fftReal[0] + fftImag[0];
        outImag[0] = 0.0f;
    } else if (k == halfN) {
        // Nyquist component
        outReal[halfN] = fftReal[0] - fftImag[0];
        outImag[halfN] = 0.0f;
    } else {
        // General case: use conjugate symmetry
        float evenReal = 0.5f * (fftReal[k] + fftReal[n - k]);
        float evenImag = 0.5f * (fftImag[k] - fftImag[n - k]);
        float oddReal = 0.5f * (fftImag[k] + fftImag[n - k]);
        float oddImag = 0.5f * (fftReal[n - k] - fftReal[k]);

        float angle = -2.0f * 3.14159265358979323846f * k / n;
        float wReal = cos(angle);
        float wImag = sin(angle);

        float twiddledReal = oddReal * wReal - oddImag * wImag;
        float twiddledImag = oddReal * wImag + oddImag * wReal;

        outReal[k] = evenReal + twiddledReal;
        outImag[k] = evenImag + twiddledImag;
    }
}

// Complex-to-real IFFT pre-processing
// Converts N/2+1 complex values to N-point complex for IFFT
__kernel void irfft_preprocess(
    __global const float* inReal,
    __global const float* inImag,
    __global float* fftReal,
    __global float* fftImag,
    const int n)
{
    int k = get_global_id(0);
    int halfN = n / 2;

    if (k >= n) return;

    if (k <= halfN) {
        fftReal[k] = inReal[k];
        fftImag[k] = inImag[k];
    } else {
        // Use conjugate symmetry: X[n-k] = conj(X[k])
        int conj_k = n - k;
        fftReal[k] = inReal[conj_k];
        fftImag[k] = -inImag[conj_k];
    }
}

// Apply window function element-wise
__kernel void apply_window(
    __global const float* input,
    __global const float* window,
    __global float* output,
    const int n)
{
    int i = get_global_id(0);
    if (i >= n) return;

    output[i] = input[i] * window[i];
}

// Batched window application for STFT frames
__kernel void apply_window_batched(
    __global const float* input,
    __global const float* window,
    __global float* output,
    const int numFrames,
    const int windowLen)
{
    int idx = get_global_id(0);
    int frame = idx / windowLen;
    int pos = idx % windowLen;

    if (frame >= numFrames) return;

    output[idx] = input[idx] * window[pos];
}

// Compute magnitude: sqrt(real^2 + imag^2)
__kernel void complex_magnitude(
    __global const float* real,
    __global const float* imag,
    __global float* magnitude,
    const int n)
{
    int i = get_global_id(0);
    if (i >= n) return;

    magnitude[i] = sqrt(real[i] * real[i] + imag[i] * imag[i]);
}

// Compute phase: atan2(imag, real)
__kernel void complex_phase(
    __global const float* real,
    __global const float* imag,
    __global float* phase,
    const int n)
{
    int i = get_global_id(0);
    if (i >= n) return;

    phase[i] = atan2(imag[i], real[i]);
}

// Convert polar to complex: real = mag*cos(phase), imag = mag*sin(phase)
__kernel void polar_to_complex(
    __global const float* magnitude,
    __global const float* phase,
    __global float* real,
    __global float* imag,
    const int n)
{
    int i = get_global_id(0);
    if (i >= n) return;

    float mag = magnitude[i];
    float ph = phase[i];
    real[i] = mag * cos(ph);
    imag[i] = mag * sin(ph);
}

// Apply Mel filterbank: melSpec = powerSpec @ filterbank^T
// powerSpec: [numFrames, numFreqs], filterbank: [nMels, numFreqs], melSpec: [numFrames, nMels]
__kernel void apply_mel_filterbank(
    __global const float* powerSpec,
    __global const float* filterbank,
    __global float* melSpec,
    const int numFrames,
    const int numFreqs,
    const int nMels)
{
    int idx = get_global_id(0);
    int frame = idx / nMels;
    int mel = idx % nMels;

    if (frame >= numFrames) return;

    float sum = 0.0f;
    for (int f = 0; f < numFreqs; f++) {
        sum += powerSpec[frame * numFreqs + f] * filterbank[mel * numFreqs + f];
    }
    melSpec[frame * nMels + mel] = sum;
}

// Power to decibel conversion: db = 10 * log10(max(power, eps)) - refDb
__kernel void power_to_db(
    __global const float* power,
    __global float* db,
    const int n,
    const float refValue,
    const float minDb)
{
    int i = get_global_id(0);
    if (i >= n) return;

    float p = max(power[i], 1e-10f);
    float refDb = 10.0f * log10(refValue);
    float val = 10.0f * log10(p) - refDb;
    db[i] = max(val, minDb);
}

// Decibel to power conversion: power = 10^((db + refDb) / 10)
__kernel void db_to_power(
    __global const float* db,
    __global float* power,
    const int n,
    const float refValue)
{
    int i = get_global_id(0);
    if (i >= n) return;

    float refDb = 10.0f * log10(refValue);
    power[i] = pow(10.0f, (db[i] + refDb) / 10.0f);
}

// Overlap-add for ISTFT reconstruction
__kernel void overlap_add(
    __global const float* frames,
    __global float* output,
    __global float* windowSum,
    const int numFrames,
    const int frameLen,
    const int hopLen,
    const int outputLen)
{
    int outIdx = get_global_id(0);
    if (outIdx >= outputLen) return;

    float sum = 0.0f;
    float wSum = 0.0f;

    // Find which frames contribute to this output sample
    int firstFrame = max(0, (outIdx - frameLen + hopLen) / hopLen);
    int lastFrame = min(numFrames - 1, outIdx / hopLen);

    for (int f = firstFrame; f <= lastFrame; f++) {
        int posInFrame = outIdx - f * hopLen;
        if (posInFrame >= 0 && posInFrame < frameLen) {
            sum += frames[f * frameLen + posInFrame];
            wSum += 1.0f; // Assuming rectangular window sum contribution
        }
    }

    output[outIdx] = sum;
    windowSum[outIdx] = max(wSum, 1e-8f);
}

// Normalize overlap-add output by window sum
__kernel void normalize_overlap_add(
    __global float* output,
    __global const float* windowSum,
    const int n)
{
    int i = get_global_id(0);
    if (i >= n) return;

    output[i] /= windowSum[i];
}

// Create Hann window
__kernel void create_hann_window(
    __global float* window,
    const int length)
{
    int i = get_global_id(0);
    if (i >= length) return;

    window[i] = 0.5f * (1.0f - cos(2.0f * 3.14159265358979323846f * i / (length - 1)));
}

// Create Hamming window
__kernel void create_hamming_window(
    __global float* window,
    const int length)
{
    int i = get_global_id(0);
    if (i >= length) return;

    window[i] = 0.54f - 0.46f * cos(2.0f * 3.14159265358979323846f * i / (length - 1));
}

// 2D FFT: row-wise FFT followed by column-wise FFT
// This kernel handles row-wise processing
__kernel void fft_rows_butterfly(
    __global float* real,
    __global float* imag,
    const int height,
    const int width,
    const int stage,
    const int inverse)
{
    int tid = get_global_id(0);
    int row = tid / (width / 2);
    int localTid = tid % (width / 2);

    if (row >= height) return;

    int halfSize = 1 << stage;
    int fullSize = halfSize << 1;
    int numGroups = width / fullSize;

    int groupIdx = localTid / halfSize;
    int pairIdx = localTid % halfSize;

    if (groupIdx >= numGroups) return;

    int baseIdx = row * width;
    int i = baseIdx + groupIdx * fullSize + pairIdx;
    int j = i + halfSize;

    float angle = (inverse ? 1.0f : -1.0f) * 2.0f * 3.14159265358979323846f * pairIdx / fullSize;
    float wReal = cos(angle);
    float wImag = sin(angle);

    float tReal = wReal * real[j] - wImag * imag[j];
    float tImag = wReal * imag[j] + wImag * real[j];

    float uReal = real[i];
    float uImag = imag[i];

    real[i] = uReal + tReal;
    imag[i] = uImag + tImag;
    real[j] = uReal - tReal;
    imag[j] = uImag - tImag;
}

// 2D FFT: column-wise butterfly
__kernel void fft_cols_butterfly(
    __global float* real,
    __global float* imag,
    const int height,
    const int width,
    const int stage,
    const int inverse)
{
    int tid = get_global_id(0);
    int col = tid / (height / 2);
    int localTid = tid % (height / 2);

    if (col >= width) return;

    int halfSize = 1 << stage;
    int fullSize = halfSize << 1;
    int numGroups = height / fullSize;

    int groupIdx = localTid / halfSize;
    int pairIdx = localTid % halfSize;

    if (groupIdx >= numGroups) return;

    int row_i = groupIdx * fullSize + pairIdx;
    int row_j = row_i + halfSize;
    int i = row_i * width + col;
    int j = row_j * width + col;

    float angle = (inverse ? 1.0f : -1.0f) * 2.0f * 3.14159265358979323846f * pairIdx / fullSize;
    float wReal = cos(angle);
    float wImag = sin(angle);

    float tReal = wReal * real[j] - wImag * imag[j];
    float tImag = wReal * imag[j] + wImag * real[j];

    float uReal = real[i];
    float uImag = imag[i];

    real[i] = uReal + tReal;
    imag[i] = uImag + tImag;
    real[j] = uReal - tReal;
    imag[j] = uImag - tImag;
}

// Row-wise bit reversal for 2D FFT
__kernel void bit_reverse_rows(
    __global float* real,
    __global float* imag,
    const int height,
    const int width,
    const int log2width)
{
    int idx = get_global_id(0);
    int row = idx / width;
    int col = idx % width;

    if (row >= height) return;

    int j = 0;
    int temp = col;
    for (int k = 0; k < log2width; k++) {
        j = (j << 1) | (temp & 1);
        temp >>= 1;
    }

    if (col < j) {
        int baseIdx = row * width;
        float tmpReal = real[baseIdx + col];
        float tmpImag = imag[baseIdx + col];
        real[baseIdx + col] = real[baseIdx + j];
        imag[baseIdx + col] = imag[baseIdx + j];
        real[baseIdx + j] = tmpReal;
        imag[baseIdx + j] = tmpImag;
    }
}

// Column-wise bit reversal for 2D FFT
__kernel void bit_reverse_cols(
    __global float* real,
    __global float* imag,
    const int height,
    const int width,
    const int log2height)
{
    int idx = get_global_id(0);
    int col = idx / height;
    int row = idx % height;

    if (col >= width) return;

    int j = 0;
    int temp = row;
    for (int k = 0; k < log2height; k++) {
        j = (j << 1) | (temp & 1);
        temp >>= 1;
    }

    if (row < j) {
        int i_idx = row * width + col;
        int j_idx = j * width + col;
        float tmpReal = real[i_idx];
        float tmpImag = imag[i_idx];
        real[i_idx] = real[j_idx];
        imag[i_idx] = imag[j_idx];
        real[j_idx] = tmpReal;
        imag[j_idx] = tmpImag;
    }
}
";

    /// <summary>
    /// Gets the kernel names for registration.
    /// </summary>
    public static string[] GetKernelNames() => new[]
    {
        "bit_reverse_permutation",
        "batched_bit_reverse",
        "fft_butterfly",
        "batched_fft_butterfly",
        "fft_scale",
        "rfft_postprocess",
        "irfft_preprocess",
        "apply_window",
        "apply_window_batched",
        "complex_magnitude",
        "complex_phase",
        "polar_to_complex",
        "apply_mel_filterbank",
        "power_to_db",
        "db_to_power",
        "overlap_add",
        "normalize_overlap_add",
        "create_hann_window",
        "create_hamming_window",
        "fft_rows_butterfly",
        "fft_cols_butterfly",
        "bit_reverse_rows",
        "bit_reverse_cols"
    };
}
