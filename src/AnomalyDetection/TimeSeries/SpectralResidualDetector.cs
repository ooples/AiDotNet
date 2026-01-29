using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.TimeSeries;

/// <summary>
/// Detects anomalies in time series using Spectral Residual method.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Spectral Residual is inspired by visual saliency detection in images.
/// It transforms the time series to the frequency domain, finds what's "unusual" in the
/// spectrum (the saliency), and transforms back to identify anomalies.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Apply FFT to convert time series to frequency domain
/// 2. Compute the log amplitude spectrum
/// 3. Extract spectral residual by subtracting smoothed spectrum
/// 4. Transform back to get saliency map (anomaly scores)
/// </para>
/// <para>
/// <b>When to use:</b>
/// - Time series with recurring patterns at multiple frequencies
/// - When anomalies disrupt the normal frequency pattern
/// - Large-scale time series (efficient O(n log n) complexity)
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Window size: 3 (for spectrum smoothing)
/// - Score threshold: Based on contamination
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Microsoft's SR-CNN: Spectral Residual based anomaly detection.
/// Originally from Hou, X., Zhang, L. (2007). "Saliency Detection: A Spectral Residual Approach."
/// </para>
/// </remarks>
public class SpectralResidualDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _windowSize;
    private double[]? _meanAmplitudes;
    private double[]? _stdAmplitudes;

    /// <summary>
    /// Gets the window size for spectrum smoothing.
    /// </summary>
    public int WindowSize => _windowSize;

    /// <summary>
    /// Creates a new Spectral Residual anomaly detector.
    /// </summary>
    /// <param name="windowSize">
    /// Window size for smoothing the log amplitude spectrum. Default is 3.
    /// </param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public SpectralResidualDetector(int windowSize = 3, double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (windowSize < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(windowSize),
                "WindowSize must be at least 1. Recommended is 3.");
        }

        _windowSize = windowSize;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        if (X.Columns != 1)
        {
            throw new ArgumentException(
                "Spectral Residual expects univariate time series (1 column).",
                nameof(X));
        }

        int n = X.Rows;

        // Extract values
        var values = new double[n];
        for (int i = 0; i < n; i++)
        {
            values[i] = NumOps.ToDouble(X[i, 0]);
        }

        // Compute baseline statistics from training data
        var saliency = ComputeSaliencyMap(values);
        _meanAmplitudes = new[] { saliency.Average() };
        _stdAmplitudes = new[] { Math.Sqrt(saliency.Select(s => Math.Pow(s - _meanAmplitudes[0], 2)).Average()) };

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    /// <inheritdoc/>
    public override Vector<T> ScoreAnomalies(Matrix<T> X)
    {
        EnsureFitted();
        return ScoreAnomaliesInternal(X);
    }

    private Vector<T> ScoreAnomaliesInternal(Matrix<T> X)
    {
        ValidateInput(X);

        if (X.Columns != 1)
        {
            throw new ArgumentException(
                "Spectral Residual expects univariate time series (1 column).",
                nameof(X));
        }

        int n = X.Rows;
        var values = new double[n];
        for (int i = 0; i < n; i++)
        {
            values[i] = NumOps.ToDouble(X[i, 0]);
        }

        // Compute saliency map
        var saliency = ComputeSaliencyMap(values);

        // Convert to scores
        var scores = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            scores[i] = NumOps.FromDouble(saliency[i]);
        }

        return scores;
    }

    private double[] ComputeSaliencyMap(double[] values)
    {
        int n = values.Length;

        // Ensure length is power of 2 for FFT (pad if necessary)
        int fftSize = 1;
        while (fftSize < n) fftSize *= 2;

        var paddedValues = new double[fftSize];
        Array.Copy(values, paddedValues, n);

        // Compute FFT
        var (real, imag) = ComputeFFT(paddedValues);

        // Compute log amplitude spectrum
        var logAmplitude = new double[fftSize];
        for (int i = 0; i < fftSize; i++)
        {
            double amplitude = Math.Sqrt(real[i] * real[i] + imag[i] * imag[i]);
            logAmplitude[i] = Math.Log(amplitude + 1e-10);
        }

        // Compute spectral residual (subtract averaged spectrum)
        var avgLogAmplitude = SmoothArray(logAmplitude, _windowSize);
        var spectralResidual = new double[fftSize];
        for (int i = 0; i < fftSize; i++)
        {
            spectralResidual[i] = logAmplitude[i] - avgLogAmplitude[i];
        }

        // Reconstruct with spectral residual amplitude
        var newReal = new double[fftSize];
        var newImag = new double[fftSize];
        for (int i = 0; i < fftSize; i++)
        {
            double amplitude = Math.Sqrt(real[i] * real[i] + imag[i] * imag[i]);
            double phase = Math.Atan2(imag[i], real[i]);
            double newAmplitude = Math.Exp(spectralResidual[i]);

            newReal[i] = newAmplitude * Math.Cos(phase);
            newImag[i] = newAmplitude * Math.Sin(phase);
        }

        // Inverse FFT
        var (saliencyReal, _) = ComputeIFFT(newReal, newImag);

        // Take squared magnitude and trim to original size
        var saliency = new double[n];
        for (int i = 0; i < n; i++)
        {
            saliency[i] = saliencyReal[i] * saliencyReal[i];
        }

        return saliency;
    }

    private (double[] real, double[] imag) ComputeFFT(double[] values)
    {
        int n = values.Length;
        var real = (double[])values.Clone();
        var imag = new double[n];

        // Cooley-Tukey FFT (iterative, radix-2)
        // Bit-reverse permutation
        int bits = (int)Math.Log(n, 2);
        for (int i = 0; i < n; i++)
        {
            int j = ReverseBits(i, bits);
            if (j > i)
            {
                double tempReal = real[i];
                real[i] = real[j];
                real[j] = tempReal;
            }
        }

        // Butterfly computations
        for (int len = 2; len <= n; len *= 2)
        {
            double angle = -2 * Math.PI / len;
            double wReal = Math.Cos(angle);
            double wImag = Math.Sin(angle);

            for (int i = 0; i < n; i += len)
            {
                double curReal = 1, curImag = 0;
                for (int j = 0; j < len / 2; j++)
                {
                    int u = i + j;
                    int v = i + j + len / 2;

                    double tReal = curReal * real[v] - curImag * imag[v];
                    double tImag = curReal * imag[v] + curImag * real[v];

                    real[v] = real[u] - tReal;
                    imag[v] = imag[u] - tImag;
                    real[u] = real[u] + tReal;
                    imag[u] = imag[u] + tImag;

                    double nextReal = curReal * wReal - curImag * wImag;
                    double nextImag = curReal * wImag + curImag * wReal;
                    curReal = nextReal;
                    curImag = nextImag;
                }
            }
        }

        return (real, imag);
    }

    private (double[] real, double[] imag) ComputeIFFT(double[] realInput, double[] imagInput)
    {
        int n = realInput.Length;

        // Standard IFFT: IFFT(X) = (1/N) * conj(FFT(conj(X)))
        // Step 1: Conjugate the input (negate imaginary part)
        var conjReal = (double[])realInput.Clone();
        var conjImag = imagInput.Select(x => -x).ToArray();

        // Step 2: Combine into a single array for FFT (real part only, with conjugated input)
        // We need to compute FFT of the complex conjugate
        // Use the FFT implementation which expects real input, so we need a different approach:
        // For complex input FFT: FFT(a + bi) = FFT(a) + i*FFT(b)
        // But our FFT takes real input. Instead, use the property:
        // IFFT(X) = conj(FFT(conj(X))) / N

        // Create combined input by interleaving real/imag into the FFT
        // Alternative: Compute FFT with complex input by modifying the FFT to accept complex

        // Simpler approach: swap real and imag, apply FFT, swap back, scale
        // IFFT via FFT: IFFT(real, imag) = FFT(real, -imag) with result scaled and imag negated

        // Correct IFFT implementation:
        // Apply FFT to (real, -imag), then scale by 1/N and negate the imaginary result
        var (fftReal, fftImag) = ComputeFFTComplex(conjReal, conjImag);

        // Step 3: Conjugate the result and scale by 1/N
        var resultReal = new double[n];
        var resultImag = new double[n];
        for (int i = 0; i < n; i++)
        {
            resultReal[i] = fftReal[i] / n;
            resultImag[i] = -fftImag[i] / n;
        }

        return (resultReal, resultImag);
    }

    private (double[] real, double[] imag) ComputeFFTComplex(double[] realInput, double[] imagInput)
    {
        int n = realInput.Length;
        var real = (double[])realInput.Clone();
        var imag = (double[])imagInput.Clone();

        // Cooley-Tukey FFT (iterative, radix-2) for complex input
        // Bit-reverse permutation
        int bits = (int)Math.Log(n, 2);
        for (int i = 0; i < n; i++)
        {
            int j = ReverseBits(i, bits);
            if (j > i)
            {
                double tempReal = real[i];
                real[i] = real[j];
                real[j] = tempReal;

                double tempImag = imag[i];
                imag[i] = imag[j];
                imag[j] = tempImag;
            }
        }

        // Butterfly computations
        for (int len = 2; len <= n; len *= 2)
        {
            double angle = -2 * Math.PI / len;
            double wReal = Math.Cos(angle);
            double wImag = Math.Sin(angle);

            for (int i = 0; i < n; i += len)
            {
                double curReal = 1, curImag = 0;
                for (int j = 0; j < len / 2; j++)
                {
                    int u = i + j;
                    int v = i + j + len / 2;

                    double tReal = curReal * real[v] - curImag * imag[v];
                    double tImag = curReal * imag[v] + curImag * real[v];

                    real[v] = real[u] - tReal;
                    imag[v] = imag[u] - tImag;
                    real[u] = real[u] + tReal;
                    imag[u] = imag[u] + tImag;

                    double nextReal = curReal * wReal - curImag * wImag;
                    double nextImag = curReal * wImag + curImag * wReal;
                    curReal = nextReal;
                    curImag = nextImag;
                }
            }
        }

        return (real, imag);
    }

    private static int ReverseBits(int num, int bits)
    {
        int result = 0;
        for (int i = 0; i < bits; i++)
        {
            result = (result << 1) | (num & 1);
            num >>= 1;
        }
        return result;
    }

    private double[] SmoothArray(double[] values, int windowSize)
    {
        int n = values.Length;
        var smoothed = new double[n];
        int halfWindow = windowSize / 2;

        for (int i = 0; i < n; i++)
        {
            int start = Math.Max(0, i - halfWindow);
            int end = Math.Min(n, i + halfWindow + 1);
            double sum = 0;
            for (int j = start; j < end; j++)
            {
                sum += values[j];
            }
            smoothed[i] = sum / (end - start);
        }

        return smoothed;
    }
}
