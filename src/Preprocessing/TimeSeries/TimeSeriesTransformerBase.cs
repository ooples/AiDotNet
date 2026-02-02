using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;

namespace AiDotNet.Preprocessing.TimeSeries;

/// <summary>
/// Abstract base class for all time series feature extractors providing shared functionality.
/// </summary>
/// <remarks>
/// <para>
/// This class provides common functionality for time series transformers including:
/// - Window validation and edge handling
/// - Parallel processing for large datasets
/// - Auto-detection of optimal window sizes
/// - Feature naming conventions
/// </para>
/// <para><b>For Beginners:</b> This base class handles the common tasks that all time series
/// transformers need, so each specific transformer can focus on its unique calculations.
///
/// Features provided:
/// - NaN padding at the start where we don't have enough history
/// - Fast parallel processing for large datasets
/// - Automatic detection of optimal window sizes
/// - Consistent naming of output features
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public abstract class TimeSeriesTransformerBase<T> : ITimeSeriesFeatureExtractor<T>
{
    #region Fields

    /// <summary>
    /// Gets the numeric operations helper for type T.
    /// </summary>
    protected INumericOperations<T> NumOps { get; }

    /// <summary>
    /// The configuration options for this transformer.
    /// </summary>
    protected readonly TimeSeriesFeatureOptions Options;

    /// <summary>
    /// The computed window sizes after auto-detection (if enabled).
    /// </summary>
    private int[] _windowSizes;

    /// <summary>
    /// The generated feature names.
    /// </summary>
    private string[] _featureNames;

    /// <summary>
    /// The input feature names learned during fitting.
    /// </summary>
    private string[] _inputFeatureNames;

    /// <summary>
    /// The number of input features learned during fitting.
    /// </summary>
    private int _inputFeatureCount;

    /// <summary>
    /// The number of output features that will be generated.
    /// </summary>
    private int _outputFeatureCount;

    /// <summary>
    /// The incremental state for streaming processing.
    /// </summary>
    private IncrementalState<T>? _incrementalState;

    #endregion

    #region IDataTransformer Implementation

    /// <inheritdoc />
    public bool IsFitted { get; protected set; }

    /// <inheritdoc />
    public int[]? ColumnIndices => null;

    /// <inheritdoc />
    public abstract bool SupportsInverseTransform { get; }

    #endregion

    #region ITimeSeriesFeatureExtractor Implementation

    /// <inheritdoc />
    public int[] WindowSizes => _windowSizes;

    /// <inheritdoc />
    public bool AutoDetectEnabled => Options.AutoDetectWindowSizes;

    /// <inheritdoc />
    public string[] FeatureNames => _featureNames;

    /// <inheritdoc />
    public int InputFeatureCount => _inputFeatureCount;

    /// <inheritdoc />
    public int OutputFeatureCount => _outputFeatureCount;

    #endregion

    #region Constructor

    /// <summary>
    /// Creates a new instance of the time series transformer.
    /// </summary>
    /// <param name="options">Configuration options, or null for defaults.</param>
    protected TimeSeriesTransformerBase(TimeSeriesFeatureOptions? options = null)
    {
        NumOps = MathHelper.GetNumericOperations<T>();
        Options = options ?? new TimeSeriesFeatureOptions();

        // Validate options
        var errors = Options.Validate();
        if (errors.Count > 0)
        {
            throw new ArgumentException(
                $"Invalid TimeSeriesFeatureOptions: {string.Join("; ", errors)}");
        }

        _windowSizes = Options.WindowSizes;
        _featureNames = [];
        _inputFeatureNames = [];
        _inputFeatureCount = 0;
        _outputFeatureCount = 0;
        IsFitted = false;
    }

    #endregion

    #region Fit/Transform Methods

    /// <inheritdoc />
    public void Fit(Tensor<T> data)
    {
        ValidateInputForFitting(data);

        // Get input dimensions
        _inputFeatureCount = data.Shape.Length > 1 ? data.Shape[^1] : 1;

        // Validate InputFeatureNames length against detected feature count
        if (Options.InputFeatureNames is { } names && names.Length != _inputFeatureCount)
        {
            throw new ArgumentException(
                $"InputFeatureNames length ({names.Length}) must match detected feature count ({_inputFeatureCount}).");
        }

        // Set up input feature names
        _inputFeatureNames = Options.InputFeatureNames ?? GenerateDefaultInputNames(_inputFeatureCount);

        // Auto-detect window sizes if enabled
        _windowSizes = Options.AutoDetectWindowSizes
            ? DetectOptimalWindowSizes(data)
            : Options.WindowSizes;

        // Generate feature names
        _featureNames = GenerateFeatureNames();
        _outputFeatureCount = _featureNames.Length;

        // Let derived class perform its specific fitting
        FitCore(data);

        IsFitted = true;
    }

    /// <inheritdoc />
    public Tensor<T> Transform(Tensor<T> data)
    {
        EnsureFitted();
        ValidateInputForTransform(data);

        // Use parallel processing if data is large enough
        if (Options.UseParallelProcessing && GetTimeSteps(data) >= Options.ParallelThreshold)
        {
            return TransformParallel(data);
        }

        return TransformCore(data);
    }

    /// <inheritdoc />
    public Tensor<T> FitTransform(Tensor<T> data)
    {
        Fit(data);
        return Transform(data);
    }

    /// <inheritdoc />
    public Tensor<T> InverseTransform(Tensor<T> data)
    {
        if (!SupportsInverseTransform)
        {
            throw new NotSupportedException(
                $"{GetType().Name} does not support inverse transformation.");
        }

        EnsureFitted();
        return InverseTransformCore(data);
    }

    /// <inheritdoc />
    public string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (!IsFitted)
        {
            throw new InvalidOperationException(
                "Transformer must be fitted before calling GetFeatureNamesOut.");
        }

        return _featureNames;
    }

    #endregion

    #region Auto-Detection

    /// <inheritdoc />
    public virtual int[] DetectOptimalWindowSizes(Tensor<T> data)
    {
        return Options.AutoDetectionMethod switch
        {
            WindowAutoDetectionMethod.Autocorrelation => DetectUsingAutocorrelation(data),
            WindowAutoDetectionMethod.SpectralAnalysis => DetectUsingSpectralAnalysis(data),
            WindowAutoDetectionMethod.Heuristic => DetectUsingHeuristics(data),
            WindowAutoDetectionMethod.GridSearch => DetectUsingGridSearch(data),
            _ => DetectUsingAutocorrelation(data)
        };
    }

    /// <summary>
    /// Detects optimal window sizes using autocorrelation function analysis.
    /// </summary>
    /// <param name="data">The time series data to analyze.</param>
    /// <returns>Detected optimal window sizes.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Autocorrelation measures how similar the data is to itself
    /// at different time lags. Peaks in the autocorrelation function indicate seasonal patterns.
    ///
    /// For example, if data has a weekly pattern, there will be a peak at lag 7 (7 days).
    /// This method finds those peaks and uses them as window sizes.
    /// </para>
    /// </remarks>
    protected virtual int[] DetectUsingAutocorrelation(Tensor<T> data)
    {
        int timeSteps = GetTimeSteps(data);
        int maxLag = Math.Min(Options.MaxWindowSize, timeSteps / 2);
        int minLag = Options.MinWindowSize;

        // Compute autocorrelation for the first feature (or average across features)
        var acf = ComputeAutocorrelation(data, maxLag);

        // Find peaks in ACF that are statistically significant
        var peaks = FindAutocorrelationPeaks(acf, minLag, maxLag);

        // If no peaks found, use heuristic defaults
        if (peaks.Count == 0)
        {
            return DetectUsingHeuristics(data);
        }

        // Take the top N peaks as window sizes
        var windowSizes = peaks
            .OrderByDescending(p => Math.Abs(NumOps.ToDouble(p.Value)))
            .Take(Options.MaxAutoDetectedWindows)
            .Select(p => p.Lag)
            .OrderBy(l => l)
            .ToArray();

        return windowSizes;
    }

    /// <summary>
    /// Detects optimal window sizes using spectral analysis (FFT).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method uses the Fast Fourier Transform (FFT) to convert
    /// the time series from the time domain to the frequency domain. Peaks in the frequency
    /// spectrum indicate periodic patterns in the data.
    ///
    /// For example, if stock prices tend to cycle every 20 days, the FFT will show a peak
    /// at the frequency 1/20 = 0.05 cycles per day. We convert this back to a period of 20 days.
    /// </para>
    /// </remarks>
    protected virtual int[] DetectUsingSpectralAnalysis(Tensor<T> data)
    {
        int timeSteps = GetTimeSteps(data);
        int features = _inputFeatureCount > 0 ? _inputFeatureCount : 1;

        // Need at least some data points for FFT
        if (timeSteps < 8)
        {
            return DetectUsingHeuristics(data);
        }

        // Pad to power of 2 for efficient FFT
        int fftSize = NextPowerOfTwo(timeSteps);

        // Aggregate power spectrum across all features
        var avgPowerSpectrum = new double[fftSize / 2];

        for (int f = 0; f < features; f++)
        {
            // Extract time series for this feature
            var series = new double[fftSize];
            for (int t = 0; t < timeSteps; t++)
            {
                series[t] = NumOps.ToDouble(GetValue(data, t, f));
            }

            // Remove mean (detrend)
            double mean = series.Take(timeSteps).Average();
            for (int t = 0; t < timeSteps; t++)
            {
                series[t] -= mean;
            }

            // Zero-pad the rest
            for (int t = timeSteps; t < fftSize; t++)
            {
                series[t] = 0;
            }

            // Apply Hann window to reduce spectral leakage
            ApplyHannWindow(series, timeSteps);

            // Compute FFT
            var (real, imag) = ComputeFFT(series);

            // Compute power spectrum and accumulate
            for (int i = 1; i < fftSize / 2; i++)  // Skip DC component
            {
                double power = real[i] * real[i] + imag[i] * imag[i];
                avgPowerSpectrum[i] += power / features;
            }
        }

        // Find peaks in the power spectrum
        var peaks = FindSpectralPeaks(avgPowerSpectrum, fftSize, timeSteps);

        // Convert frequency indices to periods (window sizes)
        var windowSizes = peaks
            .Where(p => p.Period >= Options.MinWindowSize && p.Period <= Options.MaxWindowSize)
            .OrderByDescending(p => p.Power)
            .Take(Options.MaxAutoDetectedWindows)
            .Select(p => p.Period)
            .Distinct()
            .OrderBy(p => p)
            .ToArray();

        if (windowSizes.Length == 0)
        {
            return DetectUsingHeuristics(data);
        }

        return windowSizes;
    }

    /// <summary>
    /// Applies a Hann window to reduce spectral leakage.
    /// </summary>
    private static void ApplyHannWindow(double[] data, int length)
    {
        for (int i = 0; i < length; i++)
        {
            double window = 0.5 * (1 - Math.Cos(2 * Math.PI * i / (length - 1)));
            data[i] *= window;
        }
    }

    /// <summary>
    /// Computes the FFT using the Cooley-Tukey radix-2 algorithm.
    /// </summary>
    private static (double[] real, double[] imag) ComputeFFT(double[] input)
    {
        int n = input.Length;
        var real = new double[n];
        var imag = new double[n];

        // Copy input to real part
        Array.Copy(input, real, n);

        // Bit-reversal permutation
        int j = 0;
        for (int i = 0; i < n - 1; i++)
        {
            if (i < j)
            {
                (real[i], real[j]) = (real[j], real[i]);
                (imag[i], imag[j]) = (imag[j], imag[i]);
            }
            int k = n / 2;
            while (k <= j)
            {
                j -= k;
                k /= 2;
            }
            j += k;
        }

        // Cooley-Tukey iterative FFT
        for (int len = 2; len <= n; len *= 2)
        {
            double angle = -2 * Math.PI / len;
            double wReal = Math.Cos(angle);
            double wImag = Math.Sin(angle);

            for (int i = 0; i < n; i += len)
            {
                double curReal = 1;
                double curImag = 0;

                for (int jj = 0; jj < len / 2; jj++)
                {
                    int idx1 = i + jj;
                    int idx2 = i + jj + len / 2;

                    double tReal = curReal * real[idx2] - curImag * imag[idx2];
                    double tImag = curReal * imag[idx2] + curImag * real[idx2];

                    real[idx2] = real[idx1] - tReal;
                    imag[idx2] = imag[idx1] - tImag;
                    real[idx1] += tReal;
                    imag[idx1] += tImag;

                    double newReal = curReal * wReal - curImag * wImag;
                    curImag = curReal * wImag + curImag * wReal;
                    curReal = newReal;
                }
            }
        }

        return (real, imag);
    }

    /// <summary>
    /// Finds peaks in the power spectrum.
    /// </summary>
    private List<(int Period, double Power)> FindSpectralPeaks(double[] powerSpectrum, int fftSize, int originalLength)
    {
        var peaks = new List<(int Period, double Power)>();

        // Calculate mean and std of power for significance threshold
        double meanPower = powerSpectrum.Skip(1).Average();
        double stdPower = Math.Sqrt(powerSpectrum.Skip(1).Select(p => (p - meanPower) * (p - meanPower)).Average());
        double threshold = meanPower + 2 * stdPower;  // 2 sigma above mean

        // Find local maxima above threshold
        for (int i = 2; i < powerSpectrum.Length - 1; i++)
        {
            if (powerSpectrum[i] > threshold &&
                powerSpectrum[i] > powerSpectrum[i - 1] &&
                powerSpectrum[i] > powerSpectrum[i + 1])
            {
                // Convert frequency index to period
                // Frequency = i / fftSize, Period = fftSize / i
                int period = (int)Math.Round((double)originalLength / i);

                if (period >= 2 && period < originalLength / 2)
                {
                    peaks.Add((period, powerSpectrum[i]));
                }
            }
        }

        return peaks;
    }

    /// <summary>
    /// Returns the next power of 2 greater than or equal to n.
    /// </summary>
    private static int NextPowerOfTwo(int n)
    {
        int power = 1;
        while (power < n)
            power *= 2;
        return power;
    }

    /// <summary>
    /// Detects optimal window sizes using grid search with cross-validation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method tries many different window sizes and evaluates
    /// which ones work best using time series cross-validation. It's slower but more accurate
    /// because it actually tests how well different windows capture predictive patterns.
    ///
    /// The evaluation metric is the variance explained by the rolling features - higher variance
    /// explained means the window size captures more useful information.
    /// </para>
    /// </remarks>
    protected virtual int[] DetectUsingGridSearch(Tensor<T> data)
    {
        int timeSteps = GetTimeSteps(data);
        int features = _inputFeatureCount > 0 ? _inputFeatureCount : 1;

        // Generate candidate window sizes to test
        var candidateSizes = GenerateCandidateWindowSizes(timeSteps);

        if (candidateSizes.Length == 0)
        {
            return DetectUsingHeuristics(data);
        }

        // Evaluate each candidate using variance explained
        var scores = new List<(int WindowSize, double Score)>();

        foreach (int windowSize in candidateSizes)
        {
            double score = EvaluateWindowSize(data, windowSize, timeSteps, features);
            scores.Add((windowSize, score));
        }

        // Select top-scoring window sizes
        var selectedWindows = scores
            .OrderByDescending(s => s.Score)
            .Take(Options.MaxAutoDetectedWindows)
            .Select(s => s.WindowSize)
            .OrderBy(w => w)
            .ToArray();

        if (selectedWindows.Length == 0)
        {
            return DetectUsingHeuristics(data);
        }

        return selectedWindows;
    }

    /// <summary>
    /// Generates candidate window sizes for grid search.
    /// </summary>
    private int[] GenerateCandidateWindowSizes(int dataLength)
    {
        var candidates = new List<int>();

        // Add common window sizes that fit within constraints
        int[] common = [2, 3, 5, 7, 10, 14, 20, 21, 30, 60, 90, 120, 180, 252, 365];
        candidates.AddRange(common.Where(size =>
            size >= Options.MinWindowSize &&
            size <= Options.MaxWindowSize &&
            size < dataLength / 2));

        // Add log-spaced sizes for coverage
        double logMin = Math.Log(Options.MinWindowSize);
        double logMax = Math.Log(Math.Min(Options.MaxWindowSize, dataLength / 2));
        int numSteps = 10;

        for (int i = 0; i < numSteps; i++)
        {
            double logSize = logMin + (logMax - logMin) * i / (numSteps - 1);
            int size = (int)Math.Round(Math.Exp(logSize));
            if (size >= Options.MinWindowSize && size <= Options.MaxWindowSize && size < dataLength / 2)
            {
                candidates.Add(size);
            }
        }

        return candidates.Distinct().OrderBy(x => x).ToArray();
    }

    /// <summary>
    /// Evaluates how well a window size captures patterns in the data.
    /// </summary>
    private double EvaluateWindowSize(Tensor<T> data, int windowSize, int timeSteps, int features)
    {
        // Use rolling mean to capture patterns
        // Score = (1 - MSE / Variance) = variance explained
        double totalScore = 0;
        int validFeatures = 0;

        for (int f = 0; f < features; f++)
        {
            // Extract time series
            var series = new double[timeSteps];
            for (int t = 0; t < timeSteps; t++)
            {
                series[t] = NumOps.ToDouble(GetValue(data, t, f));
            }

            // Compute rolling mean
            var rollingMean = new double[timeSteps];
            for (int t = windowSize - 1; t < timeSteps; t++)
            {
                double sum = 0;
                for (int k = 0; k < windowSize; k++)
                {
                    sum += series[t - k];
                }
                rollingMean[t] = sum / windowSize;
            }

            // Compute variance of original series
            double mean = series.Skip(windowSize - 1).Average();
            double variance = series.Skip(windowSize - 1).Select(x => (x - mean) * (x - mean)).Average();

            if (variance < 1e-10) continue;

            // Compute MSE between rolling mean and actual
            double mse = 0;
            int count = 0;
            for (int t = windowSize - 1; t < timeSteps; t++)
            {
                double diff = series[t] - rollingMean[t];
                mse += diff * diff;
                count++;
            }
            mse /= count;

            // Variance explained (higher is better)
            double r2 = 1 - mse / variance;
            totalScore += Math.Max(0, r2);
            validFeatures++;
        }

        return validFeatures > 0 ? totalScore / validFeatures : 0;
    }

    /// <summary>
    /// Detects optimal window sizes using simple heuristic rules.
    /// </summary>
    /// <param name="data">The time series data to analyze.</param>
    /// <returns>Heuristically determined window sizes.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Simple rules based on data length:
    /// - Use small windows (5-10) for short-term patterns
    /// - Use medium windows (20-30) for medium-term patterns
    /// - Use larger windows (sqrt(n)) for longer-term patterns
    /// </para>
    /// </remarks>
    protected virtual int[] DetectUsingHeuristics(Tensor<T> data)
    {
        int n = GetTimeSteps(data);
        var windows = new List<int>();

        // Common window sizes based on data length
        int[] candidates = [5, 7, 10, 14, 20, 30, 60, 90, 120, 252, 365];

        windows.AddRange(candidates
            .Where(w => w >= Options.MinWindowSize && w <= Options.MaxWindowSize && w < n / 2)
            .Take(Options.MaxAutoDetectedWindows));

        // If no candidates fit, use sqrt(n) rule
        if (windows.Count == 0)
        {
            int sqrtN = (int)Math.Sqrt(n);
            sqrtN = Math.Max(sqrtN, Options.MinWindowSize);
            sqrtN = Math.Min(sqrtN, Options.MaxWindowSize);
            windows.Add(sqrtN);
        }

        return [.. windows];
    }

    /// <summary>
    /// Computes the autocorrelation function for the time series.
    /// </summary>
    private T[] ComputeAutocorrelation(Tensor<T> data, int maxLag)
    {
        int timeSteps = GetTimeSteps(data);
        int features = _inputFeatureCount > 0 ? _inputFeatureCount : 1;

        var acf = new T[maxLag + 1];

        // Average ACF across all features
        for (int lag = 0; lag <= maxLag; lag++)
        {
            double sumAcf = 0;

            for (int f = 0; f < features; f++)
            {
                // Extract time series for this feature
                var series = new double[timeSteps];
                for (int t = 0; t < timeSteps; t++)
                {
                    series[t] = NumOps.ToDouble(GetValue(data, t, f));
                }

                // Compute mean
                double mean = series.Average();

                // Compute variance
                double variance = series.Select(x => (x - mean) * (x - mean)).Sum() / timeSteps;

                if (variance > 1e-10)
                {
                    // Compute autocorrelation at this lag
                    double cov = 0;
                    for (int t = 0; t < timeSteps - lag; t++)
                    {
                        cov += (series[t] - mean) * (series[t + lag] - mean);
                    }
                    cov /= (timeSteps - lag);
                    sumAcf += cov / variance;
                }
            }

            acf[lag] = NumOps.FromDouble(sumAcf / features);
        }

        return acf;
    }

    /// <summary>
    /// Finds peaks in the autocorrelation function.
    /// </summary>
    private List<(int Lag, T Value)> FindAutocorrelationPeaks(T[] acf, int minLag, int maxLag)
    {
        var peaks = new List<(int Lag, T Value)>();

        // Significance threshold (approximate 95% confidence interval)
        int n = acf.Length;
        double threshold = 1.96 / Math.Sqrt(n);

        for (int i = Math.Max(minLag, 1); i < Math.Min(acf.Length - 1, maxLag); i++)
        {
            double current = Math.Abs(NumOps.ToDouble(acf[i]));
            double prev = Math.Abs(NumOps.ToDouble(acf[i - 1]));
            double next = Math.Abs(NumOps.ToDouble(acf[i + 1]));

            // Is this a local maximum and significant?
            if (current > prev && current > next && current > threshold)
            {
                peaks.Add((i, acf[i]));
            }
        }

        return peaks;
    }

    #endregion

    #region Validation

    /// <inheritdoc />
    public bool ValidateInput(Tensor<T> data)
    {
        return GetValidationErrors(data).Count == 0;
    }

    /// <inheritdoc />
    public List<string> GetValidationErrors(Tensor<T> data)
    {
        var errors = new List<string>();

        if (data == null)
        {
            errors.Add("Data cannot be null.");
            return errors;
        }

        int timeSteps = GetTimeSteps(data);
        int maxWindow = _windowSizes.Length > 0
            ? _windowSizes.Max()
            : (Options.WindowSizes.Length > 0 ? Options.WindowSizes.Max() : Options.MinWindowSize);

        if (timeSteps < maxWindow)
        {
            errors.Add($"Data length ({timeSteps}) must be at least the maximum window size ({maxWindow}).");
        }

        if (IsFitted)
        {
            int dataFeatures = data.Shape.Length > 1 ? data.Shape[^1] : 1;
            if (dataFeatures != _inputFeatureCount)
            {
                errors.Add($"Data has {dataFeatures} features but transformer was fitted with {_inputFeatureCount} features.");
            }
        }

        return errors;
    }

    /// <summary>
    /// Validates input data for fitting.
    /// </summary>
    protected virtual void ValidateInputForFitting(Tensor<T> data)
    {
        if (data == null)
        {
            throw new ArgumentNullException(nameof(data));
        }

        int timeSteps = GetTimeSteps(data);
        int maxWindow = Options.WindowSizes.Length > 0 ? Options.WindowSizes.Max() : Options.MinWindowSize;

        if (timeSteps < maxWindow)
        {
            throw new ArgumentException(
                $"Data length ({timeSteps}) must be at least the maximum window size ({maxWindow}).");
        }
    }

    /// <summary>
    /// Validates input data for transformation.
    /// </summary>
    protected virtual void ValidateInputForTransform(Tensor<T> data)
    {
        var errors = GetValidationErrors(data);
        if (errors.Count > 0)
        {
            throw new ArgumentException(string.Join("; ", errors));
        }
    }

    /// <summary>
    /// Ensures the transformer has been fitted.
    /// </summary>
    protected void EnsureFitted()
    {
        if (!IsFitted)
        {
            throw new InvalidOperationException(
                "This transformer has not been fitted. Call Fit() or FitTransform() first.");
        }
    }

    #endregion

    #region Abstract Methods

    /// <summary>
    /// Performs the core fitting logic specific to this transformer.
    /// </summary>
    /// <param name="data">The training data to fit.</param>
    protected abstract void FitCore(Tensor<T> data);

    /// <summary>
    /// Performs the core transformation logic specific to this transformer.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The transformed data.</returns>
    protected abstract Tensor<T> TransformCore(Tensor<T> data);

    /// <summary>
    /// Performs the inverse transformation if supported.
    /// </summary>
    /// <param name="data">The transformed data.</param>
    /// <returns>The original data.</returns>
    protected virtual Tensor<T> InverseTransformCore(Tensor<T> data)
    {
        throw new NotSupportedException();
    }

    /// <summary>
    /// Generates the feature names for this transformer's outputs.
    /// </summary>
    /// <returns>Array of feature names.</returns>
    protected abstract string[] GenerateFeatureNames();

    /// <summary>
    /// Gets the list of statistics or operations this transformer computes.
    /// </summary>
    /// <returns>List of operation names for feature naming.</returns>
    protected abstract string[] GetOperationNames();

    #endregion

    #region Parallel Processing

    /// <summary>
    /// Transforms data using parallel processing for improved performance.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The transformed data.</returns>
    protected virtual Tensor<T> TransformParallel(Tensor<T> data)
    {
        // Default implementation delegates to sequential; derived classes can override
        return TransformCore(data);
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Gets the number of time steps in the data.
    /// </summary>
    protected int GetTimeSteps(Tensor<T> data)
    {
        // Assume shape is [time_steps] or [time_steps, features] or [batch, time_steps, features]
        return data.Shape.Length switch
        {
            1 => data.Shape[0],
            2 => data.Shape[0],
            3 => data.Shape[1],
            _ => data.Shape[0]
        };
    }

    /// <summary>
    /// Gets a value from the tensor at the specified time step and feature.
    /// </summary>
    protected T GetValue(Tensor<T> data, int timeStep, int feature)
    {
        return data.Shape.Length switch
        {
            1 => data[timeStep],
            2 => data[timeStep, feature],
            3 => data[0, timeStep, feature], // Assume batch size 1 or first batch
            _ => data[timeStep]
        };
    }

    /// <summary>
    /// Sets a value in the tensor at the specified time step and feature.
    /// </summary>
    protected void SetValue(Tensor<T> data, int timeStep, int feature, T value)
    {
        switch (data.Shape.Length)
        {
            case 1:
                data[timeStep] = value;
                break;
            case 2:
                data[timeStep, feature] = value;
                break;
            case 3:
                data[0, timeStep, feature] = value;
                break;
            default:
                data[timeStep] = value;
                break;
        }
    }

    /// <summary>
    /// Generates default input feature names.
    /// </summary>
    private string[] GenerateDefaultInputNames(int count)
    {
        var names = new string[count];
        for (int i = 0; i < count; i++)
        {
            names[i] = $"feature_{i}";
        }
        return names;
    }

    /// <summary>
    /// Gets the NaN value for type T.
    /// </summary>
    protected T GetNaN()
    {
        return NumOps.FromDouble(double.NaN);
    }

    /// <summary>
    /// Checks if a value is NaN.
    /// </summary>
    protected bool IsNaN(T value)
    {
        return double.IsNaN(NumOps.ToDouble(value));
    }

    /// <summary>
    /// Gets the input feature names (learned during fitting or from options).
    /// </summary>
    protected string[] GetInputFeatureNames()
    {
        return _inputFeatureNames;
    }

    /// <summary>
    /// Gets the feature name separator from options.
    /// </summary>
    protected string GetSeparator()
    {
        return Options.FeatureNameSeparator;
    }

    /// <summary>
    /// Gets the maximum window size from configured windows.
    /// </summary>
    protected int GetMaxWindowSize()
    {
        return _windowSizes.Length > 0 ? _windowSizes.Max() : Options.MinWindowSize;
    }

    /// <summary>
    /// Determines if a time step is in the edge region (incomplete window).
    /// </summary>
    /// <param name="timeStep">The current time step.</param>
    /// <param name="windowSize">The window size being used.</param>
    /// <returns>True if in edge region, false otherwise.</returns>
    protected bool IsEdgeRegion(int timeStep, int windowSize)
    {
        return timeStep < windowSize - 1;
    }

    /// <summary>
    /// Gets the effective window size for partial edge handling.
    /// </summary>
    /// <param name="timeStep">The current time step.</param>
    /// <param name="windowSize">The requested window size.</param>
    /// <returns>The actual number of data points available.</returns>
    protected int GetEffectiveWindowSize(int timeStep, int windowSize)
    {
        return Options.EdgeHandling == EdgeHandling.Partial
            ? Math.Min(windowSize, timeStep + 1)
            : windowSize;
    }

    /// <summary>
    /// Determines the output row count based on edge handling mode.
    /// </summary>
    /// <param name="inputTimeSteps">The number of input time steps.</param>
    /// <returns>The number of output time steps.</returns>
    protected int GetOutputTimeSteps(int inputTimeSteps)
    {
        if (Options.EdgeHandling == EdgeHandling.Truncate)
        {
            int maxWindow = GetMaxWindowSize();
            return Math.Max(0, inputTimeSteps - maxWindow + 1);
        }
        return inputTimeSteps;
    }

    /// <summary>
    /// Gets the starting time step index for output (used with Truncate mode).
    /// </summary>
    /// <returns>The starting index in the input data.</returns>
    protected int GetOutputStartIndex()
    {
        if (Options.EdgeHandling == EdgeHandling.Truncate)
        {
            return GetMaxWindowSize() - 1;
        }
        return 0;
    }

    /// <summary>
    /// Applies forward fill to output tensor for edge values.
    /// </summary>
    /// <param name="output">The output tensor to modify.</param>
    /// <param name="firstValidIndex">The first index with valid (non-edge) data.</param>
    protected void ApplyForwardFill(Tensor<T> output, int firstValidIndex)
    {
        if (Options.EdgeHandling != EdgeHandling.ForwardFill || firstValidIndex <= 0)
            return;

        int features = output.Shape[1];

        // Copy the first valid row to all preceding rows
        for (int t = 0; t < firstValidIndex; t++)
        {
            for (int f = 0; f < features; f++)
            {
                output[t, f] = output[firstValidIndex, f];
            }
        }
    }

    /// <summary>
    /// Checks if the current edge handling mode should compute partial windows.
    /// </summary>
    protected bool ShouldComputePartialWindows()
    {
        return Options.EdgeHandling == EdgeHandling.Partial ||
               Options.EdgeHandling == EdgeHandling.ForwardFill;
    }

    #endregion

    #region Incremental/Streaming Support

    /// <inheritdoc />
    public virtual bool SupportsIncrementalTransform => true;

    /// <inheritdoc />
    public virtual void InitializeIncremental(Tensor<T> historicalData)
    {
        EnsureFitted();

        int timeSteps = GetTimeSteps(historicalData);
        int maxWindow = GetMaxWindowSize();

        if (timeSteps < maxWindow)
        {
            throw new ArgumentException(
                $"Historical data length ({timeSteps}) must be at least the maximum window size ({maxWindow}).");
        }

        // Initialize the rolling buffer
        int bufferSize = maxWindow;
        var buffer = new T[_inputFeatureCount][];

        for (int f = 0; f < _inputFeatureCount; f++)
        {
            buffer[f] = new T[bufferSize];

            // Fill buffer with the last 'bufferSize' values
            int startIdx = timeSteps - bufferSize;
            for (int i = 0; i < bufferSize; i++)
            {
                buffer[f][i] = GetValue(historicalData, startIdx + i, f);
            }
        }

        _incrementalState = new IncrementalState<T>
        {
            RollingBuffer = buffer,
            BufferPosition = 0,
            PointsProcessed = timeSteps,
            BufferFilled = true,
            ExtendedState = InitializeExtendedState(historicalData)
        };
    }

    /// <inheritdoc />
    public virtual T[] TransformIncremental(T[] newDataPoint)
    {
        if (!SupportsIncrementalTransform)
        {
            throw new NotSupportedException(
                $"{GetType().Name} does not support incremental transformation.");
        }

        if (_incrementalState == null)
        {
            throw new InvalidOperationException(
                "Incremental state not initialized. Call InitializeIncremental() first.");
        }

        if (newDataPoint.Length != _inputFeatureCount)
        {
            throw new ArgumentException(
                $"Data point has {newDataPoint.Length} features but transformer was fitted with {_inputFeatureCount} features.");
        }

        // Update rolling buffer
        int maxWindow = GetMaxWindowSize();
        int pos = _incrementalState.BufferPosition;

        for (int f = 0; f < _inputFeatureCount; f++)
        {
            _incrementalState.RollingBuffer[f][pos] = newDataPoint[f];
        }

        // Compute features using the rolling buffer
        var features = ComputeIncrementalFeatures(_incrementalState, newDataPoint);

        // Update state
        _incrementalState.BufferPosition = (pos + 1) % maxWindow;
        _incrementalState.PointsProcessed++;
        if (!_incrementalState.BufferFilled && _incrementalState.PointsProcessed >= maxWindow)
        {
            _incrementalState.BufferFilled = true;
        }

        return features;
    }

    /// <inheritdoc />
    public virtual IncrementalState<T>? GetIncrementalState()
    {
        return _incrementalState;
    }

    /// <inheritdoc />
    public virtual void SetIncrementalState(IncrementalState<T> state)
    {
        if (state == null)
        {
            throw new ArgumentNullException(nameof(state));
        }

        if (state.RollingBuffer.Length != _inputFeatureCount)
        {
            throw new ArgumentException(
                $"State has {state.RollingBuffer.Length} features but transformer was fitted with {_inputFeatureCount} features.");
        }

        _incrementalState = state;
    }

    /// <summary>
    /// Initializes any extended state needed for incremental processing.
    /// Override this in derived classes to add transformer-specific state.
    /// </summary>
    /// <param name="historicalData">The initial historical data.</param>
    /// <returns>Dictionary of extended state values.</returns>
    protected virtual Dictionary<string, object> InitializeExtendedState(Tensor<T> historicalData)
    {
        return [];
    }

    /// <summary>
    /// Computes features for a single data point using the incremental state.
    /// Override this in derived classes for transformer-specific incremental computation.
    /// </summary>
    /// <param name="state">The current incremental state.</param>
    /// <param name="newDataPoint">The new data point.</param>
    /// <returns>Array of computed feature values.</returns>
    protected virtual T[] ComputeIncrementalFeatures(IncrementalState<T> state, T[] newDataPoint)
    {
        // Default implementation: extract window and compute features
        var features = new T[_outputFeatureCount];
        int featureIdx = 0;

        foreach (int windowSize in WindowSizes)
        {
            for (int f = 0; f < _inputFeatureCount; f++)
            {
                var window = ExtractIncrementalWindow(state, f, windowSize);
                int numOpsForFeature = GetOperationNames().Length;

                var computed = ComputeFeaturesForWindow(window, newDataPoint[f]);
                foreach (var value in computed)
                {
                    if (featureIdx < features.Length)
                    {
                        features[featureIdx++] = value;
                    }
                }
            }
        }

        return features;
    }

    /// <summary>
    /// Extracts a window of data from the incremental buffer.
    /// </summary>
    /// <param name="state">The incremental state.</param>
    /// <param name="featureIndex">The feature (column) index.</param>
    /// <param name="windowSize">The window size to extract.</param>
    /// <returns>Array of values in the window (most recent last).</returns>
    protected double[] ExtractIncrementalWindow(IncrementalState<T> state, int featureIndex, int windowSize)
    {
        var buffer = state.RollingBuffer[featureIndex];
        int bufferLen = buffer.Length;
        int pos = state.BufferPosition;
        var window = new double[windowSize];

        for (int i = 0; i < windowSize; i++)
        {
            // Calculate position in circular buffer
            // pos = position where current value was just written
            // Window should be [oldest, ..., newest] where newest is at pos
            // oldest is at pos - windowSize + 1
            int bufferIdx = (pos - windowSize + 1 + i + bufferLen) % bufferLen;
            window[i] = NumOps.ToDouble(buffer[bufferIdx]);
        }

        return window;
    }

    /// <summary>
    /// Computes features for a window of data. Override in derived classes.
    /// </summary>
    /// <param name="window">The window of data values.</param>
    /// <param name="currentValue">The current (most recent) value.</param>
    /// <returns>Array of computed feature values.</returns>
    protected virtual T[] ComputeFeaturesForWindow(double[] window, T currentValue)
    {
        // Default implementation - derived classes should override
        return [NumOps.FromDouble(window.Average())];
    }

    #endregion

    #region Serialization

    /// <summary>
    /// Exports the transformer's state for serialization.
    /// </summary>
    /// <returns>A serializable state object containing all transformer state.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method captures everything needed to recreate the transformer
    /// in its current state. You can serialize this object to JSON, binary, or any format you prefer.
    ///
    /// Example:
    /// <code>
    /// var state = transformer.ExportState();
    /// var json = JsonSerializer.Serialize(state);
    /// File.WriteAllText("transformer.json", json);
    /// </code>
    /// </para>
    /// </remarks>
    public virtual TransformerState<T> ExportState()
    {
        return new TransformerState<T>
        {
            TransformerType = GetType().FullName ?? GetType().Name,
            Version = 1,
            IsFitted = IsFitted,
            WindowSizes = _windowSizes,
            FeatureNames = _featureNames,
            InputFeatureNames = _inputFeatureNames,
            InputFeatureCount = _inputFeatureCount,
            OutputFeatureCount = _outputFeatureCount,
            IncrementalState = _incrementalState,
            Parameters = ExportParameters(),
            Options = ExportOptions()
        };
    }

    /// <summary>
    /// Imports a previously exported state to restore the transformer.
    /// </summary>
    /// <param name="state">The state to import.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method restores a transformer from a saved state.
    /// The transformer must be of the same type and version.
    ///
    /// Example:
    /// <code>
    /// var json = File.ReadAllText("transformer.json");
    /// var state = JsonSerializer.Deserialize&lt;TransformerState&lt;double&gt;&gt;(json);
    /// transformer.ImportState(state);
    /// </code>
    /// </para>
    /// </remarks>
    public virtual void ImportState(TransformerState<T> state)
    {
        if (state == null)
        {
            throw new ArgumentNullException(nameof(state));
        }

        string expectedType = GetType().FullName ?? GetType().Name;
        if (state.TransformerType != expectedType)
        {
            throw new ArgumentException(
                $"State is for transformer type '{state.TransformerType}' but this is '{expectedType}'.");
        }

        _windowSizes = state.WindowSizes;
        _featureNames = state.FeatureNames;
        _inputFeatureNames = state.InputFeatureNames;
        _inputFeatureCount = state.InputFeatureCount;
        _outputFeatureCount = state.OutputFeatureCount;
        _incrementalState = state.IncrementalState;
        IsFitted = state.IsFitted;

        ImportParameters(state.Parameters);
        ImportOptions(state.Options);
    }

    /// <summary>
    /// Exports transformer-specific parameters. Override in derived classes.
    /// </summary>
    /// <returns>Dictionary of parameter names and values.</returns>
    protected virtual Dictionary<string, object> ExportParameters()
    {
        return [];
    }

    /// <summary>
    /// Imports transformer-specific parameters. Override in derived classes.
    /// </summary>
    /// <param name="parameters">The parameters to import.</param>
    protected virtual void ImportParameters(Dictionary<string, object> parameters)
    {
        // Default implementation does nothing - derived classes can override
    }

    /// <summary>
    /// Exports the options used to configure this transformer.
    /// </summary>
    /// <returns>Dictionary of option names and values.</returns>
    protected virtual Dictionary<string, object> ExportOptions()
    {
        return new Dictionary<string, object>
        {
            ["WindowSizes"] = Options.WindowSizes,
            ["AutoDetectWindowSizes"] = Options.AutoDetectWindowSizes,
            ["AutoDetectionMethod"] = Options.AutoDetectionMethod.ToString(),
            ["EdgeHandling"] = Options.EdgeHandling.ToString(),
            ["UseParallelProcessing"] = Options.UseParallelProcessing,
            ["ParallelThreshold"] = Options.ParallelThreshold,
            ["FeatureNameSeparator"] = Options.FeatureNameSeparator
        };
    }

    /// <summary>
    /// Imports options from a dictionary. Override in derived classes for custom options.
    /// </summary>
    /// <param name="options">The options to import.</param>
    protected virtual void ImportOptions(Dictionary<string, object> options)
    {
        // Options are typically set during construction and shouldn't change after fitting
        // This method is available for derived classes to customize behavior if needed
    }

    #endregion
}
