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
    protected virtual int[] DetectUsingSpectralAnalysis(Tensor<T> data)
    {
        // Simplified implementation - in a full implementation, would use FFT
        // to find dominant frequencies and convert to periods
        return DetectUsingHeuristics(data);
    }

    /// <summary>
    /// Detects optimal window sizes using grid search with cross-validation.
    /// </summary>
    protected virtual int[] DetectUsingGridSearch(Tensor<T> data)
    {
        // Simplified implementation - in a full implementation, would try
        // different window sizes and evaluate using cross-validation
        return DetectUsingHeuristics(data);
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

    #endregion
}
