using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Preprocessing.TimeSeries;

/// <summary>
/// A pipeline that chains multiple time series transformers together.
/// </summary>
/// <remarks>
/// <para>
/// This class allows you to compose multiple transformers into a single pipeline,
/// applying them in sequence. Each transformer's output becomes the next transformer's input.
/// </para>
/// <para><b>For Beginners:</b> Think of this like an assembly line in a factory:
///
/// Raw Material -> [Machine 1] -> [Machine 2] -> [Machine 3] -> Final Product
///
/// Similarly, your data flows through each transformer in order:
///
/// Raw Data -> [Lag Features] -> [Rolling Stats] -> [Technical Indicators] -> Enhanced Features
///
/// This makes it easy to build complex feature engineering workflows by combining
/// simple, focused transformers.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
/// <example>
/// <code>
/// // Create individual transformers
/// var lagTransformer = new LagLeadTransformer&lt;double&gt;(lagOptions);
/// var statsTransformer = new RollingStatsTransformer&lt;double&gt;(statsOptions);
/// var indicatorTransformer = new TechnicalIndicatorsTransformer&lt;double&gt;(indOptions);
///
/// // Build a pipeline
/// var pipeline = new TimeSeriesTransformerPipeline&lt;double&gt;()
///     .AddTransformer(lagTransformer)
///     .AddTransformer(statsTransformer)
///     .AddTransformer(indicatorTransformer);
///
/// // Fit and transform
/// var enrichedData = pipeline.FitTransform(data);
/// </code>
/// </example>
public class TimeSeriesTransformerPipeline<T> : ITimeSeriesFeatureExtractor<T>
{
    #region Fields

    /// <summary>
    /// Gets the numeric operations helper for type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// The ordered list of transformers in this pipeline.
    /// </summary>
    private readonly List<ITimeSeriesFeatureExtractor<T>> _transformers;

    /// <summary>
    /// The combined feature names from all transformers.
    /// </summary>
    private string[] _featureNames;

    /// <summary>
    /// The window sizes used by the pipeline (union of all transformer windows).
    /// </summary>
    private int[] _windowSizes;

    /// <summary>
    /// The number of features expected in the input data.
    /// </summary>
    private int _inputFeatureCount;

    /// <summary>
    /// The total number of features produced by the pipeline.
    /// </summary>
    private int _outputFeatureCount;

    /// <summary>
    /// Whether to concatenate original features with transformed features.
    /// </summary>
    private readonly bool _includeOriginalFeatures;

    /// <summary>
    /// Whether each transformer receives only the original input (parallel mode)
    /// or the output of the previous transformer (sequential mode).
    /// </summary>
    private readonly PipelineMode _pipelineMode;

    #endregion

    #region Constructor

    /// <summary>
    /// Creates a new pipeline with default settings.
    /// </summary>
    public TimeSeriesTransformerPipeline()
        : this(includeOriginalFeatures: true, pipelineMode: PipelineMode.Parallel)
    {
    }

    /// <summary>
    /// Creates a new pipeline with the specified settings.
    /// </summary>
    /// <param name="includeOriginalFeatures">Whether to include original input features in output.</param>
    /// <param name="pipelineMode">How transformers receive their input (Parallel or Sequential).</param>
    /// <remarks>
    /// <para><b>Pipeline Modes:</b></para>
    /// <para>
    /// <b>Parallel mode (default):</b> Each transformer receives the original input data.
    /// Outputs from all transformers are concatenated together.
    /// Use this when transformers are independent feature extractors.
    /// </para>
    /// <para>
    /// <b>Sequential mode:</b> Each transformer receives the previous transformer's output.
    /// Use this when transformers depend on each other (e.g., normalize then extract features).
    /// </para>
    /// </remarks>
    public TimeSeriesTransformerPipeline(bool includeOriginalFeatures, PipelineMode pipelineMode)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _transformers = [];
        _featureNames = [];
        _windowSizes = [];
        _inputFeatureCount = 0;
        _outputFeatureCount = 0;
        _includeOriginalFeatures = includeOriginalFeatures;
        _pipelineMode = pipelineMode;
    }

    #endregion

    #region IDataTransformer Implementation

    /// <inheritdoc />
    public bool IsFitted { get; private set; }

    /// <inheritdoc />
    public int[]? ColumnIndices => null;

    /// <inheritdoc />
    public bool SupportsInverseTransform => false;

    #endregion

    #region ITimeSeriesFeatureExtractor Implementation

    /// <inheritdoc />
    public int[] WindowSizes => _windowSizes;

    /// <inheritdoc />
    public bool AutoDetectEnabled => _transformers.Any(t => t.AutoDetectEnabled);

    /// <inheritdoc />
    public string[] FeatureNames => _featureNames;

    /// <inheritdoc />
    public int InputFeatureCount => _inputFeatureCount;

    /// <inheritdoc />
    public int OutputFeatureCount => _outputFeatureCount;

    #endregion

    #region Pipeline Building

    /// <summary>
    /// Gets the number of transformers in the pipeline.
    /// </summary>
    public int TransformerCount => _transformers.Count;

    /// <summary>
    /// Gets the transformer at the specified index.
    /// </summary>
    /// <param name="index">The zero-based index.</param>
    /// <returns>The transformer at the specified position.</returns>
    public ITimeSeriesFeatureExtractor<T> this[int index] => _transformers[index];

    /// <summary>
    /// Adds a transformer to the end of the pipeline.
    /// </summary>
    /// <param name="transformer">The transformer to add.</param>
    /// <returns>This pipeline instance for method chaining.</returns>
    /// <exception cref="ArgumentNullException">If transformer is null.</exception>
    /// <exception cref="InvalidOperationException">If pipeline is already fitted.</exception>
    public TimeSeriesTransformerPipeline<T> AddTransformer(ITimeSeriesFeatureExtractor<T> transformer)
    {
        if (transformer == null)
        {
            throw new ArgumentNullException(nameof(transformer));
        }

        if (IsFitted)
        {
            throw new InvalidOperationException(
                "Cannot add transformers to a fitted pipeline. Create a new pipeline instead.");
        }

        _transformers.Add(transformer);
        return this;
    }

    /// <summary>
    /// Adds multiple transformers to the pipeline.
    /// </summary>
    /// <param name="transformers">The transformers to add.</param>
    /// <returns>This pipeline instance for method chaining.</returns>
    public TimeSeriesTransformerPipeline<T> AddTransformers(params ITimeSeriesFeatureExtractor<T>[] transformers)
    {
        foreach (var transformer in transformers)
        {
            AddTransformer(transformer);
        }
        return this;
    }

    /// <summary>
    /// Adds multiple transformers to the pipeline.
    /// </summary>
    /// <param name="transformers">The transformers to add.</param>
    /// <returns>This pipeline instance for method chaining.</returns>
    public TimeSeriesTransformerPipeline<T> AddTransformers(IEnumerable<ITimeSeriesFeatureExtractor<T>> transformers)
    {
        foreach (var transformer in transformers)
        {
            AddTransformer(transformer);
        }
        return this;
    }

    #endregion

    #region Fit/Transform Methods

    /// <inheritdoc />
    public void Fit(Tensor<T> data)
    {
        if (_transformers.Count == 0)
        {
            throw new InvalidOperationException("Pipeline must have at least one transformer.");
        }

        ValidateInput(data);

        _inputFeatureCount = data.Shape.Length > 1 ? data.Shape[^1] : 1;

        if (_pipelineMode == PipelineMode.Sequential)
        {
            FitSequential(data);
        }
        else
        {
            FitParallel(data);
        }

        IsFitted = true;
    }

    /// <summary>
    /// Fits transformers in sequential mode.
    /// </summary>
    private void FitSequential(Tensor<T> data)
    {
        var currentData = data;
        var allFeatureNames = new List<string>();
        var allWindowSizes = new HashSet<int>();

        // Include original feature names if configured
        if (_includeOriginalFeatures)
        {
            for (int i = 0; i < _inputFeatureCount; i++)
            {
                allFeatureNames.Add($"original_{i}");
            }
        }

        foreach (var transformer in _transformers)
        {
            transformer.Fit(currentData);
            currentData = transformer.Transform(currentData);

            allFeatureNames.AddRange(transformer.FeatureNames);
            foreach (int w in transformer.WindowSizes)
            {
                allWindowSizes.Add(w);
            }
        }

        _featureNames = [.. allFeatureNames];
        _windowSizes = [.. allWindowSizes.OrderBy(w => w)];
        _outputFeatureCount = _featureNames.Length;
    }

    /// <summary>
    /// Fits transformers in parallel mode (all receive original data).
    /// </summary>
    private void FitParallel(Tensor<T> data)
    {
        var allFeatureNames = new List<string>();
        var allWindowSizes = new HashSet<int>();

        // Include original feature names if configured
        if (_includeOriginalFeatures)
        {
            for (int i = 0; i < _inputFeatureCount; i++)
            {
                allFeatureNames.Add($"original_{i}");
            }
        }

        foreach (var transformer in _transformers)
        {
            transformer.Fit(data);
            allFeatureNames.AddRange(transformer.FeatureNames);
            foreach (int w in transformer.WindowSizes)
            {
                allWindowSizes.Add(w);
            }
        }

        _featureNames = [.. allFeatureNames];
        _windowSizes = [.. allWindowSizes.OrderBy(w => w)];
        _outputFeatureCount = _featureNames.Length;
    }

    /// <inheritdoc />
    public Tensor<T> Transform(Tensor<T> data)
    {
        EnsureFitted();
        ValidateInputForTransform(data);

        return _pipelineMode == PipelineMode.Sequential
            ? TransformSequential(data)
            : TransformParallel(data);
    }

    /// <summary>
    /// Transforms data in sequential mode.
    /// </summary>
    private Tensor<T> TransformSequential(Tensor<T> data)
    {
        int timeSteps = data.Shape[0];
        var outputs = new List<Tensor<T>>();

        // Include original features if configured
        if (_includeOriginalFeatures)
        {
            outputs.Add(data);
        }

        var currentData = data;
        foreach (var transformer in _transformers)
        {
            currentData = transformer.Transform(currentData);
        }

        outputs.Add(currentData);

        return ConcatenateFeatures(outputs, timeSteps);
    }

    /// <summary>
    /// Transforms data in parallel mode (concatenates all transformer outputs).
    /// </summary>
    private Tensor<T> TransformParallel(Tensor<T> data)
    {
        int timeSteps = data.Shape[0];
        var outputs = new List<Tensor<T>>();

        // Include original features if configured
        if (_includeOriginalFeatures)
        {
            outputs.Add(data);
        }

        // Transform with each transformer using original data
        foreach (var transformer in _transformers)
        {
            var transformed = transformer.Transform(data);
            outputs.Add(transformed);
        }

        return ConcatenateFeatures(outputs, timeSteps);
    }

    /// <summary>
    /// Concatenates multiple output tensors along the feature dimension.
    /// </summary>
    private Tensor<T> ConcatenateFeatures(List<Tensor<T>> outputs, int timeSteps)
    {
        // Calculate total features and find minimum time steps
        int totalFeatures = outputs.Sum(o => o.Shape.Length > 1 ? o.Shape[^1] : 1);
        int minTimeSteps = outputs.Min(o => o.Shape[0]);

        var result = new Tensor<T>(new[] { minTimeSteps, totalFeatures });

        int featureOffset = 0;
        foreach (var output in outputs)
        {
            int outFeatures = output.Shape.Length > 1 ? output.Shape[^1] : 1;
            int outTimeSteps = output.Shape[0];

            // Align to the end (truncate from beginning if needed)
            int startT = outTimeSteps - minTimeSteps;

            for (int t = 0; t < minTimeSteps; t++)
            {
                for (int f = 0; f < outFeatures; f++)
                {
                    T value = output.Shape.Length > 1
                        ? output[startT + t, f]
                        : output[startT + t];
                    result[t, featureOffset + f] = value;
                }
            }

            featureOffset += outFeatures;
        }

        return result;
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
        throw new NotSupportedException(
            "TimeSeriesTransformerPipeline does not support inverse transformation.");
    }

    /// <inheritdoc />
    public string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        EnsureFitted();
        return _featureNames;
    }

    #endregion

    #region Auto-Detection

    /// <inheritdoc />
    public int[] DetectOptimalWindowSizes(Tensor<T> data)
    {
        // Collect window sizes from all transformers that support auto-detection
        var allWindows = new HashSet<int>();

        foreach (var transformer in _transformers)
        {
            if (transformer.AutoDetectEnabled)
            {
                var detected = transformer.DetectOptimalWindowSizes(data);
                foreach (int w in detected)
                {
                    allWindows.Add(w);
                }
            }
        }

        return [.. allWindows.OrderBy(w => w)];
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

        if (_transformers.Count == 0)
        {
            errors.Add("Pipeline must have at least one transformer.");
            return errors;
        }

        // Validate against all transformers
        foreach (var transformer in _transformers)
        {
            errors.AddRange(transformer.GetValidationErrors(data));
        }

        return errors;
    }

    /// <summary>
    /// Validates input data for transformation.
    /// </summary>
    private void ValidateInputForTransform(Tensor<T> data)
    {
        var errors = GetValidationErrors(data);
        if (errors.Count > 0)
        {
            throw new ArgumentException(string.Join("; ", errors));
        }

        int dataFeatures = data.Shape.Length > 1 ? data.Shape[^1] : 1;
        if (dataFeatures != _inputFeatureCount)
        {
            throw new ArgumentException(
                $"Data has {dataFeatures} features but pipeline was fitted with {_inputFeatureCount} features.");
        }
    }

    /// <summary>
    /// Ensures the pipeline has been fitted.
    /// </summary>
    private void EnsureFitted()
    {
        if (!IsFitted)
        {
            throw new InvalidOperationException(
                "This pipeline has not been fitted. Call Fit() or FitTransform() first.");
        }
    }

    #endregion

    #region Utility Methods

    /// <summary>
    /// Creates a deep copy of this pipeline (transformers are not copied, only the pipeline structure).
    /// </summary>
    /// <returns>A new pipeline with the same configuration.</returns>
    public TimeSeriesTransformerPipeline<T> Clone()
    {
        var clone = new TimeSeriesTransformerPipeline<T>(_includeOriginalFeatures, _pipelineMode);
        clone._transformers.AddRange(_transformers);
        return clone;
    }

    /// <summary>
    /// Gets a summary of the pipeline configuration.
    /// </summary>
    /// <returns>A string describing the pipeline.</returns>
    public string GetSummary()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"TimeSeriesTransformerPipeline ({_pipelineMode} mode)");
        sb.AppendLine($"  Include Original Features: {_includeOriginalFeatures}");
        sb.AppendLine($"  Transformers: {_transformers.Count}");

        for (int i = 0; i < _transformers.Count; i++)
        {
            var t = _transformers[i];
            sb.AppendLine($"    [{i}] {t.GetType().Name}: {t.OutputFeatureCount} features");
        }

        if (IsFitted)
        {
            sb.AppendLine($"  Input Features: {_inputFeatureCount}");
            sb.AppendLine($"  Output Features: {_outputFeatureCount}");
            sb.AppendLine($"  Window Sizes: [{string.Join(", ", _windowSizes)}]");
        }
        else
        {
            sb.AppendLine("  Status: Not fitted");
        }

        return sb.ToString();
    }

    #endregion

    #region Incremental/Streaming Support

    /// <inheritdoc />
    public bool SupportsIncrementalTransform =>
        _pipelineMode == PipelineMode.Parallel &&
        _transformers.All(t => t.SupportsIncrementalTransform);

    /// <inheritdoc />
    public void InitializeIncremental(Tensor<T> historicalData)
    {
        if (!SupportsIncrementalTransform)
        {
            throw new NotSupportedException(
                "Pipeline does not support incremental transformation. " +
                "Sequential mode pipelines or pipelines with non-incremental transformers cannot be used incrementally.");
        }

        EnsureFitted();

        // Initialize each transformer with the historical data
        foreach (var transformer in _transformers)
        {
            transformer.InitializeIncremental(historicalData);
        }
    }

    /// <inheritdoc />
    public T[] TransformIncremental(T[] newDataPoint)
    {
        if (!SupportsIncrementalTransform)
        {
            throw new NotSupportedException(
                "Pipeline does not support incremental transformation.");
        }

        if (newDataPoint.Length != _inputFeatureCount)
        {
            throw new ArgumentException(
                $"Data point has {newDataPoint.Length} features but pipeline was fitted with {_inputFeatureCount} features.");
        }

        var allFeatures = new List<T>();

        // Include original features if configured
        if (_includeOriginalFeatures)
        {
            allFeatures.AddRange(newDataPoint);
        }

        // Transform with each transformer and collect features
        foreach (var transformer in _transformers)
        {
            var features = transformer.TransformIncremental(newDataPoint);
            allFeatures.AddRange(features);
        }

        return [.. allFeatures];
    }

    /// <inheritdoc />
    public IncrementalState<T>? GetIncrementalState()
    {
        // For pipelines, we return a composite state
        if (_transformers.Count == 0)
        {
            return null;
        }

        // Return the first transformer's state as a representative
        // Full state would need all transformers' states
        return _transformers[0].GetIncrementalState();
    }

    /// <inheritdoc />
    public void SetIncrementalState(IncrementalState<T> state)
    {
        throw new NotSupportedException(
            "Setting incremental state directly on a pipeline is not supported. " +
            "Initialize each transformer individually or use InitializeIncremental() with historical data.");
    }

    #endregion
}

/// <summary>
/// Specifies how transformers in a pipeline receive their input.
/// </summary>
public enum PipelineMode
{
    /// <summary>
    /// Each transformer receives the original input data.
    /// All transformer outputs are concatenated.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use parallel mode when your transformers are independent.
    /// For example, if you want lag features AND rolling statistics, both can compute
    /// their features from the same original data, then all features are combined.
    /// </para>
    /// </remarks>
    Parallel,

    /// <summary>
    /// Each transformer receives the previous transformer's output.
    /// The final output is the last transformer's output (plus original features if configured).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use sequential mode when transformers depend on each other.
    /// For example, first normalize the data, then extract features from the normalized data.
    /// Each step builds on the previous one.
    /// </para>
    /// </remarks>
    Sequential
}
