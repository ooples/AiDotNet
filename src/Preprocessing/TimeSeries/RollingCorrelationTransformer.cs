using AiDotNet.Models.Options;

namespace AiDotNet.Preprocessing.TimeSeries;

/// <summary>
/// Computes rolling correlation matrices for multivariate time series.
/// </summary>
/// <remarks>
/// <para>
/// This transformer calculates pairwise correlations between features over rolling windows,
/// capturing how relationships between variables change over time.
/// </para>
/// <para><b>For Beginners:</b> Correlation measures how two variables move together:
///
/// - Correlation of +1: Perfect positive relationship (when A goes up, B goes up)
/// - Correlation of 0: No relationship
/// - Correlation of -1: Perfect negative relationship (when A goes up, B goes down)
///
/// Rolling correlation shows how these relationships change over time. For example:
/// - Stock A and Stock B might be highly correlated during calm markets
/// - But become less correlated during volatile periods
///
/// This is useful for:
/// - Portfolio diversification (low correlation = better diversification)
/// - Detecting regime changes (when correlations suddenly change)
/// - Pair trading (betting on correlated stocks diverging temporarily)
///
/// Output for 3 features with upper triangle only:
/// - feature_0 vs feature_1 correlation
/// - feature_0 vs feature_2 correlation
/// - feature_1 vs feature_2 correlation
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class RollingCorrelationTransformer<T> : TimeSeriesTransformerBase<T>
{
    #region Fields

    /// <summary>
    /// The window sizes for correlation calculations.
    /// </summary>
    private readonly int[] _correlationWindowSizes;

    /// <summary>
    /// Whether to output full matrix or just upper triangle.
    /// </summary>
    private readonly bool _fullMatrix;

    #endregion

    #region Constructor

    /// <summary>
    /// Creates a new rolling correlation transformer with the specified options.
    /// </summary>
    /// <param name="options">Configuration options, or null for defaults.</param>
    public RollingCorrelationTransformer(TimeSeriesFeatureOptions? options = null)
        : base(options)
    {
        _correlationWindowSizes = Options.CorrelationWindowSizes ?? Options.WindowSizes;
        _fullMatrix = Options.FullCorrelationMatrix;
    }

    #endregion

    #region Properties

    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    #endregion

    #region Core Implementation

    /// <inheritdoc />
    protected override void FitCore(Tensor<T> data)
    {
        // Correlation doesn't need to learn parameters
    }

    /// <inheritdoc />
    protected override Tensor<T> TransformCore(Tensor<T> data)
    {
        int inputTimeSteps = GetTimeSteps(data);
        int inputFeatures = InputFeatureCount;
        int outputFeatures = OutputFeatureCount;

        // Determine output dimensions based on edge handling
        int outputTimeSteps = GetOutputTimeSteps(inputTimeSteps);
        int startIndex = GetOutputStartIndex();

        // Handle Truncate mode with empty output
        if (outputTimeSteps <= 0)
        {
            return new Tensor<T>(new[] { 0, outputFeatures });
        }

        var output = new Tensor<T>(new[] { outputTimeSteps, outputFeatures });

        // Track first valid index for forward fill
        int firstValidIndex = -1;
        int maxWindow = _correlationWindowSizes.Length > 0 ? _correlationWindowSizes.Max() : GetMaxWindowSize();

        for (int outT = 0; outT < outputTimeSteps; outT++)
        {
            int t = outT + startIndex; // Map to input time step
            int outputIdx = 0;

            foreach (int windowSize in _correlationWindowSizes)
            {
                bool isEdge = IsEdgeRegion(t, windowSize);

                // Handle edge cases based on EdgeHandling mode
                if (isEdge)
                {
                    switch (Options.EdgeHandling)
                    {
                        case EdgeHandling.NaN:
                            int pairs = CountCorrelationPairs(inputFeatures);
                            for (int i = 0; i < pairs; i++)
                            {
                                output[outT, outputIdx++] = GetNaN();
                            }
                            continue;

                        case EdgeHandling.Partial:
                        case EdgeHandling.ForwardFill:
                            // Use partial window
                            break;

                        case EdgeHandling.Truncate:
                            break;
                    }
                }

                // Track first valid index for forward fill
                if (firstValidIndex < 0 && !IsEdgeRegion(t, maxWindow))
                {
                    firstValidIndex = outT;
                }

                // Compute correlation matrix for this window (may be partial)
                int effectiveWindow = ShouldComputePartialWindows() && isEdge
                    ? GetEffectiveWindowSize(t, windowSize)
                    : windowSize;
                var corrMatrix = ComputeCorrelationMatrix(data, t, effectiveWindow, inputFeatures);

                // Output correlations
                for (int i = 0; i < inputFeatures; i++)
                {
                    int jStart = _fullMatrix ? 0 : i + 1;
                    for (int j = jStart; j < inputFeatures; j++)
                    {
                        if (i == j && !_fullMatrix) continue; // Skip diagonal for upper triangle
                        output[outT, outputIdx++] = NumOps.FromDouble(corrMatrix[i, j]);
                    }
                }
            }
        }

        // Apply forward fill if needed
        if (Options.EdgeHandling == EdgeHandling.ForwardFill && firstValidIndex > 0)
        {
            ApplyForwardFill(output, firstValidIndex);
        }

        return output;
    }

    /// <inheritdoc />
    protected override Tensor<T> TransformParallel(Tensor<T> data)
    {
        int inputTimeSteps = GetTimeSteps(data);
        int inputFeatures = InputFeatureCount;
        int outputFeatures = OutputFeatureCount;

        // Determine output dimensions based on edge handling
        int outputTimeSteps = GetOutputTimeSteps(inputTimeSteps);
        int startIndex = GetOutputStartIndex();
        int maxWindow = _correlationWindowSizes.Length > 0 ? _correlationWindowSizes.Max() : GetMaxWindowSize();

        // Handle Truncate mode with empty output
        if (outputTimeSteps <= 0)
        {
            return new Tensor<T>(new[] { 0, outputFeatures });
        }

        var output = new Tensor<T>(new[] { outputTimeSteps, outputFeatures });

        // Track first valid index for forward fill (thread-safe)
        int firstValidIndex = -1;
        object lockObj = new object();

        Parallel.For(0, outputTimeSteps, outT =>
        {
            int t = outT + startIndex; // Map to input time step
            int outputIdx = 0;

            foreach (int windowSize in _correlationWindowSizes)
            {
                bool isEdge = IsEdgeRegion(t, windowSize);

                // Handle edge cases based on EdgeHandling mode
                if (isEdge)
                {
                    switch (Options.EdgeHandling)
                    {
                        case EdgeHandling.NaN:
                            int pairs = CountCorrelationPairs(inputFeatures);
                            for (int i = 0; i < pairs; i++)
                            {
                                output[outT, outputIdx++] = GetNaN();
                            }
                            continue;

                        case EdgeHandling.Partial:
                        case EdgeHandling.ForwardFill:
                            // Use partial window
                            break;

                        case EdgeHandling.Truncate:
                            break;
                    }
                }

                // Track first valid index for forward fill (thread-safe)
                if (!IsEdgeRegion(t, maxWindow))
                {
                    lock (lockObj)
                    {
                        if (firstValidIndex < 0 || outT < firstValidIndex)
                            firstValidIndex = outT;
                    }
                }

                int effectiveWindow = ShouldComputePartialWindows() && isEdge
                    ? GetEffectiveWindowSize(t, windowSize)
                    : windowSize;
                var corrMatrix = ComputeCorrelationMatrix(data, t, effectiveWindow, inputFeatures);

                for (int i = 0; i < inputFeatures; i++)
                {
                    int jStart = _fullMatrix ? 0 : i + 1;
                    for (int j = jStart; j < inputFeatures; j++)
                    {
                        if (i == j && !_fullMatrix) continue;
                        output[outT, outputIdx++] = NumOps.FromDouble(corrMatrix[i, j]);
                    }
                }
            }
        });

        // Apply forward fill if needed (sequential post-processing)
        if (Options.EdgeHandling == EdgeHandling.ForwardFill && firstValidIndex > 0)
        {
            ApplyForwardFill(output, firstValidIndex);
        }

        return output;
    }

    #endregion

    #region Feature Naming

    /// <inheritdoc />
    protected override string[] GenerateFeatureNames()
    {
        var names = new List<string>();
        var inputNames = GetInputFeatureNames();
        var sep = GetSeparator();
        int inputFeatures = inputNames.Length;

        foreach (int windowSize in _correlationWindowSizes)
        {
            for (int i = 0; i < inputFeatures; i++)
            {
                int jStart = _fullMatrix ? 0 : i + 1;
                for (int j = jStart; j < inputFeatures; j++)
                {
                    if (i == j && !_fullMatrix) continue;
                    names.Add($"{inputNames[i]}{sep}corr{sep}{inputNames[j]}{sep}{windowSize}");
                }
            }
        }

        return [.. names];
    }

    /// <inheritdoc />
    protected override string[] GetOperationNames()
    {
        return ["correlation"];
    }

    /// <summary>
    /// Counts the number of correlation pairs for the given number of features.
    /// </summary>
    private int CountCorrelationPairs(int numFeatures)
    {
        // Full matrix: n*n, Upper triangle without diagonal: n*(n-1)/2
        return _fullMatrix
            ? numFeatures * numFeatures
            : numFeatures * (numFeatures - 1) / 2;
    }

    #endregion

    #region Incremental Computation

    /// <summary>
    /// Gets whether this transformer supports incremental transformation.
    /// Rolling correlation requires multivariate data synchronization which is complex for incremental.
    /// </summary>
    public override bool SupportsIncrementalTransform => false;

    #endregion

    #region Serialization

    /// <summary>
    /// Exports transformer-specific parameters for serialization.
    /// </summary>
    protected override Dictionary<string, object> ExportParameters()
    {
        return new Dictionary<string, object>
        {
            ["CorrelationWindowSizes"] = _correlationWindowSizes,
            ["FullMatrix"] = _fullMatrix
        };
    }

    /// <summary>
    /// Imports transformer-specific parameters for validation.
    /// </summary>
    protected override void ImportParameters(Dictionary<string, object> parameters)
    {
        if (parameters.TryGetValue("FullMatrix", out var fullMatrixObj))
        {
            bool savedFullMatrix = Convert.ToBoolean(fullMatrixObj);
            if (savedFullMatrix != _fullMatrix)
            {
                throw new ArgumentException(
                    $"Saved FullMatrix ({savedFullMatrix}) does not match current configuration ({_fullMatrix}).");
            }
        }
    }

    #endregion

    #region Correlation Computation

    /// <summary>
    /// Computes the correlation matrix for a rolling window.
    /// </summary>
    private double[,] ComputeCorrelationMatrix(Tensor<T> data, int endTime, int windowSize, int numFeatures)
    {
        var corrMatrix = new double[numFeatures, numFeatures];

        // Extract data for each feature in the window
        var featureData = new double[numFeatures][];
        for (int f = 0; f < numFeatures; f++)
        {
            featureData[f] = ExtractWindow(data, endTime, f, windowSize);
        }

        // Compute correlations
        for (int i = 0; i < numFeatures; i++)
        {
            for (int j = i; j < numFeatures; j++)
            {
                double corr = ComputeCorrelation(featureData[i], featureData[j]);
                corrMatrix[i, j] = corr;
                corrMatrix[j, i] = corr; // Symmetric
            }
        }

        return corrMatrix;
    }

    /// <summary>
    /// Extracts data for a rolling window.
    /// </summary>
    private double[] ExtractWindow(Tensor<T> data, int endTime, int feature, int windowSize)
    {
        int startTime = endTime - windowSize + 1;
        var window = new double[windowSize];

        for (int i = 0; i < windowSize; i++)
        {
            int t = startTime + i;
            window[i] = t < 0
                ? double.NaN
                : NumOps.ToDouble(GetValue(data, t, feature));
        }

        return window;
    }

    /// <summary>
    /// Computes Pearson correlation between two series.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Pearson correlation measures linear relationship:
    ///
    /// r = sum((x - mean_x) * (y - mean_y)) / (std_x * std_y * n)
    ///
    /// It ranges from -1 to +1:
    /// - +1: Perfect positive correlation
    /// - 0: No linear correlation
    /// - -1: Perfect negative correlation
    /// </para>
    /// </remarks>
    private static double ComputeCorrelation(double[] x, double[] y)
    {
        if (x.Length != y.Length || x.Length < 2)
            return double.NaN;

        // Filter pairs where both are valid
        var validPairs = x.Zip(y, (a, b) => (a, b))
            .Where(p => !double.IsNaN(p.a) && !double.IsNaN(p.b))
            .ToArray();

        if (validPairs.Length < 2)
            return double.NaN;

        double meanX = validPairs.Average(p => p.a);
        double meanY = validPairs.Average(p => p.b);

        double sumXY = 0, sumX2 = 0, sumY2 = 0;

        foreach (var (a, b) in validPairs)
        {
            double dx = a - meanX;
            double dy = b - meanY;
            sumXY += dx * dy;
            sumX2 += dx * dx;
            sumY2 += dy * dy;
        }

        double denom = Math.Sqrt(sumX2 * sumY2);
        if (denom < 1e-10)
            return 0; // No variation

        return sumXY / denom;
    }

    #endregion
}
