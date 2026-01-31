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
        int timeSteps = GetTimeSteps(data);
        int inputFeatures = InputFeatureCount;
        int outputFeatures = OutputFeatureCount;

        var output = new Tensor<T>(new[] { timeSteps, outputFeatures });

        for (int t = 0; t < timeSteps; t++)
        {
            int outputIdx = 0;

            foreach (int windowSize in _correlationWindowSizes)
            {
                // Honor EdgeHandling for incomplete windows
                if (Options.EdgeHandling == EdgeHandling.NaN && t < windowSize - 1)
                {
                    int pairs = CountCorrelationPairs(inputFeatures);
                    for (int i = 0; i < pairs; i++)
                    {
                        output[t, outputIdx++] = NumOps.FromDouble(double.NaN);
                    }
                    continue;
                }

                // Compute correlation matrix for this window
                var corrMatrix = ComputeCorrelationMatrix(data, t, windowSize, inputFeatures);

                // Output correlations
                for (int i = 0; i < inputFeatures; i++)
                {
                    int jStart = _fullMatrix ? 0 : i + 1;
                    for (int j = jStart; j < inputFeatures; j++)
                    {
                        if (i == j && !_fullMatrix) continue; // Skip diagonal for upper triangle
                        output[t, outputIdx++] = NumOps.FromDouble(corrMatrix[i, j]);
                    }
                }
            }
        }

        return output;
    }

    /// <inheritdoc />
    protected override Tensor<T> TransformParallel(Tensor<T> data)
    {
        int timeSteps = GetTimeSteps(data);
        int inputFeatures = InputFeatureCount;
        int outputFeatures = OutputFeatureCount;

        var output = new Tensor<T>(new[] { timeSteps, outputFeatures });

        Parallel.For(0, timeSteps, t =>
        {
            int outputIdx = 0;

            foreach (int windowSize in _correlationWindowSizes)
            {
                // Honor EdgeHandling for incomplete windows
                if (Options.EdgeHandling == EdgeHandling.NaN && t < windowSize - 1)
                {
                    int pairs = CountCorrelationPairs(inputFeatures);
                    for (int i = 0; i < pairs; i++)
                    {
                        output[t, outputIdx++] = NumOps.FromDouble(double.NaN);
                    }
                    continue;
                }

                var corrMatrix = ComputeCorrelationMatrix(data, t, windowSize, inputFeatures);

                for (int i = 0; i < inputFeatures; i++)
                {
                    int jStart = _fullMatrix ? 0 : i + 1;
                    for (int j = jStart; j < inputFeatures; j++)
                    {
                        if (i == j && !_fullMatrix) continue;
                        output[t, outputIdx++] = NumOps.FromDouble(corrMatrix[i, j]);
                    }
                }
            }
        });

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
