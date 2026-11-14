using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.TimeSeries;

/// <summary>
/// Detects anomalies using STL (Seasonal and Trend decomposition using Loess).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> STL decomposes a time series into three components: trend, seasonal,
/// and residual. The residual component contains the irregular variations. Large residuals
/// indicate anomalies that don't fit the expected trend and seasonal patterns.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Apply STL decomposition to extract trend, seasonal, and residual
/// 2. Standardize the residual component
/// 3. Large standardized residuals indicate anomalies
/// </para>
/// <para>
/// <b>When to use:</b>
/// - Time series with clear trend and/or seasonal patterns
/// - When anomalies disrupt expected patterns
/// - Sales, weather, sensor data with periodicity
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Season length: 7 (weekly pattern)
/// - Trend smoothness: 15
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Cleveland, R.B., et al. (1990). "STL: A Seasonal-Trend Decomposition Procedure
/// Based on Loess." Journal of Official Statistics.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.TimeSeriesModel)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("STL: A Seasonal-Trend Decomposition Procedure Based on Loess", "https://doi.org/10.6028/jres.095.015", Year = 1990, Authors = "Robert B. Cleveland, William S. Cleveland, Jean E. McRae, Irma Terpenning")]
public partial class STLDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _seasonLength;
    private readonly int _trendSmoothness;
    [Buffer]
    private Vector<T>? _trend;
    [Buffer]
    private Vector<T>? _seasonal;

    private T _residualStd;


    /// <summary>
    /// Gets the season length.
    /// </summary>
    public int SeasonLength => _seasonLength;

    /// <summary>
    /// Gets the trend smoothness parameter.
    /// </summary>
    public int TrendSmoothness => _trendSmoothness;

    /// <summary>
    /// Creates a new STL anomaly detector.
    /// </summary>
    /// <param name="seasonLength">Length of seasonal period. Default is 7.</param>
    /// <param name="trendSmoothness">Smoothness of trend component. Default is 15.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public STLDetector(int seasonLength = 7, int trendSmoothness = 15,
        double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (seasonLength < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(seasonLength),
                "SeasonLength must be at least 2. Recommended is 7.");
        }

        if (trendSmoothness < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(trendSmoothness),
                "TrendSmoothness must be at least 1. Recommended is 15.");
        }

        _seasonLength = seasonLength;
        _trendSmoothness = trendSmoothness;
        _residualStd = NumOps.Zero;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        if (X.Columns != 1)
        {
            throw new ArgumentException(
                "STL expects univariate time series (1 column).",
                nameof(X));
        }

        int n = X.Rows;

        // Validate that series length is at least the season length
        if (n < _seasonLength)
        {
            throw new ArgumentException(
                $"Time series length ({n}) must be at least the season length ({_seasonLength}). " +
                "Either provide more data or use a smaller season length.",
                nameof(X));
        }

        var values = Engine.Reshape(AnomalyTensorOps.ToTensor(X), new[] { n });

        // Perform STL decomposition
        DecomposeSTL(values);

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    private void DecomposeSTL(Tensor<T> values)
    {
        var engine = Engine;
        int n = values.Length;
        int seasonLength = _seasonLength;

        // Season position i % seasonLength of every point and the number of points at each position
        var position = new int[n];
        var positionCount = new int[seasonLength];
        for (int i = 0; i < n; i++)
        {
            position[i] = i % seasonLength;
            positionCount[position[i]]++;
        }

        var positions = Tensor<int>.FromVector(Vector<int>.Wrap(position), new[] { n });
        var counts = Tensor<int>.FromVector(Vector<int>.Wrap(positionCount), new[] { seasonLength }).Cast<T>();

        // Initialize
        var trend = Tensor<T>.CreateZeros(new[] { n });
        var seasonal = Tensor<T>.CreateZeros(new[] { n });

        // Iterative STL (simplified)
        for (int iter = 0; iter < 3; iter++)
        {
            // Step 1: Detrend
            var detrended = engine.TensorSubtract(values, trend);

            // Step 2: Extract seasonal using subseries: the mean of every season position's subseries
            // (simple average for now), assigned back to its points
            var subMeans = engine.TensorDivide(engine.ScatterAdd(detrended, positions, 0, seasonLength), counts);
            var seasonalTemp = engine.TensorGather(subMeans, positions, 0);

            // Normalize seasonal (mean = 0)
            seasonal = engine.TensorSubtractScalar(seasonalTemp, engine.TensorMean(seasonalTemp));

            // Step 3: Deseasonalize
            var deseasoned = engine.TensorSubtract(values, seasonal);

            // Step 4: Smooth to get trend (moving average); fed into the next iteration
            trend = SmoothMovingAverage(deseasoned, _trendSmoothness);
        }

        _trend = trend.ToVector();
        _seasonal = seasonal.ToVector();

        // Compute residual standard deviation sqrt(mean(residual^2)), floored at 1e-10
        var residuals = engine.TensorSubtract(engine.TensorSubtract(values, trend), seasonal);
        T residualStd = NumOps.Sqrt(engine.TensorMean(engine.TensorSquare(residuals)));
        T floor = NumOps.FromDouble(1e-10);
        _residualStd = NumOps.LessThan(residualStd, floor) ? floor : residualStd;
    }

    // Centered moving average over [i - windowSize / 2, i + windowSize / 2], truncated at the ends.
    private Tensor<T> SmoothMovingAverage(Tensor<T> values, int windowSize)
    {
        var engine = Engine;
        int n = values.Length;
        int halfWindow = windowSize / 2;

        var windowCount = new int[n];
        for (int i = 0; i < n; i++)
        {
            windowCount[i] = Math.Min(n, i + halfWindow + 1) - Math.Max(0, i - halfWindow);
        }

        var padded = values;
        if (halfWindow > 0)
        {
            var padding = Tensor<T>.CreateZeros(new[] { halfWindow });
            padded = engine.TensorConcatenate(new[] { padding, values, padding }, 0);
        }

        var sums = engine.ReduceSum(engine.TensorUnfold(padded, 0, 2 * halfWindow + 1, 1), new[] { 1 }, false);
        return engine.TensorDivide(sums, Tensor<int>.FromVector(Vector<int>.Wrap(windowCount), new[] { n }).Cast<T>());
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
                "STL expects univariate time series (1 column).",
                nameof(X));
        }

        var trend = _trend;
        var seasonal = _seasonal;

        if (trend == null || seasonal == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        // Use modular indexing for seasonal if test data is longer
        int m = X.Rows;
        var trendIndex = new int[m];
        var seasonIndex = new int[m];
        for (int i = 0; i < m; i++)
        {
            trendIndex[i] = Math.Min(i, trend.Length - 1);
            seasonIndex[i] = i % _seasonLength;
        }

        var engine = Engine;
        var values = engine.Reshape(AnomalyTensorOps.ToTensor(X), new[] { m });
        var residual = engine.TensorSubtract(
            engine.TensorSubtract(values, engine.TensorGather(trend.AsTensor(), Tensor<int>.FromVector(Vector<int>.Wrap(trendIndex), new[] { m }), 0)),
            engine.TensorGather(seasonal.AsTensor(), Tensor<int>.FromVector(Vector<int>.Wrap(seasonIndex), new[] { m }), 0));
        return engine.TensorDivideScalar(engine.TensorAbs(residual), _residualStd).ToVector();
    }
}
