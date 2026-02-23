using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.DomainSpecific;

/// <summary>
/// Combined temporal-spatial splitter for data with both time and location dimensions.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Some data has both time and space components, like:
/// - Weather observations over time at different locations
/// - Traffic patterns at intersections over days
/// - Disease spread tracking
/// </para>
/// <para>
/// This splitter considers both dimensions to prevent leakage from nearby times AND places.
/// </para>
/// <para>
/// <b>Modes:</b>
/// - Time-first: Split by time, then spatially within time periods
/// - Space-first: Split by location, then temporally within regions
/// - Combined: Use a weighted distance combining both
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class TemporalSpatialSplitter<T> : DataSplitterBase<T>
{
    private readonly double _testSize;
    private readonly double _temporalWeight;
    private readonly double _spatialWeight;
    private readonly int _timeColumn;
    private readonly int _xColumn;
    private readonly int _yColumn;

    /// <summary>
    /// Creates a new temporal-spatial splitter.
    /// </summary>
    /// <param name="testSize">Proportion for test set. Default is 0.2 (20%).</param>
    /// <param name="temporalWeight">Weight for temporal dimension. Default is 0.5.</param>
    /// <param name="spatialWeight">Weight for spatial dimension. Default is 0.5.</param>
    /// <param name="timeColumn">Column index for time. Default is 0.</param>
    /// <param name="xColumn">Column index for X coordinate. Default is 1.</param>
    /// <param name="yColumn">Column index for Y coordinate. Default is 2.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public TemporalSpatialSplitter(
        double testSize = 0.2,
        double temporalWeight = 0.5,
        double spatialWeight = 0.5,
        int timeColumn = 0,
        int xColumn = 1,
        int yColumn = 2,
        int randomSeed = 42)
        : base(shuffle: false, randomSeed) // No shuffle for temporal data
    {
        if (testSize <= 0 || testSize >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(testSize), "Test size must be between 0 and 1.");
        }

        if (temporalWeight < 0 || spatialWeight < 0)
        {
            throw new ArgumentOutOfRangeException("Weights must be non-negative.");
        }

        _testSize = testSize;
        double totalWeight = temporalWeight + spatialWeight;
        _temporalWeight = temporalWeight / totalWeight;
        _spatialWeight = spatialWeight / totalWeight;
        _timeColumn = timeColumn;
        _xColumn = xColumn;
        _yColumn = yColumn;
    }

    /// <inheritdoc/>
    public override string Description => $"Temporal-Spatial split (time={_temporalWeight:F2}, space={_spatialWeight:F2}, {_testSize * 100:F0}% test)";

    /// <inheritdoc/>
    public override DataSplitResult<T> Split(Matrix<T> X, Vector<T>? y = null)
    {
        ValidateInputs(X, y);

        int nSamples = X.Rows;
        int nFeatures = X.Columns;
        int targetTestSize = Math.Max(1, (int)(nSamples * _testSize));

        // Validate columns
        int maxCol = Math.Max(_timeColumn, Math.Max(_xColumn, _yColumn));
        if (maxCol >= nFeatures)
        {
            throw new ArgumentException(
                $"Required columns ({_timeColumn}, {_xColumn}, {_yColumn}) exceed feature count ({nFeatures}).");
        }

        // Extract and normalize coordinates
        var times = new double[nSamples];
        var xCoords = new double[nSamples];
        var yCoords = new double[nSamples];

        double minTime = double.MaxValue, maxTime = double.MinValue;
        double minX = double.MaxValue, maxX = double.MinValue;
        double minY = double.MaxValue, maxY = double.MinValue;

        for (int i = 0; i < nSamples; i++)
        {
            times[i] = Convert.ToDouble(X[i, _timeColumn]);
            xCoords[i] = Convert.ToDouble(X[i, _xColumn]);
            yCoords[i] = Convert.ToDouble(X[i, _yColumn]);

            minTime = Math.Min(minTime, times[i]);
            maxTime = Math.Max(maxTime, times[i]);
            minX = Math.Min(minX, xCoords[i]);
            maxX = Math.Max(maxX, xCoords[i]);
            minY = Math.Min(minY, yCoords[i]);
            maxY = Math.Max(maxY, yCoords[i]);
        }

        double timeRange = maxTime - minTime;
        double xRange = maxX - minX;
        double yRange = maxY - minY;

        if (timeRange <= 0) timeRange = 1;
        if (xRange <= 0) xRange = 1;
        if (yRange <= 0) yRange = 1;

        // Calculate combined score for each sample (higher = later time, farther from origin)
        var scores = new double[nSamples];
        for (int i = 0; i < nSamples; i++)
        {
            double normalizedTime = (times[i] - minTime) / timeRange;
            double normalizedX = (xCoords[i] - minX) / xRange;
            double normalizedY = (yCoords[i] - minY) / yRange;
            double spatialDist = Math.Sqrt(normalizedX * normalizedX + normalizedY * normalizedY) / Math.Sqrt(2);

            scores[i] = _temporalWeight * normalizedTime + _spatialWeight * spatialDist;
        }

        // Sort by combined score
        var sortedIndices = Enumerable.Range(0, nSamples)
            .OrderBy(i => scores[i])
            .ToArray();

        // Split: earlier/closer samples for training, later/farther for testing
        var trainIndices = sortedIndices.Take(nSamples - targetTestSize).ToArray();
        var testIndices = sortedIndices.Skip(nSamples - targetTestSize).ToArray();

        return BuildResult(X, y, trainIndices, testIndices);
    }
}
