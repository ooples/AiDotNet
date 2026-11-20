using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

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
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
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

        // Sort by combined score. Half and BFloat16 are scored in float: their ranges overflow past 65504, and scores
        // rounded back to Half would tie samples that master's double scores order.
        var time = PreprocessingTensorOps.Column(X, _timeColumn);
        var xs = PreprocessingTensorOps.Column(X, _xColumn);
        var ys = PreprocessingTensorOps.Column(X, _yColumn);
        var sortedIndices = PreprocessingTensorOps.IsNarrowFloat<T>()
            ? PreprocessingTensorOps.StableOrder(
                CombinedScores(time.Cast<float>(), xs.Cast<float>(), ys.Cast<float>(), _temporalWeight, _spatialWeight),
                descending: false)
            : PreprocessingTensorOps.StableOrder(
                CombinedScores(time, xs, ys, _temporalWeight, _spatialWeight), descending: false);

        // Split: earlier/closer samples for training, later/farther for testing
        var trainIndices = sortedIndices.Take(nSamples - targetTestSize).ToArray();
        var testIndices = sortedIndices.Skip(nSamples - targetTestSize).ToArray();

        return BuildResult(X, y, trainIndices, testIndices);
    }

    /// <summary>
    /// Combined score per sample (higher = later time, farther from origin):
    /// <c>temporalWeight * t + spatialWeight * sqrt(x^2 + y^2) / sqrt(2)</c> over min-max normalized time and
    /// coordinates (a non-positive range divides by 1).
    /// </summary>
    private static Tensor<TC> CombinedScores<TC>(
        Tensor<TC> time, Tensor<TC> xs, Tensor<TC> ys, double temporalWeight, double spatialWeight)
    {
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<TC>();
        var normalizedTime = PreprocessingTensorOps.MinMaxNormalize(time);
        var normalizedX = PreprocessingTensorOps.MinMaxNormalize(xs);
        var normalizedY = PreprocessingTensorOps.MinMaxNormalize(ys);
        var spatialDist = engine.TensorDivideScalar(
            engine.TensorSqrt(engine.TensorAdd(
                engine.TensorMultiply(normalizedX, normalizedX), engine.TensorMultiply(normalizedY, normalizedY))),
            ops.FromDouble(Math.Sqrt(2)));
        return engine.TensorAdd(
            engine.TensorMultiplyScalar(normalizedTime, ops.FromDouble(temporalWeight)),
            engine.TensorMultiplyScalar(spatialDist, ops.FromDouble(spatialWeight)));
    }
}
