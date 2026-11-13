using AiDotNet.PointCloud.Data;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Geometry.Preprocessing;

/// <summary>
/// Provides quality metrics for comparing 3D geometry (point clouds, meshes).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public static class GeometryMetrics<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Computes the Chamfer Distance between two point clouds.
    /// </summary>
    public static T ChamferDistance(PointCloudData<T> source, PointCloudData<T> target, bool squared = false)
    {
        int numSource = source.NumPoints;
        int numTarget = target.NumPoints;

        if (numSource == 0 || numTarget == 0)
        {
            return NumOps.Zero;
        }

        if (typeof(T) == typeof(Half))
        {
            // Half squares overflow above 256 and its sums above 65504, so the metric is measured in float.
            return NumOps.FromDouble(GeometryMetrics<float>.ChamferDistance(AsFloat(source), AsFloat(target), squared));
        }

        var engine = AiDotNetEngine.Current;
        var (sourceToTarget, targetToSource) = GeometryEngineOps<T>.NearestSquaredDistances(source.Points, target.Points);
        if (!squared)
        {
            sourceToTarget = engine.TensorSqrt(sourceToTarget);
            targetToSource = engine.TensorSqrt(targetToSource);
        }

        var meanSourceToTarget = NumOps.Divide(engine.TensorSum(sourceToTarget), NumOps.FromDouble(numSource));
        var meanTargetToSource = NumOps.Divide(engine.TensorSum(targetToSource), NumOps.FromDouble(numTarget));
        return NumOps.Divide(NumOps.Add(meanSourceToTarget, meanTargetToSource), NumOps.FromDouble(2));
    }

    /// <summary>
    /// Computes the F-Score between two point clouds at a given threshold.
    /// </summary>
    public static (T FScore, T Precision, T Recall) FScore(
        PointCloudData<T> prediction,
        PointCloudData<T> groundTruth,
        T threshold)
    {
        int numPred = prediction.NumPoints;
        int numGT = groundTruth.NumPoints;

        if (numPred == 0 || numGT == 0)
        {
            return (NumOps.Zero, NumOps.Zero, NumOps.Zero);
        }

        if (typeof(T) == typeof(Half))
        {
            var (f, p, r) = GeometryMetrics<float>.FScore(AsFloat(prediction), AsFloat(groundTruth), NumOps.ToFloat(threshold));
            return (NumOps.FromDouble(f), NumOps.FromDouble(p), NumOps.FromDouble(r));
        }

        var engine = AiDotNetEngine.Current;
        var thresholdSq = NumOps.Multiply(threshold, threshold);

        // A point is matched when any partner lies within the threshold. The distance matrix is walked in row
        // blocks; matches are counted in int, since 0/1 counts held in T saturate (Half stops at 2048).
        int predictionMatched = 0;
        Tensor<T>? groundTruthMatched = null;
        foreach (var (_, distancesSq) in GeometryEngineOps<T>.SquaredDistanceBlocks(prediction.Points, groundTruth.Points))
        {
            var within = engine.TensorLessEqual(distancesSq, engine.TensorFullLike(distancesSq, thresholdSq));
            predictionMatched += engine.TensorCountNonzero(engine.ReduceMax(within, new[] { 1 }, false));
            var blockMatched = engine.ReduceMax(within, new[] { 0 }, false);
            groundTruthMatched = groundTruthMatched is null
                ? blockMatched
                : engine.TensorWhere(engine.TensorGreaterThan(blockMatched, groundTruthMatched), blockMatched, groundTruthMatched);
        }

        int groundTruthMatchedCount = groundTruthMatched is null ? 0 : engine.TensorCountNonzero(groundTruthMatched);
        var precision = NumOps.FromDouble((double)predictionMatched / numPred);
        var recall = NumOps.FromDouble((double)groundTruthMatchedCount / numGT);

        var fScore = NumOps.Zero;
        var total = NumOps.Add(precision, recall);
        if (NumOps.GreaterThan(total, NumOps.Zero))
        {
            fScore = NumOps.Divide(NumOps.Multiply(NumOps.FromDouble(2), NumOps.Multiply(precision, recall)), total);
        }

        return (fScore, precision, recall);
    }

    /// <summary>
    /// Computes the Hausdorff Distance between two point clouds.
    /// </summary>
    public static T HausdorffDistance(PointCloudData<T> source, PointCloudData<T> target)
    {
        if (source.NumPoints == 0 || target.NumPoints == 0)
        {
            return NumOps.Zero;
        }

        if (typeof(T) == typeof(Half))
        {
            return NumOps.FromDouble(GeometryMetrics<float>.HausdorffDistance(AsFloat(source), AsFloat(target)));
        }

        var engine = AiDotNetEngine.Current;

        // sqrt is monotone, so the nearest distance is the root of the nearest squared distance; an all-NaN row
        // keeps the MaxValue the NaN-skipping minimum reports.
        var (sourceMinSq, targetMinSq) = GeometryEngineOps<T>.NearestSquaredDistances(source.Points, target.Points);
        var sourceToTarget = engine.TensorMaxValue(RootKeepingMax(sourceMinSq));
        var targetToSource = engine.TensorMaxValue(RootKeepingMax(targetMinSq));

        // Both directed distances start from zero, as the running maxima did.
        var worst = NumOps.GreaterThan(sourceToTarget, targetToSource) ? sourceToTarget : targetToSource;
        return NumOps.GreaterThan(worst, NumOps.Zero) ? worst : NumOps.Zero;
    }

    private static Tensor<T> RootKeepingMax(Tensor<T> squared)
    {
        var engine = AiDotNetEngine.Current;
        var isMax = engine.TensorEquals(squared, NumOps.MaxValue);
        return engine.TensorWhere(isMax, squared, engine.TensorSqrt(squared));
    }

    private static PointCloudData<float> AsFloat(PointCloudData<T> cloud)
        => new PointCloudData<float>(cloud.Points.Cast<float>());
}
