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

        var engine = AiDotNetEngine.Current;
        var distancesSq = GeometryEngineOps<T>.PairwiseSquaredDistances(source.Points, target.Points);
        var sourceToTarget = GeometryEngineOps<T>.RowMinSkippingNan(distancesSq);
        var targetToSource = GeometryEngineOps<T>.RowMinSkippingNan(engine.TensorTranspose(distancesSq));
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

        var engine = AiDotNetEngine.Current;
        var thresholdSq = NumOps.Multiply(threshold, threshold);
        var distancesSq = GeometryEngineOps<T>.PairwiseSquaredDistances(prediction.Points, groundTruth.Points);
        var within = engine.TensorLessEqual(distancesSq, engine.TensorFullLike(distancesSq, thresholdSq));

        // A point is matched when any partner lies within the threshold.
        var predictionMatched = engine.ReduceMax(within, new[] { 1 }, false);
        var groundTruthMatched = engine.ReduceMax(within, new[] { 0 }, false);

        var precision = NumOps.Divide(engine.TensorSum(predictionMatched), NumOps.FromDouble(numPred));
        var recall = NumOps.Divide(engine.TensorSum(groundTruthMatched), NumOps.FromDouble(numGT));

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

        var engine = AiDotNetEngine.Current;
        var distances = engine.TensorSqrt(
            GeometryEngineOps<T>.PairwiseSquaredDistances(source.Points, target.Points));
        var sourceToTarget = engine.TensorMaxValue(GeometryEngineOps<T>.RowMinSkippingNan(distances));
        var targetToSource = engine.TensorMaxValue(
            GeometryEngineOps<T>.RowMinSkippingNan(engine.TensorTranspose(distances)));

        // Both directed distances start from zero, as the running maxima did.
        var worst = NumOps.GreaterThan(sourceToTarget, targetToSource) ? sourceToTarget : targetToSource;
        return NumOps.GreaterThan(worst, NumOps.Zero) ? worst : NumOps.Zero;
    }
}
