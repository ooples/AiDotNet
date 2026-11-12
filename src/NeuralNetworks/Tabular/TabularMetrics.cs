using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// Loss and metric kernels shared by the tabular classifiers and regressors, expressed as whole-tensor
/// engine operations.
/// </summary>
/// <remarks>
/// Every method reproduces the scalar loops it replaced: tensors are read in flat (row-major) order, sums
/// are divided by the same counts, and degenerate cases (empty input, constant targets) return what the
/// loops returned.
/// </remarks>
internal static class TabularMetrics
{
    /// <summary>
    /// Mean of (prediction - target)^2 over the flat elements of two equally sized tensors.
    /// </summary>
    public static T MeanSquaredError<T>(Tensor<T> predictions, Tensor<T> targets)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var diff = engine.TensorSubtract(Flat(predictions, predictions.Length), Flat(targets, predictions.Length));
        return ops.Divide(engine.TensorSum(engine.TensorMultiply(diff, diff)), ops.FromDouble(predictions.Length));
    }

    /// <summary>
    /// Mean of |prediction - target| over the flat elements of two equally sized tensors.
    /// </summary>
    public static T MeanAbsoluteError<T>(Tensor<T> predictions, Tensor<T> targets)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var diff = engine.TensorSubtract(Flat(predictions, predictions.Length), Flat(targets, predictions.Length));
        return ops.Divide(engine.TensorSum(engine.TensorAbs(diff)), ops.FromDouble(predictions.Length));
    }

    /// <summary>
    /// Coefficient of determination over the first <c>targets.Length</c> flat predictions:
    /// 1 - SS_res / SS_tot, or (SS_res == 0 ? 1 : 0) when the targets are constant.
    /// </summary>
    public static T R2Score<T>(Tensor<T> predictions, Tensor<T> targets)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = targets.Length;
        var y = Flat(targets, n);
        var yHat = Flat(predictions, n);

        var targetMean = ops.Divide(engine.TensorSum(y), ops.FromDouble(n));
        var residual = engine.TensorSubtract(y, yHat);
        var deviation = engine.TensorSubtract(y, engine.TensorFullLike(y, targetMean));
        var ssRes = engine.TensorSum(engine.TensorMultiply(residual, residual));
        var ssTot = engine.TensorSum(engine.TensorMultiply(deviation, deviation));

        if (ops.Compare(ssTot, ops.Zero) == 0)
        {
            return ops.Compare(ssRes, ops.Zero) == 0 ? ops.One : ops.Zero;
        }

        return ops.Subtract(ops.One, ops.Divide(ssRes, ssTot));
    }

    /// <summary>
    /// Returns <c>values[b * numClasses + targets[b]]</c> (flat indexing) for every <c>b &lt; batchSize</c>.
    /// </summary>
    public static Tensor<T> PickTargetEntries<T>(Tensor<T> values, Vector<int> targets, int batchSize, int numClasses)
    {
        var engine = AiDotNetEngine.Current;
        var targetIndices = Tensor<int>.FromVector(targets.GetSubVector(0, batchSize));
        var rowOffsets = engine.TensorArange<int>(0, (double)batchSize * numClasses, numClasses);
        return engine.TensorTake(values, engine.TensorAdd(targetIndices, rowOffsets));
    }

    /// <summary>
    /// Mean over the batch of -log(p[b, target_b] + epsilon), reading p in flat order.
    /// </summary>
    public static T MeanNegativeLogLikelihood<T>(
        Tensor<T> probabilities, Vector<int> targets, int batchSize, int numClasses, T epsilon)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var picked = PickTargetEntries(probabilities, targets, batchSize, numClasses);
        var losses = engine.TensorNegate(engine.TensorLog(engine.TensorAddScalar(picked, epsilon)));
        return ops.Divide(engine.TensorSum(losses), ops.FromDouble(batchSize));
    }

    /// <summary>
    /// Index of the first maximum of each row of the flat [batchSize, numClasses] layout, matching the given
    /// scan (by default a <c>NumOps.Compare(x, best) &gt; 0</c> scan seeded with column 0).
    /// </summary>
    public static Vector<int> ArgMaxRows<T>(
        Tensor<T> scores, int batchSize, int numClasses, ArgMaxScan scan = ArgMaxScan.CompareFromFirst)
    {
        if (batchSize == 0)
        {
            return new Vector<int>(0);
        }

        var rows = AiDotNetEngine.Current.TensorNarrow(Flat(scores, scores.Length), 0, 0, batchSize * numClasses);
        return TieBreakArgMax.FirstMaxIndexPerRow(rows.Reshape(batchSize, numClasses), scan).ToVector();
    }

    /// <summary>
    /// Number of positions <c>i &lt; predictions.Length</c> where <c>predictions[i] == targets[i]</c>.
    /// </summary>
    public static int CountMatches(Vector<int> predictions, Vector<int> targets)
    {
        var engine = AiDotNetEngine.Current;
        var a = Tensor<int>.FromVector(predictions);
        var b = Tensor<int>.FromVector(targets.GetSubVector(0, predictions.Length));
        return engine.TensorSum(engine.TensorEquals(a, b));
    }

    /// <summary>
    /// Number of positions where <c>NumOps.Compare(a[i], b[i]) &lt;= 0</c> for two equally shaped tensors.
    /// </summary>
    /// <remarks>
    /// <c>Compare</c> is <c>CompareTo</c>, which orders NaN below every number. So a[i] sorts above b[i] exactly
    /// when a[i] &gt; b[i] (IEEE), or b[i] is NaN and a[i] is not; every other position counts.
    /// </remarks>
    public static int CountCompareAtMost<T>(Tensor<T> a, Tensor<T> b)
    {
        var engine = AiDotNetEngine.Current;
        var zeros = engine.TensorZerosLike(a);
        var aIsNumber = engine.TensorWhere(engine.TensorIsNan(a), zeros, engine.TensorOnesLike(a));
        var numberOverNaN = engine.TensorWhere(engine.TensorIsNan(b), aIsNumber, zeros);
        var above = engine.TensorAdd(engine.TensorGreaterThan(a, b), numberOverNaN);
        int countAbove = engine.TensorCountNonzero(above);
        return a.Length - countAbove;
    }

    /// <summary>
    /// Class indices stored as values: <c>(int)NumOps.ToDouble(targets[i])</c> (truncation toward zero) for the
    /// first <paramref name="count"/> flat elements.
    /// </summary>
    public static Vector<int> ClassIndices<T>(Tensor<T> targets, int count)
    {
        return AiDotNetEngine.Current.TensorTrunc(Flat(targets, count)).Cast<int>().ToVector();
    }

    /// <summary>
    /// The first <paramref name="length"/> elements of <paramref name="tensor"/> in flat order, as a 1-D tensor.
    /// </summary>
    public static Tensor<T> Flat<T>(Tensor<T> tensor, int length)
    {
        var flat = tensor.Reshape(tensor.Length);
        return length == tensor.Length ? flat : AiDotNetEngine.Current.TensorNarrow(flat, 0, 0, length);
    }

    /// <summary>
    /// Joins parameter or gradient vectors end to end with bulk copies.
    /// </summary>
    public static Vector<T> ConcatenateVectors<T>(IReadOnlyList<Vector<T>> parts)
    {
        int total = 0;
        foreach (var part in parts)
        {
            total += part.Length;
        }

        var result = new Vector<T>(total);
        var destination = result.AsWritableSpan();
        int offset = 0;
        foreach (var part in parts)
        {
            part.AsSpan().CopyTo(destination.Slice(offset, part.Length));
            offset += part.Length;
        }

        return result;
    }
}
