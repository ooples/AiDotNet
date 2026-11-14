using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Evaluation;

internal static class PredictionTypeInference
{
    public static PredictionType Infer<T>(Vector<T> actual)
    {
        if (actual.Length == 0)
        {
            return PredictionType.Regression;
        }

        return InferFromValues(Tensor<T>.FromVector(actual));
    }

    private static PredictionType InferFromValues<T>(Tensor<T> actual)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();

        const int maxTrackedClasses = 512;
        const double integerEpsilon = 1e-8;

        // Any NaN or infinite value (x - x is not 0), any value further than integerEpsilon from an integer, or any
        // integer outside the int range makes the targets regression.
        if (!AllFinite(actual))
        {
            return PredictionType.Regression;
        }

        var rounded = engine.TensorRound(actual);
        if (EvaluationTensorOps<T>.Count(engine.TensorGreaterThan(
                engine.TensorAbs(engine.TensorSubtract(actual, rounded)), numOps.FromDouble(integerEpsilon))) > 0)
        {
            return PredictionType.Regression;
        }

        var wideRounded = rounded.Cast<double>();
        if (EvaluationTensorOps<double>.Count(engine.TensorLessThan(wideRounded, (double)int.MinValue)) > 0 ||
            EvaluationTensorOps<double>.Count(engine.TensorGreaterThan(wideRounded, (double)int.MaxValue)) > 0)
        {
            return PredictionType.Regression;
        }

        // Classification inference needs at most 511 distinct integer labels.
        // Stop tracking at the regression threshold rather than materializing every unique value.
        var classes = new HashSet<int>();
        int minClass = int.MaxValue, maxClass = int.MinValue;
        foreach (double value in wideRounded)
        {
            int label = (int)value;
            classes.Add(label);
            minClass = Math.Min(minClass, label);
            maxClass = Math.Max(maxClass, label);
            if (classes.Count >= maxTrackedClasses) return PredictionType.Regression;
        }
        int classCount = classes.Count;

        if (classCount <= 2 && minClass >= 0 && maxClass <= 1)
        {
            return PredictionType.BinaryClassification;
        }

        // Heuristic: integer-valued targets are common in regression (e.g. counts, prices, IDs).
        // Treat them as classification only when labels look like a compact class set.
        //
        // - If almost every sample is its own "class", it's usually regression.
        // - If labels are not a compact contiguous range and there are many distinct values, it's usually regression.
        // Use long to avoid integer overflow when minClass is very negative and maxClass is very positive
        long range = (long)maxClass - (long)minClass;

        // If range overflows int or is larger than practical class count, treat as non-contiguous
        bool isContiguous = range <= int.MaxValue && (range + 1) == classCount;
        double uniqueRatio = classCount / (double)actual.Length;

        if (uniqueRatio > 0.8 && classCount > 20)
        {
            return PredictionType.Regression;
        }

        if (!isContiguous && uniqueRatio > 0.2)
        {
            return PredictionType.Regression;
        }

        return PredictionType.MultiClass;
    }

    // 1 where value <= limit (the values are already known to be finite).
    private static Tensor<T> AtMost<T>(Tensor<T> values, T limit)
        => EvaluationTensorOps<T>.Not(AiDotNetEngine.Current.TensorGreaterThan(values, limit));

    // x - x is 0 exactly when x is finite (NaN and infinities give NaN).
    private static bool AllFinite<T>(Tensor<T> values)
    {
        var engine = AiDotNetEngine.Current;
        var zero = MathHelper.GetNumericOperations<T>().Zero;
        return EvaluationTensorOps<T>.Count(EvaluationTensorOps<T>.EqualMask(engine.TensorSubtract(values, values), zero)) == values.Length;
    }

    public static PredictionType InferFromTargets<T, TOutput>(TOutput targets)
    {
        if (targets is Matrix<T> matrix)
        {
            return InferFromMatrixTargets(matrix);
        }

        if (targets is Tensor<T> tensor)
        {
            return InferFromTensorTargets(tensor);
        }

        try
        {
            return Infer(ConversionsHelper.ConvertToVector<T, TOutput>(targets));
        }
        catch (InvalidOperationException)
        {
            return PredictionType.Regression;
        }
        catch (ArgumentException)
        {
            return PredictionType.Regression;
        }
        catch (NotSupportedException)
        {
            return PredictionType.Regression;
        }
    }

    private static PredictionType InferFromTensorTargets<T>(Tensor<T> tensor)
    {
        if (tensor is null || tensor.Length == 0)
        {
            return PredictionType.Regression;
        }

        if (tensor.Rank == 1)
        {
            return Infer(tensor.ToVector());
        }

        if (tensor.Rank == 2)
        {
            if (tensor.Shape.Length >= 2 && tensor.Shape[1] <= 1)
            {
                return Infer(tensor.ToVector());
            }

            return InferFromMatrixTargets(tensor.ToMatrix());
        }

        return PredictionType.Regression;
    }

    private static PredictionType InferFromMatrixTargets<T>(Matrix<T> matrix)
    {
        if (matrix is null || matrix.Rows <= 0 || matrix.Columns <= 0)
        {
            return PredictionType.Regression;
        }

        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        var targets = Tensor<T>.FromMatrix(matrix);

        if (matrix.Columns == 1)
        {
            return InferFromValues(engine.Reshape(targets, new[] { matrix.Rows }));
        }

        var numOps = MathHelper.GetNumericOperations<T>();

        int sampleRows = Math.Min(matrix.Rows, 128);
        int sampleCols = Math.Min(matrix.Columns, 128);

        int nearBinaryCount = 0;
        int totalSampleCount = sampleRows * sampleCols;

        int rowsWithMultiplePositives = 0;
        int rowsWithSumGreaterThanOne = 0;
        int rowsWithSumNearOne = 0;

        const double probabilityEpsilon = 1e-3;
        const double nearBinaryEpsilon = 0.15;
        const double positiveThreshold = 0.5;
        const double sumNearOneEpsilon = 0.2;
        const double sumGreaterThanOneThreshold = 1.2;

        var sample = engine.TensorSlice(targets, new[] { 0, 0 }, new[] { sampleRows, sampleCols });
        if (!AllFinite(sample))
        {
            return PredictionType.Regression;
        }

        if (EvaluationTensorOps<T>.Count(engine.TensorLessThan(sample, numOps.FromDouble(-probabilityEpsilon))) > 0 ||
            EvaluationTensorOps<T>.Count(engine.TensorGreaterThan(sample, numOps.FromDouble(1.0 + probabilityEpsilon))) > 0)
        {
            return PredictionType.Regression;
        }

        var nearBinaryLimit = numOps.FromDouble(nearBinaryEpsilon);
        var nearZero = AtMost(engine.TensorAbs(sample), nearBinaryLimit);
        var nearOne = AtMost(engine.TensorAbs(engine.TensorSubtractScalar(sample, numOps.One)), nearBinaryLimit);
        nearBinaryCount = EvaluationTensorOps<T>.Count(engine.TensorSubtract(engine.TensorAdd(nearZero, nearOne), engine.TensorMultiply(nearZero, nearOne)));

        var positives = engine.ReduceSum(
            EvaluationTensorOps<T>.GreaterOrEqualMask(sample, numOps.FromDouble(positiveThreshold)), new[] { 1 }, false);
        var rowSums = engine.ReduceSum(sample, new[] { 1 }, false);
        rowsWithMultiplePositives = EvaluationTensorOps<T>.Count(engine.TensorGreaterThan(positives, numOps.One));
        rowsWithSumGreaterThanOne = EvaluationTensorOps<T>.Count(
            engine.TensorGreaterThan(rowSums, numOps.FromDouble(sumGreaterThanOneThreshold)));
        rowsWithSumNearOne = EvaluationTensorOps<T>.Count(AtMost(
            engine.TensorAbs(engine.TensorSubtractScalar(rowSums, numOps.One)), numOps.FromDouble(sumNearOneEpsilon)));

        double nearBinaryRatio = totalSampleCount == 0 ? 0.0 : nearBinaryCount / (double)totalSampleCount;
        if (nearBinaryRatio < 0.6)
        {
            return PredictionType.Regression;
        }

        if (rowsWithMultiplePositives > 0 || rowsWithSumGreaterThanOne > 0)
        {
            return PredictionType.MultiLabel;
        }

        if (rowsWithSumNearOne >= sampleRows / 2)
        {
            return PredictionType.MultiClass;
        }

        return PredictionType.MultiLabel;
    }
}
