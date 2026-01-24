using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
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

        var numOps = MathHelper.GetNumericOperations<T>();

        bool allNearIntegers = true;
        var classes = new HashSet<int>();

        const int maxTrackedClasses = 512;
        const double integerEpsilon = 1e-8;

        for (int i = 0; i < actual.Length; i++)
        {
            double value = numOps.ToDouble(actual[i]);

            if (double.IsNaN(value) || double.IsInfinity(value))
            {
                return PredictionType.Regression;
            }

            double rounded = Math.Round(value);
            if (Math.Abs(value - rounded) > integerEpsilon)
            {
                allNearIntegers = false;
                break;
            }

            if (classes.Count < maxTrackedClasses)
            {
                if (rounded >= int.MinValue && rounded <= int.MaxValue)
                {
                    classes.Add((int)rounded);
                }
                else
                {
                    return PredictionType.Regression;
                }
            }
        }

        if (!allNearIntegers)
        {
            return PredictionType.Regression;
        }

        if (classes.Count <= 2 && classes.All(c => c == 0 || c == 1))
        {
            return PredictionType.BinaryClassification;
        }

        if (classes.Count == 0 || classes.Count >= maxTrackedClasses)
        {
            return PredictionType.Regression;
        }

        // Heuristic: integer-valued targets are common in regression (e.g. counts, prices, IDs).
        // Treat them as classification only when labels look like a compact class set.
        //
        // - If almost every sample is its own "class", it's usually regression.
        // - If labels are not a compact contiguous range and there are many distinct values, it's usually regression.
        int minClass = classes.Min();
        int maxClass = classes.Max();

        // Use long to avoid integer overflow when minClass is very negative and maxClass is very positive
        long range = (long)maxClass - (long)minClass;

        // If range overflows int or is larger than practical class count, treat as non-contiguous
        bool isContiguous = range <= int.MaxValue && (range + 1) == classes.Count;
        double uniqueRatio = classes.Count / (double)actual.Length;

        if (uniqueRatio > 0.8 && classes.Count > 20)
        {
            return PredictionType.Regression;
        }

        if (!isContiguous && uniqueRatio > 0.2)
        {
            return PredictionType.Regression;
        }

        return PredictionType.MultiClass;
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

        if (matrix.Columns == 1)
        {
            var values = new Vector<T>(matrix.Rows);
            for (int row = 0; row < matrix.Rows; row++)
            {
                values[row] = matrix[row, 0];
            }

            return Infer(values);
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

        for (int row = 0; row < sampleRows; row++)
        {
            double rowSum = 0.0;
            int positives = 0;

            for (int col = 0; col < sampleCols; col++)
            {
                double value = numOps.ToDouble(matrix[row, col]);
                if (double.IsNaN(value) || double.IsInfinity(value))
                {
                    return PredictionType.Regression;
                }

                if (value < -probabilityEpsilon || value > 1.0 + probabilityEpsilon)
                {
                    return PredictionType.Regression;
                }

                rowSum += value;

                if (value >= positiveThreshold)
                {
                    positives++;
                }

                if (Math.Abs(value) <= nearBinaryEpsilon || Math.Abs(value - 1.0) <= nearBinaryEpsilon)
                {
                    nearBinaryCount++;
                }
            }

            if (positives > 1)
            {
                rowsWithMultiplePositives++;
            }

            if (rowSum > sumGreaterThanOneThreshold)
            {
                rowsWithSumGreaterThanOne++;
            }

            if (Math.Abs(rowSum - 1.0) <= sumNearOneEpsilon)
            {
                rowsWithSumNearOne++;
            }
        }

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
