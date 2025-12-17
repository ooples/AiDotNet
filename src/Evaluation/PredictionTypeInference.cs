using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;

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
            return PredictionType.Binary;
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
        int range = maxClass - minClass;

        bool isContiguous = (range + 1) == classes.Count;
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
}
