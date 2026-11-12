using System;
using System.Linq;
using AiDotNet;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.UncertaintyQuantification;

/// <summary>
/// Reference tests for the split-conformal (LAC) classification threshold of Sadinle, Lei and Wasserman (2019):
/// the threshold on the true-class probability is its floor((n + 1) alpha)-th smallest calibration value, which
/// gives prediction sets {c : p_c >= threshold} with coverage at least 1 - alpha.
/// </summary>
public class ConformalClassificationThresholdTests
{
    private static double Threshold(double[] scores, double confidence)
        => AiModelBuilder<double, Matrix<double>, Vector<double>>.ComputeConformalClassificationThreshold(
            new Vector<double>(scores), confidence);

    [Fact]
    public void Threshold_IsTheFloorOfNPlusOneAlphaSmallestScore()
    {
        // n = 19, alpha = 0.1: floor(20 * 0.1) = 2, the second smallest score. Master returned the third.
        var scores = Enumerable.Range(1, 19).Select(i => i / 20.0).Reverse().ToArray();
        Assert.Equal(0.10, Threshold(scores, 0.9), 12);

        // n = 49, alpha = 0.2: floor(50 * 0.2) = 10, the tenth smallest.
        var more = Enumerable.Range(1, 49).Select(i => i / 50.0).ToArray();
        Assert.Equal(0.20, Threshold(more, 0.8), 12);
    }

    [Fact]
    public void Threshold_KeepsEveryClassWhenTooFewCalibrationPoints()
    {
        // n = 8, alpha = 0.1: floor(0.9) = 0, so no finite quantile exists and every class must be kept.
        var scores = new[] { 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2 };
        Assert.Equal(0.0, Threshold(scores, 0.9), 12);
    }

    [Fact]
    public void Threshold_GivesMarginalCoverageOnExchangeableScores()
    {
        // Calibrate and test on independent uniform true-class probabilities: P(p_test >= threshold) >= 1 - alpha.
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(2024);
        const int trials = 4000, n = 30;
        const double confidence = 0.9;
        int covered = 0;
        for (int t = 0; t < trials; t++)
        {
            var calibration = Enumerable.Range(0, n).Select(_ => random.NextDouble()).ToArray();
            double threshold = Threshold(calibration, confidence);
            if (random.NextDouble() >= threshold)
            {
                covered++;
            }
        }

        // Exact coverage is 1 - floor(31 * 0.1) / 31 = 0.9032; master's rank gave 1 - 5/31 = 0.839.
        Assert.InRange(covered / (double)trials, 0.89, 0.92);
    }
}
