using System;
using AiDotNet.FitDetectors;
using AiDotNet.Models;
using AiDotNet.Models.Inputs;
using AiDotNet.Statistics;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.FitDetectors;

/// <summary>
/// Regression tests for Issue #1186: CalibratedProbabilityFitDetector crashing with
/// ArgumentOutOfRangeException on multiclass tensor probabilities + class-index labels.
/// </summary>
/// <remarks>
/// The legacy implementation flattened both Predicted and Actual with
/// ConversionsHelper.ConvertToVector, then built bin indices from the flattened predicted
/// vector and indexed actual[idx] with idx drawn from [0, N*C). For N=100, C=3 that walked
/// up to index 299 into a length-100 vector and threw. The fix detects the multiclass shape
/// ratio and reduces predictions to "probability of the true class", reusing the existing
/// binary calibration path.
/// </remarks>
public class CalibratedProbabilityFitDetectorIssue1186Tests
{
    /// <summary>
    /// The exact repro from the issue: predictions shaped [100, 3] (flattened length 300)
    /// with labels shaped [100] of class indices. Must no longer throw OOR.
    /// </summary>
    [Fact]
    public void Issue1186_MulticlassTensorPredictions_DoesNotThrow()
    {
        const int samples = 100;
        const int classes = 3;
        var rng = new Random(42);

        // Build [samples, classes] probability tensor (rows sum to 1) + class-index labels.
        var predTensor = new Tensor<double>(new[] { samples, classes });
        var actualTensor = new Tensor<double>(new[] { samples });
        for (int i = 0; i < samples; i++)
        {
            double[] logits = { rng.NextDouble(), rng.NextDouble(), rng.NextDouble() };
            double sum = logits[0] + logits[1] + logits[2];
            for (int c = 0; c < classes; c++) predTensor[i, c] = logits[c] / sum;

            // True class = argmax corrupted with 10% flip.
            int argmax = 0;
            for (int c = 1; c < classes; c++) if (predTensor[i, c] > predTensor[i, argmax]) argmax = c;
            actualTensor[i] = rng.NextDouble() < 0.9 ? argmax : rng.Next(classes);
        }

        var detector = new CalibratedProbabilityFitDetector<double, Tensor<double>, Tensor<double>>();
        var evalData = BuildEvalData(predTensor, actualTensor);

        // Before the fix this threw ArgumentOutOfRangeException inside CalculateCalibration.
        var result = detector.DetectFit(evalData);

        Assert.NotNull(result);
        Assert.True(Enum.IsDefined(typeof(AiDotNet.Enums.FitType), result.FitType));
        Assert.True(result.ConfidenceLevel >= 0 && result.ConfidenceLevel <= 1,
            $"ConfidenceLevel = {result.ConfidenceLevel} should be in [0, 1].");
        Assert.NotEmpty(result.Recommendations);
    }

    /// <summary>
    /// Classic binary case (predicted length == actual length) still works — no regression.
    /// </summary>
    [Fact]
    public void Issue1186_BinaryProbabilities_StillWork()
    {
        const int samples = 100;
        var rng = new Random(7);

        var predTensor = new Tensor<double>(new[] { samples });
        var actualTensor = new Tensor<double>(new[] { samples });
        for (int i = 0; i < samples; i++)
        {
            double p = rng.NextDouble();
            predTensor[i] = p;
            actualTensor[i] = rng.NextDouble() < p ? 1.0 : 0.0;
        }

        var detector = new CalibratedProbabilityFitDetector<double, Tensor<double>, Tensor<double>>();
        var result = detector.DetectFit(BuildEvalData(predTensor, actualTensor));

        Assert.NotNull(result);
        Assert.NotEmpty(result.Recommendations);
    }

    /// <summary>
    /// Non-integer-multiple mismatches (e.g. predicted length 100, actual length 23)
    /// should surface as a clear InvalidOperationException — not an opaque OOR.
    /// </summary>
    [Fact]
    public void Issue1186_ShapeMismatch_ThrowsClearError()
    {
        var predTensor = new Tensor<double>(new[] { 100 });
        var actualTensor = new Tensor<double>(new[] { 23 });
        for (int i = 0; i < 100; i++) predTensor[i] = 0.5;
        for (int i = 0; i < 23; i++) actualTensor[i] = 1.0;

        var detector = new CalibratedProbabilityFitDetector<double, Tensor<double>, Tensor<double>>();
        var ex = Assert.Throws<InvalidOperationException>(
            () => detector.DetectFit(BuildEvalData(predTensor, actualTensor)));
        Assert.Contains("incompatible", ex.Message);
    }

    /// <summary>
    /// 2-class case — the smallest multiclass config that triggers the same bug path.
    /// </summary>
    [Fact]
    public void Issue1186_TwoClassTensor_DoesNotThrow()
    {
        const int samples = 50;
        const int classes = 2;

        var predTensor = new Tensor<double>(new[] { samples, classes });
        var actualTensor = new Tensor<double>(new[] { samples });
        var rng = new Random(123);
        for (int i = 0; i < samples; i++)
        {
            double p0 = rng.NextDouble();
            predTensor[i, 0] = p0;
            predTensor[i, 1] = 1 - p0;
            actualTensor[i] = p0 > 0.5 ? 0 : 1;
        }

        var detector = new CalibratedProbabilityFitDetector<double, Tensor<double>, Tensor<double>>();
        var result = detector.DetectFit(BuildEvalData(predTensor, actualTensor));

        // Behavioral assertions rather than a bare NotNull — a regression
        // that silently produced a garbage FitType or a confidence
        // outside [0, 1] would otherwise pass this test.
        Assert.NotNull(result);
        Assert.True(Enum.IsDefined(typeof(AiDotNet.Enums.FitType), result.FitType),
            $"FitType must be a defined enum value; got {result.FitType}.");
        Assert.InRange(result.ConfidenceLevel, 0.0, 1.0);
        Assert.NotEmpty(result.Recommendations);
    }

    private static ModelEvaluationData<double, Tensor<double>, Tensor<double>> BuildEvalData(
        Tensor<double> predicted, Tensor<double> actual)
    {
        var statsInputs = new ModelStatsInputs<double, Tensor<double>, Tensor<double>>
        {
            Actual = actual,
            Predicted = predicted
        };
        return new ModelEvaluationData<double, Tensor<double>, Tensor<double>>
        {
            ModelStats = new ModelStats<double, Tensor<double>, Tensor<double>>(statsInputs)
        };
    }
}
