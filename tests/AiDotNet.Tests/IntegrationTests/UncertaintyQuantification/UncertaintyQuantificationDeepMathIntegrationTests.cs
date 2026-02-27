using AiDotNet.UncertaintyQuantification.Calibration;
using Xunit;
using System.Linq;

namespace AiDotNet.Tests.IntegrationTests.UncertaintyQuantification;

/// <summary>
/// Deep math-correctness integration tests for UncertaintyQuantification calibration classes.
/// These tests verify mathematical formulas with hand-computed values, not just API behavior.
/// </summary>
public class UncertaintyQuantificationDeepMathIntegrationTests
{
    private const double Tolerance = 1e-10;

    #region ECE - Hand-Computed Values

    [Fact]
    public void ECE_SingleBin_Overconfident_ExactECE025()
    {
        // 8 samples all with prob=0.75 (exact in IEEE 754), only 4 correct
        // All go to bin 7 (floor(0.75*10) = 7)
        // Accuracy = 4/8 = 0.5, AvgConfidence = 0.75
        // ECE = |0.5 - 0.75| * (8/8) = 0.25
        var ece = new ExpectedCalibrationError<double>(numBins: 10);
        var probs = new Vector<double>(new double[] { 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75 });
        var preds = new Vector<int>(new int[] { 1, 1, 1, 1, 1, 1, 1, 1 });
        var labels = new Vector<int>(new int[] { 1, 1, 1, 1, 0, 0, 0, 0 });

        double result = ece.Compute(probs, preds, labels);
        Assert.Equal(0.25, result, Tolerance);
    }

    [Fact]
    public void ECE_TwoBins_PerfectCalibration_ExactlyZero()
    {
        // Using IEEE 754-exact values (0.25 = 1/4, 0.75 = 3/4):
        // Bin 2 (prob=0.25): 4 samples, 1 correct -> acc=0.25, conf=0.25, diff=0
        // Bin 7 (prob=0.75): 4 samples, 3 correct -> acc=0.75, conf=0.75, diff=0
        // ECE = (4/8)*0 + (4/8)*0 = 0.0
        var ece = new ExpectedCalibrationError<double>(numBins: 10);
        var probs = new Vector<double>(new double[] { 0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75 });
        var preds = new Vector<int>(new int[] { 1, 1, 1, 1, 1, 1, 1, 1 });
        var labels = new Vector<int>(new int[] { 1, 0, 0, 0, 1, 1, 1, 0 });

        double result = ece.Compute(probs, preds, labels);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void ECE_SingleSample_CorrectPrediction_ExactECE()
    {
        // 1 sample with prob=0.75, correct prediction
        // Bin 7: acc=1.0, conf=0.75
        // ECE = |1.0 - 0.75| * 1.0 = 0.25
        var ece = new ExpectedCalibrationError<double>(numBins: 10);
        var probs = new Vector<double>(new double[] { 0.75 });
        var preds = new Vector<int>(new int[] { 1 });
        var labels = new Vector<int>(new int[] { 1 });

        double result = ece.Compute(probs, preds, labels);
        Assert.Equal(0.25, result, Tolerance);
    }

    [Fact]
    public void ECE_SingleSample_WrongPrediction_ExactECE()
    {
        // 1 sample with prob=0.75, wrong prediction
        // Bin 7: acc=0.0, conf=0.75
        // ECE = |0.0 - 0.75| * 1.0 = 0.75
        var ece = new ExpectedCalibrationError<double>(numBins: 10);
        var probs = new Vector<double>(new double[] { 0.75 });
        var preds = new Vector<int>(new int[] { 1 });
        var labels = new Vector<int>(new int[] { 0 });

        double result = ece.Compute(probs, preds, labels);
        Assert.Equal(0.75, result, Tolerance);
    }

    [Fact]
    public void ECE_TwoBins_BothOverconfident_WeightedAverage()
    {
        // Bin 2 (prob=0.25): 4 samples, 0 correct -> acc=0.0, conf=0.25
        // Bin 7 (prob=0.75): 4 samples, 4 correct -> acc=1.0, conf=0.75
        // ECE = (4/8)*|0.0-0.25| + (4/8)*|1.0-0.75|
        //      = 0.5 * 0.25 + 0.5 * 0.25 = 0.25
        var ece = new ExpectedCalibrationError<double>(numBins: 10);
        var probs = new Vector<double>(new double[] { 0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75 });
        var preds = new Vector<int>(new int[] { 1, 1, 1, 1, 1, 1, 1, 1 });
        var labels = new Vector<int>(new int[] { 0, 0, 0, 0, 1, 1, 1, 1 });

        double result = ece.Compute(probs, preds, labels);
        Assert.Equal(0.25, result, Tolerance);
    }

    [Fact]
    public void ECE_ProbExactlyZero_GoesToBinZero()
    {
        // prob=0.0 -> bin 0, prob=0.5 -> bin 5
        // Both predictions correct
        // Bin 0: 1 sample, acc=1.0, conf=0.0 -> |1.0 - 0.0| = 1.0
        // Bin 5: 1 sample, acc=1.0, conf=0.5 -> |1.0 - 0.5| = 0.5
        // ECE = (1/2)*1.0 + (1/2)*0.5 = 0.75
        var ece = new ExpectedCalibrationError<double>(numBins: 10);
        var probs = new Vector<double>(new double[] { 0.0, 0.5 });
        var preds = new Vector<int>(new int[] { 0, 1 });
        var labels = new Vector<int>(new int[] { 0, 1 });

        double result = ece.Compute(probs, preds, labels);
        Assert.Equal(0.75, result, Tolerance);
    }

    [Fact]
    public void ECE_ProbExactlyOne_GoesToLastBin()
    {
        // prob=1.0 -> clamped to bin 9 (last bin with numBins=10)
        // prob=0.5 -> bin 5
        // Bin 5: 1 sample, wrong -> acc=0.0, conf=0.5 -> |0.0 - 0.5| = 0.5
        // Bin 9: 1 sample, correct -> acc=1.0, conf=1.0 -> |1.0 - 1.0| = 0.0
        // ECE = (1/2)*0.5 + (1/2)*0.0 = 0.25
        var ece = new ExpectedCalibrationError<double>(numBins: 10);
        var probs = new Vector<double>(new double[] { 1.0, 0.5 });
        var preds = new Vector<int>(new int[] { 1, 1 });
        var labels = new Vector<int>(new int[] { 1, 0 });

        double result = ece.Compute(probs, preds, labels);
        Assert.Equal(0.25, result, Tolerance);
    }

    [Fact]
    public void ECE_ReliabilityDiagram_MatchesHandComputation()
    {
        // 4 samples in bin 2 (prob=0.25): 1 correct
        // 4 samples in bin 7 (prob=0.75): 3 correct
        var ece = new ExpectedCalibrationError<double>(numBins: 10);
        var probs = new Vector<double>(new double[] { 0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75 });
        var preds = new Vector<int>(new int[] { 1, 1, 1, 1, 1, 1, 1, 1 });
        var labels = new Vector<int>(new int[] { 1, 0, 0, 0, 1, 1, 1, 0 });

        var diagram = ece.GetReliabilityDiagram(probs, preds, labels);

        Assert.Equal(2, diagram.Count);

        // First bin (prob=0.25): conf=0.25, acc=1/4=0.25, count=4
        Assert.Equal(0.25, diagram[0].confidence, 1e-10);
        Assert.Equal(0.25, diagram[0].accuracy, 1e-10);
        Assert.Equal(4, diagram[0].count);

        // Second bin (prob=0.75): conf=0.75, acc=3/4=0.75, count=4
        Assert.Equal(0.75, diagram[1].confidence, 1e-10);
        Assert.Equal(0.75, diagram[1].accuracy, 1e-10);
        Assert.Equal(4, diagram[1].count);
    }

    [Fact]
    public void ECE_AllPredictionsWrong_AtMidConfidence()
    {
        // All predictions wrong at prob=0.5
        // Bin 5: 4 samples, 0 correct -> acc=0.0, conf=0.5
        // ECE = |0.0 - 0.5| = 0.5
        var ece = new ExpectedCalibrationError<double>(numBins: 10);
        var probs = new Vector<double>(new double[] { 0.5, 0.5, 0.5, 0.5 });
        var preds = new Vector<int>(new int[] { 1, 1, 1, 1 });
        var labels = new Vector<int>(new int[] { 0, 0, 0, 0 });

        double result = ece.Compute(probs, preds, labels);
        Assert.Equal(0.5, result, Tolerance);
    }

    [Fact]
    public void ECE_IncreasingOverconfidence_ECEIncreases()
    {
        // As accuracy drops below confidence, ECE should increase monotonically
        var ece = new ExpectedCalibrationError<double>(numBins: 10);
        var probs = new Vector<double>(new double[] { 0.75, 0.75, 0.75, 0.75 });
        var preds = new Vector<int>(new int[] { 1, 1, 1, 1 });

        // Scenario 1: perfectly calibrated (acc=0.75=conf)
        var labels1 = new Vector<int>(new int[] { 1, 1, 1, 0 });
        double ece1 = ece.Compute(probs, preds, labels1);

        // Scenario 2: slightly overconfident (acc=0.5)
        var labels2 = new Vector<int>(new int[] { 1, 1, 0, 0 });
        double ece2 = ece.Compute(probs, preds, labels2);

        // Scenario 3: very overconfident (acc=0.0)
        var labels3 = new Vector<int>(new int[] { 0, 0, 0, 0 });
        double ece3 = ece.Compute(probs, preds, labels3);

        Assert.Equal(0.0, ece1, Tolerance);
        Assert.Equal(0.25, ece2, Tolerance);
        Assert.Equal(0.75, ece3, Tolerance);
        Assert.True(ece1 < ece2);
        Assert.True(ece2 < ece3);
    }

    [Fact]
    public void ECE_OneBin_ECEEqualsAbsDifference()
    {
        // With numBins=1, all samples go in one bin
        // ECE = |overall_accuracy - overall_avg_confidence|
        var ece = new ExpectedCalibrationError<double>(numBins: 1);
        var probs = new Vector<double>(new double[] { 0.25, 0.5, 0.75 });
        var preds = new Vector<int>(new int[] { 1, 1, 1 });
        var labels = new Vector<int>(new int[] { 1, 0, 1 });

        // avgConf = (0.25 + 0.5 + 0.75) / 3 = 0.5
        // accuracy = 2/3
        // ECE = |2/3 - 0.5| = 1/6
        double result = ece.Compute(probs, preds, labels);
        double expectedAcc = 2.0 / 3.0;
        double expectedConf = (0.25 + 0.5 + 0.75) / 3.0;
        double expectedECE = Math.Abs(expectedAcc - expectedConf);
        Assert.Equal(expectedECE, result, 1e-10);
    }

    #endregion

    #region TemperatureScaling - ScaleLogits Math

    [Fact]
    public void TemperatureScaling_ScaleLogits_ExactDivision_HandComputed()
    {
        // logits = [6, 3, 1.5], T=1.5 -> scaled = [4, 2, 1]
        var ts = new TemperatureScaling<double>(initialTemperature: 1.5);
        var logits = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 6.0, 3.0, 1.5 }));
        var scaled = ts.ScaleLogits(logits);
        Assert.Equal(4.0, scaled[0], Tolerance);
        Assert.Equal(2.0, scaled[1], Tolerance);
        Assert.Equal(1.0, scaled[2], Tolerance);
    }

    [Fact]
    public void TemperatureScaling_ScaleLogits_NegativeLogits_HandComputed()
    {
        // logits = [-2, -4, -6], T=2 -> scaled = [-1, -2, -3]
        var ts = new TemperatureScaling<double>(initialTemperature: 2.0);
        var logits = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { -2.0, -4.0, -6.0 }));
        var scaled = ts.ScaleLogits(logits);
        Assert.Equal(-1.0, scaled[0], Tolerance);
        Assert.Equal(-2.0, scaled[1], Tolerance);
        Assert.Equal(-3.0, scaled[2], Tolerance);
    }

    [Fact]
    public void TemperatureScaling_ScaleLogits_PreservesZero()
    {
        // logits = [3, 0, -1], T=2 -> scaled = [1.5, 0, -0.5]
        var ts = new TemperatureScaling<double>(initialTemperature: 2.0);
        var logits = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 3.0, 0.0, -1.0 }));
        var scaled = ts.ScaleLogits(logits);
        Assert.Equal(1.5, scaled[0], Tolerance);
        Assert.Equal(0.0, scaled[1], Tolerance);
        Assert.Equal(-0.5, scaled[2], Tolerance);
    }

    [Fact]
    public void TemperatureScaling_SoftmaxAfterScaling_ProbsSumToOne()
    {
        // Scale logits [3, 1, -1, 0] with T=2 -> [1.5, 0.5, -0.5, 0]
        // Compute softmax manually and verify sum = 1
        var ts = new TemperatureScaling<double>(initialTemperature: 2.0);
        var logits = new Tensor<double>(new[] { 4 }, new Vector<double>(new double[] { 3.0, 1.0, -1.0, 0.0 }));
        var scaled = ts.ScaleLogits(logits);

        double[] scaledArr = { scaled[0], scaled[1], scaled[2], scaled[3] };
        double max = scaledArr.Max();
        double[] exps = scaledArr.Select(s => Math.Exp(s - max)).ToArray();
        double sum = exps.Sum();
        double[] probs = exps.Select(e => e / sum).ToArray();

        Assert.Equal(1.0, probs.Sum(), 1e-10);

        // Relative ordering preserved: highest logit -> highest prob
        Assert.True(probs[0] > probs[1], "Highest logit should have highest probability");
        Assert.True(probs[1] > probs[3], "Second logit > zero logit");
        Assert.True(probs[3] > probs[2], "Zero logit > negative logit");
    }

    [Fact]
    public void TemperatureScaling_LargeLogits_ScalingDoesNotOverflow()
    {
        // With very large logits, scaling should produce finite values
        var ts = new TemperatureScaling<double>(initialTemperature: 1.0);
        var logits = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 1000.0, 999.0, 998.0 }));
        var scaled = ts.ScaleLogits(logits);

        Assert.True((!double.IsNaN(scaled[0]) && !double.IsInfinity(scaled[0])));
        Assert.True((!double.IsNaN(scaled[1]) && !double.IsInfinity(scaled[1])));
        Assert.True((!double.IsNaN(scaled[2]) && !double.IsInfinity(scaled[2])));
        Assert.Equal(1000.0, scaled[0], Tolerance);
    }

    #endregion

    #region TemperatureScaling - Calibration Gradient Correctness

    [Fact]
    public void TemperatureScaling_GradientDirection_SingleWrongSample_TemperatureIncreases()
    {
        // Single sample: logits = [5, 0, 0], true label = 1 (model is confidently wrong)
        // At T=1: softmax(5,0,0) ~ [0.987, 0.007, 0.007]
        // Gradient of NLL w.r.t. T: (1/T^2)(z_y - sum(p_k * z_k))
        //   z_y = z_1 = 0, sum(p_k * z_k) ~ 0.987*5 = 4.933
        //   gradient = (0 - 4.933)/1 = -4.933 (negative)
        // Gradient descent: T -= lr * (-4.933) -> T increases
        // This is correct: softening predictions helps wrong predictions
        var ts = new TemperatureScaling<double>(initialTemperature: 1.0);
        var logits = new Matrix<double>(1, 3);
        logits[0, 0] = 5.0;
        logits[0, 1] = 0.0;
        logits[0, 2] = 0.0;
        var labels = new Vector<int>(new int[] { 1 });

        ts.Calibrate(logits, labels, learningRate: 0.01, maxIterations: 1);

        Assert.True(ts.Temperature > 1.0,
            $"Temperature should increase for confidently wrong prediction. Got {ts.Temperature}");
    }

    [Fact]
    public void TemperatureScaling_Calibrate_OverconfidentModel_TemperatureIncreases()
    {
        // Model predicts class 0 with ~98.7% confidence (logits [5,0,0])
        // But is only correct 60% of the time
        // Temperature must increase to soften predictions
        int numSamples = 10;
        var logits = new Matrix<double>(numSamples, 3);
        var labels = new Vector<int>(numSamples);

        for (int i = 0; i < numSamples; i++)
        {
            logits[i, 0] = 5.0;
            logits[i, 1] = 0.0;
            logits[i, 2] = 0.0;
            labels[i] = i < 6 ? 0 : 1;
        }

        var ts = new TemperatureScaling<double>(initialTemperature: 1.0);
        ts.Calibrate(logits, labels, learningRate: 0.01, maxIterations: 200);

        Assert.True(ts.Temperature > 1.0,
            $"Overconfident model should have T > 1 after calibration, got {ts.Temperature}");
    }

    [Fact]
    public void TemperatureScaling_Calibrate_OverconfidentModel_NLLDecreases()
    {
        // After proper calibration, NLL should decrease (predictions become better calibrated)
        int numSamples = 10;
        var logitsData = new double[numSamples, 3];
        var labelsData = new int[numSamples];

        for (int i = 0; i < numSamples; i++)
        {
            logitsData[i, 0] = 5.0;
            logitsData[i, 1] = 0.0;
            logitsData[i, 2] = 0.0;
            labelsData[i] = i < 6 ? 0 : 1;
        }

        var logits = new Matrix<double>(logitsData);
        var labels = new Vector<int>(labelsData);

        double nllBefore = ComputeNLL(logitsData, labelsData, 1.0);

        var ts = new TemperatureScaling<double>(initialTemperature: 1.0);
        ts.Calibrate(logits, labels, learningRate: 0.01, maxIterations: 200);

        double nllAfter = ComputeNLL(logitsData, labelsData, ts.Temperature);

        Assert.True(nllAfter < nllBefore,
            $"NLL should decrease after calibration. Before: {nllBefore:F6}, After: {nllAfter:F6}, T={ts.Temperature:F6}");
    }

    [Fact]
    public void TemperatureScaling_Calibrate_PerfectModel_TemperatureNearOne()
    {
        // All predictions correct with high confidence -> T should stay near 1.0
        int numSamples = 10;
        var logits = new Matrix<double>(numSamples, 3);
        var labels = new Vector<int>(numSamples);

        for (int i = 0; i < numSamples; i++)
        {
            logits[i, 0] = 5.0;
            logits[i, 1] = 0.0;
            logits[i, 2] = 0.0;
            labels[i] = 0;
        }

        var ts = new TemperatureScaling<double>(initialTemperature: 1.0);
        ts.Calibrate(logits, labels, learningRate: 0.01, maxIterations: 100);

        Assert.True(ts.Temperature > 0.9 && ts.Temperature < 1.1,
            $"Perfect model should have T near 1.0, got {ts.Temperature}");
    }

    [Fact]
    public void TemperatureScaling_TemperatureFloor_ClampsAtMinimum()
    {
        // With very high learning rate, temperature update could go negative
        // The code should clamp at 0.01
        var ts = new TemperatureScaling<double>(initialTemperature: 1.0);
        var logits = new Matrix<double>(1, 3);
        logits[0, 0] = 10.0;
        logits[0, 1] = 0.0;
        logits[0, 2] = 0.0;
        var labels = new Vector<int>(new int[] { 0 });

        // Very high learning rate to push temperature past zero
        ts.Calibrate(logits, labels, learningRate: 100.0, maxIterations: 10);

        Assert.True(ts.Temperature >= 0.01,
            $"Temperature should never go below 0.01, got {ts.Temperature}");
    }

    [Fact]
    public void TemperatureScaling_Calibrate_Convergence_MoreIterationsStabilizes()
    {
        // After enough iterations, temperature should stabilize
        int numSamples = 20;
        var logits = new Matrix<double>(numSamples, 3);
        var labels = new Vector<int>(numSamples);

        for (int i = 0; i < numSamples; i++)
        {
            logits[i, 0] = 3.0;
            logits[i, 1] = 0.0;
            logits[i, 2] = 0.0;
            labels[i] = i < 12 ? 0 : 1;
        }

        var ts1 = new TemperatureScaling<double>(initialTemperature: 1.0);
        ts1.Calibrate(logits, labels, learningRate: 0.01, maxIterations: 100);
        double t100 = ts1.Temperature;

        var ts2 = new TemperatureScaling<double>(initialTemperature: 1.0);
        ts2.Calibrate(logits, labels, learningRate: 0.01, maxIterations: 500);
        double t500 = ts2.Temperature;

        double diff = Math.Abs(t500 - t100);
        Assert.True(diff < 1.0,
            $"Temperature should converge. T@100={t100:F4}, T@500={t500:F4}, diff={diff:F4}");
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Computes average negative log-likelihood for given logits, labels, and temperature.
    /// Uses numerically stable softmax (max subtraction trick).
    /// </summary>
    private static double ComputeNLL(double[,] logits, int[] labels, double temperature)
    {
        int n = logits.GetLength(0);
        int c = logits.GetLength(1);
        double totalNll = 0;

        for (int i = 0; i < n; i++)
        {
            double max = double.NegativeInfinity;
            for (int j = 0; j < c; j++)
            {
                double scaled = logits[i, j] / temperature;
                if (scaled > max) max = scaled;
            }

            double sumExp = 0;
            for (int j = 0; j < c; j++)
            {
                sumExp += Math.Exp(logits[i, j] / temperature - max);
            }

            double logProb = logits[i, labels[i]] / temperature - max - Math.Log(sumExp);
            totalNll -= logProb;
        }

        return totalNll / n;
    }

    #endregion
}
