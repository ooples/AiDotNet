using AiDotNet.UncertaintyQuantification.Calibration;
using AiDotNet.UncertaintyQuantification.Layers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.UncertaintyQuantification;

/// <summary>
/// Integration tests for UncertaintyQuantification classes:
/// ExpectedCalibrationError, TemperatureScaling, BayesianDenseLayer, MCDropoutLayer.
/// </summary>
public class UncertaintyQuantificationIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region ExpectedCalibrationError - Construction

    [Fact]
    public void ECE_DefaultConstruction_DoesNotThrow()
    {
        var ece = new ExpectedCalibrationError<double>();
        Assert.NotNull(ece);
    }

    [Fact]
    public void ECE_CustomBins_DoesNotThrow()
    {
        var ece = new ExpectedCalibrationError<double>(numBins: 20);
        Assert.NotNull(ece);
    }

    [Fact]
    public void ECE_ZeroBins_Throws()
    {
        Assert.Throws<ArgumentException>(() => new ExpectedCalibrationError<double>(numBins: 0));
    }

    #endregion

    #region ExpectedCalibrationError - Computation

    [Fact]
    public void ECE_PerfectCalibration_ReturnsNonNegative()
    {
        var ece = new ExpectedCalibrationError<double>(numBins: 10);
        var probabilities = new Vector<double>(new double[] { 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9 });
        var predictions = new Vector<int>(new int[] { 1, 1, 1, 1, 1, 0, 0, 0, 0, 0 });
        var trueLabels = new Vector<int>(new int[] { 1, 1, 1, 1, 1, 0, 0, 0, 0, 0 });

        var result = ece.Compute(probabilities, predictions, trueLabels);
        Assert.True(result >= 0, "ECE should be non-negative");
        Assert.True(result <= 1.0, "ECE should be at most 1.0");
    }

    [Fact]
    public void ECE_AllWrong_HighECE()
    {
        var ece = new ExpectedCalibrationError<double>(numBins: 10);
        var probabilities = new Vector<double>(new double[] { 0.95, 0.95, 0.95, 0.95, 0.95 });
        var predictions = new Vector<int>(new int[] { 1, 1, 1, 1, 1 });
        var trueLabels = new Vector<int>(new int[] { 0, 0, 0, 0, 0 });

        var result = ece.Compute(probabilities, predictions, trueLabels);
        Assert.True(result > 0.5, $"ECE should be high when all predictions are wrong, got {result}");
    }

    [Fact]
    public void ECE_MismatchedLengths_Throws()
    {
        var ece = new ExpectedCalibrationError<double>();
        var probabilities = new Vector<double>(new double[] { 0.9, 0.9 });
        var predictions = new Vector<int>(new int[] { 1 });
        var trueLabels = new Vector<int>(new int[] { 1, 1 });

        Assert.Throws<ArgumentException>(() => ece.Compute(probabilities, predictions, trueLabels));
    }

    #endregion

    #region ExpectedCalibrationError - Reliability Diagram

    [Fact]
    public void ECE_GetReliabilityDiagram_ReturnsNonEmpty()
    {
        var ece = new ExpectedCalibrationError<double>(numBins: 5);
        var probabilities = new Vector<double>(new double[] { 0.1, 0.3, 0.5, 0.7, 0.9 });
        var predictions = new Vector<int>(new int[] { 0, 0, 1, 1, 1 });
        var trueLabels = new Vector<int>(new int[] { 0, 0, 0, 1, 1 });

        var diagram = ece.GetReliabilityDiagram(probabilities, predictions, trueLabels);
        Assert.NotEmpty(diagram);
        foreach (var bin in diagram)
        {
            Assert.True(bin.count > 0, "Each bin in diagram should have at least one sample");
            Assert.True(bin.confidence >= 0 && bin.confidence <= 1, "Confidence should be in [0,1]");
            Assert.True(bin.accuracy >= 0 && bin.accuracy <= 1, "Accuracy should be in [0,1]");
        }
    }

    #endregion

    #region TemperatureScaling - Construction

    [Fact]
    public void TemperatureScaling_DefaultConstruction_TemperatureIsOne()
    {
        var ts = new TemperatureScaling<double>();
        Assert.Equal(1.0, ts.Temperature, Tolerance);
    }

    [Fact]
    public void TemperatureScaling_CustomTemperature()
    {
        var ts = new TemperatureScaling<double>(initialTemperature: 2.0);
        Assert.Equal(2.0, ts.Temperature, Tolerance);
    }

    [Fact]
    public void TemperatureScaling_ZeroTemperature_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new TemperatureScaling<double>(initialTemperature: 0.0));
    }

    [Fact]
    public void TemperatureScaling_NegativeTemperature_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new TemperatureScaling<double>(initialTemperature: -1.0));
    }

    #endregion

    #region TemperatureScaling - ScaleLogits

    [Fact]
    public void TemperatureScaling_ScaleLogits_TemperatureOne_NoChange()
    {
        var ts = new TemperatureScaling<double>(initialTemperature: 1.0);
        var logits = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 2.0, 1.0, 0.5 }));
        var scaled = ts.ScaleLogits(logits);
        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(logits[i], scaled[i], Tolerance);
        }
    }

    [Fact]
    public void TemperatureScaling_ScaleLogits_HighTemperature_Softens()
    {
        var ts = new TemperatureScaling<double>(initialTemperature: 2.0);
        var logits = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 4.0, 2.0, 0.0 }));
        var scaled = ts.ScaleLogits(logits);
        Assert.Equal(2.0, scaled[0], Tolerance);
        Assert.Equal(1.0, scaled[1], Tolerance);
        Assert.Equal(0.0, scaled[2], Tolerance);
    }

    [Fact]
    public void TemperatureScaling_ScaleLogits_LowTemperature_Sharpens()
    {
        var ts = new TemperatureScaling<double>(initialTemperature: 0.5);
        var logits = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 2.0, 1.0, 0.0 }));
        var scaled = ts.ScaleLogits(logits);
        Assert.Equal(4.0, scaled[0], Tolerance);
        Assert.Equal(2.0, scaled[1], Tolerance);
        Assert.Equal(0.0, scaled[2], Tolerance);
    }

    #endregion

    #region TemperatureScaling - Calibration

    [Fact]
    public void TemperatureScaling_Calibrate_TemperatureStaysPositive()
    {
        var ts = new TemperatureScaling<double>(initialTemperature: 1.0);
        var logits = new Matrix<double>(5, 3);
        var labels = new Vector<int>(5);
        for (int i = 0; i < 5; i++)
        {
            logits[i, 0] = 10.0;
            logits[i, 1] = 0.0;
            logits[i, 2] = 0.0;
            labels[i] = 0;
        }
        ts.Calibrate(logits, labels, learningRate: 0.01, maxIterations: 50);
        Assert.True(ts.Temperature > 0, "Temperature should remain positive after calibration");
    }

    [Fact]
    public void TemperatureScaling_Calibrate_MismatchedLengths_Throws()
    {
        var ts = new TemperatureScaling<double>();
        var logits = new Matrix<double>(5, 3);
        var labels = new Vector<int>(3);
        Assert.Throws<ArgumentException>(() => ts.Calibrate(logits, labels));
    }

    #endregion

    #region MCDropoutLayer - Construction

    [Fact]
    public void MCDropoutLayer_DefaultConstruction_DoesNotThrow()
    {
        var layer = new MCDropoutLayer<double>(dropoutRate: 0.5);
        Assert.NotNull(layer);
        Assert.False(layer.MonteCarloMode);
    }

    [Fact]
    public void MCDropoutLayer_WithMCMode_DoesNotThrow()
    {
        var layer = new MCDropoutLayer<double>(dropoutRate: 0.3, mcMode: true);
        Assert.True(layer.MonteCarloMode);
    }

    [Fact]
    public void MCDropoutLayer_InvalidRate_Throws()
    {
        Assert.Throws<ArgumentException>(() => new MCDropoutLayer<double>(dropoutRate: 1.0));
        Assert.Throws<ArgumentException>(() => new MCDropoutLayer<double>(dropoutRate: -0.1));
    }

    [Fact]
    public void MCDropoutLayer_SupportsTraining()
    {
        var layer = new MCDropoutLayer<double>(dropoutRate: 0.5);
        Assert.True(layer.SupportsTraining);
    }

    [Fact]
    public void MCDropoutLayer_MonteCarloMode_CanToggle()
    {
        var layer = new MCDropoutLayer<double>(dropoutRate: 0.5);
        Assert.False(layer.MonteCarloMode);
        layer.MonteCarloMode = true;
        Assert.True(layer.MonteCarloMode);
        layer.MonteCarloMode = false;
        Assert.False(layer.MonteCarloMode);
    }

    #endregion

    #region MCDropoutLayer - Forward Pass

    [Fact]
    public void MCDropoutLayer_Forward_InTraining_ProducesOutput()
    {
        var layer = new MCDropoutLayer<double>(dropoutRate: 0.5, randomSeed: 42);
        var input = new Tensor<double>(new[] { 1, 4 }, new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0 }));
        layer.SetTrainingMode(true);
        var output = layer.Forward(input);
        Assert.Equal(input.Shape, output.Shape);
    }

    [Fact]
    public void MCDropoutLayer_Forward_InInference_NoDropout()
    {
        var layer = new MCDropoutLayer<double>(dropoutRate: 0.5, randomSeed: 42);
        var input = new Tensor<double>(new[] { 1, 4 }, new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0 }));
        layer.SetTrainingMode(false);
        layer.MonteCarloMode = false;
        var output = layer.Forward(input);
        for (int i = 0; i < input.Length; i++)
        {
            Assert.Equal(input[i], output[i], Tolerance);
        }
    }

    [Fact]
    public void MCDropoutLayer_Forward_InMCMode_AppliesDropout()
    {
        var layer = new MCDropoutLayer<double>(dropoutRate: 0.5, mcMode: true, randomSeed: 42);
        var data = new double[100];
        for (int i = 0; i < 100; i++) data[i] = 1.0;
        var input = new Tensor<double>(new[] { 1, 100 }, new Vector<double>(data));

        layer.SetTrainingMode(false);
        var output = layer.Forward(input);

        int zeroCount = 0;
        for (int i = 0; i < output.Length; i++)
        {
            if (Math.Abs(output[i]) < 1e-10) zeroCount++;
        }
        Assert.True(zeroCount > 0, "MC dropout should drop some activations");
        Assert.True(zeroCount < 100, "MC dropout should keep some activations");
    }

    #endregion

    #region BayesianDenseLayer - Construction

    [Fact]
    public void BayesianDenseLayer_DefaultConstruction_DoesNotThrow()
    {
        var layer = new BayesianDenseLayer<double>(inputSize: 4, outputSize: 2, randomSeed: 42);
        Assert.NotNull(layer);
        Assert.True(layer.SupportsTraining);
    }

    [Fact]
    public void BayesianDenseLayer_Forward_ProducesCorrectShape()
    {
        var layer = new BayesianDenseLayer<double>(inputSize: 4, outputSize: 3, randomSeed: 42);
        var input = new Tensor<double>(new[] { 1, 4 }, new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0 }));
        layer.SetTrainingMode(true);
        var output = layer.Forward(input);
        Assert.Equal(3, output.Shape[^1]);
    }

    [Fact]
    public void BayesianDenseLayer_Forward_MultipleCallsProduceDifferentOutputs()
    {
        var layer = new BayesianDenseLayer<double>(inputSize: 4, outputSize: 2, randomSeed: 42);
        var input = new Tensor<double>(new[] { 1, 4 }, new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0 }));
        layer.SetTrainingMode(true);

        var output1 = layer.Forward(input);
        var output2 = layer.Forward(input);

        bool anyDifferent = false;
        for (int i = 0; i < output1.Length; i++)
        {
            if (Math.Abs(output1[i] - output2[i]) > 1e-10)
            {
                anyDifferent = true;
                break;
            }
        }
        Assert.True(anyDifferent, "Bayesian layer should produce different outputs due to weight sampling");
    }

    #endregion
}
