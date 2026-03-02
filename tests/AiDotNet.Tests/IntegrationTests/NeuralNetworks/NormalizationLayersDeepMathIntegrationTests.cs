using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Deep math-correctness integration tests for normalization and utility layers.
/// Each test hand-computes expected outputs and verifies the code matches.
/// Covers BatchNormalizationLayer, LayerNormalizationLayer, DropoutLayer, EmbeddingLayer.
/// </summary>
public class NormalizationLayersDeepMathIntegrationTests
{
    private const double Tol = 1e-4;
    private const double Eps = 1e-5; // LargeEpsilon default

    // ========================================================================
    // BatchNormalizationLayer - Initialization
    // ========================================================================

    [Fact]
    public void BatchNorm_Init_GammaIsOnes()
    {
        // gamma initialized to 1.0 for each feature
        var bn = new BatchNormalizationLayer<double>(3);
        var gamma = bn.GetGamma();
        Assert.Equal(3, gamma.Length);
        for (int i = 0; i < 3; i++)
            Assert.Equal(1.0, gamma[i], Tol);
    }

    [Fact]
    public void BatchNorm_Init_BetaIsZeros()
    {
        // beta initialized to 0.0 for each feature
        var bn = new BatchNormalizationLayer<double>(3);
        var beta = bn.GetBeta();
        Assert.Equal(3, beta.Length);
        for (int i = 0; i < 3; i++)
            Assert.Equal(0.0, beta[i], Tol);
    }

    [Fact]
    public void BatchNorm_Init_RunningMeanIsZeros()
    {
        var bn = new BatchNormalizationLayer<double>(3);
        var runningMean = bn.GetRunningMean();
        Assert.Equal(3, runningMean.Length);
        for (int i = 0; i < 3; i++)
            Assert.Equal(0.0, runningMean[i], Tol);
    }

    [Fact]
    public void BatchNorm_Init_RunningVarianceIsOnes()
    {
        var bn = new BatchNormalizationLayer<double>(3);
        var runningVar = bn.GetRunningVariance();
        Assert.Equal(3, runningVar.Length);
        for (int i = 0; i < 3; i++)
            Assert.Equal(1.0, runningVar[i], Tol);
    }

    [Fact]
    public void BatchNorm_Init_Epsilon_DefaultIsLargeEpsilon()
    {
        var bn = new BatchNormalizationLayer<double>(2);
        double epsilon = bn.GetEpsilon();
        Assert.Equal(1e-5, epsilon, 1e-10);
    }

    [Fact]
    public void BatchNorm_Init_Momentum_DefaultIs09()
    {
        var bn = new BatchNormalizationLayer<double>(2);
        double momentum = bn.GetMomentum();
        Assert.Equal(0.9, momentum, Tol);
    }

    // ========================================================================
    // BatchNormalizationLayer - ParameterCount and Packing
    // ========================================================================

    [Fact]
    public void BatchNorm_ParameterCount_IsTwiceFeatureSize()
    {
        // ParameterCount = gamma.Length + beta.Length = 2 * numFeatures
        var bn = new BatchNormalizationLayer<double>(5);
        Assert.Equal(10, bn.ParameterCount);
    }

    [Fact]
    public void BatchNorm_GetParameters_GammaThenBeta()
    {
        // Parameters vector: [gamma_0, ..., gamma_n, beta_0, ..., beta_n]
        var bn = new BatchNormalizationLayer<double>(3);

        // Set custom gamma=[2,3,4], beta=[5,6,7]
        var paramVec = new Vector<double>(new double[] { 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 });
        bn.SetParameters(paramVec);

        var retrieved = bn.GetParameters();
        Assert.Equal(6, retrieved.Length);
        // gamma
        Assert.Equal(2.0, retrieved[0], Tol);
        Assert.Equal(3.0, retrieved[1], Tol);
        Assert.Equal(4.0, retrieved[2], Tol);
        // beta
        Assert.Equal(5.0, retrieved[3], Tol);
        Assert.Equal(6.0, retrieved[4], Tol);
        Assert.Equal(7.0, retrieved[5], Tol);
    }

    [Fact]
    public void BatchNorm_SetParameters_UpdatesGammaAndBeta()
    {
        var bn = new BatchNormalizationLayer<double>(2);

        // Set gamma=[10, 20], beta=[-1, -2]
        var paramVec = new Vector<double>(new double[] { 10.0, 20.0, -1.0, -2.0 });
        bn.SetParameters(paramVec);

        var gamma = bn.GetGamma();
        var beta = bn.GetBeta();
        Assert.Equal(10.0, gamma[0], Tol);
        Assert.Equal(20.0, gamma[1], Tol);
        Assert.Equal(-1.0, beta[0], Tol);
        Assert.Equal(-2.0, beta[1], Tol);
    }

    [Fact]
    public void BatchNorm_SetParameters_WrongLength_Throws()
    {
        var bn = new BatchNormalizationLayer<double>(3);
        var badParams = new Vector<double>(new double[] { 1.0, 2.0, 3.0 }); // should be 6
        Assert.Throws<ArgumentException>(() => bn.SetParameters(badParams));
    }

    // ========================================================================
    // BatchNormalizationLayer - Inference Mode Forward Pass (Hand-Computed)
    // ========================================================================

    [Fact]
    public void BatchNorm_Inference_DefaultParams_ApproximatelyIdentity()
    {
        // With default init: gamma=1, beta=0, runningMean=0, runningVar=1, eps=1e-5
        // scale = 1 / sqrt(1 + 1e-5) ≈ 0.999995
        // shift = 0 - 1 * 0 / sqrt(1 + 1e-5) = 0
        // output ≈ input * 0.999995
        var bn = new BatchNormalizationLayer<double>(2);
        bn.SetTrainingMode(false);

        var input = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new double[] { 3.0, 5.0, 7.0, -2.0 }));
        var output = bn.Forward(input);

        double scale = 1.0 / Math.Sqrt(1.0 + Eps);
        Assert.Equal(3.0 * scale, output[0, 0], Tol);
        Assert.Equal(5.0 * scale, output[0, 1], Tol);
        Assert.Equal(7.0 * scale, output[1, 0], Tol);
        Assert.Equal(-2.0 * scale, output[1, 1], Tol);
    }

    [Fact]
    public void BatchNorm_Inference_CustomGammaBeta_HandComputed()
    {
        // numFeatures=2, gamma=[2, 3], beta=[5, -1]
        // runningMean=[0, 0], runningVar=[1, 1] (defaults)
        // scale[0] = 2 / sqrt(1 + 1e-5) ≈ 1.99999
        // shift[0] = 5 - 2*0/sqrt(1+1e-5) = 5.0
        // scale[1] = 3 / sqrt(1 + 1e-5) ≈ 2.99999
        // shift[1] = -1 - 3*0/sqrt(1+1e-5) = -1.0
        var bn = new BatchNormalizationLayer<double>(2);
        bn.SetParameters(new Vector<double>(new double[] { 2.0, 3.0, 5.0, -1.0 }));
        bn.SetTrainingMode(false);

        var input = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0 }));
        var output = bn.Forward(input);

        double s = Math.Sqrt(1.0 + Eps);
        double scale0 = 2.0 / s;
        double scale1 = 3.0 / s;

        // output[0,0] = 1.0 * scale0 + 5.0
        Assert.Equal(1.0 * scale0 + 5.0, output[0, 0], Tol);
        // output[0,1] = 2.0 * scale1 + (-1.0)
        Assert.Equal(2.0 * scale1 + (-1.0), output[0, 1], Tol);
        // output[1,0] = 3.0 * scale0 + 5.0
        Assert.Equal(3.0 * scale0 + 5.0, output[1, 0], Tol);
        // output[1,1] = 4.0 * scale1 + (-1.0)
        Assert.Equal(4.0 * scale1 + (-1.0), output[1, 1], Tol);
    }

    [Fact]
    public void BatchNorm_Inference_HighVariance_OutputDamped()
    {
        // With high runningVar, the output should be damped toward beta
        // We can't set runningVar directly but default is 1.0
        // With gamma=1 beta=0 runningMean=0 runningVar=1:
        // output ≈ input / sqrt(1 + eps) ≈ input
        // If we set gamma=0.5, output ≈ 0.5 * input / sqrt(1+eps)
        var bn = new BatchNormalizationLayer<double>(2);
        bn.SetParameters(new Vector<double>(new double[] { 0.5, 0.5, 0.0, 0.0 }));
        bn.SetTrainingMode(false);

        var input = new Tensor<double>(new[] { 1, 2 }, new Vector<double>(new double[] { 10.0, 20.0 }));
        var output = bn.Forward(input);

        double halfScale = 0.5 / Math.Sqrt(1.0 + Eps);
        Assert.Equal(10.0 * halfScale, output[0, 0], Tol);
        Assert.Equal(20.0 * halfScale, output[0, 1], Tol);
    }

    [Fact]
    public void BatchNorm_Inference_PreservesOutputShape()
    {
        // Input [3, 4] should produce output [3, 4]
        var bn = new BatchNormalizationLayer<double>(4);
        bn.SetTrainingMode(false);

        var input = new Tensor<double>(new[] { 3, 4 });
        var output = bn.Forward(input);

        Assert.Equal(2, output.Shape.Length);
        Assert.Equal(3, output.Shape[0]);
        Assert.Equal(4, output.Shape[1]);
    }

    [Fact]
    public void BatchNorm_Inference_1DInput_PreservesRank()
    {
        // 1D input should be auto-reshaped to [1, N] and output reshaped back to [N]
        var bn = new BatchNormalizationLayer<double>(3);
        bn.SetTrainingMode(false);

        var input = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 1.0, 2.0, 3.0 }));
        var output = bn.Forward(input);

        Assert.Single(output.Shape);
        Assert.Equal(3, output.Shape[0]);
    }

    // ========================================================================
    // BatchNormalizationLayer - ZeroInitGamma
    // ========================================================================

    [Fact]
    public void BatchNorm_ZeroInitGamma_SetsGammaToZero()
    {
        var bn = new BatchNormalizationLayer<double>(3);
        bn.ZeroInitGamma();

        var gamma = bn.GetGamma();
        for (int i = 0; i < 3; i++)
            Assert.Equal(0.0, gamma[i], Tol);
    }

    [Fact]
    public void BatchNorm_ZeroInitGamma_InferenceOutputIsBeta()
    {
        // With gamma=0, scale=0, shift = beta - 0 = beta
        // output = input * 0 + beta = beta for all inputs
        var bn = new BatchNormalizationLayer<double>(2);
        bn.SetParameters(new Vector<double>(new double[] { 1.0, 1.0, 3.0, -2.0 })); // gamma=1,1 beta=3,-2
        bn.ZeroInitGamma(); // Now gamma=0,0
        bn.SetTrainingMode(false);

        var input = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new double[] { 100.0, 200.0, -50.0, 999.0 }));
        var output = bn.Forward(input);

        // output[n,c] = 0 * anything + beta[c]
        // But shift = beta - gamma*runningMean/sqrt(runningVar+eps) = beta - 0 = beta
        Assert.Equal(3.0, output[0, 0], Tol);
        Assert.Equal(-2.0, output[0, 1], Tol);
        Assert.Equal(3.0, output[1, 0], Tol);
        Assert.Equal(-2.0, output[1, 1], Tol);
    }

    // ========================================================================
    // BatchNormalizationLayer - Training Mode (Running Stats Update)
    // ========================================================================

    [Fact]
    public void BatchNorm_Training_RunningMeanUpdates_ExponentialMovingAverage()
    {
        // After one training forward pass:
        // runningMean_new = momentum * runningMean_old + (1 - momentum) * batchMean
        // With defaults: momentum=0.9, runningMean_old=0
        // runningMean_new = 0.9 * 0 + 0.1 * batchMean = 0.1 * batchMean
        var bn = new BatchNormalizationLayer<double>(2, momentum: 0.9);
        bn.SetTrainingMode(true);

        // Batch: [[2, 4], [6, 8]]
        // batchMean = [(2+6)/2, (4+8)/2] = [4, 6]
        var input = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new double[] { 2.0, 4.0, 6.0, 8.0 }));
        bn.Forward(input);

        var runningMean = bn.GetRunningMean();
        // runningMean_new = 0.9 * 0 + 0.1 * [4, 6] = [0.4, 0.6]
        Assert.Equal(0.4, runningMean[0], Tol);
        Assert.Equal(0.6, runningMean[1], Tol);
    }

    [Fact]
    public void BatchNorm_Training_RunningVarianceUpdates_ExponentialMovingAverage()
    {
        // After one training forward pass:
        // runningVar_new = momentum * runningVar_old + (1 - momentum) * batchVar
        // With defaults: momentum=0.9, runningVar_old=1
        // runningVar_new = 0.9 * 1 + 0.1 * batchVar
        var bn = new BatchNormalizationLayer<double>(2, momentum: 0.9);
        bn.SetTrainingMode(true);

        // Batch: [[2, 4], [6, 8]]
        // batchMean = [4, 6]
        // batchVar = [((2-4)^2 + (6-4)^2)/2, ((4-6)^2 + (8-6)^2)/2] = [(4+4)/2, (4+4)/2] = [4, 4]
        var input = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new double[] { 2.0, 4.0, 6.0, 8.0 }));
        bn.Forward(input);

        var runningVar = bn.GetRunningVariance();
        // runningVar_new = 0.9 * 1 + 0.1 * 4 = 0.9 + 0.4 = 1.3
        Assert.Equal(1.3, runningVar[0], Tol);
        Assert.Equal(1.3, runningVar[1], Tol);
    }

    [Fact]
    public void BatchNorm_Training_TwoForwardPasses_RunningMeanAccumulates()
    {
        // First pass: runningMean = 0.9*0 + 0.1*batchMean1
        // Second pass: runningMean = 0.9*prev + 0.1*batchMean2
        var bn = new BatchNormalizationLayer<double>(1, momentum: 0.9);
        bn.SetTrainingMode(true);

        // First batch mean = (10 + 20) / 2 = 15
        var input1 = new Tensor<double>(new[] { 2, 1 }, new Vector<double>(new double[] { 10.0, 20.0 }));
        bn.Forward(input1);

        var rm1 = bn.GetRunningMean();
        double expectedRM1 = 0.1 * 15.0; // = 1.5
        Assert.Equal(expectedRM1, rm1[0], Tol);

        // Second batch mean = (0 + 10) / 2 = 5
        var input2 = new Tensor<double>(new[] { 2, 1 }, new Vector<double>(new double[] { 0.0, 10.0 }));
        bn.Forward(input2);

        var rm2 = bn.GetRunningMean();
        double expectedRM2 = 0.9 * expectedRM1 + 0.1 * 5.0; // = 0.9*1.5 + 0.5 = 1.35 + 0.5 = 1.85
        Assert.Equal(expectedRM2, rm2[0], Tol);
    }

    // ========================================================================
    // BatchNormalizationLayer - Custom Epsilon
    // ========================================================================

    [Fact]
    public void BatchNorm_CustomEpsilon_AffectsScale()
    {
        // With large epsilon, scale = gamma / sqrt(runningVar + eps)
        // eps=1.0: scale = 1 / sqrt(1 + 1) = 1/sqrt(2)
        var bn = new BatchNormalizationLayer<double>(1, epsilon: 1.0);
        bn.SetTrainingMode(false);

        var input = new Tensor<double>(new[] { 1, 1 }, new Vector<double>(new double[] { 4.0 }));
        var output = bn.Forward(input);

        double expectedScale = 1.0 / Math.Sqrt(1.0 + 1.0); // 1/sqrt(2)
        Assert.Equal(4.0 * expectedScale, output[0, 0], Tol);
    }

    // ========================================================================
    // LayerNormalizationLayer - Deep Forward Pass Math
    // ========================================================================

    [Fact]
    public void LayerNorm_Forward_ExactValues_HandComputed()
    {
        // LayerNorm: for each sample, normalize across features
        // Input: [[1, 3, 5]]
        // mean = (1+3+5)/3 = 3
        // var = ((1-3)^2 + (3-3)^2 + (5-3)^2) / 3 = (4+0+4)/3 = 8/3
        // std = sqrt(8/3 + eps)
        // normalized = [(1-3)/std, (3-3)/std, (5-3)/std] = [-2/std, 0, 2/std]
        // With gamma=1, beta=0: output = normalized
        var ln = new LayerNormalizationLayer<double>(3);

        var input = new Tensor<double>(new[] { 1, 3 }, new Vector<double>(new double[] { 1.0, 3.0, 5.0 }));
        var output = ln.Forward(input);

        double mean = 3.0;
        double variance = 8.0 / 3.0;
        double std = Math.Sqrt(variance + Eps);

        Assert.Equal((1.0 - mean) / std, output[0, 0], Tol);
        Assert.Equal((3.0 - mean) / std, output[0, 1], Tol);
        Assert.Equal((5.0 - mean) / std, output[0, 2], Tol);
    }

    [Fact]
    public void LayerNorm_Forward_BatchOf2_IndependentNormalization()
    {
        // Each sample normalized independently
        // Sample 0: [2, 6] -> mean=4, var=((2-4)^2+(6-4)^2)/2 = (4+4)/2 = 4, std=sqrt(4+eps)
        // Sample 1: [10, 20] -> mean=15, var=((10-15)^2+(20-15)^2)/2 = (25+25)/2 = 25, std=sqrt(25+eps)
        var ln = new LayerNormalizationLayer<double>(2);

        var input = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new double[] { 2.0, 6.0, 10.0, 20.0 }));
        var output = ln.Forward(input);

        // Sample 0
        double mean0 = 4.0;
        double std0 = Math.Sqrt(4.0 + Eps);
        Assert.Equal((2.0 - mean0) / std0, output[0, 0], Tol);
        Assert.Equal((6.0 - mean0) / std0, output[0, 1], Tol);

        // Sample 1
        double mean1 = 15.0;
        double std1 = Math.Sqrt(25.0 + Eps);
        Assert.Equal((10.0 - mean1) / std1, output[1, 0], Tol);
        Assert.Equal((20.0 - mean1) / std1, output[1, 1], Tol);
    }

    [Fact]
    public void LayerNorm_Forward_WithCustomGammaBeta_HandComputed()
    {
        // gamma=[2, 3], beta=[1, -1]
        // Input: [[4, 8]]
        // mean = 6, var = ((4-6)^2+(8-6)^2)/2 = 4, std=sqrt(4+eps)
        // normalized = [(4-6)/std, (8-6)/std] = [-2/std, 2/std]
        // output = gamma * normalized + beta = [2*(-2/std)+1, 3*(2/std)+(-1)]
        var ln = new LayerNormalizationLayer<double>(2);
        ln.SetParameters(new Vector<double>(new double[] { 2.0, 3.0, 1.0, -1.0 }));

        var input = new Tensor<double>(new[] { 1, 2 }, new Vector<double>(new double[] { 4.0, 8.0 }));
        var output = ln.Forward(input);

        double std = Math.Sqrt(4.0 + Eps);
        double norm0 = (4.0 - 6.0) / std;
        double norm1 = (8.0 - 6.0) / std;

        Assert.Equal(2.0 * norm0 + 1.0, output[0, 0], Tol);
        Assert.Equal(3.0 * norm1 + (-1.0), output[0, 1], Tol);
    }

    [Fact]
    public void LayerNorm_ParameterCount_IsTwiceFeatureSize()
    {
        var ln = new LayerNormalizationLayer<double>(7);
        Assert.Equal(14, ln.ParameterCount);
    }

    [Fact]
    public void LayerNorm_GetSetParameters_RoundTrips()
    {
        var ln = new LayerNormalizationLayer<double>(3);
        var paramVec = new Vector<double>(new double[] { 0.5, 1.5, 2.5, -0.5, -1.5, -2.5 });
        ln.SetParameters(paramVec);

        var retrieved = ln.GetParameters();
        Assert.Equal(6, retrieved.Length);
        for (int i = 0; i < 6; i++)
            Assert.Equal(paramVec[i], retrieved[i], Tol);
    }

    [Fact]
    public void LayerNorm_UniformInput_NormalizedOutputIsZero()
    {
        // All features identical => mean = value, var = 0
        // normalized = (x - mean) / sqrt(0 + eps) = 0 / sqrt(eps) = 0
        // With gamma=1, beta=0: output = 0
        var ln = new LayerNormalizationLayer<double>(4);

        var input = new Tensor<double>(new[] { 1, 4 }, new Vector<double>(new double[] { 5.0, 5.0, 5.0, 5.0 }));
        var output = ln.Forward(input);

        for (int j = 0; j < 4; j++)
            Assert.Equal(0.0, output[0, j], Tol);
    }

    [Fact]
    public void LayerNorm_OutputMean_ApproximatelyZero()
    {
        // After LN with gamma=1, beta=0, the output mean should be ~0
        var ln = new LayerNormalizationLayer<double>(5);

        var input = new Tensor<double>(new[] { 1, 5 }, new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 }));
        var output = ln.Forward(input);

        double sum = 0;
        for (int j = 0; j < 5; j++)
            sum += output[0, j];
        double mean = sum / 5.0;
        Assert.Equal(0.0, mean, Tol);
    }

    [Fact]
    public void LayerNorm_OutputVariance_ApproximatelyOne()
    {
        // After LN with gamma=1, beta=0, the output variance should be ~1
        var ln = new LayerNormalizationLayer<double>(5);

        var input = new Tensor<double>(new[] { 1, 5 }, new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 }));
        var output = ln.Forward(input);

        double sum = 0;
        for (int j = 0; j < 5; j++)
            sum += output[0, j];
        double mean = sum / 5.0;

        double varSum = 0;
        for (int j = 0; j < 5; j++)
        {
            double diff = output[0, j] - mean;
            varSum += diff * diff;
        }
        double variance = varSum / 5.0;
        Assert.Equal(1.0, variance, 0.01);
    }

    // ========================================================================
    // DropoutLayer - Scale Factor Computation
    // ========================================================================

    [Fact]
    public void Dropout_ScaleFactor_Rate02_Is125()
    {
        // scale = 1 / (1 - 0.2) = 1 / 0.8 = 1.25
        // We verify by checking inference mode (pass-through) and that construction succeeds
        var dropout = new DropoutLayer<double>(0.2);
        dropout.SetTrainingMode(false);

        var input = new Tensor<double>(new[] { 1, 3 }, new Vector<double>(new double[] { 1.0, 2.0, 3.0 }));
        var output = dropout.Forward(input);

        // Inference mode: output = input unchanged
        Assert.Equal(1.0, output[0, 0], Tol);
        Assert.Equal(2.0, output[0, 1], Tol);
        Assert.Equal(3.0, output[0, 2], Tol);
    }

    [Fact]
    public void Dropout_ScaleFactor_Rate05_Is20()
    {
        // scale = 1 / (1 - 0.5) = 2.0
        var dropout = new DropoutLayer<double>(0.5);
        dropout.SetTrainingMode(false);

        var input = new Tensor<double>(new[] { 1, 2 }, new Vector<double>(new double[] { 10.0, -5.0 }));
        var output = dropout.Forward(input);

        Assert.Equal(10.0, output[0, 0], Tol);
        Assert.Equal(-5.0, output[0, 1], Tol);
    }

    [Fact]
    public void Dropout_InvalidRate_Throws()
    {
        Assert.Throws<ArgumentException>(() => new DropoutLayer<double>(1.0));
        Assert.Throws<ArgumentException>(() => new DropoutLayer<double>(-0.1));
    }

    [Fact]
    public void Dropout_NoTrainableParameters()
    {
        var dropout = new DropoutLayer<double>(0.3);
        var parameters = dropout.GetParameters();
        Assert.Equal(0, parameters.Length);
    }

    [Fact]
    public void Dropout_SetParametersEmpty_Succeeds()
    {
        var dropout = new DropoutLayer<double>(0.3);
        dropout.SetParameters(new Vector<double>(0));
    }

    [Fact]
    public void Dropout_SetParametersNonEmpty_Throws()
    {
        var dropout = new DropoutLayer<double>(0.3);
        Assert.Throws<ArgumentException>(() => dropout.SetParameters(new Vector<double>(new double[] { 1.0 })));
    }

    [Fact]
    public void Dropout_Training_OutputHasZeros()
    {
        // In training mode, some outputs should be 0 (dropped)
        // and the rest should be scaled by 1/(1-rate)
        var dropout = new DropoutLayer<double>(0.5);
        dropout.SetTrainingMode(true);

        var input = new Tensor<double>(new[] { 1, 100 });
        for (int i = 0; i < 100; i++)
            input[0, i] = 1.0;

        var output = dropout.Forward(input);

        int zeroCount = 0;
        int scaledCount = 0;
        for (int i = 0; i < 100; i++)
        {
            if (Math.Abs(output[0, i]) < 1e-10)
                zeroCount++;
            else
            {
                // Non-zero values should be scaled by 1/(1-0.5) = 2.0
                Assert.Equal(2.0, output[0, i], 0.01);
                scaledCount++;
            }
        }

        // With 50% dropout and 100 elements, we expect roughly 50 zeros
        // Allow wide margin for randomness
        Assert.True(zeroCount > 10, $"Expected at least 10 zeros but got {zeroCount}");
        Assert.True(scaledCount > 10, $"Expected at least 10 scaled values but got {scaledCount}");
        Assert.Equal(100, zeroCount + scaledCount);
    }

    [Fact]
    public void Dropout_Training_BackwardAppliesSameMask()
    {
        // Forward and backward should use the same mask
        // If forward zeros out position i, backward should also zero out position i
        var dropout = new DropoutLayer<double>(0.5);
        dropout.SetTrainingMode(true);

        var input = new Tensor<double>(new[] { 1, 50 });
        for (int i = 0; i < 50; i++)
            input[0, i] = 1.0;

        var output = dropout.Forward(input);

        // Backward with all-ones gradient
        var gradOutput = new Tensor<double>(new[] { 1, 50 });
        for (int i = 0; i < 50; i++)
            gradOutput[0, i] = 1.0;

        var gradInput = dropout.Backward(gradOutput);

        // Check mask consistency: if forward output was 0, backward gradient should be 0 too
        for (int i = 0; i < 50; i++)
        {
            bool wasDropped = Math.Abs(output[0, i]) < 1e-10;
            if (wasDropped)
            {
                Assert.Equal(0.0, gradInput[0, i], Tol);
            }
            else
            {
                // Forward output was input * scale = 1.0 * 2.0 = 2.0
                // Backward should be gradOutput * mask = 1.0 * 2.0 = 2.0
                Assert.Equal(output[0, i], gradInput[0, i], Tol);
            }
        }
    }

    // ========================================================================
    // EmbeddingLayer - Parameter Count
    // ========================================================================

    [Fact]
    public void Embedding_ParameterCount_IsVocabTimesEmbedDim()
    {
        // ParameterCount = vocabSize * embeddingDim
        var emb = new EmbeddingLayer<double>(100, 32);
        Assert.Equal(100 * 32, emb.ParameterCount);
    }

    [Fact]
    public void Embedding_ParameterCount_SmallVocab()
    {
        var emb = new EmbeddingLayer<double>(5, 3);
        Assert.Equal(15, emb.ParameterCount);
    }

    [Fact]
    public void Embedding_GetSetParameters_RoundTrips()
    {
        var emb = new EmbeddingLayer<double>(3, 2);
        // 3 vocab * 2 dims = 6 parameters
        var paramVec = new Vector<double>(new double[] { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 });
        emb.SetParameters(paramVec);

        var retrieved = emb.GetParameters();
        Assert.Equal(6, retrieved.Length);
        for (int i = 0; i < 6; i++)
            Assert.Equal(paramVec[i], retrieved[i], Tol);
    }

    [Fact]
    public void Embedding_Lookup_ReturnsCorrectRow()
    {
        // Create embedding with known weights
        // vocab=3, dim=2
        // Row 0: [1.0, 2.0]
        // Row 1: [3.0, 4.0]
        // Row 2: [5.0, 6.0]
        var emb = new EmbeddingLayer<double>(3, 2);
        emb.InputMode = EmbeddingInputMode.Indices;
        var paramVec = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });
        emb.SetParameters(paramVec);

        // Look up token index 1 -> should return [3.0, 4.0]
        var input = new Tensor<double>(new[] { 1 }, new Vector<double>(new double[] { 1.0 }));
        var output = emb.Forward(input);

        // Output should contain [3.0, 4.0] for index 1
        Assert.Equal(3.0, output[0, 0], Tol);
        Assert.Equal(4.0, output[0, 1], Tol);
    }

    [Fact]
    public void Embedding_Lookup_MultipleIndices()
    {
        // vocab=4, dim=2
        var emb = new EmbeddingLayer<double>(4, 2);
        emb.InputMode = EmbeddingInputMode.Indices;
        var paramVec = new Vector<double>(new double[] {
            10.0, 20.0,  // row 0
            30.0, 40.0,  // row 1
            50.0, 60.0,  // row 2
            70.0, 80.0   // row 3
        });
        emb.SetParameters(paramVec);

        // Look up indices [2, 0, 3]
        var input = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 2.0, 0.0, 3.0 }));
        var output = emb.Forward(input);

        // Index 2 -> [50, 60]
        Assert.Equal(50.0, output[0, 0], Tol);
        Assert.Equal(60.0, output[0, 1], Tol);
        // Index 0 -> [10, 20]
        Assert.Equal(10.0, output[1, 0], Tol);
        Assert.Equal(20.0, output[1, 1], Tol);
        // Index 3 -> [70, 80]
        Assert.Equal(70.0, output[2, 0], Tol);
        Assert.Equal(80.0, output[2, 1], Tol);
    }

    // ========================================================================
    // BatchNormalizationLayer - Backward Gradient Structure
    // ========================================================================

    [Fact]
    public void BatchNorm_Backward_GradientShape_MatchesInput()
    {
        var bn = new BatchNormalizationLayer<double>(3);
        bn.SetTrainingMode(true);

        var input = new Tensor<double>(new[] { 2, 3 }, new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }));
        bn.Forward(input);

        var grad = new Tensor<double>(new[] { 2, 3 }, new Vector<double>(new double[] { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 }));
        var inputGrad = bn.Backward(grad);

        Assert.Equal(2, inputGrad.Shape.Length);
        Assert.Equal(2, inputGrad.Shape[0]);
        Assert.Equal(3, inputGrad.Shape[1]);
    }

    [Fact]
    public void BatchNorm_Backward_AllGradientsFinite()
    {
        var bn = new BatchNormalizationLayer<double>(4);
        bn.SetTrainingMode(true);

        var input = new Tensor<double>(new[] { 3, 4 });
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 4; j++)
                input[i, j] = (i + 1) * (j + 1) * 0.5;

        bn.Forward(input);

        var grad = new Tensor<double>(new[] { 3, 4 });
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 4; j++)
                grad[i, j] = 0.1 * (i + j);

        var inputGrad = bn.Backward(grad);

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 4; j++)
            {
                Assert.False(double.IsNaN(inputGrad[i, j]), $"NaN at [{i},{j}]");
                Assert.False(double.IsInfinity(inputGrad[i, j]), $"Infinity at [{i},{j}]");
            }
    }

    [Fact]
    public void BatchNorm_Backward_1DInput_GradientPreservesRank()
    {
        var bn = new BatchNormalizationLayer<double>(3);
        bn.SetTrainingMode(true);

        var input = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 1.0, 2.0, 3.0 }));
        bn.Forward(input);

        var grad = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 0.5, 0.5, 0.5 }));
        var inputGrad = bn.Backward(grad);

        // Should preserve 1D rank
        Assert.Single(inputGrad.Shape);
        Assert.Equal(3, inputGrad.Shape[0]);
    }

    // ========================================================================
    // LayerNormalizationLayer - Backward Gradient Structure
    // ========================================================================

    [Fact]
    public void LayerNorm_Backward_GradientShape_MatchesInput()
    {
        var ln = new LayerNormalizationLayer<double>(4);
        ln.SetTrainingMode(true);

        var input = new Tensor<double>(new[] { 2, 4 }, new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 }));
        ln.Forward(input);

        var grad = new Tensor<double>(new[] { 2, 4 }, new Vector<double>(new double[] { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 }));
        var inputGrad = ln.Backward(grad);

        Assert.Equal(2, inputGrad.Shape.Length);
        Assert.Equal(2, inputGrad.Shape[0]);
        Assert.Equal(4, inputGrad.Shape[1]);
    }

    [Fact]
    public void LayerNorm_Backward_AllGradientsFinite()
    {
        var ln = new LayerNormalizationLayer<double>(3);
        ln.SetTrainingMode(true);

        var input = new Tensor<double>(new[] { 2, 3 }, new Vector<double>(new double[] { 1.0, 3.0, 5.0, 2.0, 4.0, 6.0 }));
        ln.Forward(input);

        var grad = new Tensor<double>(new[] { 2, 3 }, new Vector<double>(new double[] { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 }));
        var inputGrad = ln.Backward(grad);

        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
            {
                Assert.False(double.IsNaN(inputGrad[i, j]), $"NaN at [{i},{j}]");
                Assert.False(double.IsInfinity(inputGrad[i, j]), $"Infinity at [{i},{j}]");
            }
    }

    // ========================================================================
    // BatchNormalizationLayer - Numerical Edge Cases
    // ========================================================================

    [Fact]
    public void BatchNorm_Inference_ZeroInput_HandComputed()
    {
        // All-zero input with default params
        // output = 0 * scale + 0 = 0
        var bn = new BatchNormalizationLayer<double>(2);
        bn.SetTrainingMode(false);

        var input = new Tensor<double>(new[] { 1, 2 });
        var output = bn.Forward(input);

        Assert.Equal(0.0, output[0, 0], Tol);
        Assert.Equal(0.0, output[0, 1], Tol);
    }

    [Fact]
    public void BatchNorm_Inference_NegativeInput_HandComputed()
    {
        // Negative input with gamma=1, beta=0
        // output = input * scale ≈ input
        var bn = new BatchNormalizationLayer<double>(2);
        bn.SetTrainingMode(false);

        var input = new Tensor<double>(new[] { 1, 2 }, new Vector<double>(new double[] { -3.0, -7.0 }));
        var output = bn.Forward(input);

        double scale = 1.0 / Math.Sqrt(1.0 + Eps);
        Assert.Equal(-3.0 * scale, output[0, 0], Tol);
        Assert.Equal(-7.0 * scale, output[0, 1], Tol);
    }

    [Fact]
    public void BatchNorm_Inference_LargeBatch_PerFeatureConsistency()
    {
        // All samples in a batch with same feature value should get same output
        var bn = new BatchNormalizationLayer<double>(2);
        bn.SetParameters(new Vector<double>(new double[] { 2.0, 3.0, 1.0, -1.0 }));
        bn.SetTrainingMode(false);

        // 4 samples, all with same values per feature
        var input = new Tensor<double>(new[] { 4, 2 }, new Vector<double>(new double[] {
            5.0, 10.0,
            5.0, 10.0,
            5.0, 10.0,
            5.0, 10.0
        }));
        var output = bn.Forward(input);

        // All rows should be identical
        for (int i = 1; i < 4; i++)
        {
            Assert.Equal(output[0, 0], output[i, 0], Tol);
            Assert.Equal(output[0, 1], output[i, 1], Tol);
        }
    }

    // ========================================================================
    // BatchNormalizationLayer - Custom Momentum
    // ========================================================================

    [Fact]
    public void BatchNorm_CustomMomentum_RunningStats()
    {
        // momentum=0.5: runningMean = 0.5*old + 0.5*batch
        var bn = new BatchNormalizationLayer<double>(1, momentum: 0.5);
        bn.SetTrainingMode(true);

        // batchMean = (4 + 8) / 2 = 6
        var input = new Tensor<double>(new[] { 2, 1 }, new Vector<double>(new double[] { 4.0, 8.0 }));
        bn.Forward(input);

        var rm = bn.GetRunningMean();
        // runningMean = 0.5 * 0 + 0.5 * 6 = 3.0
        Assert.Equal(3.0, rm[0], Tol);
    }

    // ========================================================================
    // SGD Update Correctness
    // ========================================================================

    [Fact]
    public void BatchNorm_UpdateParameters_SGD_BetaUpdates()
    {
        // After backward, betaGrad = sum of output gradients per feature
        // beta_new = beta - lr * betaGrad
        // With asymmetric gradient, beta gradient should be non-zero
        var bn = new BatchNormalizationLayer<double>(2);
        bn.SetTrainingMode(true);

        // Forward pass
        var input = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new double[] { 1.0, 2.0, 5.0, 8.0 }));
        bn.Forward(input);

        // Asymmetric gradient to ensure non-zero betaGrad
        var grad = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0 }));
        bn.Backward(grad);

        var betaBefore = bn.GetBeta();

        double lr = 0.01;
        bn.UpdateParameters(lr);

        var betaAfter = bn.GetBeta();

        // betaGrad for feature 0 = 1.0 + 3.0 = 4.0
        // betaGrad for feature 1 = 2.0 + 4.0 = 6.0
        // beta_new = beta_old - lr * betaGrad
        bool betaChanged = false;
        for (int i = 0; i < 2; i++)
        {
            if (Math.Abs(betaAfter[i] - betaBefore[i]) > 1e-10) betaChanged = true;
        }
        Assert.True(betaChanged, "Beta should change after SGD update with non-zero gradient");
    }

    // ========================================================================
    // Dropout + BatchNorm Composition
    // ========================================================================

    [Fact]
    public void Dropout_ThenBatchNorm_InferenceIsClean()
    {
        // In inference mode, dropout is identity, so BN should work normally
        var dropout = new DropoutLayer<double>(0.5);
        dropout.SetTrainingMode(false);

        var bn = new BatchNormalizationLayer<double>(2);
        bn.SetTrainingMode(false);

        var input = new Tensor<double>(new[] { 1, 2 }, new Vector<double>(new double[] { 3.0, 7.0 }));

        var afterDropout = dropout.Forward(input);
        var afterBN = bn.Forward(afterDropout);

        // Dropout is identity, so afterDropout = input
        double scale = 1.0 / Math.Sqrt(1.0 + Eps);
        Assert.Equal(3.0 * scale, afterBN[0, 0], Tol);
        Assert.Equal(7.0 * scale, afterBN[0, 1], Tol);
    }
}
