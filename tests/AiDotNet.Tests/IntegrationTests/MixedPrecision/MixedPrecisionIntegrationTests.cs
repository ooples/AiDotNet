using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.MixedPrecision;
using Xunit;

namespace AiDotNetTests.IntegrationTests.MixedPrecision;

/// <summary>
/// Integration tests for the MixedPrecision module.
/// Tests the full workflow of mixed-precision training including loss scaling,
/// context management, and precision conversions.
/// </summary>
public class MixedPrecisionIntegrationTests
{
    private const double Tolerance = 1e-5;

    #region LossScaler Integration Tests

    [Fact]
    public void LossScaler_FullWorkflow_ScalesAndUnscalesCorrectly()
    {
        // Arrange
        var scaler = new LossScaler<float>(initialScale: 1000.0);
        var loss = 0.001f; // Small loss value

        // Act - Scale the loss
        var scaledLoss = scaler.ScaleLoss(loss);

        // Simulate gradients that would result from scaled loss
        var gradients = new Vector<float>(new[] { 10.0f, 20.0f, 30.0f }); // Scaled gradients

        // Unscale gradients
        scaler.UnscaleGradients(gradients);

        // Assert
        Assert.Equal(1.0f, scaledLoss); // 0.001 * 1000 = 1.0
        Assert.Equal(0.01f, gradients[0], precision: 5); // 10 / 1000 = 0.01
        Assert.Equal(0.02f, gradients[1], precision: 5); // 20 / 1000 = 0.02
        Assert.Equal(0.03f, gradients[2], precision: 5); // 30 / 1000 = 0.03
    }

    [Fact]
    public void LossScaler_TensorGradients_HandlesLargeTensors()
    {
        // Arrange
        var scaler = new LossScaler<float>(initialScale: 100.0);
        var size = 1000;
        var gradients = new Tensor<float>([size], new Vector<float>(size));

        // Fill with scaled gradient values
        for (int i = 0; i < size; i++)
        {
            gradients.SetFlatIndex(i, i * 10.0f); // Values 0, 10, 20, ..., 9990
        }

        // Act
        scaler.UnscaleGradients(gradients);

        // Assert
        for (int i = 0; i < size; i++)
        {
            var expected = i * 0.1f; // Divided by 100
            Assert.Equal(expected, gradients.GetFlatIndexValue(i), precision: 4);
        }
    }

    [Fact]
    public void LossScaler_DynamicScaling_AdaptsToOverflowPatterns()
    {
        // Arrange
        var scaler = new LossScaler<float>(
            initialScale: 1000.0,
            dynamicScaling: true,
            growthInterval: 3,
            growthFactor: 2.0,
            backoffFactor: 0.5
        );

        // Act - Simulate training with occasional overflows
        var normalGradients = new Vector<float>(new[] { 10.0f, 20.0f });
        var overflowGradients = new Vector<float>(new[] { float.NaN, 20.0f });

        // 3 successful updates should trigger growth
        scaler.UnscaleGradientsAndCheck(new Vector<float>(new[] { 10.0f, 20.0f }));
        scaler.UnscaleGradientsAndCheck(new Vector<float>(new[] { 10.0f, 20.0f }));
        scaler.UnscaleGradientsAndCheck(new Vector<float>(new[] { 10.0f, 20.0f }));

        var scaleAfterGrowth = scaler.Scale;

        // 1 overflow should trigger backoff
        scaler.UnscaleGradientsAndCheck(new Vector<float>(new[] { float.NaN }));
        var scaleAfterOverflow = scaler.Scale;

        // Assert
        Assert.Equal(2000.0, scaleAfterGrowth); // 1000 * 2.0
        Assert.Equal(1000.0, scaleAfterOverflow); // 2000 * 0.5
    }

    [Fact]
    public void LossScaler_DoubleType_WorksCorrectly()
    {
        // Arrange
        var scaler = new LossScaler<double>(initialScale: 65536.0);
        var loss = 0.0000001; // Very small loss

        // Act
        var scaledLoss = scaler.ScaleLoss(loss);
        var unscaledGradient = scaler.UnscaleGradient(scaledLoss);

        // Assert - Round-trip should preserve value
        Assert.Equal(loss, unscaledGradient, precision: 10);
    }

    [Fact]
    public void LossScaler_EdgeCase_VerySmallScale()
    {
        // Arrange
        var scaler = new LossScaler<float>(initialScale: 1.0, minScale: 1.0);
        var gradients = new Vector<float>(new[] { 1.0f, 2.0f, 3.0f });

        // Act - Unscale should be identity when scale is 1.0
        scaler.UnscaleGradients(gradients);

        // Assert
        Assert.Equal(1.0f, gradients[0], precision: 5);
        Assert.Equal(2.0f, gradients[1], precision: 5);
        Assert.Equal(3.0f, gradients[2], precision: 5);
    }

    [Fact]
    public void LossScaler_EdgeCase_VeryLargeScale()
    {
        // Arrange
        var scaler = new LossScaler<float>(initialScale: 16777216.0); // 2^24
        var loss = 0.0000001f;

        // Act
        var scaledLoss = scaler.ScaleLoss(loss);

        // Assert - Should be representable
        Assert.False(float.IsInfinity(scaledLoss));
        Assert.False(float.IsNaN(scaledLoss));
        Assert.True(scaledLoss > 0);
    }

    [Fact]
    public void LossScaler_MultipleOverflows_ScaleReachesMinimum()
    {
        // Arrange
        var scaler = new LossScaler<float>(
            initialScale: 100.0,
            dynamicScaling: true,
            backoffFactor: 0.5,
            minScale: 10.0
        );

        // Act - Trigger many overflows
        for (int i = 0; i < 10; i++)
        {
            scaler.UnscaleGradientsAndCheck(new Vector<float>(new[] { float.NaN }));
        }

        // Assert - Should be at minimum
        Assert.Equal(10.0, scaler.Scale);
        Assert.Equal(10, scaler.SkippedUpdates);
    }

    [Fact]
    public void LossScaler_MultipleSuccesses_ScaleReachesMaximum()
    {
        // Arrange
        var scaler = new LossScaler<float>(
            initialScale: 100.0,
            dynamicScaling: true,
            growthInterval: 1,
            growthFactor: 2.0,
            maxScale: 500.0
        );

        // Act - Trigger many successful updates
        for (int i = 0; i < 20; i++)
        {
            scaler.UnscaleGradientsAndCheck(new Vector<float>(new[] { 10.0f }));
        }

        // Assert - Should be at maximum
        Assert.Equal(500.0, scaler.Scale);
    }

    [Fact]
    public void LossScaler_DetectOverflow_Tensor_HandlesNegativeInfinity()
    {
        // Arrange
        var scaler = new LossScaler<float>();
        var gradients = new Tensor<float>([2, 2], new Vector<float>(new[] { 1.0f, 2.0f, float.NegativeInfinity, 4.0f }));

        // Act
        var hasOverflow = scaler.DetectOverflow(gradients);

        // Assert
        Assert.True(hasOverflow);
    }

    [Fact]
    public void LossScaler_Statistics_TrackCorrectly()
    {
        // Arrange
        var scaler = new LossScaler<float>(initialScale: 100.0, growthInterval: 10);

        // Act - Mix of successful and failed updates
        for (int i = 0; i < 5; i++)
        {
            scaler.UnscaleGradientsAndCheck(new Vector<float>(new[] { 10.0f }));
        }
        scaler.UnscaleGradientsAndCheck(new Vector<float>(new[] { float.NaN }));
        scaler.UnscaleGradientsAndCheck(new Vector<float>(new[] { float.PositiveInfinity }));

        // Assert
        Assert.Equal(7, scaler.TotalUpdates);
        Assert.Equal(2, scaler.SkippedUpdates);
        Assert.Equal(2.0 / 7.0, scaler.OverflowRate, precision: 5);
    }

    #endregion

    #region MixedPrecisionConfig Integration Tests

    [Fact]
    public void MixedPrecisionConfig_DefaultValues_FollowNvidiaRecommendations()
    {
        // Arrange & Act
        var config = new MixedPrecisionConfig();

        // Assert - NVIDIA recommended defaults
        Assert.Equal(65536.0, config.InitialLossScale); // 2^16
        Assert.True(config.EnableDynamicScaling);
        Assert.Equal(2000, config.ScaleGrowthInterval);
        Assert.Equal(2.0, config.ScaleGrowthFactor);
        Assert.Equal(0.5, config.ScaleBackoffFactor);
        Assert.Equal(1.0, config.MinLossScale);
        Assert.Equal(16777216.0, config.MaxLossScale); // 2^24
        Assert.True(config.Fp32BatchNorm);
        Assert.True(config.Fp32Loss);
        Assert.True(config.Fp32GradientAccumulation);
    }

    [Fact]
    public void MixedPrecisionConfig_CustomConfiguration_SetsAllProperties()
    {
        // Arrange
        var config = new MixedPrecisionConfig
        {
            InitialLossScale = 4096.0,
            EnableDynamicScaling = false,
            ScaleGrowthInterval = 500,
            ScaleGrowthFactor = 1.5,
            ScaleBackoffFactor = 0.25,
            MinLossScale = 2.0,
            MaxLossScale = 8192.0,
            Fp32BatchNorm = false,
            Fp32Loss = false,
            Fp32GradientAccumulation = false
        };

        // Assert
        Assert.Equal(4096.0, config.InitialLossScale);
        Assert.False(config.EnableDynamicScaling);
        Assert.Equal(500, config.ScaleGrowthInterval);
        Assert.Equal(1.5, config.ScaleGrowthFactor);
        Assert.Equal(0.25, config.ScaleBackoffFactor);
        Assert.Equal(2.0, config.MinLossScale);
        Assert.Equal(8192.0, config.MaxLossScale);
        Assert.False(config.Fp32BatchNorm);
        Assert.False(config.Fp32Loss);
        Assert.False(config.Fp32GradientAccumulation);
    }

    [Fact]
    public void MixedPrecisionConfig_ToString_ContainsAllSettings()
    {
        // Arrange
        var config = new MixedPrecisionConfig();

        // Act
        var str = config.ToString();

        // Assert
        Assert.Contains("MixedPrecisionConfig", str);
        Assert.Contains("65536", str); // InitialScale
        Assert.Contains("Dynamic=True", str);
        Assert.Contains("2000", str); // GrowthInterval
    }

    #endregion

    #region MixedPrecisionContext Integration Tests

    [Fact]
    public void MixedPrecisionContext_Initialize_SetsUpCorrectly()
    {
        // Arrange
        var context = new MixedPrecisionContext();
        var parameters = new Vector<float>(new[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f });

        // Act
        context.Initialize(parameters, "layer1");

        // Assert
        Assert.True(context.IsInitialized);
        Assert.Equal(5, context.ParameterCount);
        Assert.Contains("layer1", context.ParameterNames);
    }

    [Fact]
    public void MixedPrecisionContext_InitializeMultiple_TracksAllParameters()
    {
        // Arrange
        var context = new MixedPrecisionContext();
        var namedParams = new Dictionary<string, Vector<float>>
        {
            ["weights"] = new Vector<float>(new[] { 1.0f, 2.0f, 3.0f }),
            ["biases"] = new Vector<float>(new[] { 0.1f, 0.2f })
        };

        // Act
        context.Initialize(namedParams);

        // Assert
        Assert.True(context.IsInitialized);
        Assert.Equal(5, context.ParameterCount); // 3 + 2
        Assert.Contains("weights", context.ParameterNames);
        Assert.Contains("biases", context.ParameterNames);
    }

    [Fact]
    public void MixedPrecisionContext_CastWeightsToFP16_ConvertsCorrectly()
    {
        // Arrange
        var context = new MixedPrecisionContext();
        var parameters = new Vector<float>(new[] { 1.5f, 2.5f, 3.5f });
        context.Initialize(parameters);

        // Act
        context.CastWeightsToFP16();
        var workingWeights = context.GetWorkingWeights();

        // Assert
        Assert.Equal(3, workingWeights.Length);
        Assert.Equal((Half)1.5f, workingWeights[0]);
        Assert.Equal((Half)2.5f, workingWeights[1]);
        Assert.Equal((Half)3.5f, workingWeights[2]);
    }

    [Fact]
    public void MixedPrecisionContext_GetMasterWeights_ReturnsCopy()
    {
        // Arrange
        var context = new MixedPrecisionContext();
        var parameters = new Vector<float>(new[] { 1.0f, 2.0f, 3.0f });
        context.Initialize(parameters);

        // Act
        var masterWeights = context.GetMasterWeights();

        // Assert
        Assert.Equal(3, masterWeights.Length);
        Assert.Equal(1.0f, masterWeights[0], precision: 5);
        Assert.Equal(2.0f, masterWeights[1], precision: 5);
        Assert.Equal(3.0f, masterWeights[2], precision: 5);
    }

    [Fact]
    public void MixedPrecisionContext_UpdateMasterWeights_AppliesGradientDescent()
    {
        // Arrange
        var context = new MixedPrecisionContext();
        var parameters = new Vector<float>(new[] { 10.0f, 20.0f, 30.0f });
        context.Initialize(parameters);

        var gradients = new Vector<float>(new[] { 1.0f, 2.0f, 3.0f });
        var learningRate = 0.1f;

        // Act
        context.UpdateMasterWeights(gradients, learningRate);
        var updatedWeights = context.GetMasterWeights();

        // Assert - weights -= lr * gradients
        Assert.Equal(9.9f, updatedWeights[0], precision: 5); // 10 - 0.1 * 1 = 9.9
        Assert.Equal(19.8f, updatedWeights[1], precision: 5); // 20 - 0.1 * 2 = 19.8
        Assert.Equal(29.7f, updatedWeights[2], precision: 5); // 30 - 0.1 * 3 = 29.7
    }

    [Fact]
    public void MixedPrecisionContext_PrepareGradientsForUpdate_HandlesValidGradients()
    {
        // Arrange
        var context = new MixedPrecisionContext();
        var parameters = new Vector<float>(new[] { 1.0f, 2.0f, 3.0f });
        context.Initialize(parameters);

        // Simulate FP16 gradients (scaled)
        var gradientsHalf = new Vector<Half>(new[] { (Half)100.0f, (Half)200.0f, (Half)300.0f });

        // Act
        var isValid = context.PrepareGradientsForUpdate(gradientsHalf, out var gradientsFloat);

        // Assert
        Assert.True(isValid);
        Assert.Equal(3, gradientsFloat.Length);
        // The gradients are unscaled by the default scale (65536)
    }

    [Fact]
    public void MixedPrecisionContext_PrepareGradientsForUpdate_DetectsOverflow()
    {
        // Arrange
        var context = new MixedPrecisionContext();
        var parameters = new Vector<float>(new[] { 1.0f, 2.0f, 3.0f });
        context.Initialize(parameters);

        // Simulate FP16 gradients with overflow (Half.NaN or Half.PositiveInfinity)
        var gradientsHalf = new Vector<Half>(new[] { (Half)100.0f, Half.NaN, (Half)300.0f });

        // Act
        var isValid = context.PrepareGradientsForUpdate(gradientsHalf, out _);

        // Assert
        Assert.False(isValid);
    }

    [Fact]
    public void MixedPrecisionContext_Reset_ClearsAllState()
    {
        // Arrange
        var context = new MixedPrecisionContext();
        var parameters = new Vector<float>(new[] { 1.0f, 2.0f, 3.0f });
        context.Initialize(parameters);
        context.CastWeightsToFP16();

        // Act
        context.Reset();

        // Assert
        Assert.False(context.IsInitialized);
        Assert.Equal(0, context.ParameterCount);
        Assert.Empty(context.ParameterNames);
    }

    [Fact]
    public void MixedPrecisionContext_NotInitialized_ThrowsOnCast()
    {
        // Arrange
        var context = new MixedPrecisionContext();

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => context.CastWeightsToFP16());
    }

    [Fact]
    public void MixedPrecisionContext_AlreadyInitialized_ThrowsOnReinitialize()
    {
        // Arrange
        var context = new MixedPrecisionContext();
        var parameters = new Vector<float>(new[] { 1.0f, 2.0f, 3.0f });
        context.Initialize(parameters);

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => context.Initialize(parameters));
    }

    [Fact]
    public void MixedPrecisionContext_GetNonExistentParameter_ThrowsKeyNotFound()
    {
        // Arrange
        var context = new MixedPrecisionContext();
        var parameters = new Vector<float>(new[] { 1.0f, 2.0f, 3.0f });
        context.Initialize(parameters, "layer1");

        // Act & Assert
        Assert.Throws<KeyNotFoundException>(() => context.GetMasterWeights("nonexistent"));
    }

    [Fact]
    public void MixedPrecisionContext_WithCustomConfig_UsesConfigSettings()
    {
        // Arrange
        var config = new MixedPrecisionConfig
        {
            InitialLossScale = 4096.0,
            EnableDynamicScaling = false
        };
        var context = new MixedPrecisionContext(config);

        // Assert
        Assert.Equal(4096.0, context.LossScaler.Scale);
        Assert.False(context.LossScaler.DynamicScaling);
    }

    [Fact]
    public void MixedPrecisionContext_Dispose_ClearsResources()
    {
        // Arrange
        var context = new MixedPrecisionContext();
        var parameters = new Vector<float>(new[] { 1.0f, 2.0f, 3.0f });
        context.Initialize(parameters);

        // Act
        context.Dispose();

        // Assert - Should not throw
        Assert.Empty(context.ParameterNames);
    }

    [Fact]
    public void MixedPrecisionContext_ToString_ContainsStateInfo()
    {
        // Arrange
        var context = new MixedPrecisionContext();
        var parameters = new Vector<float>(new[] { 1.0f, 2.0f, 3.0f });
        context.Initialize(parameters);

        // Act
        var str = context.ToString();

        // Assert
        Assert.Contains("MixedPrecisionContext", str);
        Assert.Contains("Initialized=True", str);
        Assert.Contains("Parameters=3", str);
    }

    #endregion

    #region Full Workflow Integration Tests

    [Fact]
    public void MixedPrecision_FullTrainingIterationWorkflow_CompletesSuccessfully()
    {
        // Arrange
        var config = new MixedPrecisionConfig
        {
            InitialLossScale = 1000.0,
            EnableDynamicScaling = true,
            ScaleGrowthInterval = 5
        };
        var context = new MixedPrecisionContext(config);

        // Simulate model parameters
        var parameters = new Vector<float>(new[] { 0.5f, 0.5f, 0.5f, 0.5f });
        context.Initialize(parameters);

        // Simulate multiple training iterations
        var iterations = 10;
        var successfulUpdates = 0;

        for (int i = 0; i < iterations; i++)
        {
            // Step 1: Cast weights to FP16
            context.CastWeightsToFP16();
            var workingWeights = context.GetWorkingWeights();
            Assert.Equal(4, workingWeights.Length);

            // Step 2: Simulate forward pass (just get working weights)
            // In real scenario, this would be done by the network

            // Step 3: Simulate loss computation (scaled)
            var loss = 0.001f;
            var scaledLoss = context.LossScaler.ScaleLoss(loss);

            // Step 4: Simulate backward pass producing gradients
            var gradientsHalf = new Vector<Half>(new[]
            {
                (Half)(0.1f * (float)context.LossScaler.Scale),
                (Half)(0.2f * (float)context.LossScaler.Scale),
                (Half)(0.1f * (float)context.LossScaler.Scale),
                (Half)(0.2f * (float)context.LossScaler.Scale)
            });

            // Step 5: Prepare gradients for update
            var isValid = context.PrepareGradientsForUpdate(gradientsHalf, out var gradientsFloat);

            // Step 6: Update master weights if valid
            if (isValid)
            {
                context.UpdateMasterWeights(gradientsFloat, learningRate: 0.01f);
                successfulUpdates++;
            }
        }

        // Assert - All updates should succeed with normal gradients
        Assert.Equal(iterations, successfulUpdates);

        // Weights should have changed
        var finalWeights = context.GetMasterWeights();
        Assert.NotEqual(0.5f, finalWeights[0]); // Should have been updated
    }

    [Fact]
    public void MixedPrecision_SimulatedOverflowRecovery_ScaleAdjustsCorrectly()
    {
        // Arrange
        var config = new MixedPrecisionConfig
        {
            InitialLossScale = 1000.0,
            EnableDynamicScaling = true,
            ScaleGrowthInterval = 3,
            ScaleGrowthFactor = 2.0,
            ScaleBackoffFactor = 0.5
        };
        var context = new MixedPrecisionContext(config);
        var parameters = new Vector<float>(new[] { 1.0f, 2.0f, 3.0f });
        context.Initialize(parameters);

        var scaleHistory = new List<double> { context.LossScaler.Scale };

        // Simulate training with mixed success/overflow pattern
        var gradientPatterns = new[]
        {
            new Vector<Half>(new[] { (Half)100.0f, (Half)200.0f, (Half)300.0f }), // Normal
            new Vector<Half>(new[] { (Half)100.0f, (Half)200.0f, (Half)300.0f }), // Normal
            new Vector<Half>(new[] { (Half)100.0f, (Half)200.0f, (Half)300.0f }), // Normal - triggers growth
            new Vector<Half>(new[] { Half.NaN, (Half)200.0f, (Half)300.0f }), // Overflow - triggers backoff
            new Vector<Half>(new[] { (Half)100.0f, (Half)200.0f, (Half)300.0f }), // Normal
            new Vector<Half>(new[] { (Half)100.0f, (Half)200.0f, (Half)300.0f }), // Normal
            new Vector<Half>(new[] { (Half)100.0f, (Half)200.0f, (Half)300.0f }), // Normal - triggers growth
        };

        foreach (var gradients in gradientPatterns)
        {
            context.PrepareGradientsForUpdate(gradients, out _);
            scaleHistory.Add(context.LossScaler.Scale);
        }

        // Assert - Verify scale adjustments
        Assert.Equal(1000.0, scaleHistory[0]); // Initial
        Assert.Equal(2000.0, scaleHistory[3]); // After 3 successes, growth
        Assert.Equal(1000.0, scaleHistory[4]); // After overflow, backoff
        Assert.Equal(2000.0, scaleHistory[7]); // After 3 more successes, growth again
    }

    [Fact]
    public void MixedPrecision_FP32ToFP16Precision_LossesSomeAccuracy()
    {
        // Arrange
        var context = new MixedPrecisionContext();

        // Use values that will have precision loss in FP16
        var parameters = new Vector<float>(new[]
        {
            1.0000001f,  // Very small difference from 1.0
            123.456789f, // Many decimal places
            0.00012345f  // Small value with precision
        });
        context.Initialize(parameters);

        // Act
        context.CastWeightsToFP16();
        var workingWeights = context.GetWorkingWeights();

        // Assert - FP16 has less precision
        // These values should be close but not exact due to FP16 precision limits
        Assert.Equal(1.0f, (float)workingWeights[0], precision: 2); // FP16 can't represent 1.0000001
        // 123.456789f becomes 123.4375f in FP16 (Half has ~3.3 decimal digits precision)
        Assert.Equal(123.4f, (float)workingWeights[1], precision: 0); // 123.4375 rounds to 123
        Assert.True((float)workingWeights[2] > 0); // Should still be representable
    }

    [Fact]
    public void MixedPrecision_MultipleParameterGroups_UpdatesIndependently()
    {
        // Arrange
        var context = new MixedPrecisionContext();
        var namedParams = new Dictionary<string, Vector<float>>
        {
            ["encoder"] = new Vector<float>(new[] { 1.0f, 2.0f }),
            ["decoder"] = new Vector<float>(new[] { 3.0f, 4.0f, 5.0f })
        };
        context.Initialize(namedParams);

        // Act - Update only encoder weights
        var encoderGradients = new Vector<float>(new[] { 0.1f, 0.2f });
        context.UpdateMasterWeights(encoderGradients, 1.0f, "encoder");

        // Assert
        var encoderWeights = context.GetMasterWeights("encoder");
        var decoderWeights = context.GetMasterWeights("decoder");

        Assert.Equal(0.9f, encoderWeights[0], precision: 5); // Updated
        Assert.Equal(1.8f, encoderWeights[1], precision: 5); // Updated
        Assert.Equal(3.0f, decoderWeights[0], precision: 5); // Unchanged
        Assert.Equal(4.0f, decoderWeights[1], precision: 5); // Unchanged
        Assert.Equal(5.0f, decoderWeights[2], precision: 5); // Unchanged
    }

    [Fact]
    public void MixedPrecision_GradientMismatch_ThrowsArgumentException()
    {
        // Arrange
        var context = new MixedPrecisionContext();
        var parameters = new Vector<float>(new[] { 1.0f, 2.0f, 3.0f });
        context.Initialize(parameters);

        var wrongSizeGradients = new Vector<float>(new[] { 0.1f, 0.2f }); // Size mismatch

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            context.UpdateMasterWeights(wrongSizeGradients, 0.1f));
    }

    #endregion

    #region Edge Cases and Numerical Stability

    [Fact]
    public void LossScaler_ZeroLoss_HandlesGracefully()
    {
        // Arrange
        var scaler = new LossScaler<float>(initialScale: 65536.0);
        var loss = 0.0f;

        // Act
        var scaledLoss = scaler.ScaleLoss(loss);

        // Assert
        Assert.Equal(0.0f, scaledLoss);
    }

    [Fact]
    public void LossScaler_NegativeLoss_ScalesCorrectly()
    {
        // Arrange
        var scaler = new LossScaler<float>(initialScale: 100.0);
        var loss = -0.5f;

        // Act
        var scaledLoss = scaler.ScaleLoss(loss);

        // Assert
        Assert.Equal(-50.0f, scaledLoss);
    }

    [Fact]
    public void LossScaler_ZeroGradients_UnscalesToZero()
    {
        // Arrange
        var scaler = new LossScaler<float>(initialScale: 65536.0);
        var gradients = new Vector<float>(new[] { 0.0f, 0.0f, 0.0f });

        // Act
        scaler.UnscaleGradients(gradients);

        // Assert
        Assert.Equal(0.0f, gradients[0]);
        Assert.Equal(0.0f, gradients[1]);
        Assert.Equal(0.0f, gradients[2]);
    }

    [Fact]
    public void MixedPrecisionContext_EmptyParameters_HandlesCorrectly()
    {
        // Arrange
        var context = new MixedPrecisionContext();
        var parameters = new Vector<float>(0); // Empty

        // Act
        context.Initialize(parameters);
        context.CastWeightsToFP16();
        var workingWeights = context.GetWorkingWeights();

        // Assert
        Assert.True(context.IsInitialized);
        Assert.Equal(0, context.ParameterCount);
        Assert.Equal(0, workingWeights.Length);
    }

    [Fact]
    public void LossScaler_TensorWithAllNaN_DetectsOverflow()
    {
        // Arrange
        var scaler = new LossScaler<float>();
        var gradients = new Tensor<float>([3], new Vector<float>(new[] { float.NaN, float.NaN, float.NaN }));

        // Act
        var hasOverflow = scaler.DetectOverflow(gradients);

        // Assert
        Assert.True(hasOverflow);
    }

    [Fact]
    public void LossScaler_EmptyTensor_NoOverflow()
    {
        // Arrange
        var scaler = new LossScaler<float>();
        var gradients = new Tensor<float>([0], new Vector<float>(0));

        // Act
        var hasOverflow = scaler.DetectOverflow(gradients);

        // Assert
        Assert.False(hasOverflow);
    }

    [Fact]
    public void LossScaler_EmptyVector_NoOverflow()
    {
        // Arrange
        var scaler = new LossScaler<float>();
        var gradients = new Vector<float>(0);

        // Act
        var hasOverflow = scaler.DetectOverflow(gradients);

        // Assert
        Assert.False(hasOverflow);
    }

    #endregion

    #region FP8 Type Tests

    [Fact]
    public void Float8E4M3_Conversion_RoundTripsCorrectly()
    {
        // Arrange - Use smaller values where E4M3 has better precision
        var testValues = new[] { 0f, 1f, -1f, 0.5f, 2.5f, 8f, -4f };

        foreach (var value in testValues)
        {
            // Act
            var e4m3 = Float8E4M3.FromFloat(value);
            var roundTrip = e4m3.ToFloat();

            // Assert - E4M3 has limited precision (3 mantissa bits), allow some tolerance
            // Relative error can be up to ~12.5% for E4M3
            var tolerance = Math.Max(1.0f, Math.Abs(value) * 0.15f);
            Assert.True(Math.Abs(value - roundTrip) <= tolerance,
                $"Value {value} became {roundTrip}, exceeding tolerance {tolerance}");
        }
    }

    [Fact]
    public void Float8E4M3_Clamps_ValuesOutOfRange()
    {
        // Arrange
        const float maxE4M3 = 448f;

        // Act
        var tooLarge = Float8E4M3.FromFloat(1000f);
        var tooLargeBack = tooLarge.ToFloat();

        // Assert - Should be clamped to max
        Assert.True(tooLargeBack <= maxE4M3);
    }

    [Fact]
    public void Float8E4M3_HandlesZero()
    {
        // Act
        var zero = Float8E4M3.FromFloat(0f);
        var negZero = Float8E4M3.FromFloat(-0f);

        // Assert
        Assert.True(zero.IsZero);
        Assert.Equal(0f, zero.ToFloat());
    }

    [Fact]
    public void Float8E4M3_HandlesNaN()
    {
        // Act
        var nan = Float8E4M3.FromFloat(float.NaN);

        // Assert
        Assert.True(nan.IsNaN);
        Assert.True(float.IsNaN(nan.ToFloat()));
    }

    [Fact]
    public void Float8E5M2_Conversion_RoundTripsCorrectly()
    {
        // Arrange
        var testValues = new[] { 0f, 1f, -1f, 0.5f, 10f, 1000f, -500f };

        foreach (var value in testValues)
        {
            // Act
            var e5m2 = Float8E5M2.FromFloat(value);
            var roundTrip = e5m2.ToFloat();

            // Assert - E5M2 has even less precision, allow more tolerance
            Assert.True(Math.Abs(value - roundTrip) < Math.Max(1.0, Math.Abs(value) * 0.5));
        }
    }

    [Fact]
    public void Float8E5M2_HasLargerRange_ThanE4M3()
    {
        // Arrange
        const float largeValue = 50000f;

        // Act
        var e4m3 = Float8E4M3.FromFloat(largeValue);
        var e5m2 = Float8E5M2.FromFloat(largeValue);

        // Assert - E4M3 should clamp, E5M2 should represent (or be close)
        Assert.True(e4m3.ToFloat() < largeValue); // E4M3 max is 448
        Assert.True(Math.Abs(e5m2.ToFloat() - largeValue) < 10000); // E5M2 can represent up to 57344
    }

    [Fact]
    public void Float8E5M2_HandlesInfinity()
    {
        // Act
        var posInf = Float8E5M2.FromFloat(float.PositiveInfinity);
        var negInf = Float8E5M2.FromFloat(float.NegativeInfinity);

        // Assert
        Assert.True(posInf.IsInfinity);
        Assert.True(negInf.IsInfinity);
        Assert.True(float.IsPositiveInfinity(posInf.ToFloat()));
        Assert.True(float.IsNegativeInfinity(negInf.ToFloat()));
    }

    [Fact]
    public void Float8Extensions_BulkConversion_Works()
    {
        // Arrange
        var values = new[] { 1f, 2f, 3f, 4f, 5f };

        // Act
        var e4m3Array = values.ToE4M3();
        var backToFloat = e4m3Array.ToFloatArray();

        // Assert
        Assert.Equal(values.Length, e4m3Array.Length);
        Assert.Equal(values.Length, backToFloat.Length);
        for (int i = 0; i < values.Length; i++)
        {
            Assert.Equal(values[i], backToFloat[i], precision: 0);
        }
    }

    [Fact]
    public void Float8_E4M3ToE5M2Conversion_Works()
    {
        // Arrange
        var e4m3 = Float8E4M3.FromFloat(10f);

        // Act
        var e5m2 = e4m3.ToE5M2();
        var backToE4M3 = e5m2.ToE4M3();

        // Assert - Should preserve value through conversion
        Assert.Equal(e4m3.ToFloat(), backToE4M3.ToFloat(), precision: 0);
    }

    #endregion

    #region Layer Precision Policy Tests

    [Fact]
    public void LayerPrecisionPolicy_DefaultPrecision_AppliedCorrectly()
    {
        // Arrange
        var policy = new LayerPrecisionPolicy(MixedPrecisionType.FP16);

        // Act
        var precision = policy.GetPrecision("SomeRandomLayer");

        // Assert
        Assert.Equal(MixedPrecisionType.FP16, precision);
    }

    [Fact]
    public void LayerPrecisionPolicy_ExactMatch_TakesPrecedence()
    {
        // Arrange
        var policy = new LayerPrecisionPolicy(MixedPrecisionType.FP16)
            .SetPrecision("layer1", MixedPrecisionType.None);

        // Act
        var precision = policy.GetPrecision("layer1");

        // Assert
        Assert.Equal(MixedPrecisionType.None, precision);
    }

    [Fact]
    public void LayerPrecisionPolicy_PatternMatch_Works()
    {
        // Arrange
        var policy = new LayerPrecisionPolicy(MixedPrecisionType.FP8_Hybrid)
            .KeepInFP32("Norm");

        // Act & Assert
        Assert.Equal(MixedPrecisionType.None, policy.GetPrecision("LayerNorm"));
        Assert.Equal(MixedPrecisionType.None, policy.GetPrecision("BatchNorm"));
        Assert.Equal(MixedPrecisionType.None, policy.GetPrecision("RMSNorm"));
        Assert.Equal(MixedPrecisionType.FP8_Hybrid, policy.GetPrecision("Linear"));
    }

    [Fact]
    public void LayerPrecisionPolicy_ForFP8_ExcludesNormalization()
    {
        // Arrange
        var policy = LayerPrecisionPolicy.ForFP8();

        // Act & Assert
        Assert.True(policy.ShouldSkipMixedPrecision("LayerNorm"));
        Assert.True(policy.ShouldSkipMixedPrecision("BatchNorm1d"));
        Assert.True(policy.ShouldUseHigherPrecision("Softmax"));
        Assert.False(policy.ShouldSkipMixedPrecision("Linear"));
    }

    [Fact]
    public void LayerPrecisionPolicy_ForFP8Transformers_ConfiguredCorrectly()
    {
        // Arrange
        var policy = LayerPrecisionPolicy.ForFP8Transformers();

        // Act & Assert
        Assert.True(policy.ShouldSkipMixedPrecision("LayerNorm"));
        Assert.True(policy.ShouldUseHigherPrecision("self_attn"));
        Assert.True(policy.ShouldUseHigherPrecision("Embedding"));
        Assert.False(policy.ShouldSkipMixedPrecision("mlp.fc1")); // MLP stays in FP8
    }

    #endregion

    #region MixedPrecisionConfig Factory Tests

    [Fact]
    public void MixedPrecisionConfig_ForFP8_HasCorrectSettings()
    {
        // Act
        var config = MixedPrecisionConfig.ForFP8();

        // Assert
        Assert.Equal(MixedPrecisionType.FP8_Hybrid, config.PrecisionType);
        Assert.Equal(MixedPrecisionType.FP8_E4M3, config.FP8ForwardFormat);
        Assert.Equal(MixedPrecisionType.FP8_E5M2, config.FP8BackwardFormat);
        Assert.True(config.FP8PerTensorScaling);
        Assert.True(config.EnableDynamicScaling);
        Assert.Contains("LayerNorm", config.FP8ExcludedLayers);
    }

    [Fact]
    public void MixedPrecisionConfig_ForBF16_HasCorrectSettings()
    {
        // Act
        var config = MixedPrecisionConfig.ForBF16();

        // Assert
        Assert.Equal(MixedPrecisionType.BF16, config.PrecisionType);
        Assert.False(config.EnableDynamicScaling); // BF16 doesn't need loss scaling
        Assert.Equal(1.0, config.InitialLossScale);
    }

    [Fact]
    public void MixedPrecisionConfig_Conservative_HasSaferSettings()
    {
        // Act
        var config = MixedPrecisionConfig.Conservative();

        // Assert
        Assert.Equal(4096.0, config.InitialLossScale); // Lower than default
        Assert.Equal(4000, config.ScaleGrowthInterval); // More conservative
        Assert.Equal(0.25, config.ScaleBackoffFactor); // More aggressive backoff
        Assert.True(config.Fp32BatchNorm);
        Assert.True(config.Fp32GradientAccumulation);
    }

    #endregion
}
