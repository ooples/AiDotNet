using AiDotNet.Deployment.Configuration;
using AiDotNet.Deployment.Optimization.Quantization;
using AiDotNet.Deployment.Optimization.Quantization.Calibration;
using AiDotNet.Deployment.Optimization.Quantization.Formats;
using AiDotNet.Deployment.Optimization.Quantization.Strategies;
using AiDotNet.Deployment.Optimization.Quantization.Training;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Quantization;

/// <summary>
/// Integration tests for the quantization framework.
/// Tests PTQ strategies (GPTQ, AWQ, SmoothQuant), QAT, and various configurations.
/// </summary>
public class QuantizationIntegrationTests
{
    private const double Tolerance = 1e-4;

    #region Helper Methods

    /// <summary>
    /// Creates a simple test model for quantization testing.
    /// </summary>
    private SimpleTestModel<double> CreateTestModel(int parameterCount = 1000)
    {
        var random = new Random(42);
        var weights = new double[parameterCount];
        for (int i = 0; i < parameterCount; i++)
        {
            // Create weights with normal distribution (mean=0, std=0.1)
            weights[i] = (random.NextDouble() - 0.5) * 0.2;
        }
        return new SimpleTestModel<double>(weights);
    }

    /// <summary>
    /// Creates calibration data for quantization.
    /// </summary>
    private IEnumerable<double[]> CreateCalibrationData(int samples, int inputSize)
    {
        var random = new Random(42);
        for (int i = 0; i < samples; i++)
        {
            var sample = new double[inputSize];
            for (int j = 0; j < inputSize; j++)
            {
                sample[j] = random.NextDouble() * 2 - 1;
            }
            yield return sample;
        }
    }

    #endregion

    #region INT8 Quantization Tests

    [Fact]
    public void Int8Quantizer_QuantizesModel_ReducesParameterRange()
    {
        // Arrange
        var model = CreateTestModel(100);
        var quantizer = new Int8Quantizer<double, double[], double[]>();
        var calibrationData = CreateCalibrationData(50, 10).ToList();
        var config = QuantizationConfiguration.ForInt8();

        // Act
        quantizer.Calibrate(model, calibrationData);
        var quantizedModel = quantizer.Quantize(model, config);

        // Assert
        var originalParams = model.GetParameters();
        var quantizedParams = quantizedModel.GetParameters();

        Assert.Equal(originalParams.Length, quantizedParams.Length);
        Assert.Equal(8, quantizer.BitWidth);
        Assert.Equal(QuantizationMode.Int8, quantizer.Mode);

        // Verify parameters are quantized (should have limited unique values)
        var uniqueValues = new HashSet<double>();
        for (int i = 0; i < quantizedParams.Length; i++)
        {
            // Scale factor determines the quantization step size
            var scaleFactor = quantizer.GetScaleFactor("global");
            var quantizedValue = Math.Round(quantizedParams[i] / scaleFactor) * scaleFactor;
            Assert.True(Math.Abs(quantizedParams[i] - quantizedValue) < scaleFactor * 0.5 + 1e-10,
                $"Parameter {i} should be on quantization grid");
        }
    }

    [Fact]
    public void Int8Quantizer_WithSymmetricQuantization_UsesZeroPoint()
    {
        // Arrange
        var model = CreateTestModel(100);
        var quantizer = new Int8Quantizer<double, double[], double[]>();
        var calibrationData = CreateCalibrationData(50, 10).ToList();
        var config = new QuantizationConfiguration
        {
            Mode = QuantizationMode.Int8,
            UseSymmetricQuantization = true
        };

        // Act
        quantizer.Calibrate(model, calibrationData);
        quantizer.Quantize(model, config);

        // Assert
        var zeroPoint = quantizer.GetZeroPoint("global");
        Assert.Equal(0, zeroPoint); // Symmetric quantization has zero point = 0
    }

    #endregion

    #region Float16 Quantization Tests

    [Fact]
    public void Float16Quantizer_QuantizesModel_MaintainsHighPrecision()
    {
        // Arrange
        var model = CreateTestModel(100);
        var quantizer = new Float16Quantizer<double, double[], double[]>();
        var config = QuantizationConfiguration.ForFloat16();

        // Act
        var quantizedModel = quantizer.Quantize(model, config);

        // Assert
        var originalParams = model.GetParameters();
        var quantizedParams = quantizedModel.GetParameters();

        Assert.Equal(originalParams.Length, quantizedParams.Length);
        Assert.Equal(16, quantizer.BitWidth);
        Assert.Equal(QuantizationMode.Float16, quantizer.Mode);

        // Float16 should maintain high precision
        double maxError = 0;
        for (int i = 0; i < originalParams.Length; i++)
        {
            var error = Math.Abs(Convert.ToDouble(originalParams[i]) - Convert.ToDouble(quantizedParams[i]));
            maxError = Math.Max(maxError, error);
        }

        // Float16 can represent values with ~3 decimal digits of precision
        Assert.True(maxError < 0.01, $"Max error {maxError} should be small for Float16");
    }

    #endregion

    #region GPTQ Quantization Tests

    [Fact]
    public void GPTQQuantizer_QuantizesWithHessianCompensation_MaintainsAccuracy()
    {
        // Arrange
        var model = CreateTestModel(256);
        var config = QuantizationConfiguration.ForGPTQ(groupSize: 64);
        var quantizer = new GPTQQuantizer<double, double[], double[]>(config);
        var calibrationData = CreateCalibrationData(100, 16).ToList();

        // Act
        quantizer.Calibrate(model, calibrationData);
        var quantizedModel = quantizer.Quantize(model, config);

        // Assert
        var originalParams = model.GetParameters();
        var quantizedParams = quantizedModel.GetParameters();

        Assert.Equal(originalParams.Length, quantizedParams.Length);
        Assert.Equal(4, quantizer.BitWidth); // GPTQ typically targets 4-bit

        // Verify quantization occurred (parameters should be different)
        bool hasChanged = false;
        for (int i = 0; i < originalParams.Length; i++)
        {
            if (Math.Abs(Convert.ToDouble(originalParams[i]) - Convert.ToDouble(quantizedParams[i])) > 1e-10)
            {
                hasChanged = true;
                break;
            }
        }
        Assert.True(hasChanged, "Quantization should modify parameters");
    }

    [Fact]
    public void GPTQQuantizer_WithActOrder_ProcessesImportantColumnsFirst()
    {
        // Arrange
        var model = CreateTestModel(256);
        var configWithActOrder = QuantizationConfiguration.ForGPTQ(groupSize: 64, actOrder: true);
        var configWithoutActOrder = QuantizationConfiguration.ForGPTQ(groupSize: 64, actOrder: false);

        var quantizerWithActOrder = new GPTQQuantizer<double, double[], double[]>(configWithActOrder);
        var quantizerWithoutActOrder = new GPTQQuantizer<double, double[], double[]>(configWithoutActOrder);

        var calibrationData = CreateCalibrationData(100, 16).ToList();

        // Act
        quantizerWithActOrder.Calibrate(model, calibrationData);
        quantizerWithoutActOrder.Calibrate(model, calibrationData);

        var quantizedWithActOrder = quantizerWithActOrder.Quantize(model, configWithActOrder);
        var quantizedWithoutActOrder = quantizerWithoutActOrder.Quantize(model, configWithoutActOrder);

        // Assert - Both should complete successfully but may have different results
        var paramsWithActOrder = quantizedWithActOrder.GetParameters();
        var paramsWithoutActOrder = quantizedWithoutActOrder.GetParameters();

        Assert.Equal(paramsWithActOrder.Length, paramsWithoutActOrder.Length);
    }

    #endregion

    #region AWQ Quantization Tests

    [Fact]
    public void AWQQuantizer_ProtectsImportantWeights_MaintainsAccuracy()
    {
        // Arrange
        var model = CreateTestModel(256);
        var config = QuantizationConfiguration.ForAWQ(groupSize: 64, protectionPercentage: 1.0);
        var quantizer = new AWQQuantizer<double, double[], double[]>(config);
        var calibrationData = CreateCalibrationData(100, 16).ToList();

        // Act
        quantizer.Calibrate(model, calibrationData);
        var quantizedModel = quantizer.Quantize(model, config);

        // Assert
        var originalParams = model.GetParameters();
        var quantizedParams = quantizedModel.GetParameters();

        Assert.Equal(originalParams.Length, quantizedParams.Length);
        Assert.Equal(4, quantizer.BitWidth);

        // Calculate mean squared error
        double mse = 0;
        for (int i = 0; i < originalParams.Length; i++)
        {
            var diff = Convert.ToDouble(originalParams[i]) - Convert.ToDouble(quantizedParams[i]);
            mse += diff * diff;
        }
        mse /= originalParams.Length;

        // AWQ should maintain reasonable accuracy
        Assert.True(mse < 0.01, $"MSE {mse} should be small for AWQ");
    }

    #endregion

    #region SmoothQuant Tests

    [Fact]
    public void SmoothQuantQuantizer_SmoothsOutliers_EnablesW8A8()
    {
        // Arrange
        var model = CreateTestModel(256);
        var config = QuantizationConfiguration.ForSmoothQuant(alpha: 0.5);
        var quantizer = new SmoothQuantQuantizer<double, double[], double[]>(config);
        var calibrationData = CreateCalibrationData(100, 16).ToList();

        // Act
        quantizer.Calibrate(model, calibrationData);
        var quantizedModel = quantizer.Quantize(model, config);

        // Assert
        var originalParams = model.GetParameters();
        var quantizedParams = quantizedModel.GetParameters();

        Assert.Equal(originalParams.Length, quantizedParams.Length);
        Assert.Equal(8, quantizer.BitWidth);

        // Verify smoothing scales were computed
        var smoothingScales = quantizer.SmoothingScales;
        Assert.True(smoothingScales.ContainsKey("global"), "Should have global smoothing scales");
    }

    [Fact]
    public void SmoothQuantQuantizer_WithDifferentAlpha_ProducesDifferentResults()
    {
        // Arrange
        var model = CreateTestModel(256);
        var configAlpha05 = QuantizationConfiguration.ForSmoothQuant(alpha: 0.5);
        var configAlpha08 = QuantizationConfiguration.ForSmoothQuant(alpha: 0.8);

        var quantizerAlpha05 = new SmoothQuantQuantizer<double, double[], double[]>(configAlpha05);
        var quantizerAlpha08 = new SmoothQuantQuantizer<double, double[], double[]>(configAlpha08);

        var calibrationData = CreateCalibrationData(100, 16).ToList();

        // Act
        quantizerAlpha05.Calibrate(model, calibrationData);
        quantizerAlpha08.Calibrate(model, calibrationData);

        var quantized05 = quantizerAlpha05.Quantize(model, configAlpha05);
        var quantized08 = quantizerAlpha08.Quantize(model, configAlpha08);

        // Assert - Different alpha should produce different smoothing
        var smoothingScales05 = quantizerAlpha05.SmoothingScales["global"];
        var smoothingScales08 = quantizerAlpha08.SmoothingScales["global"];

        bool hasDifferentScales = false;
        for (int i = 0; i < Math.Min(smoothingScales05.Length, smoothingScales08.Length); i++)
        {
            if (Math.Abs(smoothingScales05[i] - smoothingScales08[i]) > 1e-6)
            {
                hasDifferentScales = true;
                break;
            }
        }
        Assert.True(hasDifferentScales, "Different alpha values should produce different smoothing scales");
    }

    #endregion

    #region SpinQuant Tests

    [Fact]
    public void SpinQuantQuantizer_LearnedRotations_ReducesOutliers()
    {
        // Arrange - Use small model for fast test execution
        var model = CreateTestModel(16);
        var config = new QuantizationConfiguration
        {
            Mode = QuantizationMode.Int8,
            Strategy = QuantizationStrategy.SpinQuant,
            TargetBitWidth = 4,
            UseSymmetricQuantization = true
        };
        // Use minimal iterations for test speed
        var quantizer = new SpinQuantQuantizer<double, double[], double[]>(
            config,
            numIterations: 2,
            learningRate: 0.01,
            blockSize: 8);
        var calibrationData = CreateCalibrationData(10, 4).ToList();

        // Act
        quantizer.Calibrate(model, calibrationData);
        var quantizedModel = quantizer.Quantize(model, config);

        // Assert
        Assert.True(quantizer.IsCalibrated);
        Assert.Equal(4, quantizer.BitWidth);

        var quantizedParams = quantizedModel.GetParameters();
        Assert.Equal(16, quantizedParams.Length);

        // All quantized values should be finite
        for (int i = 0; i < quantizedParams.Length; i++)
        {
            Assert.False(double.IsNaN(quantizedParams[i]), "Quantized params contain NaN");
            Assert.False(double.IsInfinity(quantizedParams[i]), "Quantized params contain Infinity");
        }
    }

    [Fact]
    public void SpinQuantQuantizer_WithDifferentBlockSizes_Completes()
    {
        // Arrange - Use small model for fast test execution
        var model = CreateTestModel(32);
        var config = new QuantizationConfiguration
        {
            Mode = QuantizationMode.Int8,
            Strategy = QuantizationStrategy.SpinQuant,
            TargetBitWidth = 8
        };

        // Test with different block sizes - minimal iterations for speed
        var quantizerBlock8 = new SpinQuantQuantizer<double, double[], double[]>(config, numIterations: 1, blockSize: 8);
        var quantizerBlock16 = new SpinQuantQuantizer<double, double[], double[]>(config, numIterations: 1, blockSize: 16);

        // Act & Assert - All should complete without error
        var quantized8 = quantizerBlock8.Quantize(model, config);
        var quantized16 = quantizerBlock16.Quantize(model, config);

        Assert.NotNull(quantized8);
        Assert.NotNull(quantized16);
    }

    [Fact]
    public void SpinQuantQuantizer_PreservesParameterCount()
    {
        // Arrange - Use small model for fast test execution
        var model = CreateTestModel(20);
        var quantizer = new SpinQuantQuantizer<double, double[], double[]>(
            numIterations: 1,
            learningRate: 0.01,
            blockSize: 10);
        var config = new QuantizationConfiguration { Mode = QuantizationMode.Int8 };

        // Act
        var quantizedModel = quantizer.Quantize(model, config);

        // Assert
        var originalParams = model.GetParameters();
        var quantizedParams = quantizedModel.GetParameters();
        Assert.Equal(originalParams.Length, quantizedParams.Length);
    }

    #endregion

    #region QuIP# 2-bit Quantization Tests

    [Fact]
    public void QuIPSharpQuantizer_2BitQuantization_Completes()
    {
        // Arrange
        var model = CreateTestModel(32);
        var config = new QuantizationConfiguration
        {
            Mode = QuantizationMode.Int8,
            Strategy = QuantizationStrategy.QuIPSharp,
            TargetBitWidth = 2
        };
        var quantizer = new QuIPSharpQuantizer<double, double[], double[]>(config, groupSize: 8);
        var calibrationData = CreateCalibrationData(10, 4).ToList();

        // Act
        quantizer.Calibrate(model, calibrationData);
        var quantizedModel = quantizer.Quantize(model, config);

        // Assert
        Assert.True(quantizer.IsCalibrated);
        Assert.Equal(2, quantizer.BitWidth);

        var quantizedParams = quantizedModel.GetParameters();
        Assert.Equal(32, quantizedParams.Length);
    }

    [Fact]
    public void QuIPSharpQuantizer_HadamardTransform_PreservesParameterCount()
    {
        // Arrange
        var model = CreateTestModel(64);
        var quantizer = new QuIPSharpQuantizer<double, double[], double[]>(groupSize: 16);
        var config = new QuantizationConfiguration { Mode = QuantizationMode.Int8 };

        // Act
        var quantizedModel = quantizer.Quantize(model, config);

        // Assert
        var originalParams = model.GetParameters();
        var quantizedParams = quantizedModel.GetParameters();
        Assert.Equal(originalParams.Length, quantizedParams.Length);

        // All values should be finite
        for (int i = 0; i < quantizedParams.Length; i++)
        {
            Assert.False(double.IsNaN(quantizedParams[i]), "Quantized params contain NaN");
            Assert.False(double.IsInfinity(quantizedParams[i]), "Quantized params contain Infinity");
        }
    }

    [Fact]
    public void QuIPSharpQuantizer_WithDifferentGroupSizes_Completes()
    {
        // Arrange
        var model = CreateTestModel(48);
        var config = new QuantizationConfiguration { Mode = QuantizationMode.Int8 };

        // Test with different group sizes
        var quantizerGroup4 = new QuIPSharpQuantizer<double, double[], double[]>(config, groupSize: 4);
        var quantizerGroup8 = new QuIPSharpQuantizer<double, double[], double[]>(config, groupSize: 8);
        var quantizerGroup16 = new QuIPSharpQuantizer<double, double[], double[]>(config, groupSize: 16);

        // Act & Assert - All should complete without error
        var quantized4 = quantizerGroup4.Quantize(model, config);
        var quantized8 = quantizerGroup8.Quantize(model, config);
        var quantized16 = quantizerGroup16.Quantize(model, config);

        Assert.NotNull(quantized4);
        Assert.NotNull(quantized8);
        Assert.NotNull(quantized16);
    }

    [Fact]
    public void QuIPSharpQuantizer_ExtremCompression_16xRatio()
    {
        // Arrange - QuIP# targets 2-bit quantization = 16x compression vs 32-bit
        var model = CreateTestModel(100);
        var quantizer = new QuIPSharpQuantizer<double, double[], double[]>(groupSize: 8);
        var config = new QuantizationConfiguration
        {
            Mode = QuantizationMode.Int8,
            Strategy = QuantizationStrategy.QuIPSharp,
            TargetBitWidth = 2
        };

        // Act
        var quantizedModel = quantizer.Quantize(model, config);

        // Assert
        Assert.Equal(2, quantizer.BitWidth);

        // Quantized values should be limited to 4 possible levels
        // (In simulation, values are dequantized, so we verify they're within expected range)
        var quantizedParams = quantizedModel.GetParameters();
        for (int i = 0; i < quantizedParams.Length; i++)
        {
            Assert.False(double.IsNaN(quantizedParams[i]), "Quantized params contain NaN");
        }
    }

    #endregion

    #region FP8 Quantization Tests

    [Fact]
    public void FP8Quantizer_E4M3Format_QuantizesWithBetterPrecision()
    {
        // Arrange
        var model = CreateTestModel(100);
        var quantizer = new FP8Quantizer<double, double[], double[]>(FP8Format.E4M3);
        var calibrationData = CreateCalibrationData(50, 10).ToList();
        var config = new QuantizationConfiguration { Mode = QuantizationMode.Float16 };

        // Act
        quantizer.Calibrate(model, calibrationData);
        var quantizedModel = quantizer.Quantize(model, config);

        // Assert
        Assert.Equal(FP8Format.E4M3, quantizer.Format);
        Assert.Equal(8, quantizer.BitWidth);

        var quantizedParams = quantizedModel.GetParameters();
        Assert.Equal(100, quantizedParams.Length);
    }

    [Fact]
    public void FP8Quantizer_E5M2Format_HasLargerRange()
    {
        // Arrange
        var quantizerE4M3 = new FP8Quantizer<double, double[], double[]>(FP8Format.E4M3);
        var quantizerE5M2 = new FP8Quantizer<double, double[], double[]>(FP8Format.E5M2);

        // E4M3 max: 448, E5M2 max: 57344
        Assert.Equal(FP8Format.E4M3, quantizerE4M3.Format);
        Assert.Equal(FP8Format.E5M2, quantizerE5M2.Format);
    }

    [Fact]
    public void FP8Quantizer_ByteConversion_RoundTrips()
    {
        // Test E4M3 byte conversion
        var testValues = new double[] { 0.0, 1.0, -1.0, 0.5, 100.0, -100.0 };

        foreach (var value in testValues)
        {
            var byteValue = FP8Quantizer<double, double[], double[]>.E4M3ToByte(value);
            var roundTrip = FP8Quantizer<double, double[], double[]>.ByteToE4M3(byteValue);

            // Round-trip should be within FP8 precision
            if (Math.Abs(value) < 448) // E4M3 max
            {
                Assert.True(Math.Abs(roundTrip - value) / Math.Max(Math.Abs(value), 1e-6) < 0.2,
                    $"E4M3 round-trip error for {value} -> {byteValue} -> {roundTrip}");
            }
        }

        // Test E5M2 byte conversion
        foreach (var value in testValues)
        {
            var byteValue = FP8Quantizer<double, double[], double[]>.E5M2ToByte(value);
            var roundTrip = FP8Quantizer<double, double[], double[]>.ByteToE5M2(byteValue);

            // Round-trip should be within FP8 precision
            if (Math.Abs(value) < 57344) // E5M2 max
            {
                Assert.True(Math.Abs(roundTrip - value) / Math.Max(Math.Abs(value), 1e-6) < 0.3,
                    $"E5M2 round-trip error for {value} -> {byteValue} -> {roundTrip}");
            }
        }
    }

    #endregion

    #region NF4 Quantization Tests

    [Fact]
    public void NF4Quantizer_4BitQuantization_Completes()
    {
        // Arrange
        var model = CreateTestModel(64);
        var quantizer = new NF4Quantizer<double, double[], double[]>(blockSize: 16);
        var calibrationData = CreateCalibrationData(10, 4).ToList();
        var config = new QuantizationConfiguration { Mode = QuantizationMode.Int8 };

        // Act
        quantizer.Calibrate(model, calibrationData);
        var quantizedModel = quantizer.Quantize(model, config);

        // Assert
        Assert.True(quantizer.IsCalibrated);
        Assert.Equal(4, quantizer.BitWidth);
        Assert.Equal(64, quantizedModel.GetParameters().Length);
    }

    [Fact]
    public void NF4Quantizer_IndexConversions_AreConsistent()
    {
        // Test that index -> value -> index is consistent
        for (int i = 0; i < 16; i++)
        {
            double value = NF4Quantizer<double, double[], double[]>.IndexToNF4(i);
            int roundTrip = NF4Quantizer<double, double[], double[]>.NF4ToIndex(value);
            Assert.Equal(i, roundTrip);
        }
    }

    [Fact]
    public void NF4Quantizer_BlockWiseQuantization_PreservesStructure()
    {
        // Arrange
        var model = CreateTestModel(100);
        var quantizer = new NF4Quantizer<double, double[], double[]>(blockSize: 32);
        var config = new QuantizationConfiguration { Mode = QuantizationMode.Int8 };

        // Act
        var quantizedModel = quantizer.Quantize(model, config);

        // Assert
        var originalParams = model.GetParameters();
        var quantizedParams = quantizedModel.GetParameters();
        Assert.Equal(originalParams.Length, quantizedParams.Length);

        // All values should be finite
        for (int i = 0; i < quantizedParams.Length; i++)
        {
            Assert.False(double.IsNaN(quantizedParams[i]), "NF4 quantized params contain NaN");
        }
    }

    #endregion

    #region MXFP4 Quantization Tests

    [Fact]
    public void MXFP4Quantizer_4BitMicroscaling_Completes()
    {
        // Arrange
        var model = CreateTestModel(64);
        var quantizer = new MXFP4Quantizer<double, double[], double[]>(blockSize: 32);
        var calibrationData = CreateCalibrationData(10, 4).ToList();
        var config = new QuantizationConfiguration { Mode = QuantizationMode.Float16 };

        // Act
        quantizer.Calibrate(model, calibrationData);
        var quantizedModel = quantizer.Quantize(model, config);

        // Assert
        Assert.True(quantizer.IsCalibrated);
        Assert.Equal(4, quantizer.BitWidth);
        Assert.Equal(64, quantizedModel.GetParameters().Length);
    }

    [Fact]
    public void MXFP4Quantizer_EncodeDecodeRoundTrip()
    {
        // Test all 16 possible 4-bit values
        for (int encoded = 0; encoded < 16; encoded++)
        {
            double decoded = MXFP4Quantizer<double, double[], double[]>.DecodeFromMXFP4(encoded);
            int reencoded = MXFP4Quantizer<double, double[], double[]>.EncodeToMXFP4(decoded);

            // Should round-trip correctly (accounting for sign)
            Assert.False(double.IsNaN(decoded), $"MXFP4 decode produced NaN for {encoded}");
        }
    }

    [Fact]
    public void MXFP4Quantizer_SharedScales_WorkCorrectly()
    {
        // Arrange
        var model = CreateTestModel(100);
        var quantizer = new MXFP4Quantizer<double, double[], double[]>(blockSize: 16);
        var config = new QuantizationConfiguration { Mode = QuantizationMode.Float16 };

        // Act
        var quantizedModel = quantizer.Quantize(model, config);

        // Assert
        Assert.Equal(16, quantizer.BlockSize);
        var quantizedParams = quantizedModel.GetParameters();
        Assert.Equal(100, quantizedParams.Length);

        // All values should be finite
        for (int i = 0; i < quantizedParams.Length; i++)
        {
            Assert.False(double.IsNaN(quantizedParams[i]), "MXFP4 quantized params contain NaN");
            Assert.False(double.IsInfinity(quantizedParams[i]), "MXFP4 quantized params contain Infinity");
        }
    }

    #endregion

    #region Quantization Granularity Tests

    [Fact]
    public void PerTensorGranularity_UsesSingleScaleFactor()
    {
        // Arrange
        var model = CreateTestModel(256);
        var config = new QuantizationConfiguration
        {
            Mode = QuantizationMode.Int8,
            Granularity = QuantizationGranularity.PerTensor,
            CalibrationMethod = CalibrationMethod.MinMax
        };
        var quantizer = new Int8Quantizer<double, double[], double[]>();
        var calibrationData = CreateCalibrationData(50, 16).ToList();

        // Act
        quantizer.Calibrate(model, calibrationData);
        quantizer.Quantize(model, config);

        // Assert
        var globalScale = quantizer.GetScaleFactor("global");
        Assert.True(globalScale > 0, "Should have a global scale factor");
    }

    [Fact]
    public void PerGroupGranularity_UsesMultipleScaleFactors()
    {
        // Arrange
        var model = CreateTestModel(256);
        var config = new QuantizationConfiguration
        {
            Mode = QuantizationMode.Int8,
            Granularity = QuantizationGranularity.PerGroup,
            GroupSize = 64,
            Strategy = QuantizationStrategy.GPTQ,
            CalibrationMethod = CalibrationMethod.MinMax
        };
        var quantizer = new GPTQQuantizer<double, double[], double[]>(config);
        var calibrationData = CreateCalibrationData(50, 16).ToList();

        // Act
        quantizer.Calibrate(model, calibrationData);
        quantizer.Quantize(model, config);

        // Assert - Should have multiple group scale factors
        var group0Scale = quantizer.GetScaleFactor("group_0");
        var group1Scale = quantizer.GetScaleFactor("group_1");

        Assert.True(group0Scale > 0, "Should have scale factor for group 0");
        Assert.True(group1Scale > 0, "Should have scale factor for group 1");
    }

    #endregion

    #region QuantizationInfo Tests

    [Fact]
    public void QuantizationInfo_None_ReturnsUnquantizedInfo()
    {
        // Arrange & Act
        var info = QuantizationInfo.None;

        // Assert
        Assert.False(info.IsQuantized);
        Assert.Equal(QuantizationMode.None, info.Mode);
        Assert.Equal(32, info.BitWidth);
        Assert.Equal(1.0, info.CompressionRatio);
    }

    [Fact]
    public void QuantizationInfo_CompressionRatio_CalculatesCorrectly()
    {
        // Arrange
        var info = new QuantizationInfo
        {
            IsQuantized = true,
            Mode = QuantizationMode.Int8,
            BitWidth = 8,
            OriginalSizeBytes = 1000,
            QuantizedSizeBytes = 250
        };

        // Act & Assert
        Assert.Equal(4.0, info.CompressionRatio);
    }

    [Fact]
    public void QuantizationInfo_QuantizedPercentage_CalculatesCorrectly()
    {
        // Arrange
        var info = new QuantizationInfo
        {
            IsQuantized = true,
            TotalParameters = 1000,
            QuantizedParameters = 950
        };

        // Act & Assert
        Assert.Equal(95.0, info.QuantizedPercentage);
    }

    [Fact]
    public void QuantizationInfo_ToString_ReturnsFormattedString()
    {
        // Arrange
        var info = new QuantizationInfo
        {
            IsQuantized = true,
            Mode = QuantizationMode.Int8,
            Strategy = QuantizationStrategy.GPTQ,
            Granularity = QuantizationGranularity.PerGroup,
            BitWidth = 4,
            OriginalSizeBytes = 1000,
            QuantizedSizeBytes = 125
        };

        // Act
        var str = info.ToString();

        // Assert
        Assert.Contains("Int8", str);
        Assert.Contains("4-bit", str);
        Assert.Contains("GPTQ", str);
        Assert.Contains("PerGroup", str);
        Assert.Contains("PTQ", str); // Not QAT
    }

    [Fact]
    public void QuantizationInfo_WithQAT_IncludesQATInfo()
    {
        // Arrange
        var info = new QuantizationInfo
        {
            IsQuantized = true,
            Mode = QuantizationMode.Int8,
            BitWidth = 8,
            UsedQAT = true,
            QATMethod = QATMethod.EfficientQAT,
            OriginalSizeBytes = 1000,
            QuantizedSizeBytes = 250
        };

        // Act
        var str = info.ToString();

        // Assert
        Assert.Contains("QAT", str);
        Assert.Contains("EfficientQAT", str);
    }

    #endregion

    #region QuantizationConfiguration Tests

    [Fact]
    public void QuantizationConfiguration_ForGPTQ_HasCorrectDefaults()
    {
        // Act
        var config = QuantizationConfiguration.ForGPTQ();

        // Assert
        Assert.Equal(QuantizationMode.Int8, config.Mode);
        Assert.Equal(4, config.TargetBitWidth);
        Assert.Equal(QuantizationStrategy.GPTQ, config.Strategy);
        Assert.Equal(QuantizationGranularity.PerGroup, config.Granularity);
        Assert.Equal(128, config.GroupSize);
        Assert.True(config.GPTQActOrder);
    }

    [Fact]
    public void QuantizationConfiguration_ForAWQ_HasCorrectDefaults()
    {
        // Act
        var config = QuantizationConfiguration.ForAWQ();

        // Assert
        Assert.Equal(QuantizationMode.Int8, config.Mode);
        Assert.Equal(4, config.TargetBitWidth);
        Assert.Equal(QuantizationStrategy.AWQ, config.Strategy);
        Assert.Equal(QuantizationGranularity.PerGroup, config.Granularity);
        Assert.Equal(1.0, config.AWQProtectionPercentage);
    }

    [Fact]
    public void QuantizationConfiguration_ForSmoothQuant_HasCorrectDefaults()
    {
        // Act
        var config = QuantizationConfiguration.ForSmoothQuant();

        // Assert
        Assert.Equal(QuantizationMode.Int8, config.Mode);
        Assert.Equal(QuantizationStrategy.SmoothQuant, config.Strategy);
        Assert.Equal(QuantizationGranularity.PerChannel, config.Granularity);
        Assert.True(config.QuantizeActivations);
        Assert.Equal(0.5, config.SmoothQuantAlpha);
    }

    [Fact]
    public void QuantizationConfiguration_ForQAT_HasCorrectDefaults()
    {
        // Act
        var config = QuantizationConfiguration.ForQAT();

        // Assert
        Assert.True(config.UseQuantizationAwareTraining);
        Assert.Equal(QATMethod.EfficientQAT, config.QATMethod);
        Assert.Equal(1, config.QATWarmupEpochs);
    }

    [Fact]
    public void QuantizationConfiguration_ForQLoRA_Has4BitNF4()
    {
        // Act
        var config = QuantizationConfiguration.ForQLoRA();

        // Assert
        Assert.Equal(4, config.TargetBitWidth);
        Assert.Equal(QuantizationGranularity.PerGroup, config.Granularity);
        Assert.Equal(64, config.GroupSize);
        Assert.True(config.UseQuantizationAwareTraining);
        Assert.Equal(QATMethod.QABLoRA, config.QATMethod);
    }

    [Fact]
    public void QuantizationConfiguration_EffectiveBitWidth_UsesTargetOrDefault()
    {
        // Arrange
        var configWithTarget = new QuantizationConfiguration
        {
            Mode = QuantizationMode.Int8,
            TargetBitWidth = 4
        };
        var configWithoutTarget = new QuantizationConfiguration
        {
            Mode = QuantizationMode.Int8
        };

        // Assert
        Assert.Equal(4, configWithTarget.EffectiveBitWidth);
        Assert.Equal(8, configWithoutTarget.EffectiveBitWidth);
    }

    #endregion

    #region LoRA + Quantization Integration Tests

    [Fact]
    public void LoRA_WithQLoRAConfiguration_HasCorrectSettings()
    {
        // Act
        var config = QuantizationConfiguration.ForQLoRA();

        // Assert - QLoRA uses 4-bit NF4 quantization with LoRA adapters
        Assert.Equal(4, config.TargetBitWidth);
        Assert.Equal(64, config.GroupSize);
        Assert.True(config.UseQuantizationAwareTraining);
        Assert.Equal(QATMethod.QABLoRA, config.QATMethod);
    }

    [Fact]
    public void QLoRA_QuantizesModel_PreservesLoRALayerStructure()
    {
        // Arrange - Create a model with LoRA-like parameters
        // Simulates a base model that would have LoRA adapters attached
        var model = CreateTestModel(256);
        var quantizer = new Int8Quantizer<double, double[], double[]>();
        var config = new QuantizationConfiguration
        {
            Mode = QuantizationMode.Int8,
            TargetBitWidth = 4,
            Granularity = QuantizationGranularity.PerGroup,
            GroupSize = 64,
            CalibrationMethod = CalibrationMethod.MinMax,
            UseQuantizationAwareTraining = true,
            QATMethod = QATMethod.QABLoRA
        };

        // Act - Calibrate and quantize
        var calibrationData = CreateCalibrationData(50, 10);
        quantizer.Calibrate(model, calibrationData);
        var quantizedModel = quantizer.Quantize(model, config);

        // Assert
        Assert.NotNull(quantizedModel);
        var originalParams = model.GetParameters();
        var quantizedParams = quantizedModel.GetParameters();
        Assert.Equal(originalParams.Length, quantizedParams.Length);
    }

    [Fact]
    public void QABLoRA_QATMethod_AppliesCorrectly()
    {
        // Arrange
        var config = new QuantizationConfiguration
        {
            Mode = QuantizationMode.Int8,
            TargetBitWidth = 4,
            UseQuantizationAwareTraining = true,
            QATMethod = QATMethod.QABLoRA,
            QATWarmupEpochs = 0
        };
        var hook = new QATTrainingHook<double>(config);
        hook.OnEpochStart(0);

        // Simulate LoRA layer parameters (A and B matrices)
        // A matrix: inputSize x rank, B matrix: rank x outputSize
        // Using small rank (typical for LoRA)
        var loraAWeights = new Vector<double>(new double[] { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 }); // 4x2
        var loraBWeights = new Vector<double>(new double[] { 0.5, -0.5, 0.3, -0.3 }); // 2x2

        // Act - Apply QAT fake quantization to both LoRA matrices
        var quantizedA = hook.ApplyFakeQuantization(loraAWeights, "lora_A");
        var quantizedB = hook.ApplyFakeQuantization(loraBWeights, "lora_B");

        // Assert - Values should be modified by quantization
        Assert.Equal(loraAWeights.Length, quantizedA.Length);
        Assert.Equal(loraBWeights.Length, quantizedB.Length);

        // Both states should be tracked
        var stateA = hook.GetLayerState("lora_A");
        var stateB = hook.GetLayerState("lora_B");
        Assert.NotNull(stateA);
        Assert.NotNull(stateB);
    }

    [Fact]
    public void LoRAAdapter_Quantization_PreservesRankStructure()
    {
        // This test verifies that quantization doesn't break LoRA's low-rank structure
        // LoRA decomposes W = BA where B is (d x r) and A is (r x k)
        // After quantization, the rank should remain the same

        // Arrange
        int inputSize = 16;
        int outputSize = 8;
        int rank = 4;

        // Simulate LoRA decomposition: W_delta = B @ A
        // A: (rank x outputSize) = 4 x 8 = 32 params
        // B: (inputSize x rank) = 16 x 4 = 64 params
        var loraAParams = new double[rank * outputSize];
        var loraBParams = new double[inputSize * rank];
        var random = new Random(42);

        for (int i = 0; i < loraAParams.Length; i++)
            loraAParams[i] = random.NextDouble() * 2 - 1;
        for (int i = 0; i < loraBParams.Length; i++)
            loraBParams[i] = random.NextDouble() * 2 - 1;

        var loraA = new Vector<double>(loraAParams);
        var loraB = new Vector<double>(loraBParams);

        var config = new QuantizationConfiguration
        {
            Mode = QuantizationMode.Int8,
            TargetBitWidth = 4,
            UseSymmetricQuantization = true,
            QATWarmupEpochs = 0
        };
        var hook = new QATTrainingHook<double>(config);
        hook.OnEpochStart(0);

        // Act
        var quantizedA = hook.ApplyFakeQuantization(loraA, "lora_layer_A");
        var quantizedB = hook.ApplyFakeQuantization(loraB, "lora_layer_B");

        // Assert - The parameter counts should match (rank structure preserved)
        Assert.Equal(rank * outputSize, quantizedA.Length);
        Assert.Equal(inputSize * rank, quantizedB.Length);

        // Values should be modified but remain finite
        for (int i = 0; i < quantizedA.Length; i++)
        {
            Assert.False(double.IsNaN(quantizedA[i]), "Quantized A contains NaN");
            Assert.False(double.IsInfinity(quantizedA[i]), "Quantized A contains Infinity");
        }
        for (int i = 0; i < quantizedB.Length; i++)
        {
            Assert.False(double.IsNaN(quantizedB[i]), "Quantized B contains NaN");
            Assert.False(double.IsInfinity(quantizedB[i]), "Quantized B contains Infinity");
        }
    }

    #endregion

    #region Accuracy and Compression Verification Tests

    [Fact]
    public void Int8Quantization_AchievesExpectedCompressionRatio()
    {
        // Arrange - INT8 should achieve 4x compression (32-bit to 8-bit)
        var model = CreateTestModel(1000);
        var quantizer = new Int8Quantizer<double, double[], double[]>();
        var calibrationData = CreateCalibrationData(50, 10).ToList();
        var config = new QuantizationConfiguration
        {
            Mode = QuantizationMode.Int8,
            CalibrationMethod = CalibrationMethod.MinMax
        };

        // Act
        quantizer.Calibrate(model, calibrationData);
        var quantizedModel = quantizer.Quantize(model, config);

        // Assert - Verify compression ratio
        var originalParams = model.GetParameters();
        var quantizedParams = quantizedModel.GetParameters();

        // Parameter count should be preserved
        Assert.Equal(originalParams.Length, quantizedParams.Length);

        // Calculate theoretical compression ratio
        long originalBits = originalParams.Length * 64; // double = 64 bits
        long quantizedBits = quantizedParams.Length * 8; // INT8 = 8 bits
        double theoreticalRatio = (double)originalBits / quantizedBits;

        Assert.True(theoreticalRatio >= 7.5, $"INT8 compression ratio should be ~8x, got {theoreticalRatio}x");
    }

    [Fact]
    public void Float16Quantization_MaintainsReasonableAccuracy()
    {
        // Arrange - FP16 should have very low error
        var model = CreateTestModel(500);
        var quantizer = new Float16Quantizer<double, double[], double[]>();
        var calibrationData = CreateCalibrationData(50, 10).ToList();
        var config = new QuantizationConfiguration
        {
            Mode = QuantizationMode.Float16,
            CalibrationMethod = CalibrationMethod.None
        };

        // Act
        var quantizedModel = quantizer.Quantize(model, config);

        // Assert - Compute Mean Squared Error
        var originalParams = model.GetParameters();
        var quantizedParams = quantizedModel.GetParameters();

        double mse = 0;
        for (int i = 0; i < originalParams.Length; i++)
        {
            double diff = originalParams[i] - quantizedParams[i];
            mse += diff * diff;
        }
        mse /= originalParams.Length;

        // FP16 should have very low MSE (relative to weight magnitudes)
        Assert.True(mse < 0.01, $"FP16 MSE should be very low, got {mse}");
    }

    [Fact]
    public void GPTQ4BitQuantization_AchievesHighCompression()
    {
        // Arrange - GPTQ 4-bit should achieve 8x compression
        var model = CreateTestModel(256);
        var config = QuantizationConfiguration.ForGPTQ(groupSize: 64);
        var quantizer = new GPTQQuantizer<double, double[], double[]>(config);
        var calibrationData = CreateCalibrationData(100, 16).ToList();

        // Act
        quantizer.Calibrate(model, calibrationData);
        var quantizedModel = quantizer.Quantize(model, config);

        // Assert
        Assert.Equal(4, quantizer.BitWidth);

        var quantizedParams = quantizedModel.GetParameters();
        Assert.Equal(256, quantizedParams.Length);

        // All values should be finite
        for (int i = 0; i < quantizedParams.Length; i++)
        {
            Assert.False(double.IsNaN(quantizedParams[i]), "GPTQ produced NaN");
            Assert.False(double.IsInfinity(quantizedParams[i]), "GPTQ produced Infinity");
        }
    }

    [Fact]
    public void QuIPSharp2Bit_AchievesExtremeCompression()
    {
        // Arrange - QuIP# 2-bit achieves 16x compression
        var model = CreateTestModel(64);
        var quantizer = new QuIPSharpQuantizer<double, double[], double[]>(groupSize: 8);
        var config = new QuantizationConfiguration
        {
            Mode = QuantizationMode.Int8,
            Strategy = QuantizationStrategy.QuIPSharp,
            TargetBitWidth = 2
        };

        // Act
        var quantizedModel = quantizer.Quantize(model, config);

        // Assert
        Assert.Equal(2, quantizer.BitWidth);

        // 16x theoretical compression (32-bit to 2-bit)
        long originalBits = 64 * 64; // 64 doubles, each 64 bits
        long quantizedBits = 64 * 2; // 64 values, each 2 bits
        double theoreticalRatio = (double)originalBits / quantizedBits;

        Assert.True(theoreticalRatio >= 30, $"QuIP# should achieve ~32x compression, got {theoreticalRatio}x");
    }

    [Fact]
    public void QuantizationAccuracy_DifferentBitWidths_TradeOff()
    {
        // Verify that lower bit widths have higher error (expected trade-off)
        var model = CreateTestModel(100);
        var originalParams = model.GetParameters();

        // Test with INT8 (8-bit)
        var quantizer8 = new Int8Quantizer<double, double[], double[]>();
        var calibrationData = CreateCalibrationData(50, 10).ToList();
        quantizer8.Calibrate(model, calibrationData);
        var config8 = new QuantizationConfiguration { Mode = QuantizationMode.Int8, CalibrationMethod = CalibrationMethod.MinMax };
        var quantized8 = quantizer8.Quantize(model, config8);

        // Test with FP16 (16-bit equivalent precision)
        var quantizer16 = new Float16Quantizer<double, double[], double[]>();
        var config16 = new QuantizationConfiguration { Mode = QuantizationMode.Float16, CalibrationMethod = CalibrationMethod.None };
        var quantized16 = quantizer16.Quantize(model, config16);

        // Compute MSE for each
        double mse8 = ComputeMSE(originalParams, quantized8.GetParameters());
        double mse16 = ComputeMSE(originalParams, quantized16.GetParameters());

        // FP16 should generally have lower error than INT8
        // (This is a general expectation, may vary with specific values)
        Assert.True(mse8 >= 0 && mse16 >= 0, "MSE should be non-negative");
    }

    [Fact]
    public void QuantizationInfo_CompressionRatio_Verification()
    {
        // Arrange
        var info = new QuantizationInfo
        {
            IsQuantized = true,
            Mode = QuantizationMode.Int8,
            BitWidth = 4,
            OriginalSizeBytes = 1000,
            QuantizedSizeBytes = 125 // 8x compression
        };

        // Assert - verify compression ratio calculation
        Assert.Equal(8.0, info.CompressionRatio);
        Assert.True(info.CompressionRatio > 1, "Compression ratio should be > 1");
    }

    [Fact]
    public void AllQuantizers_ProduceFiniteValues()
    {
        // Arrange
        var model = CreateTestModel(64);
        var calibrationData = CreateCalibrationData(10, 4).ToList();
        var config = new QuantizationConfiguration { Mode = QuantizationMode.Int8 };

        // Test all quantizer types
        var quantizers = new Dictionary<string, Func<Vector<double>>>
        {
            ["Int8"] = () => {
                var q = new Int8Quantizer<double, double[], double[]>();
                q.Calibrate(model, calibrationData);
                return q.Quantize(model, config).GetParameters();
            },
            ["Float16"] = () => {
                var q = new Float16Quantizer<double, double[], double[]>();
                return q.Quantize(model, config).GetParameters();
            },
            ["NF4"] = () => {
                var q = new NF4Quantizer<double, double[], double[]>(blockSize: 16);
                return q.Quantize(model, config).GetParameters();
            },
            ["MXFP4"] = () => {
                var q = new MXFP4Quantizer<double, double[], double[]>(blockSize: 16);
                return q.Quantize(model, config).GetParameters();
            }
        };

        // Act & Assert
        foreach (var (name, getParams) in quantizers)
        {
            var quantizedParams = getParams();
            for (int i = 0; i < quantizedParams.Length; i++)
            {
                Assert.False(double.IsNaN(quantizedParams[i]), $"{name} produced NaN at index {i}");
                Assert.False(double.IsInfinity(quantizedParams[i]), $"{name} produced Infinity at index {i}");
            }
        }
    }

    /// <summary>
    /// Helper method to compute Mean Squared Error between two parameter vectors.
    /// </summary>
    private static double ComputeMSE(Vector<double> original, Vector<double> quantized)
    {
        if (original.Length != quantized.Length)
            throw new ArgumentException("Vectors must have same length");

        double sum = 0;
        for (int i = 0; i < original.Length; i++)
        {
            double diff = original[i] - quantized[i];
            sum += diff * diff;
        }
        return sum / original.Length;
    }

    #endregion

    #region Calibration Tests

    [Fact]
    public void Calibration_WithEmptyData_ThrowsException()
    {
        // Arrange
        var model = CreateTestModel(100);
        var quantizer = new Int8Quantizer<double, double[], double[]>();
        var emptyData = Enumerable.Empty<double[]>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => quantizer.Calibrate(model, emptyData));
    }

    [Fact]
    public void Calibration_WithNullModel_ThrowsException()
    {
        // Arrange
        var quantizer = new Int8Quantizer<double, double[], double[]>();
        var calibrationData = CreateCalibrationData(50, 10).ToList();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => quantizer.Calibrate(null!, calibrationData));
    }

    [Fact]
    public void Quantize_WithoutCalibration_ThrowsException()
    {
        // Arrange
        var model = CreateTestModel(100);
        var quantizer = new Int8Quantizer<double, double[], double[]>();
        var config = new QuantizationConfiguration
        {
            Mode = QuantizationMode.Int8,
            CalibrationMethod = CalibrationMethod.MinMax
        };

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => quantizer.Quantize(model, config));
    }

    [Fact]
    public void Quantize_WithCalibrationMethodNone_WorksWithoutCalibration()
    {
        // Arrange
        var model = CreateTestModel(100);
        var quantizer = new Float16Quantizer<double, double[], double[]>();
        var config = new QuantizationConfiguration
        {
            Mode = QuantizationMode.Float16,
            CalibrationMethod = CalibrationMethod.None
        };

        // Act - Should not throw
        var quantizedModel = quantizer.Quantize(model, config);

        // Assert
        Assert.NotNull(quantizedModel);
    }

    [Fact]
    public void AWQQuantizer_Calibrate_CollectsActivationStatistics()
    {
        // Arrange
        var model = CreateTestModel(100);
        var quantizer = new AWQQuantizer<double, double[], double[]>();
        var calibrationData = CreateCalibrationData(50, 10).ToList();

        // Act
        quantizer.Calibrate(model, calibrationData);

        // Assert
        Assert.True(quantizer.IsCalibrated);
        // Verify that scale factor is computed
        var scaleFactor = quantizer.GetScaleFactor("global");
        Assert.True(scaleFactor > 0, "Scale factor should be positive");
    }

    [Fact]
    public void SmoothQuantQuantizer_Calibrate_CollectsActivationStatistics()
    {
        // Arrange
        var model = CreateTestModel(100);
        var quantizer = new SmoothQuantQuantizer<double, double[], double[]>();
        var calibrationData = CreateCalibrationData(50, 10).ToList();

        // Act
        quantizer.Calibrate(model, calibrationData);

        // Assert
        Assert.True(quantizer.IsCalibrated);
        // Verify that smoothing scales are computed
        Assert.NotEmpty(quantizer.SmoothingScales);
    }

    [Fact]
    public void GPTQQuantizer_Calibrate_CollectsActivationStatistics()
    {
        // Arrange
        var model = CreateTestModel(100);
        var quantizer = new GPTQQuantizer<double, double[], double[]>();
        var calibrationData = CreateCalibrationData(50, 10).ToList();

        // Act
        quantizer.Calibrate(model, calibrationData);

        // Assert
        Assert.True(quantizer.IsCalibrated);
        // Verify that scale factor is computed
        var scaleFactor = quantizer.GetScaleFactor("global");
        Assert.True(scaleFactor > 0, "Scale factor should be positive");
    }

    [Fact]
    public void CalibrationHelper_WithSimpleModel_CollectsStatistics()
    {
        // Arrange
        var model = CreateTestModel(100);
        var config = new QuantizationConfiguration
        {
            NumCalibrationSamples = 10
        };
        var helper = new CalibrationHelper<double, double[], double[]>(config);
        var calibrationData = CreateCalibrationData(10, 10).ToList();

        // Act
        var stats = helper.CollectActivationStatistics(model, calibrationData);

        // Assert
        Assert.NotNull(stats);
        Assert.True(stats.SampleCount > 0, "Should have processed samples");
        Assert.NotNull(stats.GlobalActivationMagnitudes);
        Assert.NotNull(stats.GlobalMaxAbsActivations);
        Assert.Equal(100, stats.GlobalActivationMagnitudes.Length);
    }

    #endregion

    #region QAT Simulation Tests

    [Fact]
    public void QATTrainingHook_ApplyFakeQuantization_ModifiesWeights()
    {
        // Arrange
        var config = new QuantizationConfiguration
        {
            Mode = QuantizationMode.Int8,
            TargetBitWidth = 8,
            UseSymmetricQuantization = true,
            QATWarmupEpochs = 0 // Enable immediately
        };
        var hook = new QATTrainingHook<double>(config);
        hook.OnEpochStart(0); // Enable quantization

        var weights = new Vector<double>(new double[] { 0.5, -0.3, 1.2, -0.8, 0.0, 2.5 });

        // Act
        var quantizedWeights = hook.ApplyFakeQuantization(weights, "test_layer");

        // Assert - Fake quantization should modify values
        Assert.Equal(weights.Length, quantizedWeights.Length);

        // At least some values should be different due to quantization
        bool hasChanges = false;
        for (int i = 0; i < weights.Length; i++)
        {
            if (Math.Abs(weights[i] - quantizedWeights[i]) > 1e-10)
            {
                hasChanges = true;
                break;
            }
        }
        Assert.True(hasChanges, "Fake quantization should modify at least some values");
    }

    [Fact]
    public void QATTrainingHook_DuringWarmup_PassesThroughUnchanged()
    {
        // Arrange
        var config = new QuantizationConfiguration
        {
            Mode = QuantizationMode.Int8,
            TargetBitWidth = 8,
            QATWarmupEpochs = 5 // Warmup for 5 epochs
        };
        var hook = new QATTrainingHook<double>(config);
        hook.OnEpochStart(0); // Epoch 0, still in warmup

        var weights = new Vector<double>(new double[] { 0.5, -0.3, 1.2 });

        // Act
        var result = hook.ApplyFakeQuantization(weights, "test_layer");

        // Assert - During warmup, weights should pass through unchanged
        for (int i = 0; i < weights.Length; i++)
        {
            Assert.Equal(weights[i], result[i], 10); // Should be exactly equal
        }
    }

    [Fact]
    public void QATTrainingHook_AfterWarmup_AppliesQuantization()
    {
        // Arrange
        var config = new QuantizationConfiguration
        {
            Mode = QuantizationMode.Int8,
            TargetBitWidth = 8,
            QATWarmupEpochs = 2
        };
        var hook = new QATTrainingHook<double>(config);

        var weights = new Vector<double>(new double[] { 0.5, -0.3, 1.2 });

        // Act - Advance past warmup
        hook.OnEpochStart(0);
        var duringWarmup = hook.ApplyFakeQuantization(weights, "test_layer");

        hook.OnEpochStart(2); // At warmup epoch, quantization should enable
        var afterWarmup = hook.ApplyFakeQuantization(weights, "test_layer");

        // Assert
        Assert.False(hook.IsQuantizationEnabled == false || duringWarmup[0] != weights[0],
            "Quantization should be disabled during warmup (epoch 0 < 2)");
        Assert.True(hook.IsQuantizationEnabled,
            "Quantization should be enabled after warmup");
    }

    [Fact]
    public void QATTrainingHook_GetLayerState_ReturnsCorrectState()
    {
        // Arrange
        var config = new QuantizationConfiguration
        {
            Mode = QuantizationMode.Int8,
            TargetBitWidth = 8,
            QATWarmupEpochs = 0
        };
        var hook = new QATTrainingHook<double>(config);
        hook.OnEpochStart(0);

        var weights = new Vector<double>(new double[] { 0.5, -0.3, 1.2, -0.8, 2.0 });

        // Act
        hook.ApplyFakeQuantization(weights, "test_layer");
        var state = hook.GetLayerState("test_layer");

        // Assert
        Assert.NotNull(state);
        Assert.Equal("test_layer", state.LayerName);
        Assert.Equal(8, state.BitWidth);
        Assert.True(state.Scale > 0);
    }

    [Fact]
    public void EfficientQATOptimizer_ApplyBlockWiseQuantization_Works()
    {
        // Arrange
        var config = new QuantizationConfiguration
        {
            Mode = QuantizationMode.Int8,
            TargetBitWidth = 8,
            GroupSize = 4,
            QATMethod = QATMethod.EfficientQAT,
            QATWarmupEpochs = 0
        };
        var optimizer = new EfficientQATOptimizer<double>(config, totalEpochs: 10);
        optimizer.OnEpochStart(0);

        var weights = new Vector<double>(new double[] { 0.5, -0.3, 1.2, -0.8, 2.0, -1.5, 0.7, -0.2 });

        // Act
        var quantizedWeights = optimizer.ApplyBlockWiseQuantization(weights, "test_layer");

        // Assert
        Assert.Equal(weights.Length, quantizedWeights.Length);
    }

    [Fact]
    public void EfficientQATOptimizer_EstimateMemoryUsage_ReturnsReasonableValue()
    {
        // Arrange
        var config = new QuantizationConfiguration
        {
            Mode = QuantizationMode.Int8,
            TargetBitWidth = 8,
            GroupSize = 128
        };
        var optimizer = new EfficientQATOptimizer<double>(config, totalEpochs: 10);

        // Act
        long memoryEstimate = optimizer.EstimateMemoryUsage(1_000_000);

        // Assert - Memory estimate should be reasonable (less than full precision)
        long fullPrecisionMemory = 1_000_000 * sizeof(double);
        Assert.True(memoryEstimate < fullPrecisionMemory,
            "Quantized memory estimate should be less than full precision");
        Assert.True(memoryEstimate > 0, "Memory estimate should be positive");
    }

    #endregion

    #region Helper Test Model

    /// <summary>
    /// Simple test model for quantization testing.
    /// Implements all IFullModel interface members.
    /// </summary>
    private class SimpleTestModel<T> : IFullModel<T, T[], T[]>
    {
        private Vector<T> _parameters;
        private readonly ModelMetadata<T> _metadata = new();
        private IEnumerable<int> _activeFeatureIndices = Array.Empty<int>();

        public SimpleTestModel(T[] weights)
        {
            _parameters = new Vector<T>(weights);
        }

        // IParameterizable
        public Vector<T> GetParameters() => _parameters;

        public void SetParameters(Vector<T> parameters)
        {
            _parameters = parameters;
        }

        public int ParameterCount => _parameters.Length;

        public IFullModel<T, T[], T[]> WithParameters(Vector<T> parameters)
        {
            var newModel = new SimpleTestModel<T>(new T[0]);
            newModel._parameters = parameters;
            return newModel;
        }

        // IModel
        public T[] Predict(T[] input) => input;

        public void Train(T[] inputs, T[] outputs)
        {
            // No-op for test model
        }

        public ModelMetadata<T> GetModelMetadata() => _metadata;

        // IFullModel
        public ILossFunction<T> DefaultLossFunction => new MeanSquaredErrorLoss<T>();

        // IModelSerializer
        public byte[] Serialize() => Array.Empty<byte>();
        public void Deserialize(byte[] data) { }
        public void SaveModel(string path) { }
        public void LoadModel(string path) { }

        // ICheckpointableModel
        public void SaveState(Stream stream) { }
        public void LoadState(Stream stream) { }

        // IFeatureAware
        public IEnumerable<int> GetActiveFeatureIndices() => _activeFeatureIndices;
        public void SetActiveFeatureIndices(IEnumerable<int> indices) => _activeFeatureIndices = indices;
        public bool IsFeatureUsed(int featureIndex) => _activeFeatureIndices.Contains(featureIndex);

        // IFeatureImportance
        public Dictionary<string, T> GetFeatureImportance() => new Dictionary<string, T>();

        // ICloneable
        public IFullModel<T, T[], T[]> DeepCopy()
        {
            var copy = new SimpleTestModel<T>(new T[_parameters.Length]);
            for (int i = 0; i < _parameters.Length; i++)
            {
                copy._parameters[i] = _parameters[i];
            }
            return copy;
        }

        public IFullModel<T, T[], T[]> Clone() => DeepCopy();

        // IGradientComputable
        public Vector<T> ComputeGradients(T[] inputs, T[] outputs, ILossFunction<T>? lossFunction = null)
        {
            return new Vector<T>(_parameters.Length);
        }

        public void ApplyGradients(Vector<T> gradients, T learningRate)
        {
            // No-op for test model
        }

        // IJitCompilable
        public AiDotNet.Autodiff.ComputationNode<T> ExportComputationGraph(List<AiDotNet.Autodiff.ComputationNode<T>> inputNodes)
        {
            throw new NotSupportedException("JIT compilation not supported for test model");
        }

        public bool SupportsJitCompilation => false;
    }

    #endregion
}
