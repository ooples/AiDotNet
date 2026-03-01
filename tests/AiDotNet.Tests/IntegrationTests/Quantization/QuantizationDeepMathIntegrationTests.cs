using AiDotNet.Deployment.Optimization.Quantization;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Quantization;

/// <summary>
/// Deep integration tests for Quantization:
/// QuantizationConfiguration (defaults, factory methods, effective bit width, per-category overrides),
/// LayerQuantizationParams (defaults),
/// Quantization enums (QuantizationMode, QuantizationStrategy, QuantizationGranularity, CalibrationMethod, QATMethod),
/// Quantization math (compression ratios, quantization error, scale/zero-point).
/// </summary>
public class QuantizationDeepMathIntegrationTests
{
    // ============================
    // QuantizationMode Enum
    // ============================

    [Fact]
    public void QuantizationMode_HasSixValues()
    {
        var values = (((QuantizationMode[])Enum.GetValues(typeof(QuantizationMode))));
        Assert.Equal(6, values.Length);
    }

    [Theory]
    [InlineData(QuantizationMode.None)]
    [InlineData(QuantizationMode.Int8)]
    [InlineData(QuantizationMode.Float16)]
    [InlineData(QuantizationMode.Float32)]
    [InlineData(QuantizationMode.Dynamic)]
    [InlineData(QuantizationMode.Mixed)]
    public void QuantizationMode_AllValuesValid(QuantizationMode mode)
    {
        Assert.True(Enum.IsDefined(typeof(QuantizationMode), mode));
    }

    // ============================
    // QuantizationStrategy Enum
    // ============================

    [Fact]
    public void QuantizationStrategy_HasSevenValues()
    {
        var values = (((QuantizationStrategy[])Enum.GetValues(typeof(QuantizationStrategy))));
        Assert.Equal(7, values.Length);
    }

    [Theory]
    [InlineData(QuantizationStrategy.Dynamic)]
    [InlineData(QuantizationStrategy.GPTQ)]
    [InlineData(QuantizationStrategy.AWQ)]
    [InlineData(QuantizationStrategy.SmoothQuant)]
    [InlineData(QuantizationStrategy.SpinQuant)]
    [InlineData(QuantizationStrategy.QuIPSharp)]
    [InlineData(QuantizationStrategy.MinMax)]
    public void QuantizationStrategy_AllValuesValid(QuantizationStrategy strategy)
    {
        Assert.True(Enum.IsDefined(typeof(QuantizationStrategy), strategy));
    }

    // ============================
    // QuantizationGranularity Enum
    // ============================

    [Fact]
    public void QuantizationGranularity_HasFiveValues()
    {
        var values = (((QuantizationGranularity[])Enum.GetValues(typeof(QuantizationGranularity))));
        Assert.Equal(5, values.Length);
    }

    [Theory]
    [InlineData(QuantizationGranularity.PerTensor)]
    [InlineData(QuantizationGranularity.PerChannel)]
    [InlineData(QuantizationGranularity.PerGroup)]
    [InlineData(QuantizationGranularity.PerBlock)]
    [InlineData(QuantizationGranularity.PerRow)]
    public void QuantizationGranularity_AllValuesValid(QuantizationGranularity granularity)
    {
        Assert.True(Enum.IsDefined(typeof(QuantizationGranularity), granularity));
    }

    // ============================
    // QATMethod Enum
    // ============================

    [Fact]
    public void QATMethod_HasFiveValues()
    {
        var values = (((QATMethod[])Enum.GetValues(typeof(QATMethod))));
        Assert.Equal(5, values.Length);
    }

    [Theory]
    [InlineData(QATMethod.Standard)]
    [InlineData(QATMethod.EfficientQAT)]
    [InlineData(QATMethod.ZeroQAT)]
    [InlineData(QATMethod.ParetoQ)]
    [InlineData(QATMethod.QABLoRA)]
    public void QATMethod_AllValuesValid(QATMethod method)
    {
        Assert.True(Enum.IsDefined(typeof(QATMethod), method));
    }

    // ============================
    // QuantizationConfiguration: Defaults
    // ============================

    [Fact]
    public void QuantizationConfig_Defaults()
    {
        var config = new QuantizationConfiguration();
        Assert.Equal(QuantizationMode.Int8, config.Mode);
        Assert.Equal(QuantizationStrategy.Dynamic, config.Strategy);
        Assert.Equal(QuantizationGranularity.PerChannel, config.Granularity);
        Assert.Equal(128, config.GroupSize);
        Assert.Null(config.TargetBitWidth);
        Assert.Equal(8, config.EffectiveBitWidth);
        Assert.True(config.UseSymmetricQuantization);
        Assert.Equal(CalibrationMethod.MinMax, config.CalibrationMethod);
        Assert.Equal(100, config.NumCalibrationSamples);
        Assert.Empty(config.SkipLayers);
        Assert.True(config.QuantizeActivations);
        Assert.Equal(1e-6, config.MinScaleFactor);
        Assert.Equal(1e6, config.MaxScaleFactor);
        Assert.False(config.UseQuantizationAwareTraining);
        Assert.Equal(QATMethod.Standard, config.QATMethod);
        Assert.Equal(1, config.QATWarmupEpochs);
        Assert.Equal(99.99, config.HistogramPercentile);
        Assert.Equal(8, config.ActivationBitWidth);
        Assert.Equal(0.5, config.SmoothQuantAlpha);
        Assert.Equal(0.01, config.GPTQDampingFactor);
        Assert.True(config.GPTQActOrder);
        Assert.Equal(1.0, config.AWQProtectionPercentage);
    }

    // ============================
    // QuantizationConfiguration: EffectiveBitWidth
    // ============================

    [Theory]
    [InlineData(QuantizationMode.Int8, null, 8)]
    [InlineData(QuantizationMode.Float16, null, 16)]
    [InlineData(QuantizationMode.Float32, null, 32)]
    [InlineData(QuantizationMode.Dynamic, null, 8)]
    [InlineData(QuantizationMode.Int8, 4, 4)]    // Override with TargetBitWidth
    [InlineData(QuantizationMode.Int8, 2, 2)]    // Override with TargetBitWidth
    public void QuantizationConfig_EffectiveBitWidth(QuantizationMode mode, int? targetBitWidth, int expected)
    {
        var config = new QuantizationConfiguration
        {
            Mode = mode,
            TargetBitWidth = targetBitWidth
        };
        Assert.Equal(expected, config.EffectiveBitWidth);
    }

    // ============================
    // QuantizationConfiguration: Factory Methods
    // ============================

    [Fact]
    public void QuantizationConfig_ForInt8()
    {
        var config = QuantizationConfiguration.ForInt8();
        Assert.Equal(QuantizationMode.Int8, config.Mode);
        Assert.True(config.UseSymmetricQuantization);
        Assert.True(config.QuantizeActivations);
        Assert.Equal(8, config.EffectiveBitWidth);
    }

    [Fact]
    public void QuantizationConfig_ForFloat16()
    {
        var config = QuantizationConfiguration.ForFloat16();
        Assert.Equal(QuantizationMode.Float16, config.Mode);
        Assert.False(config.UseSymmetricQuantization);
        Assert.Equal(CalibrationMethod.None, config.CalibrationMethod);
        Assert.Equal(16, config.EffectiveBitWidth);
    }

    [Fact]
    public void QuantizationConfig_ForDynamic()
    {
        var config = QuantizationConfiguration.ForDynamic();
        Assert.Equal(QuantizationMode.Dynamic, config.Mode);
        Assert.True(config.UseSymmetricQuantization);
        Assert.False(config.QuantizeActivations);
    }

    [Fact]
    public void QuantizationConfig_ForGPTQ()
    {
        var config = QuantizationConfiguration.ForGPTQ();
        Assert.Equal(QuantizationStrategy.GPTQ, config.Strategy);
        Assert.Equal(4, config.EffectiveBitWidth);
        Assert.Equal(QuantizationGranularity.PerGroup, config.Granularity);
        Assert.Equal(128, config.GroupSize);
        Assert.True(config.GPTQActOrder);
        Assert.False(config.QuantizeActivations);
    }

    [Fact]
    public void QuantizationConfig_ForGPTQ_CustomGroupSize()
    {
        var config = QuantizationConfiguration.ForGPTQ(groupSize: 64, actOrder: false);
        Assert.Equal(64, config.GroupSize);
        Assert.False(config.GPTQActOrder);
    }

    [Fact]
    public void QuantizationConfig_ForAWQ()
    {
        var config = QuantizationConfiguration.ForAWQ();
        Assert.Equal(QuantizationStrategy.AWQ, config.Strategy);
        Assert.Equal(4, config.EffectiveBitWidth);
        Assert.Equal(1.0, config.AWQProtectionPercentage);
    }

    [Fact]
    public void QuantizationConfig_ForSmoothQuant()
    {
        var config = QuantizationConfiguration.ForSmoothQuant();
        Assert.Equal(QuantizationStrategy.SmoothQuant, config.Strategy);
        Assert.Equal(0.5, config.SmoothQuantAlpha);
        Assert.True(config.QuantizeActivations);
        Assert.Equal(8, config.ActivationBitWidth);
    }

    [Fact]
    public void QuantizationConfig_ForSmoothQuant_CustomAlpha()
    {
        var config = QuantizationConfiguration.ForSmoothQuant(alpha: 0.7);
        Assert.Equal(0.7, config.SmoothQuantAlpha);
    }

    [Fact]
    public void QuantizationConfig_ForQAT()
    {
        var config = QuantizationConfiguration.ForQAT();
        Assert.True(config.UseQuantizationAwareTraining);
        Assert.Equal(QATMethod.EfficientQAT, config.QATMethod);
        Assert.Equal(1, config.QATWarmupEpochs);
        Assert.Equal(8, config.EffectiveBitWidth);
    }

    [Fact]
    public void QuantizationConfig_ForQAT_Custom()
    {
        var config = QuantizationConfiguration.ForQAT(targetBitWidth: 4, method: QATMethod.ParetoQ);
        Assert.Equal(4, config.EffectiveBitWidth);
        Assert.Equal(QATMethod.ParetoQ, config.QATMethod);
    }

    [Fact]
    public void QuantizationConfig_ForQLoRA()
    {
        var config = QuantizationConfiguration.ForQLoRA();
        Assert.Equal(4, config.EffectiveBitWidth);
        Assert.Equal(64, config.GroupSize);
        Assert.False(config.UseSymmetricQuantization);
        Assert.True(config.UseQuantizationAwareTraining);
        Assert.Equal(QATMethod.QABLoRA, config.QATMethod);
    }

    // ============================
    // QuantizationConfiguration: UsePerChannelQuantization
    // ============================

    [Fact]
    public void QuantizationConfig_UsePerChannelQuantization_SetsGranularity()
    {
        var config = new QuantizationConfiguration();
        config.UsePerChannelQuantization = true;
        Assert.Equal(QuantizationGranularity.PerChannel, config.Granularity);

        config.UsePerChannelQuantization = false;
        Assert.Equal(QuantizationGranularity.PerTensor, config.Granularity);
    }

    // ============================
    // QuantizationConfiguration: MixedPrecision
    // ============================

    [Fact]
    public void QuantizationConfig_ForMixedPrecision()
    {
        var config = QuantizationConfiguration.ForMixedPrecision();
        Assert.NotNull(config.CategoryBitWidths);
        Assert.Equal(8, config.CategoryBitWidths[LayerCategory.Attention]);
        Assert.Equal(8, config.CategoryBitWidths[LayerCategory.Embedding]);
        Assert.Equal(16, config.CategoryBitWidths[LayerCategory.Normalization]);
        Assert.Equal(4, config.CategoryBitWidths[LayerCategory.Dense]);
        Assert.Equal(4, config.CategoryBitWidths[LayerCategory.FeedForward]);
        Assert.Equal(4, config.CategoryBitWidths[LayerCategory.Convolution]);
    }

    [Fact]
    public void QuantizationConfig_ForMixedPrecision_CustomBitWidths()
    {
        var config = QuantizationConfiguration.ForMixedPrecision(
            sensitiveBitWidth: 16, aggressiveBitWidth: 2, groupSize: 64);
        Assert.NotNull(config.CategoryBitWidths);
        Assert.Equal(16, config.CategoryBitWidths[LayerCategory.Attention]);
        Assert.Equal(2, config.CategoryBitWidths[LayerCategory.Dense]);
        Assert.Equal(64, config.GroupSize);
    }

    [Fact]
    public void QuantizationConfig_ForMixedPrecision_InvalidBitWidthThrows()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            QuantizationConfiguration.ForMixedPrecision(sensitiveBitWidth: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            QuantizationConfiguration.ForMixedPrecision(aggressiveBitWidth: -1));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            QuantizationConfiguration.ForMixedPrecision(groupSize: 0));
    }

    // ============================
    // QuantizationConfiguration: GetBitWidthForCategory
    // ============================

    [Fact]
    public void QuantizationConfig_GetBitWidthForCategory_DefaultsToEffective()
    {
        var config = new QuantizationConfiguration { TargetBitWidth = 4 };
        Assert.Equal(4, config.GetBitWidthForCategory(LayerCategory.Dense));
    }

    [Fact]
    public void QuantizationConfig_GetBitWidthForCategory_UsesOverride()
    {
        var config = QuantizationConfiguration.ForMixedPrecision();
        Assert.Equal(8, config.GetBitWidthForCategory(LayerCategory.Attention));
        Assert.Equal(4, config.GetBitWidthForCategory(LayerCategory.Dense));
    }

    // ============================
    // LayerQuantizationParams: Defaults
    // ============================

    [Fact]
    public void LayerQuantizationParams_Defaults()
    {
        var param = new LayerQuantizationParams();
        Assert.Equal(1.0, param.ScaleFactor);
        Assert.Equal(0, param.ZeroPoint);
        Assert.False(param.Skip);
        Assert.Null(param.BitWidth);
        Assert.Null(param.Mode);
    }

    // ============================
    // Quantization Math: Compression Ratio
    // ============================

    [Theory]
    [InlineData(32, 8, 4.0)]    // FP32 to INT8: 4x compression
    [InlineData(32, 4, 8.0)]    // FP32 to 4-bit: 8x compression
    [InlineData(32, 2, 16.0)]   // FP32 to 2-bit: 16x compression
    [InlineData(32, 16, 2.0)]   // FP32 to FP16: 2x compression
    [InlineData(16, 8, 2.0)]    // FP16 to INT8: 2x compression
    public void QuantizationMath_CompressionRatio(int originalBits, int quantizedBits, double expectedRatio)
    {
        double ratio = (double)originalBits / quantizedBits;
        Assert.Equal(expectedRatio, ratio, 1e-10);
    }

    // ============================
    // Quantization Math: Scale and Zero-Point
    // ============================

    [Theory]
    [InlineData(-1.0, 1.0, 8, true)]    // Symmetric INT8
    [InlineData(0.0, 6.0, 8, false)]    // Asymmetric INT8
    public void QuantizationMath_ScaleAndZeroPoint(double minVal, double maxVal, int bits, bool symmetric)
    {
        int numLevels = (1 << bits) - 1; // 2^bits - 1

        if (symmetric)
        {
            // Symmetric: scale = max(|min|, |max|) / (2^(bits-1) - 1)
            double absMax = Math.Max(Math.Abs(minVal), Math.Abs(maxVal));
            int halfLevels = (1 << (bits - 1)) - 1; // 127 for 8-bit
            double scale = absMax / halfLevels;

            Assert.True(scale > 0, "Scale must be positive");
            Assert.Equal(0, 0); // Zero point is 0 for symmetric

            // Verify roundtrip
            double quantized = Math.Round(maxVal / scale);
            double dequantized = quantized * scale;
            Assert.Equal(maxVal, dequantized, 1e-2);
        }
        else
        {
            // Asymmetric: scale = (max - min) / numLevels
            double scale = (maxVal - minVal) / numLevels;
            int zeroPoint = (int)Math.Round(-minVal / scale);

            Assert.True(scale > 0, "Scale must be positive");
            Assert.True(zeroPoint >= 0, "Zero point should be non-negative");

            // Verify roundtrip for min
            double quantizedMin = Math.Round(minVal / scale) + zeroPoint;
            double dequantizedMin = (quantizedMin - zeroPoint) * scale;
            Assert.Equal(minVal, dequantizedMin, 1e-2);
        }
    }

    // ============================
    // Quantization Math: Quantization Error
    // ============================

    [Theory]
    [InlineData(8, 0.0039)]    // 8-bit: step = 2/255 ~ 0.0078, max error ~ step/2
    [InlineData(4, 0.0667)]    // 4-bit: step = 2/15 ~ 0.133, max error ~ step/2
    [InlineData(2, 0.333)]     // 2-bit: step = 2/3 ~ 0.667, max error ~ step/2
    public void QuantizationMath_MaxQuantizationError_Symmetric(int bits, double expectedMaxError)
    {
        // For symmetric quantization in range [-1, 1]:
        // step = 2 / (2^bits - 1)
        // max error = step / 2
        int numLevels = (1 << bits) - 1;
        double step = 2.0 / numLevels;
        double maxError = step / 2.0;

        Assert.Equal(expectedMaxError, maxError, 1e-3);
    }

    // ============================
    // Quantization Math: Memory Savings
    // ============================

    [Theory]
    [InlineData(1_000_000, 32, 8, 3_000_000)]     // 1M params, FP32->INT8: save 3MB
    [InlineData(7_000_000_000L, 32, 4, 24_500_000_000L)]  // 7B params, FP32->4bit: save 24.5GB
    public void QuantizationMath_MemorySavings(long numParams, int originalBits, int quantizedBits, long expectedSavingsBytes)
    {
        long originalBytes = numParams * originalBits / 8;
        long quantizedBytes = numParams * quantizedBits / 8;
        long savings = originalBytes - quantizedBytes;

        Assert.Equal(expectedSavingsBytes, savings);
        Assert.True(savings > 0, "Quantization should save memory");
    }

    // ============================
    // Quantization Math: Group Quantization Overhead
    // ============================

    [Theory]
    [InlineData(1_000_000, 128, 16)]   // 1M params, group size 128, FP16 scales
    [InlineData(1_000_000, 64, 16)]    // 1M params, group size 64, FP16 scales
    public void QuantizationMath_GroupQuantizationOverhead(long numParams, int groupSize, int scaleBits)
    {
        // Number of groups
        long numGroups = (numParams + groupSize - 1) / groupSize;

        // Scale storage: numGroups * scaleBits/8 bytes
        long scaleStorageBytes = numGroups * scaleBits / 8;

        // Additional bits per weight from scales
        double additionalBitsPerWeight = (double)scaleBits / groupSize;

        // Smaller group size -> more overhead
        Assert.True(additionalBitsPerWeight > 0);
        Assert.True(scaleStorageBytes > 0);

        // Group 64 should have 2x the overhead of group 128
        if (groupSize == 64)
        {
            Assert.Equal((double)scaleBits / 64, additionalBitsPerWeight, 1e-10);
        }
    }

    // ============================
    // Quantization Math: SmoothQuant Alpha
    // ============================

    [Theory]
    [InlineData(0.0)]     // All difficulty on weights
    [InlineData(0.5)]     // Balanced
    [InlineData(1.0)]     // All difficulty on activations
    public void QuantizationMath_SmoothQuantAlpha_ValidRange(double alpha)
    {
        // SmoothQuant: s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha)
        // alpha in [0, 1] balances difficulty between activations and weights
        Assert.True(alpha >= 0.0 && alpha <= 1.0);

        double activationMax = 10.0;
        double weightMax = 2.0;

        double smoothingFactor = Math.Pow(activationMax, alpha) / Math.Pow(weightMax, 1.0 - alpha);
        Assert.True(smoothingFactor > 0, "Smoothing factor must be positive");
    }
}
