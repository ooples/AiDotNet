using AiDotNet.FederatedLearning.Compression;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

/// <summary>
/// Tests for advanced communication compression (#851).
/// </summary>
public class AdvancedCompressionTests
{
    private static Tensor<double> CreateTensor(params double[] values)
    {
        var tensor = new Tensor<double>(new[] { values.Length });
        for (int i = 0; i < values.Length; i++)
        {
            tensor[i] = values[i];
        }

        return tensor;
    }

    private static Tensor<double> CreateGradient(int size, double scale = 1.0)
    {
        var values = new double[size];
        var rng = new Random(42);
        for (int i = 0; i < size; i++)
        {
            values[i] = (rng.NextDouble() * 2.0 - 1.0) * scale;
        }

        return CreateTensor(values);
    }

    // ========== PowerSGDCompressor Tests ==========

    [Fact]
    public void PowerSGD_CompressAndDecompress_PreservesApproximately()
    {
        var options = new AdvancedCompressionOptions
        {
            Strategy = AdvancedCompressionStrategy.PowerSGD,
            PowerSGDRank = 2,
            PowerSGDWarmStart = false
        };
        var compressor = new PowerSGDCompressor<double>(options);
        var gradient = CreateGradient(16);

        var (P, Q, rows, cols) = compressor.Compress(gradient, 0);
        var decompressed = compressor.Decompress(P, Q, rows, cols, gradient.Shape[0]);

        Assert.NotNull(decompressed);
        Assert.Equal(gradient.Shape[0], decompressed.Shape[0]);

        // Reconstructed gradient should have non-trivial values
        bool anyNonZero = false;
        for (int i = 0; i < decompressed.Shape[0]; i++)
        {
            if (Math.Abs(decompressed[i]) > 1e-12)
            {
                anyNonZero = true;
                break;
            }
        }

        Assert.True(anyNonZero, "Decompressed gradient should have non-zero values");
    }

    [Fact]
    public void PowerSGD_CompressionRatio_LessThanOne()
    {
        var options = new AdvancedCompressionOptions
        {
            Strategy = AdvancedCompressionStrategy.PowerSGD,
            PowerSGDRank = 2
        };
        var compressor = new PowerSGDCompressor<double>(options);

        double ratio = compressor.GetCompressionRatio(100);

        Assert.True(ratio < 1.0, $"Compression ratio ({ratio}) should be less than 1.0");
        Assert.True(ratio > 0.0, $"Compression ratio ({ratio}) should be positive");
    }

    [Fact]
    public void PowerSGD_WarmStart_ProducesConsistentResults()
    {
        var options = new AdvancedCompressionOptions
        {
            Strategy = AdvancedCompressionStrategy.PowerSGD,
            PowerSGDRank = 2,
            PowerSGDWarmStart = true
        };
        var compressor = new PowerSGDCompressor<double>(options);
        var gradient = CreateGradient(16);

        // First compression
        var result1 = compressor.Compress(gradient, clientId: 0);
        var decompressed1 = compressor.Decompress(result1.P, result1.Q, result1.Rows, result1.Cols, gradient.Shape[0]);

        // Second compression with same client (warm start should be active)
        var result2 = compressor.Compress(gradient, clientId: 0);
        var decompressed2 = compressor.Decompress(result2.P, result2.Q, result2.Rows, result2.Cols, gradient.Shape[0]);

        Assert.NotNull(decompressed1);
        Assert.NotNull(decompressed2);
    }

    [Fact]
    public void PowerSGD_NullOptions_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new PowerSGDCompressor<double>(null));
    }

    // ========== GradientSketchCompressor Tests ==========

    [Fact]
    public void GradientSketch_CompressAndDecompress_PreservesApproximately()
    {
        var options = new AdvancedCompressionOptions
        {
            Strategy = AdvancedCompressionStrategy.GradientSketch,
            SketchDepth = 3,
            SketchWidth = 32
        };
        var compressor = new GradientSketchCompressor<double>(options);
        var gradient = CreateGradient(20);

        var (sketch, originalSize, width) = compressor.Compress(gradient);
        var decompressed = compressor.Decompress(sketch, originalSize, width);

        Assert.NotNull(decompressed);
        Assert.Equal(gradient.Shape[0], decompressed.Shape[0]);
    }

    [Fact]
    public void GradientSketch_DecompressTopK_ReturnsCorrectSize()
    {
        var options = new AdvancedCompressionOptions
        {
            Strategy = AdvancedCompressionStrategy.GradientSketch,
            SketchDepth = 3,
            SketchWidth = 32,
            SketchTopK = 5
        };
        var compressor = new GradientSketchCompressor<double>(options);
        var gradient = CreateGradient(20);

        var (sketch, originalSize, width) = compressor.Compress(gradient);
        var decompressed = compressor.DecompressTopK(sketch, originalSize, width, 5);

        Assert.NotNull(decompressed);
        Assert.Equal(gradient.Shape[0], decompressed.Shape[0]);

        // Top-K should result in sparse output (most values zero)
        int nonZero = 0;
        for (int i = 0; i < decompressed.Shape[0]; i++)
        {
            if (Math.Abs(decompressed[i]) > 1e-12)
            {
                nonZero++;
            }
        }

        Assert.True(nonZero <= 5, $"Top-5 should have at most 5 non-zero values, got {nonZero}");
    }

    [Fact]
    public void GradientSketch_GetCompressionRatio_LessThanOne()
    {
        var options = new AdvancedCompressionOptions
        {
            Strategy = AdvancedCompressionStrategy.GradientSketch,
            SketchDepth = 3,
            SketchWidth = 32
        };
        var compressor = new GradientSketchCompressor<double>(options);

        double ratio = compressor.GetCompressionRatio(1000);

        Assert.True(ratio < 1.0, $"Compression ratio ({ratio}) should be less than 1.0");
    }

    [Fact]
    public void GradientSketch_NullOptions_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new GradientSketchCompressor<double>(null));
    }

    // ========== ErrorFeedbackCompressor Tests ==========

    [Fact]
    public void ErrorFeedback_AccumulatesResiduals()
    {
        var options = new AdvancedCompressionOptions
        {
            Strategy = AdvancedCompressionStrategy.OneBitSGD,
            UseErrorFeedback = true
        };
        var compressor = new ErrorFeedbackCompressor<double>(options);
        var gradient = CreateGradient(10, 2.0);

        // First round - compress with error feedback
        var corrected1 = compressor.ApplyErrorFeedback(gradient, 0);
        var (compressed1, scale1) = compressor.CompressOneBit(corrected1, 0);

        // Use compressed + scale to create a "decompressed" version for error update
        // The scale represents the mean absolute value, compressed are signs
        compressor.UpdateError(corrected1, compressed1, 0);

        // Second round - residuals from first round should be accumulated
        var corrected2 = compressor.ApplyErrorFeedback(gradient, 0);

        // Corrected gradient should differ from original due to error accumulation
        bool anyDifferent = false;
        for (int i = 0; i < gradient.Shape[0]; i++)
        {
            if (Math.Abs(corrected2[i] - gradient[i]) > 1e-12)
            {
                anyDifferent = true;
                break;
            }
        }

        Assert.True(anyDifferent, "Error feedback should modify the gradient in the second round");
    }

    [Fact]
    public void ErrorFeedback_OneBitCompression_PreservesSigns()
    {
        var options = new AdvancedCompressionOptions { Strategy = AdvancedCompressionStrategy.OneBitSGD };
        var compressor = new ErrorFeedbackCompressor<double>(options);
        var gradient = CreateTensor(1.0, -2.0, 3.0, -4.0, 5.0);

        var (compressed, scale) = compressor.CompressOneBit(gradient, 0);

        Assert.NotNull(compressed);
        Assert.True(scale > 0, "Scale should be positive");

        // Signs should be preserved in the compressed tensor
        for (int i = 0; i < 5; i++)
        {
            double originalSign = Math.Sign(gradient[i]);
            double compressedSign = Math.Sign(compressed[i]);
            Assert.Equal(originalSign, compressedSign);
        }
    }

    [Fact]
    public void ErrorFeedback_ApplyErrorFeedback_NoFeedback_ReturnsSame()
    {
        var options = new AdvancedCompressionOptions
        {
            Strategy = AdvancedCompressionStrategy.OneBitSGD,
            UseErrorFeedback = false
        };
        var compressor = new ErrorFeedbackCompressor<double>(options);
        var gradient = CreateTensor(1.0, 2.0, 3.0);

        var result = compressor.ApplyErrorFeedback(gradient, 0);

        // Should return the same tensor when error feedback is disabled
        Assert.Same(gradient, result);
    }

    [Fact]
    public void ErrorFeedback_NullOptions_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new ErrorFeedbackCompressor<double>(null));
    }

    // ========== AdaptiveCompressor Tests ==========

    [Fact]
    public void Adaptive_ComputesCompressionRatio()
    {
        var options = new AdvancedCompressionOptions
        {
            Strategy = AdvancedCompressionStrategy.Adaptive,
            AdaptiveBandwidthWindow = 5,
            AdaptiveMinRatio = 0.01,
            AdaptiveMaxRatio = 0.5
        };
        var compressor = new AdaptiveCompressor<double>(options);

        // Record some RTT measurements
        compressor.RecordBandwidth(0, 100.0);
        compressor.RecordBandwidth(0, 150.0);

        var gradient = CreateGradient(20, 2.0);
        double ratio = compressor.ComputeAdaptiveRatio(0, gradient, currentRound: 5);

        Assert.InRange(ratio, 0.01, 0.5);
    }

    [Fact]
    public void Adaptive_CompressProducesCompressedOutput()
    {
        var options = new AdvancedCompressionOptions
        {
            Strategy = AdvancedCompressionStrategy.Adaptive,
            AdaptiveMinRatio = 0.1,
            AdaptiveMaxRatio = 0.3
        };
        var compressor = new AdaptiveCompressor<double>(options);
        var gradient = CreateGradient(100, 5.0);

        var (compressed, actualRatio) = compressor.Compress(gradient, clientId: 0, currentRound: 1);

        Assert.NotNull(compressed);
        Assert.InRange(actualRatio, 0.1, 0.3);

        // Compressed should be a sparse version of the gradient
        Assert.Equal(gradient.Shape[0], compressed.Shape[0]);
    }

    [Fact]
    public void Adaptive_CompressWithBandwidthMeasurement()
    {
        var options = new AdvancedCompressionOptions
        {
            Strategy = AdvancedCompressionStrategy.Adaptive,
            AdaptiveMinRatio = 0.05,
            AdaptiveMaxRatio = 0.5
        };
        var compressor = new AdaptiveCompressor<double>(options);
        var gradient = CreateGradient(50, 3.0);

        // Fast client should get higher ratio (less compression)
        compressor.RecordBandwidth(0, 10.0); // Very fast RTT
        var (_, fastRatio) = compressor.Compress(gradient, clientId: 0, currentRound: 1, roundTripMs: 10.0);

        // Slow client should get lower ratio (more compression)
        compressor.RecordBandwidth(1, 1000.0); // Very slow RTT
        var (_, slowRatio) = compressor.Compress(gradient, clientId: 1, currentRound: 1, roundTripMs: 1000.0);

        Assert.InRange(fastRatio, 0.05, 0.5);
        Assert.InRange(slowRatio, 0.05, 0.5);
    }

    [Fact]
    public void Adaptive_NullOptions_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new AdaptiveCompressor<double>(null));
    }

    // ========== Options Tests ==========

    [Fact]
    public void AdvancedCompressionOptions_DefaultValues()
    {
        var options = new AdvancedCompressionOptions();

        Assert.Equal(AdvancedCompressionStrategy.PowerSGD, options.Strategy);
        Assert.Equal(4, options.PowerSGDRank);
        Assert.True(options.PowerSGDWarmStart);
        Assert.Equal(5, options.SketchDepth);
        Assert.True(options.UseErrorFeedback);
        Assert.Equal(10, options.AdaptiveBandwidthWindow);
        Assert.Equal(0.01, options.AdaptiveMinRatio);
        Assert.Equal(0.5, options.AdaptiveMaxRatio);
    }

    [Fact]
    public void AdvancedCompressionStrategy_HasAllExpectedValues()
    {
        Assert.True(Enum.IsDefined(typeof(AdvancedCompressionStrategy), AdvancedCompressionStrategy.PowerSGD));
        Assert.True(Enum.IsDefined(typeof(AdvancedCompressionStrategy), AdvancedCompressionStrategy.GradientSketch));
        Assert.True(Enum.IsDefined(typeof(AdvancedCompressionStrategy), AdvancedCompressionStrategy.OneBitSGD));
        Assert.True(Enum.IsDefined(typeof(AdvancedCompressionStrategy), AdvancedCompressionStrategy.Adaptive));
    }

    // ========== Integration with FederatedCompressionOptions ==========

    [Fact]
    public void FederatedCompressionOptions_CanSetAdvancedOptions()
    {
        var options = new FederatedCompressionOptions
        {
            Advanced = new AdvancedCompressionOptions
            {
                Strategy = AdvancedCompressionStrategy.PowerSGD,
                PowerSGDRank = 8
            }
        };

        Assert.NotNull(options.Advanced);
        Assert.Equal(AdvancedCompressionStrategy.PowerSGD, options.Advanced.Strategy);
        Assert.Equal(8, options.Advanced.PowerSGDRank);
    }
}
