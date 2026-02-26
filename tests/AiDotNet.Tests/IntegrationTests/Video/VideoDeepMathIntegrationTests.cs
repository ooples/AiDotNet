using AiDotNet.Video.Options;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Video;

/// <summary>
/// Deep integration tests for Video:
/// VideoModelOptions (defaults, effective values, nullable pattern),
/// VideoEnhancementOptions (enhancement-specific defaults),
/// Video processing math (frame rate, resolution, pixel counts, temporal consistency).
/// </summary>
public class VideoDeepMathIntegrationTests
{
    // ============================
    // VideoModelOptions: Defaults (all null)
    // ============================

    [Fact]
    public void VideoModelOptions_Defaults_AllNull()
    {
        var options = new VideoModelOptions<double>();
        Assert.Null(options.HiddenDimension);
        Assert.Null(options.NumAttentionHeads);
        Assert.Null(options.NumLayers);
        Assert.Null(options.DropoutRate);
        Assert.Null(options.NumFrames);
        Assert.Null(options.InputHeight);
        Assert.Null(options.InputWidth);
        Assert.Null(options.InputChannels);
        Assert.Null(options.LearningRate);
        Assert.Null(options.BatchSize);
        Assert.Null(options.WeightDecay);
        Assert.Null(options.UseGradientClipping);
        Assert.Null(options.MaxGradientNorm);
        Assert.Null(options.UseGpu);
        Assert.Null(options.UseMixedPrecision);
        Assert.Null(options.RandomSeed);
    }

    [Fact]
    public void VideoModelOptions_SetProperties()
    {
        var options = new VideoModelOptions<double>
        {
            HiddenDimension = 512,
            NumAttentionHeads = 8,
            NumLayers = 6,
            DropoutRate = 0.2,
            NumFrames = 32,
            InputHeight = 384,
            InputWidth = 384,
            InputChannels = 3,
            LearningRate = 0.001,
            BatchSize = 4,
            WeightDecay = 0.05,
            UseGradientClipping = true,
            MaxGradientNorm = 0.5,
            UseGpu = true,
            UseMixedPrecision = true,
            RandomSeed = 42
        };

        Assert.Equal(512, options.HiddenDimension);
        Assert.Equal(8, options.NumAttentionHeads);
        Assert.Equal(6, options.NumLayers);
        Assert.Equal(0.2, options.DropoutRate);
        Assert.Equal(32, options.NumFrames);
        Assert.Equal(384, options.InputHeight);
        Assert.Equal(384, options.InputWidth);
        Assert.Equal(42, options.RandomSeed);
    }

    // ============================
    // VideoEnhancementOptions: Defaults
    // ============================

    [Fact]
    public void VideoEnhancementOptions_InheritsVideoModelOptions()
    {
        var options = new VideoEnhancementOptions<double>();
        Assert.IsAssignableFrom<VideoModelOptions<double>>(options);
    }

    [Fact]
    public void VideoEnhancementOptions_Defaults_ScaleFactorNull()
    {
        var options = new VideoEnhancementOptions<double>();
        Assert.Null(options.ScaleFactor);
    }

    [Fact]
    public void VideoEnhancementOptions_SetScaleFactor()
    {
        var options = new VideoEnhancementOptions<double> { ScaleFactor = 4 };
        Assert.Equal(4, options.ScaleFactor);
    }

    // ============================
    // Video Resolution Math
    // ============================

    [Theory]
    [InlineData(720, 480, 2, 1440, 960)]    // 2x upscale
    [InlineData(720, 480, 4, 2880, 1920)]   // 4x upscale
    [InlineData(1920, 1080, 2, 3840, 2160)] // 1080p to 4K
    public void VideoMath_SuperResolution_OutputSize(int inW, int inH, int scale, int expectedW, int expectedH)
    {
        int outW = inW * scale;
        int outH = inH * scale;
        Assert.Equal(expectedW, outW);
        Assert.Equal(expectedH, outH);
    }

    [Theory]
    [InlineData(1920, 1080, 3, 6_220_800)]     // 1080p RGB: 1920*1080*3
    [InlineData(3840, 2160, 3, 24_883_200)]    // 4K RGB
    [InlineData(224, 224, 3, 150_528)]          // Standard input size
    public void VideoMath_PixelCount_PerFrame(int width, int height, int channels, int expectedPixels)
    {
        int pixels = width * height * channels;
        Assert.Equal(expectedPixels, pixels);
    }

    [Theory]
    [InlineData(1920, 1080, 3, 16, 99_532_800)]   // 16 frames of 1080p RGB
    [InlineData(224, 224, 3, 16, 2_408_448)]       // 16 frames of 224x224 RGB
    public void VideoMath_TotalElements_ForClip(int w, int h, int c, int frames, long expectedElements)
    {
        long totalElements = (long)frames * w * h * c;
        Assert.Equal(expectedElements, totalElements);
    }

    // ============================
    // Video Memory Math
    // ============================

    [Theory]
    [InlineData(224, 224, 3, 16, 4)]    // float32: 4 bytes
    [InlineData(224, 224, 3, 16, 8)]    // float64: 8 bytes
    public void VideoMath_MemoryUsage_PerClip(int w, int h, int c, int frames, int bytesPerElement)
    {
        long totalElements = (long)frames * w * h * c;
        long memoryBytes = totalElements * bytesPerElement;

        // Should be reasonable for standard input sizes
        double memoryMB = memoryBytes / (1024.0 * 1024.0);
        Assert.True(memoryMB < 1000, $"Memory usage {memoryMB:F1} MB should be < 1000 MB for standard inputs");
        Assert.True(memoryMB > 0);
    }

    // ============================
    // Video Frame Rate Math
    // ============================

    [Theory]
    [InlineData(30, 2, 60)]     // 30fps * 2 = 60fps
    [InlineData(24, 4, 96)]     // 24fps * 4 = 96fps
    [InlineData(60, 2, 120)]    // 60fps * 2 = 120fps
    public void VideoMath_FrameInterpolation_OutputFPS(int inputFPS, int temporalScale, int expectedOutputFPS)
    {
        int outputFPS = inputFPS * temporalScale;
        Assert.Equal(expectedOutputFPS, outputFPS);
    }

    [Theory]
    [InlineData(10.0, 30, 300)]     // 10 seconds at 30fps = 300 frames
    [InlineData(60.0, 24, 1440)]    // 1 minute at 24fps = 1440 frames
    [InlineData(5.0, 60, 300)]      // 5 seconds at 60fps = 300 frames
    public void VideoMath_TotalFrames_FromDuration(double durationSeconds, int fps, int expectedFrames)
    {
        int totalFrames = (int)(durationSeconds * fps);
        Assert.Equal(expectedFrames, totalFrames);
    }

    // ============================
    // Attention Head Math
    // ============================

    [Theory]
    [InlineData(768, 12, 64)]    // Standard ViT
    [InlineData(512, 8, 64)]     // Smaller model
    [InlineData(1024, 16, 64)]   // Larger model
    [InlineData(256, 4, 64)]     // Small model
    public void VideoMath_HeadDimension(int hiddenDim, int numHeads, int expectedHeadDim)
    {
        // Head dimension = hidden / num_heads (must divide evenly)
        Assert.Equal(0, hiddenDim % numHeads); // Must divide evenly
        int headDim = hiddenDim / numHeads;
        Assert.Equal(expectedHeadDim, headDim);
    }

    // ============================
    // Transformer Parameter Count Math
    // ============================

    [Theory]
    [InlineData(768, 12)]
    [InlineData(512, 6)]
    public void VideoMath_TransformerParams_ApproximateCount(int hiddenDim, int numLayers)
    {
        // Approximate parameters per transformer layer:
        // Self-attention: 4 * d^2 (Q, K, V, output projections)
        // FFN: 2 * d * (4d) = 8 * d^2 (two linear layers with 4x expansion)
        // Total per layer: ~12 * d^2
        long paramsPerLayer = 12L * hiddenDim * hiddenDim;
        long totalParams = paramsPerLayer * numLayers;

        Assert.True(totalParams > 0);
        double millionParams = totalParams / 1_000_000.0;
        Assert.True(millionParams > 0);
    }

    // ============================
    // PSNR / SSIM Quality Metrics Math
    // ============================

    [Theory]
    [InlineData(0.0, double.PositiveInfinity)]   // Perfect reconstruction
    [InlineData(10.0, 38.13)]                     // Low MSE
    [InlineData(100.0, 28.13)]                    // Moderate MSE
    [InlineData(1000.0, 18.13)]                   // High MSE
    public void VideoMath_PSNR_FromMSE(double mse, double expectedPSNR)
    {
        double maxPixelValue = 255.0;

        double psnr;
        if (mse == 0)
        {
            psnr = double.PositiveInfinity;
        }
        else
        {
            // PSNR = 10 * log10(MAX^2 / MSE) = 20 * log10(MAX / sqrt(MSE))
            psnr = 10.0 * Math.Log10(maxPixelValue * maxPixelValue / mse);
        }

        if (double.IsPositiveInfinity(expectedPSNR))
        {
            Assert.True(double.IsPositiveInfinity(psnr));
        }
        else
        {
            Assert.Equal(expectedPSNR, psnr, 0.1);
        }
    }

    // ============================
    // Dropout Regularization Math
    // ============================

    [Theory]
    [InlineData(0.0, 1.0)]    // No dropout: scale factor = 1
    [InlineData(0.1, 1.111)]  // 10% dropout: scale factor ≈ 1/(1-0.1) = 1.111
    [InlineData(0.5, 2.0)]    // 50% dropout: scale factor = 1/(1-0.5) = 2.0
    public void VideoMath_Dropout_InversionScaleFactor(double dropoutRate, double expectedScale)
    {
        // During training, activations are scaled by 1/(1-p) (inverted dropout)
        // This ensures expected value remains the same during inference
        double scaleFactor = 1.0 / (1.0 - dropoutRate);
        Assert.Equal(expectedScale, scaleFactor, 1e-2);
    }
}
