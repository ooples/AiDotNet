using System;
using AiDotNet.Metrics;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNetTests.UnitTests.Metrics;

/// <summary>
/// Tests for audio, image, and video evaluation metrics.
/// </summary>
public class EvaluationMetricsTests
{
    #region Audio Metrics Tests

    [Fact]
    public void WordErrorRate_IdenticalStrings_ReturnsZero()
    {
        // Arrange
        var wer = new WordErrorRate();
        string reference = "the quick brown fox";
        string hypothesis = "the quick brown fox";

        // Act
        double result = wer.Compute(hypothesis, reference);

        // Assert
        Assert.Equal(0.0, result, 6);
    }

    [Fact]
    public void WordErrorRate_CompletelyDifferent_ReturnsOne()
    {
        // Arrange
        var wer = new WordErrorRate();
        string reference = "the quick brown fox";
        string hypothesis = "a lazy dog jumped";

        // Act
        double result = wer.Compute(hypothesis, reference);

        // Assert
        Assert.Equal(1.0, result, 6);
    }

    [Fact]
    public void WordErrorRate_OneSubstitution_ReturnsCorrectRate()
    {
        // Arrange
        var wer = new WordErrorRate();
        string reference = "the quick brown fox";
        string hypothesis = "the fast brown fox";

        // Act
        double result = wer.Compute(hypothesis, reference);

        // Assert
        // 1 substitution / 4 words = 0.25
        Assert.Equal(0.25, result, 6);
    }

    [Fact]
    public void WordErrorRate_EmptyReference_ReturnsOne()
    {
        // Arrange
        var wer = new WordErrorRate();
        string reference = "";
        string hypothesis = "hello world";

        // Act
        double result = wer.Compute(hypothesis, reference);

        // Assert
        Assert.Equal(1.0, result, 6);
    }

    [Fact]
    public void WordErrorRate_EmptyBoth_ReturnsZero()
    {
        // Arrange
        var wer = new WordErrorRate();
        string reference = "";
        string hypothesis = "";

        // Act
        double result = wer.Compute(hypothesis, reference);

        // Assert
        Assert.Equal(0.0, result, 6);
    }

    [Fact]
    public void WordErrorRate_DetailedOutput_ContainsCorrectCounts()
    {
        // Arrange
        var wer = new WordErrorRate();
        string reference = "the quick brown fox";
        string hypothesis = "the fast brown dog jumped";

        // Act
        var (werScore, subs, ins, dels, refCount) = wer.ComputeDetailed(hypothesis, reference);

        // Assert
        Assert.Equal(4, refCount);
        Assert.True(werScore > 0);
    }

    [Fact]
    public void WordErrorRate_BatchCompute_ReturnsAverageWER()
    {
        // Arrange
        var wer = new WordErrorRate();
        var references = new[] { "hello world", "the quick brown fox" };
        var hypotheses = new[] { "hello world", "the fast brown fox" };

        // Act
        double result = wer.ComputeBatch(hypotheses, references);

        // Assert
        // First pair: 0, Second pair: 0.25, Average: 0.125
        Assert.Equal(0.125, result, 6);
    }

    [Fact]
    public void CharacterErrorRate_IdenticalStrings_ReturnsZero()
    {
        // Arrange
        var cer = new CharacterErrorRate();
        string reference = "hello";
        string hypothesis = "hello";

        // Act
        double result = cer.Compute(hypothesis, reference);

        // Assert
        Assert.Equal(0.0, result, 6);
    }

    [Fact]
    public void CharacterErrorRate_OneSubstitution_ReturnsCorrectRate()
    {
        // Arrange
        var cer = new CharacterErrorRate();
        string reference = "hello";
        string hypothesis = "hallo";

        // Act
        double result = cer.Compute(hypothesis, reference);

        // Assert
        // 1 substitution / 5 characters = 0.2
        Assert.Equal(0.2, result, 6);
    }

    [Fact]
    public void CharacterErrorRate_IgnoreWhitespace_Works()
    {
        // Arrange
        var cer = new CharacterErrorRate();
        string reference = "hello world";
        string hypothesis = "helloworld";

        // Act
        double resultWithWs = cer.Compute(hypothesis, reference, ignoreWhitespace: false);
        double resultNoWs = cer.Compute(hypothesis, reference, ignoreWhitespace: true);

        // Assert
        Assert.True(resultWithWs > resultNoWs);
    }

    [Fact]
    public void ShortTimeObjectiveIntelligibility_IdenticalSignals_ReturnsHighScore()
    {
        // Arrange
        var stoi = new ShortTimeObjectiveIntelligibility<double>(sampleRate: 16000);
        var signal = new Tensor<double>(new[] { 1600 }); // 100ms at 16kHz
        var random = new Random(42);
        for (int i = 0; i < signal.Length; i++)
        {
            signal[i] = Math.Sin(2 * Math.PI * 440 * i / 16000.0) + random.NextDouble() * 0.1;
        }

        // Act
        double result = stoi.Compute(signal, signal);

        // Assert
        // Identical signals should have high STOI (close to 1)
        Assert.True(result > 0.5);
    }

    [Fact]
    public void ScaleInvariantSDR_IdenticalSignals_ReturnsHighValue()
    {
        // Arrange
        var siSdr = new ScaleInvariantSignalToDistortionRatio<double>();
        var signal = new Tensor<double>(new[] { 1000 });
        for (int i = 0; i < signal.Length; i++)
        {
            signal[i] = Math.Sin(2 * Math.PI * 100 * i / 1000.0);
        }

        // Act
        double result = siSdr.Compute(signal, signal);

        // Assert
        // Identical signals should have very high SI-SDR
        Assert.True(result > 50);
    }

    [Fact]
    public void ScaleInvariantSDR_ScaledSignal_ReturnsHighValue()
    {
        // Arrange
        var siSdr = new ScaleInvariantSignalToDistortionRatio<double>();
        var target = new Tensor<double>(new[] { 1000 });
        var estimated = new Tensor<double>(new[] { 1000 });
        for (int i = 0; i < target.Length; i++)
        {
            target[i] = Math.Sin(2 * Math.PI * 100 * i / 1000.0);
            estimated[i] = target[i] * 2.0; // Scaled version
        }

        // Act
        double result = siSdr.Compute(estimated, target);

        // Assert
        // Scale-invariant: scaled version should have high SI-SDR
        Assert.True(result > 50);
    }

    [Fact]
    public void SignalToNoiseRatio_NoisySignal_ReturnsPositiveValue()
    {
        // Arrange
        var snr = new SignalToNoiseRatio<double>();
        var clean = new Tensor<double>(new[] { 1000 });
        var noisy = new Tensor<double>(new[] { 1000 });
        var random = new Random(42);

        for (int i = 0; i < clean.Length; i++)
        {
            clean[i] = Math.Sin(2 * Math.PI * 100 * i / 1000.0);
            noisy[i] = clean[i] + random.NextDouble() * 0.1 - 0.05; // Add small noise
        }

        // Act
        double result = snr.Compute(clean, noisy);

        // Assert
        // Clean signal with small noise should have positive SNR
        Assert.True(result > 0);
    }

    #endregion

    #region Image Metrics Tests

    [Fact]
    public void PSNR_IdenticalImages_ReturnsHighValue()
    {
        // Arrange
        var psnr = new PeakSignalToNoiseRatio<double>();
        var image = new Tensor<double>(new[] { 64, 64, 3 });
        var random = new Random(42);
        for (int i = 0; i < image.Length; i++)
        {
            image[i] = random.NextDouble();
        }

        // Act
        double result = psnr.Compute(image, image);

        // Assert
        // Identical images should have very high PSNR (capped at 100)
        Assert.Equal(100.0, result, 6);
    }

    [Fact]
    public void PSNR_DifferentImages_ReturnsLowerValue()
    {
        // Arrange
        var psnr = new PeakSignalToNoiseRatio<double>();
        var image1 = new Tensor<double>(new[] { 64, 64, 3 });
        var image2 = new Tensor<double>(new[] { 64, 64, 3 });
        var random = new Random(42);

        for (int i = 0; i < image1.Length; i++)
        {
            image1[i] = random.NextDouble();
            image2[i] = random.NextDouble(); // Different random values
        }

        // Act
        double result = psnr.Compute(image1, image2);

        // Assert
        // Different images should have finite PSNR
        Assert.True(result > 0 && result < 100);
    }

    [Fact]
    public void SSIM_IdenticalImages_ReturnsOne()
    {
        // Arrange
        var ssim = new StructuralSimilarity<double>();
        var image = new Tensor<double>(new[] { 64, 64, 3 });
        var random = new Random(42);
        for (int i = 0; i < image.Length; i++)
        {
            image[i] = random.NextDouble();
        }

        // Act
        double result = ssim.Compute(image, image);

        // Assert
        // SSIM of identical images should be 1
        Assert.Equal(1.0, result, 3);
    }

    [Fact]
    public void SSIM_DifferentImages_ReturnsLowerValue()
    {
        // Arrange
        var ssim = new StructuralSimilarity<double>();
        var image1 = new Tensor<double>(new[] { 64, 64, 3 });
        var image2 = new Tensor<double>(new[] { 64, 64, 3 });
        var random = new Random(42);

        for (int i = 0; i < image1.Length; i++)
        {
            image1[i] = random.NextDouble();
            image2[i] = random.NextDouble();
        }

        // Act
        double result = ssim.Compute(image1, image2);

        // Assert
        // Different images should have SSIM < 1
        Assert.True(result < 1.0 && result > -1.0);
    }

    [Fact]
    public void MeanIoU_PerfectPrediction_ReturnsOne()
    {
        // Arrange
        var miou = new MeanIntersectionOverUnion<double>(numClasses: 3);
        var prediction = new Tensor<double>(new[] { 100 });
        var groundTruth = new Tensor<double>(new[] { 100 });

        for (int i = 0; i < 100; i++)
        {
            int classLabel = i % 3;
            prediction[i] = classLabel;
            groundTruth[i] = classLabel;
        }

        // Act
        double result = miou.Compute(prediction, groundTruth);

        // Assert
        Assert.Equal(1.0, result, 6);
    }

    [Fact]
    public void MeanIoU_RandomPrediction_ReturnsLowerValue()
    {
        // Arrange
        var miou = new MeanIntersectionOverUnion<double>(numClasses: 3);
        var prediction = new Tensor<double>(new[] { 100 });
        var groundTruth = new Tensor<double>(new[] { 100 });
        var random = new Random(42);

        for (int i = 0; i < 100; i++)
        {
            prediction[i] = random.Next(3);
            groundTruth[i] = i % 3;
        }

        // Act
        double result = miou.Compute(prediction, groundTruth);

        // Assert
        // Random predictions should have low mIoU
        Assert.True(result < 0.5);
    }

    #endregion

    #region Video Metrics Tests

    [Fact]
    public void VideoPSNR_IdenticalVideos_ReturnsHighValue()
    {
        // Arrange
        var vpsnr = new VideoPSNR<double>();
        // Create a 10-frame video with 32x32 RGB
        var video = new Tensor<double>(new[] { 10, 32, 32, 3 });
        var random = new Random(42);
        for (int i = 0; i < video.Length; i++)
        {
            video[i] = random.NextDouble();
        }

        // Act
        var (mean, min, max, perFrame) = vpsnr.ComputeWithStats(video, video);

        // Assert
        Assert.Equal(100.0, mean, 6);
        Assert.Equal(10, perFrame.Length);
    }

    [Fact]
    public void VideoSSIM_IdenticalVideos_ReturnsOne()
    {
        // Arrange
        var vssim = new VideoSSIM<double>();
        var video = new Tensor<double>(new[] { 5, 32, 32, 3 });
        var random = new Random(42);
        for (int i = 0; i < video.Length; i++)
        {
            video[i] = random.NextDouble();
        }

        // Act
        double result = vssim.Compute(video, video);

        // Assert
        Assert.Equal(1.0, result, 3);
    }

    [Fact]
    public void TemporalConsistency_SmoothVideo_ReturnsHighValue()
    {
        // Arrange
        var tc = new TemporalConsistency<double>();
        // Create a smooth video where each frame is similar to the previous
        var video = new Tensor<double>(new[] { 10, 32, 32, 3 });

        for (int t = 0; t < 10; t++)
        {
            for (int h = 0; h < 32; h++)
            {
                for (int w = 0; w < 32; w++)
                {
                    for (int c = 0; c < 3; c++)
                    {
                        // Gradually changing pattern
                        video[t, h, w, c] = (double)t / 10.0 + (double)(h + w) / 64.0;
                    }
                }
            }
        }

        // Act
        double result = tc.ComputeSimple(video);

        // Assert
        // Smooth video should have high temporal consistency
        Assert.True(result > 0.8);
    }

    [Fact]
    public void TemporalConsistency_FlickeringVideo_ReturnsLowerValue()
    {
        // Arrange
        var tc = new TemporalConsistency<double>();
        // Create a flickering video where alternating frames are very different
        var video = new Tensor<double>(new[] { 10, 32, 32, 3 });

        for (int t = 0; t < 10; t++)
        {
            double baseValue = (t % 2 == 0) ? 0.0 : 1.0;
            for (int h = 0; h < 32; h++)
            {
                for (int w = 0; w < 32; w++)
                {
                    for (int c = 0; c < 3; c++)
                    {
                        video[t, h, w, c] = baseValue;
                    }
                }
            }
        }

        // Act
        double result = tc.ComputeSimple(video);

        // Assert
        // Flickering video should have low temporal consistency
        Assert.True(result < 0.5);
    }

    [Fact]
    public void Flicker_SmoothVideo_ReturnsLowValue()
    {
        // Arrange
        var tc = new TemporalConsistency<double>();
        var video = new Tensor<double>(new[] { 10, 32, 32, 3 });

        for (int t = 0; t < 10; t++)
        {
            for (int h = 0; h < 32; h++)
            {
                for (int w = 0; w < 32; w++)
                {
                    for (int c = 0; c < 3; c++)
                    {
                        video[t, h, w, c] = (double)t / 10.0;
                    }
                }
            }
        }

        // Act
        double result = tc.ComputeFlicker(video);

        // Assert
        // Smooth video should have low flicker
        Assert.True(result < 0.3);
    }

    [Fact]
    public void VideoQualityIndex_ReturnsComprehensiveResults()
    {
        // Arrange
        var vqi = new VideoQualityIndex<double>();
        var video1 = new Tensor<double>(new[] { 5, 32, 32, 3 });
        var video2 = new Tensor<double>(new[] { 5, 32, 32, 3 });
        var random = new Random(42);

        for (int i = 0; i < video1.Length; i++)
        {
            video1[i] = random.NextDouble();
            video2[i] = video1[i] + (random.NextDouble() * 0.1 - 0.05); // Small perturbation
        }

        // Act
        var result = vqi.Compute(video1, video2);

        // Assert
        Assert.True(result.MeanPSNR > 0);
        Assert.True(result.MeanSSIM > 0 && result.MeanSSIM <= 1);
        Assert.True(result.TemporalConsistency >= 0 && result.TemporalConsistency <= 1);
        Assert.True(result.FlickerScore >= 0 && result.FlickerScore <= 1);
        Assert.True(result.OverallScore >= 0 && result.OverallScore <= 1);
        Assert.Equal(5, result.PerFramePSNR.Length);
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void WordErrorRate_BatchMismatch_ThrowsException()
    {
        // Arrange
        var wer = new WordErrorRate();
        var references = new[] { "hello" };
        var hypotheses = new[] { "hello", "world" };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => wer.ComputeBatch(hypotheses, references));
    }

    [Fact]
    public void CharacterErrorRate_BatchMismatch_ThrowsException()
    {
        // Arrange
        var cer = new CharacterErrorRate();
        var references = new[] { "hello" };
        var hypotheses = new[] { "hello", "world" };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => cer.ComputeBatch(hypotheses, references));
    }

    [Fact]
    public void PSNR_ShapeMismatch_ThrowsException()
    {
        // Arrange
        var psnr = new PeakSignalToNoiseRatio<double>();
        var image1 = new Tensor<double>(new[] { 64, 64, 3 });
        var image2 = new Tensor<double>(new[] { 32, 32, 3 });

        // Act & Assert
        Assert.Throws<ArgumentException>(() => psnr.Compute(image1, image2));
    }

    [Fact]
    public void SiSdr_LengthMismatch_ThrowsException()
    {
        // Arrange
        var siSdr = new ScaleInvariantSignalToDistortionRatio<double>();
        var signal1 = new Tensor<double>(new[] { 100 });
        var signal2 = new Tensor<double>(new[] { 200 });

        // Act & Assert
        Assert.Throws<ArgumentException>(() => siSdr.Compute(signal1, signal2));
    }

    [Fact]
    public void STOI_LengthMismatch_ThrowsException()
    {
        // Arrange
        var stoi = new ShortTimeObjectiveIntelligibility<double>();
        var signal1 = new Tensor<double>(new[] { 100 });
        var signal2 = new Tensor<double>(new[] { 200 });

        // Act & Assert
        Assert.Throws<ArgumentException>(() => stoi.Compute(signal1, signal2));
    }

    #endregion
}
