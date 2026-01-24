using System;
using System.Collections.Generic;
using AiDotNet.Metrics;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Metrics;

/// <summary>
/// Integration tests for the Metrics module.
/// These tests verify image quality, audio, and geometry metrics implementations.
/// </summary>
public class MetricsIntegrationTests
{
    #region PeakSignalToNoiseRatio Tests

    [Fact]
    public void PSNR_IdenticalImages_ReturnsHighValue()
    {
        var psnr = new PeakSignalToNoiseRatio<double>();

        // Create identical 4x4 images
        var data = new double[4 * 4 * 3];
        for (int i = 0; i < data.Length; i++) data[i] = 0.5;

        var image1 = new Tensor<double>(new[] { 4, 4, 3 }, new Vector<double>(data));
        var image2 = new Tensor<double>(new[] { 4, 4, 3 }, new Vector<double>(data));

        var result = psnr.Compute(image1, image2);

        // PSNR should be very high (100 dB) for identical images
        Assert.True(result >= 99.0);
    }

    [Fact]
    public void PSNR_DifferentImages_ReturnsReasonableValue()
    {
        var psnr = new PeakSignalToNoiseRatio<double>();

        // Create two different images
        var data1 = new double[4 * 4 * 3];
        var data2 = new double[4 * 4 * 3];
        for (int i = 0; i < data1.Length; i++)
        {
            data1[i] = 0.5;
            data2[i] = 0.6; // Slight difference
        }

        var image1 = new Tensor<double>(new[] { 4, 4, 3 }, new Vector<double>(data1));
        var image2 = new Tensor<double>(new[] { 4, 4, 3 }, new Vector<double>(data2));

        var result = psnr.Compute(image1, image2);

        // PSNR should be positive for similar images
        Assert.True(result > 0 && result < 100);
    }

    [Fact]
    public void PSNR_WithCustomMaxValue_ComputesCorrectly()
    {
        var psnr = new PeakSignalToNoiseRatio<double>(255.0);

        var data1 = new double[4 * 4 * 3];
        var data2 = new double[4 * 4 * 3];
        for (int i = 0; i < data1.Length; i++)
        {
            data1[i] = 128.0;
            data2[i] = 128.0;
        }

        var image1 = new Tensor<double>(new[] { 4, 4, 3 }, new Vector<double>(data1));
        var image2 = new Tensor<double>(new[] { 4, 4, 3 }, new Vector<double>(data2));

        var result = psnr.Compute(image1, image2);
        Assert.True(result >= 99.0);
    }

    [Fact]
    public void PSNR_NullPredicted_ThrowsArgumentNullException()
    {
        var psnr = new PeakSignalToNoiseRatio<double>();
        var data = new double[16];
        var image = new Tensor<double>(new[] { 4, 4 }, new Vector<double>(data));

        Assert.Throws<ArgumentNullException>(() => psnr.Compute(null!, image));
    }

    [Fact]
    public void PSNR_NullGroundTruth_ThrowsArgumentNullException()
    {
        var psnr = new PeakSignalToNoiseRatio<double>();
        var data = new double[16];
        var image = new Tensor<double>(new[] { 4, 4 }, new Vector<double>(data));

        Assert.Throws<ArgumentNullException>(() => psnr.Compute(image, null!));
    }

    [Fact]
    public void PSNR_ShapeMismatch_ThrowsArgumentException()
    {
        var psnr = new PeakSignalToNoiseRatio<double>();

        var image1 = new Tensor<double>(new[] { 4, 4 }, new Vector<double>(new double[16]));
        var image2 = new Tensor<double>(new[] { 8, 8 }, new Vector<double>(new double[64]));

        Assert.Throws<ArgumentException>(() => psnr.Compute(image1, image2));
    }

    [Fact]
    public void PSNR_ComputeBatch_ReturnsCorrectArraySize()
    {
        var psnr = new PeakSignalToNoiseRatio<double>();

        // Create batch of 2 images, each 4x4 with 3 channels [B, H, W, C]
        var batch1 = new Tensor<double>(new[] { 2, 4, 4, 3 }, new Vector<double>(new double[2 * 4 * 4 * 3]));
        var batch2 = new Tensor<double>(new[] { 2, 4, 4, 3 }, new Vector<double>(new double[2 * 4 * 4 * 3]));

        var results = psnr.ComputeBatch(batch1, batch2);

        Assert.Equal(2, results.Length);
    }

    [Fact]
    public void PSNR_ComputeBatch_Non4DTensor_ThrowsArgumentException()
    {
        var psnr = new PeakSignalToNoiseRatio<double>();

        var tensor = new Tensor<double>(new[] { 4, 4 }, new Vector<double>(new double[16]));

        Assert.Throws<ArgumentException>(() => psnr.ComputeBatch(tensor, tensor));
    }

    #endregion

    #region StructuralSimilarity Tests

    [Fact]
    public void SSIM_IdenticalImages_ReturnsOne()
    {
        var ssim = new StructuralSimilarity<double>();

        var data = new double[8 * 8];
        for (int i = 0; i < data.Length; i++) data[i] = 0.5;

        var image1 = new Tensor<double>(new[] { 8, 8 }, new Vector<double>(data));
        var image2 = new Tensor<double>(new[] { 8, 8 }, new Vector<double>(data));

        var result = ssim.Compute(image1, image2);

        // SSIM should be approximately 1 for identical images
        Assert.True(result > 0.99);
    }

    [Fact]
    public void SSIM_DifferentImages_ReturnsLessThanOne()
    {
        var ssim = new StructuralSimilarity<double>();

        var data1 = new double[8 * 8];
        var data2 = new double[8 * 8];
        var random = new Random(42);

        for (int i = 0; i < data1.Length; i++)
        {
            data1[i] = random.NextDouble();
            data2[i] = random.NextDouble();
        }

        var image1 = new Tensor<double>(new[] { 8, 8 }, new Vector<double>(data1));
        var image2 = new Tensor<double>(new[] { 8, 8 }, new Vector<double>(data2));

        var result = ssim.Compute(image1, image2);

        // SSIM should be less than 1 for different images
        Assert.True(result < 1.0);
    }

    [Fact]
    public void SSIM_MultiChannelImage_ComputesCorrectly()
    {
        var ssim = new StructuralSimilarity<double>();

        // 8x8 image with 3 channels [H, W, C]
        var data = new double[8 * 8 * 3];
        for (int i = 0; i < data.Length; i++) data[i] = 0.5;

        var image1 = new Tensor<double>(new[] { 8, 8, 3 }, new Vector<double>(data));
        var image2 = new Tensor<double>(new[] { 8, 8, 3 }, new Vector<double>(data));

        var result = ssim.Compute(image1, image2);

        Assert.True(result > 0.99);
    }

    [Fact]
    public void SSIM_NullPredicted_ThrowsArgumentNullException()
    {
        var ssim = new StructuralSimilarity<double>();
        var image = new Tensor<double>(new[] { 8, 8 }, new Vector<double>(new double[64]));

        Assert.Throws<ArgumentNullException>(() => ssim.Compute(null!, image));
    }

    [Fact]
    public void SSIM_NullGroundTruth_ThrowsArgumentNullException()
    {
        var ssim = new StructuralSimilarity<double>();
        var image = new Tensor<double>(new[] { 8, 8 }, new Vector<double>(new double[64]));

        Assert.Throws<ArgumentNullException>(() => ssim.Compute(image, null!));
    }

    [Fact]
    public void SSIM_CustomParameters_WorksCorrectly()
    {
        var ssim = new StructuralSimilarity<double>(maxValue: 255.0, k1: 0.02, k2: 0.04, windowSize: 7);

        var data = new double[8 * 8];
        for (int i = 0; i < data.Length; i++) data[i] = 128.0;

        var image = new Tensor<double>(new[] { 8, 8 }, new Vector<double>(data));

        var result = ssim.Compute(image, image);

        Assert.True(result > 0.99);
    }

    #endregion

    #region MeanIntersectionOverUnion Tests

    [Fact]
    public void mIoU_PerfectPrediction_ReturnsOne()
    {
        var miou = new MeanIntersectionOverUnion<double>(numClasses: 3);

        var data = new double[] { 0, 1, 2, 0, 1, 2, 0, 1, 2 };
        var predicted = new Tensor<double>(new[] { 9 }, new Vector<double>(data));
        var groundTruth = new Tensor<double>(new[] { 9 }, new Vector<double>(data));

        var result = miou.Compute(predicted, groundTruth);

        Assert.Equal(1.0, result, 3);
    }

    [Fact]
    public void mIoU_CompleteMismatch_ReturnsZero()
    {
        var miou = new MeanIntersectionOverUnion<double>(numClasses: 2);

        var predicted = new Tensor<double>(new[] { 4 }, new Vector<double>(new double[] { 0, 0, 0, 0 }));
        var groundTruth = new Tensor<double>(new[] { 4 }, new Vector<double>(new double[] { 1, 1, 1, 1 }));

        var result = miou.Compute(predicted, groundTruth);

        Assert.Equal(0.0, result, 3);
    }

    [Fact]
    public void mIoU_PartialMatch_ReturnsBetweenZeroAndOne()
    {
        var miou = new MeanIntersectionOverUnion<double>(numClasses: 2);

        var predicted = new Tensor<double>(new[] { 4 }, new Vector<double>(new double[] { 0, 1, 0, 1 }));
        var groundTruth = new Tensor<double>(new[] { 4 }, new Vector<double>(new double[] { 0, 0, 1, 1 }));

        var result = miou.Compute(predicted, groundTruth);

        Assert.True(result > 0 && result < 1);
    }

    [Fact]
    public void mIoU_IgnoreBackground_ExcludesClass0()
    {
        var miou = new MeanIntersectionOverUnion<double>(numClasses: 2, ignoreBackground: true);

        var predicted = new Tensor<double>(new[] { 4 }, new Vector<double>(new double[] { 0, 1, 1, 1 }));
        var groundTruth = new Tensor<double>(new[] { 4 }, new Vector<double>(new double[] { 0, 1, 1, 1 }));

        var result = miou.Compute(predicted, groundTruth);

        // With background ignored, only class 1 is evaluated (perfect match)
        Assert.Equal(1.0, result, 3);
    }

    [Fact]
    public void mIoU_TooFewClasses_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() => new MeanIntersectionOverUnion<double>(numClasses: 1));
    }

    [Fact]
    public void mIoU_NullPredicted_ThrowsArgumentNullException()
    {
        var miou = new MeanIntersectionOverUnion<double>(numClasses: 2);
        var groundTruth = new Tensor<double>(new[] { 4 }, new Vector<double>(new double[4]));

        Assert.Throws<ArgumentNullException>(() => miou.Compute(null!, groundTruth));
    }

    [Fact]
    public void mIoU_ComputePerClass_ReturnsCorrectArraySize()
    {
        var miou = new MeanIntersectionOverUnion<double>(numClasses: 3);

        var data = new double[] { 0, 1, 2, 0, 1, 2 };
        var predicted = new Tensor<double>(new[] { 6 }, new Vector<double>(data));
        var groundTruth = new Tensor<double>(new[] { 6 }, new Vector<double>(data));

        var results = miou.ComputePerClass(predicted, groundTruth);

        Assert.Equal(3, results.Length);
        Assert.Equal(1.0, results[0], 3);
        Assert.Equal(1.0, results[1], 3);
        Assert.Equal(1.0, results[2], 3);
    }

    #endregion

    #region OverallAccuracy Tests

    [Fact]
    public void OverallAccuracy_PerfectPrediction_ReturnsOne()
    {
        var accuracy = new OverallAccuracy<double>();

        var data = new double[] { 0, 1, 2, 0, 1, 2 };
        var predicted = new Tensor<double>(new[] { 6 }, new Vector<double>(data));
        var groundTruth = new Tensor<double>(new[] { 6 }, new Vector<double>(data));

        var result = accuracy.Compute(predicted, groundTruth);

        Assert.Equal(1.0, result, 3);
    }

    [Fact]
    public void OverallAccuracy_HalfCorrect_ReturnsPointFive()
    {
        var accuracy = new OverallAccuracy<double>();

        var predicted = new Tensor<double>(new[] { 4 }, new Vector<double>(new double[] { 0, 1, 0, 1 }));
        var groundTruth = new Tensor<double>(new[] { 4 }, new Vector<double>(new double[] { 0, 1, 1, 0 }));

        var result = accuracy.Compute(predicted, groundTruth);

        Assert.Equal(0.5, result, 3);
    }

    [Fact]
    public void OverallAccuracy_NullPredicted_ThrowsArgumentNullException()
    {
        var accuracy = new OverallAccuracy<double>();
        var groundTruth = new Tensor<double>(new[] { 4 }, new Vector<double>(new double[4]));

        Assert.Throws<ArgumentNullException>(() => accuracy.Compute(null!, groundTruth));
    }

    [Fact]
    public void OverallAccuracy_LengthMismatch_ThrowsArgumentException()
    {
        var accuracy = new OverallAccuracy<double>();

        var predicted = new Tensor<double>(new[] { 4 }, new Vector<double>(new double[4]));
        var groundTruth = new Tensor<double>(new[] { 6 }, new Vector<double>(new double[6]));

        Assert.Throws<ArgumentException>(() => accuracy.Compute(predicted, groundTruth));
    }

    #endregion

    #region WordErrorRate Tests

    [Fact]
    public void WER_IdenticalText_ReturnsZero()
    {
        var wer = new WordErrorRate();

        var result = wer.Compute("hello world", "hello world");

        Assert.Equal(0.0, result, 3);
    }

    [Fact]
    public void WER_CompletelyDifferent_ReturnsOne()
    {
        var wer = new WordErrorRate();

        var result = wer.Compute("foo bar", "hello world");

        Assert.Equal(1.0, result, 3);
    }

    [Fact]
    public void WER_OneDeletion_ComputesCorrectly()
    {
        var wer = new WordErrorRate();

        // Reference has 2 words, hypothesis has 1 deletion
        var result = wer.Compute("hello", "hello world");

        // 1 deletion / 2 reference words = 0.5
        Assert.Equal(0.5, result, 3);
    }

    [Fact]
    public void WER_OneInsertion_ComputesCorrectly()
    {
        var wer = new WordErrorRate();

        var result = wer.Compute("hello world extra", "hello world");

        // 1 insertion / 2 reference words = 0.5
        Assert.Equal(0.5, result, 3);
    }

    [Fact]
    public void WER_OneSubstitution_ComputesCorrectly()
    {
        var wer = new WordErrorRate();

        var result = wer.Compute("hello universe", "hello world");

        // 1 substitution / 2 reference words = 0.5
        Assert.Equal(0.5, result, 3);
    }

    [Fact]
    public void WER_EmptyReference_ReturnsCorrectValue()
    {
        var wer = new WordErrorRate();

        Assert.Equal(1.0, wer.Compute("hello", ""));
        Assert.Equal(0.0, wer.Compute("", ""));
    }

    [Fact]
    public void WER_ComputeBatch_ReturnsAverage()
    {
        var wer = new WordErrorRate();

        var hypotheses = new[] { "hello world", "foo bar" };
        var references = new[] { "hello world", "hello world" };

        var result = wer.ComputeBatch(hypotheses, references);

        // First pair: 0.0, Second pair: 1.0 (2 substitutions / 2 words)
        // Average: 0.5
        Assert.Equal(0.5, result, 3);
    }

    [Fact]
    public void WER_ComputeBatch_MismatchedArrays_ThrowsArgumentException()
    {
        var wer = new WordErrorRate();

        Assert.Throws<ArgumentException>(() =>
            wer.ComputeBatch(new[] { "a", "b" }, new[] { "a" }));
    }

    [Fact]
    public void WER_ComputeDetailed_ReturnsAllComponents()
    {
        var wer = new WordErrorRate();

        var (werValue, subs, ins, dels, refCount) = wer.ComputeDetailed("hello universe extra", "hello world");

        Assert.True(werValue > 0);
        Assert.Equal(2, refCount);
        // 1 substitution (world->universe) + 1 insertion (extra)
        Assert.Equal(1, subs);
        Assert.Equal(1, ins);
        Assert.Equal(0, dels);
    }

    [Fact]
    public void WER_CaseInsensitive_MatchesCorrectly()
    {
        var wer = new WordErrorRate();

        var result = wer.Compute("HELLO WORLD", "hello world");

        Assert.Equal(0.0, result, 3);
    }

    #endregion

    #region CharacterErrorRate Tests

    [Fact]
    public void CER_IdenticalText_ReturnsZero()
    {
        var cer = new CharacterErrorRate();

        var result = cer.Compute("hello", "hello");

        Assert.Equal(0.0, result, 3);
    }

    [Fact]
    public void CER_OneCharacterDifferent_ComputesCorrectly()
    {
        var cer = new CharacterErrorRate();

        var result = cer.Compute("hallo", "hello");

        // 1 substitution / 5 characters = 0.2
        Assert.Equal(0.2, result, 3);
    }

    [Fact]
    public void CER_EmptyReference_ReturnsCorrectValue()
    {
        var cer = new CharacterErrorRate();

        Assert.Equal(1.0, cer.Compute("hello", ""));
        Assert.Equal(0.0, cer.Compute("", ""));
    }

    [Fact]
    public void CER_IgnoreWhitespace_RemovesSpaces()
    {
        var cer = new CharacterErrorRate();

        var result = cer.Compute("hello world", "helloworld", ignoreWhitespace: true);

        Assert.Equal(0.0, result, 3);
    }

    [Fact]
    public void CER_ComputeBatch_ReturnsAverage()
    {
        var cer = new CharacterErrorRate();

        var hypotheses = new[] { "hello", "hallo" };
        var references = new[] { "hello", "hello" };

        var result = cer.ComputeBatch(hypotheses, references);

        // First: 0.0, Second: 0.2 (1/5), Average: 0.1
        Assert.Equal(0.1, result, 3);
    }

    #endregion

    #region SignalToNoiseRatio Tests

    [Fact]
    public void SNR_IdenticalSignals_ReturnsHighValue()
    {
        var snr = new SignalToNoiseRatio<double>();

        var data = new double[100];
        for (int i = 0; i < data.Length; i++) data[i] = Math.Sin(i * 0.1);

        var clean = new Tensor<double>(new[] { 100 }, new Vector<double>(data));
        var noisy = new Tensor<double>(new[] { 100 }, new Vector<double>(data));

        var result = snr.Compute(clean, noisy);

        Assert.True(result >= 99.0); // Very high SNR for identical signals
    }

    [Fact]
    public void SNR_NoisySignal_ReturnsPositiveValue()
    {
        var snr = new SignalToNoiseRatio<double>();

        var random = new Random(42);
        var clean = new double[100];
        var noisy = new double[100];

        for (int i = 0; i < clean.Length; i++)
        {
            clean[i] = Math.Sin(i * 0.1);
            noisy[i] = clean[i] + random.NextDouble() * 0.1; // Add small noise
        }

        var cleanTensor = new Tensor<double>(new[] { 100 }, new Vector<double>(clean));
        var noisyTensor = new Tensor<double>(new[] { 100 }, new Vector<double>(noisy));

        var result = snr.Compute(cleanTensor, noisyTensor);

        Assert.True(result > 0); // Should be positive
    }

    [Fact]
    public void SNR_LengthMismatch_ThrowsArgumentException()
    {
        var snr = new SignalToNoiseRatio<double>();

        var clean = new Tensor<double>(new[] { 100 }, new Vector<double>(new double[100]));
        var noisy = new Tensor<double>(new[] { 50 }, new Vector<double>(new double[50]));

        Assert.Throws<ArgumentException>(() => snr.Compute(clean, noisy));
    }

    [Fact]
    public void SNR_ComputeSegmental_ReturnsReasonableValue()
    {
        var snr = new SignalToNoiseRatio<double>();

        var random = new Random(42);
        var clean = new double[512];
        var noisy = new double[512];

        for (int i = 0; i < clean.Length; i++)
        {
            clean[i] = Math.Sin(i * 0.1);
            noisy[i] = clean[i] + random.NextDouble() * 0.1;
        }

        var cleanTensor = new Tensor<double>(new[] { 512 }, new Vector<double>(clean));
        var noisyTensor = new Tensor<double>(new[] { 512 }, new Vector<double>(noisy));

        var result = snr.ComputeSegmental(cleanTensor, noisyTensor, frameLength: 128);

        Assert.True(result > -10 && result < 35); // Within clamped range
    }

    #endregion

    #region ScaleInvariantSignalToDistortionRatio Tests

    [Fact]
    public void SISDR_IdenticalSignals_ReturnsHighValue()
    {
        var sisdr = new ScaleInvariantSignalToDistortionRatio<double>();

        var data = new double[100];
        for (int i = 0; i < data.Length; i++) data[i] = Math.Sin(i * 0.1);

        var estimated = new Tensor<double>(new[] { 100 }, new Vector<double>(data));
        var target = new Tensor<double>(new[] { 100 }, new Vector<double>(data));

        var result = sisdr.Compute(estimated, target);

        Assert.True(result >= 99.0); // Very high SI-SDR for identical signals
    }

    [Fact]
    public void SISDR_ScaledSignal_HandlesScaleInvariance()
    {
        var sisdr = new ScaleInvariantSignalToDistortionRatio<double>();

        var target = new double[100];
        var estimated = new double[100];

        for (int i = 0; i < target.Length; i++)
        {
            target[i] = Math.Sin(i * 0.1);
            estimated[i] = target[i] * 2.0; // Scaled version
        }

        var estimatedTensor = new Tensor<double>(new[] { 100 }, new Vector<double>(estimated));
        var targetTensor = new Tensor<double>(new[] { 100 }, new Vector<double>(target));

        var result = sisdr.Compute(estimatedTensor, targetTensor);

        // SI-SDR should be high since signal is just scaled
        Assert.True(result > 90.0);
    }

    [Fact]
    public void SISDR_LengthMismatch_ThrowsArgumentException()
    {
        var sisdr = new ScaleInvariantSignalToDistortionRatio<double>();

        var estimated = new Tensor<double>(new[] { 100 }, new Vector<double>(new double[100]));
        var target = new Tensor<double>(new[] { 50 }, new Vector<double>(new double[50]));

        Assert.Throws<ArgumentException>(() => sisdr.Compute(estimated, target));
    }

    [Fact]
    public void SISDR_ComputeImprovement_ReturnsCorrectDifference()
    {
        var sisdr = new ScaleInvariantSignalToDistortionRatio<double>();

        var target = new double[100];
        var estimated = new double[100];
        var baseline = new double[100];

        var random = new Random(42);
        for (int i = 0; i < target.Length; i++)
        {
            target[i] = Math.Sin(i * 0.1);
            estimated[i] = target[i] + random.NextDouble() * 0.1; // Good estimate
            baseline[i] = target[i] + random.NextDouble() * 0.5; // Worse baseline
        }

        var estimatedTensor = new Tensor<double>(new[] { 100 }, new Vector<double>(estimated));
        var targetTensor = new Tensor<double>(new[] { 100 }, new Vector<double>(target));
        var baselineTensor = new Tensor<double>(new[] { 100 }, new Vector<double>(baseline));

        var improvement = sisdr.ComputeImprovement(estimatedTensor, targetTensor, baselineTensor);

        // Improvement should be positive since estimated is better than baseline
        Assert.True(improvement > 0);
    }

    #endregion

    #region ShortTimeObjectiveIntelligibility Tests

    [Fact]
    public void STOI_IdenticalSignals_ReturnsHighValue()
    {
        var stoi = new ShortTimeObjectiveIntelligibility<double>(sampleRate: 16000);

        var data = new double[1600]; // 100ms at 16kHz
        for (int i = 0; i < data.Length; i++) data[i] = Math.Sin(i * 0.1);

        var degraded = new Tensor<double>(new[] { 1600 }, new Vector<double>(data));
        var clean = new Tensor<double>(new[] { 1600 }, new Vector<double>(data));

        var result = stoi.Compute(degraded, clean);

        Assert.True(result >= 0.9); // High STOI for identical signals
    }

    [Fact]
    public void STOI_LengthMismatch_ThrowsArgumentException()
    {
        var stoi = new ShortTimeObjectiveIntelligibility<double>();

        var degraded = new Tensor<double>(new[] { 1000 }, new Vector<double>(new double[1000]));
        var clean = new Tensor<double>(new[] { 500 }, new Vector<double>(new double[500]));

        Assert.Throws<ArgumentException>(() => stoi.Compute(degraded, clean));
    }

    [Fact]
    public void STOI_InvalidSampleRate_ThrowsArgumentOutOfRangeException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ShortTimeObjectiveIntelligibility<double>(sampleRate: 0));
    }

    [Fact]
    public void STOI_InvalidFrameLength_ThrowsArgumentOutOfRangeException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ShortTimeObjectiveIntelligibility<double>(frameLength: 0));
    }

    [Fact]
    public void STOI_InvalidHopLength_ThrowsArgumentOutOfRangeException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ShortTimeObjectiveIntelligibility<double>(hopLength: 0));

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ShortTimeObjectiveIntelligibility<double>(frameLength: 256, hopLength: 512));
    }

    [Fact]
    public void STOI_InvalidNumBands_ThrowsArgumentOutOfRangeException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ShortTimeObjectiveIntelligibility<double>(numBands: 0));
    }

    #endregion

    #region PerceptualSpeechQuality Tests

    [Fact]
    public void PESQ_IdenticalSignals_ReturnsHighScore()
    {
        var pesq = new PerceptualSpeechQuality<double>(sampleRate: 16000);

        var data = new double[1600];
        for (int i = 0; i < data.Length; i++) data[i] = Math.Sin(i * 0.1);

        var degraded = new Tensor<double>(new[] { 1600 }, new Vector<double>(data));
        var reference = new Tensor<double>(new[] { 1600 }, new Vector<double>(data));

        var result = pesq.Compute(degraded, reference);

        Assert.True(result >= 3.0); // Should be good quality for identical signals
    }

    [Fact]
    public void PESQ_InvalidSampleRate_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            new PerceptualSpeechQuality<double>(sampleRate: 44100));
    }

    [Fact]
    public void PESQ_ValidSampleRates_DoNotThrow()
    {
        var pesq8k = new PerceptualSpeechQuality<double>(sampleRate: 8000);
        var pesq16k = new PerceptualSpeechQuality<double>(sampleRate: 16000);

        Assert.NotNull(pesq8k);
        Assert.NotNull(pesq16k);
    }

    #endregion

    #region ChamferDistance Tests

    [Fact]
    public void ChamferDistance_IdenticalPointClouds_ReturnsZero()
    {
        var chamfer = new ChamferDistance<double>();

        var data = new double[] { 0, 0, 0, 1, 0, 0, 0, 1, 0 }; // 3 points
        var pointsA = new Tensor<double>(new[] { 3, 3 }, new Vector<double>(data));
        var pointsB = new Tensor<double>(new[] { 3, 3 }, new Vector<double>(data));

        var result = chamfer.Compute(pointsA, pointsB);

        Assert.Equal(0.0, result, 5);
    }

    [Fact]
    public void ChamferDistance_DifferentPointClouds_ReturnsPositive()
    {
        var chamfer = new ChamferDistance<double>();

        var dataA = new double[] { 0, 0, 0, 1, 0, 0, 0, 1, 0 };
        var dataB = new double[] { 0.1, 0.1, 0.1, 1.1, 0.1, 0.1, 0.1, 1.1, 0.1 };

        var pointsA = new Tensor<double>(new[] { 3, 3 }, new Vector<double>(dataA));
        var pointsB = new Tensor<double>(new[] { 3, 3 }, new Vector<double>(dataB));

        var result = chamfer.Compute(pointsA, pointsB);

        Assert.True(result > 0);
    }

    [Fact]
    public void ChamferDistance_Squared_ReturnsSquaredDistances()
    {
        var chamferSquared = new ChamferDistance<double>(squared: true);
        var chamferEuclidean = new ChamferDistance<double>(squared: false);

        var dataA = new double[] { 0, 0, 0 };
        var dataB = new double[] { 1, 0, 0 };

        var pointsA = new Tensor<double>(new[] { 1, 3 }, new Vector<double>(dataA));
        var pointsB = new Tensor<double>(new[] { 1, 3 }, new Vector<double>(dataB));

        var squaredResult = chamferSquared.Compute(pointsA, pointsB);
        var euclideanResult = chamferEuclidean.Compute(pointsA, pointsB);

        // Squared distance = 1^2 = 1, Euclidean = 1
        // Both directions: squaredResult = 2, euclideanResult = 2
        Assert.Equal(2.0, squaredResult, 5);
        Assert.Equal(2.0, euclideanResult, 5);
    }

    [Fact]
    public void ChamferDistance_NullPointsA_ThrowsArgumentNullException()
    {
        var chamfer = new ChamferDistance<double>();
        var points = new Tensor<double>(new[] { 3, 3 }, new Vector<double>(new double[9]));

        Assert.Throws<ArgumentNullException>(() => chamfer.Compute(null!, points));
    }

    [Fact]
    public void ChamferDistance_Non2DTensor_ThrowsArgumentException()
    {
        var chamfer = new ChamferDistance<double>();
        var tensor = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[3]));

        Assert.Throws<ArgumentException>(() => chamfer.Compute(tensor, tensor));
    }

    [Fact]
    public void ChamferDistance_DimensionMismatch_ThrowsArgumentException()
    {
        var chamfer = new ChamferDistance<double>();

        var points3D = new Tensor<double>(new[] { 3, 3 }, new Vector<double>(new double[9]));
        var points2D = new Tensor<double>(new[] { 3, 2 }, new Vector<double>(new double[6]));

        Assert.Throws<ArgumentException>(() => chamfer.Compute(points3D, points2D));
    }

    [Fact]
    public void ChamferDistance_ComputeBatch_ReturnsCorrectArraySize()
    {
        var chamfer = new ChamferDistance<double>();

        var batchA = new Tensor<double>(new[] { 2, 3, 3 }, new Vector<double>(new double[18]));
        var batchB = new Tensor<double>(new[] { 2, 3, 3 }, new Vector<double>(new double[18]));

        var results = chamfer.ComputeBatch(batchA, batchB);

        Assert.Equal(2, results.Length);
    }

    [Fact]
    public void ChamferDistance_ComputeOneWay_ReturnsCorrectDirection()
    {
        var chamfer = new ChamferDistance<double>();

        // Source: 1 point at origin
        // Target: 2 points at (1,0,0) and (2,0,0)
        var source = new Tensor<double>(new[] { 1, 3 }, new Vector<double>(new double[] { 0, 0, 0 }));
        var target = new Tensor<double>(new[] { 2, 3 }, new Vector<double>(new double[] { 1, 0, 0, 2, 0, 0 }));

        var oneWay = chamfer.ComputeOneWay(source, target);

        // Nearest point to origin is (1,0,0), squared distance = 1
        Assert.Equal(1.0, oneWay, 5);
    }

    #endregion

    #region EarthMoversDistance Tests

    [Fact]
    public void EMD_IdenticalPointClouds_ReturnsZero()
    {
        var emd = new EarthMoversDistance<double>();

        var data = new double[] { 0, 0, 0, 1, 0, 0, 0, 1, 0 };
        var pointsA = new Tensor<double>(new[] { 3, 3 }, new Vector<double>(data));
        var pointsB = new Tensor<double>(new[] { 3, 3 }, new Vector<double>(data));

        var result = emd.Compute(pointsA, pointsB);

        Assert.True(result < 0.01); // Near zero for identical point clouds
    }

    [Fact]
    public void EMD_DifferentPointClouds_ReturnsPositive()
    {
        var emd = new EarthMoversDistance<double>();

        var dataA = new double[] { 0, 0, 0 };
        var dataB = new double[] { 1, 0, 0 };

        var pointsA = new Tensor<double>(new[] { 1, 3 }, new Vector<double>(dataA));
        var pointsB = new Tensor<double>(new[] { 1, 3 }, new Vector<double>(dataB));

        var result = emd.Compute(pointsA, pointsB);

        Assert.True(result > 0);
    }

    [Fact]
    public void EMD_NullPointClouds_ThrowsArgumentNullException()
    {
        var emd = new EarthMoversDistance<double>();
        var points = new Tensor<double>(new[] { 3, 3 }, new Vector<double>(new double[9]));

        Assert.Throws<ArgumentNullException>(() => emd.Compute(null!, points));
        Assert.Throws<ArgumentNullException>(() => emd.Compute(points, null!));
    }

    [Fact]
    public void EMD_Non2DTensor_ThrowsArgumentException()
    {
        var emd = new EarthMoversDistance<double>();
        var tensor = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[3]));

        Assert.Throws<ArgumentException>(() => emd.Compute(tensor, tensor));
    }

    [Fact]
    public void EMD_CustomIterationsAndEpsilon_WorksCorrectly()
    {
        var emd = new EarthMoversDistance<double>(iterations: 50, epsilon: 0.05);

        var data = new double[] { 0, 0, 0, 1, 0, 0 };
        var points = new Tensor<double>(new[] { 2, 3 }, new Vector<double>(data));

        var result = emd.Compute(points, points);

        Assert.True(result >= 0);
    }

    #endregion

    #region FScore Tests

    [Fact]
    public void FScore_IdenticalPointClouds_ReturnsOne()
    {
        var fscore = new FScore<double>(threshold: 0.1);

        var data = new double[] { 0, 0, 0, 1, 0, 0, 0, 1, 0 };
        var predicted = new Tensor<double>(new[] { 3, 3 }, new Vector<double>(data));
        var groundTruth = new Tensor<double>(new[] { 3, 3 }, new Vector<double>(data));

        var result = fscore.Compute(predicted, groundTruth);

        Assert.Equal(1.0, result, 3);
    }

    [Fact]
    public void FScore_FarPointClouds_ReturnsZero()
    {
        var fscore = new FScore<double>(threshold: 0.1);

        var dataA = new double[] { 0, 0, 0 };
        var dataB = new double[] { 10, 10, 10 };

        var predicted = new Tensor<double>(new[] { 1, 3 }, new Vector<double>(dataA));
        var groundTruth = new Tensor<double>(new[] { 1, 3 }, new Vector<double>(dataB));

        var result = fscore.Compute(predicted, groundTruth);

        Assert.Equal(0.0, result, 3);
    }

    [Fact]
    public void FScore_PartialMatch_ReturnsBetweenZeroAndOne()
    {
        var fscore = new FScore<double>(threshold: 0.5);

        var predicted = new double[] { 0, 0, 0, 0.3, 0, 0, 10, 0, 0 }; // 2 close, 1 far
        var groundTruth = new double[] { 0, 0, 0, 0.3, 0, 0, 0.6, 0, 0 }; // 3 points

        var predTensor = new Tensor<double>(new[] { 3, 3 }, new Vector<double>(predicted));
        var gtTensor = new Tensor<double>(new[] { 3, 3 }, new Vector<double>(groundTruth));

        var result = fscore.Compute(predTensor, gtTensor);

        Assert.True(result > 0 && result < 1);
    }

    [Fact]
    public void FScore_ComputePrecisionRecall_ReturnsTuple()
    {
        var fscore = new FScore<double>(threshold: 0.1);

        var data = new double[] { 0, 0, 0, 1, 0, 0 };
        var points = new Tensor<double>(new[] { 2, 3 }, new Vector<double>(data));

        var (precision, recall) = fscore.ComputePrecisionRecall(points, points);

        Assert.Equal(1.0, precision, 3);
        Assert.Equal(1.0, recall, 3);
    }

    [Fact]
    public void FScore_EmptyPointClouds_HandleCorrectly()
    {
        var fscore = new FScore<double>(threshold: 0.1);

        var emptyPoints = new Tensor<double>(new[] { 0, 3 }, new Vector<double>(new double[0]));
        var nonEmptyPoints = new Tensor<double>(new[] { 1, 3 }, new Vector<double>(new double[3]));

        // Both empty -> perfect match
        var (p1, r1) = fscore.ComputePrecisionRecall(emptyPoints, emptyPoints);
        Assert.Equal(1.0, p1, 3);
        Assert.Equal(1.0, r1, 3);

        // Predicted empty, GT not empty -> (0, 0)
        var (p2, r2) = fscore.ComputePrecisionRecall(emptyPoints, nonEmptyPoints);
        Assert.Equal(0.0, p2, 3);
        Assert.Equal(0.0, r2, 3);
    }

    [Fact]
    public void FScore_NullPredicted_ThrowsArgumentNullException()
    {
        var fscore = new FScore<double>();
        var groundTruth = new Tensor<double>(new[] { 3, 3 }, new Vector<double>(new double[9]));

        Assert.Throws<ArgumentNullException>(() => fscore.Compute(null!, groundTruth));
    }

    #endregion

    #region IoU3D Tests

    [Fact]
    public void IoU3D_IdenticalVoxels_ReturnsOne()
    {
        var iou = new IoU3D<double>();

        // Create 2x2x2 voxel grid with all occupied
        var data = new double[] { 1, 1, 1, 1, 1, 1, 1, 1 };
        var voxelsA = new Tensor<double>(new[] { 2, 2, 2 }, new Vector<double>(data));
        var voxelsB = new Tensor<double>(new[] { 2, 2, 2 }, new Vector<double>(data));

        var result = iou.ComputeVoxelIoU(voxelsA, voxelsB);

        Assert.Equal(1.0, result, 3);
    }

    [Fact]
    public void IoU3D_NoOverlap_ReturnsZero()
    {
        var iou = new IoU3D<double>();

        var dataA = new double[] { 1, 1, 0, 0 };
        var dataB = new double[] { 0, 0, 1, 1 };

        var voxelsA = new Tensor<double>(new[] { 2, 2, 1 }, new Vector<double>(dataA));
        var voxelsB = new Tensor<double>(new[] { 2, 2, 1 }, new Vector<double>(dataB));

        var result = iou.ComputeVoxelIoU(voxelsA, voxelsB);

        Assert.Equal(0.0, result, 3);
    }

    [Fact]
    public void IoU3D_PartialOverlap_ReturnsBetweenZeroAndOne()
    {
        var iou = new IoU3D<double>();

        var dataA = new double[] { 1, 1, 1, 0 };
        var dataB = new double[] { 0, 1, 1, 1 };

        var voxelsA = new Tensor<double>(new[] { 2, 2, 1 }, new Vector<double>(dataA));
        var voxelsB = new Tensor<double>(new[] { 2, 2, 1 }, new Vector<double>(dataB));

        var result = iou.ComputeVoxelIoU(voxelsA, voxelsB);

        // Intersection = 2, Union = 4, IoU = 0.5
        Assert.Equal(0.5, result, 3);
    }

    [Fact]
    public void IoU3D_VoxelSizeMismatch_ThrowsArgumentException()
    {
        var iou = new IoU3D<double>();

        var voxelsA = new Tensor<double>(new[] { 2, 2, 2 }, new Vector<double>(new double[8]));
        var voxelsB = new Tensor<double>(new[] { 3, 3, 3 }, new Vector<double>(new double[27]));

        Assert.Throws<ArgumentException>(() => iou.ComputeVoxelIoU(voxelsA, voxelsB));
    }

    [Fact]
    public void IoU3D_NullVoxels_ThrowsArgumentNullException()
    {
        var iou = new IoU3D<double>();
        var voxels = new Tensor<double>(new[] { 2, 2, 2 }, new Vector<double>(new double[8]));

        Assert.Throws<ArgumentNullException>(() => iou.ComputeVoxelIoU(null!, voxels));
    }

    [Fact]
    public void IoU3D_BoxIoU_IdenticalBoxes_ReturnsOne()
    {
        var iou = new IoU3D<double>();

        var box = new double[] { 0, 0, 0, 1, 1, 1 };

        var result = iou.ComputeBoxIoU(box, box);

        Assert.Equal(1.0, result, 3);
    }

    [Fact]
    public void IoU3D_BoxIoU_NoOverlap_ReturnsZero()
    {
        var iou = new IoU3D<double>();

        var boxA = new double[] { 0, 0, 0, 1, 1, 1 };
        var boxB = new double[] { 2, 2, 2, 3, 3, 3 };

        var result = iou.ComputeBoxIoU(boxA, boxB);

        Assert.Equal(0.0, result, 3);
    }

    [Fact]
    public void IoU3D_BoxIoU_PartialOverlap_ReturnsBetweenZeroAndOne()
    {
        var iou = new IoU3D<double>();

        var boxA = new double[] { 0, 0, 0, 2, 2, 2 }; // Volume = 8
        var boxB = new double[] { 1, 1, 1, 3, 3, 3 }; // Volume = 8

        var result = iou.ComputeBoxIoU(boxA, boxB);

        // Intersection: [1,1,1] to [2,2,2] = 1x1x1 = 1
        // Union = 8 + 8 - 1 = 15
        // IoU = 1/15
        Assert.True(result > 0 && result < 1);
    }

    [Fact]
    public void IoU3D_BoxIoU_InvalidBoxFormat_ThrowsArgumentException()
    {
        var iou = new IoU3D<double>();

        var validBox = new double[] { 0, 0, 0, 1, 1, 1 };
        var invalidBox = new double[] { 0, 0, 0, 1, 1 }; // Only 5 values

        Assert.Throws<ArgumentException>(() => iou.ComputeBoxIoU(validBox, invalidBox));
    }

    #endregion
}
