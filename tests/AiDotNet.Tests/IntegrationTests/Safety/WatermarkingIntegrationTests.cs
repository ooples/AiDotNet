#nullable disable
using AiDotNet.Safety.Watermarking;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.IntegrationTests.Safety;

/// <summary>
/// Integration tests for watermarking modules.
/// Tests TextWatermarker variants, ImageWatermarker variants, AudioWatermarker variants,
/// and the composite WatermarkDetector.
/// </summary>
public class WatermarkingIntegrationTests
{
    #region TextWatermarker Tests

    [Fact]
    public void TextWatermarker_DetectsWatermark()
    {
        var watermarker = new TextWatermarker<double>();
        var findings = watermarker.EvaluateText(
            "This is a sample text to test watermark detection capabilities in generated content");

        Assert.NotNull(findings);
    }

    [Fact]
    public void TextWatermarker_ShortText_HandlesGracefully()
    {
        var watermarker = new TextWatermarker<double>();
        var findings = watermarker.EvaluateText("Short text");

        Assert.NotNull(findings);
    }

    #endregion

    #region SamplingWatermarker Tests

    [Fact]
    public void Sampling_DetectsWatermarkPattern()
    {
        var watermarker = new SamplingWatermarker<double>();
        var score = watermarker.DetectWatermark(
            "This is a longer piece of text that contains enough content for watermark detection analysis");

        Assert.True(score >= 0 && score <= 1);
    }

    [Fact]
    public void Sampling_CustomVocabSize_Works()
    {
        var watermarker = new SamplingWatermarker<double>(
            watermarkStrength: 0.7, vocabSize: 10000, greenListFraction: 0.3);
        var score = watermarker.DetectWatermark(
            "Another sample text for watermark analysis with different configuration settings");

        Assert.True(score >= 0 && score <= 1);
    }

    #endregion

    #region LexicalWatermarker Tests

    [Fact]
    public void Lexical_DetectsWatermarkPattern()
    {
        var watermarker = new LexicalWatermarker<double>();
        var score = watermarker.DetectWatermark(
            "The quick brown fox jumps over the lazy dog in the park on a sunny day");

        Assert.True(score >= 0 && score <= 1);
    }

    #endregion

    #region SyntacticWatermarker Tests

    [Fact]
    public void Syntactic_DetectsWatermarkPattern()
    {
        var watermarker = new SyntacticWatermarker<double>();
        var score = watermarker.DetectWatermark(
            "Complex sentences with multiple clauses are often used in formal writing and academic papers");

        Assert.True(score >= 0 && score <= 1);
    }

    #endregion

    #region ImageWatermarker Tests

    [Fact]
    public void ImageWatermarker_ProcessesTensor()
    {
        var watermarker = new ImageWatermarker<double>();
        var tensor = CreateRandomImageTensor(3, 16, 16);
        var findings = watermarker.EvaluateImage(tensor);

        Assert.NotNull(findings);
    }

    #endregion

    #region FrequencyImageWatermarker Tests

    [Fact]
    public void FrequencyImage_DetectsWatermark()
    {
        var watermarker = new FrequencyImageWatermarker<double>();
        var tensor = CreateRandomImageTensor(3, 32, 32);
        var score = watermarker.DetectWatermark(tensor);

        Assert.True(score >= 0 && score <= 1);
    }

    #endregion

    #region NeuralImageWatermarker Tests

    [Fact]
    public void NeuralImage_DetectsWatermark()
    {
        var watermarker = new NeuralImageWatermarker<double>();
        var tensor = CreateRandomImageTensor(3, 32, 32);
        var score = watermarker.DetectWatermark(tensor);

        Assert.True(score >= 0 && score <= 1);
    }

    #endregion

    #region InvisibleImageWatermarker Tests

    [Fact]
    public void InvisibleImage_DetectsWatermark()
    {
        var watermarker = new InvisibleImageWatermarker<double>();
        var tensor = CreateRandomImageTensor(3, 32, 32);
        var score = watermarker.DetectWatermark(tensor);

        Assert.True(score >= 0 && score <= 1);
    }

    [Fact]
    public void InvisibleImage_CustomStrength_Works()
    {
        var watermarker = new InvisibleImageWatermarker<double>(watermarkStrength: 0.1);
        var tensor = CreateRandomImageTensor(3, 16, 16);
        var score = watermarker.DetectWatermark(tensor);

        Assert.True(score >= 0 && score <= 1);
    }

    #endregion

    #region AudioWatermarker Tests

    [Fact]
    public void AudioWatermarker_ProcessesAudio()
    {
        var watermarker = new AudioWatermarker<double>();
        var audio = GenerateSineWave(16000, 440.0, 16000);
        var findings = watermarker.EvaluateAudio(audio, 16000);

        Assert.NotNull(findings);
    }

    #endregion

    #region SpectralAudioWatermarker Tests

    [Fact]
    public void SpectralAudio_DetectsWatermark()
    {
        var watermarker = new SpectralAudioWatermarker<double>();
        var audio = GenerateSineWave(16000, 440.0, 16000);
        var score = watermarker.DetectWatermark(audio, 16000);

        Assert.True(score >= 0 && score <= 1);
    }

    #endregion

    #region AudioSealWatermarker Tests

    [Fact]
    public void AudioSeal_DetectsWatermark()
    {
        var watermarker = new AudioSealWatermarker<double>();
        var audio = GenerateSineWave(16000, 440.0, 16000);
        var score = watermarker.DetectWatermark(audio, 16000);

        Assert.True(score >= 0 && score <= 1);
    }

    [Fact]
    public void AudioSeal_CustomSegmentSize_Works()
    {
        var watermarker = new AudioSealWatermarker<double>(
            watermarkStrength: 0.7, segmentSamples: 2048);
        var audio = GenerateSineWave(16000, 440.0, 16000);
        var score = watermarker.DetectWatermark(audio, 16000);

        Assert.True(score >= 0 && score <= 1);
    }

    #endregion

    #region WatermarkDetector (Composite) Tests

    [Fact]
    public void Detector_TextAnalysis_ProcessesWithoutError()
    {
        var detector = new WatermarkDetector<double>();
        var findings = detector.EvaluateText(
            "This generated text may contain embedded watermark patterns for provenance tracking");

        Assert.NotNull(findings);
    }

    #endregion

    #region Cross-Module Tests

    [Fact]
    public void AllTextWatermarkers_SameText_ProduceScores()
    {
        var text = "A moderately long piece of text that is suitable for watermark detection analysis";

        var sampling = new SamplingWatermarker<double>().DetectWatermark(text);
        var lexical = new LexicalWatermarker<double>().DetectWatermark(text);
        var syntactic = new SyntacticWatermarker<double>().DetectWatermark(text);

        Assert.True(sampling >= 0 && sampling <= 1);
        Assert.True(lexical >= 0 && lexical <= 1);
        Assert.True(syntactic >= 0 && syntactic <= 1);
    }

    [Fact]
    public void AllImageWatermarkers_SameTensor_ProduceScores()
    {
        var tensor = CreateRandomImageTensor(3, 32, 32);

        var frequency = new FrequencyImageWatermarker<double>().DetectWatermark(tensor);
        var neural = new NeuralImageWatermarker<double>().DetectWatermark(tensor);
        var invisible = new InvisibleImageWatermarker<double>().DetectWatermark(tensor);

        Assert.True(frequency >= 0 && frequency <= 1);
        Assert.True(neural >= 0 && neural <= 1);
        Assert.True(invisible >= 0 && invisible <= 1);
    }

    [Fact]
    public void AllAudioWatermarkers_SameAudio_ProduceScores()
    {
        var audio = GenerateSineWave(16000, 440.0, 16000);

        var spectral = new SpectralAudioWatermarker<double>().DetectWatermark(audio, 16000);
        var audioSeal = new AudioSealWatermarker<double>().DetectWatermark(audio, 16000);

        Assert.True(spectral >= 0 && spectral <= 1);
        Assert.True(audioSeal >= 0 && audioSeal <= 1);
    }

    #endregion

    #region Helpers

    private static Tensor<double> CreateRandomImageTensor(int channels, int height, int width)
    {
        var data = new double[channels * height * width];
        var rng = RandomHelper.CreateSeededRandom(42);
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = rng.NextDouble();
        }

        return new Tensor<double>(data, new[] { channels, height, width });
    }

    private static Vector<double> GenerateSineWave(int length, double frequency, int sampleRate)
    {
        var data = new double[length];
        for (int i = 0; i < length; i++)
        {
            data[i] = 0.5 * Math.Sin(2 * Math.PI * frequency * i / sampleRate);
        }

        return new Vector<double>(data);
    }

    #endregion
}
