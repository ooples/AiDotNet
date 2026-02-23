#nullable disable
using AiDotNet.Enums;
using AiDotNet.Safety;
using AiDotNet.Safety.Audio;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Safety;

/// <summary>
/// Integration tests for audio safety modules.
/// Tests SpectralDeepfakeDetector, VoiceprintDeepfakeDetector, WatermarkDeepfakeDetector,
/// TranscriptionToxicityDetector, and AcousticToxicityDetector.
/// </summary>
public class AudioSafetyIntegrationTests
{
    #region SpectralDeepfakeDetector Tests

    [Fact]
    public void Spectral_SineWave_ProcessesWithoutError()
    {
        var detector = new SpectralDeepfakeDetector<double>();
        var audio = GenerateSineWave(16000, 440.0, 16000);
        var findings = detector.EvaluateAudio(audio, 16000);

        Assert.NotNull(findings);
    }

    [Fact]
    public void Spectral_MultiFrequency_ProcessesWithoutError()
    {
        var detector = new SpectralDeepfakeDetector<double>();
        var audio = GenerateMultiFrequencyWave(16000, new[] { 220.0, 440.0, 880.0 }, 16000);
        var findings = detector.EvaluateAudio(audio, 16000);

        Assert.NotNull(findings);
    }

    [Fact]
    public void Spectral_ShortAudio_HandlesGracefully()
    {
        var detector = new SpectralDeepfakeDetector<double>();
        var audio = GenerateSineWave(1000, 440.0, 16000);
        var findings = detector.EvaluateAudio(audio, 16000);

        Assert.NotNull(findings);
    }

    [Fact]
    public void Spectral_DifferentSampleRates_Work()
    {
        var detector = new SpectralDeepfakeDetector<double>();

        foreach (var sampleRate in new[] { 8000, 16000, 44100 })
        {
            var audio = GenerateSineWave(sampleRate, 440.0, sampleRate);
            var findings = detector.EvaluateAudio(audio, sampleRate);
            Assert.NotNull(findings);
        }
    }

    #endregion

    #region VoiceprintDeepfakeDetector Tests

    [Fact]
    public void Voiceprint_SineWave_ProcessesWithoutError()
    {
        var detector = new VoiceprintDeepfakeDetector<double>();
        var audio = GenerateSineWave(16000, 440.0, 16000);
        var findings = detector.EvaluateAudio(audio, 16000);

        Assert.NotNull(findings);
    }

    [Fact]
    public void Voiceprint_LongerAudio_ProcessesWithoutError()
    {
        var detector = new VoiceprintDeepfakeDetector<double>();
        var audio = GenerateSineWave(32000, 440.0, 16000);
        var findings = detector.EvaluateAudio(audio, 16000);

        Assert.NotNull(findings);
    }

    #endregion

    #region WatermarkDeepfakeDetector Tests

    [Fact]
    public void WatermarkDeepfake_SineWave_ProcessesWithoutError()
    {
        var detector = new WatermarkDeepfakeDetector<double>();
        var audio = GenerateSineWave(16000, 440.0, 16000);
        var findings = detector.EvaluateAudio(audio, 16000);

        Assert.NotNull(findings);
    }

    [Fact]
    public void WatermarkDeepfake_CustomThreshold_Works()
    {
        var detector = new WatermarkDeepfakeDetector<double>(threshold: 0.3);
        var audio = GenerateSineWave(16000, 440.0, 16000);
        var findings = detector.EvaluateAudio(audio, 16000);

        Assert.NotNull(findings);
    }

    #endregion

    #region TranscriptionToxicityDetector Tests

    [Fact]
    public void Transcription_SineWave_ProcessesWithoutError()
    {
        var detector = new TranscriptionToxicityDetector<double>();
        var audio = GenerateSineWave(16000, 440.0, 16000);
        var findings = detector.EvaluateAudio(audio, 16000);

        Assert.NotNull(findings);
    }

    [Fact]
    public void Transcription_CustomThreshold_Works()
    {
        var detector = new TranscriptionToxicityDetector<double>(threshold: 0.3);
        var audio = GenerateSineWave(8000, 440.0, 16000);
        var findings = detector.EvaluateAudio(audio, 16000);

        Assert.NotNull(findings);
    }

    #endregion

    #region AcousticToxicityDetector Tests

    [Fact]
    public void Acoustic_SineWave_ProcessesWithoutError()
    {
        var detector = new AcousticToxicityDetector<double>();
        var audio = GenerateSineWave(16000, 440.0, 16000);
        var findings = detector.EvaluateAudio(audio, 16000);

        Assert.NotNull(findings);
    }

    [Fact]
    public void Acoustic_MultiFrequency_ProcessesWithoutError()
    {
        var detector = new AcousticToxicityDetector<double>();
        var audio = GenerateMultiFrequencyWave(16000, new[] { 100.0, 500.0, 2000.0 }, 16000);
        var findings = detector.EvaluateAudio(audio, 16000);

        Assert.NotNull(findings);
    }

    #endregion

    #region Cross-Module Tests

    [Fact]
    public void AllDeepfakeDetectors_SameAudio_ProduceResults()
    {
        var audio = GenerateSineWave(16000, 440.0, 16000);

        Assert.NotNull(new SpectralDeepfakeDetector<double>().EvaluateAudio(audio, 16000));
        Assert.NotNull(new VoiceprintDeepfakeDetector<double>().EvaluateAudio(audio, 16000));
        Assert.NotNull(new WatermarkDeepfakeDetector<double>().EvaluateAudio(audio, 16000));
    }

    [Fact]
    public void AllToxicityDetectors_SameAudio_ProduceResults()
    {
        var audio = GenerateSineWave(16000, 440.0, 16000);

        Assert.NotNull(new TranscriptionToxicityDetector<double>().EvaluateAudio(audio, 16000));
        Assert.NotNull(new AcousticToxicityDetector<double>().EvaluateAudio(audio, 16000));
    }

    #endregion

    #region Helpers

    private static Vector<double> GenerateSineWave(int length, double frequency, int sampleRate)
    {
        var data = new double[length];
        for (int i = 0; i < length; i++)
        {
            data[i] = 0.5 * Math.Sin(2 * Math.PI * frequency * i / sampleRate);
        }

        return new Vector<double>(data);
    }

    private static Vector<double> GenerateMultiFrequencyWave(
        int length, double[] frequencies, int sampleRate)
    {
        var data = new double[length];
        for (int i = 0; i < length; i++)
        {
            double sum = 0;
            foreach (var freq in frequencies)
            {
                sum += Math.Sin(2 * Math.PI * freq * i / sampleRate);
            }

            data[i] = sum / frequencies.Length * 0.5;
        }

        return new Vector<double>(data);
    }

    #endregion
}
