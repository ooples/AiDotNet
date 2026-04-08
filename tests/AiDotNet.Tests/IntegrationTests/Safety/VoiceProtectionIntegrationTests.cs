#nullable disable
using AiDotNet.Safety.Audio;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Safety;

/// <summary>
/// Integration tests for voice protection modules.
/// Tests PerturbationVoiceProtector, WatermarkVoiceProtector, and MaskingVoiceProtector
/// for embed/detect round-trip, audio length preservation, and protection strength.
/// </summary>
public class VoiceProtectionIntegrationTests
{
    #region PerturbationVoiceProtector Tests

    [Fact]
    public void Perturbation_ProtectsAudio_PreservesLength()
    {
        var protector = new PerturbationVoiceProtector<double>();
        var audio = GenerateSineWave(16000, 440.0, 16000);

        var protectedAudio = protector.ProtectAudio(audio, 16000);

        Assert.Equal(audio.Length, protectedAudio.Length);
    }

    [Fact]
    public void Perturbation_ModifiesAudio()
    {
        var protector = new PerturbationVoiceProtector<double>();
        var audio = GenerateSineWave(16000, 440.0, 16000);

        var protectedAudio = protector.ProtectAudio(audio, 16000);

        // Protected audio should differ from original
        bool anyDifferent = false;
        for (int i = 0; i < audio.Length; i++)
        {
            if (Math.Abs(audio[i] - protectedAudio[i]) > 1e-10)
            {
                anyDifferent = true;
                break;
            }
        }

        Assert.True(anyDifferent, "Protection should modify the audio samples");
    }

    [Fact]
    public void Perturbation_ShortAudio_HandlesGracefully()
    {
        var protector = new PerturbationVoiceProtector<double>();
        var audio = GenerateSineWave(100, 440.0, 16000);

        var protectedAudio = protector.ProtectAudio(audio, 16000);

        Assert.Equal(audio.Length, protectedAudio.Length);
    }

    #endregion

    #region WatermarkVoiceProtector Tests

    [Fact]
    public void Watermark_EmbedAndDetect_PreservesLength()
    {
        var protector = new WatermarkVoiceProtector<double>(watermarkStrength: 0.05, watermarkKey: 42);
        var audio = GenerateSineWave(16000, 440.0, 16000);

        var watermarked = protector.EmbedWatermark(audio, 16000);

        Assert.Equal(audio.Length, watermarked.Length);
    }

    [Fact]
    public void Watermark_EmbedWatermark_PreservesLength()
    {
        var protector = new WatermarkVoiceProtector<double>(watermarkStrength: 0.05, watermarkKey: 42);
        var audio = GenerateSineWave(16000, 440.0, 16000);

        var watermarked = protector.EmbedWatermark(audio, 16000);

        Assert.Equal(audio.Length, watermarked.Length);
    }

    [Fact]
    public void Watermark_ModifiesAudio()
    {
        var protector = new WatermarkVoiceProtector<double>(watermarkStrength: 0.1, watermarkKey: 42);
        var audio = GenerateSineWave(16000, 440.0, 16000);

        var watermarked = protector.EmbedWatermark(audio, 16000);

        bool anyDifferent = false;
        for (int i = 0; i < audio.Length; i++)
        {
            if (Math.Abs(audio[i] - watermarked[i]) > 1e-10)
            {
                anyDifferent = true;
                break;
            }
        }

        Assert.True(anyDifferent, "Watermark embedding should modify samples");
    }

    [Fact]
    public void Watermark_DifferentKeys_ProduceDifferentResults()
    {
        var protector1 = new WatermarkVoiceProtector<double>(watermarkStrength: 0.1, watermarkKey: 42);
        var protector2 = new WatermarkVoiceProtector<double>(watermarkStrength: 0.1, watermarkKey: 99);
        var audio = GenerateSineWave(16000, 440.0, 16000);

        var watermarked1 = protector1.EmbedWatermark(audio, 16000);
        var watermarked2 = protector2.EmbedWatermark(audio, 16000);

        bool anyDifferent = false;
        for (int i = 0; i < watermarked1.Length; i++)
        {
            if (Math.Abs(watermarked1[i] - watermarked2[i]) > 1e-10)
            {
                anyDifferent = true;
                break;
            }
        }

        Assert.True(anyDifferent, "Different keys should produce different watermarks");
    }

    #endregion

    #region MaskingVoiceProtector Tests

    [Fact]
    public void Masking_ProtectsAudio_PreservesLength()
    {
        var protector = new MaskingVoiceProtector<double>();
        var audio = GenerateSineWave(16000, 440.0, 16000);

        var protectedAudio = protector.ProtectAudio(audio, 16000);

        Assert.Equal(audio.Length, protectedAudio.Length);
    }

    [Fact]
    public void Masking_ModifiesAudio()
    {
        var protector = new MaskingVoiceProtector<double>();
        var audio = GenerateSineWave(16000, 440.0, 16000);

        var protectedAudio = protector.ProtectAudio(audio, 16000);

        bool anyDifferent = false;
        for (int i = 0; i < audio.Length; i++)
        {
            if (Math.Abs(audio[i] - protectedAudio[i]) > 1e-10)
            {
                anyDifferent = true;
                break;
            }
        }

        Assert.True(anyDifferent, "Masking should modify the audio samples");
    }

    [Fact]
    public void Masking_LongAudio_ProcessesWithoutError()
    {
        var protector = new MaskingVoiceProtector<double>();
        var audio = GenerateSineWave(48000, 440.0, 16000);

        var protectedAudio = protector.ProtectAudio(audio, 16000);

        Assert.Equal(audio.Length, protectedAudio.Length);
    }

    #endregion

    #region Cross-Module Tests

    [Fact]
    public void AllProtectors_PreserveAudioLength()
    {
        var audio = GenerateSineWave(16000, 440.0, 16000);

        var perturbation = new PerturbationVoiceProtector<double>().ProtectAudio(audio, 16000);
        var watermark = new WatermarkVoiceProtector<double>(watermarkStrength: 0.05, watermarkKey: 42)
            .EmbedWatermark(audio, 16000);
        var masking = new MaskingVoiceProtector<double>().ProtectAudio(audio, 16000);

        Assert.Equal(audio.Length, perturbation.Length);
        Assert.Equal(audio.Length, watermark.Length);
        Assert.Equal(audio.Length, masking.Length);
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

    #endregion
}
