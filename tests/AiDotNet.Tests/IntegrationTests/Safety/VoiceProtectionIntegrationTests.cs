#nullable disable
using AiDotNet.Safety.Audio;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.Safety;

/// <summary>
/// Integration tests for voice protection modules.
/// Tests PerturbationVoiceProtector, WatermarkVoiceProtector, and MaskingVoiceProtector
/// for embed/detect round-trip, audio length preservation, and protection strength.
/// </summary>
public class VoiceProtectionIntegrationTests
{
    #region PerturbationVoiceProtector Tests

    [Fact(Timeout = 120000)]
    public async Task Perturbation_ProtectsAudio_PreservesLength()
    {
        var protector = new PerturbationVoiceProtector<double>();
        var audio = GenerateSineWave(16000, 440.0, 16000);

        var protectedAudio = protector.ProtectAudio(audio, 16000);

        Assert.Equal(audio.Length, protectedAudio.Length);
    }

    [Fact(Timeout = 120000)]
    public async Task Perturbation_ModifiesAudio()
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

    [Fact(Timeout = 120000)]
    public async Task Perturbation_ShortAudio_HandlesGracefully()
    {
        var protector = new PerturbationVoiceProtector<double>();
        var audio = GenerateSineWave(100, 440.0, 16000);

        var protectedAudio = protector.ProtectAudio(audio, 16000);

        Assert.Equal(audio.Length, protectedAudio.Length);
    }

    #endregion

    #region WatermarkVoiceProtector Tests

    [Fact(Timeout = 120000)]
    public async Task Watermark_EmbedAndDetect_PreservesLength()
    {
        var protector = new WatermarkVoiceProtector<double>(watermarkStrength: 0.05, watermarkKey: 42);
        var audio = GenerateSineWave(16000, 440.0, 16000);

        var watermarked = protector.EmbedWatermark(audio, 16000);

        Assert.Equal(audio.Length, watermarked.Length);
    }

    [Fact(Timeout = 120000)]
    public async Task Watermark_EmbedWatermark_PreservesLength()
    {
        var protector = new WatermarkVoiceProtector<double>(watermarkStrength: 0.05, watermarkKey: 42);
        var audio = GenerateSineWave(16000, 440.0, 16000);

        var watermarked = protector.EmbedWatermark(audio, 16000);

        Assert.Equal(audio.Length, watermarked.Length);
    }

    [Fact(Timeout = 120000)]
    public async Task Watermark_ModifiesAudio()
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

    [Fact(Timeout = 120000)]
    public async Task Watermark_DifferentKeys_ProduceDifferentResults()
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

    [Fact(Timeout = 120000)]
    public async Task Masking_ProtectsAudio_PreservesLength()
    {
        var protector = new MaskingVoiceProtector<double>();
        var audio = GenerateSineWave(16000, 440.0, 16000);

        var protectedAudio = protector.ProtectAudio(audio, 16000);

        Assert.Equal(audio.Length, protectedAudio.Length);
    }

    [Fact(Timeout = 120000)]
    public async Task Masking_ModifiesAudio()
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

    [Fact(Timeout = 120000)]
    public async Task Masking_LongAudio_ProcessesWithoutError()
    {
        var protector = new MaskingVoiceProtector<double>();
        var audio = GenerateSineWave(48000, 440.0, 16000);

        var protectedAudio = protector.ProtectAudio(audio, 16000);

        Assert.Equal(audio.Length, protectedAudio.Length);
    }

    #endregion

    #region Cross-Module Tests

    [Fact(Timeout = 120000)]
    public async Task AllProtectors_PreserveAudioLength()
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

    [Fact(Timeout = 120000)]
    public async Task Watermark_FrameBatchesMatchOneBatch()
    {
        await Task.Yield();
        // 30 frames in batches of 3: the overlap carried between batches and the global chip index must make the
        // batched result identical to the single-batch one.
        var protector = new WatermarkVoiceProtector<double>(watermarkStrength: 0.05, watermarkKey: 42);
        var audio = GenerateSineWave(8000, 440.0, 16000);

        var whole = protector.EmbedWatermark(audio, 16000, frameBatch: 1 << 20);
        var batched = protector.EmbedWatermark(audio, 16000, frameBatch: 3);

        Assert.Equal(whole.Length, batched.Length);
        for (int i = 0; i < whole.Length; i++)
        {
            Assert.Equal(whole[i], batched[i], 12);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task VoiceprintDetector_HalfMatchesFloat()
    {
        await Task.Yield();
        // A 0.5-amplitude 1 kHz tone: its Half spectral first moment (~128,000) overflows Half's 65504.
        var tone = new float[8192];
        for (int i = 0; i < tone.Length; i++) tone[i] = 0.5f * (float)Math.Sin(2 * Math.PI * 1000.0 * i / 16000.0);
        var asHalf = new Half[tone.Length];
        for (int i = 0; i < tone.Length; i++) asHalf[i] = (Half)tone[i];
        var widened = new float[tone.Length];
        for (int i = 0; i < tone.Length; i++) widened[i] = (float)asHalf[i];

        var fromHalf = new AiDotNet.Safety.Audio.VoiceprintDeepfakeDetector<Half>(0.0, 16000)
            .EvaluateAudio(new Vector<Half>(asHalf), 16000);
        var fromFloat = new AiDotNet.Safety.Audio.VoiceprintDeepfakeDetector<float>(0.0, 16000)
            .EvaluateAudio(new Vector<float>(widened), 16000);

        Assert.Equal(fromFloat.Count, fromHalf.Count);
        for (int i = 0; i < fromFloat.Count; i++)
        {
            Assert.Equal(fromFloat[i].Confidence, fromHalf[i].Confidence, 6);
        }
    }
}
