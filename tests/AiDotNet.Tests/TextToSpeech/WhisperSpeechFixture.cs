using System;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.MultiModal;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// A tiny WhisperSpeech ready to run, every part shrunk:
/// <list type="bullet">
/// <item>Whisper: 16 wide, 2 heads, one encoder and one decoder layer, a 16-position audio window (0.32 s), 32 text
/// positions, the real 51,865-token multilingual vocabulary.</item>
/// <item>Semantic tokenizer: one rotary block, 8 codes of width 4 (eight tokens a window).</item>
/// <item>T2S and S2A: depth 2 (one encoder, three decoder blocks), 2 heads of 8; four EnCodec codebooks of 64 codes.</item>
/// <item>Speaker: ECAPA-TDNN of width 8 (24 at the MFA), embedding 6; EnCodec and Vocos as in the MARS5 fixture (2560-sample
/// hop).</item>
/// <item>Training: no warmup, a 30-step one-cycle schedule.</item>
/// </list>
/// The model speaks as a speaker made from a quarter second of tone.
/// </summary>
internal static class WhisperSpeechFixture
{
    public const int SampleRate = 24000;

    public static WhisperSpeechOptions Options() => new()
    {
        MaxTextLength = 32, SemanticCodes = 8, SemanticWidth = 4,
        T2SDepth = 2, T2SHeads = 2, T2SHeadWidth = 8, T2SWarmupSteps = 0,
        S2ADepth = 2, S2AHeads = 2, S2AHeadWidth = 8, S2AWarmupSteps = 0,
        WhisperWidth = 16, WhisperHeads = 2, WhisperLayers = 1, WhisperAudioContext = 16, WhisperTextContext = 32,
        VqHeads = 2, VqWarmupSteps = 0,
        SpeakerChannels = new[] { 8, 8, 8, 8, 24 }, SpeakerAttentionChannels = 4, SpeakerSeChannels = 4, SpeakerEmbeddingSize = 6,
        HopSize = 2560,
        Codec = new AiDotNet.Audio.Generation.EnCodecOptions
        {
            SampleRate = 24000, NumQuantizers = 4, CodebookSize = 64, Filters = 4, Ratios = [8, 5, 4, 4, 4], Dimension = 8,
            ResidualKernelSizes = [3, 1], TargetBandwidthKbps = 0.225,
        },
        VocosDim = 16, VocosIntermediateDim = 32, VocosLayers = 1,
        TotalSteps = 30,
    };

    public static WhisperSpeech<T> Tts<T>(WhisperSpeechOptions? options = null)
    {
        var model = new WhisperSpeech<T>(new NeuralNetworkArchitecture<T>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1),
            options ?? Options());
        model.SpeakerEmbedding = model.CreateSpeaker(Tones<T>(0.25), SampleRate);
        return model;
    }

    /// <summary>A recording: two tones at <see cref="SampleRate"/>.</summary>
    public static Tensor<T> Tones<T>(double seconds, double pitch = 220)
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        int n = (int)(seconds * SampleRate);
        var audio = new Tensor<T>(new[] { n });
        for (int i = 0; i < n; i++)
            audio[i] = ops.FromDouble(0.3 * Math.Sin(2 * Math.PI * pitch * i / SampleRate) + 0.1 * Math.Sin(2 * Math.PI * 3.1 * pitch * i / SampleRate));
        return audio;
    }
}
