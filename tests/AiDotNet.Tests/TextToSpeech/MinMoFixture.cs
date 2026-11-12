using System;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.MultiModal;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// A tiny MinMo ready to run, every part shrunk:
/// <list type="bullet">
/// <item>Voice encoder: a one-layer, 16-wide SenseVoice-Large (Whisper-style) encoder over 80 log-mel bands (Whisper supports 80 or 128), or a one-block,
/// 16-wide SAN-M with <see cref="MinMoVoiceEncoder.SenseVoiceSmall"/>.</item>
/// <item>Input projector: one 16-wide Transformer layer of 2 heads after a 32-wide bottleneck.</item>
/// <item>LLM: a one-layer, 16-wide Qwen2 over Qwen2.5's real tokenizer (151,665 rows), LoRA rank 2.</item>
/// <item>Voice decoder: CosyVoice 2 at the generated suite's tiny size (16 wide, one block each).</item>
/// <item>Duplex predictor: 16 wide, 2 heads.</item>
/// </list>
/// </summary>
internal static class MinMoFixture
{
    public const int SampleRate = 24000;

    public static MinMoOptions Options(MinMoVoiceEncoder encoder = MinMoVoiceEncoder.SenseVoiceLarge) => new()
    {
        Encoder = encoder,
        LargeEncoderMels = 80, LargeEncoderWidth = 16, LargeEncoderHeads = 2, LargeEncoderLayers = 1,
        SmallEncoderMels = 8, SmallEncoderWidth = 16, SmallEncoderHeads = 2, SmallEncoderFeedForward = 32, SmallEncoderBlocks = 1, SmallEncoderTpBlocks = 1,
        ProjectorFeedForward = 32, ProjectorLayers = 1, ProjectorHeads = 2,
        HiddenDim = 16, NumHeads = 2, KeyValueHeads = 1, Layers = 1, Intermediate = 32, VocabSize = 151665, LoraRank = 2,
        MaxNewTokens = 6, DuplexWidth = 16, DuplexHeads = 2, DuplexFeedForward = 32, LearningRate = 1e-2,
        // Plain training steps train the voice decoder (stage 2): pre-alignment moves only the input projector, which a
        // one-layer LLM over 151,665 tokens barely feels in a few steps.
        Stage = MinMoTrainingStage.TextToSpeech,
        VoiceDecoder = new CosyVoice2Options
        {
            LLMDim = 16, NumLLMLayers = 1, NumHeads = 2, KeyValueHeads = 1, LmFeedForwardDim = 32, FlowDim = 16, FlowHeads = 2,
            FlowFeedForwardDim = 32, FlowEncoderLayers = 1, StaticChunkSize = 2, DecoderChannels = new[] { 16 }, DecoderHeads = 2,
            DecoderHeadDim = 8, DecoderBlocks = 1, DecoderMidBlocks = 1, FlowTimesteps = 2, HiftBaseChannels = 16, HiftHarmonics = 2,
            SpeechTokenizerState = 64, SpeechTokenizerHeads = 1, SpeechTokenizerLayers = 1, SpeakerEncoderChannels = 8,
            SpeakerEncoderGrowthRate = 4, SpeakerEmbeddingDim = 8, MelChannels = 8, DropoutRate = 0.0,
        },
    };

    public static MinMo<T> Tts<T>(MinMoOptions? options = null) =>
        new(new NeuralNetworkArchitecture<T>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1),
            options ?? Options());

    /// <summary>A chirp and a tone at 24 kHz, <paramref name="seconds"/> long.</summary>
    public static Tensor<T> Tones<T>(double seconds, double low = 140.0, double high = 420.0)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int samples = (int)(SampleRate * seconds);
        var audio = new T[samples];
        for (int i = 0; i < samples; i++)
        {
            double t = (double)i / SampleRate;
            audio[i] = ops.FromDouble(0.4 * Math.Sin(2 * Math.PI * low * (1 + t) * t) + 0.2 * Math.Sin(2 * Math.PI * high * t));
        }
        return new Tensor<T>(new[] { samples }, new Vector<T>(audio));
    }
}
