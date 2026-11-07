using AiDotNet.Audio.Effects;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.Tests.TextToSpeech;

internal static class ZonosFixture
{
    internal static ZonosOptions Options() => new()
    {
        HiddenDim = 32, Layers = 2, Heads = 4, KeyValueHeads = 2, AttentionIntermediate = 128,
        RotaryDim = 8, StateDim = 8, MambaHeadDim = 8, ChunkSize = 3,
        MaxNewTokens = 24, SampleRate = 16000, HopSize = 8,
        Codec = new DACOptions
        {
            SampleRate = 16000, EncoderDim = 4, EncoderRates = new[] { 2, 4 }, DecoderDim = 16,
            DecoderRates = new[] { 4, 2 }, NumQuantizers = 9, CodebookSize = 1024, CodebookDim = 2,
            TargetBandwidthKbps = 180,
        },
    };

    internal static Zonos<T> Create<T>() => new(new NeuralNetworkArchitecture<T>(InputType.OneDimensional,
        NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1), Options());
}
