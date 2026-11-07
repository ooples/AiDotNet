using System;
using System.IO;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.VoiceCloning;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>A small instance of every released Chatterbox component for invariant and paper tests.</summary>
internal static class ChatterboxFixture
{
    internal static void WriteFrontendContractGraph(string path)
    {
        var builder = new AiDotNet.Onnx.OnnxGraphBuilder(new AiDotNet.Onnx.OnnxExportOptions());
        builder.AddFloatInput("features", new[] { 1, 128, -1 });
        builder.AddFloatInput("length", new[] { 1 });
        var node = new AiDotNet.Onnx.Protobuf.NodeProto { OpType = "Identity" };
        node.Input.Add("length"); node.Output.Add("tokens"); builder.AddNode(node);
        builder.AddFloatOutput("tokens", new[] { 1 });
        using var stream = File.Create(path);
        builder.WriteTo(stream);
    }

    private static readonly Lazy<string> Tokenizer = new(CreateTokenizer);

    internal static ChatterboxOptions Options() => new()
    {
        TokenizerPath = Tokenizer.Value,
        SampleRate = 24000,
        MelChannels = 8,
        HopSize = 8,
        FftSize = 32,
        NumCodebooks = 1,
        // S3Tokenizer-v2's eight ternary FSQ dimensions intrinsically produce 3^8 codes.
        CodebookSize = 6561,
        CodecFrameRate = 25,
        VocabSize = 32,
        TextStartToken = 30,
        TextStopToken = 1,
        MaxTextLength = 32,
        SpeechStartToken = 6561,
        SpeechStopToken = 6562,
        SpeechVocabularySize = 6563,
        MaxSpeechTokens = 16,
        MaxGeneratedSpeechTokens = 2,
        LLMDim = 16,
        NumLLMLayers = 1,
        NumHeads = 2,
        KeyValueHeads = 2,
        LmFeedForwardDim = 32,
        PerceiverQueries = 2,
        PerceiverHeads = 2,
        PerceiverDropout = 0.0,
        T3SpeakerEmbeddingDim = 8,
        SpeechConditionPromptTokens = 2,
        VoiceEncoderHiddenDim = 8,
        VoiceEncoderLayers = 1,
        VoiceEncoderPartialFrames = 4,
        VoiceEncoderPartialRate = 1.3,
        FlowDim = 16,
        FlowHeads = 2,
        FlowFeedForwardDim = 32,
        FlowEncoderLayers = 1,
        FlowAttentionDropoutRate = 0.0,
        StaticChunkSize = 0,
        TokenMelRatio = 2,
        Lookahead = 1,
        DecoderChannels = new[] { 8 },
        DecoderHeads = 2,
        DecoderHeadDim = 4,
        DecoderBlocks = 1,
        DecoderMidBlocks = 1,
        FlowTimesteps = 1,
        HiftBaseChannels = 16,
        HiftHarmonics = 1,
        HiftUpsampleRates = new[] { 2 },
        HiftUpsampleKernels = new[] { 4 },
        HiftSourceKernels = new[] { 3 },
        SpeechTokenizerState = 64,
        SpeechTokenizerHeads = 1,
        SpeechTokenizerLayers = 1,
        SpeakerEncoderChannels = 8,
        SpeakerEncoderGrowthRate = 4,
        SpeakerEmbeddingDim = 8,
        EncoderReferenceSeconds = 1,
        DecoderReferenceSeconds = 1,
        DropoutRate = 0.0,
        SamplingSeed = 7,
        WarmupSteps = 0,
        IncludeDiscriminators = false,
        VocoderSegmentLength = 32,
    };

    internal static Chatterbox<T> Create<T>() =>
        new(new NeuralNetworkArchitecture<T>(InputType.OneDimensional, NeuralNetworkTaskType.Generative,
            inputSize: 1, outputSize: 1) { RandomSeed = 7 }, Options());

    /// <summary>Generates beyond the release's 20 ms zero prefix and 20 ms fade-in.</summary>
    internal static Chatterbox<T> CreateForInvariants<T>()
    {
        var options = Options();
        options.MaxTextLength = 128;
        options.MaxSpeechTokens = 256;
        options.MaxGeneratedSpeechTokens = 64;
        // Match the previous invariant fixture's effective AdamW rate now that native training reads its options.
        options.LearningRate = 0.001;
        return new Chatterbox<T>(new NeuralNetworkArchitecture<T>(InputType.OneDimensional,
            NeuralNetworkTaskType.Generative, inputSize: 8, outputSize: 1024) { RandomSeed = 7 }, options);
    }

    internal static Tensor<T> Recording<T>(int seconds = 1)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var recording = new Tensor<T>(new[] { seconds * 24000 });
        for (int i = 0; i < recording.Length; i++)
            recording[i] = ops.FromDouble(0.4 * Math.Sin(2 * Math.PI * 180.0 * i / 24000)
                + 0.2 * Math.Sin(2 * Math.PI * 470.0 * i / 24000));
        return recording;
    }

    private static string CreateTokenizer()
    {
        string path = Path.Combine(Path.GetTempPath(), "aidotnet-chatterbox-tiny-tokenizer-v1.json");
        if (!File.Exists(path))
        {
            File.WriteAllText(path, """
                {
                  "added_tokens": [
                    { "id": 0, "content": "[UNK]" },
                    { "id": 1, "content": "[STOP]" },
                    { "id": 2, "content": "[SPACE]" },
                    { "id": 30, "content": "[START]" }
                  ],
                  "model": {
                    "type": "BPE",
                    "vocab": {
                      "[UNK]": 0, "[STOP]": 1, "[SPACE]": 2,
                      "h": 3, "e": 4, "l": 5, "o": 6,
                      "he": 7, "hel": 8, "hell": 9, "hello": 10,
                      "w": 11, "r": 12, "d": 13, ".": 14, ",": 15,
                      "!": 16, "?": 17, "a": 18, "t": 19, "i": 20,
                      "n": 21, "s": 22, "u": 23, "c": 24, "k": 25,
                      "b": 26, "f": 27, "x": 28, "y": 29, "[START]": 30
                    },
                    "merges": [["h", "e"], ["he", "l"], ["hel", "l"], ["hell", "o"]]
                  }
                }
                """);
        }
        return path;
    }
}
