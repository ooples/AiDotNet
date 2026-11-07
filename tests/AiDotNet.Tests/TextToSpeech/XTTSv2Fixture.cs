using System;
using System.IO;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.VoiceCloning;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>A bounded instance of every released XTTS-v2 component for generated model invariants.</summary>
internal static class XTTSv2Fixture
{
    private static readonly Lazy<string> Tokenizer = new(CreateTokenizer);

    internal static XTTSv2<T> Create<T>() =>
        new(Architecture<T>(), Configure(new XTTSv2Options()));

    internal static XTTSv2Clone<T> CreateClone<T>() =>
        new(Architecture<T>(), Configure(new XTTSv2CloneOptions { MinReferenceDurationSec = 0.1 }));

    private static NeuralNetworkArchitecture<T> Architecture<T>() =>
        new(InputType.OneDimensional, NeuralNetworkTaskType.Generative, inputSize: 16, outputSize: 16)
        {
            RandomSeed = 7,
        };

    private static TOptions Configure<TOptions>(TOptions options) where TOptions : XTTSv2Options
    {
        options.TokenizerPath = Tokenizer.Value;
        options.NumCodebooks = 1;
        options.CodebookSize = 16;
        options.AudioVocabularySize = 18;
        options.StartAudioToken = 16;
        options.StopAudioToken = 17;
        options.TextVocabularySize = 32;
        options.LLMDim = 32;
        options.NumLLMLayers = 1;
        options.NumHeads = 4;
        options.ConditioningBlocks = 1;
        options.ConditioningHeads = 4;
        options.PerceiverDepth = 1;
        options.PerceiverLatents = 2;
        options.PerceiverHeads = 1;
        options.DvaeHiddenDim = 8;
        options.DvaeCodebookDim = 8;
        options.DvaeResidualBlocks = 1;
        options.SpeakerEmbeddingDim = 8;
        options.SpeakerEncoderFilters = [8, 8, 8, 8];
        options.DecoderInitialChannels = 16;
        options.DecoderUpsampleRates = [2, 2];
        options.DecoderUpsampleKernels = [4, 4];
        options.DecoderResblockKernels = [3];
        options.DecoderResblockDilations = [[1]];
        options.ConditioningFftSize = 256;
        options.ConditioningWindowSize = 256;
        options.ConditioningHopSize = 64;
        options.DvaeFftSize = 256;
        options.DvaeWindowSize = 256;
        options.DvaeHopSize = 64;
        options.MaxTextLength = 64;
        options.MaxCodecFrames = 2;
        options.MaximumReferenceSeconds = 0.34;
        options.ConditioningSeconds = 0.34;
        options.ConditioningChunkSeconds = 0.34;
        options.MinimumConditioningChunkSeconds = 0.1;
        options.DropoutRate = 0;
        options.TopK = 1;
        options.TopP = 1;
        options.LearningRate = 1e-3;
        options.SamplingSeed = 7;
        return options;
    }

    private static string CreateTokenizer()
    {
        string path = Path.Combine(Path.GetTempPath(), "aidotnet-xttsv2-tiny-tokenizer-v1.json");
        if (!File.Exists(path))
        {
            File.WriteAllText(path, """
                {
                  "added_tokens": [
                    { "id": 0, "content": "[START]", "special": true },
                    { "id": 1, "content": "[STOP]", "special": true },
                    { "id": 2, "content": "[en]", "special": true },
                    { "id": 3, "content": "[SPACE]", "special": true }
                  ],
                  "pre_tokenizer": {
                    "type": "ByteLevel",
                    "add_prefix_space": false,
                    "use_regex": false
                  },
                  "model": {
                    "type": "BPE",
                    "vocab": {
                      "a": 4, "b": 5, "c": 6, "d": 7, "e": 8, "f": 9, "g": 10,
                      "h": 11, "i": 12, "j": 13, "k": 14, "l": 15, "m": 16, "n": 17,
                      "o": 18, "p": 19, "q": 20, "r": 21, "s": 22, "t": 23, "u": 24,
                      "v": 25, "w": 26, "x": 27, "y": 28, "z": 29, "!": 30, "?": 31
                    },
                    "merges": []
                  }
                }
                """);
        }
        return path;
    }
}
