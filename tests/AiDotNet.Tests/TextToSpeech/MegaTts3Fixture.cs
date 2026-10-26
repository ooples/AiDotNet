using System;
using System.IO;
using System.Linq;
using AiDotNet.Audio.Whisper;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.Latest;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// The generated suite's MegaTTS 3: the tiny configuration of the parity tests (a width-16 diffusion transformer with one
/// layer, the duration model of megatts3_duration_reference.py with its weights, a tiny WaveVAE) with
/// megatts3_speaking_fixture.py's front end: its tokenizer, a tiny Qwen2 g2p model that answers every text with one of
/// four phone sequences (chosen by the text's first character) and a tiny Whisper aligner that aligns every recording,
/// so synthesis and training run the release's own path. Two diffusion steps, no warm-up; the voice is half a second of
/// noise.
/// </summary>
internal static class MegaTts3Fixture
{
    private static JObject Load(string fileName)
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        return JObject.Parse(File.ReadAllText(path));
    }

    private static readonly Lazy<(JObject Speaking, JObject Duration)> Data =
        new(() => (Load("megatts3_speaking.json"), (JObject)Load("megatts3_duration_reference.json")["state"]!));

    public static MegaTTS3<T> Create<T>()
    {
        var (speaking, duration) = Data.Value;
        var g2p = speaking["g2p_config"]!;
        var options = MegaTts3PipelineReferenceTests.TinyOptions();
        options.G2pVocabSize = (int)g2p["vocab_size"]!;
        options.G2pTokenizerJson = (string)speaking["tokenizer"]!;
        options.G2pMaxNewTokens = 16;
        options.AlignerVocabulary = (int)speaking["aligner_vocab"]!;
        options.Steps = 2;
        // The paper warms up from a rate of 0 over 10K steps; the suite's few steps train at the rate itself.
        options.WarmupSteps = 0;
        var model = new MegaTTS3<T>(new NeuralNetworkArchitecture<T>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1),
            options);
        double[] Read(JObject state, string name) => state[name]?.Select(v => (double)v).ToArray()
            ?? throw new InvalidDataException($"The fixture has no tensor '{name}'.");
        var g2pState = (JObject)speaking["g2p_state"]!;
        model.G2p.Network.LoadTorchWeights((name, _) => Read(g2pState, name));
        var alignerState = (JObject)speaking["aligner_state"]!;
        model.Aligner.LoadTorch(name => name == "encoder.positional_embedding" ? OpenAiWhisperNetwork<T>.Sinusoids(3000, 16) : Read(alignerState, name));
        model.Duration.LoadTorch(AiDotNet.Tensors.Engines.AiDotNetEngine.Current, name => Read(duration, name));
        var random = new Random(201);
        var ops = MathHelper.GetNumericOperations<T>();
        var voice = new T[12000];
        for (int i = 0; i < voice.Length; i++) voice[i] = ops.FromDouble(0.2 * (2 * random.NextDouble() - 1));
        model.Voice = new TtsVoice<T> { ReferenceAudio = new Tensor<T>(new[] { voice.Length }, new Vector<T>(voice)) };
        return model;
    }
}
