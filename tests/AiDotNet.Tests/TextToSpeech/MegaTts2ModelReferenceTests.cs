using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.FrontEnd;
using AiDotNet.TextToSpeech.Latest;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// Mega-TTS 2's autoencoder and language models against their composition from the authors' modules and CVQ-VAE's
/// quantizer (tools/reference-data/megatts2_model_reference.py): the content, the VQ encoder and its codes and loss, the
/// MRTE, the decoded mel spectrogram and the L2 and adversarial terms, CVQ-VAE's codebook update, and the P-LLM's and
/// duration model's outputs and losses over three sentences of two speakers.
/// </summary>
public class MegaTts2ModelReferenceTests
{
    /// <summary>The script's sizes: width 16 (filter 32), two layers or blocks everywhere, 10 mel bins, 12 codes of width 8,
    /// pooling by 4, a 16-wide P-LLM of 4 heads and an 8-wide duration model of 2 heads reading durations up to 3.</summary>
    internal static MegaTTS2Options TinyOptions() => new()
    {
        HiddenDim = 16, ContentLayers = 2, ContentHeads = 2, ContentFilter = 32, ProsodyHidden = 12, ProsodyBlocks = 2, Compression = 4,
        Codes = 12, CodeDimension = 8, TimbreHidden = 8, TimbreBlocks = 2, Downsampling = 4, TimbreHeads = 2, DecoderBlocks = 2,
        MelChannels = 10, LmLayers = 2, LmHidden = 16, LmHeads = 4, DurationLayers = 2, DurationHidden = 8, DurationHeads = 2,
        DurationCodes = 3, DiscriminatorWindows = new[] { 8, 16 }, DiscriminatorHidden = 6, TopK = 3, VocoderInitialChannels = 32,
        VocoderResBlockKernels = new[] { 3 }, VocoderResBlockDilations = new[] { new[] { 1, 3, 5 } },
    };

    private static MegaTTS2<double> Create(MegaTTS2Options options) =>
        new(new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1), options);

    [Fact(Timeout = 300000)]
    public async Task Model_MatchesAuthorsComposition()
    {
        await Task.Yield();
        const string fileName = "megatts2_model_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        var f = JObject.Parse(File.ReadAllText(path));
        double[] Doubles(JToken t) => t.Select(v => (double)v).ToArray();
        int[] Ints(JToken t) => t.Select(v => (int)v).ToArray();
        var state = (JObject)f["state"]!;
        var used = new HashSet<string>();
        double[] Read(string name)
        {
            used.Add(name);
            return Doubles(state[name] ?? throw new InvalidDataException(name));
        }
        var options = TinyOptions();
        var model = Create(options);
        model.Network.LoadTorch(Read);
        model.SetTrainingMode(false);
        Assert.Empty(state.Properties().Select(p => p.Name).Where(n => !used.Contains(n)));

        void Close(JToken expected, Tensor<double> actual, string what, double tolerance = 1e-9)
        {
            var e = expected.Type == JTokenType.Array ? Doubles(expected) : new[] { (double)expected };
            Assert.Equal(e.Length, actual.Length);
            for (int i = 0; i < e.Length; i++)
                Assert.True(Math.Abs(e[i] - actual[i]) <= tolerance * Math.Max(1, Math.Abs(e[i])), $"{what}[{i}]: {e[i]} vs {actual[i]}");
        }
        using var _ = new NoGradScope<double>();
        var network = model.Network;
        var phones = NatSpeechFrontEnd.Encode(f["phones"]!.Select(v => (string)v!), model.Vocabulary);
        var durations = Ints(f["durations"]!);
        int frames = durations.Sum();
        var mel = new Tensor<double>(new[] { frames, 10 }, new Vector<double>(Doubles(f["mel"]!)));
        var reference = new Tensor<double>(new[] { 9, 10 }, new Vector<double>(Doubles(f["reference"]!)));
        var content = model.Content(phones);
        Close(f["content"]!, content, "content");
        var hidden = network.ProsodyHidden(mel);
        Close(f["hidden"]!, hidden, "hidden");
        Assert.Equal(Ints(f["codes"]!), network.Quantizer.Encode(hidden));
        var euclidean = new MegaTts2Quantizer<double>(AiDotNet.Tensors.Engines.AiDotNetEngine.Current, new List<LayerBase<double>>(), 12, 8,
            MegaTts2CodeDistance.Euclidean, MegaTts2CodeAnchor.Closest, 0.25, 0.99);
        euclidean.Codebook.LoadTable(Doubles(state["codebook.weight"]!));
        Assert.Equal(Ints(f["l2_codes"]!), euclidean.Encode(hidden));
        var states = network.TimbreStates(reference);
        Close(f["timbre_states"]!, states, "timbre states");
        Close(f["timbre"]!, network.Timbre(content, states), "timbre");

        var u = new MegaTTS2<double>.Utterance(phones, durations, MegaTts3Aligner<double>.LengthRegulate(durations), mel, reference);
        var r = model.Reconstruct(u, update: false);
        Close(f["decoded"]!, r.Mel, "decoded");
        Close(f["vq"]!, r.VqLoss, "vq");
        Close(f["total"]!, model.AcousticLoss(u, null, update: false).Loss, "total");
        Close(new JArray((double)f["total"]! + 0.05 * (double)f["adversarial"]!), model.AcousticLoss(u, Ints(f["starts"]!), update: false).Loss, "with adversarial");

        // The language models: three sentences of two speakers.
        var sentences = ((JArray)f["sentences"]!).Select((s, i) => (Speaker: (int)s["speaker"]!, Sample: new TtsTrainingSample<double>
        {
            Tokens = new Tensor<double>(new[] { s["phones"]!.Count() },
                new Vector<double>(NatSpeechFrontEnd.Encode(s["phones"]!.Select(v => (string)v!), model.Vocabulary).Select(v => (double)v).ToArray())),
            Durations = Ints(s["durations"]!),
            Mel = new Tensor<double>(new[] { Ints(s["durations"]!).Sum(), 10 }, new Vector<double>(Doubles(f["mels"]![i]!))),
        })).ToList();
        var speakers = sentences.GroupBy(s => s.Speaker).OrderBy(g => g.Key)
            .Select(g => (IReadOnlyList<TtsTrainingSample<double>>)g.Select(s => s.Sample).ToList()).ToList();
        var sequence = model.LanguageSequenceOf(speakers);
        Assert.Equal(Ints(f["lm_tokens"]!), sequence.Tokens);
        Assert.Equal(Ints(f["lm_positions"]!), sequence.Positions);
        Assert.Equal(Ints(f["lm_targets"]!), sequence.Targets);
        Assert.Equal(Ints(f["dur_tokens"]!), sequence.DurationTokens);
        Assert.Equal(Ints(f["dur_positions"]!), sequence.DurationPositions);
        Close(f["logits"]!, network.ProsodyModel.Forward(sequence.Tokens, sequence.Features, sequence.Positions, sequence.Speakers), "logits");
        Close(f["ce"]!, model.ProsodyLoss(sequence), "ce");
        Close(f["dur_pred"]!, network.DurationModel.Forward(sequence.DurationTokens, sequence.DurationFeatures, sequence.DurationPositions,
            sequence.DurationSpeakers), "dur_pred");
        Close(f["dur_mse"]!, model.DurationLoss(sequence), "dur_mse");

        // CVQ-VAE's training-mode step: the usage and the codebook after the dynamic initialization (closest anchors).
        network.Quantizer.Forward(hidden, training: true, new Random(0));
        Close(f["usage"]!, new Tensor<double>(new[] { 12 }, new Vector<double>(network.Quantizer.Usage)), "usage");
        Close(f["updated"]!, network.Quantizer.Lookup(Enumerable.Range(0, 12).ToArray()), "updated codebook");
    }

    /// <summary>The cached decoding the synthesis uses gives the training forward's outputs, prefilled several tokens at
    /// once or one at a time.</summary>
    [Fact(Timeout = 300000)]
    public async Task CodeDecoder_CachedSteps_MatchFullForward()
    {
        await Task.Yield();
        var model = Create(TinyOptions());
        model.SetTrainingMode(false);
        using var _ = new NoGradScope<double>();
        foreach (var decoder in new[] { model.Network.ProsodyModel, model.Network.DurationModel })
        {
            var tokens = new[] { decoder.Vocabulary - 1, 3, 1, 0, 2, decoder.Vocabulary - 1, 4 }.Select(t => Math.Min(t, decoder.Vocabulary - 1)).ToArray();
            int n = tokens.Length;
            var features = new Tensor<double>(new[] { n, 16 });
            var random = new Random(7);
            for (int i = 0; i < features.Length; i++) features[i] = random.NextDouble() - 0.5;
            var full = decoder.Forward(tokens, features, Enumerable.Range(0, n).ToArray(), null).ToArray();
            var cache = new MegaTts3DurationCache<double>();
            var head = decoder.Extend(tokens.Take(4).ToArray(), AiDotNet.Tensors.Engines.AiDotNetEngine.Current.TensorSlice(features, new[] { 0, 0 }, new[] { 4, 16 }), cache).ToArray();
            var steps = new List<double>(head);
            for (int i = 4; i < n; i++)
                steps.AddRange(decoder.Extend(new[] { tokens[i] }, AiDotNet.Tensors.Engines.AiDotNetEngine.Current.TensorSlice(features, new[] { i, 0 }, new[] { 1, 16 }), cache).ToArray());
            Assert.Equal(full.Length, steps.Count);
            for (int i = 0; i < full.Length; i++) Assert.True(Math.Abs(full[i] - steps[i]) <= 1e-10, $"[{i}] {full[i]} vs {steps[i]}");
        }
    }

    private static TtsVoice<double> Prompt(MegaTTS2<double> model, MegaTTS2Options options, string text, double hz)
    {
        var tokens = model.TextToTokens(text);
        int count = tokens.Length;
        var durations = Enumerable.Range(0, count).Select(i => 1 + i % 3).ToArray();
        int samples = (durations.Sum() - 1) * options.HopSize;
        var audio = new Tensor<double>(new[] { samples });
        for (int i = 0; i < samples; i++) audio[i] = 0.3 * Math.Sin(2 * Math.PI * hz * i / options.SampleRate);
        return new TtsVoice<double> { ReferenceAudio = audio, ReferenceTokens = tokens, ReferenceDurations = durations };
    }

    /// <summary>Synthesis from two prompt sentences: the same seed gives the same audio, the audio is the vocoder's frames
    /// of the mel spectrogram Predict returns, and prosody prompts mixed in with γ = 0 change nothing.</summary>
    [Fact(Timeout = 300000)]
    public async Task Synthesis_WithPrompts_IsDeterministicFramedAndInterpolates()
    {
        await Task.Yield();
        var options = TinyOptions();
        var model = Create(options);
        model.PromptSentences = new[] { Prompt(model, options, "Hello world.", 200), Prompt(model, options, "Good day.", 260) };
        var first = model.Synthesize("Good morning.");
        var second = model.Synthesize("Good morning.");
        Assert.Equal(first.ToArray(), second.ToArray());
        Assert.True(first.Length > 0);
        Assert.Equal(0, first.Length % options.HopSize);
        var mel = model.Predict(model.TextToTokens("Good morning."));
        Assert.Equal(first.Length, mel.Shape[0] * options.HopSize);

        options.ProsodyInterpolation = 0;
        model.ProsodyPrompts = new[] { Prompt(model, options, "Yes.", 330) };
        Assert.Equal(mel.ToArray(), model.Predict(model.TextToTokens("Good morning.")).ToArray());
        options.ProsodyInterpolation = 1;
        var other = model.Predict(model.TextToTokens("Good morning."));
        Assert.Equal(mel.Shape.ToArray(), other.Shape.ToArray());
    }
    /// <summary>The second stage trains the P-LLM and the duration model on two speakers' sentences (the autoencoder
    /// frozen): the summed loss falls, and the autoencoder's weights stay where they were.</summary>
    [Fact(Timeout = 300000)]
    public async Task LanguageModels_TrainOnSpeakers_LossFallsAndAutoencoderStays()
    {
        await Task.Yield();
        var options = TinyOptions();
        options.Stage = MegaTts2Stage.LanguageModels;
        options.WarmupSteps = 1;
        options.ScheduleFactor = 0.004;
        var model = Create(options);
        var random = new Random(11);
        TtsTrainingSample<double> Sentence(string text)
        {
            var tokens = model.TextToTokens(text);
            var durations = Enumerable.Range(0, tokens.Length).Select(_ => 1 + random.Next(3)).ToArray();
            var mel = new Tensor<double>(new[] { durations.Sum(), 10 });
            for (int i = 0; i < mel.Length; i++) mel[i] = random.NextDouble() - 2;
            return new TtsTrainingSample<double> { Tokens = tokens, Durations = durations, Mel = mel };
        }
        var speakers = new[]
        {
            new[] { Sentence("Hello world."), Sentence("Good day.") },
            new[] { Sentence("Yes, of course.") },
        };
        var content = model.Network.Content.Embedding.GetParameters().ToArray();
        double first = model.TrainLanguageModels(speakers), last = first;
        for (int step = 0; step < 15; step++) last = model.TrainLanguageModels(speakers);
        Assert.True(last < first, $"The language models' loss went from {first} to {last}.");
        Assert.Equal(content, model.Network.Content.Embedding.GetParameters().ToArray());
        double objective = model.EvaluateTrainingObjective(speakers[0][0]);
        Assert.False(double.IsNaN(objective) || double.IsInfinity(objective));
    }
}
