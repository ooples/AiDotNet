using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// Llasa against its release (tools/reference-data/llasa_reference.py): Llasa-1B's tokenizer, the model card's chat
/// prompts and reading of the generated codes, the recipe's training rows (ids, labels and attention mask), and a tiny
/// model's synthesis and training step.
/// </summary>
public class LlasaModelTests
{
    private static JsonElement Load(string name)
    {
        string relative = Path.Combine("TextToSpeech", "ReferenceData", name);
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
        string path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    }

    private static int[] Ints(JsonElement e) => e.EnumerateArray().Select(v => v.GetInt32()).ToArray();

    [Fact(Timeout = 120000)]
    public async Task Tokenizer_MatchesLlasa()
    {
        await Task.Yield();
        var reference = Load("llasa_reference.json");
        Assert.Equal(reference.GetProperty("base").GetInt32(), LlasaTokens.FirstSpeechToken);
        foreach (var c in reference.GetProperty("encodings").EnumerateArray())
            Assert.Equal(Ints(c.GetProperty("ids")), LlasaTokens.Tokenizer.Encode(c.GetProperty("text").GetString() ?? ""));
    }

    [Fact(Timeout = 120000)]
    public async Task Prompts_MatchTheModelCard()
    {
        await Task.Yield();
        var reference = Load("llasa_reference.json");
        string date = reference.GetProperty("date").GetString()!;
        foreach (var c in reference.GetProperty("prompts").EnumerateArray())
            Assert.Equal(Ints(c.GetProperty("ids")),
                LlasaTokens.Prompt(LlasaTokens.Tokenizer, c.GetProperty("text").GetString() ?? "", date, Ints(c.GetProperty("prefix"))));
    }

    [Fact(Timeout = 120000)]
    public async Task TrainingRows_MatchTheRecipe()
    {
        await Task.Yield();
        var reference = Load("llasa_reference.json");
        string date = reference.GetProperty("date").GetString()!;
        foreach (var c in reference.GetProperty("training").EnumerateArray())
        {
            var row = LlasaTokens.TrainingRow(LlasaTokens.Tokenizer, c.GetProperty("text").GetString() ?? "", date, Ints(c.GetProperty("codes")), 2048);
            Assert.Equal(Ints(c.GetProperty("ids")), row.Ids);
            Assert.Equal(Ints(c.GetProperty("labels")), row.Labels);
            Assert.Equal(Ints(c.GetProperty("mask")).Select(v => v == 1).ToArray(), row.Mask);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task ReadCodes_MatchTheModelCard()
    {
        await Task.Yield();
        var reference = Load("llasa_reference.json");
        Assert.Equal(Ints(reference.GetProperty("codes")), LlasaTokens.ReadCodes(Ints(reference.GetProperty("stream"))));
    }

    [Fact(Timeout = 120000)]
    public async Task TrainingRow_CropsToTheMaximumLength()
    {
        await Task.Yield();
        var row = LlasaTokens.TrainingRow(LlasaTokens.Tokenizer, "Hello.", "26 Jul 2024", Enumerable.Range(0, 100).ToArray(), 64);
        Assert.Equal(64, row.Ids.Length);
        Assert.InRange(row.Ids[63], LlasaTokens.FirstSpeechToken, LlasaTokens.FirstSpeechToken + LlasaTokens.CodebookSize - 1);
        Assert.DoesNotContain(LlasaTokens.SpeechGenerationEnd, row.Ids);
    }

    // ---------------------------------------------------------------- a tiny model

    private static LlasaOptions TinyOptions() => new()
    {
        HiddenDim = 16, Layers = 1, Heads = 2, KeyValueHeads = 1, HeadDim = 8, Intermediate = 32, MaxLength = 80, ChatDate = "26 Jul 2024",
        ConstrainToSpeechTokens = true, CodecEncoderDim = 2, CodecAcousticDim = 32, CodecSemanticDim = 32, CodecDecoderDim = 32,
        CodecDecoderDepth = 1, CodecDecoderHeads = 4, SemanticHiddenSize = 32, SemanticHeads = 4, SemanticIntermediateSize = 64,
        SemanticLayer = 2, LearningRate = 1e-3,
    };

    private static Llasa<double> CreateModel(LlasaOptions? options = null) => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1)
        { RandomSeed = 3 },
        options ?? TinyOptions());

    private static Tensor<double> Recording(int n) =>
        new(new[] { n }, new Vector<double>(Enumerable.Range(0, n).Select(i => 0.3 * Math.Sin(0.02 * i) + 0.1 * Math.Sin(0.11 * i)).ToArray()));

    [Fact(Timeout = 300000)]
    public async Task Synthesis_DecodesTheGeneratedCodes()
    {
        await Task.Yield();
        var model = CreateModel();
        var prompt = LlasaTokens.Prompt(model.Tokenizer, "Hello there.", "26 Jul 2024", Array.Empty<int>());
        var generated = model.Generate(prompt);
        Assert.Equal(80 - prompt.Length, generated.Count);
        int codes = LlasaTokens.ReadCodes(generated).Count;
        Assert.Equal(generated.Count, codes);
        var audio = model.Synthesize("Hello there.");
        Assert.Equal(codes * 320, audio.Length);
        Assert.Equal(audio.ToArray(), model.Synthesize("Hello there.").ToArray());
    }

    [Fact(Timeout = 300000)]
    public async Task PromptedSynthesis_KeepsOrCutsThePrompt()
    {
        await Task.Yield();
        var model = CreateModel();
        var prompt = Recording(1000);
        int promptCodes = model.EncodeToTokens(prompt).Shape[0];
        Assert.Equal(4, promptCodes);                             // 1000 samples padded to 1280
        var full = model.SynthesizeWithPrompt("General Kenobi.", prompt, "Hello there.");
        Assert.True(full.Length > promptCodes * 320, "the model should continue after the prompt's codes");
        Assert.Equal(0, full.Length % 320);

        var options = TinyOptions();
        options.IncludePromptAudio = false;
        var cut = CreateModel(options).SynthesizeWithPrompt("General Kenobi.", prompt, "Hello there.");
        Assert.Equal(full.Length - 1000, cut.Length);
        Assert.Equal(full.ToArray().Skip(1000).ToArray(), cut.ToArray());
    }

    [Fact(Timeout = 300000)]
    public async Task TrainingStep_UpdatesTheLanguageModelOnly()
    {
        await Task.Yield();
        var model = CreateModel();
        var audio = Recording(3200);
        var tokens = model.TextToTokens("Hello there.");
        var sample = new TtsTrainingSample<double> { Tokens = tokens, Audio = audio };
        var fromCodes = new TtsTrainingSample<double> { Tokens = tokens, CodecTokens = model.EncodeToTokens(audio) };
        Assert.Equal(model.TrainingRow(sample).Ids, model.TrainingRow(fromCodes).Ids);

        double before = model.EvaluateTrainingObjective(sample);
        var codecBefore = model.CodecLayers.SelectMany(l => l.GetParameters().ToArray()).ToArray();
        var lmBefore = model.LanguageModelLayers.SelectMany(l => l.GetParameters().ToArray()).ToArray();
        double loss = model.Train(sample);
        Assert.Equal(before, loss, 9);
        var codecAfter = model.CodecLayers.SelectMany(l => l.GetParameters().ToArray()).ToArray();
        var lmAfter = model.LanguageModelLayers.SelectMany(l => l.GetParameters().ToArray()).ToArray();
        Assert.Equal(codecBefore, codecAfter);
        Assert.NotEqual(lmBefore, lmAfter);
        Assert.True(model.EvaluateTrainingObjective(sample) < before, "one step should lower the objective on its own sample");
    }
}
