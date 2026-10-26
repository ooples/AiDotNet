using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// Orpheus against its release: Llama 3 / Orpheus tokenization (tools/reference-data/llama3_tokenizer_resources.py), the
/// notebook's training sequences, the package's prompts and streaming windows (tools/reference-data/orpheus_reference.py),
/// and a tiny model's synthesis and training step.
/// </summary>
public class OrpheusTTSModelTests
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

    private static string? Voice(JsonElement e) => e.GetProperty("voice").ValueKind == JsonValueKind.Null ? null : e.GetProperty("voice").GetString();

    [Fact(Timeout = 120000)]
    public async Task Tokenizer_MatchesOrpheus()
    {
        await Task.Yield();
        foreach (var c in Load("llama3_tokenizer_reference.json").GetProperty("cases").EnumerateArray())
        {
            var text = c.GetProperty("text").GetString() ?? "";
            var actual = new[] { 128000 }.Concat(OrpheusTokens.Tokenizer.Encode(text)).ToArray();
            Assert.Equal(Ints(c.GetProperty("ids")), actual);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task TrainingSequences_MatchTheNotebook()
    {
        await Task.Yield();
        foreach (var c in Load("orpheus_reference.json").GetProperty("training").EnumerateArray())
        {
            var codes = c.GetProperty("codes").EnumerateArray().Select(Ints).ToArray();
            var frames = OrpheusTokens.RemoveDuplicateFrames(OrpheusTokens.Frames(codes));
            Assert.True(frames.Count < codes[0].Length, "the case should drop a repeated frame");
            var ids = OrpheusTokens.TrainingSequence(OrpheusTokens.Tokenizer, c.GetProperty("text").GetString() ?? "", Voice(c), frames);
            Assert.Equal(Ints(c.GetProperty("ids")), ids);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Prompts_MatchThePackage()
    {
        await Task.Yield();
        foreach (var c in Load("orpheus_reference.json").GetProperty("prompts").EnumerateArray())
            Assert.Equal(Ints(c.GetProperty("ids")), OrpheusTokens.Prompt(OrpheusTokens.Tokenizer, c.GetProperty("text").GetString() ?? "", Voice(c)));
    }

    [Fact(Timeout = 120000)]
    public async Task StreamingWindows_MatchThePackage()
    {
        await Task.Yield();
        var reference = Load("orpheus_reference.json");
        var codes = OrpheusTokens.ReadCodes(Ints(reference.GetProperty("stream")));
        var windows = new List<int[][]>();
        int skipped = 0;
        for (int count = 28; count <= codes.Count; count += 7)
        {
            var window = OrpheusTokens.WholeFrames(codes.GetRange(count - 28, 28));
            if (window.All(OrpheusTokens.IsValid)) windows.Add(OrpheusTokens.Codes(window));
            else skipped++;
        }
        Assert.True(skipped > 0, "the stream should spoil some windows");
        var expected = reference.GetProperty("windows").EnumerateArray().Select(w => w.EnumerateArray().Select(Ints).ToArray()).ToList();
        Assert.Equal(expected.Count, windows.Count);
        for (int w = 0; w < expected.Count; w++)
            for (int q = 0; q < 3; q++) Assert.Equal(expected[w][q], windows[w][q]);
    }

    [Fact(Timeout = 120000)]
    public async Task ReadCodes_KeepsCodeZero_AndSkipsOrdinaryTokens()
    {
        await Task.Yield();
        // The package keeps codes above 0; a code of 0 is a real code.
        var codes = OrpheusTokens.ReadCodes(new[] { 128266, 1234, 128266 + 4096 + 7, OrpheusTokens.EndOfAi, 128266 + 3 });
        // The marker and the slot-2 token of slot 0's code come out negative and are not read.
        Assert.Equal(new[] { 0, 7 }, codes);
    }

    // ---------------------------------------------------------------- a tiny model

    private static OrpheusTTSOptions TinyOptions() => new()
    {
        HiddenDim = 16, Layers = 1, Heads = 2, KeyValueHeads = 1, HeadDim = 8, Intermediate = 32, MaxNewTokens = 35,
        ConstrainToAudioTokens = true, CodecEncoderDim = 2, CodecDecoderDim = 32, CodecCodebookDim = 2, LearningRate = 1e-3,
    };

    private static OrpheusTTS<double> CreateModel(OrpheusTTSOptions? options = null) => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1)
        { RandomSeed = 3 },
        options ?? TinyOptions());

    private static Tensor<double> Recording(int n) =>
        new(new[] { n }, new Vector<double>(Enumerable.Range(0, n).Select(i => 0.3 * Math.Sin(0.02 * i) + 0.1 * Math.Sin(0.11 * i)).ToArray()));

    [Fact(Timeout = 300000)]
    public async Task Synthesis_DecodesTheGeneratedFrames()
    {
        await Task.Yield();
        var model = CreateModel();
        var prompt = OrpheusTokens.Prompt(model.Tokenizer, "Hello there.", "tara");
        var generated = model.Generate(prompt);
        int frames = OrpheusTokens.ReadCodes(generated).Count / 7;
        Assert.True(frames >= 4, $"the constrained model wrote {frames} frames");
        var audio = model.Synthesize("Hello there.");
        Assert.Equal(frames * 2048, audio.Length);
        Assert.All(audio.ToArray(), v => Assert.True(Math.Abs(v) <= 1));
        Assert.Equal(audio.ToArray(), model.Synthesize("Hello there.").ToArray());

        var streaming = TinyOptions();
        streaming.Decoding = OrpheusDecoding.Streaming;
        Assert.Equal((frames - 3) * 2048, CreateModel(streaming).Synthesize("Hello there.").Length);
    }

    [Fact(Timeout = 300000)]
    public async Task UnconstrainedSynthesis_DropsMalformedFrames()
    {
        await Task.Yield();
        var options = TinyOptions();
        options.ConstrainToAudioTokens = false;
        var model = CreateModel(options);
        var generated = model.Generate(OrpheusTokens.Prompt(model.Tokenizer, "Hello there.", "tara"));
        int valid = OrpheusTokens.WholeFrames(OrpheusTokens.ReadCodes(generated)).Count(OrpheusTokens.IsValid);
        Assert.Equal(valid * 2048, model.Synthesize("Hello there.").Length);
    }

    [Fact(Timeout = 300000)]
    public async Task TrainingStep_UpdatesTheLanguageModelOnly()
    {
        await Task.Yield();
        var model = CreateModel();
        var audio = Recording(6000);
        var tokens = model.TextToTokens("Hello there.");
        var sample = new TtsTrainingSample<double> { Tokens = tokens, Audio = audio };
        var fromCodes = new TtsTrainingSample<double> { Tokens = tokens, CodecTokens = model.EncodeToTokens(audio) };
        Assert.Equal(model.TrainingIds(sample), model.TrainingIds(fromCodes));

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
