using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Audio.MusicGen;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// MusicGen's training and sampling rules from the paper (Copet et al. 2023, §2–3) and audiocraft where the paper is
/// silent: the delay pattern, condition merging, description and word dropout, top-k sampling with ties, the
/// optimizer per size, the EMA, continuation and the capabilities it refuses.
/// </summary>
public class MusicGenPaperTests
{
    private static MusicGenModel<double> Create(Action<MusicGenOptions>? configure = null)
    {
        var options = MusicGenReferenceTests.TinyOptions();
        options.UseSampling = true;
        options.TextVocabularySize = 32128;                       // the real tokenizer's ids
        configure?.Invoke(options);
        return new MusicGenModel<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Generative, inputSize: 1, outputSize: 1) { RandomSeed = 5 }, options);
    }

    private static Tensor<double> Codes(int frames, int offset = 0)
    {
        var codes = new Tensor<double>(new[] { frames, 4 });
        for (int f = 0; f < frames; f++)
            for (int q = 0; q < 4; q++) codes[f, q] = (3 * f + 5 * q + offset) % 16;
        return codes;
    }

    [Fact(Timeout = 30000)]
    public async Task DelayPattern_LagsEachCodebookByItsDelay()
    {
        await Task.Yield();
        var codes = new int[4, 3];
        for (int q = 0; q < 4; q++)
            for (int t = 0; t < 3; t++) codes[q, t] = 10 * q + t;
        var sequence = MusicGenDelayPattern.TrainingSequence(codes, new[] { 0, 1, 2, 3 }, special: 99);
        // Step 0 is empty; step s holds code (s − 1 − d_k, k) (Eq. 6).
        Assert.Equal(new[] { 99, 0, 1, 2 }, Enumerable.Range(0, 4).Select(s => sequence[0, s]));
        Assert.Equal(new[] { 99, 99, 10, 11 }, Enumerable.Range(0, 4).Select(s => sequence[1, s]));
        Assert.Equal(new[] { 99, 99, 99, 20 }, Enumerable.Range(0, 4).Select(s => sequence[2, s]));
        Assert.Equal(new[] { 99, 99, 99, 99 }, Enumerable.Range(0, 4).Select(s => sequence[3, s]));
    }

    [Fact(Timeout = 30000)]
    public async Task ConditionMerging_FollowsTheReferenceFormat()
    {
        await Task.Yield();
        using var always = Create(o =>
        {
            o.MergeMetadataProbability = 1;
            o.KeepMetadataFieldProbability = 1;
            o.DropDescriptionProbability = 0;
        });
        var sample = new MusicGenTrainingSample<double>
        {
            Codes = Codes(4), Description = "A calm piano piece.",
            Metadata = new Dictionary<string, string> { ["genre"] = "classical", ["bpm"] = "70" },
        };
        string merged = always.AugmentDescriptionForTests(sample, new Random(1)) ?? "";
        // "description. name: value. name: value" with the fields shuffled and the description's period trimmed.
        Assert.StartsWith("A calm piano piece. ", merged);
        Assert.Contains("genre: classical", merged);
        Assert.Contains("bpm: 70", merged);

        using var dropping = Create(o =>
        {
            o.MergeMetadataProbability = 1;
            o.KeepMetadataFieldProbability = 1;
            o.DropDescriptionProbability = 1;
        });
        string metadataOnly = dropping.AugmentDescriptionForTests(sample, new Random(1)) ?? "";
        Assert.DoesNotContain("piano", metadataOnly);
        Assert.Contains("genre: classical", metadataOnly);

        using var never = Create(o => o.MergeMetadataProbability = 0);
        Assert.Equal("A calm piano piece.", never.AugmentDescriptionForTests(sample, new Random(1)));
    }

    [Fact(Timeout = 30000)]
    public async Task WordDropout_DropsWordsAtTheConfiguredRate()
    {
        await Task.Yield();
        using var model = Create();
        var text = string.Join(" ", Enumerable.Range(0, 2000).Select(i => "w" + i));
        var kept = model.DropWordsForTests(text, new Random(2))!.Split(' ').Length;
        Assert.InRange(kept / 2000.0, 0.65, 0.75);                    // keep 1 − 0.3
        using var none = Create(o => o.WordDropout = 1.0);
        Assert.Null(none.DropWordsForTests("all words go", new Random(2)));
    }

    [Fact(Timeout = 30000)]
    public async Task TopK_KeepsTiesAtTheKthProbability_AndNeverDrawsBelowIt()
    {
        await Task.Yield();
        using var model = Create(o => o.TopK = 2);
        // Logits 3, 1, 1, 0: the 2nd largest probability is shared by codes 1 and 2, so both survive; code 3 never.
        var logits = new[] { 3.0, 1.0, 1.0, 0.0 };
        var random = new Random(4);
        var drawn = Enumerable.Range(0, 2000).Select(_ => model.ChooseForTests(logits, random)).ToArray();
        Assert.DoesNotContain(3, drawn);
        Assert.Contains(1, drawn);
        Assert.Contains(2, drawn);
        using var greedy = Create(o => o.UseSampling = false);
        Assert.Equal(0, greedy.ChooseForTests(logits, random));
    }

    [Fact(Timeout = 30000)]
    public async Task Optimizer_IsDAdaptationForSmall_AndAdamWOtherwise()
    {
        await Task.Yield();
        using var small = Create();
        Assert.IsType<DAdaptAdamOptimizer<double, Tensor<double>, Tensor<double>>>(small.OptimizerForTests);
        using var medium = Create(o => o.ModelSize = MusicGenModelSize.Medium);
        Assert.IsType<AdamWOptimizer<double, Tensor<double>, Tensor<double>>>(medium.OptimizerForTests);
    }

    [Fact(Timeout = 300000)]
    public async Task Training_LowersTheObjective_AndTheEmaTakesOver()
    {
        await Task.Yield();
        using var model = Create(o =>
        {
            o.ModelSize = MusicGenModelSize.Medium;
            o.LearningRate = 3e-3;
            o.WarmupSteps = 0;
            o.ConditionDropout = 0;
            o.EmaEveryUpdates = 5;
        });
        var tokens = model.TextToTokens("an upbeat synth track");
        var codes = Codes(10);
        double before = model.EvaluateTrainingObjective(tokens, codes);
        for (int i = 0; i < 30; i++) model.Train(tokens, codes);
        double after = model.EvaluateTrainingObjective(tokens, codes);
        Assert.True(after < 0.8 * before, $"The objective did not fall ({before} -> {after}).");
        Assert.True(model.HasEmaForTests, "Thirty updates every five should have produced EMA weights.");
    }

    [Fact(Timeout = 300000)]
    public async Task Generation_IsSeeded_DecodesFramesTimesHop_AndContinuationKeepsThePrompt()
    {
        await Task.Yield();
        using var model = Create(o => o.DurationSeconds = 0.1);
        var first = model.GenerateMusic("lofi beat", durationSeconds: 0.1, seed: 9);
        var again = model.GenerateMusic("lofi beat", durationSeconds: 0.1, seed: 9);
        Assert.Equal(first.ToArray(), again.ToArray());
        Assert.Equal(5 * 640, first.Length);                          // 0.1 s at 50 Hz frames of 640 samples
        var prompt = new int[4, 3];
        for (int q = 0; q < 4; q++)
            for (int t = 0; t < 3; t++) prompt[q, t] = (q + t) % 16;
        var continued = model.GenerateCodes(model.TextToTokens("lofi beat"), 6, seed: 1, prompt: prompt);
        for (int q = 0; q < 4; q++)
            for (int t = 0; t < 3; t++) Assert.Equal(prompt[q, t], continued[q, t]);
    }

    [Fact(Timeout = 30000)]
    public async Task Refuses_NegativePromptsAndInpainting()
    {
        await Task.Yield();
        using var model = Create();
        Assert.Throws<NotSupportedException>(() => model.GenerateMusic("a", negativePrompt: "b", durationSeconds: 0.02));
        Assert.Throws<NotSupportedException>(() => model.InpaintAudio(new Tensor<double>(new[] { 640 }), new Tensor<double>(new[] { 640 })));
    }

    private static Tensor<double> Tone(double hz, int samples = 16000)
        => new(new[] { samples }, new Vector<double>(Enumerable.Range(0, samples).Select(i => 0.5 * Math.Sin(2 * Math.PI * hz * i / 32000)).ToArray()));

    [Fact(Timeout = 300000)]
    public async Task MelodyModel_TrainsOnItsOwnChroma_AndTheMelodySteersGeneration()
    {
        await Task.Yield();
        using var model = Create(o =>
        {
            o.MelodyConditioning = true;
            o.ModelSize = MusicGenModelSize.Medium;
            o.LearningRate = 3e-3;
            o.WarmupSteps = 0;
            o.ConditionDropout = 0;
            o.MelodyDropout = 0;
            o.UseSampling = false;
        });
        var clip = Tone(261.63, 3200);
        var tokens = model.TextToTokens("a piano melody");
        double before = model.EvaluateTrainingObjective(tokens, clip);
        for (int i = 0; i < 25; i++) model.Train(new MusicGenTrainingSample<double> { Audio = clip, Description = "a piano melody" });
        double after = model.EvaluateTrainingObjective(tokens, clip);
        Assert.True(after < before, $"The objective did not fall ({before} -> {after}).");
        // The melody is part of the condition: a C and an A melody give different prefixes.
        var c = model.MelodyConditionForTests(tokens, Tone(261.63));
        var a = model.MelodyConditionForTests(tokens, Tone(440.0));
        Assert.Equal(c.Shape.ToArray(), a.Shape.ToArray());
        Assert.NotEqual(c.ToArray(), a.ToArray());
        Assert.Equal(5 * 640, model.GenerateWithMelody("a piano melody", Tone(440.0), durationSeconds: 0.1).Length);
    }

    [Fact(Timeout = 30000)]
    public async Task TextModel_RefusesAMelody()
    {
        await Task.Yield();
        using var model = Create();
        Assert.Throws<InvalidOperationException>(() => model.GenerateWithMelody("a", Tone(440.0), durationSeconds: 0.02));
    }
}
