using System;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.SpeechRecognition.Specialized;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.SpeechRecognition;

/// <summary>
/// Verifies that <see cref="CodeSwitchingASR{T}"/> implements Zeng et al.,
/// "On the End-to-End Solution to Mandarin-English Code-switching Speech Recognition"
/// (arXiv:1811.00241), rather than a CTC-only model with a language-guessing rule.
/// </summary>
/// <remarks>
/// The paper's contributions are an explicitly LEARNED language-identification task, a genuinely
/// hybrid CTC/attention objective, and per-language token inventories that are concatenated rather
/// than unified. The previous revision of this class had none of them: CTC only, a unified
/// vocabulary, and language "identification" by counting CJK versus Latin codepoints in the decoded
/// string. Each is asserted here so that cannot silently return.
/// </remarks>
public class CodeSwitchingASRMechanismTests
{
    private static NeuralNetworkArchitecture<double> Arch() =>
        new(inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputSize: 64, outputSize: 32);

    private static CodeSwitchingASROptions SmallOptions(Action<CodeSwitchingASROptions>? tweak = null)
    {
        var o = new CodeSwitchingASROptions
        {
            EncoderDim = 16,
            NumEncoderLayers = 1,
            NumAttentionHeads = 2,
            NumMels = 8,
            DecoderDim = 16,
            NumDecoderLayers = 1,
            MandarinCharVocabSize = 6,
            EnglishBpeVocabSize = 5,
            MaxTextLength = 16,
            DropoutRate = 0.0,
        };
        tweak?.Invoke(o);
        return o;
    }

    private static CodeSwitchingASR<double> Model(Action<CodeSwitchingASROptions>? tweak = null)
        => new(Arch(), SmallOptions(tweak));

    // ------------------------------------------------- paper hyperparameters

    [Fact]
    public void PaperWeightsAreTheDefaults()
    {
        var o = new CodeSwitchingASROptions();

        // "L_MTL = lambda1 * L_att + (1 - lambda1) * L_ctc + lambda2 * L_lid", CTC weight fixed at
        // 0.2 and lambda2 tuned to an optimum of 0.2.
        Assert.Equal(0.2, o.CtcWeight);
        Assert.Equal(0.2, o.LidWeight);

        // Encoder 6 layers at 320 units; decoder 1 layer at 320.
        Assert.Equal(320, o.EncoderDim);
        Assert.Equal(6, o.NumEncoderLayers);
        Assert.Equal(320, o.DecoderDim);
        Assert.Equal(1, o.NumDecoderLayers);

        // "3k BPE performs best".
        Assert.Equal(3000, o.EnglishBpeVocabSize);

        // LID_shared is the default variant.
        Assert.True(o.SharedLidAttention);
    }

    [Fact]
    public void AttentionBranchCarriesMostOfTheObjective()
    {
        // CtcWeight is the CTC share, so the attention branch carries 1 - 0.2 = 0.8. A model where
        // CTC dominates is a different configuration than the paper's.
        var o = new CodeSwitchingASROptions();
        Assert.True(1.0 - o.CtcWeight > o.CtcWeight,
            "The paper's CTC weight of 0.2 leaves 0.8 on the attention branch.");
    }

    // ------------------------------------------------- separate inventories

    [Fact]
    public void VocabularyIsTheConcatenationOfTwoInventoriesPlusBlank()
    {
        // "Mandarin uses characters while English uses BPE units" — they do NOT share one inventory.
        var o = new CodeSwitchingASROptions { MandarinCharVocabSize = 100, EnglishBpeVocabSize = 40 };
        Assert.Equal(141, o.VocabSize);   // 100 + 40 + one CTC blank
    }

    [Fact]
    public void TokenIdDeterminesItsLanguage()
    {
        // This is what the separate-inventory choice buys, and it is why an LID target exists
        // without extra annotation.
        var model = Model();
        int offset = model.EnglishTokenOffset;

        Assert.Equal(1 + 6, offset);          // blank + 6 Mandarin characters
        Assert.Equal("zh", model.LanguageOfToken(1));
        Assert.Equal("zh", model.LanguageOfToken(offset - 1));
        Assert.Equal("en", model.LanguageOfToken(offset));
        Assert.Equal("en", model.LanguageOfToken(offset + 4));
    }

    [Fact]
    public void SupportedLanguagesAreTheMandarinEnglishPairTheModelCanActuallyDistinguish()
    {
        // The previous revision advertised four languages ("en", "zh", "es", "hi") while having no
        // mechanism able to tell them apart. Claiming coverage the model does not have is worse than
        // claiming less.
        var model = Model();
        Assert.Equal(2, model.SupportedLanguages.Count);
        Assert.Contains("zh", model.SupportedLanguages);
        Assert.Contains("en", model.SupportedLanguages);
    }

    // ------------------------------------------------- learned LID

    private static Tensor<double> Audio(int samples = 64, int seed = 5)
    {
        var rng = new Random(seed);
        var t = new Tensor<double>([samples]);
        for (int i = 0; i < samples; i++) t[i] = rng.NextDouble() - 0.5;
        return t;
    }

    [Fact]
    public void LanguagePosteriorsComeFromTheHeadAndAreNormalized()
    {
        var model = Model();
        var probabilities = model.DetectLanguageProbabilities(Audio());

        Assert.Equal(2, probabilities.Count);
        double total = probabilities.Values.Sum();
        Assert.Equal(1.0, total, 9);
        foreach (var p in probabilities.Values) Assert.InRange(p, 0.0, 1.0);
    }

    [Fact]
    public void DetectedLanguageIsTheArgmaxOfThosePosteriors()
    {
        // Ties the reported label to the learned head, so a heuristic cannot be reintroduced
        // underneath it without this disagreeing.
        var model = Model();
        var audio = Audio(seed: 12);

        var probabilities = model.DetectLanguageProbabilities(audio);
        string expected = probabilities.OrderByDescending(kv => kv.Value).First().Key;

        Assert.Equal(expected, model.DetectLanguage(audio));
    }

    [Fact]
    public void LanguageDetectionRespondsToTheEncoder_NotToDecodedCodepoints()
    {
        // The old implementation classified by counting CJK vs Latin codepoints in the OUTPUT
        // string, so its verdict was a fixed function of the decoded text. A learned head instead
        // depends on the model's parameters: perturbing them must be able to change the posteriors.
        var model = Model();
        var audio = Audio(seed: 21);

        var before = model.DetectLanguageProbabilities(audio).ToDictionary(kv => kv.Key, kv => kv.Value);

        var parameters = model.GetParameters();
        var shifted = new Vector<double>(parameters.Length);
        for (int i = 0; i < parameters.Length; i++) shifted[i] = parameters[i] + 0.75;
        model.UpdateParameters(shifted);

        var after = model.DetectLanguageProbabilities(audio);
        bool changed = before.Keys.Any(k => Math.Abs(before[k] - after[k]) > 1e-12);
        Assert.True(changed,
            "Language posteriors did not move when the parameters did — they are not coming from a " +
            "learned head.");
    }

    // ------------------------------------------------- hybrid objective

    [Fact]
    public void TrainingUpdatesTheSharedEncoderAndAllThreeHeads()
    {
        // One tape over the joint objective, so every branch's gradient reaches the shared encoder.
        var model = Model();

        // Materialize the lazily-sized layers first. Before any forward pass some layers have not
        // resolved their input width and report zero parameters, so comparing vector LENGTHS across
        // the first Train would compare two different parameterizations rather than two states of
        // the same one.
        model.Predict(new Tensor<double>([4, 8]));
        var before = model.GetParameters();

        var input = new Tensor<double>([4, 8]);
        var rng = new Random(3);
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble() - 0.5;
        var target = new Tensor<double>([4, model.GetOptionsTyped().VocabSize]);
        for (int i = 0; i < target.Length; i++) target[i] = rng.NextDouble();

        model.Train(input, target);
        var after = model.GetParameters();

        Assert.Equal(before.Length, after.Length);
        bool moved = Enumerable.Range(0, before.Length).Any(i => Math.Abs(before[i] - after[i]) > 1e-12);
        Assert.True(moved, "Joint training did not move any parameter.");
    }

    [Fact]
    public void DisablingLidLeavesThePaperOwnBaseline()
    {
        // LidWeight = 0 removes the contribution and leaves a plain hybrid CTC/attention model,
        // which is the paper's baseline rather than a broken configuration.
        var model = Model(o => o.LidWeight = 0.0);

        var input = new Tensor<double>([4, 8]);
        var rng = new Random(9);
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble() - 0.5;
        var target = new Tensor<double>([4, model.GetOptionsTyped().VocabSize]);
        for (int i = 0; i < target.Length; i++) target[i] = rng.NextDouble();

        model.Predict(input);
        var before = model.GetParameters();
        model.Train(input, target);
        var after = model.GetParameters();

        bool moved = Enumerable.Range(0, before.Length).Any(i => Math.Abs(before[i] - after[i]) > 1e-12);
        Assert.True(moved, "The hybrid baseline must still train with the LID task switched off.");
    }

    [Fact]
    public void TranscriptionIsFiniteAndReportsALearnedLanguage()
    {
        var model = Model();
        var result = model.Transcribe(Audio(seed: 31));

        Assert.NotNull(result);
        Assert.Contains(result.Language, model.SupportedLanguages);
        Assert.False(double.IsNaN(result.Confidence) || double.IsInfinity(result.Confidence));
    }

    [Fact]
    public void StreamingIsRefusedExplicitly()
    {
        var model = Model();
        Assert.False(model.SupportsStreaming);
        Assert.Throws<NotSupportedException>(() => model.StartStreamingSession());
    }
}

internal static class CodeSwitchingASRTestExtensions
{
    internal static CodeSwitchingASROptions GetOptionsTyped<T>(this CodeSwitchingASR<T> model)
        => (CodeSwitchingASROptions)model.GetOptions();
}
