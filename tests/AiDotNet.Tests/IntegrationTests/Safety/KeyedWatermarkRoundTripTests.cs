using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AiDotNet.Agentic.Models.Local;
using AiDotNet.Safety;
using AiDotNet.Safety.Watermarking;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Safety;

/// <summary>
/// Shared fixtures for the keyed-watermark round-trip families: a deterministic toy language model with a word-level
/// tokenizer, template prose with substitution sites, and speech-like audio.
/// </summary>
internal static class WatermarkFixtures
{
    public const int Vocabulary = 500;
    public const int GeneratedTokens = 200;

    private static readonly string[] Fillers =
    {
        "river", "stone", "garden", "window", "market", "signal", "harbor", "forest", "engine", "pencil",
        "valley", "silver", "rocket", "bridge", "castle", "meadow", "planet", "copper", "island", "lantern",
        "orchard", "canyon", "glacier", "harvest", "compass", "thunder", "marble", "velvet", "saddle", "quarry",
        "tunnel", "beacon", "cobalt", "falcon", "juniper", "lagoon", "mosaic", "nectar", "prairie", "ribbon"
    };

    /// <summary>
    /// A toy causal language model whose next-token logits are 2 x N(0, 1) values seeded by the last token: a
    /// distribution with enough entropy for the green-list bias to act on, and deterministic.
    /// </summary>
    public sealed class ToyLanguageModel : ICausalLanguageModel<double>
    {
        public int VocabularySize => Vocabulary;

        public Vector<double> NextTokenLogits(IReadOnlyList<int> tokenIds)
        {
            var rng = RandomHelper.CreateSeededRandom(1000 + tokenIds[tokenIds.Count - 1]);
            var logits = new Vector<double>(Vocabulary);
            for (int i = 0; i < Vocabulary; i++) logits[i] = 2.0 * Gaussian(rng);
            return logits;
        }
    }

    /// <summary>The incremental form of <see cref="ToyLanguageModel"/>, tracking its own context like a KV cache.</summary>
    public sealed class ToyIncrementalLanguageModel : IIncrementalCausalLanguageModel<double>
    {
        private readonly ToyLanguageModel _model = new();
        private readonly List<int> _context = new();

        public int VocabularySize => Vocabulary;

        public Vector<double> NextTokenLogits(IReadOnlyList<int> tokenIds) => _model.NextTokenLogits(tokenIds);

        public void ResetCache() => _context.Clear();

        public Vector<double> StartSequence(IReadOnlyList<int> promptTokenIds)
        {
            _context.Clear();
            _context.AddRange(promptTokenIds);
            return _model.NextTokenLogits(_context);
        }

        public Vector<double> AppendToken(int tokenId)
        {
            _context.Add(tokenId);
            return _model.NextTokenLogits(_context);
        }
    }

    /// <summary>A word-level tokenizer for the toy vocabulary: token id i is the word "w{i}".</summary>
    public sealed class WordTokenizer : IGenerationTokenizer
    {
        public int EosTokenId => Vocabulary - 1;

        public IReadOnlyList<int> Encode(string text) =>
            text.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries)
                .Select(w => int.Parse(w.Substring(1), CultureInfo.InvariantCulture)).ToList();

        public string Decode(IReadOnlyList<int> tokenIds) =>
            string.Join(" ", tokenIds.Select(t => "w" + t.ToString(CultureInfo.InvariantCulture)));
    }

    /// <summary>
    /// Samples <see cref="GeneratedTokens"/> tokens after the prompt token <paramref name="prompt"/> from
    /// <paramref name="model"/>'s softmax, with a sampler seeded by the prompt, and returns prompt plus generation.
    /// </summary>
    public static int[] Generate(ICausalLanguageModel<double> model, int prompt)
    {
        var rng = RandomHelper.CreateSeededRandom(50000 + prompt);
        var tokens = new List<int> { prompt };
        var incremental = model as IIncrementalCausalLanguageModel<double>;
        var logits = incremental is null ? model.NextTokenLogits(tokens) : incremental.StartSequence(tokens);
        for (int step = 0; step < GeneratedTokens; step++)
        {
            int next = SampleSoftmax(logits, rng);
            tokens.Add(next);
            if (step + 1 < GeneratedTokens)
                logits = incremental is null ? model.NextTokenLogits(tokens) : incremental.AppendToken(next);
        }

        return tokens.ToArray();
    }

    /// <summary>Replaces 10% of the generated tokens with random ids: an edit of one word in ten.</summary>
    public static int[] ReplaceTenPercent(int[] tokens, int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(7000 + seed);
        var result = (int[])tokens.Clone();
        foreach (int i in Enumerable.Range(1, tokens.Length - 1).OrderBy(_ => rng.Next()).Take(tokens.Length / 10))
            result[i] = rng.Next(Vocabulary);
        return result;
    }

    /// <summary>
    /// Prose of 80 sentences, each a filler word followed by a random member of a random substitution group and two
    /// more fillers, so every sentence holds one site with a varied preceding word.
    /// </summary>
    public static string SubstitutionProse(IReadOnlyList<IReadOnlyList<string>> groups, int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(900 + seed);
        var text = new StringBuilder();
        for (int s = 0; s < 80; s++)
        {
            var group = groups[rng.Next(groups.Count)];
            string lead = Fillers[rng.Next(Fillers.Length)];
            text.Append(char.ToUpperInvariant(lead[0])).Append(lead.Substring(1)).Append(' ')
                .Append(group[rng.Next(group.Count)]).Append(' ')
                .Append(Fillers[rng.Next(Fillers.Length)]).Append(' ')
                .Append(Fillers[rng.Next(Fillers.Length)]).Append(". ");
        }

        return text.ToString().TrimEnd();
    }

    /// <summary>Replaces the first word of one sentence in ten with a different filler, which re-keys the site after it.</summary>
    public static string EditTenPercentOfSentences(string text, int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(3000 + seed);
        var sentences = text.Split(new[] { ". " }, StringSplitOptions.None);
        foreach (int i in Enumerable.Range(0, sentences.Length).OrderBy(_ => rng.Next()).Take(sentences.Length / 10))
        {
            int space = sentences[i].IndexOf(' ');
            string replacement = Fillers[rng.Next(Fillers.Length)];
            sentences[i] = char.ToUpperInvariant(replacement[0]) + replacement.Substring(1) + sentences[i].Substring(space);
        }

        return string.Join(". ", sentences);
    }

    /// <summary>One second of 16 kHz speech-like audio: three harmonics of a seed-dependent pitch plus a little noise.</summary>
    public static Vector<double> Voiced(int seed, int sampleRate = 16000)
    {
        var rng = RandomHelper.CreateSeededRandom(2139 + seed);
        double f0 = 110 + 10 * (seed % 7);
        var samples = new Vector<double>(sampleRate);
        for (int i = 0; i < sampleRate; i++)
        {
            double t = (double)i / sampleRate;
            samples[i] = 0.4 * Math.Sin(2 * Math.PI * f0 * t) + 0.2 * Math.Sin(2 * Math.PI * 2 * f0 * t)
                + 0.1 * Math.Sin(2 * Math.PI * 3 * f0 * t) + 0.02 * (rng.NextDouble() * 2 - 1);
        }

        return samples;
    }

    /// <summary>Adds Gaussian noise 30 dB below the signal and re-quantizes to 16 bits.</summary>
    public static Vector<double> NoisyRequantized(Vector<double> audio, int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(4000 + seed);
        double power = 0;
        for (int i = 0; i < audio.Length; i++) power += audio[i] * audio[i];
        double sigma = Math.Sqrt(power / audio.Length) * Math.Pow(10, -30.0 / 20);
        var result = new Vector<double>(audio.Length);
        for (int i = 0; i < audio.Length; i++)
            result[i] = Math.Round((audio[i] + sigma * Gaussian(rng)) * 32767) / 32767;
        return result;
    }

    public static bool SameAudio(Vector<double> a, Vector<double> b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
            if (a[i] != b[i]) return false;
        return true;
    }

    private static double Gaussian(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble(), u2 = rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}

/// <summary>Green-list watermark (Kirchenbauer et al. 2023) on token ids, embedded by wrapping the model.</summary>
public class SamplingWatermarkerRoundTripTests : WatermarkRoundTripTestBase<int[]>
{
    protected override int[] Sample(int seed) => WatermarkFixtures.Generate(new WatermarkFixtures.ToyLanguageModel(), seed % WatermarkFixtures.Vocabulary);

    protected override int[] Embed(string key, int[] content) =>
        WatermarkFixtures.Generate(new SamplingWatermarker<double>(key: key).WatermarkModel(new WatermarkFixtures.ToyLanguageModel()), content[0]);

    protected override WatermarkResult Detect(string key, int[] content) => new SamplingWatermarker<double>(key: key).DetectTokens(content);

    protected override int[] Perturb(int[] content, int seed) => WatermarkFixtures.ReplaceTenPercent(content, seed);

    protected override bool SameContent(int[] a, int[] b) => a.SequenceEqual(b);

    [Fact(Timeout = 120000)]
    public async Task IncrementalModel_IsWatermarkedExactlyLikeTheFullForwardModel()
    {
        await Task.Yield();
        var watermarker = new SamplingWatermarker<double>(key: TestKey);
        var incremental = watermarker.WatermarkModel(new WatermarkFixtures.ToyIncrementalLanguageModel());
        Assert.IsAssignableFrom<IIncrementalCausalLanguageModel<double>>(incremental);
        var full = watermarker.WatermarkModel(new WatermarkFixtures.ToyLanguageModel());
        Assert.Equal(WatermarkFixtures.Generate(full, 11), WatermarkFixtures.Generate(incremental, 11));
    }

    [Fact(Timeout = 120000)]
    public async Task BiasLogits_AddsDeltaToExactlyTheGreenTokens()
    {
        await Task.Yield();
        var watermarker = new SamplingWatermarker<double>(key: TestKey);
        var context = new[] { 3, 17 };
        var logits = new Vector<double>(WatermarkFixtures.Vocabulary);
        watermarker.BiasLogits(logits, context);
        int green = 0;
        for (int token = 0; token < logits.Length; token++)
        {
            double expected = watermarker.IsGreen(context, token) ? watermarker.GreenListBias : 0.0;
            Assert.Equal(expected, logits[token]);
            if (expected > 0) green++;
        }

        // The green list holds about gamma = 25% of the vocabulary: 125 +/- 4 standard deviations of a binomial.
        Assert.InRange(green, 125 - 39, 125 + 39);
    }
}

/// <summary>The pipeline's <see cref="TextWatermarker{T}"/> on text, scored through the generator's tokenizer.</summary>
public class TextWatermarkerRoundTripTests : WatermarkRoundTripTestBase<string>
{
    private static readonly WatermarkFixtures.WordTokenizer Tokenizer = new();

    protected override string Sample(int seed) =>
        Tokenizer.Decode(WatermarkFixtures.Generate(new WatermarkFixtures.ToyLanguageModel(), seed % WatermarkFixtures.Vocabulary));

    protected override string Embed(string key, string content) =>
        Tokenizer.Decode(WatermarkFixtures.Generate(
            new TextWatermarker<double>(key: key, tokenizer: Tokenizer).WatermarkModel(new WatermarkFixtures.ToyLanguageModel()),
            Tokenizer.Encode(content)[0]));

    protected override WatermarkResult Detect(string key, string content) => new TextWatermarker<double>(key: key, tokenizer: Tokenizer).Detect(content);

    protected override string Perturb(string content, int seed) =>
        Tokenizer.Decode(WatermarkFixtures.ReplaceTenPercent(Tokenizer.Encode(content).ToArray(), seed));

    protected override bool SameContent(string a, string b) => string.Equals(a, b, StringComparison.Ordinal);

    [Fact(Timeout = 120000)]
    public async Task EvaluateText_ReportsMarkedText_AndNotUnmarkedText()
    {
        await Task.Yield();
        var detector = new TextWatermarker<double>(key: TestKey, tokenizer: Tokenizer);
        var finding = Assert.Single(detector.EvaluateText(Embed(TestKey, Sample(1))));
        Assert.Equal(SafetyCategory.Watermarked, finding.Category);
        Assert.Empty(detector.EvaluateText(Sample(1)));
    }

    [Fact(Timeout = 120000)]
    public async Task WithoutATokenizer_TextIsNoEvidence()
    {
        await Task.Yield();
        var result = new TextWatermarker<double>(key: TestKey).Detect(Embed(TestKey, Sample(1)));
        Assert.Equal(0, result.ScoredUnits);
        Assert.Equal(0.0, result.Confidence);
    }
}

/// <summary>Keyed synonym substitution, embedded into finished text.</summary>
public class LexicalWatermarkerRoundTripTests : WatermarkRoundTripTestBase<string>
{
    protected override string Sample(int seed) => WatermarkFixtures.SubstitutionProse(LexicalWatermarker<double>.DefaultSynonymGroups, seed);

    protected override string Embed(string key, string content) => new LexicalWatermarker<double>(key: key).EmbedWatermark(content);

    protected override WatermarkResult Detect(string key, string content) => new LexicalWatermarker<double>(key: key).Detect(content);

    protected override string Perturb(string content, int seed) => WatermarkFixtures.EditTenPercentOfSentences(content, seed);

    protected override bool SameContent(string a, string b) => string.Equals(a, b, StringComparison.Ordinal);

    [Fact(Timeout = 120000)]
    public async Task Embedding_ChangesOnlyWholeSynonyms_AndKeepsCapitalisation()
    {
        await Task.Yield();
        var watermarker = new LexicalWatermarker<double>(key: TestKey);
        // "Mainland" and "remains" contain group members as substrings and must not be touched.
        const string text = "Mainland ports remains open. Perhaps the river rarely floods. MAYBE NOT.";
        string marked = watermarker.EmbedWatermark(text);
        Assert.StartsWith("Mainland ports remains open. ", marked);
        Assert.True(marked.Contains("Perhaps the river") || marked.Contains("Maybe the river"), marked);
        Assert.True(marked.Contains(" rarely floods") || marked.Contains(" seldom floods"), marked);
        Assert.True(marked.EndsWith("MAYBE NOT.") || marked.EndsWith("PERHAPS NOT."), marked);
        Assert.Equal(marked, watermarker.EmbedWatermark(marked));
    }
}

/// <summary>Keyed contraction and expansion, embedded into finished text.</summary>
public class SyntacticWatermarkerRoundTripTests : WatermarkRoundTripTestBase<string>
{
    protected override string Sample(int seed) => WatermarkFixtures.SubstitutionProse(SyntacticWatermarker<double>.ContractionGroups, seed);

    protected override string Embed(string key, string content) => new SyntacticWatermarker<double>(key: key).EmbedWatermark(content);

    protected override WatermarkResult Detect(string key, string content) => new SyntacticWatermarker<double>(key: key).Detect(content);

    protected override string Perturb(string content, int seed) => WatermarkFixtures.EditTenPercentOfSentences(content, seed);

    protected override bool SameContent(string a, string b) => string.Equals(a, b, StringComparison.Ordinal);

    [Fact(Timeout = 120000)]
    public async Task Embedding_RewritesContractionsBothWays_AndKeepsCapitalisation()
    {
        await Task.Yield();
        var watermarker = new SyntacticWatermarker<double>(key: TestKey);
        const string text = "Do not panic. They don’t know it's late.";
        string marked = watermarker.EmbedWatermark(text);
        Assert.True(marked.StartsWith("Do not panic.") || marked.StartsWith("Don't panic."), marked);
        Assert.True(marked.Contains("They don't know") || marked.Contains("They do not know"), marked);
        Assert.EndsWith("it's late.", marked);
        Assert.Equal(marked, watermarker.EmbedWatermark(marked));
    }
}

/// <summary>The composite detector, finding a synonym watermark among the three it checks.</summary>
public class WatermarkDetectorRoundTripTests : WatermarkRoundTripTestBase<string>
{
    protected override bool ReportsStandardNormalZ => false;

    protected override string Sample(int seed) => WatermarkFixtures.SubstitutionProse(LexicalWatermarker<double>.DefaultSynonymGroups, seed);

    protected override string Embed(string key, string content) => new LexicalWatermarker<double>(key: key).EmbedWatermark(content);

    protected override WatermarkResult Detect(string key, string content) => new WatermarkDetector<double>(key: key).Detect(content);

    protected override string Perturb(string content, int seed) => WatermarkFixtures.EditTenPercentOfSentences(content, seed);

    protected override bool SameContent(string a, string b) => string.Equals(a, b, StringComparison.Ordinal);
}

/// <summary>Additive spread spectrum in the 1-7 kHz block-DCT band.</summary>
public class AudioWatermarkerRoundTripTests : WatermarkRoundTripTestBase<Vector<double>>
{
    protected override Vector<double> Sample(int seed) => WatermarkFixtures.Voiced(seed);

    protected override Vector<double> Embed(string key, Vector<double> content) => new AudioWatermarker<double>(key: key).EmbedWatermark(content, 16000);

    protected override WatermarkResult Detect(string key, Vector<double> content) => new AudioWatermarker<double>(key: key).Detect(content, 16000);

    protected override Vector<double> Perturb(Vector<double> content, int seed) => WatermarkFixtures.NoisyRequantized(content, seed);

    protected override bool SameContent(Vector<double> a, Vector<double> b) => WatermarkFixtures.SameAudio(a, b);

    [Fact(Timeout = 120000)]
    public async Task Embedding_IsQuiet_StaysInRange_AndLeavesTheInputUntouched()
    {
        await Task.Yield();
        var audio = Sample(1);
        var copy = new Vector<double>(audio.Length);
        for (int i = 0; i < audio.Length; i++) copy[i] = audio[i];
        var marked = Embed(TestKey, audio);
        Assert.True(WatermarkFixtures.SameAudio(copy, audio), "Embedding must not modify its input.");
        double signal = 0, noise = 0;
        for (int i = 0; i < audio.Length; i++)
        {
            Assert.InRange(marked[i], -1.0, 1.0);
            signal += audio[i] * audio[i];
            noise += (marked[i] - audio[i]) * (marked[i] - audio[i]);
        }

        double snr = 10 * Math.Log10(signal / noise);
        Assert.True(snr >= 35, $"The mark is only {snr:F1} dB below the signal.");
    }

    [Fact(Timeout = 120000)]
    public async Task SilenceAndTooShortAudio_CannotBeMarked_AndScoreZero()
    {
        await Task.Yield();
        var watermarker = new AudioWatermarker<double>(key: TestKey);
        var silence = new Vector<double>(16000);
        Assert.Throws<ArgumentException>(() => watermarker.EmbedWatermark(silence, 16000));
        Assert.Equal(0.0, watermarker.DetectWatermark(silence, 16000));
        var tiny = new Vector<double>(100);
        Assert.Throws<ArgumentException>(() => watermarker.EmbedWatermark(tiny, 16000));
        Assert.Equal(0.0, watermarker.DetectWatermark(tiny, 16000));
    }
}

/// <summary>Patchwork: keyed multiplicative magnitude changes in the 0.5-6 kHz block-DCT band.</summary>
public class SpectralAudioWatermarkerRoundTripTests : WatermarkRoundTripTestBase<Vector<double>>
{
    protected override Vector<double> Sample(int seed) => WatermarkFixtures.Voiced(seed);

    protected override Vector<double> Embed(string key, Vector<double> content) => new SpectralAudioWatermarker<double>(key: key).EmbedWatermark(content, 16000);

    protected override WatermarkResult Detect(string key, Vector<double> content) => new SpectralAudioWatermarker<double>(key: key).Detect(content, 16000);

    protected override Vector<double> Perturb(Vector<double> content, int seed) => WatermarkFixtures.NoisyRequantized(content, seed);

    protected override bool SameContent(Vector<double> a, Vector<double> b) => WatermarkFixtures.SameAudio(a, b);
}

/// <summary>Localized spread spectrum: detection per segment, and of any marked segment.</summary>
public class AudioSealWatermarkerRoundTripTests : WatermarkRoundTripTestBase<Vector<double>>
{
    protected override bool ReportsStandardNormalZ => false;

    protected override Vector<double> Sample(int seed) => WatermarkFixtures.Voiced(seed);

    protected override Vector<double> Embed(string key, Vector<double> content) => new AudioSealWatermarker<double>(key: key).EmbedWatermark(content, 16000);

    protected override WatermarkResult Detect(string key, Vector<double> content) => new AudioSealWatermarker<double>(key: key).Detect(content, 16000);

    protected override Vector<double> Perturb(Vector<double> content, int seed) => WatermarkFixtures.NoisyRequantized(content, seed);

    protected override bool SameContent(Vector<double> a, Vector<double> b) => WatermarkFixtures.SameAudio(a, b);

    [Fact(Timeout = 120000)]
    public async Task DetectSegments_LocatesTheMarkedHalfOfASplicedClip()
    {
        await Task.Yield();
        // Two seconds of speech, marked as a whole, then the first 16384 samples (32 frames of 512) put back unmarked:
        // a clip whose second half is generated. With 8192-sample segments, segments 0-1 are real and 2-3 marked.
        var watermarker = new AudioSealWatermarker<double>(segmentSamples: 8192, key: TestKey);
        var original = Concatenate(WatermarkFixtures.Voiced(5), WatermarkFixtures.Voiced(6));
        var marked = watermarker.EmbedWatermark(original, 16000);
        var splice = new Vector<double>(original.Length);
        for (int i = 0; i < original.Length; i++) splice[i] = i < 16384 ? original[i] : marked[i];

        var segments = watermarker.DetectSegments(splice, 16000);
        Assert.Equal(4, segments.Count);
        Assert.True(segments[0].ZScore < 3 && segments[1].ZScore < 3,
            $"Unmarked segments scored z = {segments[0].ZScore:F2} and {segments[1].ZScore:F2}.");
        Assert.True(segments[2].ZScore >= 4.5 && segments[3].ZScore >= 4.5,
            $"Marked segments scored z = {segments[2].ZScore:F2} and {segments[3].ZScore:F2}.");
        Assert.True(watermarker.Detect(splice, 16000).Confidence > 0.5);
    }

    private static Vector<double> Concatenate(Vector<double> a, Vector<double> b)
    {
        var result = new Vector<double>(a.Length + b.Length);
        for (int i = 0; i < a.Length; i++) result[i] = a[i];
        for (int i = 0; i < b.Length; i++) result[a.Length + i] = b[i];
        return result;
    }
}
