using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Safety;
using AiDotNet.Safety.Watermarking;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Safety;

/// <summary>
/// The shared keyed-watermark contract: key handling, the detection statistics, and the pipeline's embed entry points.
/// </summary>
public class WatermarkKeyAndPipelineTests
{
    private const string Secret = "pipeline-secret-key-81d2";

    [Fact(Timeout = 60000)]
    public async Task Key_RejectsBlankText_AndRequireSecretRejectsThePublicKey()
    {
        await Task.Yield();
        Assert.Throws<ArgumentException>(() => new WatermarkKey(""));
        Assert.Throws<ArgumentException>(() => new WatermarkKey("   "));
        Assert.True(WatermarkKey.FromOptional(null).IsPublic);
        Assert.False(WatermarkKey.FromOptional(Secret).IsPublic);
        Assert.Throws<InvalidOperationException>(() => WatermarkKey.RequireSecret(null, "Watermarking.TextKey"));
        Assert.Throws<InvalidOperationException>(() => WatermarkKey.RequireSecret(" ", "Watermarking.TextKey"));
        Assert.Throws<InvalidOperationException>(() => WatermarkKey.RequireSecret(WatermarkKey.PublicKeyText, "Watermarking.TextKey"));
        Assert.False(WatermarkKey.RequireSecret(Secret, "Watermarking.TextKey").IsPublic);
    }

    [Fact(Timeout = 60000)]
    public async Task Key_DerivationsAreDeterministic_AndSeparatedByKeySaltAndStream()
    {
        await Task.Yield();
        var a = new WatermarkKey(Secret);
        var b = new WatermarkKey(Secret);
        Assert.Equal(a.SignPattern(1, 2, 600), b.SignPattern(1, 2, 600));
        Assert.Equal(a.Seed(1, new[] { 4, 5 }), b.Seed(1, new[] { 4, 5 }));
        Assert.NotEqual(a.SignPattern(1, 2, 600), new WatermarkKey(Secret + "x").SignPattern(1, 2, 600));
        Assert.NotEqual(a.SignPattern(1, 2, 600), a.SignPattern(9, 2, 600));
        Assert.NotEqual(a.SignPattern(1, 2, 600), a.SignPattern(1, 3, 600));
        Assert.NotEqual(a.Seed(1, new[] { 4, 5 }), a.Seed(1, new[] { 5, 4 }));

        // The pattern is +/-1 and balanced: 6000 fair bits sum within 5 standard deviations (5 x sqrt(6000) = 387).
        var pattern = a.SignPattern(1, 2, 6000);
        Assert.All(pattern, v => Assert.True(v == 1.0 || v == -1.0));
        Assert.InRange(pattern.Sum(), -387, 387);
    }

    [Fact(Timeout = 60000)]
    public async Task Statistics_MatchTheStandardNormalTail()
    {
        await Task.Yield();
        Assert.Equal(0.5, WatermarkStatistics.UpperTailProbability(0), 6);
        Assert.Equal(0.0013498980316301, WatermarkStatistics.UpperTailProbability(3), 9);
        Assert.Equal(9.8658764503769e-10, WatermarkStatistics.UpperTailProbability(6), 14);
        foreach (double z in new[] { -2.0, 0.5, 3.0, 4.5, 8.0 })
            Assert.Equal(z, WatermarkStatistics.ZForUpperTail(WatermarkStatistics.UpperTailProbability(z)), 5);

        Assert.Equal(0.0, WatermarkStatistics.Confidence(3.0));
        Assert.Equal(0.5, WatermarkStatistics.Confidence(4.5), 12);
        Assert.Equal(1.0, WatermarkStatistics.Confidence(6.0));
        Assert.Equal(0.0, WatermarkStatistics.Confidence(double.NaN));
        Assert.Equal(5.0, WatermarkStatistics.CountZ(75, 100, 0.5), 12);

        // Bonferroni over 3 tests: P(Z >= z') = 3 P(Z >= 4), and never negative.
        double corrected = WatermarkStatistics.BonferroniZ(4.0, 3);
        Assert.Equal(3 * WatermarkStatistics.UpperTailProbability(4.0), WatermarkStatistics.UpperTailProbability(corrected), 9);
        Assert.Equal(0.0, WatermarkStatistics.BonferroniZ(-1.0, 3));
        Assert.Equal(40.0, WatermarkStatistics.BonferroniZ(40.0, 3));
    }

    [Fact(Timeout = 60000)]
    public async Task Pipeline_RefusesToEmbedWithoutAKeyedWatermarker()
    {
        await Task.Yield();
        var empty = new SafetyPipelineBuilder<double>().Build();
        Assert.Throws<InvalidOperationException>(() => empty.EmbedAudioWatermark(WatermarkFixtures.Voiced(1), 16000));
        Assert.Throws<InvalidOperationException>(() => empty.EmbedTextWatermark("Perhaps it rarely rains."));
        Assert.Throws<InvalidOperationException>(() => empty.WatermarkLanguageModel(new WatermarkFixtures.ToyLanguageModel()));

        // A watermarker on the published key is a configuration error, not a silent forgeable mark.
        var publicKey = new SafetyPipelineBuilder<double>()
            .AddAudioModule(new AudioWatermarker<double>())
            .AddTextModule(new LexicalWatermarker<double>())
            .AddTextModule(new TextWatermarker<double>())
            .Build();
        Assert.Throws<InvalidOperationException>(() => publicKey.EmbedAudioWatermark(WatermarkFixtures.Voiced(1), 16000));
        Assert.Throws<InvalidOperationException>(() => publicKey.EmbedTextWatermark("Perhaps it rarely rains."));
        Assert.Throws<InvalidOperationException>(() => publicKey.WatermarkLanguageModel(new WatermarkFixtures.ToyLanguageModel()));
    }

    [Fact(Timeout = 120000)]
    public async Task Pipeline_EmbedsWithItsKeyedWatermarkers_AndDetectsTheMarks()
    {
        await Task.Yield();
        var tokenizer = new WatermarkFixtures.WordTokenizer();
        var pipeline = new SafetyPipelineBuilder<double>()
            .AddAudioModule(new AudioWatermarker<double>())                 // public key: skipped for embedding
            .AddAudioModule(new AudioWatermarker<double>(key: Secret))
            .AddTextModule(new LexicalWatermarker<double>(key: Secret))
            .AddTextModule(new TextWatermarker<double>(key: Secret, tokenizer: tokenizer))
            .Build();

        var audio = pipeline.EmbedAudioWatermark(WatermarkFixtures.Voiced(2), 16000);
        Assert.True(new AudioWatermarker<double>(key: Secret).DetectWatermark(audio, 16000) >= 0.9);
        Assert.Contains(pipeline.EvaluateAudio(audio, 16000).Findings, f => f.Category == SafetyCategory.Watermarked);

        string prose = WatermarkFixtures.SubstitutionProse(LexicalWatermarker<double>.DefaultSynonymGroups, 2);
        string marked = pipeline.EmbedTextWatermark(prose);
        Assert.True(new LexicalWatermarker<double>(key: Secret).DetectWatermark(marked) >= 0.9);

        var model = pipeline.WatermarkLanguageModel(new WatermarkFixtures.ToyLanguageModel());
        string generated = tokenizer.Decode(WatermarkFixtures.Generate(model, 4));
        Assert.Contains(pipeline.EvaluateText(generated).Findings,
            f => f.Category == SafetyCategory.Watermarked && f.SourceModule == "TextWatermarker");
    }
}
