using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Safety.Video;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Safety;

/// <summary>
/// Verifies that <see cref="MultimodalVideoModerator{T}"/> implements the six-category harm taxonomy
/// of arXiv:2411.05854 rather than an ad-hoc ensemble.
/// </summary>
/// <remarks>
/// The properties that matter are the taxonomy's own stated principles: exactly six categories,
/// NON-MUTUALLY EXCLUSIVE so a video can be reported under several at once, and a fixed multimodal
/// sampling budget rather than a rate. Without tests for those, the rebuild is asserted only by its
/// comments.
/// </remarks>
public class HarmTaxonomyModeratorTests
{
    private static List<Tensor<double>> Frames(int count, int seed = 11)
    {
        var rng = new Random(seed);
        var frames = new List<Tensor<double>>(count);
        for (int f = 0; f < count; f++)
        {
            var t = new Tensor<double>([3, 8, 8]);
            for (int i = 0; i < t.Length; i++) t[i] = rng.NextDouble();
            frames.Add(t);
        }
        return frames;
    }

    [Fact]
    public void TaxonomyHasExactlySixCategories()
    {
        // "Our taxonomy contains six non-mutually exclusive categories: Information harms, Hate and
        // harassment harms, Clickbait harms, Addictive harms, Sexual harms, and Physical harms."
        // Enum.GetValues<T>() is .NET 5+. This suite still builds for net471, where only the
        // non-generic overload exists, so cast its Array result to the typed array.
        var categories = (HarmCategory[])Enum.GetValues(typeof(HarmCategory));
        Assert.Equal(6, categories.Length);
        Assert.Contains(HarmCategory.Information, categories);
        Assert.Contains(HarmCategory.HateAndHarassment, categories);
        Assert.Contains(HarmCategory.Addictive, categories);
        Assert.Contains(HarmCategory.Clickbait, categories);
        Assert.Contains(HarmCategory.Sexual, categories);
        Assert.Contains(HarmCategory.Physical, categories);
    }

    [Fact]
    public void EverySignalMapsIntoTheTaxonomyOrIsDeliberatelyExcluded()
    {
        // A content-harm signal that maps nowhere would be silently dropped by the fold. Model
        // -integrity and provenance signals are excluded on purpose: they are not harms to a viewer,
        // and the taxonomy excludes anything not discernible as harm from the content itself.
        var deliberatelyUnmapped = new HashSet<SafetyCategory>
        {
            SafetyCategory.PromptInjection, SafetyCategory.JailbreakAttempt,
            SafetyCategory.TrainingDataLeakage, SafetyCategory.ModelExtraction,
            SafetyCategory.Watermarked, SafetyCategory.AIGenerated,
            SafetyCategory.CopyrightViolation, SafetyCategory.LegalAdvice,
            SafetyCategory.PIIExposure, SafetyCategory.SurveillanceEnabling,
            SafetyCategory.Malware, SafetyCategory.Bias,
            SafetyCategory.TransparencyViolation,
            // A generic catch-all, not one of the six specific harms. The taxonomy explicitly
            // "avoid[s] broad categories, such as 'problematic'" in favour of categories that are
            // "as objective and verifiable as possible", so a non-specific policy label has no home
            // in it by design.
            SafetyCategory.PolicyViolation,
        };

        foreach (SafetyCategory signal in (SafetyCategory[])Enum.GetValues(typeof(SafetyCategory)))
        {
            var harm = HarmTaxonomyMap.ToHarmCategory(signal);
            if (harm is null)
            {
                Assert.True(deliberatelyUnmapped.Contains(signal),
                    $"{signal} maps to no harm category and is not on the deliberate-exclusion list. " +
                    "Either map it or record why it is not a content harm.");
            }
        }
    }

    [Fact]
    public void EveryHarmCategoryHasARepresentativeSignal()
    {
        foreach (HarmCategory harm in (HarmCategory[])Enum.GetValues(typeof(HarmCategory)))
        {
            var signal = HarmTaxonomyMap.RepresentativeSignal(harm);
            Assert.Equal(harm, HarmTaxonomyMap.ToHarmCategory(signal));
        }
    }

    [Fact]
    public void CategoriesAreNonMutuallyExclusive_AtMostOneFindingEach()
    {
        // Multi-label by design: a video may be reported under several categories, but each category
        // yields a single video-level finding rather than one per frame.
        var moderator = new MultimodalVideoModerator<double>();
        var findings = moderator.EvaluateVideo(Frames(60), 30.0);

        Assert.NotNull(findings);
        var duplicated = findings.GroupBy(f => f.Category).Where(g => g.Count() > 1).ToList();
        Assert.True(duplicated.Count == 0,
            "Each harm category folds to one video-level finding; got duplicates for: " +
            string.Join(", ", duplicated.Select(g => $"{g.Key} x{g.Count()}")));
    }

    [Fact]
    public void SamplingBudgetIsFixed_NotAFrameRate()
    {
        // "14 image frames, 1 thumbnail, and text metadata" per video — a constant budget, so a
        // 20x longer video is not 20x more work.
        Assert.Equal(14, MultimodalVideoModerator<double>.TaxonomyFrameBudget);

        var moderator = new MultimodalVideoModerator<double>();
        Assert.NotNull(moderator.EvaluateVideo(Frames(30), 30.0));
        Assert.NotNull(moderator.EvaluateVideo(Frames(600), 30.0));
    }

    [Fact]
    public void FindingsAreVideoLevel()
    {
        var frames = Frames(90);
        foreach (var finding in new MultimodalVideoModerator<double>().EvaluateVideo(frames, 30.0))
        {
            Assert.Equal(0, finding.SpanStart);
            Assert.True(finding.SpanEnd > 0);
        }
    }

    [Fact]
    public void EvaluationIsDeterministic()
    {
        var frames = Frames(45);
        var moderator = new MultimodalVideoModerator<double>();
        var first = moderator.EvaluateVideo(frames, 30.0);
        var second = moderator.EvaluateVideo(frames, 30.0);

        Assert.Equal(first.Count, second.Count);
        for (int i = 0; i < first.Count; i++)
        {
            Assert.Equal(first[i].Category, second[i].Category);
            Assert.Equal(first[i].Confidence, second[i].Confidence, 12);
        }
    }

    [Fact]
    public void DegenerateInputs_ProduceNoFindings()
    {
        var moderator = new MultimodalVideoModerator<double>();
        Assert.Empty(moderator.EvaluateVideo(new List<Tensor<double>>(), 30.0));
        Assert.Empty(moderator.EvaluateVideo(Frames(10), 0.0));
        Assert.NotNull(moderator.EvaluateVideo(Frames(1), 30.0));
    }
}
