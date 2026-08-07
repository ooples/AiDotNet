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

    /// <summary>
    /// A video built to trip SEVERAL harm categories at once, one representative signal each.
    /// </summary>
    /// <remarks>
    /// <see cref="Frames(int, int)"/> is uniform noise, which is exactly the content the taxonomy is
    /// meant to find nothing in — so it cannot be used to assert that findings ARE produced. The
    /// multi-label and non-empty assertions need a fixture whose content is detectable, and it must
    /// be detectable under more than one category for the non-mutually-exclusive claim to mean
    /// anything.
    /// </remarks>
    private static List<Tensor<double>> MultiHarmFrames()
    {
        var frames = Frames(60);

        // Saturate distinct channel/region combinations so more than one detector responds. The
        // exact values are not meaningful; what matters is that the content is far from the uniform
        // noise the benign fixture uses, in more than one way.
        for (int f = 0; f < frames.Count; f++)
        {
            var t = frames[f];
            for (int c = 0; c < 3; c++)
            {
                for (int y = 0; y < 8; y++)
                {
                    for (int x = 0; x < 8; x++)
                    {
                        int i = (c * 8 + y) * 8 + x;
                        if (i >= t.Length) continue;
                        t[i] = c == 0 && y < 4 ? 1.0
                             : c == 1 && x < 4 ? 0.0
                             : t[i];
                    }
                }
            }
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

        // BOTH DIRECTIONS. Checking only that every unmapped signal is on the list lets the list
        // rot the other way: once PolicyViolation gains a mapping, its entry sits here forever and
        // nothing reports that the deliberate exclusion is now a lie. A stale exclusion is exactly
        // as invisible as an unmapped signal, so it gets the same assertion.
        var actuallyUnmapped = new HashSet<SafetyCategory>();

        foreach (SafetyCategory signal in (SafetyCategory[])Enum.GetValues(typeof(SafetyCategory)))
        {
            var harm = HarmTaxonomyMap.ToHarmCategory(signal);
            if (harm is null)
            {
                actuallyUnmapped.Add(signal);
                Assert.True(deliberatelyUnmapped.Contains(signal),
                    $"{signal} maps to no harm category and is not on the deliberate-exclusion list. " +
                    "Either map it or record why it is not a content harm.");
            }
        }

        var staleExclusions = deliberatelyUnmapped.Except(actuallyUnmapped).ToList();
        Assert.True(staleExclusions.Count == 0,
            "These signals are on the deliberate-exclusion list but now DO map into the taxonomy: " +
            string.Join(", ", staleExclusions) +
            ". Remove them from the list -- an exclusion that no longer excludes anything documents " +
            "a decision that has since been reversed.");
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
        var findings = moderator.EvaluateVideo(MultiHarmFrames(), 30.0);

        Assert.NotNull(findings);

        // AN EMPTY LIST SATISFIED THE DUPLICATE CHECK. A moderator that detected nothing at all
        // passed this test, and the property the name claims -- that SEVERAL categories can be
        // reported for one video -- was never asserted at all. Both halves are asserted now.
        Assert.NotEmpty(findings);

        var categories = findings.Select(f => f.Category).Distinct().ToList();
        Assert.True(categories.Count > 1,
            "Non-mutually-exclusive means a single video can be reported under SEVERAL categories. " +
            "This fixture is built to trip more than one, yet only these were reported: " +
            string.Join(", ", categories));

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

        // THE CONSTANT ALONE PROVES NOTHING. Asserting the budget field and then only that two
        // calls returned non-null left the actual cost property untested: a moderator that reverted
        // to a per-frame rate satisfied every one of those assertions. The number of frames the
        // moderator actually looked at is the observable, so that is what is asserted.
        moderator.EvaluateVideo(Frames(30), 30.0);
        int sampledForShort = moderator.LastSampledFrameCount;

        moderator.EvaluateVideo(Frames(600), 30.0);
        int sampledForLong = moderator.LastSampledFrameCount;

        Assert.Equal(sampledForShort, sampledForLong);
        Assert.True(sampledForLong <= MultimodalVideoModerator<double>.TaxonomyFrameBudget,
            $"A 600-frame video was sampled {sampledForLong} times against a budget of " +
            $"{MultimodalVideoModerator<double>.TaxonomyFrameBudget}. The budget is a rate, not a constant.");
    }

    [Fact]
    public void FindingsAreVideoLevel()
    {
        var frames = MultiHarmFrames();
        var findings = new MultimodalVideoModerator<double>().EvaluateVideo(frames, 30.0);

        // A foreach over an empty list runs its body zero times and passes. Nothing else in this
        // file established that the fixture produces findings at all, so a moderator that stopped
        // detecting everything turned this test -- and two others -- green.
        Assert.NotEmpty(findings);

        foreach (var finding in findings)
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

        // THE CONTRACT FOR ONE FRAME, not NotNull -- which the moderator satisfies either way and
        // which therefore contradicted this method's own name. A single frame is a valid video: the
        // taxonomy's harms are judged from content, and one frame of benign noise carries none of
        // them. It is degenerate in LENGTH, not in kind, so it must produce no findings for benign
        // content rather than being exempt from the assertion.
        Assert.Empty(moderator.EvaluateVideo(Frames(1), 30.0));
    }
}
