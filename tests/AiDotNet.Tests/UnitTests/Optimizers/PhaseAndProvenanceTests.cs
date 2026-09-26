using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Optimizers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// Tests the v2 record: a model declares every stage its paper states, and resolution picks one (#1928).
/// </summary>
/// <remarks>
/// <para>
/// The v1 design held one recipe per model, so a paper stating several — 25 of 122 surveyed do —
/// had to have all but one described in prose. That put the paper's own words beyond the reach of
/// any check, and it was the largest single source of the 18% of declarations that narrated a
/// limitation instead of expressing it.
/// </para>
/// <para>
/// These assert the two behaviours that make a multi-phase record usable rather than merely
/// present: the right phase is selected, and a phase that says it reuses another's settings gets
/// them without either declaration being rewritten.
/// </para>
/// </remarks>
public class PhaseAndProvenanceTests
{
    [Fact]
    public void EachPhaseResolvesToItsOwnRecipe()
    {
        var pre = PaperOptimizerFactory.Find(new TwoStage(), phase: TrainingPhase.PreTraining);
        var fine = PaperOptimizerFactory.Find(new TwoStage(), phase: TrainingPhase.FineTuning);

        Assert.NotNull(pre);
        Assert.NotNull(fine);
        Assert.Equal(2e-3, pre!.LearningRate, precision: 12);
        Assert.Equal(2e-5, fine!.LearningRate, precision: 12);
        Assert.Equal(128, pre.ReferenceBatchSize);
        Assert.Equal(32, fine.ReferenceBatchSize);
    }

    [Fact]
    public void AFineTuningCallerIsNotHandedThePreTrainingRate()
    {
        // The failure this key exists to prevent. LLaVA pre-trains at 2e-3 and fine-tunes at 2e-5:
        // a hundredfold difference, and applying the wrong one destroys the pretrained weights.
        var fine = PaperOptimizerFactory.Find(new TwoStage(), phase: TrainingPhase.FineTuning);

        Assert.NotNull(fine);
        Assert.NotEqual(2e-3, fine!.LearningRate);
    }

    [Fact]
    public void AnInheritingPhaseTakesWhatItDoesNotState()
    {
        // SPEAR-TTS states its second stage as using the same optimizer and schedule as its first,
        // and gives only the rate that differs.
        var second = PaperOptimizerFactory.Find(new InheritingStages(), phase: TrainingPhase.FineTuning);

        Assert.NotNull(second);
        Assert.Equal(5e-5, second!.LearningRate, precision: 12);      // its own
        Assert.Equal(OptimizerKind.AdamW, second.Optimizer);           // inherited
        Assert.Equal(0.98, second.Beta2, precision: 12);               // inherited
        Assert.Equal(LearningRateSchedulerType.CosineAnnealing, second.Schedule); // inherited
    }

    [Fact]
    public void InheritanceDoesNotMutateEitherDeclaration()
    {
        // A declaration is the record of what a paper says. It must not change because something
        // else was read alongside it, or the second reader sees a recipe the paper never stated.
        PaperOptimizerFactory.Find(new InheritingStages(), phase: TrainingPhase.FineTuning);

        var parent = typeof(InheritingStages)
            .GetCustomAttributes(typeof(PaperOptimizerAttribute), false)
            .Cast<PaperOptimizerAttribute>()
            .Single(r => r.Phase == TrainingPhase.PreTraining);

        Assert.Equal(1e-4, parent.LearningRate, precision: 12);
    }

    [Fact]
    public void ASearchedRateIsRecordedWithItsAlternatives()
    {
        // A search is not a recommendation. Before provenance existed the only honest option was to
        // declare no rate at all, which lost the paper's actual claim entirely.
        var recipe = PaperOptimizerFactory.Find(new Searched());

        Assert.NotNull(recipe);
        Assert.Equal(RecipeProvenance.Searched, recipe!.Provenance);
        Assert.Equal(3, recipe.SearchedValues.Length);
        Assert.Contains(5e-4, recipe.SearchedValues);
        Assert.True(recipe.DeclaresAnyHyperparameter);
    }

    [Fact]
    public void AShippedModelResolvesEachOfItsDeclaredPhases()
    {
        // The fixtures above prove the mechanism; this proves a real declaration uses it.
        // LLaVA pre-trains at 2e-3 with batch 128 and fine-tunes at 2e-5 with batch 32 (Liu et
        // al. 2023, Sec. 5). Before the phase key, one of those was recorded and the other was a
        // sentence in the Source that nothing could check.
        var model = typeof(AiDotNet.NeuralNetworks.LLaVANeuralNetwork<double>);
        var rows = (PaperOptimizerAttribute[])model.GetCustomAttributes(
            typeof(PaperOptimizerAttribute), inherit: false);

        var pre = Assert.Single(rows.Where(r => r.Phase == TrainingPhase.PreTraining));
        var fine = Assert.Single(rows.Where(r => r.Phase == TrainingPhase.FineTuning));

        Assert.Equal(2e-3, pre.LearningRate, precision: 12);
        Assert.Equal(2e-5, fine.LearningRate, precision: 12);
        Assert.Equal(128, pre.ReferenceBatchSize);
        Assert.Equal(32, fine.ReferenceBatchSize);

        // Both state the paper explicitly rather than one inheriting silently.
        Assert.All(rows, r => Assert.False(string.IsNullOrWhiteSpace(r.Source)));
    }

    [PaperOptimizer(OptimizerKind.Adam, LearningRate = 2e-3, ReferenceBatchSize = 128,
                    Phase = TrainingPhase.PreTraining, Source = "fixture: pre-training stage")]
    [PaperOptimizer(OptimizerKind.Adam, LearningRate = 2e-5, ReferenceBatchSize = 32,
                    Phase = TrainingPhase.FineTuning, Source = "fixture: fine-tuning stage")]
    private sealed class TwoStage { }

    [PaperOptimizer(OptimizerKind.AdamW, Beta2 = 0.98, LearningRate = 1e-4,
                    Schedule = LearningRateSchedulerType.CosineAnnealing,
                    Phase = TrainingPhase.PreTraining, Source = "fixture: first stage")]
    [PaperOptimizer(OptimizerKind.Unspecified, LearningRate = 5e-5,
                    Phase = TrainingPhase.FineTuning, InheritsFrom = TrainingPhase.PreTraining,
                    Source = "fixture: second stage, same optimizer and schedule as the first")]
    private sealed class InheritingStages { }

    [PaperOptimizer(OptimizerKind.Adam, LearningRate = 5e-4,
                    Provenance = RecipeProvenance.Searched,
                    SearchedValues = [1e-3, 5e-4, 1e-4],
                    Source = "fixture: a paper that searched three rates")]
    private sealed class Searched { }
}
