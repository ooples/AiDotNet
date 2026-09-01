using AiDotNet.Evolution;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution;

public sealed class EvolutionSelectionPolicyTests
{
    [Fact]
    public void DoubleSelectionUsesDistinctQualityRankedInspirations()
    {
        var archive = new MapElitesArchive<TestGenome>(new[]
        {
            new EvolutionDescriptorDefinition("x", 0, 4, 4)
        });
        for (int i = 0; i < 4; i++) MapElitesArchiveTests.Add(archive, i, $"g{i}", i, i + 0.1);
        var policy = new DoubleEvolutionSelectionPolicy<TestGenome>();

        EvolutionSelection<TestGenome> selection = Assert.IsType<EvolutionSelection<TestGenome>>(
            policy.Select(archive, new StableRandom(3), inspirationCount: 3));

        Assert.DoesNotContain(selection.Inspirations,
            entry => entry.Evaluation.GenomeId == selection.Parent.Evaluation.GenomeId);
        Assert.Equal(selection.Inspirations.Count,
            selection.Inspirations.Select(entry => entry.Evaluation.GenomeId).Distinct().Count());
        Assert.Equal(selection.Inspirations.OrderByDescending(entry => entry.Evaluation.Quality)
            .Select(entry => entry.Evaluation.GenomeId), selection.Inspirations.Select(entry => entry.Evaluation.GenomeId));
    }

    [Fact]
    public void CuriosityStateRoundTripsAndRewardsSuccessfulParents()
    {
        var first = new CuriosityEvolutionSelectionPolicy<TestGenome>();
        (_, EvolutionEvaluation evaluation) = MapElitesArchiveTests.Create(2, "child", 2, 0.2);
        var lineage = new EvolutionLineage(new[] { "parent" }, null, "mut", null, 1, 0, 2);
        var childEvaluation = new EvolutionEvaluation(2, "child", EvolutionEvaluationStatus.Completed, 2,
            EvolutionOptimizationDirection.Maximize, evaluation.Descriptors, Array.Empty<double>(), Array.Empty<double>(),
            evaluation.Cost, lineage, EvolutionCacheStatus.Miss, Array.Empty<EvolutionDiagnostic>(), "task", "eval", "config");

        first.Observe(childEvaluation, EvolutionArchiveInsertionResult.Inserted);
        string state = first.CaptureState();
        var restored = new CuriosityEvolutionSelectionPolicy<TestGenome>();
        restored.RestoreState(state);

        Assert.Equal(2.0, first.Scores["parent"]);
        Assert.Equal(first.Scores, restored.Scores);
        Assert.Equal(state, restored.CaptureState());
    }
}
