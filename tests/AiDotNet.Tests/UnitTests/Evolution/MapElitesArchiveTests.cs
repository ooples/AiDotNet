using AiDotNet.Enums;
using AiDotNet.Evolution;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution;

public sealed class MapElitesArchiveTests
{
    [Fact]
    public void MaximizeAndMinimizeUseOppositeQualityDirections()
    {
        MapElitesArchive<TestGenome> maximize = Archive(EvolutionOptimizationDirection.Maximize);
        MapElitesArchive<TestGenome> minimize = Archive(EvolutionOptimizationDirection.Minimize);

        Add(maximize, 1, "a", 1, 0.2);
        Add(maximize, 2, "b", 2, 0.2);
        Add(minimize, 1, "a", 1, 0.2);
        Add(minimize, 2, "b", 2, 0.2);

        Assert.NotNull(maximize.Best);
        Assert.NotNull(minimize.Best);
        Assert.Equal("b", maximize.Best?.Evaluation.GenomeId);
        Assert.Equal("a", minimize.Best?.Evaluation.GenomeId);
    }

    [Fact]
    public void EqualQualityUsesCanonicalGenomeIdTieBreaker()
    {
        MapElitesArchive<TestGenome> archive = Archive();

        Assert.Equal(EvolutionArchiveInsertionResult.Inserted, Add(archive, 1, "z", 5, 0.2));
        Assert.Equal(EvolutionArchiveInsertionResult.Replaced, Add(archive, 2, "a", 5, 0.2));
        Assert.NotNull(archive.Best);
        Assert.Equal("a", archive.Best?.Evaluation.GenomeId);
    }

    [Fact]
    public void BoundedEvictionIsIndependentOfInsertionOrder()
    {
        string[][] orders =
        {
            new[] { "a", "b", "c", "d" },
            new[] { "d", "b", "a", "c" },
            new[] { "c", "a", "d", "b" }
        };

        string[]? expected = null;
        foreach (string[] order in orders)
        {
            var archive = new MapElitesArchive<TestGenome>(new[]
            {
                new EvolutionDescriptorDefinition("x", 0, 4, 4)
            }, capacity: 2);
            var values = new Dictionary<string, (double Quality, double Descriptor)>
            {
                ["a"] = (1, 0.2), ["b"] = (4, 1.2), ["c"] = (3, 2.2), ["d"] = (2, 3.2)
            };
            int id = 0;
            foreach (string genomeId in order)
            {
                (double quality, double descriptor) = values[genomeId];
                Add(archive, id++, genomeId, quality, descriptor);
            }

            string[] actual = archive.Entries.Select(entry => entry.Evaluation.GenomeId).OrderBy(value => value).ToArray();
            expected ??= actual;
            Assert.Equal(expected, actual);
        }
        Assert.Equal(new[] { "b", "c" }, expected);
    }

    [Fact]
    public void InvalidEvaluationDoesNotEnterArchive()
    {
        MapElitesArchive<TestGenome> archive = Archive();
        (EvolutionCandidate<TestGenome> candidate, EvolutionEvaluation evaluation) = Create(1, "a", 1, 0.2);
        var failed = new EvolutionEvaluation(1, "a", EvolutionEvaluationStatus.Failed, null,
            EvolutionOptimizationDirection.Maximize, new Dictionary<string, double>(), Array.Empty<double>(),
            Array.Empty<double>(), new EvolutionEvaluationCost(TimeSpan.Zero, 1, 0), evaluation.Lineage,
            EvolutionCacheStatus.NotChecked, Array.Empty<EvolutionDiagnostic>(), "task", "eval", "config");

        Assert.Equal(EvolutionArchiveInsertionResult.Rejected, archive.TryAdd(candidate, failed));
        Assert.Empty(archive.Entries);
    }

    internal static MapElitesArchive<TestGenome> Archive(
        EvolutionOptimizationDirection direction = EvolutionOptimizationDirection.Maximize) => new(new[]
    {
        new EvolutionDescriptorDefinition("x", 0, 1, 5, EvolutionOutOfRangePolicy.Reject)
    }, direction);

    internal static EvolutionArchiveInsertionResult Add(MapElitesArchive<TestGenome> archive, long id,
        string genomeId, double quality, double descriptor)
    {
        (EvolutionCandidate<TestGenome> candidate, EvolutionEvaluation evaluation) = Create(
            id, genomeId, quality, descriptor, archive.Direction);
        return archive.TryAdd(candidate, evaluation);
    }

    internal static (EvolutionCandidate<TestGenome>, EvolutionEvaluation) Create(long id, string genomeId,
        double quality, double descriptor,
        EvolutionOptimizationDirection direction = EvolutionOptimizationDirection.Maximize)
    {
        var lineage = new EvolutionLineage(null, null, "test", null, 0, 0, (ulong)id);
        var candidate = new EvolutionCandidate<TestGenome>(id,
            new EvolutionCanonicalGenome<TestGenome>(new TestGenome((int)id), genomeId), lineage);
        var evaluation = new EvolutionEvaluation(id, genomeId, EvolutionEvaluationStatus.Completed, quality,
            direction, new Dictionary<string, double> { ["x"] = descriptor }, Array.Empty<double>(),
            Array.Empty<double>(), new EvolutionEvaluationCost(TimeSpan.Zero, 1, 0), lineage,
            EvolutionCacheStatus.Miss, Array.Empty<EvolutionDiagnostic>(), "task", "eval", "config");
        return (candidate, evaluation);
    }
}
