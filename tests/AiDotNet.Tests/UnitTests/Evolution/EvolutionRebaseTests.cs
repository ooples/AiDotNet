using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.Interfaces;
using AiDotNet.ProgramSynthesis.Enums;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution;

/// <summary>
/// Covers moving a relative descriptor onto a new reference set and putting the whole archive back on one ruler.
/// A descriptor that measures a candidate against other candidates gives a different answer as the population
/// moves, so an archive that never re-measures fills with coordinates taken against references that no longer
/// exist: cells stop being comparable with one another, and nothing reports it.
/// </summary>
public sealed class EvolutionRebaseTests
{
    [Fact]
    public void RebasingProducesANewDescriptorRatherThanChangingTheOldOne()
    {
        // The old reading may still be in use while the new one is prepared, and a version hash is supposed to
        // identify one exact reading.
        var original = new ProgramDiversityDescriptor(new[] { "def a():\n    return 1\n" });
        IRebasableProgramDescriptor rebased = original.Rebase(new[] { Genome("def b():\n    return 2\n") });

        Assert.NotSame(original, rebased);
        Assert.Equal(original.Name, rebased.Name);
        Assert.NotEqual(original.VersionHash, rebased.VersionHash);
        Assert.Single(original.ReferenceSources);
    }

    [Fact]
    public void RebasingAWholeSetMovesEveryRelativeDescriptorAndLeavesTheAbsoluteOnesAlone()
    {
        var set = new ProgramDescriptorSet(
            new ProgramLengthDescriptor(),
            new ProgramDiversityDescriptor(new[] { "def a():\n    return 1\n" }));

        // A reference of a very different size, so the distance to it genuinely differs from the distance to the
        // original one-line reference.
        ProgramDescriptorSet rebased = set.Rebase(new[]
        {
            Genome("def b(values):\n    total = 0\n    for value in values:\n        total += value * 3\n    return total\n")
        });

        Assert.Equal(set.Names, rebased.Names);
        Assert.NotEqual(set.VersionHash, rebased.VersionHash);

        // The absolute axis reads the same before and after; only the relative one moved.
        ProgramGenome probe = Genome("def c():\n    return 3\n");
        Assert.Equal(set.Compute(probe)["length"], rebased.Compute(probe)["length"], 10);
        Assert.NotEqual(set.Compute(probe)["diversity"], rebased.Compute(probe)["diversity"]);
    }

    [Fact]
    public void ASetOfAbsoluteDescriptorsHasNothingToRebase()
    {
        // Re-measuring a whole archive for a set that would produce identical values is pure cost.
        var set = new ProgramDescriptorSet(new ProgramLengthDescriptor());

        Assert.False(set.HasRebasableDescriptors);
        Assert.Same(set, set.Rebase(new[] { Genome("def b():\n    return 2\n") }));
    }

    [Fact]
    public void RemeasuringPutsEveryEliteBackOnOneRuler()
    {
        // The defect this exists to prevent: elites filed at different times keep coordinates taken against
        // different references, so two cells cannot meaningfully be compared.
        MapElitesArchive<TestGenome> archive = Archive(
            Entry(1, quality: 10, descriptor: 10),
            Entry(2, quality: 20, descriptor: 30),
            Entry(3, quality: 30, descriptor: 50));

        Assert.Equal(3, archive.Count);

        // A new reading that halves every value: the whole archive should move together.
        int remaining = archive.Remeasure(genome => new Dictionary<string, double>(StringComparer.Ordinal)
        {
            ["x"] = genome.Value * 10 / 2.0
        });

        Assert.Equal(remaining, archive.Count);
        foreach (EvolutionArchiveEntry<TestGenome> entry in archive.Entries)
        {
            Assert.Equal(entry.Candidate.CanonicalGenome.Genome.Value * 5.0, entry.Evaluation.Descriptors["x"], 10);
            Assert.Equal(archive.TryCreateKey(entry.Evaluation.Descriptors)!.StableKey, entry.Cell.StableKey);
        }
    }

    [Fact]
    public void TwoElitesColldingAfterRemeasurementKeepTheBetterOne()
    {
        // A new reading can map two elites onto one cell. Resolving it the way an ordinary insertion would is what
        // keeps the outcome independent of traversal order.
        MapElitesArchive<TestGenome> archive = Archive(
            Entry(1, quality: 10, descriptor: 10),
            Entry(2, quality: 90, descriptor: 50));

        archive.Remeasure(_ => new Dictionary<string, double>(StringComparer.Ordinal) { ["x"] = 25 });

        EvolutionArchiveEntry<TestGenome> survivor = Assert.Single(archive.Entries);
        Assert.Equal(90, survivor.Evaluation.Quality);
        Assert.Equal(90, archive.Best?.Evaluation.Quality);
    }

    [Fact]
    public void AnEliteTheNewReadingCannotPlaceKeepsTheCellItHad()
    {
        // Losing an elite to a re-measurement would throw away search progress that was really made.
        MapElitesArchive<TestGenome> archive = Archive(Entry(1, quality: 10, descriptor: 10));
        string before = Assert.Single(archive.Entries).Cell.StableKey;

        archive.Remeasure(_ => null);

        EvolutionArchiveEntry<TestGenome> after = Assert.Single(archive.Entries);
        Assert.Equal(before, after.Cell.StableKey);
        Assert.Equal(10, after.Evaluation.Descriptors["x"], 10);
    }

    [Fact]
    public void RemeasuringIsIndependentOfTheOrderTheArchiveWasFilledIn()
    {
        // Two runs that reached the same archive by different routes must re-measure to the same archive, or the
        // state hash stops meaning anything.
        MapElitesArchive<TestGenome> forwards = Archive(
            Entry(1, quality: 10, descriptor: 10),
            Entry(2, quality: 20, descriptor: 30),
            Entry(3, quality: 30, descriptor: 50));
        MapElitesArchive<TestGenome> backwards = Archive(
            Entry(3, quality: 30, descriptor: 50),
            Entry(1, quality: 10, descriptor: 10),
            Entry(2, quality: 20, descriptor: 30));

        Func<TestGenome, IReadOnlyDictionary<string, double>?> reading = genome =>
            new Dictionary<string, double>(StringComparer.Ordinal) { ["x"] = 100 - genome.Value * 10 };

        forwards.Remeasure(reading);
        backwards.Remeasure(reading);

        Assert.Equal(
            forwards.Entries.Select(entry => entry.Cell.StableKey + ":" + entry.Evaluation.GenomeId),
            backwards.Entries.Select(entry => entry.Cell.StableKey + ":" + entry.Evaluation.GenomeId));
    }

    [Fact]
    public void ARemeasurementReportsTheArchiveChangedSoReadersCanNotice()
    {
        MapElitesArchive<TestGenome> archive = Archive(Entry(1, quality: 10, descriptor: 10));
        long before = archive.Version;

        archive.Remeasure(genome => new Dictionary<string, double>(StringComparer.Ordinal) { ["x"] = 40 });

        Assert.True(archive.Version > before);
    }

    [Fact]
    public void RemeasuringAnEmptyArchiveDoesNothing()
    {
        var archive = new MapElitesArchive<TestGenome>(new[]
        {
            new EvolutionDescriptorDefinition("x", 0, 100, 10, EvolutionOutOfRangePolicy.Grow)
        });

        Assert.Equal(0, archive.Remeasure(_ => throw new InvalidOperationException("should not be called")));
    }

    [Fact]
    public void ANewReadingOutsideTheGridWidensItRatherThanLosingTheElite()
    {
        MapElitesArchive<TestGenome> archive = Archive(Entry(1, quality: 10, descriptor: 10));

        archive.Remeasure(_ => new Dictionary<string, double>(StringComparer.Ordinal) { ["x"] = 500 });

        EvolutionArchiveEntry<TestGenome> entry = Assert.Single(archive.Entries);
        Assert.Equal(500, entry.Evaluation.Descriptors["x"], 10);
        Assert.True(archive.Descriptors[0].Maximum >= 500);
    }

    [Fact]
    public void ARemeasuredEvaluationKeepsEverythingButItsDescriptors()
    {
        // The copy is the same measurement re-expressed against a different reference, not a new measurement.
        (EvolutionCandidate<TestGenome> candidate, EvolutionEvaluation evaluation) = Entry(1, quality: 10, descriptor: 10);
        EvolutionEvaluation moved = evaluation.WithDescriptors(
            new Dictionary<string, double>(StringComparer.Ordinal) { ["x"] = 40 });

        Assert.Equal(evaluation.EvaluationId, moved.EvaluationId);
        Assert.Equal(evaluation.GenomeId, moved.GenomeId);
        Assert.Equal(evaluation.Quality, moved.Quality);
        Assert.Equal(evaluation.TaskVersionHash, moved.TaskVersionHash);
        Assert.Equal(evaluation.EvaluatorVersionHash, moved.EvaluatorVersionHash);
        Assert.Equal(evaluation.Cost.CostUnits, moved.Cost.CostUnits);
        Assert.Equal(40, moved.Descriptors["x"], 10);
        Assert.Equal(10, evaluation.Descriptors["x"], 10);
        Assert.Equal(candidate.EvaluationId, moved.EvaluationId);
    }

    private static ProgramGenome Genome(string source) => new(source, ProgramLanguage.Python);

    private static MapElitesArchive<TestGenome> Archive(params (EvolutionCandidate<TestGenome> Candidate,
        EvolutionEvaluation Evaluation)[] entries)
    {
        var archive = new MapElitesArchive<TestGenome>(new[]
        {
            new EvolutionDescriptorDefinition("x", 0, 100, 10, EvolutionOutOfRangePolicy.Grow)
        });

        foreach ((EvolutionCandidate<TestGenome> candidate, EvolutionEvaluation evaluation) in entries)
        {
            Assert.NotEqual(EvolutionArchiveInsertionResult.Rejected, archive.TryAdd(candidate, evaluation));
        }

        return archive;
    }

    private static (EvolutionCandidate<TestGenome>, EvolutionEvaluation) Entry(int value, double quality, double descriptor)
    {
        var genome = new TestGenome(value);
        string genomeId = "genome-" + value.ToString(System.Globalization.CultureInfo.InvariantCulture);
        var lineage = new EvolutionLineage(null, null, "seed", null, 0, 0, 0UL);
        var candidate = new EvolutionCandidate<TestGenome>(
            value, new EvolutionCanonicalGenome<TestGenome>(genome, genomeId), lineage);
        var evaluation = new EvolutionEvaluation(
            value,
            genomeId,
            EvolutionEvaluationStatus.Completed,
            quality,
            EvolutionOptimizationDirection.Maximize,
            new Dictionary<string, double>(StringComparer.Ordinal) { ["x"] = descriptor },
            Array.Empty<double>(),
            Array.Empty<double>(),
            new EvolutionEvaluationCost(TimeSpan.Zero, 1, 1),
            lineage,
            EvolutionCacheStatus.Miss,
            Array.Empty<EvolutionDiagnostic>(),
            "task-v1",
            "evaluator-v1",
            "config-v1");

        return (candidate, evaluation);
    }
}
