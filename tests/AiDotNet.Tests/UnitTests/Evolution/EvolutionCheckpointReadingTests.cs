using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Interfaces;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution;

/// <summary>
/// Covers reading a checkpoint back without resuming: recovering every candidate it holds, and rebuilding the chain
/// of ancestors that produced one of them. A checkpoint is the only complete record a finished run leaves, so being
/// able to open it is what makes a run auditable after the fact rather than only continuable.
/// </summary>
public sealed class EvolutionCheckpointReadingTests
{
    [Fact]
    public async Task ACheckpointCanBeReadBackAsTheCandidatesTheRunHeld()
    {
        var store = new InMemoryEvolutionCheckpointStore();
        EvolutionRunResult<TestGenome> run = await Engine(store).RunAsync(Seeds(4));
        EvolutionCheckpoint checkpoint = Assert.IsType<EvolutionCheckpoint>(await store.LoadLatestAsync("read-run"));

        EvolutionCheckpointContents<TestGenome> contents =
            EvolutionEngine<TestGenome>.ReadCheckpoint(checkpoint, new TestGenomeCodec());

        Assert.Equal("read-run", contents.RunId);
        Assert.Equal(checkpoint.CompatibilityHash, contents.CompatibilityHash);
        Assert.NotEmpty(contents.Entries);

        // Every elite the run finished with is recoverable, genome and score included, with no engine involved.
        string[] archived = run.Islands.SelectMany(island => island.Entries)
            .Select(entry => entry.Evaluation.GenomeId).OrderBy(id => id, StringComparer.Ordinal).ToArray();
        string[] recovered = contents.Entries
            .Where(entry => entry.Source == EvolutionCheckpointEntrySource.IslandArchive)
            .Select(entry => entry.GenomeId).OrderBy(id => id, StringComparer.Ordinal).ToArray();
        Assert.Equal(archived, recovered);

        EvolutionCheckpointEntry<TestGenome> sample = contents.Entries[0];
        Assert.Equal(sample.GenomeId, sample.Entry.Evaluation.GenomeId);
        Assert.Equal(EvolutionEvaluationStatus.Completed, sample.Entry.Evaluation.Status);
        Assert.NotNull(sample.Entry.Evaluation.Quality);
        Assert.Equal(sample.Genome.Value.ToString(System.Globalization.CultureInfo.InvariantCulture), sample.GenomeId);
    }

    [Fact]
    public async Task AnAncestryChainIsRebuiltFromTheCheckpointAndSaysWhenItIsPartial()
    {
        var store = new InMemoryEvolutionCheckpointStore();
        await Engine(store).RunAsync(Seeds(4));
        EvolutionCheckpoint checkpoint = Assert.IsType<EvolutionCheckpoint>(await store.LoadLatestAsync("read-run"));
        EvolutionCheckpointContents<TestGenome> contents =
            EvolutionEngine<TestGenome>.ReadCheckpoint(checkpoint, new TestGenomeCodec());

        // The variation operator only increments, so the deepest candidate has the longest chain behind it.
        EvolutionCheckpointEntry<TestGenome> deepest = contents.DistinctCandidates
            .OrderByDescending(entry => entry.Entry.Evaluation.Lineage.Generation)
            .ThenBy(entry => entry.GenomeId, StringComparer.Ordinal)
            .First();

        bool complete = contents.TryGetAncestry(deepest.GenomeId, out IReadOnlyList<EvolutionCheckpointEntry<TestGenome>> chain);

        Assert.NotEmpty(chain);
        Assert.Equal(deepest.GenomeId, chain[chain.Count - 1].GenomeId);
        Assert.Equal(chain.Count, chain.Select(entry => entry.GenomeId).Distinct(StringComparer.Ordinal).Count());
        for (int index = 1; index < chain.Count; index++)
        {
            // Each link is the recorded parent of the next, which is what makes this ancestry rather than a listing.
            Assert.Equal(chain[index - 1].GenomeId, chain[index].Entry.Evaluation.Lineage.ParentIds[0]);
        }

        // A complete chain ends at a seed, which has no parent; a partial one stopped at a candidate the bounded
        // archive no longer holds. Both are legitimate, and the flag is what tells them apart.
        if (complete) Assert.Empty(chain[0].Entry.Evaluation.Lineage.ParentIds);
        else Assert.NotEmpty(chain[0].Entry.Evaluation.Lineage.ParentIds);

        Assert.Null(contents.Find("no-such-genome"));
        Assert.False(contents.TryGetAncestry("no-such-genome", out IReadOnlyList<EvolutionCheckpointEntry<TestGenome>> missing));
        Assert.Empty(missing);
    }

    [Fact]
    public async Task AReadRefusesACheckpointItCannotDecode()
    {
        var store = new InMemoryEvolutionCheckpointStore();
        await Engine(store).RunAsync(Seeds(4));
        EvolutionCheckpoint checkpoint = Assert.IsType<EvolutionCheckpoint>(await store.LoadLatestAsync("read-run"));

        Assert.Throws<InvalidDataException>(() =>
            EvolutionEngine<TestGenome>.ReadCheckpoint(checkpoint, new RefusingGenomeCodec()));

        var corrupt = new EvolutionCheckpoint(checkpoint.RunId, checkpoint.Sequence, checkpoint.CompatibilityHash,
            "{ this is not engine state ");
        Assert.Throws<InvalidDataException>(() =>
            EvolutionEngine<TestGenome>.ReadCheckpoint(corrupt, new TestGenomeCodec()));

        Assert.Throws<ArgumentNullException>(() =>
            EvolutionEngine<TestGenome>.ReadCheckpoint(checkpoint, null!));
    }

    private static TestGenome[] Seeds(int count) =>
        Enumerable.Range(1, count).Select(value => new TestGenome(value)).ToArray();

    private static EvolutionEngine<TestGenome> Engine(IEvolutionCheckpointStore store) => new(
        new SyntheticEvolutionTask(), new IncrementVariation(), _ => new MapElitesArchive<TestGenome>(new[]
        {
            new EvolutionDescriptorDefinition("x", 0, 100, 20, EvolutionOutOfRangePolicy.Clamp)
        }), new EvolutionEngineOptions
        {
            RunId = "read-run",
            Seed = 17,
            MaxEvaluationAttempts = 16,
            MaxProposals = 100,
            MaxGenerations = 100,
            ProposalBatchSize = 2,
            MaxDegreeOfParallelism = 1,
            IslandCount = 1,
            MigrationInterval = 0,
            MigrantsPerIsland = 1,
            CheckpointInterval = 0,
            HistorySize = 8,
            GlobalEliteCount = 4
        }, checkpointStore: store, genomeCodec: new TestGenomeCodec());

    /// <summary>A codec that cannot read anything, standing in for a checkpoint written by different code.</summary>
    private sealed class RefusingGenomeCodec : IEvolutionGenomeCodec<TestGenome>
    {
        public string Id => "refusing";
        public string VersionHash => "refusing-v1";
        public string Serialize(TestGenome genome) => string.Empty;
        public TestGenome Deserialize(string payload) => throw new FormatException("unreadable payload");
    }
}
