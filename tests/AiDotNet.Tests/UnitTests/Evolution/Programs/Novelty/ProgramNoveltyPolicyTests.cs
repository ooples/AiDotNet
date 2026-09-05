using AiDotNet.Agentic.Embeddings;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Evolution.Programs;
using AiDotNet.Evolution.Programs.Novelty;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution.Programs.Novelty;

public sealed class ProgramNoveltyPolicyTests
{
    private static ProgramGenome Genome(string source) => new(source);

    [Fact]
    public async Task AnEmptyKnownSetAdmitsTheCandidateForFree()
    {
        var policy = new ProgramNoveltyPolicy();

        ProgramNoveltyDecision decision = await policy.EvaluateAsync(
            Genome("print(1)"), Array.Empty<ProgramGenome>());

        Assert.True(decision.IsNovel);
        Assert.Equal(ProgramNoveltyStage.None, decision.DecidedBy);
        Assert.True(decision.WasFree);
        Assert.Equal(0, decision.StructuralComparisons);
    }

    [Fact]
    public async Task ADistinctCandidateIsAdmittedWithoutAnyEmbeddingRequestOrModelCall()
    {
        // Both paid rungs are configured and instrumented; the point is that neither is reached.
        var provider = new DeterministicEmbeddingClient(dimensions: 32);
        var judge = new ScriptedNoveltyJudge(ProgramNoveltyVerdict.NotNovel);
        var policy = new ProgramNoveltyPolicy(embeddingClient: provider, judge: judge);

        var known = new[]
        {
            Genome("def add(a, b):\n    return a + b\n"),
            Genome("def mul(a, b):\n    return a * b\n")
        };

        ProgramNoveltyDecision decision = await policy.EvaluateAsync(
            Genome("import socket\nserver = socket.socket()\nserver.listen()\n"), known);

        Assert.True(decision.IsNovel);
        Assert.Equal(ProgramNoveltyStage.Structural, decision.DecidedBy);
        Assert.True(decision.WasFree);
        Assert.Equal(2, decision.StructuralComparisons);
        Assert.Equal(0, decision.EmbeddingRequests);
        Assert.Equal(0, decision.JudgeRequests);

        // The exceed, asserted at the source: upstream would have issued one embedding request here, because its
        // gate begins by embedding the candidate before it compares anything.
        Assert.Equal(0L, provider.Calls);
        Assert.Equal(0, judge.Calls);
        Assert.Equal(1L, policy.FreeDecisions);
        Assert.Equal(0L, policy.EmbeddingRequests);
        Assert.Equal(0L, policy.JudgeRequests);
    }

    [Fact]
    public async Task WithNoProviderAndNoJudgeANearDuplicateIsRejectedStructurallyAndForFree()
    {
        var policy = new ProgramNoveltyPolicy();
        var known = new[] { Genome("def add(a, b):\n    return a + b\n") };

        ProgramNoveltyDecision decision = await policy.EvaluateAsync(
            Genome("def add(a, b):\n    return a + b\n\n"), known);

        Assert.False(decision.IsNovel);
        Assert.Equal(ProgramNoveltyStage.Structural, decision.DecidedBy);
        Assert.True(decision.WasFree);
        Assert.Equal(known[0].Id, decision.NearestGenomeId);
        Assert.Equal(0.0, decision.NearestStructuralDistance ?? -1.0, 10);

        // Usable with nothing configured at all, which is precisely what upstream's gate is not.
        Assert.False(policy.HasEmbeddingStage);
        Assert.False(policy.HasJudgeStage);
    }

    [Fact]
    public async Task AStructuralNearDuplicateWhoseEmbeddingsDifferIsAdmittedByTheEmbeddingRung()
    {
        var provider = new OrthogonalEmbeddingClient();
        var policy = new ProgramNoveltyPolicy(embeddingClient: provider);
        var known = new[] { Genome("total = 0\nfor x in xs:\n    total += x\n") };

        ProgramNoveltyDecision decision = await policy.EvaluateAsync(
            Genome("total = 0\nfor x in xs:\n    total += x\n#\n"), known);

        Assert.True(decision.IsNovel);
        Assert.Equal(ProgramNoveltyStage.Embedding, decision.DecidedBy);
        Assert.Equal(1, decision.EmbeddingRequests);
        Assert.Equal(0, decision.JudgeRequests);
        Assert.NotNull(decision.EmbeddingSimilarity);
        Assert.Equal(0.0, decision.EmbeddingSimilarity ?? -1, 10);
    }

    [Fact]
    public async Task AStructuralAndSemanticDuplicateIsRejectedByTheEmbeddingRungWhenNoJudgeIsConfigured()
    {
        var provider = new ConstantEmbeddingClient();
        var policy = new ProgramNoveltyPolicy(embeddingClient: provider);
        var known = new[] { Genome("total = 0\nfor x in xs:\n    total += x\n") };

        ProgramNoveltyDecision decision = await policy.EvaluateAsync(
            Genome("total = 0\nfor x in xs:\n    total += x\n#\n"), known);

        Assert.False(decision.IsNovel);
        Assert.Equal(ProgramNoveltyStage.Embedding, decision.DecidedBy);
        Assert.Equal(1.0, decision.EmbeddingSimilarity ?? 0, 10);
        Assert.Equal(1, decision.EmbeddingRequests);
    }

    [Fact]
    public async Task TheJudgeIsConsultedOnlyAfterBothCheaperRungsAreInconclusive()
    {
        var provider = new ConstantEmbeddingClient();
        var judge = new ScriptedNoveltyJudge(ProgramNoveltyVerdict.Novel);
        var policy = new ProgramNoveltyPolicy(embeddingClient: provider, judge: judge);
        var known = new[] { Genome("total = 0\nfor x in xs:\n    total += x\n") };

        ProgramNoveltyDecision decision = await policy.EvaluateAsync(
            Genome("total = 0\nfor x in xs:\n    total += x\n#\n"), known);

        Assert.True(decision.IsNovel);
        Assert.Equal(ProgramNoveltyStage.LanguageModel, decision.DecidedBy);
        Assert.Equal(1, decision.JudgeRequests);
        Assert.Equal(1, judge.Calls);
        Assert.Equal(1, provider.Calls);
    }

    [Fact]
    public async Task AJudgeThatRejectsTurnsTheCandidateAway()
    {
        var judge = new ScriptedNoveltyJudge(ProgramNoveltyVerdict.NotNovel);
        var policy = new ProgramNoveltyPolicy(judge: judge);
        var known = new[] { Genome("def add(a, b):\n    return a + b\n") };

        ProgramNoveltyDecision decision = await policy.EvaluateAsync(
            Genome("def add(a, b):\n    return  a + b\n"), known);

        Assert.False(decision.IsNovel);
        Assert.Equal(ProgramNoveltyStage.LanguageModel, decision.DecidedBy);
        Assert.Equal(0, decision.EmbeddingRequests);
        Assert.Equal(1, decision.JudgeRequests);
    }

    [Theory]
    [InlineData(true, true)]
    [InlineData(false, false)]
    public async Task AnUnreachableProviderResolvesThroughTheConfiguredFailurePolicy(bool failOpen, bool expected)
    {
        var provider = new UnavailableEmbeddingClient();
        var options = new EmbeddingNoveltyOptions(failOpenOnEmbeddingFailure: failOpen);
        var policy = new ProgramNoveltyPolicy(options, embeddingClient: provider);
        var known = new[] { Genome("def add(a, b):\n    return a + b\n") };

        ProgramNoveltyDecision decision = await policy.EvaluateAsync(
            Genome("def add(a, b):\n    return a + b\n#\n"), known);

        Assert.Equal(expected, decision.IsNovel);
        Assert.Equal(ProgramNoveltyStage.Embedding, decision.DecidedBy);
        Assert.Equal(1, decision.EmbeddingRequests);
        Assert.Equal(1, provider.Calls);
    }

    [Theory]
    [InlineData(true, true)]
    [InlineData(false, false)]
    public async Task AnUnusableJudgementResolvesThroughTheConfiguredFailurePolicy(bool failOpen, bool expected)
    {
        var judge = new ScriptedNoveltyJudge(ProgramNoveltyVerdict.Unavailable);
        var options = new EmbeddingNoveltyOptions(failOpenOnJudgeFailure: failOpen);
        var policy = new ProgramNoveltyPolicy(options, judge: judge);
        var known = new[] { Genome("def add(a, b):\n    return a + b\n") };

        ProgramNoveltyDecision decision = await policy.EvaluateAsync(
            Genome("def add(a, b):\n    return a + b\n#\n"), known);

        Assert.Equal(expected, decision.IsNovel);
        Assert.Equal(ProgramNoveltyStage.LanguageModel, decision.DecidedBy);
    }

    [Fact]
    public async Task TheEmbeddingRungComparesAtMostTheConfiguredNumberOfNeighbours()
    {
        var provider = new DeterministicEmbeddingClient(dimensions: 32);
        var caching = new CachingEmbeddingClient(provider);
        var options = new EmbeddingNoveltyOptions(maxEmbeddingComparisons: 2);
        var policy = new ProgramNoveltyPolicy(options, embeddingClient: caching);

        var known = new List<ProgramGenome>();
        for (int index = 0; index < 10; index++) known.Add(Genome("def add(a, b):\n    return a + b\n" + new string('#', index + 1)));

        await policy.EvaluateAsync(Genome("def add(a, b):\n    return a + b\n"), known);

        // One candidate plus two neighbours, never the whole island as upstream scans.
        Assert.Equal(3L, provider.TextsEmbedded);
        Assert.Equal(1L, caching.InnerCalls);
    }

    [Fact]
    public async Task TheSameInputsAlwaysProduceTheSameDecision()
    {
        var known = new[]
        {
            Genome("def add(a, b):\n    return a + b\n"),
            Genome("def add(x, y):\n    return x + y\n")
        };
        var candidate = Genome("def add(a, b):\n    return b + a\n");

        for (int repeat = 0; repeat < 5; repeat++)
        {
            var policy = new ProgramNoveltyPolicy();
            ProgramNoveltyDecision decision = await policy.EvaluateAsync(candidate, known);
            Assert.Equal(known[0].Id, decision.NearestGenomeId);
            Assert.Equal(0.0, decision.NearestStructuralDistance ?? -1.0, 10);
            Assert.False(decision.IsNovel);
        }
    }

    [Fact]
    public async Task ANullEntryInTheKnownSetIsARejectedArgument()
    {
        var policy = new ProgramNoveltyPolicy();
#pragma warning disable CS8625
        await Assert.ThrowsAsync<ArgumentNullException>(async () =>
            await policy.EvaluateAsync(null, Array.Empty<ProgramGenome>()));
        await Assert.ThrowsAsync<ArgumentException>(async () =>
            await policy.EvaluateAsync(Genome("a"), new ProgramGenome[] { null }));
#pragma warning restore CS8625
    }

    [Fact]
    public void ThePolicyVersionChangesWithEveryConfiguredRung()
    {
        string bare = new ProgramNoveltyPolicy().VersionHash;
        string withEmbedding = new ProgramNoveltyPolicy(
            embeddingClient: new DeterministicEmbeddingClient()).VersionHash;
        string withJudge = new ProgramNoveltyPolicy(
            judge: new ScriptedNoveltyJudge()).VersionHash;
        string withDifferentThresholds = new ProgramNoveltyPolicy(
            new EmbeddingNoveltyOptions(structuralNoveltyThreshold: 0.4)).VersionHash;

        Assert.NotEqual(bare, withEmbedding);
        Assert.NotEqual(bare, withJudge);
        Assert.NotEqual(bare, withDifferentThresholds);
        Assert.Equal(bare, new ProgramNoveltyPolicy().VersionHash);
    }

    [Theory]
    [InlineData(-0.1)]
    [InlineData(1.1)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void OptionsRejectAThresholdOutsideTheUnitInterval(double value)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new EmbeddingNoveltyOptions(structuralNoveltyThreshold: value));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new EmbeddingNoveltyOptions(embeddingSimilarityThreshold: value));
    }

    [Fact]
    public void OptionsRejectInvalidCountsAndKeepTheirDefaults()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new EmbeddingNoveltyOptions(maxEmbeddingComparisons: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new EmbeddingNoveltyOptions(maxEmbeddingComparisons: 257));
        Assert.Throws<ArgumentOutOfRangeException>(() => new EmbeddingNoveltyOptions(maxTrackedGenomes: 0));

        var options = new EmbeddingNoveltyOptions();
        Assert.Equal(EmbeddingNoveltyOptions.DefaultStructuralNoveltyThreshold, options.StructuralNoveltyThreshold);

        // Matches the reference implementation's similarity_threshold default of 0.99.
        Assert.Equal(0.99, options.EmbeddingSimilarityThreshold);
        Assert.True(options.FailOpenOnEmbeddingFailure);
        Assert.True(options.FailOpenOnJudgeFailure);
    }
}
