using System.Linq;
using System.Text;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.Evolution.Programs.Provenance;
using AiDotNet.Evolution.Prompts;
using AiDotNet.Interfaces;
using AiDotNet.ProgramSynthesis.Enums;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution.Programs;

/// <summary>
/// Covers per-candidate provenance: the record, the crash-safe JSON Lines sink, the tolerant reader, lineage
/// reconstruction, and the variation operator's wiring. Nothing here touches a network or a real model.
/// </summary>
public sealed class ProposalProvenanceTests : IDisposable
{
    private const string ParentSource = "def solve(x):\n    return x\n";

    // DateTimeOffset.UnixEpoch does not exist on .NET Framework, which this test project also targets.
    private static readonly DateTimeOffset Epoch = new(1970, 1, 1, 0, 0, 0, TimeSpan.Zero);

    private readonly string _directory = Path.Combine(
        Path.GetTempPath(), "aidotnet-provenance-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_directory)) Directory.Delete(_directory, recursive: true);
        }
        catch (IOException)
        {
            // A leftover temp directory must not fail a test run.
        }
    }

    private static ProposalProvenanceRecord Record(
        string proposalId,
        long evaluationId,
        string parentId,
        string childId,
        ProgramProposalOutcome outcome = ProgramProposalOutcome.Accepted,
        int attempt = 1,
        DateTimeOffset requestedAt = default) =>
        new(proposalId, evaluationId, parentId, attempt, outcome)
        {
            ChildGenomeId = childId,
            ModelId = "fake-model",
            OperatorId = "llm-program-variation",
            PromptHash = "hash-" + proposalId,
            PromptText = "### USER\nimprove\n",
            ResponseText = "answer",
            InputTokens = 11,
            OutputTokens = 7,
            RequestedAtUtc = requestedAt == default ? Epoch.AddSeconds(evaluationId) : requestedAt,
            LatencyMilliseconds = 12.5,
            Detail = outcome == ProgramProposalOutcome.Accepted ? string.Empty : "unusable"
        };

    private static EvolutionArchiveEntry<ProgramGenome> Entry(ProgramGenome genome, long evaluationId = 0)
    {
        var lineage = new EvolutionLineage(new[] { "grandparent-1" }, null, "seed", null, 0, 0, 0UL);
        var candidate = new EvolutionCandidate<ProgramGenome>(
            evaluationId, new EvolutionCanonicalGenome<ProgramGenome>(genome, genome.Id), lineage);
        var evaluation = new EvolutionEvaluation(
            evaluationId,
            genome.Id,
            EvolutionEvaluationStatus.Completed,
            0.5,
            EvolutionOptimizationDirection.Maximize,
            new Dictionary<string, double>(StringComparer.Ordinal) { ["x"] = 0.5 },
            Array.Empty<double>(),
            Array.Empty<double>(),
            new EvolutionEvaluationCost(TimeSpan.Zero, 1, 1),
            lineage,
            EvolutionCacheStatus.Miss,
            Array.Empty<EvolutionDiagnostic>(),
            "task-v1",
            "evaluator-v1",
            "config-v1");

        return new EvolutionArchiveEntry<ProgramGenome>(new EvolutionCellKey(new[] { 1, 2 }), candidate, evaluation);
    }

    private static EvolutionVariationContext<ProgramGenome> Context(ProgramGenome? parent = null) =>
        new(Entry(parent ?? new ProgramGenome(ParentSource, ProgramLanguage.Python), 42),
            Array.Empty<EvolutionArchiveEntry<ProgramGenome>>(),
            new StableRandom(1234UL, 7UL),
            3,
            1);

    private static string DiffResponse(string search, string replace) =>
        "<<<<<<< SEARCH\n" + search + "\n=======\n" + replace + "\n>>>>>>> REPLACE\n";

    [Fact]
    public async Task RecordsRoundTripThroughTheJsonLinesSinkAndReader()
    {
        using (var sink = new JsonLinesProposalProvenanceSink(
                   _directory, new ProposalProvenanceOptions { FlushEveryRecords = 2 }))
        {
            await sink.RecordAsync(Record("p1", 1, "a", "b"));
            await sink.RecordAsync(Record("p2", 2, "b", "c"));
            await sink.RecordAsync(Record("p3", 3, "c", "d"));
        }

        ProposalProvenanceReadResult result = ProposalProvenanceReader.Read(_directory);

        Assert.True(result.IsComplete);
        Assert.Equal(3, result.Records.Count);
        Assert.Equal(new[] { "p1", "p2", "p3" }, result.Records.Select(r => r.ProposalId).ToArray());

        ProposalProvenanceRecord first = result.Records[0];
        Assert.Equal(1L, first.EvaluationId);
        Assert.Equal("a", first.ParentGenomeId);
        Assert.Equal("b", first.ChildGenomeId);
        Assert.Equal("fake-model", first.ModelId);
        Assert.Equal("hash-p1", first.PromptHash);
        Assert.Equal(11, first.InputTokens);
        Assert.Equal(7, first.OutputTokens);
        Assert.Equal(12.5, first.LatencyMilliseconds);
        Assert.Equal(ProgramProposalOutcome.Accepted, first.Outcome);
        Assert.Equal(Epoch.AddSeconds(1), first.RequestedAtUtc);
    }

    [Fact]
    public async Task SegmentsAreWrittenAtomicallyAndLeaveNoTemporaryFiles()
    {
        var sink = new JsonLinesProposalProvenanceSink(
            _directory, new ProposalProvenanceOptions { FlushEveryRecords = 1 });
        try
        {
            await sink.RecordAsync(Record("p1", 1, "a", "b"));
            await sink.RecordAsync(Record("p2", 2, "b", "c"));
        }
        finally
        {
            sink.Dispose();
        }

        Assert.Equal(2, sink.SegmentsWritten);
        Assert.Equal(2L, sink.RecordsWritten);
        Assert.Empty(Directory.GetFiles(_directory, "*.tmp"));
        Assert.Equal(2, Directory.GetFiles(_directory, "*.jsonl").Length);
    }

    [Fact]
    public async Task ASecondSinkResumesNumberingInsteadOfOverwriting()
    {
        using (var first = new JsonLinesProposalProvenanceSink(
                   _directory, new ProposalProvenanceOptions { FlushEveryRecords = 1 }))
        {
            await first.RecordAsync(Record("p1", 1, "a", "b"));
        }

        using (var second = new JsonLinesProposalProvenanceSink(
                   _directory, new ProposalProvenanceOptions { FlushEveryRecords = 1 }))
        {
            await second.RecordAsync(Record("p2", 2, "b", "c"));
        }

        ProposalProvenanceReadResult result = ProposalProvenanceReader.Read(_directory);
        Assert.Equal(2, result.Records.Count);
        Assert.Equal(new[] { "p1", "p2" }, result.Records.Select(r => r.ProposalId).ToArray());
    }

    [Fact]
    public async Task ATruncatedTailLosesOnlyItsOwnLine()
    {
        using (var sink = new JsonLinesProposalProvenanceSink(
                   _directory, new ProposalProvenanceOptions { FlushEveryRecords = 2 }))
        {
            await sink.RecordAsync(Record("p1", 1, "a", "b"));
            await sink.RecordAsync(Record("p2", 2, "b", "c"));
        }

        string segment = Directory.GetFiles(_directory, "*.jsonl").Single();
        string complete = File.ReadAllText(segment);
        File.WriteAllText(segment, complete + "{\"ProposalId\":\"p3\",\"Evalu", Encoding.UTF8);

        ProposalProvenanceReadResult result = ProposalProvenanceReader.Read(_directory);

        Assert.Equal(2, result.Records.Count);
        Assert.Equal(1, result.MalformedLineCount);
        Assert.True(result.HasIncompleteTail);
        Assert.False(result.IsComplete);
    }

    [Fact]
    public async Task AMalformedLineInTheMiddleIsCountedNotFatal()
    {
        using (var sink = new JsonLinesProposalProvenanceSink(
                   _directory, new ProposalProvenanceOptions { FlushEveryRecords = 8 }))
        {
            await sink.RecordAsync(Record("p1", 1, "a", "b"));
            await sink.RecordAsync(Record("p2", 2, "b", "c"));
        }

        string segment = Directory.GetFiles(_directory, "*.jsonl").Single();
        List<string> lines = File.ReadAllLines(segment).ToList();
        lines.Insert(1, "{ this is not json");
        File.WriteAllLines(segment, lines);

        ProposalProvenanceReadResult result = ProposalProvenanceReader.Read(_directory);

        Assert.Equal(2, result.Records.Count);
        Assert.Equal(1, result.MalformedLineCount);
        Assert.False(result.HasIncompleteTail);
    }

    [Fact]
    public void LineageRebuildsTheChainOfAcceptedEditsInOrder()
    {
        var records = new List<ProposalProvenanceRecord>
        {
            Record("p3", 3, "c", "d"),
            Record("p1", 1, "a", "b"),
            Record("pf", 2, "b", string.Empty, ProgramProposalOutcome.ParseFailed),
            Record("p2", 2, "b", "c")
        };

        IReadOnlyList<ProposalProvenanceLineage> lineages = ProposalProvenanceReader.BuildLineages(records);

        ProposalProvenanceLineage lineage = Assert.Single(lineages);
        Assert.Equal("d", lineage.FinalGenomeId);
        Assert.Equal("a", lineage.RootGenomeId);
        Assert.Equal(3, lineage.Depth);
        Assert.Equal(new[] { "a", "b", "c" }, lineage.Steps.Select(s => s.ParentGenomeId).ToArray());
        Assert.Equal(new[] { "b", "c", "d" }, lineage.Steps.Select(s => s.ChildGenomeId).ToArray());
        Assert.Equal(new[] { 0, 1, 2 }, lineage.Steps.Select(s => s.StepIndex).ToArray());
        Assert.Equal(54L, lineage.TotalTokens);

        // The failed attempt is in the stream but is not ancestry, so it is not a link in the chain.
        Assert.DoesNotContain(lineage.Steps, s => s.Record.ProposalId == "pf");
    }

    [Fact]
    public void SeparateBranchesEachProduceTheirOwnLineage()
    {
        var records = new List<ProposalProvenanceRecord>
        {
            Record("p1", 1, "a", "b"),
            Record("p2", 2, "b", "c"),
            Record("p3", 3, "b", "d")
        };

        IReadOnlyList<ProposalProvenanceLineage> lineages = ProposalProvenanceReader.BuildLineages(records);

        Assert.Equal(2, lineages.Count);
        Assert.All(lineages, l => Assert.Equal("a", l.RootGenomeId));
        Assert.Equal(new[] { "c", "d" }, lineages.Select(l => l.FinalGenomeId).OrderBy(x => x, StringComparer.Ordinal).ToArray());
    }

    [Fact]
    public void ACycleInTheStreamCannotSpinTheWalk()
    {
        var records = new List<ProposalProvenanceRecord>
        {
            Record("p1", 1, "a", "b"),
            Record("p2", 2, "b", "a")
        };

        ProposalProvenanceLineage lineage = ProposalProvenanceReader.BuildLineage("b", records);

        Assert.True(lineage.Depth <= 2);
        Assert.Equal("b", lineage.FinalGenomeId);
    }

    [Fact]
    public void RecordFieldsAreBoundedOnConstruction()
    {
        string huge = new('x', ProposalProvenanceRecord.MaxTextLength + 500);
        var record = new ProposalProvenanceRecord("p1", 0, "a", 1, ProgramProposalOutcome.Accepted)
        {
            PromptText = huge,
            ResponseText = huge,
            Detail = huge,
            InputTokens = -5,
            LatencyMilliseconds = double.NaN,
            Generation = -3
        };

        Assert.Equal(ProposalProvenanceRecord.MaxTextLength, record.PromptText.Length);
        Assert.Equal(ProposalProvenanceRecord.MaxTextLength, record.ResponseText.Length);
        Assert.Equal(ProposalProvenanceRecord.MaxDetailLength, record.Detail.Length);
        Assert.Equal(0, record.InputTokens);
        Assert.Equal(0.0, record.LatencyMilliseconds);
        Assert.Equal(0L, record.Generation);
    }

    [Fact]
    public async Task TheOperatorRecordsEveryRequestIncludingTheOnesThatFailed()
    {
        var client = new FakeChatClient(
            DiffResponse("    return y", "    return 0"),
            DiffResponse("    return x", "    return x + 1"))
        {
            Usage = new AiDotNet.Agentic.Models.ChatUsage(120, 45)
        };

        var sink = new InMemoryProposalProvenanceSink();
        var operatorUnderTest = new LlmProgramVariationOperator<double>(
            client, null, null, "llm-program-variation", null, sink);

        ProgramGenome child = await operatorUnderTest.ProposeAsync(Context());

        IReadOnlyList<ProposalProvenanceRecord> records = sink.GetRecords();
        Assert.Equal(2, records.Count);

        Assert.Equal(ProgramProposalOutcome.ParseFailed, records[0].Outcome);
        Assert.Equal(1, records[0].AttemptNumber);
        Assert.Equal(string.Empty, records[0].ChildGenomeId);

        Assert.Equal(ProgramProposalOutcome.Accepted, records[1].Outcome);
        Assert.Equal(2, records[1].AttemptNumber);
        Assert.Equal(child.Id, records[1].ChildGenomeId);
        Assert.True(records[1].IsAccepted);

        foreach (ProposalProvenanceRecord record in records)
        {
            Assert.Equal("fake-model", record.ModelId);
            Assert.Equal("llm-program-variation", record.OperatorId);
            Assert.Equal(operatorUnderTest.VersionHash, record.OperatorVersionHash);
            Assert.Equal(42L, record.EvaluationId);
            Assert.Equal(3L, record.Generation);
            Assert.Equal(1, record.Island);
            Assert.Equal(new[] { "grandparent-1" }, record.ParentIds.ToArray());
            Assert.NotEqual(string.Empty, record.PromptHash);
            Assert.NotEqual(string.Empty, record.PromptText);
            Assert.Equal(120, record.InputTokens);
            Assert.Equal(45, record.OutputTokens);
            Assert.NotEqual(default(DateTimeOffset), record.RequestedAtUtc);
        }

        // The retry sends a longer conversation, so it is provably a different request.
        Assert.NotEqual(records[0].PromptHash, records[1].PromptHash);
        Assert.Equal(records[0].ProposalId, records[1].ProposalId);
        Assert.Equal(0L, operatorUnderTest.ProvenanceFailureCount);
    }

    [Fact]
    public async Task ProposalIdentifiersAreDeterministicAcrossIdenticalRuns()
    {
        string[] first = await CaptureProposalIdsAsync();
        string[] second = await CaptureProposalIdsAsync();

        Assert.Equal(first, second);
        Assert.NotEmpty(first[0]);
    }

    private static async Task<string[]> CaptureProposalIdsAsync()
    {
        var client = new FakeChatClient(DiffResponse("    return x", "    return x * 2"));
        var sink = new InMemoryProposalProvenanceSink();
        var operatorUnderTest = new LlmProgramVariationOperator<double>(
            client, null, null, "llm-program-variation", null, sink);

        await operatorUnderTest.ProposeAsync(Context());
        return sink.GetRecords().Select(r => r.ProposalId).ToArray();
    }

    [Fact]
    public async Task CredentialShapedTextInAnAnswerIsRedactedBeforeItIsRecorded()
    {
        const string leaked = "sk-livekey0123456789abcdefghij";
        var client = new FakeChatClient(
            DiffResponse("    return x", "    return x * 2") + "\napi_key=" + leaked + "\n");

        var sink = new InMemoryProposalProvenanceSink();
        var operatorUnderTest = new LlmProgramVariationOperator<double>(
            client, null, null, "llm-program-variation", null, sink);

        await operatorUnderTest.ProposeAsync(Context());

        IReadOnlyList<ProposalProvenanceRecord> records = sink.GetRecords();
        Assert.NotEmpty(records);
        Assert.DoesNotContain(leaked, records[0].ResponseText, StringComparison.Ordinal);
        Assert.Contains(PromptTextRedactor.RedactionMarker, records[0].ResponseText, StringComparison.Ordinal);
    }

    [Fact]
    public async Task TextBudgetsTruncateAndTheRecordSaysSo()
    {
        // Ordinary prose padding, not one long alphanumeric run: the redactor treats a 40-character unbroken
        // token as an opaque secret and would replace the whole tail, leaving nothing for the budget to cut.
        string padding = string.Concat(Enumerable.Repeat("padding ", 3_000));
        var client = new FakeChatClient(
            DiffResponse("    return x", "    return x * 2") + "\n" + padding);

        var sink = new InMemoryProposalProvenanceSink();
        var operatorUnderTest = new LlmProgramVariationOperator<double>(
            client,
            null,
            null,
            "llm-program-variation",
            null,
            sink,
            new ProposalProvenanceOptions { MaxResponseBytes = 256, MaxPromptBytes = 128 });

        await operatorUnderTest.ProposeAsync(Context());

        IReadOnlyList<ProposalProvenanceRecord> records = sink.GetRecords();
        Assert.NotEmpty(records);
        Assert.True(records[0].ResponseTruncated);
        Assert.True(records[0].PromptTruncated);
        Assert.True(Encoding.UTF8.GetByteCount(records[0].ResponseText) <= 256);
        Assert.True(Encoding.UTF8.GetByteCount(records[0].PromptText) <= 128);
    }

    [Fact]
    public async Task TurningTextOffKeepsTheIdentityAndTheCost()
    {
        var client = new FakeChatClient(DiffResponse("    return x", "    return x * 2"));
        var sink = new InMemoryProposalProvenanceSink();
        var operatorUnderTest = new LlmProgramVariationOperator<double>(
            client,
            null,
            null,
            "llm-program-variation",
            null,
            sink,
            new ProposalProvenanceOptions { IncludePromptText = false, IncludeResponseText = false });

        await operatorUnderTest.ProposeAsync(Context());

        ProposalProvenanceRecord record = Assert.Single(sink.GetRecords());
        Assert.Equal(string.Empty, record.PromptText);
        Assert.Equal(string.Empty, record.ResponseText);
        Assert.NotEqual(string.Empty, record.PromptHash);
        Assert.NotEqual(string.Empty, record.ChildGenomeId);
    }

    [Fact]
    public async Task OnlyAcceptedAttemptsAreKeptWhenFailuresAreTurnedOff()
    {
        var client = new FakeChatClient(
            DiffResponse("    return y", "    return 0"),
            DiffResponse("    return x", "    return x + 1"));

        var sink = new InMemoryProposalProvenanceSink();
        var operatorUnderTest = new LlmProgramVariationOperator<double>(
            client,
            null,
            null,
            "llm-program-variation",
            null,
            sink,
            new ProposalProvenanceOptions { RecordFailedAttempts = false });

        await operatorUnderTest.ProposeAsync(Context());

        ProposalProvenanceRecord record = Assert.Single(sink.GetRecords());
        Assert.Equal(ProgramProposalOutcome.Accepted, record.Outcome);
    }

    [Fact]
    public async Task ProvenanceIsSkippedEntirelyWhenItIsDisabled()
    {
        var client = new FakeChatClient(DiffResponse("    return x", "    return x * 2"));
        var sink = new InMemoryProposalProvenanceSink();
        var operatorUnderTest = new LlmProgramVariationOperator<double>(
            client, null, null, "llm-program-variation", null, sink,
            new ProposalProvenanceOptions { Enabled = false });

        ProgramGenome child = await operatorUnderTest.ProposeAsync(Context());

        Assert.Empty(sink.GetRecords());
        Assert.Equal("def solve(x):\n    return x * 2\n", child.Source);
    }

    [Fact]
    public async Task ABrokenSinkIsCountedAndNeverEndsTheRun()
    {
        var client = new FakeChatClient(DiffResponse("    return x", "    return x * 2"));
        var operatorUnderTest = new LlmProgramVariationOperator<double>(
            client, null, null, "llm-program-variation", null, new ThrowingProvenanceSink());

        ProgramGenome child = await operatorUnderTest.ProposeAsync(Context());

        Assert.Equal("def solve(x):\n    return x * 2\n", child.Source);
        Assert.Equal(1L, operatorUnderTest.ProvenanceFailureCount);
    }

    [Fact]
    public async Task AProvenanceStreamFromARealProposalRebuildsItsOwnLineage()
    {
        var client = new FakeChatClient(DiffResponse("    return x", "    return x * 2"));
        var sink = new InMemoryProposalProvenanceSink();
        var operatorUnderTest = new LlmProgramVariationOperator<double>(
            client, null, null, "llm-program-variation", null, sink);

        var parent = new ProgramGenome(ParentSource, ProgramLanguage.Python);
        ProgramGenome child = await operatorUnderTest.ProposeAsync(Context(parent));

        IReadOnlyList<ProposalProvenanceLineage> lineages =
            ProposalProvenanceReader.BuildLineages(sink.GetRecords());

        ProposalProvenanceLineage lineage = Assert.Single(lineages);
        Assert.Equal(child.Id, lineage.FinalGenomeId);
        Assert.Equal(parent.Id, lineage.RootGenomeId);
        Assert.Equal(1, lineage.Depth);
        Assert.Contains("return x * 2", lineage.Steps[0].Record.ResponseText, StringComparison.Ordinal);
    }

    private sealed class ThrowingProvenanceSink : IProposalProvenanceSink
    {
        public Task RecordAsync(ProposalProvenanceRecord record, CancellationToken cancellationToken = default) =>
            throw new IOException("the provenance volume is full");
    }
}
