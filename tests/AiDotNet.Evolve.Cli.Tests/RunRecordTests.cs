using System.Text;
using AiDotNet.Configuration;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.ProgramSynthesis.Enums;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Evolve.Cli.Tests;

public sealed class RunRecordTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "run-record-test-" + Guid.NewGuid().ToString("N"));
    private string Destination(string name) => Path.Combine(_root, name);

    private static YamlModelConfig Configuration() => JsonConvert.DeserializeObject<YamlModelConfig>("""
        {"Evolution":{"RunId":"private-run-name","Seed":7,"MaxEvaluationAttempts":1},
         "ProgramEvolution":{"Language":"Python"},
         "ChatClient":{"Type":"ManualChatClient","Params":{"apiKey":"configured-private-token","queueDirectory":"private-queue-path"}}}
        """)!;

    private static ProgramEvolutionResult Winner(string source = "print(7)") => new(
        EvolutionStopReason.EvaluationBudgetReached, "state", new EvolutionRunCounters(1, 1, 1,
            new Dictionary<EvolutionEvaluationStatus, long> { [EvolutionEvaluationStatus.Completed] = 1 }),
        EvolutionOptimizationDirection.Maximize, new ProgramGenome(source, ProgramLanguage.Python), 1, null, null, 1, 1);

    private static RunInspection.Snapshot Snapshot(ProgramEvolutionResult? winner = null)
    {
        using var cancellation = new CancellationTokenSource();
        var snapshot = new RunInspection(new EvolutionRunControl(), cancellation).Read();
        return snapshot with
        {
            State = winner is null ? "aborted" : "stopped", Finished = true,
            BestFeasible = winner is null ? null : new RunInspection.BestSnapshot(winner.BestProgram!.Id, 1,
                "Maximize", new string('a', 64), new string('b', 64), new string('c', 64))
        };
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SourceRequiresOptInAndProviderParametersAndPrivatePathsAreNotExported(bool includeSource)
    {
        string target = Destination("record");
        var winner = Winner();
        using (var record = new RunRecord(target, Configuration(), includeSource))
            record.Complete(null, null, winner, Snapshot(winner), 0);
        var files = RunEvidenceBundle.ReadAndVerify(target);
        Assert.Equal(includeSource, files.ContainsKey("program.txt"));
        foreach (byte[] bytes in files.Values)
        {
            string text = Encoding.UTF8.GetString(bytes);
            Assert.DoesNotContain("configured-private-token", text);
            Assert.DoesNotContain("private-queue-path", text);
            Assert.DoesNotContain("private-run-name", text);
        }
        var environment = JObject.Parse(Encoding.UTF8.GetString(files["environment.json"]));
        Assert.Equal(3, ((JArray)environment["Binaries"]!).Count);
        Assert.All((JArray)environment["Binaries"]!, binary => Assert.Equal(64, binary.Value<string>("Sha256")!.Length));
        using var output = new StringWriter();
        Assert.Equal(0, EvidenceCommands.Inspect(target, output));
        Assert.Contains("\"AuthenticityVerified\": false", output.ToString());
    }

    [Fact]
    public void ConfiguredSecretInExactSourceRefusesPublicationWithoutRewritingSource()
    {
        string target = Destination("secret");
        var winner = Winner("print('configured-private-token')");
        using var record = new RunRecord(target, Configuration(), true);
        Assert.Throws<InvalidDataException>(() => record.Complete(null, null, winner, Snapshot(winner), 0));
        Assert.False(Directory.Exists(target));
        Assert.Equal("print('configured-private-token')", winner.BestProgram!.Source);
    }

    [Fact]
    public void WinnerMustMatchTheArchiveReceipt()
    {
        using var record = new RunRecord(Destination("mismatch"), Configuration(), true);
        var winner = Winner();
        Assert.Throws<InvalidDataException>(() => record.Complete(null, null, winner, Snapshot(Winner("print(6)")), 0));
        Assert.False(Directory.Exists(Destination("mismatch")));
    }

    [Fact]
    public void ActiveRecordDestinationsAreExclusiveAndPublishedRecordsCannotBeReplaced()
    {
        string target = Destination("exclusive");
        using (var record = new RunRecord(target, Configuration(), false))
        {
            Assert.Throws<IOException>(() => new RunRecord(target, Configuration(), false));
            record.Complete(null, null, null, Snapshot(), 3);
        }
        Assert.Throws<IOException>(() => new RunRecord(target, Configuration(), false));
        Assert.False(File.Exists(Path.Combine(target, "program.txt")));
        Assert.Equal(3, JObject.Parse(File.ReadAllText(Path.Combine(target, "result.json"))).Value<int>("ExitCode"));
    }

    [Fact]
    public async Task CliExportsAndComparesVerifiedRecordsWithBothExactSources()
    {
        var winner = Winner();
        string original = Destination("original"), exported = Destination("exported"), comparison = Destination("comparison");
        using (var record = new RunRecord(original, Configuration(), true))
            record.Complete(null, null, winner, Snapshot(winner), 0);
        using var output = new StringWriter();
        using var error = new StringWriter();
        Assert.Equal(0, await EvolveCommandLine.ExecuteAsync(new[] { "export", "--record", original, "--out", exported }, output, error));
        Assert.Equal(0, await EvolveCommandLine.ExecuteAsync(new[] { "compare", "--left", original, "--right", exported, "--out", comparison }, output, error));
        foreach (string child in new[] { "left", "right" })
            Assert.Equal(Encoding.UTF8.GetBytes("print(7)"), RunEvidenceBundle.ReadAndVerify(Path.Combine(comparison, child))["program.txt"]);
        Assert.Contains("no significance, causal or competitor-superiority claim", output.ToString());
        Assert.Empty(error.ToString());
        Assert.Equal(1, await EvolveCommandLine.ExecuteAsync(new[] { "export", "--record", original, "--out", exported }, output, error));
    }

    [Fact]
    public void ARehashedBundleWithAMismatchedSourceStillFailsReceiptValidation()
    {
        var winner = Winner();
        string original = Destination("original");
        using (var record = new RunRecord(original, Configuration(), true))
            record.Complete(null, null, winner, Snapshot(winner), 0);
        var files = RunEvidenceBundle.ReadAndVerify(original).ToDictionary(pair => pair.Key, pair => pair.Value);
        files["program.txt"] = Encoding.UTF8.GetBytes("print(6)");
        RunEvidenceBundle.Create(Destination("forged"), files, Array.Empty<string>());
        using var output = new StringWriter();
        Assert.Throws<InvalidDataException>(() => EvidenceCommands.Inspect(Destination("forged"), output));
        Assert.Empty(output.ToString());
    }

    [Fact]
    public void DirectoryLeasesRejectAliasesAndReleaseAlreadyAcquiredRootsAfterFailure()
    {
        string first = Destination("a"), second = Destination("b");
        using (RunDirectoryLease.Acquire(new[] { second }))
        {
            Assert.Throws<IOException>(() => RunDirectoryLease.Acquire(new[] { first, Path.Combine(second, ".") }));
            using var available = RunDirectoryLease.Acquire(new[] { first });
        }
        using var released = RunDirectoryLease.Acquire(new[] { second });
    }

    [Fact]
    public async Task SourceFlagCannotSilentlyEnableWithoutRecording()
    {
        using var output = new StringWriter();
        using var error = new StringWriter();
        Assert.Equal(1, await EvolveCommandLine.ExecuteAsync(new[] { "run", "--include-source" }, output, error));
        Assert.Contains("--include-source requires --record", error.ToString());
        error.GetStringBuilder().Clear();
        Assert.Equal(1, await EvolveCommandLine.ExecuteAsync(new[] { "run", "--include-source", "false" }, output, error));
        Assert.Contains("does not accept a value", error.ToString());
    }

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true); // Only this instance's GUID-owned directory.
    }
}
