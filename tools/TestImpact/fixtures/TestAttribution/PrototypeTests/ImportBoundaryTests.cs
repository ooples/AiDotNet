using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using AiDotNet.TestImpact;
using Xunit;

namespace PrototypeTests;

[Trait("Scenario", "WorkflowProtocol")]
public sealed class ImportBoundaryTests
{
    private static T WithNull<T>(T value, params string[] path) where T : class
    {
        var options = new JsonSerializerOptions();
        options.Converters.Add(new JsonStringEnumConverter());
        JsonNode document = JsonSerializer.SerializeToNode(value, options) ?? throw new InvalidOperationException();
        JsonNode parent = document;
        foreach (string part in path[..^1]) parent = parent[part] ?? throw new InvalidOperationException();
        parent[path[^1]] = null;
        return ExecutionEvidence.ReadDocument<T>(document.ToJsonString());
    }

    [Fact]
    public void ExplicitNullWorkflowFieldsAreTypedFailuresBeforeNetworkAccess()
    {
        var request = new WorkflowImportRequest(new("owner/repo", 1, 1, new string('a', 40),
            ".github/workflows/test.yml", "tests", "proof-1-1", WorkflowCompletionScope.WholeRun, ValidationScope.FullWorkload),
            "inventory.json", "plan.json", "reports", "results.trx");
        foreach (string field in new[] { "Policy", "Inventory", "Plan", "Reports", "Trx" })
            Assert.Equal(EvidenceFailure.Format, Assert.Throws<EvidenceException>(() =>
                GitHubEvidenceReader.ValidateRequest(WithNull(request, field))).Reason);
        foreach (string field in new[] { "Repository", "HeadSha", "WorkflowPath", "JobName", "ArtifactName" })
            Assert.Equal(EvidenceFailure.Format, Assert.Throws<EvidenceException>(() =>
                GitHubEvidenceReader.ValidateRequest(WithNull(request, "Policy", field))).Reason);
        GitHubEvidenceReader.ValidateRequest(request);
    }

    [Fact]
    public void ExplicitNullReuseRecordsAndPathsAreTypedFailures()
    {
        var revision = new SourceRevisionInput("source.json", "inventory.json", "bundle");
        var request = new SourceReuseRequest("repository", revision, revision,
            new("plan.json", "reports", "results.trx", "run", new("owner/repo", 1, 1)));
        foreach (string field in new[] { "Repository", "Before", "After", "Baseline" })
            Assert.Equal(EvidenceFailure.Format, Assert.Throws<EvidenceException>(() =>
                SourceReuseCommands.Validate(WithNull(request, field))).Reason);
        foreach (string revisionField in new[] { "Before", "After" })
            foreach (string field in new[] { "Snapshot", "Inventory", "Bundle" })
                Assert.Equal(EvidenceFailure.Format, Assert.Throws<EvidenceException>(() =>
                    SourceReuseCommands.Validate(WithNull(request, revisionField, field))).Reason);
        foreach (string field in new[] { "Plan", "Reports", "Trx", "CollectionRun", "Origin" })
            Assert.Equal(EvidenceFailure.Format, Assert.Throws<EvidenceException>(() =>
                SourceReuseCommands.Validate(WithNull(request, "Baseline", field))).Reason);
        SourceReuseCommands.Validate(request);
    }

    [Fact]
    public async Task DeadlineTerminatesProcessesStalledInOutputOrExit()
    {
        foreach (bool stalledOutput in new[] { true, false })
        {
            using Process child = Start(stalledOutput ? "StalledOutput" : "StalledExit");
            Assert.Equal("ready", await child.StandardOutput.ReadLineAsync().WaitAsync(TimeSpan.FromSeconds(15)));
            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => ProcessDeadline.Run(child, TimeSpan.FromMilliseconds(100),
                async token => stalledOutput ? await child.StandardOutput.ReadToEndAsync(token) : "output complete"));
            Assert.True(child.HasExited);
        }
    }

    [Fact]
    public async Task DeadlinePreservesSuccessfulOutputAndReaderFailure()
    {
        using (Process child = Start("Missing"))
        {
            string result = await ProcessDeadline.Run(child, TimeSpan.FromSeconds(15), token => child.StandardOutput.ReadToEndAsync(token));
            Assert.Equal("", result);
            Assert.Equal(0, child.ExitCode);
        }
        using (Process child = Start("StalledOutput"))
        {
            Assert.Equal("ready", await child.StandardOutput.ReadLineAsync().WaitAsync(TimeSpan.FromSeconds(15)));
            var expected = new IOException("reader rejected archive");
            Assert.Same(expected, await Assert.ThrowsAsync<IOException>(() => ProcessDeadline.Run<string>(child,
                TimeSpan.FromSeconds(15), _ => Task.FromException<string>(expected))));
            Assert.True(child.HasExited);
        }
    }

    private static Process Start(string scenario)
    {
        string worker = Environment.GetEnvironmentVariable("ATTRIBUTION_WORKER_DLL") ?? throw new InvalidOperationException("Missing worker fixture.");
        var start = new ProcessStartInfo("dotnet") { UseShellExecute = false, CreateNoWindow = true,
            RedirectStandardOutput = true, RedirectStandardError = true };
        start.ArgumentList.Add(worker);
        start.ArgumentList.Add(scenario);
        return Process.Start(start) ?? throw new IOException("Cannot start worker fixture.");
    }
}
