using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using AiDotNet.Agentic.Models;
using Xunit;
using static AiDotNet.Evolution.CSharp.Tests.CompilerTestSupport;

namespace AiDotNet.Evolution.CSharp.Tests;

public sealed class CSharpProposalSourceTests
{
    private static ResourceMeteredVariationOperator<Programs.ProgramGenome> Meter(CSharpProposalSource<double> source, EvolutionResourceLedger ledger) =>
        new(source, ledger, source.MaximumProposalResources, source.CostUnitVersionHash);

    [Fact]
    public void Setup_denial_and_bad_reference_images_cannot_disappear_from_accounting()
    {
        var options = Options();
        var denied = Ledger(0.05m);
        var client = new ScriptedClient();
        Assert.Throws<EvolutionResourceBudgetException>(() => CSharpProposalSource<double>.Create(client, options, Program(), denied));
        Assert.False(Directory.Exists(options.AuditDirectory));
        Assert.Equal(0, denied.Snapshot().Settled);
        string invalidImage = Path.GetTempFileName();
        try
        {
            options.ReferencePaths = new[] { invalidImage };
            var ledger = Ledger();
            Assert.Throws<ArgumentException>(() => CSharpProposalSource<double>.Create(client, options, Program(), ledger));
            Assert.Equal(1, ledger.Snapshot().Unknown);
            Assert.Equal(0.1m, ledger.Snapshot().Spent["cost_units"]);
            Assert.Equal(CSharpPatchCompiler.MaximumReferenceBytes, ledger.Snapshot().Spent["reference_bytes"]);
            File.WriteAllBytes(invalidImage, new byte[] { 1, 2, 3, 4 });
            Assert.Throws<BadImageFormatException>(() => CSharpProposalSource<double>.Create(client, options, Program(), Ledger()));
            using (var stream = new FileStream(invalidImage, FileMode.Open, FileAccess.Write, FileShare.Read))
                stream.SetLength(32 * 1024 * 1024 + 1);
            Assert.Throws<ArgumentException>(() => new CSharpPatchCompiler(options, Program()));
        }
        finally { File.Delete(invalidImage); }
        Assert.Empty(client.Conversations);
    }

    [Fact]
    public async Task Cancellation_and_fatal_errors_after_model_dispatch_keep_unknown_maxima_and_propagate()
    {
        var options = Options();
        var client = new ScriptedClient();
        var ledger = Ledger();
        var source = CSharpProposalSource<double>.Create(client, options, Program(), ledger);
        using var cancelled = new CancellationTokenSource();
        cancelled.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => Meter(source, ledger).ProposeAsync(Context(), cancelled.Token).AsTask());
        Assert.Equal(0.1m, ledger.Snapshot().Spent["cost_units"]);
        Assert.Empty(client.Conversations);
        foreach (Exception failure in new Exception[] { new OperationCanceledException(), new OutOfMemoryException("synthetic") })
        {
            ledger = Ledger();
            client = new ScriptedClient { Handler = (_, _) => throw failure };
            source = CSharpProposalSource<double>.Create(client, Options(), Program(), ledger);
            var thrown = await Record.ExceptionAsync(() => Meter(source, ledger).ProposeAsync(Context()).AsTask());
            Assert.Same(failure, thrown);
            Assert.Equal(1, ledger.Snapshot().Unknown);
            Assert.Equal(0.1m + source.MaximumProposalResources["cost_units"], ledger.Snapshot().Spent["cost_units"]);
            Assert.Equal(1, source.GetUsage().ChatCalls);
        }
    }

    [Fact]
    public void Restoring_incomplete_ambiguous_or_incompatible_usage_is_transactional()
    {
        var source = CSharpProposalSource<double>.Create(new ScriptedClient(), Options(), Program(), Ledger());
        string initial = source.CaptureState();
        foreach (string invalid in new[]
        {
            "null", "{}", initial.Replace("\"Calls\":0", "\"Calls\":0,\"Calls\":0", StringComparison.Ordinal),
            initial.Replace("\"Calls\":0", "\"unknown\":0", StringComparison.Ordinal),
            initial.Replace("\"Errors\":0", "\"Errors\":1", StringComparison.Ordinal),
            initial.Replace("\"Abandoned\":0", "\"Abandoned\":1", StringComparison.Ordinal),
            initial.Replace("\"Retries\":0", "\"Retries\":1", StringComparison.Ordinal),
            initial.Replace("\"InputTokens\":0", "\"InputTokens\":1000000000001", StringComparison.Ordinal)
        })
        {
            Assert.Throws<ArgumentException>(() => source.RestoreState(invalid));
            Assert.Equal(initial, source.CaptureState());
        }
        Assert.ThrowsAny<JsonException>(() => source.RestoreState("{"));
        Assert.Equal(initial, source.CaptureState());
    }

    [Fact]
    public async Task Repair_history_and_reported_model_identity_remain_bounded()
    {
        var options = Options();
        options.MaxRepairs = 3;
        options.MaxResponseChars = 512;
        var client = new ScriptedClient
        {
            Handler = (_, _) => new ChatResponse(ChatMessage.Assistant(new string('x', 600)),
            modelId: new string('m', 300), usage: new ChatUsage(100, 50))
        };
        var ledger = Ledger();
        var source = CSharpProposalSource<double>.Create(client, options, Program(), ledger);
        await Assert.ThrowsAsync<InvalidOperationException>(() => Meter(source, ledger).ProposeAsync(Context()).AsTask());
        Assert.Equal(4, client.Conversations.Count);
        Assert.All(client.Conversations.Skip(1), messages =>
        {
            Assert.Equal(4, messages.Count);
            Assert.True(messages[2].Text.Length < 512);
        });
        foreach (string path in Directory.GetFiles(options.AuditDirectory, "*.json"))
        {
            using JsonDocument record = JsonDocument.Parse(File.ReadAllText(path));
            Assert.StartsWith("sha256:", record.RootElement.GetProperty("reportedModel").GetString());
        }
        Assert.Equal(3, source.GetUsage().Retries);
        Assert.Equal(0, ledger.Snapshot().Spent.GetValueOrDefault("build_calls"));
        Assert.Equal(4, ledger.Snapshot().Spent["audit_calls"]);
    }

    [Fact]
    public async Task Compile_failure_is_repaired_against_original_parent_and_all_work_is_charged()
    {
        var client = new ScriptedClient { Handler = (call, messages) => ScriptedClient.Response(Reply(messages, call == 1 ? "MISSING" : "2")) };
        var options = Options();
        var ledger = Ledger();
        var source = CSharpProposalSource<double>.Create(client, options, Program(), ledger);
        var metered = Meter(source, ledger);
        var child = await metered.ProposeAsync(Context());
        Assert.Equal(Source.Replace("return 1", "return 2", StringComparison.Ordinal), child.Source);
        Assert.Equal(2, client.Conversations.Count);
        Assert.Equal(2, client.Conversations[0].Count);
        Assert.Equal(4, client.Conversations[1].Count);
        Assert.Equal(client.Conversations[0][1].Text, client.Conversations[1][1].Text);
        Assert.Contains("CS0103", client.Conversations[1][3].Text);
        Assert.DoesNotContain("MISSING", client.Conversations[1][3].Text);
        Assert.Equal(options.MaxOutputTokens, client.LastOptions?.MaxOutputTokens);
        Assert.NotNull(client.LastOptions?.Seed);
        Assert.Equal(2, source.GetUsage().ChatCalls);
        Assert.Equal(1, source.GetUsage().Retries);
        Assert.Equal(200, source.GetUsage().InputTokens);
        Assert.Equal(100, source.GetUsage().OutputTokens);
        var snapshot = ledger.Snapshot();
        Assert.Equal(2.654m, snapshot.Spent["cost_units"]); // setup .1 + calls 2 + parse .03 + emit .5 + audit .02 + tokens .004
        Assert.Equal(2m, snapshot.Spent["model_calls"]);
        Assert.Equal(3m, snapshot.Spent["parse_calls"]);
        Assert.Equal(2m, snapshot.Spent["build_calls"]);
        Assert.Equal(0, snapshot.Unknown);
        Assert.False(snapshot.MaximumViolated);
        Assert.Equal(2.554m, metered.GetProposalCost(1).Charged["cost_units"]);
        string[] evidence = Directory.GetFiles(options.AuditDirectory, "*.json");
        Assert.Equal(2, evidence.Length);
        Assert.Equal(evidence.Sum(file => new FileInfo(file).Length), snapshot.Spent["artifact_bytes"]);
        var records = evidence.Select(file => JsonNode.Parse(File.ReadAllText(file))!).OrderBy(record => (int)record["attempt"]!).ToArray();
        Assert.False((bool)records[0]["compiled"]!);
        Assert.True((bool)records[1]["compiled"]!);
        Assert.Equal(Source, (string?)records[0]["parentSource"]);
        Assert.Equal(child.Source, (string?)records[1]["proposedSource"]);
        Assert.Equal(64, ((string)records[1]["emittedSha256"]!).Length);
        Assert.Single(records[1]["referenceSha256"]!.AsArray());
        Assert.Equal(options.ModelVersionIdentity, (string?)records[1]["declaredModelVersion"]);
        Assert.Equal(source.VersionHash, (string?)records[1]["operatorVersion"]);
        Assert.Equal(2.554m, (decimal)records[1]["cumulativeCostUnits"]!);
    }

    [Fact]
    public async Task Exhausted_repairs_keep_failed_costs_and_never_return_an_uncompiled_child()
    {
        var client = new ScriptedClient { Handler = (_, messages) => ScriptedClient.Response(Reply(messages, "UNKNOWN")) };
        var options = Options();
        var ledger = Ledger();
        var source = CSharpProposalSource<double>.Create(client, options, Program(), ledger);
        var metered = Meter(source, ledger);
        await Assert.ThrowsAsync<InvalidOperationException>(() => metered.ProposeAsync(Context()).AsTask());
        Assert.Equal(2, source.GetUsage().ChatCalls);
        Assert.Equal(1, source.GetUsage().AbandonedProposals);
        Assert.Equal(EvolutionResourceOutcome.Rejected, metered.GetProposalCost(1).Outcome);
        Assert.Equal(2.654m, ledger.Snapshot().Spent["cost_units"]);
        Assert.Equal(0, ledger.Snapshot().Unknown);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Unknown_provider_usage_is_not_free_and_raw_exception_text_is_not_exposed(bool missingUsage)
    {
        var client = new ScriptedClient
        {
            Handler = (_, _) => missingUsage ? new ChatResponse(ChatMessage.Assistant("{}")) : throw new IOException("SECRET_CREDENTIAL")
        };
        var ledger = Ledger();
        var source = CSharpProposalSource<double>.Create(client, Options(), Program(), ledger);
        var metered = Meter(source, ledger);
        var error = await Assert.ThrowsAsync<InvalidOperationException>(() => metered.ProposeAsync(Context()).AsTask());
        Assert.DoesNotContain("SECRET_CREDENTIAL", error.ToString());
        Assert.Equal(1, ledger.Snapshot().Unknown);
        Assert.Equal(0.1m + source.MaximumProposalResources["cost_units"], ledger.Snapshot().Spent["cost_units"]);
        Assert.True(metered.GetProposalCost(1).IsUnknown);
        Assert.Single(client.Conversations);
    }

    [Fact]
    public async Task Denied_reservation_dispatches_neither_model_nor_compiler()
    {
        var client = new ScriptedClient();
        var ledger = Ledger(0.2m);
        var source = CSharpProposalSource<double>.Create(client, Options(), Program(), ledger);
        await Assert.ThrowsAsync<EvolutionResourceBudgetException>(() => Meter(source, ledger).ProposeAsync(Context()).AsTask());
        Assert.Empty(client.Conversations);
        Assert.Equal(0.1m, ledger.Snapshot().Spent["cost_units"]);
        Assert.Equal(0, source.GetUsage().Proposals);
        Assert.Equal(1, ledger.Snapshot().Denied);
    }

    [Fact]
    public async Task Actual_token_overrun_is_retained_and_blocks_further_ledger_admission()
    {
        var options = Options();
        options.MaxRepairs = 0;
        var client = new ScriptedClient { Handler = (_, _) => ScriptedClient.Response("{}", new ChatUsage(100, options.MaxOutputTokens + 1)) };
        var ledger = Ledger();
        var source = CSharpProposalSource<double>.Create(client, options, Program(), ledger);
        await Assert.ThrowsAsync<InvalidOperationException>(() => Meter(source, ledger).ProposeAsync(Context()).AsTask());
        Assert.Single(client.Conversations);
        Assert.True(ledger.Snapshot().MaximumViolated);
        Assert.Equal(options.MaxOutputTokens + 1, ledger.Snapshot().Spent["output_tokens"]);
        Assert.Equal(0, ledger.Snapshot().Spent.GetValueOrDefault("build_calls"));
        Assert.Equal(1, source.GetUsage().AbandonedProposals);
    }

    [Fact]
    public async Task Invalid_parent_and_prompt_admission_failures_have_parse_and_audit_costs_but_no_model_calls()
    {
        foreach (bool invalidParent in new[] { false, true })
        {
            var options = Options();
            if (!invalidParent) options.MaxInputTokens = 1;
            var client = new ScriptedClient();
            var ledger = Ledger();
            var source = CSharpProposalSource<double>.Create(client, options, Program(), ledger);
            await Assert.ThrowsAsync<InvalidOperationException>(() => Meter(source, ledger).ProposeAsync(Context(Parent(invalidParent ? "class C {}" : Source))).AsTask());
            Assert.Empty(client.Conversations);
            Assert.Equal(0.12m, ledger.Snapshot().Spent["cost_units"]);
            Assert.Single(Directory.GetFiles(options.AuditDirectory, "*.json"));
            Assert.Equal(0, ledger.Snapshot().Unknown);
        }
    }

    [Fact]
    public async Task Evidence_collisions_fail_closed_and_usage_state_restores_without_losing_attempt_identity()
    {
        var options = Options();
        var ledger = Ledger();
        var source = CSharpProposalSource<double>.Create(new ScriptedClient(), options, Program(), ledger);
        string emptyState = source.CaptureState();
        await source.ProposeAsync(Context());
        string once = source.CaptureState();
        await source.ProposeAsync(Context());
        Assert.Equal(2, Directory.GetFiles(options.AuditDirectory, "*.json").Length);
        source.RestoreState(once);
        Assert.Equal(1, source.GetUsage().Proposals);
        var metered = Meter(source, ledger);
        await Assert.ThrowsAsync<InvalidOperationException>(() => metered.ProposeAsync(Context()).AsTask());
        Assert.Equal(1, ledger.Snapshot().Unknown);
        Assert.Equal(2, Directory.GetFiles(options.AuditDirectory, "*.json").Length);
        var corrupted = JsonNode.Parse(once)!;
        corrupted["Calls"] = -1;
        string before = source.CaptureState();
        Assert.Throws<ArgumentException>(() => source.RestoreState(corrupted.ToJsonString()));
        Assert.Equal(before, source.CaptureState());
        corrupted["Calls"] = 1000;
        Assert.Throws<ArgumentException>(() => source.RestoreState(corrupted.ToJsonString()));
        corrupted["Version"] = "wrong";
        Assert.Throws<ArgumentException>(() => source.RestoreState(corrupted.ToJsonString()));
        Assert.Throws<ArgumentException>(() => source.RestoreState(new string('x', 4097)));
        source.RestoreState(emptyState);
        Assert.Equal(0, source.GetUsage().ChatCalls);
    }

    [Fact]
    public async Task Malformed_unicode_and_overlong_answers_preserve_bounded_exact_evidence()
    {
        foreach (string response in new[] { "bad" + '\ud800', new string('x', 600) })
        {
            var options = Options();
            options.MaxRepairs = 0;
            options.MaxResponseChars = 512;
            var ledger = Ledger();
            var client = new ScriptedClient { Handler = (_, _) => ScriptedClient.Response(response) };
            var source = CSharpProposalSource<double>.Create(client, options, Program(), ledger);
            await Assert.ThrowsAsync<InvalidOperationException>(() => Meter(source, ledger).ProposeAsync(Context()).AsTask());
            using JsonDocument record = JsonDocument.Parse(File.ReadAllText(Directory.GetFiles(options.AuditDirectory, "*.json").Single()));
            byte[] retained = Convert.FromBase64String(record.RootElement.GetProperty("retainedResponseUtf16Base64").GetString()!);
            int length = Math.Min(response.Length, options.MaxResponseChars);
            Assert.Equal(length * 2, retained.Length);
            for (int index = 0; index < length; index++) Assert.Equal(response[index], (char)(retained[index * 2] | retained[index * 2 + 1] << 8));
            Assert.Equal(response.Length > 512, record.RootElement.GetProperty("responseTruncated").GetBoolean());
        }
    }

    [Fact]
    public async Task Captured_configuration_cannot_be_changed_by_mutating_caller_options_or_model_alias()
    {
        var options = Options();
        var client = new ScriptedClient();
        var ledger = Ledger();
        var source = CSharpProposalSource<double>.Create(client, options, Program(), ledger);
        string version = source.VersionHash;
        options.ModelCallCostUnits = 999;
        options.AuditDirectory = "not-the-configured-directory";
        await Meter(source, ledger).ProposeAsync(Context());
        Assert.Equal(1.382m, ledger.Snapshot().Spent["cost_units"]);
        Assert.Equal(version, source.VersionHash);
        client.ModelId = "changed";
        await Assert.ThrowsAsync<InvalidOperationException>(() => source.ProposeAsync(Context(generation: 2)).AsTask());
        Assert.Single(client.Conversations);
    }
}
