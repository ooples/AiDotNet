using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.RegularExpressions;
using AiDotNet.TestImpact;
using AttributionRuntime;

internal sealed record WorkflowImportRequest(WorkflowImportPolicy Policy, string Inventory, string Plan, string Reports, string Trx);

internal static partial class GitHubEvidenceReader
{
    public static async Task<VerifiedExecution> Verify(WorkflowImportRequest request, string directory)
        => (await VerifyObserved(request, directory)).Execution;

    public static async Task<VerifiedObservedExecution> VerifyObserved(WorkflowImportRequest request, string directory)
    {
        ValidateRequest(request);
        WorkflowImportPolicy policy = request.Policy;
        if (!RepositoryName().IsMatch(policy.Repository) || policy.RunId <= 0 || policy.Attempt <= 0)
            throw new EvidenceException(EvidenceFailure.Provenance, "Invalid GitHub workflow identity.");
        string endpoint = $"repos/{policy.Repository}/actions/runs/{policy.RunId}";
        WorkflowRunEvidence run = ReadRun(await Json(endpoint));
        WorkflowJobEvidence[] jobs = (await Pages(endpoint + $"/attempts/{policy.Attempt}/jobs?per_page=100", "jobs"))
            .Select(job => new WorkflowJobEvidence(job.GetProperty("run_id").GetInt64(), policy.Attempt,
                Text(job, "name"), Status(Text(job, "status")), Conclusion(Text(job, "conclusion")))).ToArray();
        JsonElement[] artifacts = await Pages(endpoint + "/artifacts?per_page=100", "artifacts");
        JsonElement[] named = artifacts.Where(artifact => Text(artifact, "name") == policy.ArtifactName).ToArray();
        if (named.Length != 1) throw new EvidenceException(EvidenceFailure.Provenance, "Expected exactly one workflow artifact.");
        JsonElement metadata = named[0];
        long artifactId = metadata.GetProperty("id").GetInt64();
        var artifact = new WorkflowArtifactEvidence(artifactId, metadata.GetProperty("workflow_run").GetProperty("id").GetInt64(),
            Text(metadata, "name"), metadata.GetProperty("expired").GetBoolean());
        // Reject unfinished/wrong-origin runs before retrieving any untrusted archive.
        WorkflowEvidence.Validate(policy, run, jobs, artifact, new(run.HeadSha, []));
        string digest = Text(metadata, "digest");
        if (!ArtifactDigest().IsMatch(digest)) throw new EvidenceException(EvidenceFailure.Provenance, "GitHub artifact digest is missing.");
        if (metadata.GetProperty("size_in_bytes").GetInt64() is <= 0 or > 268435456)
            throw new EvidenceException(EvidenceFailure.Format, "Artifact archive exceeds the import size limit.");
        string root = Path.GetFullPath(directory);
        if (Directory.Exists(root) || File.Exists(root)) throw new IOException("Import destination must be new.");
        Directory.CreateDirectory(root);
        string zip = Path.Combine(root, "artifact.zip");
        await Download($"repos/{policy.Repository}/actions/artifacts/{artifactId}/zip", zip);
        using (var stream = File.OpenRead(zip))
            if ("sha256:" + Convert.ToHexStringLower(SHA256.HashData(stream)) != digest)
                throw new EvidenceException(EvidenceFailure.Provenance, "Downloaded archive differs from GitHub's artifact digest.");
        string extracted = Path.Combine(root, "contents");
        ArtifactArchive.Extract(zip, extracted);
        DiscoveryManifest manifest = SourceReuseCommands.Read<DiscoveryManifest>(Contained(extracted, request.Inventory));
        ExecutionPlan plan = SourceReuseCommands.Read<ExecutionPlan>(Contained(extracted, request.Plan));
        if (!CommitSha().IsMatch(manifest.Context.SourceTree)) throw new EvidenceException(EvidenceFailure.Context, "Invalid tested commit.");
        JsonElement commit = await Json($"repos/{policy.Repository}/git/commits/{manifest.Context.SourceTree}");
        var source = new WorkflowSourceEvidence(Text(commit, "sha"), commit.GetProperty("parents").EnumerateArray()
            .Select(parent => Text(parent, "sha")).ToArray());
        if (source.Parents.Length == 2 && source.Parents[1] == run.HeadSha && source.Sha != run.HeadSha)
        {
            WorkflowPullIdentity[] pulls = run.Pulls.Where(pull => pull.HeadSha == run.HeadSha).ToArray();
            if (pulls.Length == 1 && pulls[0].BaseSha != source.Parents[0])
            {
                WorkflowPullIdentity pull = pulls[0];
                JsonElement tip = await Json($"repos/{policy.Repository}/git/ref/heads/{Uri.EscapeDataString(pull.BaseBranch)}");
                string baseTip = Text(tip.GetProperty("object"), "sha");
                JsonElement recordedToTested = await Json($"repos/{policy.Repository}/compare/{pull.BaseSha}...{source.Parents[0]}");
                JsonElement testedToCurrent = await Json($"repos/{policy.Repository}/compare/{source.Parents[0]}...{baseTip}");
                source = source with { BaseBinding = new(pull.BaseSha, source.Parents[0], pull.BaseBranch, baseTip,
                    Text(recordedToTested.GetProperty("merge_base_commit"), "sha") == pull.BaseSha,
                    Text(testedToCurrent.GetProperty("merge_base_commit"), "sha") == source.Parents[0]) };
            }
        }
        WorkflowEvidence.Validate(policy, run, jobs, artifact, source);
        if (plan.Scope != policy.RequiredScope)
            throw new EvidenceException(EvidenceFailure.Scope, "Imported execution has the wrong required scope.");
        string reports = Contained(extracted, request.Reports);
        string[] files = Directory.GetFileSystemEntries(reports);
        if (files.Length != 1 || !File.Exists(files[0]) || Path.GetExtension(files[0]) != ".json")
            throw new EvidenceException(EvidenceFailure.Outcome, "Incomplete or revoked remote report.");
        AttributionReport report = SourceReuseCommands.Read<AttributionReport>(files[0]);
        if (Path.GetFileNameWithoutExtension(files[0]) != report.Token)
            throw new EvidenceException(EvidenceFailure.Provenance, "Remote report filename differs from its identity.");
        // Completed GitHub job is the remote termination barrier. Never inspect
        // the producer's PID on the importing host.
        VerifiedObservedExecution execution = PlannedEvidence.VerifyObserved(manifest, plan, report, Contained(extracted, request.Trx), report.Run,
            new(policy.Repository, policy.RunId, policy.Attempt));
        WorkflowRunEvidence finalRun = ReadRun(await Json(endpoint));
        JsonElement finalArtifact = await Json($"repos/{policy.Repository}/actions/artifacts/{artifactId}");
        WorkflowEvidence.Validate(policy, finalRun, jobs,
            new(finalArtifact.GetProperty("id").GetInt64(), finalArtifact.GetProperty("workflow_run").GetProperty("id").GetInt64(),
                Text(finalArtifact, "name"), finalArtifact.GetProperty("expired").GetBoolean()), source);
        if (Text(finalArtifact, "digest") != digest)
            throw new EvidenceException(EvidenceFailure.Provenance, "Artifact changed during import.");
        return execution;
    }

    internal static void ValidateRequest(WorkflowImportRequest request)
    {
        if (request is null || request.Policy is null || string.IsNullOrWhiteSpace(request.Policy.Repository) ||
            string.IsNullOrWhiteSpace(request.Policy.HeadSha) || string.IsNullOrWhiteSpace(request.Policy.WorkflowPath) ||
            string.IsNullOrWhiteSpace(request.Policy.JobName) || string.IsNullOrWhiteSpace(request.Policy.ArtifactName) ||
            string.IsNullOrWhiteSpace(request.Inventory) || string.IsNullOrWhiteSpace(request.Plan) ||
            string.IsNullOrWhiteSpace(request.Reports) || string.IsNullOrWhiteSpace(request.Trx))
            throw new EvidenceException(EvidenceFailure.Format, "Missing workflow import request fields.");
    }

    private static WorkflowRunEvidence ReadRun(JsonElement run) => new(
        Text(run.GetProperty("repository"), "full_name"), Text(run.GetProperty("head_repository"), "full_name"),
        run.GetProperty("id").GetInt64(), run.GetProperty("run_attempt").GetInt32(), Text(run, "head_sha"), Text(run, "path"),
        Text(run, "event") switch { "pull_request" => WorkflowEvent.PullRequest, "push" => WorkflowEvent.Push,
            "merge_group" => WorkflowEvent.MergeGroup, "workflow_dispatch" => WorkflowEvent.WorkflowDispatch,
            _ => throw new EvidenceException(EvidenceFailure.Provenance, "Unsupported workflow event.") },
        Status(Text(run, "status")), Conclusion(Text(run, "conclusion")),
        run.GetProperty("pull_requests").EnumerateArray().Select(pull => new WorkflowPullIdentity(
            Text(pull.GetProperty("base"), "sha"), Text(pull.GetProperty("head"), "sha"), Text(pull.GetProperty("base"), "ref"))).ToArray());

    private static WorkflowStatus Status(string text) => text switch { "completed" => WorkflowStatus.Completed,
        "in_progress" => WorkflowStatus.InProgress, "queued" or "pending" or "waiting" or "requested" => WorkflowStatus.Queued,
        _ => throw new EvidenceException(EvidenceFailure.Provenance, "Unsupported workflow status.") };
    private static WorkflowConclusion Conclusion(string text) => text switch { "" => WorkflowConclusion.None,
        "success" => WorkflowConclusion.Success, "failure" => WorkflowConclusion.Failure, "cancelled" => WorkflowConclusion.Cancelled,
        "skipped" => WorkflowConclusion.Skipped, "timed_out" => WorkflowConclusion.TimedOut, "action_required" => WorkflowConclusion.ActionRequired,
        "neutral" => WorkflowConclusion.Neutral, "stale" => WorkflowConclusion.Stale,
        _ => throw new EvidenceException(EvidenceFailure.Provenance, "Unsupported workflow conclusion.") };
    private static string Text(JsonElement value, string property) => value.GetProperty(property).GetString() ?? "";
    private static async Task<JsonElement> Json(string endpoint)
    {
        using Process process = Start("api", endpoint);
        (string output, string errors) = await ProcessDeadline.Run(process, TimeSpan.FromMinutes(2), async token =>
        {
            Task<string> stderr = process.StandardError.ReadToEndAsync(token);
            string stdout = await process.StandardOutput.ReadToEndAsync(token);
            return (stdout, await stderr);
        });
        if (process.ExitCode != 0) throw new IOException("GitHub metadata request failed: " + errors);
        using JsonDocument document = JsonDocument.Parse(output);
        return document.RootElement.Clone();
    }
    private static async Task<JsonElement[]> Pages(string endpoint, string field)
    {
        var result = new List<JsonElement>();
        for (int page = 1; page <= 100; page++)
        {
            JsonElement response = await Json(endpoint + "&page=" + page);
            JsonElement[] items = response.GetProperty(field).EnumerateArray().Select(item => item.Clone()).ToArray();
            result.AddRange(items);
            if (items.Length < 100) return result.ToArray();
        }
        throw new EvidenceException(EvidenceFailure.Provenance, "GitHub inventory exceeded pagination limit.");
    }
    private static Process Start(params string[] arguments)
    {
        var start = new ProcessStartInfo("gh") { RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false, CreateNoWindow = true };
        foreach (string argument in arguments) start.ArgumentList.Add(argument);
        start.ArgumentList.Add("--hostname"); start.ArgumentList.Add("github.com");
        return Process.Start(start) ?? throw new IOException("Cannot start authenticated GitHub client.");
    }
    private static async Task Download(string endpoint, string output)
    {
        await using var destination = new FileStream(output, FileMode.CreateNew, FileAccess.Write, FileShare.None);
        using Process process = Start("api", endpoint);
        string errors = await ProcessDeadline.Run(process, TimeSpan.FromMinutes(5), async token =>
        {
            Task<string> stderr = process.StandardError.ReadToEndAsync(token);
            byte[] buffer = new byte[81920];
            long length = 0;
            int read;
            while ((read = await process.StandardOutput.BaseStream.ReadAsync(buffer, token)) != 0)
            {
                length += read;
                if (length > 268435456) throw new IOException("Artifact archive exceeds the import size limit.");
                await destination.WriteAsync(buffer.AsMemory(0, read), token);
            }
            return await stderr;
        });
        if (process.ExitCode != 0) throw new IOException("GitHub artifact download failed: " + errors);
    }
    private static string Contained(string root, string relative)
        => ArtifactArchive.ResolveContained(root, relative);
    [GeneratedRegex(@"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", RegexOptions.CultureInvariant)]
    private static partial Regex RepositoryName();
    [GeneratedRegex(@"^sha256:[0-9a-f]{64}$", RegexOptions.CultureInvariant)]
    private static partial Regex ArtifactDigest();
    [GeneratedRegex(@"^[0-9a-f]{40}$", RegexOptions.CultureInvariant)]
    private static partial Regex CommitSha();
}
