using AiDotNet.TestImpact;
using AttributionRuntime;

internal sealed record SourceRevisionInput(string Snapshot, string Inventory, string Bundle);
internal sealed record LocalExecutionInput(string Plan, string Reports, string Trx, string CollectionRun, RunIdentity Origin);
internal sealed record SourceReuseRequest(string Repository, SourceRevisionInput Before, SourceRevisionInput After, LocalExecutionInput Baseline);

internal static class SourceReuseCommands
{
    public static ReusePartition Prepare(string path)
        => Prepare(Read<SourceReuseRequest>(path));

    private static ReusePartition Prepare(SourceReuseRequest request)
    {
        SourceBundleSnapshot before = LocalEvidenceReader.ReadSource(request.Before.Snapshot);
        SourceBundleSnapshot after = LocalEvidenceReader.ReadSource(request.After.Snapshot);
        DiscoveryManifest oldInventory = Read<DiscoveryManifest>(request.Before.Inventory);
        DiscoveryManifest currentInventory = Read<DiscoveryManifest>(request.After.Inventory);
        LocalEvidenceReader.ValidateBundle(before, oldInventory, request.Before.Bundle);
        LocalEvidenceReader.ValidateBundle(after, currentInventory, request.After.Bundle);
        VerifiedExecution baseline = LocalEvidenceReader.Verify(oldInventory, request.Baseline);
        return SourceImpact.PrepareReuse(baseline, before, after, currentInventory,
            GitSourceDelta.Read(request.Repository, before.SourceTree, after.SourceTree));
    }

    public static VerifiedReusePartition Complete(string requestPath, string? executionPath)
    {
        SourceReuseRequest request = Read<SourceReuseRequest>(requestPath);
        ReusePartition partition = Prepare(request);
        VerifiedExecution? current = executionPath is null ? null : LocalEvidenceReader.Verify(
            Read<DiscoveryManifest>(request.After.Inventory), Read<LocalExecutionInput>(executionPath));
        return ExecutionReuse.Complete(partition, current);
    }

    internal static T Read<T>(string path) where T : class => ExecutionEvidence.ReadDocument<T>(File.ReadAllText(path));
}

internal static class LocalEvidenceReader
{
    public static SourceBundleSnapshot ReadSource(string path)
    {
        string json = File.ReadAllText(path);
        using var document = System.Text.Json.JsonDocument.Parse(json);
        if (document.RootElement.TryGetProperty(nameof(SourceBundleSnapshot.Assemblies), out _))
            return ExecutionEvidence.ReadDocument<SourceBundleSnapshot>(json);
        SourceSnapshot single = ExecutionEvidence.ReadDocument<SourceSnapshot>(json);
        return new(1, single.SourceTree, single.AssemblyFile, [single]);
    }

    public static VerifiedExecution Verify(DiscoveryManifest manifest, LocalExecutionInput input)
    {
        string[] files = Directory.GetFileSystemEntries(input.Reports);
        if (files.Length != 1 || !File.Exists(files[0]) || Path.GetExtension(files[0]) != ".json")
            throw new EvidenceException(EvidenceFailure.Outcome, "Expected one complete, non-revoked single-bundle host report.");
        AttributionReport report = SourceReuseCommands.Read<AttributionReport>(files[0]);
        // Local invocation only. Remote imports must authenticate completed jobs,
        // not compare a remote PID with processes on the importing machine.
        if (report.ProcessId <= 0) throw new EvidenceException(EvidenceFailure.Provenance, "Missing local host process identity.");
        try
        {
            using var host = System.Diagnostics.Process.GetProcessById(report.ProcessId);
            if (!host.WaitForExit(10_000))
                throw new EvidenceException(EvidenceFailure.Outcome, "Local test host is still running; evidence is not final.");
        }
        catch (ArgumentException) { /* Already exited. A reused live PID blocks conservatively. */ }
        string[] finalFiles = Directory.GetFileSystemEntries(input.Reports);
        if (finalFiles.Length != 1 || finalFiles[0] != files[0])
            throw new EvidenceException(EvidenceFailure.Outcome, "Host shutdown revoked or changed the evidence set.");
        if (Path.GetFileNameWithoutExtension(files[0]) != report.Token)
            throw new EvidenceException(EvidenceFailure.Provenance, "Report filename differs from its process identity.");
        return PlannedEvidence.Verify(manifest, SourceReuseCommands.Read<ExecutionPlan>(input.Plan), report, input.Trx,
            input.CollectionRun, input.Origin);
    }

    public static void ValidateSnapshot(SourceSnapshot snapshot, DiscoveryManifest inventory, string bundle)
        => ValidateBundle(new(1, snapshot.SourceTree, snapshot.AssemblyFile, [snapshot]), inventory, bundle);

    public static void ValidateBundle(SourceBundleSnapshot source, DiscoveryManifest inventory, string bundle)
    {
        if (source.Schema != 1 || source.SourceTree != inventory.Context.SourceTree || source.Assemblies is null || source.Assemblies.Length == 0 ||
            source.Assemblies.Count(snapshot => snapshot.AssemblyFile == source.TestAssembly) != 1 ||
            source.Assemblies.Select(snapshot => snapshot.AssemblyFile).Distinct(StringComparer.OrdinalIgnoreCase).Count() != source.Assemblies.Length ||
            RunnerBinding.HashBundle(bundle) != inventory.Context.BuildFingerprint)
            throw new EvidenceException(EvidenceFailure.Context, "Source snapshot differs from the discovered binary bundle.");
        foreach (SourceSnapshot snapshot in source.Assemblies) ValidateAssembly(snapshot, inventory, bundle);
    }

    private static void ValidateAssembly(SourceSnapshot snapshot, DiscoveryManifest inventory, string bundle)
    {
        if (inventory.Schema != 1 || snapshot.Status != SourceMapStatus.Verified || snapshot.SourceTree != inventory.Context.SourceTree ||
            string.IsNullOrWhiteSpace(snapshot.AssemblyFile) || Path.GetFileName(snapshot.AssemblyFile) != snapshot.AssemblyFile ||
            snapshot.AssemblyFile.Contains('/') || snapshot.AssemblyFile.Contains('\\') ||
            Hash(Path.Combine(bundle, snapshot.AssemblyFile)) != snapshot.AssemblyHash ||
            Hash(Path.ChangeExtension(Path.Combine(bundle, snapshot.AssemblyFile), ".pdb")) != snapshot.PdbHash)
            throw new EvidenceException(EvidenceFailure.Context, "Source snapshot differs from the discovered binary bundle.");
    }

    private static string Hash(string path)
    {
        using var stream = File.OpenRead(path);
        return Convert.ToHexStringLower(System.Security.Cryptography.SHA256.HashData(stream));
    }
}
