using System.Collections.Concurrent;
using System.Security.Cryptography;
using AiDotNet.TestImpact;
using Mono.Cecil;

internal enum GpuStartupContract { Unresolved, DisabledWithoutDiagnosticCallbacks }

// Review of the pinned Tensors module's opt-out branch. EnsureRegistered runs
// even when GPU detection is disabled: an empty dump path is separately required
// to exclude its timer, shutdown handlers and immediate file dump. This does not
// prove the surrounding test module's CPU/license initialization or host lifetime.
internal static class ReviewedGpuStartup
{
    internal static GpuStartupContract Read(AssemblyDefinition tensors, RuntimeContractProfile? profile)
    {
        try
        {
            if (!Inputs(profile) || Hash(tensors.MainModule.FileName) != ReviewedNumericProvider.PackageHash ||
                Hash(typeof(object).Assembly.Location) != ReviewedOwnerCompletion.RuntimeHash ||
                Hash(typeof(ConcurrentDictionary<,>).Assembly.Location) != RuntimeCacheInitializationReader.CollectionsHash)
                return GpuStartupContract.Unresolved;
            // Re-read the exact file rather than interpreting a caller-mutated
            // Cecil object as if its FileName authenticated its method bodies.
            using var pinned = AssemblyDefinition.ReadAssembly(tensors.MainModule.FileName,
                new ReaderParameters { AssemblyResolver = tensors.MainModule.AssemblyResolver });
            if (pinned.MainModule.Mvid != tensors.MainModule.Mvid || pinned.Name.FullName != tensors.Name.FullName ||
                pinned.Modules.Count != 1) return GpuStartupContract.Unresolved;
            // In these bytes, the no-dump diagnostics initializer creates only
            // its private arrays, dictionary and capture object. Its early return
            // precedes Timer construction, event registration, and DumpTo. The
            // module's nonempty GPU opt-out then returns before native discovery.
            // Default IntPtr dictionary comparison cannot invoke user callbacks.
            return Hash(tensors.MainModule.FileName) == ReviewedNumericProvider.PackageHash &&
                Hash(typeof(object).Assembly.Location) == ReviewedOwnerCompletion.RuntimeHash &&
                Hash(typeof(ConcurrentDictionary<,>).Assembly.Location) == RuntimeCacheInitializationReader.CollectionsHash
                ? GpuStartupContract.DisabledWithoutDiagnosticCallbacks : GpuStartupContract.Unresolved;
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or ArgumentException or BadImageFormatException or
            AssemblyResolutionException or ResolutionException or InvalidOperationException)
        { return GpuStartupContract.Unresolved; }
    }

    internal static bool Inputs(RuntimeContractProfile? profile) => RuntimeProfileEvidence.HasObservedCpuResetPreconditions(profile);

    private static string Hash(string path)
    {
        for (FileSystemInfo? entry = new FileInfo(Path.GetFullPath(path)); entry is not null;
            entry = entry is FileInfo file ? file.Directory : ((DirectoryInfo)entry).Parent)
            if (!entry.Exists || (entry.Attributes & FileAttributes.ReparsePoint) != 0 || entry.LinkTarget is not null)
                throw new InvalidDataException("Missing or linked startup-contract binary.");
        using var stream = File.OpenRead(path);
        return Convert.ToHexStringLower(SHA256.HashData(stream));
    }
}
