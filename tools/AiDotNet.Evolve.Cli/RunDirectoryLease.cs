using AiDotNet.Configuration;
using AiDotNet.Evolution;

namespace AiDotNet.Evolve.Cli;

/// <summary>Cooperating CLI writers fail before preflight when their output directories overlap.</summary>
internal sealed class RunDirectoryLease : IDisposable
{
    private readonly List<FileStream> _held = new();

    internal static RunDirectoryLease Acquire(EvolutionOptions options)
    {
        var roots = new List<string>();
        if (options.OutputDirectory is { } output) roots.Add(output);
        else if (options.Resume || options.CheckpointInterval > 0 || options.CheckpointDirectory is not null || options.Trace.Enabled)
            roots.Add(Path.Combine(Path.GetTempPath(), "aidotnet-evolve", EvolutionOutputLayout.CreateStem(options.RunId)));
        if (options.CheckpointDirectory is { } checkpoint) roots.Add(checkpoint);
        if (options.Trace.Enabled && options.Trace.Path is { } trace)
            roots.Add(Path.GetDirectoryName(Path.GetFullPath(trace))!);
        return Acquire(roots);
    }

    internal static RunDirectoryLease Acquire(IEnumerable<string> roots)
    {
        var lease = new RunDirectoryLease();
        try
        {
            var comparer = OperatingSystem.IsWindows() ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal;
            foreach (string root in roots.Select(Path.GetFullPath).Distinct(comparer).OrderBy(path => path, comparer))
            {
                Directory.CreateDirectory(root);
                if ((File.GetAttributes(root) & FileAttributes.ReparsePoint) != 0)
                    throw new IOException("Linked run output roots are not supported.");
                string path = Path.Combine(root, ".aidotnet-cli.lock");
                if (File.Exists(path) && (File.GetAttributes(path) & FileAttributes.ReparsePoint) != 0)
                    throw new IOException("Linked run leases are not supported.");
                lease._held.Add(new FileStream(path, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None));
            }
            return lease;
        }
        catch { lease.Dispose(); throw; }
    }

    public void Dispose()
    {
        foreach (var stream in _held) stream.Dispose();
        _held.Clear(); // Keep lock files: unlinking them introduces a second-inode lock race on Unix.
    }
}
