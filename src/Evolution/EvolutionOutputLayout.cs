using System.Globalization;
using System.Text;
using AiDotNet.Enums;
using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>Derives every file path an evolution run writes from one output directory and its run identifier.</summary>
/// <remarks>
/// <para>
/// Set <c>EvolutionEngineOptions.OutputDirectory</c> and a run has a single root under which its checkpoint and its
/// trace live at deterministic, collision-free paths. The paths depend only on the resolved root and the run
/// identifier, so two processes given the same pair address the same files and a run resumed tomorrow finds
/// yesterday's checkpoint without being told where it went.
/// </para>
/// <para>
/// The run identifier is turned into a file-name stem by replacing every character outside
/// <c>A-Z a-z 0-9 . _ -</c> with an underscore. Because that mapping is many-to-one, a stem that needed any
/// replacement also carries a short hash of the original identifier, so <c>run/a</c> and <c>run:a</c> cannot silently
/// share one checkpoint. OpenEvolve does the opposite: every run under one output directory writes
/// <c>evolution_trace.jsonl</c> and opens it in append mode (controller.py:142-160 with
/// trace_export_utils.py:153-179), so a second run interleaves its records into the first run's file with nothing to
/// tell them apart.
/// </para>
/// <para>
/// Nothing here touches the file system; it computes strings. The checkpoint store and the trace observer each create
/// their own parent directory when they first write, so a configured output directory costs nothing until something
/// is actually saved.
/// </para>
/// <para><b>For Beginners:</b> Rather than telling the engine three separate paths - where the checkpoint goes, where
/// the trace goes, and what to call them - you give it one folder and it works the rest out. Ask this type for
/// <see cref="CheckpointPath"/> when you build a checkpoint store, and for <see cref="TracePath"/> when you build a
/// trace observer, and both land under the same folder named after your run. Because the names are derived rather
/// than random, running the same job twice overwrites the same checkpoint instead of littering the folder, and two
/// different jobs never collide.</para>
/// </remarks>
public sealed class EvolutionOutputLayout
{
    /// <summary>The subdirectory of <see cref="Root"/> that holds checkpoints.</summary>
    public const string CheckpointsFolderName = "checkpoints";

    /// <summary>The subdirectory of <see cref="Root"/> that holds traces.</summary>
    public const string TracesFolderName = "traces";

    /// <summary>Initializes a layout rooted at one output directory.</summary>
    /// <param name="outputDirectory">The non-blank output directory; resolved to a full path.</param>
    /// <param name="runId">The non-blank run identifier the file names are derived from.</param>
    /// <exception cref="ArgumentNullException">An argument is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">An argument is empty or white space, or the directory is not a valid path.</exception>
    public EvolutionOutputLayout(string outputDirectory, string runId)
    {
        Guard.NotNullOrWhiteSpace(outputDirectory);
        Guard.NotNullOrWhiteSpace(runId);
        Root = ResolveDirectory(outputDirectory, nameof(outputDirectory));
        RunId = runId.Trim();
        Stem = CreateStem(RunId);
    }

    /// <summary>Gets the resolved absolute output directory.</summary>
    public string Root { get; }

    /// <summary>Gets the run identifier the file names are derived from.</summary>
    public string RunId { get; }

    /// <summary>Gets the file-name stem derived from <see cref="RunId"/>.</summary>
    public string Stem { get; }

    /// <summary>Gets the directory checkpoints are written to.</summary>
    public string CheckpointsDirectory => Path.Combine(Root, CheckpointsFolderName);

    /// <summary>Gets the directory traces are written to.</summary>
    public string TracesDirectory => Path.Combine(Root, TracesFolderName);

    /// <summary>Gets the path <see cref="JsonEvolutionCheckpointStore"/> should be pointed at for this run.</summary>
    public string CheckpointPath => Path.Combine(CheckpointsDirectory, Stem + ".checkpoint.json");

    /// <summary>Returns the trace path for one format and compression setting.</summary>
    /// <param name="format">The on-disk layout the trace will be written in.</param>
    /// <param name="compress">Whether the trace will be gzip-compressed, which appends a <c>.gz</c> suffix.</param>
    /// <returns>The absolute trace path.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="format"/> is undefined.</exception>
    public string TracePath(EvolutionTraceFormat format, bool compress)
    {
        if (!Enum.IsDefined(typeof(EvolutionTraceFormat), format)) throw new ArgumentOutOfRangeException(nameof(format));
        string extension = format == EvolutionTraceFormat.Json ? ".trace.json" : ".trace.jsonl";
        if (compress) extension += ".gz";
        return Path.Combine(TracesDirectory, Stem + extension);
    }

    /// <summary>Returns the sidecar metadata path that accompanies a trace file.</summary>
    /// <param name="tracePath">The non-blank trace path the sidecar describes.</param>
    /// <returns>The trace path with a <c>.meta.json</c> suffix appended.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="tracePath"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="tracePath"/> is empty or white space.</exception>
    public static string SummaryPathFor(string tracePath)
    {
        Guard.NotNullOrWhiteSpace(tracePath);
        return tracePath.Trim() + ".meta.json";
    }

    /// <summary>Converts a run identifier into a stable, collision-free file-name stem.</summary>
    /// <param name="runId">The non-blank run identifier.</param>
    /// <returns>A stem containing only <c>A-Z a-z 0-9 . _ -</c>, suffixed with a hash when characters were replaced.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="runId"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="runId"/> is empty or white space.</exception>
    public static string CreateStem(string runId)
    {
        Guard.NotNullOrWhiteSpace(runId);
        string trimmed = runId.Trim();
        var builder = new StringBuilder(trimmed.Length);
        bool replaced = false;
        foreach (char character in trimmed)
        {
            if ((character >= 'a' && character <= 'z') || (character >= 'A' && character <= 'Z') ||
                (character >= '0' && character <= '9') || character == '.' || character == '_' || character == '-')
            {
                builder.Append(character);
                continue;
            }
            builder.Append('_');
            replaced = true;
        }

        string stem = builder.ToString();
        if (stem.Length > 64)
        {
            stem = stem.Substring(0, 64);
            replaced = true;
        }
        if (!replaced) return stem;
        return stem + "-" + EvolutionHash.Compute(trimmed).Substring(0, 12);
    }

    private static string ResolveDirectory(string outputDirectory, string parameterName)
    {
        try
        {
            return Path.GetFullPath(outputDirectory.Trim());
        }
        catch (Exception exception) when (exception is ArgumentException or NotSupportedException or PathTooLongException)
        {
            throw new ArgumentException(
                string.Format(CultureInfo.InvariantCulture, "'{0}' is not a valid output directory.", outputDirectory),
                parameterName, exception);
        }
    }
}
