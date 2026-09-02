namespace AiDotNet.Configuration;

/// <summary>Names the directories and files a run writes its best program into.</summary>
/// <remarks>
/// <para>
/// The defaults reproduce the reference OpenEvolve output layout
/// (<c>openevolve/controller.py</c> <c>_save_checkpoint</c> and <c>_save_best_program</c>): a checkpoint snapshot
/// lands in <c>checkpoints/checkpoint_&lt;n&gt;/</c> and the final answer in <c>best/</c>, each holding
/// <c>best_program</c> with the source file's own extension plus <c>best_program_info.json</c>. A run ported from a
/// Python configuration therefore leaves an output tree that existing tooling can read unchanged.
/// </para>
/// <para>
/// <see cref="MaxSourceBytes"/> has no upstream counterpart and exists because the program text is model-generated:
/// upstream writes whatever the model produced straight to disk, whereas a bounded write records the truncation in
/// the info document instead of letting one runaway response fill a volume.
/// </para>
/// <para><b>For Beginners:</b> These settings decide where the winning program is saved and what the files are
/// called. Leave them alone unless you already have tooling that expects different names. Turn
/// <see cref="WriteAtCheckpoints"/> off if you only want the final answer and not a copy at every checkpoint.</para>
/// </remarks>
public sealed class ProgramRunOutputOptions
{
    /// <summary>The largest source size accepted by <see cref="Validate"/>, in bytes.</summary>
    public const int MaxAllowedSourceBytes = 64 * 1024 * 1024;

    /// <summary>Gets or sets the directory holding the run's final answer. Defaults to <c>best</c>.</summary>
    public string BestDirectoryName { get; set; } = "best";

    /// <summary>Gets or sets the directory holding per-checkpoint snapshots. Defaults to <c>checkpoints</c>.</summary>
    public string CheckpointsDirectoryName { get; set; } = "checkpoints";

    /// <summary>Gets or sets the prefix of one checkpoint's directory. Defaults to <c>checkpoint_</c>.</summary>
    public string CheckpointDirectoryPrefix { get; set; } = "checkpoint_";

    /// <summary>Gets or sets the program file name without its extension. Defaults to <c>best_program</c>.</summary>
    /// <remarks>The extension is chosen from the genome's language, so a Python winner is written as <c>best_program.py</c>.</remarks>
    public string ProgramFileNameStem { get; set; } = "best_program";

    /// <summary>Gets or sets the info document's file name. Defaults to <c>best_program_info.json</c>.</summary>
    public string InfoFileName { get; set; } = "best_program_info.json";

    /// <summary>Gets or sets whether a snapshot is written at every checkpoint. Defaults to <c>true</c>.</summary>
    public bool WriteAtCheckpoints { get; set; } = true;

    /// <summary>Gets or sets whether the final answer is written when the run stops. Defaults to <c>true</c>.</summary>
    public bool WriteAtRunEnd { get; set; } = true;

    /// <summary>Gets or sets the largest program written, in UTF-8 bytes. Defaults to 4 MB.</summary>
    /// <remarks>
    /// Longer sources are cut on a character boundary, so the written file stays valid UTF-8, and the info document
    /// records that the program on disk is incomplete.
    /// </remarks>
    public int MaxSourceBytes { get; set; } = 4 * 1024 * 1024;

    /// <summary>Gets or sets an optional run identifier recorded in the info document. Defaults to <c>null</c>.</summary>
    public string? RunId { get; set; }

    /// <summary>Creates an independent copy so a running writer is unaffected by later mutation.</summary>
    /// <returns>A new instance carrying the same settings.</returns>
    public ProgramRunOutputOptions Clone() => new()
    {
        BestDirectoryName = BestDirectoryName,
        CheckpointsDirectoryName = CheckpointsDirectoryName,
        CheckpointDirectoryPrefix = CheckpointDirectoryPrefix,
        ProgramFileNameStem = ProgramFileNameStem,
        InfoFileName = InfoFileName,
        WriteAtCheckpoints = WriteAtCheckpoints,
        WriteAtRunEnd = WriteAtRunEnd,
        MaxSourceBytes = MaxSourceBytes,
        RunId = RunId
    };

    /// <summary>Rejects names that are not usable file-system names and sizes that cannot be enforced.</summary>
    /// <exception cref="ArgumentException">A name is empty, white space, or contains a path separator or other invalid character.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><see cref="MaxSourceBytes"/> is not positive or exceeds its ceiling.</exception>
    public void Validate()
    {
        RequireName(BestDirectoryName, nameof(BestDirectoryName));
        RequireName(CheckpointsDirectoryName, nameof(CheckpointsDirectoryName));
        RequireName(CheckpointDirectoryPrefix, nameof(CheckpointDirectoryPrefix));
        RequireName(ProgramFileNameStem, nameof(ProgramFileNameStem));
        RequireName(InfoFileName, nameof(InfoFileName));
        if (MaxSourceBytes <= 0 || MaxSourceBytes > MaxAllowedSourceBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(MaxSourceBytes), MaxSourceBytes,
                $"Value must be between 1 and {MaxAllowedSourceBytes} bytes.");
        }
    }

    private static void RequireName(string value, string parameterName)
    {
        if (string.IsNullOrWhiteSpace(value))
            throw new ArgumentException($"{parameterName} cannot be empty or white space.", parameterName);
        if (value.IndexOf('/') >= 0 || value.IndexOf('\\') >= 0 || value.IndexOf(':') >= 0 ||
            value.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0 || value == "." || value == "..")
        {
            throw new ArgumentException($"{parameterName} must be a single valid file-system name.", parameterName);
        }
    }
}
