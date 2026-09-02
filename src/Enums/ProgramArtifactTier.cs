namespace AiDotNet.Enums;

/// <summary>Says where an artifact store physically keeps one artifact's bytes.</summary>
/// <remarks>
/// <para>
/// Evaluation artifacts vary in size by orders of magnitude: a one-line assertion message and a fifty-megabyte
/// profiler trace arrive through the same channel. The reference OpenEvolve database splits them at
/// <c>artifact_size_threshold</c> (32 KB by default, <c>openevolve/config.py</c>), serializing everything at or
/// below the threshold into the program record and writing everything above it to a per-program directory. This
/// enumeration reports which side of that split an artifact landed on, so a caller can tell a cheap read from an
/// expensive one before making it.
/// </para>
/// <para><b>For Beginners:</b> Small outputs are kept together in one small index file, which is fast to read.
/// Large outputs get a file of their own so the index stays small. This tells you which of the two happened for a
/// particular output.</para>
/// </remarks>
public enum ProgramArtifactTier
{
    /// <summary>The bytes are stored inside the per-genome index document.</summary>
    Inline = 0,

    /// <summary>The bytes are stored in a file of their own beside the index document.</summary>
    OnDisk = 1
}
