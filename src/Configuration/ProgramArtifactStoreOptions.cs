namespace AiDotNet.Configuration;

/// <summary>Limits and retention rules applied to stored evaluation artifacts.</summary>
/// <remarks>
/// <para>
/// The two defaults that matter are taken from the reference OpenEvolve database configuration
/// (<c>openevolve/config.py</c>): <see cref="InlineSizeThresholdBytes"/> is upstream's
/// <c>artifact_size_threshold</c> of 32 KB, the point at which an artifact stops being serialized into the program
/// record and gets a file of its own, and <see cref="RetentionPeriod"/> is upstream's
/// <c>artifact_retention_days</c> of 30. Keeping the numbers identical means a run ported from a Python
/// configuration splits and expires its artifacts at the same points.
/// </para>
/// <para>
/// The remaining settings are bounds upstream does not enforce. Artifact content is produced by code a model wrote
/// against data the library did not choose, so a single evaluation can hand over a gigabyte of standard error or
/// ten thousand named outputs. <see cref="MaxArtifactBytes"/> caps one artifact, <see cref="MaxArtifactsPerGenome"/>
/// caps how many a genome may accumulate, and <see cref="MaxTotalBytesPerGenome"/> caps their combined size.
/// Exceeding the byte caps truncates and flags rather than failing, so a noisy program is recorded honestly instead
/// of aborting a run; exceeding the count cap is rejected, because that is a contract error rather than noise.
/// </para>
/// <para><b>For Beginners:</b> These settings decide how much of what your candidate programs print is kept, and
/// for how long. Small outputs go in one shared index file; anything over
/// <see cref="InlineSizeThresholdBytes"/> gets its own file. Anything older than <see cref="RetentionPeriod"/> is
/// deleted the next time you ask the store to tidy up. The defaults are sensible; raise the caps if you are
/// capturing large traces on purpose, and lower them if disk space matters more than detail.</para>
/// </remarks>
public sealed class ProgramArtifactStoreOptions
{
    /// <summary>The reference implementation's <c>artifact_size_threshold</c>, 32 KB.</summary>
    public const int DefaultInlineSizeThresholdBytes = 32 * 1024;

    /// <summary>The reference implementation's <c>artifact_retention_days</c>, 30 days.</summary>
    public const int DefaultRetentionDays = 30;

    /// <summary>The largest inline threshold accepted by <see cref="Validate"/>, in bytes.</summary>
    public const int MaxInlineSizeThresholdBytes = 4 * 1024 * 1024;

    /// <summary>Gets or sets the size at or below which an artifact is stored inline. Defaults to 32 KB.</summary>
    /// <remarks>Matches the reference implementation, which stores content at or below the threshold in the record.</remarks>
    public int InlineSizeThresholdBytes { get; set; } = DefaultInlineSizeThresholdBytes;

    /// <summary>Gets or sets the largest content kept for one artifact, in bytes. Defaults to 8 MB.</summary>
    /// <remarks>Longer content is truncated and the stored descriptor reports the truncation.</remarks>
    public int MaxArtifactBytes { get; set; } = 8 * 1024 * 1024;

    /// <summary>Gets or sets how many artifacts one genome may accumulate. Defaults to 64.</summary>
    /// <remarks>A store call that would exceed this is rejected rather than silently dropping evidence.</remarks>
    public int MaxArtifactsPerGenome { get; set; } = 64;

    /// <summary>Gets or sets the combined size of one genome's artifacts, in bytes. Defaults to 64 MB.</summary>
    /// <remarks>A store call that would exceed this is rejected rather than silently dropping evidence.</remarks>
    public long MaxTotalBytesPerGenome { get; set; } = 64L * 1024L * 1024L;

    /// <summary>Gets or sets how long artifacts are retained. Defaults to 30 days.</summary>
    /// <remarks><see cref="TimeSpan.Zero"/> disables age-based expiry so nothing is removed by age.</remarks>
    public TimeSpan RetentionPeriod { get; set; } = TimeSpan.FromDays(DefaultRetentionDays);

    /// <summary>Gets or sets how many genomes keep artifacts, oldest evicted first. Defaults to 0 for unlimited.</summary>
    public int MaxRetainedGenomes { get; set; }

    /// <summary>Creates an independent copy so a running store is unaffected by later mutation.</summary>
    /// <returns>A new instance carrying the same limits.</returns>
    public ProgramArtifactStoreOptions Clone() => new()
    {
        InlineSizeThresholdBytes = InlineSizeThresholdBytes,
        MaxArtifactBytes = MaxArtifactBytes,
        MaxArtifactsPerGenome = MaxArtifactsPerGenome,
        MaxTotalBytesPerGenome = MaxTotalBytesPerGenome,
        RetentionPeriod = RetentionPeriod,
        MaxRetainedGenomes = MaxRetainedGenomes
    };

    /// <summary>Rejects limits that cannot be enforced or that describe an unusable store.</summary>
    /// <exception cref="ArgumentOutOfRangeException">
    /// A limit is not positive where a positive value is required, is negative where a non-negative value is
    /// required, or exceeds its hard ceiling.
    /// </exception>
    public void Validate()
    {
        if (InlineSizeThresholdBytes < 0 || InlineSizeThresholdBytes > MaxInlineSizeThresholdBytes)
            throw new ArgumentOutOfRangeException(nameof(InlineSizeThresholdBytes), InlineSizeThresholdBytes,
                $"Value must be between 0 and {MaxInlineSizeThresholdBytes} bytes.");
        if (MaxArtifactBytes <= 0)
            throw new ArgumentOutOfRangeException(nameof(MaxArtifactBytes), MaxArtifactBytes, "Value must be positive.");
        if (MaxArtifactsPerGenome <= 0)
            throw new ArgumentOutOfRangeException(nameof(MaxArtifactsPerGenome), MaxArtifactsPerGenome, "Value must be positive.");
        if (MaxTotalBytesPerGenome <= 0)
            throw new ArgumentOutOfRangeException(nameof(MaxTotalBytesPerGenome), MaxTotalBytesPerGenome, "Value must be positive.");
        if (RetentionPeriod < TimeSpan.Zero)
            throw new ArgumentOutOfRangeException(nameof(RetentionPeriod), RetentionPeriod, "Value cannot be negative.");
        if (MaxRetainedGenomes < 0)
            throw new ArgumentOutOfRangeException(nameof(MaxRetainedGenomes), MaxRetainedGenomes, "Value cannot be negative.");
    }
}
