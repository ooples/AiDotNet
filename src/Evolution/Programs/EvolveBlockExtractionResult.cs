using System.Collections.ObjectModel;
using AiDotNet.Enums;

namespace AiDotNet.Evolution.Programs;

/// <summary>The outcome of scanning a program source for evolve-block markers.</summary>
/// <remarks>
/// <para>
/// The reference OpenEvolve parser returns only the list of well-formed blocks: a start marker with no matching
/// end, a second start marker inside an open block, and a stray end marker are all discarded in silence, so a file
/// that lost a marker looks exactly like a file that never had one. This result keeps the recovered
/// <see cref="Regions"/> and additionally reports a <see cref="Status"/> naming the most severe anomaly plus one
/// bounded human-readable <see cref="Diagnostics"/> entry per anomaly, so a caller can distinguish "this file has
/// no evolve block" from "the model deleted my end marker" and respond differently.
/// </para>
/// <para>
/// When several anomalies occur the reported <see cref="Status"/> is the most severe one, ordered
/// <see cref="EvolveBlockStatus.Unterminated"/>, <see cref="EvolveBlockStatus.RestartedBlock"/>,
/// <see cref="EvolveBlockStatus.UnmatchedEnd"/>, then <see cref="EvolveBlockStatus.Complete"/> or
/// <see cref="EvolveBlockStatus.NotPresent"/>. Diagnostics are capped at 32 entries so a pathological response
/// cannot bloat a checkpoint or a log.
/// </para>
/// <para><b>For Beginners:</b> This is the report you get back after looking for the editable region of a file. It
/// tells you how many editable regions were found, hands you each one ready to rewrite, and explains anything that
/// looked wrong. Check <see cref="Status"/> first: if it is <see cref="EvolveBlockStatus.Complete"/> you can use
/// <see cref="Regions"/> with confidence, and anything else means the file's markers were damaged and you should
/// probably ask for the code again rather than guess.</para>
/// </remarks>
public sealed class EvolveBlockExtractionResult
{
    /// <summary>The maximum number of diagnostics retained by one extraction.</summary>
    public const int MaxDiagnostics = 32;

    private readonly ReadOnlyCollection<EvolveBlockRegion> _regions;
    private readonly ReadOnlyCollection<string> _diagnostics;

    internal EvolveBlockExtractionResult(
        EvolveBlockStatus status,
        IReadOnlyList<EvolveBlockRegion> regions,
        IReadOnlyList<string> diagnostics)
    {
        Status = status;
        var regionCopy = new EvolveBlockRegion[regions.Count];
        for (int index = 0; index < regions.Count; index++) regionCopy[index] = regions[index];
        _regions = Array.AsReadOnly(regionCopy);

        int diagnosticCount = Math.Min(diagnostics.Count, MaxDiagnostics);
        var diagnosticCopy = new string[diagnosticCount];
        for (int index = 0; index < diagnosticCount; index++) diagnosticCopy[index] = diagnostics[index];
        _diagnostics = Array.AsReadOnly(diagnosticCopy);
    }

    /// <summary>Gets the most severe outcome observed while scanning the source.</summary>
    public EvolveBlockStatus Status { get; }

    /// <summary>Gets the well-formed blocks, in the order their start markers appear.</summary>
    public IReadOnlyList<EvolveBlockRegion> Regions => _regions;

    /// <summary>Gets bounded human-readable notes describing every anomaly that was found.</summary>
    public IReadOnlyList<string> Diagnostics => _diagnostics;

    /// <summary>Gets whether at least one well-formed block was recovered.</summary>
    public bool HasRegions => _regions.Count > 0;

    /// <summary>Gets whether the source was free of marker anomalies.</summary>
    public bool IsWellFormed => Status == EvolveBlockStatus.Complete || Status == EvolveBlockStatus.NotPresent;

    /// <summary>Gets the first well-formed block, which is the one a single-region rewrite targets.</summary>
    /// <param name="region">The first block when one exists; otherwise a default region.</param>
    /// <returns><c>true</c> when at least one block was recovered.</returns>
    public bool TryGetPrimaryRegion(out EvolveBlockRegion region)
    {
        if (_regions.Count == 0)
        {
            region = default;
            return false;
        }

        region = _regions[0];
        return true;
    }
}
