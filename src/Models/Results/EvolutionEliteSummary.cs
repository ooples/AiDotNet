using System.Collections.Generic;

namespace AiDotNet.Models.Results;

/// <summary>One archived elite, described by identity and coordinates rather than by its contents.</summary>
/// <remarks>
/// <para>
/// An evolution archive keeps the best candidate found for each combination of behaviour values. This is the
/// redacted record of one of them: which cell it occupies, how it scored, and which island found it. It carries no
/// genome, because a genome can be arbitrarily large and is often the very thing that should not travel into a log
/// or a saved model file. Program evolution's own summary keeps a size-capped copy of the source alongside this
/// when the caller asks for it.
/// </para>
/// <para><b>For Beginners:</b> Think of the archive as a shelf with one slot per combination of properties — for
/// instance "short and fast" or "long and accurate". This is the label on one slot: what scored there, how well it
/// scored, and where in the grid the slot sits. <see cref="Descriptors"/> gives the measured values by name and
/// <see cref="Cell"/> gives the bin each value fell into.</para>
/// </remarks>
public sealed class EvolutionEliteSummary
{
    private IDictionary<string, double>? _descriptors;
    private IList<int>? _cell;

    /// <summary>Gets or sets the canonical identifier of the archived candidate.</summary>
    public string GenomeId { get; set; } = string.Empty;

    /// <summary>Gets or sets the island the elite was archived on.</summary>
    public int Island { get; set; }

    /// <summary>Gets or sets the quality that was recorded, or <c>null</c> when the candidate was never scored.</summary>
    public double? Quality { get; set; }

    /// <summary>Gets or sets the sequential evaluation identifier that produced this entry.</summary>
    public long EvaluationId { get; set; }

    /// <summary>Gets or sets the measured behaviour values, keyed by descriptor name.</summary>
    public IDictionary<string, double> Descriptors
    {
        get => _descriptors ??= new Dictionary<string, double>(StringComparer.Ordinal);
        set => _descriptors = value;
    }

    /// <summary>Gets or sets the archive cell coordinates, one bin index per descriptor, in descriptor order.</summary>
    public IList<int> Cell
    {
        get => _cell ??= new List<int>();
        set => _cell = value;
    }
}
