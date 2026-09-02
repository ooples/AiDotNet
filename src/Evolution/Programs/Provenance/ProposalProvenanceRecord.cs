using System.Collections.ObjectModel;
using System.Globalization;
using AiDotNet.Enums;
using AiDotNet.Validation;

namespace AiDotNet.Evolution.Programs.Provenance;

/// <summary>
/// Everything known about one request-and-answer round that a language model was asked to turn into a candidate
/// program: who asked, what was asked, what came back, what it cost, and what the answer turned into.
/// </summary>
/// <remarks>
/// <para>
/// A finished evolutionary run is a tree of programs whose scores are known and whose <em>causes</em> normally are
/// not. Recording one of these per attempt closes that gap: the archive says a program scored 0.83, and the
/// provenance stream says which parent it came from, which model produced it, on which attempt, from which prompt,
/// with which answer, at what token cost, and how long the call took. Chained through
/// <see cref="ParentGenomeId"/> and <see cref="ChildGenomeId"/>, the stream reconstructs the full lineage of any
/// program in the archive — the post-hoc audit and reinforcement-learning data-collection story the reference
/// implementation gets from its per-program JSON files and its checkpoint trace extractors.
/// </para>
/// <para>
/// Failed attempts are recorded too, and are the more interesting half. A run whose model answered with prose
/// nine times out of ten looks, in the archive alone, exactly like a run that was merely unlucky; in the
/// provenance stream the difference is one query. Upstream logs "No valid diffs found" and moves on, leaving
/// nothing to count.
/// </para>
/// <para>
/// Every text field is bounded and every one that can carry model or program output is redacted before it gets
/// here, and re-bounded on construction so a hand-built record cannot smuggle an unbounded string into a file.
/// <see cref="PromptTruncated"/> and <see cref="ResponseTruncated"/> mark the cut so a reader never mistakes a
/// clipped prompt for the whole one. An API key is never part of this record: the operator that writes it has no
/// access to one, and the redaction pass replaces credential-shaped text on the way in.
/// </para>
/// <para><b>For Beginners:</b> This is one page of the lab notebook for an AI-driven search. It says: at this
/// moment, we showed model M this prompt about program P, it replied with this text, the reply cost this many
/// tokens and took this long, and the reply did or did not become a new program. Save these and you can answer
/// "why does this program exist?" months later, or feed the whole record of successes and failures to a training
/// process. Long text is trimmed and anything that looks like a password is replaced before it is written.</para>
/// </remarks>
public sealed class ProposalProvenanceRecord
{
    /// <summary>The schema version stamped on every serialized record.</summary>
    public const int CurrentSchemaVersion = 1;

    /// <summary>The longest identifier retained; a longer one is truncated.</summary>
    public const int MaxIdentifierLength = 160;

    /// <summary>The longest prompt or response text retained; longer text is truncated.</summary>
    public const int MaxTextLength = 32_768;

    /// <summary>The longest detail text retained; a longer reason is truncated.</summary>
    public const int MaxDetailLength = 240;

    /// <summary>The most parent or inspiration identifiers retained; extra entries are dropped.</summary>
    public const int MaxRelatedIdentifiers = 16;

    private static readonly ReadOnlyCollection<string> NoIdentifiers =
        new(new List<string>());

    private readonly string _operatorId = string.Empty;
    private readonly string _operatorVersionHash = string.Empty;
    private readonly string _childGenomeId = string.Empty;
    private readonly string _modelId = string.Empty;
    private readonly string _promptTemplateKey = string.Empty;
    private readonly string _promptHash = string.Empty;
    private readonly string _promptText = string.Empty;
    private readonly string _responseText = string.Empty;
    private readonly string _detail = string.Empty;
    private readonly IReadOnlyList<string> _parentIds = NoIdentifiers;
    private readonly IReadOnlyList<string> _inspirationIds = NoIdentifiers;
    private readonly long _generation;
    private readonly int _island;
    private readonly int _inputTokens;
    private readonly int _outputTokens;
    private readonly double _latencyMilliseconds;

    /// <summary>Initializes a provenance record with the identity every record must carry.</summary>
    /// <param name="proposalId">A stable identifier for the proposal this attempt belongs to.</param>
    /// <param name="evaluationId">
    /// The engine-assigned evaluation identifier of the parent this proposal was derived from. The child's own
    /// identifier does not exist yet when the request is made, so the parent's is what links a record to the run.
    /// </param>
    /// <param name="parentGenomeId">The canonical identity of the program the model was asked to improve.</param>
    /// <param name="attemptNumber">The one-based position of this request within the proposal.</param>
    /// <param name="outcome">What happened to the request.</param>
    /// <exception cref="ArgumentNullException"><paramref name="proposalId"/> or <paramref name="parentGenomeId"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="proposalId"/> or <paramref name="parentGenomeId"/> is empty or white space.</exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <paramref name="evaluationId"/> is negative, <paramref name="attemptNumber"/> is not positive, or
    /// <paramref name="outcome"/> is undefined.
    /// </exception>
    public ProposalProvenanceRecord(
        string proposalId,
        long evaluationId,
        string parentGenomeId,
        int attemptNumber,
        ProgramProposalOutcome outcome)
    {
        Guard.NotNullOrWhiteSpace(proposalId);
        Guard.NotNullOrWhiteSpace(parentGenomeId);
        Guard.Positive(attemptNumber);
        if (evaluationId < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(evaluationId), evaluationId, "Value cannot be negative.");
        }

        if (!Enum.IsDefined(typeof(ProgramProposalOutcome), outcome))
        {
            throw new ArgumentOutOfRangeException(nameof(outcome), outcome, "Value must be a defined outcome.");
        }

        ProposalId = Bound(proposalId.Trim(), MaxIdentifierLength);
        EvaluationId = evaluationId;
        ParentGenomeId = Bound(parentGenomeId.Trim(), MaxIdentifierLength);
        AttemptNumber = attemptNumber;
        Outcome = outcome;
    }

    /// <summary>Gets the schema version this record was built against.</summary>
    public int SchemaVersion => CurrentSchemaVersion;

    /// <summary>Gets the stable identifier of the proposal this attempt belongs to.</summary>
    public string ProposalId { get; }

    /// <summary>Gets the evaluation identifier of the parent this proposal was derived from.</summary>
    public long EvaluationId { get; }

    /// <summary>Gets the canonical identity of the program the model was asked to improve.</summary>
    public string ParentGenomeId { get; }

    /// <summary>Gets the one-based position of this request within its proposal.</summary>
    public int AttemptNumber { get; }

    /// <summary>Gets what happened to the request.</summary>
    public ProgramProposalOutcome Outcome { get; }

    /// <summary>Gets the identifier of the variation operator that made the request.</summary>
    public string OperatorId
    {
        get => _operatorId;
        init => _operatorId = Bound(value, MaxIdentifierLength);
    }

    /// <summary>Gets the operator's version hash, which changes when its prompt or settings change.</summary>
    public string OperatorVersionHash
    {
        get => _operatorVersionHash;
        init => _operatorVersionHash = Bound(value, MaxIdentifierLength);
    }

    /// <summary>Gets the canonical identities of the parent's own parents; empty for a seed program.</summary>
    public IReadOnlyList<string> ParentIds
    {
        get => _parentIds;
        init => _parentIds = BoundIdentifiers(value);
    }

    /// <summary>Gets the canonical identities of the programs offered to the model as inspiration.</summary>
    public IReadOnlyList<string> InspirationIds
    {
        get => _inspirationIds;
        init => _inspirationIds = BoundIdentifiers(value);
    }

    /// <summary>Gets the canonical identity of the program this attempt produced; empty when it produced none.</summary>
    public string ChildGenomeId
    {
        get => _childGenomeId;
        init => _childGenomeId = Bound(value, MaxIdentifierLength);
    }

    /// <summary>Gets the logical generation the proposal belonged to.</summary>
    public long Generation
    {
        get => _generation;
        init => _generation = value < 0L ? 0L : value;
    }

    /// <summary>Gets the zero-based island the proposal targeted.</summary>
    public int Island
    {
        get => _island;
        init => _island = value < 0 ? 0 : value;
    }

    /// <summary>Gets the identifier of the model that answered, as the chat client reported it.</summary>
    public string ModelId
    {
        get => _modelId;
        init => _modelId = Bound(value, MaxIdentifierLength);
    }

    /// <summary>Gets the prompt template the request was rendered from.</summary>
    public string PromptTemplateKey
    {
        get => _promptTemplateKey;
        init => _promptTemplateKey = Bound(value, MaxIdentifierLength);
    }

    /// <summary>Gets a hash of the exact messages sent, so identical requests are recognisable without storing them.</summary>
    /// <remarks>
    /// The hash covers the full untruncated conversation, whereas <see cref="PromptText"/> may be clipped. Two
    /// records with the same hash were sent the same request even when both stored prompts were cut short.
    /// </remarks>
    public string PromptHash
    {
        get => _promptHash;
        init => _promptHash = Bound(value, MaxIdentifierLength);
    }

    /// <summary>Gets the redacted, bounded rendering of the messages that were sent.</summary>
    public string PromptText
    {
        get => _promptText;
        init => _promptText = Bound(value, MaxTextLength);
    }

    /// <summary>Gets whether <see cref="PromptText"/> was cut to fit its budget.</summary>
    public bool PromptTruncated { get; init; }

    /// <summary>Gets the redacted, bounded text the model answered with.</summary>
    public string ResponseText
    {
        get => _responseText;
        init => _responseText = Bound(value, MaxTextLength);
    }

    /// <summary>Gets whether <see cref="ResponseText"/> was cut to fit its budget.</summary>
    public bool ResponseTruncated { get; init; }

    /// <summary>Gets the prompt tokens the provider reported, or zero when it reported none.</summary>
    public int InputTokens
    {
        get => _inputTokens;
        init => _inputTokens = value < 0 ? 0 : value;
    }

    /// <summary>Gets the answer tokens the provider reported, or zero when it reported none.</summary>
    public int OutputTokens
    {
        get => _outputTokens;
        init => _outputTokens = value < 0 ? 0 : value;
    }

    /// <summary>Gets the sum of the reported prompt and answer tokens.</summary>
    public int TotalTokens => InputTokens + OutputTokens;

    /// <summary>Gets when the request was issued, in UTC.</summary>
    public DateTimeOffset RequestedAtUtc { get; init; }

    /// <summary>Gets how long the call took, in milliseconds; zero when it was not measured.</summary>
    public double LatencyMilliseconds
    {
        get => _latencyMilliseconds;
        init => _latencyMilliseconds = value > 0.0 && !double.IsNaN(value) && !double.IsInfinity(value) ? value : 0.0;
    }

    /// <summary>Gets a short, sanitized reason; empty when the attempt succeeded or carried no reason.</summary>
    public string Detail
    {
        get => _detail;
        init => _detail = Bound(Sanitize(value), MaxDetailLength);
    }

    /// <summary>Gets whether this attempt produced a usable child program.</summary>
    public bool IsAccepted =>
        Outcome == ProgramProposalOutcome.Accepted && ChildGenomeId.Length > 0;

    /// <summary>Returns a description that never echoes prompt text, response text, or program source.</summary>
    /// <returns>The proposal, the attempt number, the outcome, and the token total.</returns>
    public override string ToString() => string.Format(
        CultureInfo.InvariantCulture,
        "ProposalProvenanceRecord({0}, #{1}, {2}, {3} tokens)",
        ProposalId,
        AttemptNumber,
        Outcome,
        TotalTokens);

    private static string Bound(string? text, int maximumLength)
    {
        // Narrowed by an explicit null check rather than string.IsNullOrEmpty, which carries no nullable
        // annotation on .NET Framework and would leave the following member access flagged there.
        if (text is null || text.Length == 0) return string.Empty;
        if (text.Length <= maximumLength) return text;
        return maximumLength <= 3 ? text.Substring(0, maximumLength) : text.Substring(0, maximumLength - 3) + "...";
    }

    private static string Sanitize(string? text)
    {
        if (text is null || text.Length == 0) return string.Empty;
        var builder = new System.Text.StringBuilder(text.Length);
        foreach (char character in text)
        {
            if (character == '\t' || character == '\n' || character == '\r') builder.Append(' ');
            else if (char.IsControl(character)) builder.Append('·');
            else builder.Append(character);
        }

        return builder.ToString().Trim();
    }

    private static IReadOnlyList<string> BoundIdentifiers(IReadOnlyList<string>? values)
    {
        if (values is null || values.Count == 0) return NoIdentifiers;

        var bounded = new List<string>(Math.Min(values.Count, MaxRelatedIdentifiers));
        foreach (string value in values)
        {
            if (bounded.Count >= MaxRelatedIdentifiers) break;
            if (string.IsNullOrWhiteSpace(value)) continue;
            bounded.Add(Bound(value.Trim(), MaxIdentifierLength));
        }

        return bounded.Count == 0 ? NoIdentifiers : new ReadOnlyCollection<string>(bounded);
    }
}
