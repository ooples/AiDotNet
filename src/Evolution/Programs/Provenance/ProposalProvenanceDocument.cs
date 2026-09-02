using System.Globalization;
using AiDotNet.Enums;

namespace AiDotNet.Evolution.Programs.Provenance;

/// <summary>
/// The on-disk shape of one provenance record. Kept separate from the public record so the file format can be
/// versioned without constraining the public type, exactly as the checkpoint store separates its document from
/// <c>EvolutionCheckpoint</c>.
/// </summary>
/// <remarks>
/// Every property is settable and every value type is nullable-free so Newtonsoft can round-trip it without
/// constructor binding. Timestamps travel as ISO-8601 round-trip strings rather than as a serializer-dependent
/// date representation, so a file written on one machine reads identically on another.
/// </remarks>
internal sealed class ProposalProvenanceDocument
{
    public int SchemaVersion { get; set; }
    public string ProposalId { get; set; } = string.Empty;
    public long EvaluationId { get; set; }
    public string ParentGenomeId { get; set; } = string.Empty;
    public int AttemptNumber { get; set; }
    public string Outcome { get; set; } = string.Empty;
    public string OperatorId { get; set; } = string.Empty;
    public string OperatorVersionHash { get; set; } = string.Empty;
    public List<string> ParentIds { get; set; } = new();
    public List<string> InspirationIds { get; set; } = new();
    public string ChildGenomeId { get; set; } = string.Empty;
    public long Generation { get; set; }
    public int Island { get; set; }
    public string ModelId { get; set; } = string.Empty;
    public string PromptTemplateKey { get; set; } = string.Empty;
    public string PromptHash { get; set; } = string.Empty;
    public string PromptText { get; set; } = string.Empty;
    public bool PromptTruncated { get; set; }
    public string ResponseText { get; set; } = string.Empty;
    public bool ResponseTruncated { get; set; }
    public int InputTokens { get; set; }
    public int OutputTokens { get; set; }
    public string RequestedAtUtc { get; set; } = string.Empty;
    public double LatencyMilliseconds { get; set; }
    public string Detail { get; set; } = string.Empty;

    /// <summary>Projects a public record onto its serialization shape.</summary>
    public static ProposalProvenanceDocument From(ProposalProvenanceRecord record) => new()
    {
        SchemaVersion = record.SchemaVersion,
        ProposalId = record.ProposalId,
        EvaluationId = record.EvaluationId,
        ParentGenomeId = record.ParentGenomeId,
        AttemptNumber = record.AttemptNumber,
        Outcome = record.Outcome.ToString(),
        OperatorId = record.OperatorId,
        OperatorVersionHash = record.OperatorVersionHash,
        ParentIds = new List<string>(record.ParentIds),
        InspirationIds = new List<string>(record.InspirationIds),
        ChildGenomeId = record.ChildGenomeId,
        Generation = record.Generation,
        Island = record.Island,
        ModelId = record.ModelId,
        PromptTemplateKey = record.PromptTemplateKey,
        PromptHash = record.PromptHash,
        PromptText = record.PromptText,
        PromptTruncated = record.PromptTruncated,
        ResponseText = record.ResponseText,
        ResponseTruncated = record.ResponseTruncated,
        InputTokens = record.InputTokens,
        OutputTokens = record.OutputTokens,
        RequestedAtUtc = record.RequestedAtUtc == default
            ? string.Empty
            : record.RequestedAtUtc.ToUniversalTime().ToString("o", CultureInfo.InvariantCulture),
        LatencyMilliseconds = record.LatencyMilliseconds,
        Detail = record.Detail
    };

    /// <summary>Rebuilds the public record, rejecting a document whose required identity is missing.</summary>
    /// <exception cref="InvalidDataException">A required field is absent or malformed.</exception>
    public ProposalProvenanceRecord ToRecord()
    {
        if (string.IsNullOrWhiteSpace(ProposalId))
        {
            throw new InvalidDataException("A provenance record needs a proposal identifier.");
        }

        if (string.IsNullOrWhiteSpace(ParentGenomeId))
        {
            throw new InvalidDataException("A provenance record needs a parent genome identifier.");
        }

        if (AttemptNumber <= 0)
        {
            throw new InvalidDataException("A provenance record needs a positive attempt number.");
        }

        if (EvaluationId < 0L)
        {
            throw new InvalidDataException("A provenance record cannot carry a negative evaluation identifier.");
        }

        ProgramProposalOutcome outcome;
        try
        {
            outcome = (ProgramProposalOutcome)Enum.Parse(typeof(ProgramProposalOutcome), Outcome, ignoreCase: true);
        }
        catch (ArgumentException exception)
        {
            throw new InvalidDataException("A provenance record carries an unknown outcome.", exception);
        }

        if (!Enum.IsDefined(typeof(ProgramProposalOutcome), outcome))
        {
            throw new InvalidDataException("A provenance record carries an unknown outcome.");
        }

        DateTimeOffset requestedAt = default;
        if (!string.IsNullOrWhiteSpace(RequestedAtUtc) &&
            DateTimeOffset.TryParse(
                RequestedAtUtc,
                CultureInfo.InvariantCulture,
                DateTimeStyles.RoundtripKind,
                out DateTimeOffset parsed))
        {
            requestedAt = parsed;
        }

        return new ProposalProvenanceRecord(ProposalId, EvaluationId, ParentGenomeId, AttemptNumber, outcome)
        {
            OperatorId = OperatorId,
            OperatorVersionHash = OperatorVersionHash,
            ParentIds = ParentIds,
            InspirationIds = InspirationIds,
            ChildGenomeId = ChildGenomeId,
            Generation = Generation,
            Island = Island,
            ModelId = ModelId,
            PromptTemplateKey = PromptTemplateKey,
            PromptHash = PromptHash,
            PromptText = PromptText,
            PromptTruncated = PromptTruncated,
            ResponseText = ResponseText,
            ResponseTruncated = ResponseTruncated,
            InputTokens = InputTokens,
            OutputTokens = OutputTokens,
            RequestedAtUtc = requestedAt,
            LatencyMilliseconds = LatencyMilliseconds,
            Detail = Detail
        };
    }
}
