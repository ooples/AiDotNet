using System.Globalization;
using AiDotNet.Enums;
using AiDotNet.Validation;

namespace AiDotNet.Evolution.Programs;

/// <summary>One recorded request-and-answer round while a language model was asked to improve a program.</summary>
/// <remarks>
/// <para>
/// The operator keeps a bounded, ordered log of these so a finished run can be explained without re-reading provider
/// logs: which parent was being improved, how many times the model had to be asked, what went wrong each time, and
/// how many tokens the round cost. Nothing here echoes the model's answer or the program text. <see cref="Detail"/>
/// carries only the short, control-character-sanitized reason the operator itself generated, and a provider failure
/// contributes its exception <em>type name</em> and nothing else, so an endpoint or a key cannot reach a log through
/// this type.
/// </para>
/// <para>
/// Attempts are numbered from one within a single proposal, so <see cref="AttemptNumber"/> restarts for each parent
/// rather than counting across the run; pair it with <see cref="ParentGenomeId"/> to group a conversation. The
/// terminal <see cref="ProgramProposalOutcome.Exhausted"/> record shares the number of the last failed request.
/// </para>
/// <para><b>For Beginners:</b> Think of this as one line in a diary of the conversation with the AI. It says which
/// program was being improved, whether that particular reply was usable, and roughly what it cost. After a run you
/// can read the diary to see whether the model was mostly succeeding or mostly being asked to try again, which is
/// usually the quickest way to tell whether a prompt or a model choice is the problem.</para>
/// </remarks>
public sealed class ProgramProposalAttempt
{
    /// <summary>The longest detail text retained; longer reasons are truncated.</summary>
    public const int MaxDetailLength = 240;

    /// <summary>Initializes a recorded attempt.</summary>
    /// <param name="parentGenomeId">The identity of the program the model was asked to improve.</param>
    /// <param name="attemptNumber">The one-based position of this request within the proposal.</param>
    /// <param name="outcome">What happened to the request.</param>
    /// <param name="detail">A short reason, truncated and sanitized; <c>null</c> becomes empty.</param>
    /// <param name="inputTokens">Prompt tokens reported by the provider, or zero when it reported none.</param>
    /// <param name="outputTokens">Answer tokens reported by the provider, or zero when it reported none.</param>
    /// <exception cref="ArgumentException"><paramref name="parentGenomeId"/> is empty or white space.</exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <paramref name="attemptNumber"/> is not positive, <paramref name="outcome"/> is undefined, or a token count
    /// is negative.
    /// </exception>
    public ProgramProposalAttempt(
        string parentGenomeId,
        int attemptNumber,
        ProgramProposalOutcome outcome,
        string? detail = null,
        int inputTokens = 0,
        int outputTokens = 0)
    {
        Guard.NotNullOrWhiteSpace(parentGenomeId);
        Guard.Positive(attemptNumber);
        if (!Enum.IsDefined(typeof(ProgramProposalOutcome), outcome))
        {
            throw new ArgumentOutOfRangeException(nameof(outcome), outcome, "Value must be a defined outcome.");
        }

        if (inputTokens < 0) throw new ArgumentOutOfRangeException(nameof(inputTokens), inputTokens, "Value cannot be negative.");
        if (outputTokens < 0) throw new ArgumentOutOfRangeException(nameof(outputTokens), outputTokens, "Value cannot be negative.");

        ParentGenomeId = parentGenomeId.Trim();
        AttemptNumber = attemptNumber;
        Outcome = outcome;
        Detail = detail is null ? string.Empty : ProgramText.Bound(ProgramText.Sanitize(detail), MaxDetailLength);
        InputTokens = inputTokens;
        OutputTokens = outputTokens;
    }

    /// <summary>Gets the identity of the program the model was asked to improve.</summary>
    public string ParentGenomeId { get; }

    /// <summary>Gets the one-based position of this request within its proposal.</summary>
    public int AttemptNumber { get; }

    /// <summary>Gets what happened to the request.</summary>
    public ProgramProposalOutcome Outcome { get; }

    /// <summary>Gets a short, sanitized reason; empty when the attempt succeeded or carried no reason.</summary>
    public string Detail { get; }

    /// <summary>Gets the prompt tokens the provider reported, or zero when it reported none.</summary>
    public int InputTokens { get; }

    /// <summary>Gets the answer tokens the provider reported, or zero when it reported none.</summary>
    public int OutputTokens { get; }

    /// <summary>Gets the sum of the reported prompt and answer tokens.</summary>
    public int TotalTokens => InputTokens + OutputTokens;

    /// <summary>Gets whether this attempt produced a usable child program.</summary>
    public bool IsAccepted => Outcome == ProgramProposalOutcome.Accepted;

    /// <summary>Returns a description that never echoes the model's answer or the program text.</summary>
    /// <returns>The parent identity prefix, the attempt number, and the outcome.</returns>
    public override string ToString() =>
        "ProgramProposalAttempt(" +
        (ParentGenomeId.Length > 12 ? ParentGenomeId.Substring(0, 12) : ParentGenomeId) +
        ", #" + AttemptNumber.ToString(CultureInfo.InvariantCulture) + ", " + Outcome + ")";
}
