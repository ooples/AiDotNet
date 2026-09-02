using AiDotNet.Evolution.Programs.Provenance;

namespace AiDotNet.Interfaces;

/// <summary>Receives one provenance record per language-model request made while proposing a candidate program.</summary>
/// <remarks>
/// <para>
/// An evolutionary run produces programs whose scores are recorded and whose causes usually are not. A sink is
/// where the causes go: every request the variation operator makes, successful or not, is offered here with the
/// parent it started from, the model that answered, the bounded and redacted prompt and answer, the token cost,
/// the timing, and what the answer parsed into. That stream is what makes a finished run auditable afterwards and
/// what turns it into training data.
/// </para>
/// <para>
/// Implementations must tolerate concurrent calls, because several evolution workers normally share one operator.
/// They should also avoid throwing: the operator treats a sink failure as a recording problem, counts it, and lets
/// the run continue, since losing a note is never a reason to lose a search. <c>JsonLinesProposalProvenanceSink</c>
/// writes crash-safe JSON Lines segments to a directory; <c>InMemoryProposalProvenanceSink</c> keeps a bounded list
/// for tests and short runs.
/// </para>
/// <para>
/// Records may contain untrusted model output, already bounded and redacted by the writer. A sink must not
/// interpret that text — never execute it, never expand it into a template, never use it to build a path.
/// </para>
/// <para><b>For Beginners:</b> This is where the library sends its notes about every conversation it had with the
/// AI while searching for a better program. Point it at the built-in file writer and you get a log you can read
/// later to answer "where did this program come from?" — or delete the sink entirely and the search runs exactly
/// as before, just without the notes.</para>
/// </remarks>
public interface IProposalProvenanceSink
{
    /// <summary>Records one request-and-answer round.</summary>
    /// <param name="record">The provenance record; never <c>null</c>.</param>
    /// <param name="cancellationToken">Token used to cancel the write.</param>
    /// <returns>A task that completes when the record has been accepted, though not necessarily flushed to disk.</returns>
    /// <remarks>
    /// Called once per language-model request, possibly from several threads at once. Implementations should be
    /// cheap and should not throw; a buffering implementation may complete synchronously.
    /// </remarks>
    Task RecordAsync(ProposalProvenanceRecord record, CancellationToken cancellationToken = default);
}
