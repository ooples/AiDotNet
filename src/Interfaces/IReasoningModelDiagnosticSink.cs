using AiDotNet.Agentic.Models.Connectors;

namespace AiDotNet.Interfaces;

/// <summary>Receives the record of every adjustment a reasoning-model profile makes to an outgoing request.</summary>
/// <remarks>
/// <para>
/// A reasoning model rejects some of the sampling settings an ordinary chat model accepts, so the library removes
/// them, renames one, and sometimes adds a deliberation level. Dropping a caller's setting without saying so would
/// make a run impossible to explain, so every edit is offered to this sink. Implement it to forward the record to
/// whatever logging, metrics, or audit trail the host already uses; <c>CollectingReasoningModelDiagnosticSink</c>
/// is a ready-made in-memory implementation for tests and short runs.
/// </para>
/// <para>
/// An implementation must be safe to call from several threads at once, because one chat client is normally shared
/// by concurrent evolution workers. It must also not throw: the connectors treat a sink failure as a reporting
/// problem and continue, but a sink that throws on every call turns a diagnostic path into a hot exception path.
/// Keep the implementation cheap — it runs once per adjusted setting per request.
/// </para>
/// <para><b>For Beginners:</b> When the library has to change your request so a newer "reasoning" model will accept
/// it, it tells you. This interface is where those notices go. Write a class that implements it and, say, prints
/// each notice to your log; hand it to the chat client and you will never be surprised by a setting that quietly
/// stopped applying.</para>
/// </remarks>
public interface IReasoningModelDiagnosticSink
{
    /// <summary>Records one adjustment made to an outgoing request.</summary>
    /// <param name="diagnostic">The adjustment; never <c>null</c>.</param>
    /// <remarks>
    /// Called once per adjusted setting per request, possibly from several threads at once. Implementations should
    /// return quickly and should not throw.
    /// </remarks>
    void Report(ReasoningModelDiagnostic diagnostic);
}
