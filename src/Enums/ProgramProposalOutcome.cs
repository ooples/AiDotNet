namespace AiDotNet.Enums;

/// <summary>Records what happened to one request sent to a language model while proposing a candidate program.</summary>
/// <remarks>
/// <para>
/// A single proposal may take several requests: the operator asks for a change, and if the answer cannot be turned
/// into a program it explains the problem and asks again. Every request produces exactly one of these outcomes, so
/// the recorded sequence for a proposal reads as the story of that conversation — for example two
/// <see cref="ParseFailed"/> entries followed by an <see cref="Accepted"/> one. The reference OpenEvolve worker has
/// no equivalent: an answer whose edits do not apply is logged as "No valid diffs found" and the iteration is lost,
/// leaving nothing to count or diagnose afterwards.
/// </para>
/// <para>
/// <see cref="Exhausted"/> is the only value that is not tied to a single request. It is recorded once when every
/// permitted attempt has failed and the operator gives the parent genome back unchanged, which the engine then
/// recognises as a duplicate so that the failed proposal costs no evaluation budget at all.
/// </para>
/// <para><b>For Beginners:</b> When an AI is asked to improve a program, its answer is not always usable — it might
/// reply with prose instead of code, ask to replace text that is not in the file, or hand back exactly what it was
/// given. This list names each of those situations so you can see, after a run, how often the model succeeded and
/// what went wrong the rest of the time. A run with many <see cref="ParseFailed"/> entries usually means the prompt
/// needs to state the required answer format more firmly, or that the model is too small for the task.</para>
/// </remarks>
public enum ProgramProposalOutcome
{
    /// <summary>The answer produced a usable child program that differs from its parent.</summary>
    Accepted = 0,

    /// <summary>The model returned nothing, or only white space.</summary>
    EmptyResponse = 1,

    /// <summary>The answer could not be parsed: no usable edit block, or no fenced code block in rewrite mode.</summary>
    ParseFailed = 2,

    /// <summary>The proposed program was byte-identical to the parent once normalized.</summary>
    Unchanged = 3,

    /// <summary>The proposed program exceeded the configured program size limit.</summary>
    TooLong = 4,

    /// <summary>The chat client threw; only the exception type is recorded, never its message.</summary>
    ProviderError = 5,

    /// <summary>Every permitted attempt failed, so the parent was returned unchanged and no budget was spent.</summary>
    Exhausted = 6
}
