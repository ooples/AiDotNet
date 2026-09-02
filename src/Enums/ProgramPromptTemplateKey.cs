namespace AiDotNet.Enums;

/// <summary>Names one overridable prompt template used when a language model evolves a program.</summary>
/// <remarks>
/// <para>
/// Every value corresponds to exactly one piece of prompt text that
/// <see cref="AiDotNet.Evolution.Prompts.ProgramPromptTemplateSet"/> resolves, and each has a fixed set of
/// placeholders declared alongside it. Because the keys are an enumeration rather than free-form strings, an
/// override for a template that does not exist is a compile error instead of a run-time lookup failure, and the
/// complete set of overridable texts is discoverable from the type itself.
/// </para>
/// <para>
/// The reference OpenEvolve implementation names templates with strings that are matched against whatever
/// <c>*.txt</c> files happen to be in a directory, so a typo silently leaves the shipped default in place and a
/// missing template raises an error only when the first prompt is built. Typed keys plus configure-time
/// validation remove both failure modes.
/// </para>
/// <para><b>For Beginners:</b> A prompt is assembled from several reusable pieces of text: the standing
/// instructions, the request itself, and the little sections that describe past attempts and example programs.
/// This list names each of those pieces so you can replace any one of them with your own wording. Pick the key
/// you want to change, supply your text, and the rest of the prompt keeps working exactly as before. If your
/// replacement text refers to a value the piece does not receive, you are told immediately rather than halfway
/// through a long run.</para>
/// </remarks>
public enum ProgramPromptTemplateKey
{
    /// <summary>The standing instructions sent as the system message when proposing a change.</summary>
    SystemMessage = 0,

    /// <summary>The standing instructions sent as the system message when a model judges a program.</summary>
    EvaluatorSystemMessage = 1,

    /// <summary>The request asking for SEARCH/REPLACE edit blocks.</summary>
    DiffUser = 2,

    /// <summary>The request asking for a complete replacement program.</summary>
    FullRewriteUser = 3,

    /// <summary>The wrapper that joins previous attempts, top programs, and inspirations.</summary>
    EvolutionHistory = 4,

    /// <summary>The rendering of one earlier attempt and how it turned out.</summary>
    PreviousAttempt = 5,

    /// <summary>The rendering of one high-scoring program shown as an example.</summary>
    TopProgram = 6,

    /// <summary>The wrapper around the list of inspiration programs.</summary>
    InspirationsSection = 7,

    /// <summary>The rendering of one inspiration program.</summary>
    InspirationProgram = 8,

    /// <summary>The request asking a model to score a program on named criteria.</summary>
    Evaluation = 9,

    /// <summary>The extra system instructions that require a changes description to be maintained.</summary>
    SystemMessageChangesDescription = 10,

    /// <summary>The wrapper that appends the changes-description instructions to the system message.</summary>
    SystemMessageWithChangesDescription = 11,

    /// <summary>The wrapper that appends the current changes description to the user message.</summary>
    UserMessageWithChangesDescription = 12
}
