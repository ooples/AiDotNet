namespace AiDotNet.ProgramSynthesis.Options;

/// <summary>
/// Defines the built-in program synthesis model implementations that can be configured via the primary builder/result APIs.
/// </summary>
public enum ProgramSynthesisModelKind
{
    /// <summary>
    /// CodeBERT (encoder-only) for code understanding tasks.
    /// </summary>
    CodeBERT,

    /// <summary>
    /// GraphCodeBERT (encoder-only) with optional data-flow modeling.
    /// </summary>
    GraphCodeBERT,

    /// <summary>
    /// CodeT5 (encoder-decoder) for code generation and transformation tasks.
    /// </summary>
    CodeT5
}
