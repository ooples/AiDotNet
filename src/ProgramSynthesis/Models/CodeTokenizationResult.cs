using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.Tokenization.Models;

namespace AiDotNet.ProgramSynthesis.Models;

/// <summary>
/// Represents the result of code-aware tokenization, including token IDs and token-to-source spans.
/// </summary>
/// <remarks>
/// <para>
/// This type is designed to reuse the existing tokenization stack (ITokenizer) while providing
/// code-specific structural metadata (spans) that is useful for downstream tasks (search, review,
/// diagnostics, etc.).
/// </para>
/// </remarks>
public sealed class CodeTokenizationResult
{
    public ProgramLanguage Language { get; set; } = ProgramLanguage.Generic;

    public string? FilePath { get; set; }

    public TokenizationResult Tokenization { get; set; } = new();

    /// <summary>
    /// Best-effort mapping from each token to a span in the original source code.
    /// </summary>
    /// <remarks>
    /// When the underlying tokenizer does not provide offsets, this may be empty.
    /// When offsets are provided, this is aligned with <see cref="TokenizationResult.Tokens"/>.
    /// </remarks>
    public List<CodeSpan> TokenSpans { get; set; } = new();

    /// <summary>
    /// Optional AST nodes extracted during tokenization (when enabled and supported for the language).
    /// </summary>
    public List<CodeAstNode> AstNodes { get; set; } = new();

    /// <summary>
    /// Optional relationships between AST nodes (when enabled).
    /// </summary>
    public List<CodeAstEdge> AstEdges { get; set; } = new();
}
