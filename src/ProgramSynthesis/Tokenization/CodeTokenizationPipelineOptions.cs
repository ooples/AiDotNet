namespace AiDotNet.ProgramSynthesis.Tokenization;

/// <summary>
/// Options for <see cref="CodeTokenizationPipeline"/> that control optional structural extraction.
/// </summary>
public sealed class CodeTokenizationPipelineOptions
{
    /// <summary>
    /// Gets or sets whether to extract a syntax tree (AST nodes and edges) alongside the tokens.
    /// </summary>
    /// <value>
    /// Default: <c>false</c>. Extraction needs a registered code syntax provider (for example the TreeSitter add-on);
    /// without one, or when the provider cannot load its native grammar, the result has tokens only and
    /// <see cref="AiDotNet.ProgramSynthesis.Models.CodeTokenizationResult.AstUnavailableReason"/> says why when the grammar failed to load.
    /// </value>
    public bool IncludeAst { get; set; }

    /// <summary>
    /// Gets or sets the most syntax-tree nodes extracted from one input; extraction stops once this many are collected.
    /// </summary>
    /// <value>Default: 10,000. Values below 1 are treated as 1.</value>
    public int MaxAstNodes { get; set; } = 10_000;
}

