namespace AiDotNet.ProgramSynthesis.Tokenization;

/// <summary>
/// Options for <see cref="CodeTokenizationPipeline"/> that control optional structural extraction.
/// </summary>
public sealed class CodeTokenizationPipelineOptions
{
    public bool IncludeAst { get; set; }

    public int MaxAstNodes { get; set; } = 10_000;
}

