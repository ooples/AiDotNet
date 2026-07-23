using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Models;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.Tokenization.Models;

namespace AiDotNet.ProgramSynthesis.Tokenization;

/// <summary>
/// Default implementation of <see cref="ICodeTokenizationPipeline"/>.
/// </summary>
public sealed class CodeTokenizationPipeline : ICodeTokenizationPipeline
{
    public CodeTokenizationResult Tokenize(
        string code,
        ProgramLanguage language,
        ITokenizer tokenizer,
        EncodingOptions? options = null,
        string? filePath = null)
    {
        if (tokenizer is null)
        {
            throw new ArgumentNullException(nameof(tokenizer));
        }

        var encodingOptions = options ?? new EncodingOptions { AddSpecialTokens = true };
        var tokenization = tokenizer.Encode(code ?? string.Empty, encodingOptions);

        var result = new CodeTokenizationResult
        {
            Language = language,
            FilePath = filePath,
            Tokenization = tokenization
        };

        if (tokenization.Offsets.Count > 0 && tokenization.Offsets.Count == tokenization.Tokens.Count)
        {
            result.TokenSpans = BuildSpansFromOffsets(code ?? string.Empty, tokenization.Offsets);
        }

        return result;
    }

    public CodeTokenizationResult TokenizeWithStructure(
        string code,
        ProgramLanguage language,
        ITokenizer tokenizer,
        CodeTokenizationPipelineOptions pipelineOptions,
        EncodingOptions? options = null,
        string? filePath = null)
    {
        if (pipelineOptions is null)
        {
            throw new ArgumentNullException(nameof(pipelineOptions));
        }

        var result = Tokenize(code, language, tokenizer, options, filePath);

        if (pipelineOptions.IncludeAst &&
            TreeSitterAstExtractor.TryExtractAst(code, language, pipelineOptions, out var nodes, out var edges))
        {
            result.AstNodes = nodes;
            result.AstEdges = edges;
        }

        return result;
    }

    private static List<CodeSpan> BuildSpansFromOffsets(string text, List<(int Start, int End)> offsets)
    {
        var lineStarts = CodeSpanBuilder.ComputeLineStarts(text);
        var spans = new List<CodeSpan>(offsets.Count);

        foreach (var (start, end) in offsets)
        {
            spans.Add(CodeSpanBuilder.CreateSpan(lineStarts, start, end));
        }

        return spans;
    }
}
