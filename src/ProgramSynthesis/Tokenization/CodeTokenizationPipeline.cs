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
    private readonly Interfaces.ICodeSyntaxProvider? _syntaxProvider;

    /// <summary>Creates a pipeline that uses the host's fallback parser, if any, for syntax trees.</summary>
    public CodeTokenizationPipeline()
        : this(null)
    {
    }

    /// <summary>Creates a pipeline that extracts syntax trees with <paramref name="syntaxProvider"/>.</summary>
    /// <param name="syntaxProvider">The parser to use; null uses the host's fallback parser, if any.</param>
    public CodeTokenizationPipeline(Interfaces.ICodeSyntaxProvider? syntaxProvider)
    {
        _syntaxProvider = syntaxProvider;
    }

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

        if (pipelineOptions.IncludeAst && (_syntaxProvider ?? Engines.CodeSyntaxProviders.Provider) is { } syntaxProvider)
        {
            // A provider's native grammar is optional: when it cannot load, the tokenization still stands and the
            // result says why it has no syntax tree.
            try
            {
                if (syntaxProvider.TryExtractAst(code, language, pipelineOptions, out var nodes, out var edges))
                {
                    result.AstNodes = nodes;
                    result.AstEdges = edges;
                }
            }
            catch (Exception ex) when (IsNativeLoadFailure(ex))
            {
                result.AstUnavailableReason =
                    $"The syntax provider could not load its native grammar for {language} ({ex.GetType().Name}); the result has no syntax tree.";
            }
        }

        return result;
    }

    /// <summary>
    /// True when a syntax provider failed because its native grammar library could not load (missing, wrong
    /// architecture, missing entry point), including a failure wrapped by a type initializer. Only these warrant a
    /// fallback to generic handling; a parser's own errors are real failures.
    /// </summary>
    internal static bool IsNativeLoadFailure(Exception ex) =>
        ex is DllNotFoundException or BadImageFormatException or EntryPointNotFoundException
        || (ex is TypeInitializationException { InnerException: { } inner } && IsNativeLoadFailure(inner));

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
