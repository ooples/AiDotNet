using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Models;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.Tokenization.Models;

namespace AiDotNet.ProgramSynthesis.Tokenization;

/// <summary>
/// Defines a code tokenization pipeline that builds on the core tokenizer stack and adds code-oriented metadata.
/// </summary>
public interface ICodeTokenizationPipeline
{
    CodeTokenizationResult Tokenize(
        string code,
        ProgramLanguage language,
        ITokenizer tokenizer,
        EncodingOptions? options = null,
        string? filePath = null);
}
