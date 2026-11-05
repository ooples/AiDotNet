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
    /// <summary>Tokenizes source code and records where each token sits in it.</summary>
    /// <param name="code">The source code; null is treated as empty.</param>
    /// <param name="language">The programming language of <paramref name="code"/>.</param>
    /// <param name="tokenizer">The tokenizer that splits the code into tokens.</param>
    /// <param name="options">Encoding options; null adds the tokenizer's special tokens.</param>
    /// <param name="filePath">The file the code came from, recorded on the result; null when there is none.</param>
    /// <returns>The tokenization, with each token's character span when the tokenizer reports offsets.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="tokenizer"/> is null.</exception>
    CodeTokenizationResult Tokenize(
        string code,
        ProgramLanguage language,
        ITokenizer tokenizer,
        EncodingOptions? options = null,
        string? filePath = null);
}
