using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.Tokenization.Algorithms;
using AiDotNet.Tokenization.CodeTokenization;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.Tokenization.Models;

namespace AiDotNet.ProgramSynthesis.Tokenization;

/// <summary>
/// Creates safe, code-aware default tokenizers for <see cref="ProgramLanguage"/> values.
/// </summary>
public static class ProgramSynthesisTokenizerFactory
{
    public static ITokenizer CreateDefault(ProgramLanguage language, bool splitIdentifiers = true)
    {
        var baseTokenizer = CharacterTokenizer.CreateAscii(SpecialTokens.Bert(), lowercase: false);

        var codeLanguage = language switch
        {
            ProgramLanguage.CSharp => ProgrammingLanguage.CSharp,
            ProgramLanguage.Python => ProgrammingLanguage.Python,
            ProgramLanguage.Java => ProgrammingLanguage.Java,
            ProgramLanguage.JavaScript => ProgrammingLanguage.JavaScript,
            ProgramLanguage.TypeScript => ProgrammingLanguage.TypeScript,
            ProgramLanguage.C => ProgrammingLanguage.C,
            ProgramLanguage.CPlusPlus => ProgrammingLanguage.Cpp,
            ProgramLanguage.Go => ProgrammingLanguage.Go,
            ProgramLanguage.Rust => ProgrammingLanguage.Rust,
            ProgramLanguage.SQL => ProgrammingLanguage.SQL,
            _ => ProgrammingLanguage.Generic
        };

        return new CodeTokenizer(baseTokenizer, codeLanguage, splitIdentifiers: splitIdentifiers);
    }
}

