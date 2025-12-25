using System.Linq;
using System.Reflection;
using AiDotNet.Tokenization.CodeTokenization;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.Tokenization.Models;
using AiDotNet.Tokenization.Vocabulary;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Tokenization;

public sealed class TreeSitterTokenizerFallbackTests
{
    [Fact]
    public void Tokenize_FallsBackToBaseTokenizer_WhenTreeSitterUnavailable()
    {
        var baseTokenizer = new SimpleTestTokenizer();
        var tokenizer = new TreeSitterTokenizer(baseTokenizer, language: TreeSitterLanguage.Python);

        ForceTreeSitterUnavailable(tokenizer);

        var tokens = tokenizer.Tokenize("print('hi')");
        Assert.Equal(baseTokenizer.Tokenize("print('hi')"), tokens);
    }

    [Fact]
    public void FactoryHelpers_CreateTokenizers_AndDispose()
    {
        var baseTokenizer = new SimpleTestTokenizer();

        using var python = TreeSitterTokenizer.CreatePython(baseTokenizer);
        using var csharp = TreeSitterTokenizer.CreateCSharp(baseTokenizer);
        using var java = TreeSitterTokenizer.CreateJava(baseTokenizer);

        ForceTreeSitterUnavailable(python);
        ForceTreeSitterUnavailable(csharp);
        ForceTreeSitterUnavailable(java);

        Assert.NotEmpty(python.Tokenize("print('hi')"));
        Assert.NotEmpty(csharp.Tokenize("class C {}"));
        Assert.NotEmpty(java.Tokenize("class C {}"));
    }

    private static void ForceTreeSitterUnavailable(TreeSitterTokenizer tokenizer)
    {
        var treeSitterAvailable = typeof(TreeSitterTokenizer).GetField("_treeSitterAvailable", BindingFlags.NonPublic | BindingFlags.Instance);
        Assert.NotNull(treeSitterAvailable);
        treeSitterAvailable!.SetValue(tokenizer, false);

        var parser = typeof(TreeSitterTokenizer).GetField("_parser", BindingFlags.NonPublic | BindingFlags.Instance);
        Assert.NotNull(parser);
        parser!.SetValue(tokenizer, null);
    }

    private sealed class SimpleTestTokenizer : ITokenizer
    {
        private readonly Vocabulary _vocabulary = new();
        private readonly SpecialTokens _specialTokens = SpecialTokens.Default();

        public IVocabulary Vocabulary => _vocabulary;

        public SpecialTokens SpecialTokens => _specialTokens;

        public int VocabularySize => _vocabulary.Size;

        public TokenizationResult Encode(string text, EncodingOptions? options = null)
        {
            var tokens = Tokenize(text);
            var ids = ConvertTokensToIds(tokens);
            return new TokenizationResult(tokens, ids);
        }

        public List<TokenizationResult> EncodeBatch(List<string> texts, EncodingOptions? options = null)
        {
            return texts.Select(text => Encode(text, options)).ToList();
        }

        public string Decode(List<int> tokenIds, bool skipSpecialTokens = true)
        {
            var tokens = ConvertIdsToTokens(tokenIds);
            return string.Join(" ", tokens);
        }

        public List<string> DecodeBatch(List<List<int>> tokenIdsBatch, bool skipSpecialTokens = true)
        {
            return tokenIdsBatch.Select(batch => Decode(batch, skipSpecialTokens)).ToList();
        }

        public List<string> Tokenize(string text)
        {
            return string.IsNullOrWhiteSpace(text)
                ? []
                : text.Split(new[] { ' ', '\t', '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries).ToList();
        }

        public List<int> ConvertTokensToIds(List<string> tokens)
        {
            return tokens.Select(token => _vocabulary.AddToken(token)).ToList();
        }

        public List<string> ConvertIdsToTokens(List<int> ids)
        {
            return ids.Select(id => _vocabulary.GetToken(id) ?? _specialTokens.UnkToken).ToList();
        }
    }
}
