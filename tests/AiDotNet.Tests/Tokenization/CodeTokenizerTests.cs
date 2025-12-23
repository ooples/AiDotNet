using System.Collections.Generic;
using AiDotNet.Tokenization.Algorithms;
using AiDotNet.Tokenization.CodeTokenization;
using AiDotNet.Tokenization.Models;
using AiDotNet.Tokenization.Vocabulary;
using Xunit;

namespace AiDotNet.Tests.Tokenization
{
    public class CodeTokenizerTests
    {
        [Fact]
        public void Tokenize_SplitsCamelCaseIdentifiers()
        {
            // Arrange
            var vocab = new Vocabulary("[UNK]");
            vocab.AddTokens(new[] { "get", "user", "name", "by", "id" });

            var baseTokenizer = new WordPieceTokenizer(vocab, SpecialTokens.Bert());
            var codeTokenizer = new CodeTokenizer(baseTokenizer, ProgrammingLanguage.CSharp, splitIdentifiers: true);

            // Act
            var tokens = codeTokenizer.Tokenize("getUserNameById");

            // Assert
            Assert.Contains("get", tokens);
            Assert.Contains("user", tokens);
            Assert.Contains("name", tokens);
        }

        [Fact]
        public void Tokenize_SplitsSnakeCaseIdentifiers()
        {
            // Arrange
            var vocab = new Vocabulary("[UNK]");
            vocab.AddTokens(new[] { "get", "user", "name" });

            var baseTokenizer = new WordPieceTokenizer(vocab, SpecialTokens.Bert());
            var codeTokenizer = new CodeTokenizer(baseTokenizer, ProgrammingLanguage.Python, splitIdentifiers: true);

            // Act
            var tokens = codeTokenizer.Tokenize("get_user_name");

            // Assert
            Assert.Contains("get", tokens);
            Assert.Contains("user", tokens);
            Assert.Contains("name", tokens);
        }

        [Fact]
        public void Tokenize_RecognizesCSharpKeywords()
        {
            // Arrange
            var vocab = new Vocabulary("[UNK]");
            vocab.AddTokens(new[] { "public", "class", "void", "if", "return" });

            var baseTokenizer = new WordPieceTokenizer(vocab, SpecialTokens.Bert());
            var codeTokenizer = new CodeTokenizer(baseTokenizer, ProgrammingLanguage.CSharp);

            // Act
            var tokens = codeTokenizer.Tokenize("public class if void return");

            // Assert
            Assert.Contains("public", tokens);
            Assert.Contains("class", tokens);
            Assert.Contains("void", tokens);
            Assert.Contains("if", tokens);
            Assert.Contains("return", tokens);
        }

        [Fact]
        public void Tokenize_RecognizesPythonKeywords()
        {
            // Arrange
            var vocab = new Vocabulary("[UNK]");
            vocab.AddTokens(new[] { "def", "class", "if", "return", "import" });

            var baseTokenizer = new WordPieceTokenizer(vocab, SpecialTokens.Bert());
            var codeTokenizer = new CodeTokenizer(baseTokenizer, ProgrammingLanguage.Python);

            // Act
            var tokens = codeTokenizer.Tokenize("def class if return import");

            // Assert
            Assert.Contains("def", tokens);
            Assert.Contains("class", tokens);
            Assert.Contains("if", tokens);
            Assert.Contains("return", tokens);
            Assert.Contains("import", tokens);
        }

        [Fact]
        public void CodeBertTokenizer_EncodesCodeAndNL()
        {
            // Arrange
            var vocab = new Vocabulary("[UNK]");
            vocab.AddTokens(new[] { "[PAD]", "[CLS]", "[SEP]", "[MASK]" });
            vocab.AddTokens(new[] { "return", "sum", "of", "two", "numbers", "a", "b", "+", "def", "add" });

            var codeBert = new CodeBertTokenizer(vocab, ProgrammingLanguage.Python);

            // Act
            var result = codeBert.EncodeCodeAndNL(
                code: "def add(a, b): return a + b",
                naturalLanguage: "return sum of two numbers");

            // Assert
            Assert.NotNull(result);
            Assert.NotEmpty(result.TokenIds);
            Assert.NotEmpty(result.TokenTypeIds);
            Assert.Contains(0, result.TokenTypeIds); // NL segment
            Assert.Contains(1, result.TokenTypeIds); // Code segment
        }

        [Fact]
        public void CodeBertTokenizer_EncodesCodeOnly()
        {
            // Arrange
            var vocab = new Vocabulary("[UNK]");
            vocab.AddTokens(new[] { "[PAD]", "[CLS]", "[SEP]", "def", "add", "return", "a", "b", "+", "(", ")", ":", "," });

            var codeBert = new CodeBertTokenizer(vocab, ProgrammingLanguage.Python);

            // Act
            var result = codeBert.EncodeCodeAndNL(code: "def add(a, b): return a + b");

            // Assert
            Assert.NotNull(result);
            Assert.NotEmpty(result.TokenIds);
            Assert.Contains("[CLS]", result.Tokens);
            Assert.Contains("[SEP]", result.Tokens);
        }
    }
}
