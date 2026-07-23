using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tokenization.Algorithms;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.Tokenization.Models;

namespace AiDotNet.Tokenization.CodeTokenization
{
    /// <summary>
    /// CodeBERT-compatible tokenizer for program synthesis and code understanding tasks.
    /// Combines WordPiece tokenization with code-aware preprocessing.
    /// </summary>
    public class CodeBertTokenizer
    {
        private readonly CodeTokenizer _codeTokenizer;
        private readonly WordPieceTokenizer _wordPieceTokenizer;

        /// <summary>
        /// Gets the underlying tokenizer.
        /// </summary>
        public ITokenizer Tokenizer => _codeTokenizer;

        /// <summary>
        /// Creates a new CodeBERT tokenizer.
        /// </summary>
        /// <param name="vocabulary">The vocabulary.</param>
        /// <param name="language">The programming language.</param>
        /// <param name="specialTokens">The special tokens (BERT-style by default).</param>
        public CodeBertTokenizer(
            IVocabulary vocabulary,
            ProgrammingLanguage language = ProgrammingLanguage.Generic,
            SpecialTokens? specialTokens = null)
        {
            specialTokens ??= SpecialTokens.Bert();
            _wordPieceTokenizer = new WordPieceTokenizer(vocabulary, specialTokens);
            _codeTokenizer = new CodeTokenizer(_wordPieceTokenizer, language, splitIdentifiers: true);
        }

        /// <summary>
        /// Encodes code and natural language for CodeBERT.
        /// </summary>
        /// <param name="code">The code snippet.</param>
        /// <param name="naturalLanguage">The natural language description (optional).</param>
        /// <param name="options">Encoding options.</param>
        /// <returns>The tokenization result.</returns>
        public TokenizationResult EncodeCodeAndNL(
            string code,
            string? naturalLanguage = null,
            EncodingOptions? options = null)
        {
            options ??= new EncodingOptions { AddSpecialTokens = true };

            var codeTokens = _codeTokenizer.Tokenize(code);
            var allTokens = new List<string>();

            // Add [CLS] token
            allTokens.Add(_codeTokenizer.SpecialTokens.ClsToken);

            // Add natural language tokens if provided
            if (naturalLanguage is string nl && !string.IsNullOrEmpty(nl))
            {
                var nlTokens = _wordPieceTokenizer.Tokenize(nl);
                allTokens.AddRange(nlTokens);
                allTokens.Add(_codeTokenizer.SpecialTokens.SepToken);
            }

            // Add code tokens
            allTokens.AddRange(codeTokens);
            allTokens.Add(_codeTokenizer.SpecialTokens.SepToken);

            // Truncate if necessary, preserving NL/code boundary when naturalLanguage is provided
            if (options.Truncation && options.MaxLength.HasValue && allTokens.Count > options.MaxLength.Value)
            {
                var maxLen = options.MaxLength.Value;

                if (!string.IsNullOrEmpty(naturalLanguage))
                {
                    // Find the first SEP position to preserve NL/code boundary
                    var firstSepIndex = allTokens.IndexOf(_codeTokenizer.SpecialTokens.SepToken);

                    if (firstSepIndex >= 0 && firstSepIndex < maxLen - 1)
                    {
                        // Preserve tokens up to first SEP, truncate remaining
                        var tokensBeforeSep = allTokens.Take(firstSepIndex + 1).ToList();
                        var remaining = maxLen - tokensBeforeSep.Count - 1; // -1 for final SEP
                        var tokensAfterSep = allTokens.Skip(firstSepIndex + 1).Take(remaining).ToList();
                        allTokens = tokensBeforeSep.Concat(tokensAfterSep).ToList();
                    }
                    else
                    {
                        allTokens = allTokens.Take(maxLen - 1).ToList();
                    }
                }
                else
                {
                    allTokens = allTokens.Take(maxLen - 1).ToList();
                }

                // Add SEP only if not already present at end to avoid duplicates
                if (allTokens.Count == 0 || allTokens[allTokens.Count - 1] != _codeTokenizer.SpecialTokens.SepToken)
                {
                    allTokens.Add(_codeTokenizer.SpecialTokens.SepToken);
                }
            }

            // Convert to IDs
            var tokenIds = _codeTokenizer.ConvertTokensToIds(allTokens);

            // Create attention mask and token type IDs
            var attentionMask = Enumerable.Repeat(1, tokenIds.Count).ToList();

            var tokenTypeIds = new List<int>();
            if (!string.IsNullOrEmpty(naturalLanguage))
            {
                // Segment IDs: 0 for NL, 1 for code
                var sepIndices = new List<int>();
                for (int i = 0; i < allTokens.Count; i++)
                {
                    if (allTokens[i] == _codeTokenizer.SpecialTokens.SepToken)
                        sepIndices.Add(i);
                }

                for (int i = 0; i < allTokens.Count; i++)
                {
                    if (sepIndices.Count > 0 && i <= sepIndices[0])
                        tokenTypeIds.Add(0); // NL segment
                    else
                        tokenTypeIds.Add(1); // Code segment
                }
            }
            else
            {
                tokenTypeIds = new List<int>(new int[tokenIds.Count]); // All zeros
            }

            // Pad if necessary
            if (options.Padding && options.MaxLength.HasValue)
            {
                var paddingLength = options.MaxLength.Value - tokenIds.Count;
                if (paddingLength > 0)
                {
                    var padTokenId = _codeTokenizer.Vocabulary.GetTokenId(_codeTokenizer.SpecialTokens.PadToken);
                    for (int i = 0; i < paddingLength; i++)
                    {
                        tokenIds.Add(padTokenId);
                        allTokens.Add(_codeTokenizer.SpecialTokens.PadToken);
                        attentionMask.Add(0);
                        tokenTypeIds.Add(0);
                    }
                }
            }

            return new TokenizationResult
            {
                Tokens = allTokens,
                TokenIds = tokenIds,
                AttentionMask = attentionMask,
                TokenTypeIds = tokenTypeIds
            };
        }

        /// <summary>
        /// Decodes token IDs back to code.
        /// </summary>
        public string Decode(List<int> tokenIds, bool skipSpecialTokens = true)
        {
            return _codeTokenizer.Decode(tokenIds, skipSpecialTokens);
        }
    }
}
