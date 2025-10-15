using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace AiDotNet.FoundationModels.Tokenizers
{
    /// <summary>
    /// WordPiece tokenizer implementation used by BERT and similar models.
    /// Splits words into subword units to handle out-of-vocabulary words.
    /// </summary>
    public class WordPieceTokenizer : TokenizerBase
    {
        private readonly string _vocabFile;
        private readonly bool _doLowerCase;
        private readonly int _maxInputCharsPerWord;
        private readonly string _unkToken;
        private readonly string _continuationPrefix;

        /// <summary>
        /// Initializes a new instance of the WordPieceTokenizer class
        /// </summary>
        /// <param name="vocabFile">Path to vocabulary file</param>
        /// <param name="doLowerCase">Whether to lowercase input text</param>
        /// <param name="maxInputCharsPerWord">Maximum characters per word</param>
        public WordPieceTokenizer(
            string vocabFile, 
            bool doLowerCase = true,
            int maxInputCharsPerWord = 100)
        {
            _vocabFile = vocabFile;
            _doLowerCase = doLowerCase;
            _maxInputCharsPerWord = maxInputCharsPerWord;
            _unkToken = "[UNK]";
            _continuationPrefix = "##";
        }

        /// <inheritdoc/>
        public override int MaxSequenceLength => 512; // BERT's default

        /// <inheritdoc/>
        protected override void InitializeSpecialTokens()
        {
            base.InitializeSpecialTokens();
            
            // BERT-specific special tokens
            _specialTokens["[PAD]"] = 0;
            _specialTokens["[UNK]"] = 100;
            _specialTokens["[CLS]"] = 101;
            _specialTokens["[SEP]"] = 102;
            _specialTokens["[MASK]"] = 103;
        }

        #region Protected Methods

        /// <inheritdoc/>
        protected override async Task LoadVocabularyAsync()
        {
            if (File.Exists(_vocabFile))
            {
                var lines = await File.ReadAllLinesAsync(_vocabFile);
                for (int i = 0; i < lines.Length; i++)
                {
                    var token = lines[i].Trim();
                    if (!string.IsNullOrEmpty(token))
                    {
                        AddToVocabulary(token, i);
                    }
                }
            }
            else
            {
                // Create default BERT vocabulary for demo
                CreateDefaultBertVocabulary();
            }
        }

        /// <inheritdoc/>
        protected override async Task<List<string>> TokenizeInternalAsync(string text)
        {
            var tokens = new List<string>();
            
            // Normalize and clean text
            text = NormalizeText(text);
            
            if (_doLowerCase)
            {
                text = text.ToLower();
            }
            
            // Split on whitespace and punctuation
            var words = BasicTokenize(text);
            
            // Apply WordPiece to each word
            foreach (var word in words)
            {
                if (word.Length > _maxInputCharsPerWord)
                {
                    tokens.Add(_unkToken);
                    continue;
                }
                
                var wordPieceTokens = WordPieceTokenize(word);
                tokens.AddRange(wordPieceTokens);
            }
            
            return await Task.FromResult(tokens);
        }

        /// <inheritdoc/>
        protected override async Task<string> PostProcessTokensAsync(List<string> tokens)
        {
            var result = new StringBuilder();
            
            for (int i = 0; i < tokens.Count; i++)
            {
                var token = tokens[i];
                
                // Skip special tokens
                if (IsSpecialToken(_vocabulary.ContainsKey(token) ? _vocabulary[token] : -1))
                {
                    continue;
                }
                
                if (token.StartsWith(_continuationPrefix) && result.Length > 0)
                {
                    // Remove ## prefix and append directly
                    result.Append(token.Substring(_continuationPrefix.Length));
                }
                else
                {
                    // Add space before token if not first
                    if (result.Length > 0)
                    {
                        result.Append(" ");
                    }
                    result.Append(token);
                }
            }
            
            return await Task.FromResult(result.ToString());
        }

        /// <inheritdoc/>
        public override async Task<IReadOnlyList<string>> TokenizeAsync(string text)
        {
            var tokens = await TokenizeInternalAsync(text);
            return tokens;
        }

        #endregion

        #region Private Methods

        /// <summary>
        /// Normalizes text by handling unicode and special characters
        /// </summary>
        private string NormalizeText(string text)
        {
            // Remove control characters and normalize whitespace
            var normalized = Regex.Replace(text, @"\s+", " ");
            normalized = Regex.Replace(normalized, @"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "");
            
            return normalized.Trim();
        }

        /// <summary>
        /// Basic tokenization splitting on whitespace and punctuation
        /// </summary>
        private List<string> BasicTokenize(string text)
        {
            var tokens = new List<string>();
            var currentToken = new StringBuilder();
            
            foreach (char c in text)
            {
                if (char.IsWhiteSpace(c))
                {
                    if (currentToken.Length > 0)
                    {
                        tokens.Add(currentToken.ToString());
                        currentToken.Clear();
                    }
                }
                else if (IsPunctuation(c))
                {
                    if (currentToken.Length > 0)
                    {
                        tokens.Add(currentToken.ToString());
                        currentToken.Clear();
                    }
                    tokens.Add(c.ToString());
                }
                else
                {
                    currentToken.Append(c);
                }
            }
            
            if (currentToken.Length > 0)
            {
                tokens.Add(currentToken.ToString());
            }
            
            return tokens;
        }

        /// <summary>
        /// Checks if a character is punctuation
        /// </summary>
        private bool IsPunctuation(char c)
        {
            // Common punctuation marks
            return ".,!?;:'\"-()[]{}@#$%^&*+=<>/\\|`~".Contains(c);
        }

        /// <summary>
        /// Applies WordPiece tokenization to a single word
        /// </summary>
        private List<string> WordPieceTokenize(string word)
        {
            var outputTokens = new List<string>();
            var start = 0;
            
            while (start < word.Length)
            {
                var end = word.Length;
                string? curSubstr = null;
                
                while (start < end)
                {
                    var substr = word.Substring(start, end - start);
                    
                    if (start > 0)
                    {
                        substr = _continuationPrefix + substr;
                    }
                    
                    if (_vocabulary.ContainsKey(substr))
                    {
                        curSubstr = substr;
                        break;
                    }
                    
                    end--;
                }
                
                if (curSubstr == null)
                {
                    // Couldn't find any valid subword
                    outputTokens.Clear();
                    outputTokens.Add(_unkToken);
                    break;
                }
                
                outputTokens.Add(curSubstr);
                start = end;
            }
            
            return outputTokens;
        }

        /// <summary>
        /// Creates a default BERT vocabulary for demonstration
        /// </summary>
        private void CreateDefaultBertVocabulary()
        {
            var tokenId = 0;
            
            // Special tokens
            AddToVocabulary("[PAD]", tokenId++);
            AddToVocabulary("[UNK]", tokenId++);
            AddToVocabulary("[CLS]", tokenId++);
            AddToVocabulary("[SEP]", tokenId++);
            AddToVocabulary("[MASK]", tokenId++);
            
            // Unused tokens (BERT convention)
            for (int i = 0; i < 100; i++)
            {
                AddToVocabulary($"[unused{i}]", tokenId++);
            }
            
            // Single characters
            for (char c = 'a'; c <= 'z'; c++)
            {
                AddToVocabulary(c.ToString(), tokenId++);
                if (_doLowerCase == false)
                {
                    AddToVocabulary(c.ToString().ToUpper(), tokenId++);
                }
            }
            
            // Numbers
            for (char c = '0'; c <= '9'; c++)
            {
                AddToVocabulary(c.ToString(), tokenId++);
            }
            
            // Common punctuation
            var punctuation = ".,!?;:'\"-()[]{}@#$%^&*+=<>/\\|`~ \n\t";
            foreach (char c in punctuation)
            {
                AddToVocabulary(c.ToString(), tokenId++);
            }
            
            // Common subwords with ## prefix
            var commonSubwords = new[] {
                "ing", "ed", "er", "est", "ly", "tion", "ment", "ness", "able",
                "ful", "less", "ish", "ous", "ive", "ize", "ise", "ate", "ity",
                "al", "ic", "ian", "ist", "ism", "ship", "hood", "dom", "age"
            };
            
            foreach (var subword in commonSubwords)
            {
                AddToVocabulary("##" + subword, tokenId++);
            }
            
            // Common words
            var commonWords = new[] {
                "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
                "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
                "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
                "or", "an", "will", "my", "one", "all", "would", "there", "their",
                "what", "so", "up", "out", "if", "about", "who", "get", "which", "go"
            };
            
            foreach (var word in commonWords)
            {
                AddToVocabulary(word, tokenId++);
            }
            
            // Common prefixes and suffixes
            var prefixes = new[] { "un", "re", "pre", "dis", "over", "under", "mis", "out" };
            foreach (var prefix in prefixes)
            {
                AddToVocabulary(prefix, tokenId++);
                AddToVocabulary("##" + prefix, tokenId++);
            }
        }

        #endregion
    }
}