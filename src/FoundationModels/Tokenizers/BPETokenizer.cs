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
    /// Byte Pair Encoding (BPE) tokenizer implementation.
    /// Used by models like GPT-2, GPT-3, and RoBERTa.
    /// </summary>
    public class BPETokenizer : TokenizerBase
    {
        private Dictionary<string, int> _bpeMerges;
        private Dictionary<string, string> _cache;
        private readonly string _vocabFile;
        private readonly string _mergesFile;
        private readonly Regex _pattern;

        /// <summary>
        /// Initializes a new instance of the BPETokenizer class
        /// </summary>
        /// <param name="vocabFile">Path to vocabulary file</param>
        /// <param name="mergesFile">Path to BPE merges file</param>
        public BPETokenizer(string vocabFile, string mergesFile)
        {
            _vocabFile = vocabFile;
            _mergesFile = mergesFile;
            _bpeMerges = new Dictionary<string, int>();
            _cache = new Dictionary<string, string>();
            
            // GPT-2 style pattern for tokenization
            _pattern = new Regex(
                @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
                RegexOptions.Compiled);
        }

        /// <inheritdoc/>
        public override int MaxSequenceLength => 1024;

        #region Protected Methods

        /// <inheritdoc/>
        protected override async Task LoadVocabularyAsync()
        {
            // Load vocabulary
            if (File.Exists(_vocabFile))
            {
                var lines = await File.ReadAllLinesAsync(_vocabFile);
                for (int i = 0; i < lines.Length; i++)
                {
                    var parts = lines[i].Split('\t');
                    if (parts.Length >= 1)
                    {
                        AddToVocabulary(parts[0], i);
                    }
                }
            }
            else
            {
                // Create default vocabulary for demo
                CreateDefaultVocabulary();
            }

            // Load BPE merges
            if (File.Exists(_mergesFile))
            {
                var lines = await File.ReadAllLinesAsync(_mergesFile);
                for (int i = 0; i < lines.Length; i++)
                {
                    if (!string.IsNullOrWhiteSpace(lines[i]) && !lines[i].StartsWith("#"))
                    {
                        _bpeMerges[lines[i]] = i;
                    }
                }
            }
            else
            {
                // Create default merges for demo
                CreateDefaultMerges();
            }
        }

        /// <inheritdoc/>
        protected override async Task<List<string>> TokenizeInternalAsync(string text)
        {
            var tokens = new List<string>();
            
            // Split text using regex pattern
            var matches = _pattern.Matches(text);
            
            foreach (Match match in matches)
            {
                var word = match.Value;
                
                // Apply BPE to each word
                var bpeTokens = ApplyBPE(word);
                tokens.AddRange(bpeTokens);
            }
            
            return await Task.FromResult(tokens);
        }

        /// <inheritdoc/>
        protected override async Task<string> PostProcessTokensAsync(List<string> tokens)
        {
            // Join tokens and handle special characters
            var text = string.Join("", tokens);
            
            // Replace special tokens used in GPT-2
            text = text.Replace("Ġ", " ");  // Space token
            text = text.Replace("Ċ", "\n"); // Newline token
            
            return await Task.FromResult(text);
        }

        /// <inheritdoc/>
        public override async Task<IReadOnlyList<string>> TokenizeAsync(string text)
        {
            return await TokenizeInternalAsync(text);
        }

        #endregion

        #region Private Methods

        /// <summary>
        /// Applies BPE algorithm to a word
        /// </summary>
        private List<string> ApplyBPE(string word)
        {
            // Check cache first
            if (_cache.TryGetValue(word, out var cached))
            {
                return cached.Split(' ').ToList();
            }

            // Convert word to list of characters
            var wordTokens = new List<string>();
            
            // Add special character for spaces
            if (word.StartsWith(" "))
            {
                wordTokens.Add("Ġ");
                word = word.Substring(1);
            }
            
            // Split into characters
            foreach (char c in word)
            {
                wordTokens.Add(c.ToString());
            }
            
            if (wordTokens.Count == 0)
            {
                return new List<string>();
            }

            // Apply BPE merges
            while (wordTokens.Count > 1)
            {
                var pairs = GetPairs(wordTokens);
                if (pairs.Count == 0)
                {
                    break;
                }

                var bigram = GetMostFrequentPair(pairs);
                if (bigram == null)
                {
                    break;
                }

                wordTokens = MergePair(wordTokens, bigram);
            }

            // Cache result
            var result = string.Join(" ", wordTokens);
            _cache[word] = result;
            
            return wordTokens;
        }

        /// <summary>
        /// Gets all adjacent pairs in the word
        /// </summary>
        private List<Tuple<string, string>> GetPairs(List<string> wordTokens)
        {
            var pairs = new List<Tuple<string, string>>();
            
            for (int i = 0; i < wordTokens.Count - 1; i++)
            {
                pairs.Add(Tuple.Create(wordTokens[i], wordTokens[i + 1]));
            }
            
            return pairs;
        }

        /// <summary>
        /// Gets the most frequent pair based on BPE merges
        /// </summary>
        private Tuple<string, string>? GetMostFrequentPair(List<Tuple<string, string>> pairs)
        {
            Tuple<string, string>? bestPair = null;
            int bestRank = int.MaxValue;
            
            foreach (var pair in pairs.Distinct())
            {
                var mergeKey = $"{pair.Item1} {pair.Item2}";
                if (_bpeMerges.TryGetValue(mergeKey, out var rank) && rank < bestRank)
                {
                    bestRank = rank;
                    bestPair = pair;
                }
            }
            
            return bestPair;
        }

        /// <summary>
        /// Merges a pair in the word tokens
        /// </summary>
        private List<string> MergePair(List<string> wordTokens, Tuple<string, string> pair)
        {
            var newTokens = new List<string>();
            var i = 0;
            
            while (i < wordTokens.Count)
            {
                if (i < wordTokens.Count - 1 && 
                    wordTokens[i] == pair.Item1 && 
                    wordTokens[i + 1] == pair.Item2)
                {
                    newTokens.Add(pair.Item1 + pair.Item2);
                    i += 2;
                }
                else
                {
                    newTokens.Add(wordTokens[i]);
                    i++;
                }
            }
            
            return newTokens;
        }

        /// <summary>
        /// Creates a default vocabulary for demonstration
        /// </summary>
        private void CreateDefaultVocabulary()
        {
            // Add special tokens
            var specialTokens = new[] { "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]" };
            for (int i = 0; i < specialTokens.Length; i++)
            {
                AddToVocabulary(specialTokens[i], i);
            }

            // Add common characters and subwords
            var nextId = specialTokens.Length;
            
            // Single characters
            for (char c = 'a'; c <= 'z'; c++)
            {
                AddToVocabulary(c.ToString(), nextId++);
                AddToVocabulary(c.ToString().ToUpper(), nextId++);
            }
            
            for (char c = '0'; c <= '9'; c++)
            {
                AddToVocabulary(c.ToString(), nextId++);
            }
            
            // Common punctuation
            var punctuation = " .,!?;:'\"-()[]{}@#$%^&*+=<>/\\|`~_\n\t";
            foreach (char c in punctuation)
            {
                AddToVocabulary(c.ToString(), nextId++);
            }
            
            // Space variants
            AddToVocabulary("Ġ", nextId++); // Space token
            AddToVocabulary("Ċ", nextId++); // Newline token
            
            // Common subwords
            var commonSubwords = new[] {
                "ing", "ed", "er", "est", "ly", "tion", "ment", "ness",
                "the", "and", "of", "to", "in", "is", "it", "that"
            };
            
            foreach (var subword in commonSubwords)
            {
                AddToVocabulary(subword, nextId++);
                AddToVocabulary("Ġ" + subword, nextId++); // With space prefix
            }
        }

        /// <summary>
        /// Creates default BPE merges for demonstration
        /// </summary>
        private void CreateDefaultMerges()
        {
            var merges = new[] {
                "t h", "th e", "i n", "e r", "a n", "r e", "o n", "a t",
                "e n", "o r", "t o", "e d", "i t", "a l", "a r", "o u",
                "i s", "l e", "s e", "v e", "c e", "o m", "d e", "b e"
            };
            
            for (int i = 0; i < merges.Length; i++)
            {
                _bpeMerges[merges[i]] = i;
            }
        }

        #endregion
    }
}