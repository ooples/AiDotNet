using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.FoundationModels.Tokenizers
{
    /// <summary>
    /// Byte-level BPE tokenizer similar to the one used by GPT-2/GPT-3.
    /// Converts text to bytes and applies BPE on the byte sequence.
    /// </summary>
    public class ByteLevelBPETokenizer : TokenizerBase
    {
        private Dictionary<Tuple<string, string>, int> _merges;
        private Dictionary<string, int> _bytesToUnicode;
        private Dictionary<int, string> _unicodeToBytes;
        private readonly int _vocabSize;
        private readonly Regex _pattern;
        private Dictionary<string, string> _cache;

        /// <inheritdoc/>
        public override int MaxSequenceLength => 2048; // GPT-2 default

        /// <summary>
        /// Initializes a new instance of the ByteLevelBPETokenizer class
        /// </summary>
        /// <param name="vocabSize">Target vocabulary size</param>
        public ByteLevelBPETokenizer(int vocabSize = 50257) // GPT-2 vocab size
        {
            _vocabSize = vocabSize;
            _merges = new Dictionary<Tuple<string, string>, int>();
            _bytesToUnicode = new Dictionary<string, int>();
            _unicodeToBytes = new Dictionary<int, string>();
            _cache = new Dictionary<string, string>();
            
            // GPT-2 style regex pattern for tokenization
            _pattern = new Regex(
                @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
                RegexOptions.Compiled | RegexOptions.IgnoreCase
            );
            
            InitializeBytesToUnicode();
        }

        /// <summary>
        /// Initializes the byte to unicode mapping
        /// </summary>
        private void InitializeBytesToUnicode()
        {
            // Create a mapping from bytes to unicode characters that are "safe" to use
            var bytes = new List<int>();
            
            // Add printable ASCII (except some problematic ones)
            for (int i = 33; i < 127; i++) // Skip space (32) and DEL (127)
            {
                bytes.Add(i);
            }
            
            for (int i = 161; i < 173; i++)
            {
                bytes.Add(i);
            }
            
            for (int i = 174; i < 256; i++)
            {
                bytes.Add(i);
            }

            var chars = new List<int>(bytes);
            int n = 0;
            
            // For bytes not in our "safe" list, map to unused Unicode points
            for (int b = 0; b < 256; b++)
            {
                if (!bytes.Contains(b))
                {
                    bytes.Add(b);
                    chars.Add(256 + n);
                    n++;
                }
            }

            // Create bidirectional mapping
            for (int i = 0; i < bytes.Count; i++)
            {
                string byteStr = ((char)bytes[i]).ToString();
                _bytesToUnicode[byteStr] = chars[i];
                _unicodeToBytes[chars[i]] = byteStr;
            }
        }

        /// <summary>
        /// Loads the vocabulary and merges
        /// </summary>
        protected override async Task LoadVocabularyAsync()
        {
            _vocabulary.Clear();
            _reverseVocabulary.Clear();

            int tokenId = 0;

            // Add special tokens
            foreach (var specialToken in _specialTokens)
            {
                _vocabulary[specialToken.Key] = tokenId;
                _reverseVocabulary[tokenId] = specialToken.Key;
                tokenId++;
            }

            // Add byte tokens
            for (int i = 0; i < 256; i++)
            {
                string token = ConvertBytesToUnicode(new[] { (byte)i });
                _vocabulary[token] = tokenId;
                _reverseVocabulary[tokenId] = token;
                tokenId++;
            }

            // In a real implementation, we would load pre-trained merges here
            // For now, we'll create some basic merges
            await CreateBasicMerges(tokenId);
        }

        /// <summary>
        /// Creates basic BPE merges
        /// </summary>
        private async Task CreateBasicMerges(int startTokenId)
        {
            int tokenId = startTokenId;
            
            // Common English bigrams and trigrams
            var commonMerges = new List<Tuple<string, string>>
            {
                Tuple.Create("t", "h"), Tuple.Create("h", "e"), Tuple.Create("i", "n"), Tuple.Create("e", "r"), Tuple.Create("a", "n"),
                Tuple.Create("r", "e"), Tuple.Create("n", "d"), Tuple.Create("o", "n"), Tuple.Create("e", "n"), Tuple.Create("a", "t"),
                Tuple.Create("o", "u"), Tuple.Create("e", "d"), Tuple.Create("i", "t"), Tuple.Create("i", "s"), Tuple.Create("a", "l"),
                Tuple.Create("o", "r"), Tuple.Create("a", "r"), Tuple.Create("t", "o"), Tuple.Create("e", "s"), Tuple.Create("l", "e"),
                Tuple.Create("th", "e"), Tuple.Create("in", "g"), Tuple.Create("an", "d"), Tuple.Create("er", "e"), Tuple.Create("on", "e"),
                Tuple.Create("re", "d"), Tuple.Create("ou", "r"), Tuple.Create("ti", "on"), Tuple.Create("en", "t"), Tuple.Create("es", "t")
            };

            foreach (var tuple in commonMerges)
            {
                var first = tuple.Item1;
                var second = tuple.Item2;
                if (tokenId >= _vocabSize) break;
                
                string merged = first + second;
                if (!_vocabulary.ContainsKey(merged))
                {
                    _merges[Tuple.Create(first, second)] = _merges.Count;
                    _vocabulary[merged] = tokenId;
                    _reverseVocabulary[tokenId] = merged;
                    tokenId++;
                }
            }

            await Task.CompletedTask;
        }

        /// <summary>
        /// Converts bytes to unicode string representation
        /// </summary>
        private string ConvertBytesToUnicode(byte[] bytes)
        {
            var sb = new StringBuilder();
            foreach (byte b in bytes)
            {
                string byteStr = ((char)b).ToString();
                if (_bytesToUnicode.TryGetValue(byteStr, out int unicode))
                {
                    sb.Append((char)unicode);
                }
                else
                {
                    sb.Append((char)b);
                }
            }
            return sb.ToString();
        }

        /// <summary>
        /// Converts unicode string back to bytes
        /// </summary>
        private byte[] ConvertUnicodeToBytes(string text)
        {
            var bytes = new List<byte>();
            foreach (char c in text)
            {
                if (_unicodeToBytes.TryGetValue((int)c, out string? byteStr))
                {
                    bytes.Add((byte)byteStr[0]);
                }
                else if (c < 256)
                {
                    bytes.Add((byte)c);
                }
            }
            return bytes.ToArray();
        }

        /// <summary>
        /// Applies BPE to a word
        /// </summary>
        private string ApplyBPE(string word)
        {
            // Check cache
            if (_cache.TryGetValue(word, out string? cachedResult))
            {
                return cachedResult;
            }

            // Convert word to list of characters
            var wordTokens = word.Select(c => c.ToString()).ToList();
            
            if (wordTokens.Count == 1)
            {
                _cache[word] = word;
                return word;
            }

            // Apply merges
            while (true)
            {
                var pairs = GetPairs(wordTokens);
                if (pairs.Count == 0) break;

                // Find the best pair to merge
                var bestPair = pairs
                    .Where(p => _merges.ContainsKey(p))
                    .OrderBy(p => _merges[p])
                    .FirstOrDefault();

                if (bestPair == null) break;

                // Merge the best pair
                var newWordTokens = new List<string>();
                int i = 0;
                
                while (i < wordTokens.Count)
                {
                    if (i < wordTokens.Count - 1 && 
                        wordTokens[i] == bestPair.Item1 && 
                        wordTokens[i + 1] == bestPair.Item2)
                    {
                        newWordTokens.Add(bestPair.Item1 + bestPair.Item2);
                        i += 2;
                    }
                    else
                    {
                        newWordTokens.Add(wordTokens[i]);
                        i++;
                    }
                }
                
                wordTokens = newWordTokens;
            }

            string result = string.Join(" ", wordTokens);
            _cache[word] = result;
            return result;
        }

        /// <summary>
        /// Gets all consecutive pairs in a list
        /// </summary>
        private HashSet<Tuple<string, string>> GetPairs(List<string> tokens)
        {
            var pairs = new HashSet<Tuple<string, string>>();
            for (int i = 0; i < tokens.Count - 1; i++)
            {
                pairs.Add(Tuple.Create(tokens[i], tokens[i + 1]));
            }
            return pairs;
        }

        /// <summary>
        /// Tokenizes text using byte-level BPE
        /// </summary>
        protected override async Task<List<string>> TokenizeInternalAsync(string text)
        {
            var tokens = new List<string>();
            
            // Split text using regex pattern
            var matches = _pattern.Matches(text);
            
            foreach (Match match in matches)
            {
                string piece = match.Value;
                
                // Convert to bytes then to unicode representation
                byte[] bytes = Encoding.UTF8.GetBytes(piece);
                string unicodePiece = ConvertBytesToUnicode(bytes);
                
                // Apply BPE
                string bpePiece = ApplyBPE(unicodePiece);
                
                // Add tokens
                foreach (string token in bpePiece.Split(' '))
                {
                    if (!string.IsNullOrEmpty(token))
                    {
                        tokens.Add(token);
                    }
                }
            }

            return await Task.FromResult(tokens);
        }

        /// <summary>
        /// Post-processes tokens to reconstruct the original text
        /// </summary>
        protected override async Task<string> PostProcessTokensAsync(List<string> tokens)
        {
            // Tokens are already provided as strings from DecodeAsync

            // Concatenate tokens
            string unicodeText = string.Join("", tokens);
            
            // Convert back to bytes
            byte[] bytes = ConvertUnicodeToBytes(unicodeText);
            
            // Convert bytes to text
            string result = Encoding.UTF8.GetString(bytes);

            return await Task.FromResult(result);
        }

        /// <inheritdoc/>
        public override async Task<IReadOnlyList<string>> TokenizeAsync(string text)
        {
            EnsureInitialized();
            return await TokenizeInternalAsync(text);
        }

        /// <summary>
        /// Trains the BPE tokenizer on a corpus
        /// </summary>
        public async Task TrainAsync(IEnumerable<string> corpus, int numMerges)
        {
            // Count token frequencies
            var tokenFreqs = new Dictionary<string, int>();
            
            foreach (var text in corpus)
            {
                var matches = _pattern.Matches(text);
                foreach (Match match in matches)
                {
                    string piece = match.Value;
                    byte[] bytes = Encoding.UTF8.GetBytes(piece);
                    string unicodePiece = ConvertBytesToUnicode(bytes);
                    
                    tokenFreqs[unicodePiece] = tokenFreqs.ContainsKey(unicodePiece) ? tokenFreqs[unicodePiece] + 1 : 1;
                }
            }

            // Initialize vocabulary with individual bytes
            var vocab = new Dictionary<string, int>();
            foreach (var token in tokenFreqs.Keys)
            {
                var wordTokens = token.Select(c => c.ToString()).ToArray();
                foreach (var t in wordTokens)
                {
                    vocab[t] = vocab.ContainsKey(t) ? vocab[t] + tokenFreqs[token] : tokenFreqs[token];
                }
            }

            // Perform BPE merges
            _merges.Clear();
            
            for (int i = 0; i < numMerges; i++)
            {
                var pairs = new Dictionary<Tuple<string, string>, int>();
                
                // Count pair frequencies
                foreach (var token in tokenFreqs.Keys)
                {
                    var wordTokens = token.Split(' ').Where(t => !string.IsNullOrEmpty(t)).ToList();
                    if (wordTokens.Count < 2) continue;
                    
                    var tokenPairs = GetPairs(wordTokens);
                    foreach (var pair in tokenPairs)
                    {
                        pairs[pair] = pairs.ContainsKey(pair) ? pairs[pair] + tokenFreqs[token] : tokenFreqs[token];
                    }
                }

                if (pairs.Count == 0) break;

                // Find most frequent pair
                var bestPair = pairs.OrderByDescending(p => p.Value).First().Key;
                _merges[bestPair] = i;

                // Update vocabulary
                string newToken = bestPair.Item1 + bestPair.Item2;
                vocab[newToken] = pairs[bestPair];
                
                // Clear cache since merges changed
                _cache.Clear();
            }

            await Task.CompletedTask;
        }
    }
}