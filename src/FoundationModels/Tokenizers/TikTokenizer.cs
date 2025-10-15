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
    /// TikToken tokenizer implementation similar to what's used by GPT-3.5/GPT-4.
    /// Uses a refined BPE algorithm with better handling of whitespace, numbers, and special patterns.
    /// </summary>
    public class TikTokenizer : TokenizerBase
    {
        private readonly Dictionary<Tuple<string, string>, int> _merges;
        private readonly Dictionary<string, int> _encoder;
        private readonly Dictionary<int, string> _decoder;
        private readonly Dictionary<int, byte[]> _tokenBytes;
        private readonly Regex _pattern;
        private readonly HashSet<string> _specialTokensSet;
        private readonly int _maxTokenLength;

        /// <inheritdoc/>
        public override int MaxSequenceLength => 8192; // GPT-4 default

        /// <summary>
        /// Initializes a new instance of the TikTokenizer class
        /// </summary>
        /// <param name="encodingName">Encoding name (cl100k_base for GPT-4, p50k_base for GPT-3)</param>
        public TikTokenizer(string encodingName = "cl100k_base")
        {
            _merges = new Dictionary<Tuple<string, string>, int>();
            _encoder = new Dictionary<string, int>();
            _decoder = new Dictionary<int, string>();
            _tokenBytes = new Dictionary<int, byte[]>();
            _specialTokensSet = new HashSet<string>();
            _maxTokenLength = 256;

            // GPT-4 (cl100k_base) uses this improved regex pattern
            if (encodingName == "cl100k_base")
            {
                _pattern = new Regex(
                    @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
                    RegexOptions.Compiled
                );
            }
            else // p50k_base for GPT-3
            {
                _pattern = new Regex(
                    @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
                    RegexOptions.Compiled
                );
            }

            InitializeSpecialTokens();
        }

        /// <summary>
        /// Initializes special tokens for TikToken
        /// </summary>
        protected override void InitializeSpecialTokens()
        {
            base.InitializeSpecialTokens();
            
            // GPT-4 specific special tokens
            var gpt4SpecialTokens = new Dictionary<string, int>
            {
                ["<|endoftext|>"] = 100257,
                ["<|fim_prefix|>"] = 100258,
                ["<|fim_middle|>"] = 100259,
                ["<|fim_suffix|>"] = 100260,
                ["<|endofprompt|>"] = 100261,
                ["<|startoftext|>"] = 100262,
                // System/user/assistant markers
                ["<|system|>"] = 100263,
                ["<|user|>"] = 100264,
                ["<|assistant|>"] = 100265,
                ["<|end|>"] = 100266,
            };

            foreach (var kvp in gpt4SpecialTokens)
            {
                var token = kvp.Key;
                var id = kvp.Value;
                _specialTokens[token] = id;
                _specialTokensSet.Add(token);
            }
        }

        /// <summary>
        /// Loads the vocabulary with base tokens and merges
        /// </summary>
        protected override async Task LoadVocabularyAsync()
        {
            _encoder.Clear();
            _decoder.Clear();
            _tokenBytes.Clear();

            int tokenId = 0;

            // Add special tokens to encoder/decoder
            foreach (var kvp in _specialTokens)
            {
                var token = kvp.Key;
                var id = kvp.Value;
                _encoder[token] = id;
                _decoder[id] = token;
                _tokenBytes[id] = Encoding.UTF8.GetBytes(token);
            }

            // Initialize base vocabulary (all single bytes)
            for (int i = 0; i < 256; i++)
            {
                var bytes = new byte[] { (byte)i };
                var token = BytesToUnicode(bytes);
                
                if (!_encoder.ContainsKey(token))
                {
                    _encoder[token] = tokenId;
                    _decoder[tokenId] = token;
                    _tokenBytes[tokenId] = bytes;
                    tokenId++;
                }
            }

            // Add common multi-byte sequences (simplified for example)
            await AddCommonMerges(tokenId);
        }

        /// <summary>
        /// Adds common BPE merges for TikToken
        /// </summary>
        private async Task AddCommonMerges(int startTokenId)
        {
            int tokenId = startTokenId;

            // Common English word pieces and subwords
            var commonMerges = new List<string[]>
            {
                // Common contractions
                new[] { "n", "'t" }, new[] { "s", "'s" }, new[] { "r", "'re" },
                new[] { "v", "'ve" }, new[] { "l", "'ll" }, new[] { "m", "'m" },
                new[] { "d", "'d" },
                
                // Common prefixes
                new[] { "u", "n" }, new[] { "r", "e" }, new[] { "i", "n" },
                new[] { "d", "e" }, new[] { "d", "is" }, new[] { "p", "re" },
                new[] { "c", "on" }, new[] { "e", "x" },
                
                // Common suffixes  
                new[] { "i", "ng" }, new[] { "e", "d" }, new[] { "e", "r" },
                new[] { "e", "st" }, new[] { "l", "y" }, new[] { "t", "ion" },
                new[] { "a", "tion" }, new[] { "m", "ent" }, new[] { "n", "ess" },
                
                // Common words
                new[] { "t", "he" }, new[] { "a", "nd" }, new[] { "o", "f" },
                new[] { "t", "o" }, new[] { "i", "s" }, new[] { "i", "t" },
                new[] { "f", "or" }, new[] { "a", "s" }, new[] { "w", "ith" },
                new[] { "th", "at" }, new[] { "b", "e" }, new[] { "o", "n" },
                new[] { "h", "ave" }, new[] { "fr", "om" }, new[] { "b", "y" },
                
                // Numbers and punctuation patterns
                new[] { "0", "0" }, new[] { "1", "1" }, new[] { ".", "." },
                new[] { ",", " " }, new[] { "!", "!" }, new[] { "?", "?" },
                new[] { " ", " " }, new[] { "\n", "\n" }, new[] { "\r", "\n" }
            };

            foreach (var parts in commonMerges)
            {
                if (tokenId >= 100256) break; // Stay below special token range
                
                var merged = string.Join("", parts);
                if (!_encoder.ContainsKey(merged))
                {
                    _encoder[merged] = tokenId;
                    _decoder[tokenId] = merged;
                    _tokenBytes[tokenId] = Encoding.UTF8.GetBytes(merged);
                    
                    if (parts.Length == 2)
                    {
                        _merges[Tuple.Create(parts[0], parts[1])] = _merges.Count;
                    }
                    
                    tokenId++;
                }
            }

            // Update vocabulary
            foreach (var kvp in _encoder)
            {
                var token = kvp.Key;
                var id = kvp.Value;
                _vocabulary[token] = id;
            }
            
            foreach (var kvp in _decoder)
            {
                var id = kvp.Key;
                var token = kvp.Value;
                _reverseVocabulary[id] = token;
            }

            await Task.CompletedTask;
        }

        /// <summary>
        /// Converts bytes to a unicode string representation
        /// </summary>
        private string BytesToUnicode(byte[] bytes)
        {
            // TikToken uses a specific byte-to-unicode mapping
            var result = new StringBuilder();
            
            foreach (byte b in bytes)
            {
                // For printable ASCII, use directly
                if (b >= 33 && b <= 126 && b != 92) // Exclude backslash
                {
                    result.Append((char)b);
                }
                else
                {
                    // Map to unicode private use area
                    result.Append((char)(0x100 + b));
                }
            }
            
            return result.ToString();
        }

        /// <summary>
        /// Tokenizes text using TikToken algorithm
        /// </summary>
        protected override async Task<List<string>> TokenizeInternalAsync(string text)
        {
            var tokens = new List<string>();
            
            // First, check for special tokens
            var processedText = text;
            var specialTokenPositions = new List<Tuple<int, int, string>>();
            
            foreach (var specialToken in _specialTokensSet.OrderByDescending(t => t.Length))
            {
                int index = 0;
                while ((index = processedText.IndexOf(specialToken, index)) != -1)
                {
                    specialTokenPositions.Add(Tuple.Create(index, index + specialToken.Length, specialToken));
                    index += specialToken.Length;
                }
            }
            
            // Sort by position
            specialTokenPositions.Sort((a, b) => a.Item1.CompareTo(b.Item1));
            
            // Process text segments
            int currentPos = 0;
            foreach (var tuple in specialTokenPositions)
            {
                var start = tuple.Item1;
                var end = tuple.Item2;
                var token = tuple.Item3;
                if (start > currentPos)
                {
                    // Process text before special token
                    var segment = processedText.Substring(currentPos, start - currentPos);
                    tokens.AddRange(await TokenizeSegment(segment));
                }
                
                // Add special token
                tokens.Add(token);
                currentPos = end;
            }
            
            // Process remaining text
            if (currentPos < processedText.Length)
            {
                var segment = processedText.Substring(currentPos);
                tokens.AddRange(await TokenizeSegment(segment));
            }
            
            return tokens;
        }

        /// <summary>
        /// Tokenizes a text segment (without special tokens)
        /// </summary>
        private async Task<List<string>> TokenizeSegment(string text)
        {
            var tokens = new List<string>();
            var matches = _pattern.Matches(text);
            
            foreach (Match match in matches)
            {
                var piece = match.Value;
                var bytes = Encoding.UTF8.GetBytes(piece);
                
                // Apply BPE to the piece
                var bpeTokens = ApplyBPE(bytes);
                tokens.AddRange(bpeTokens);
            }
            
            return await Task.FromResult(tokens);
        }

        /// <summary>
        /// Applies BPE algorithm to a byte sequence
        /// </summary>
        private List<string> ApplyBPE(byte[] bytes)
        {
            if (bytes.Length == 0) return new List<string>();
            
            // Convert bytes to unicode representation
            var unicodeChars = new List<string>();
            foreach (byte b in bytes)
            {
                unicodeChars.Add(BytesToUnicode(new[] { b }));
            }
            
            if (unicodeChars.Count == 1)
            {
                return unicodeChars;
            }
            
            // Apply merges
            while (true)
            {
                var pairs = GetPairs(unicodeChars);
                if (pairs.Count == 0) break;
                
                // Find the best merge
                var bestPair = pairs
                    .Where(p => _merges.ContainsKey(p))
                    .OrderBy(p => _merges[p])
                    .FirstOrDefault();
                
                if (bestPair == null) break;
                
                // Apply merge
                var newTokens = new List<string>();
                int i = 0;
                
                while (i < unicodeChars.Count)
                {
                    if (i < unicodeChars.Count - 1 &&
                        unicodeChars[i] == bestPair.Item1 &&
                        unicodeChars[i + 1] == bestPair.Item2)
                    {
                        newTokens.Add(bestPair.Item1 + bestPair.Item2);
                        i += 2;
                    }
                    else
                    {
                        newTokens.Add(unicodeChars[i]);
                        i++;
                    }
                }
                
                unicodeChars = newTokens;
                
                // Prevent infinite loops and excessive token length
                if (unicodeChars.Any(t => t.Length > _maxTokenLength))
                {
                    break;
                }
            }
            
            return unicodeChars;
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
        /// Post-processes tokens back to text
        /// </summary>
        protected override async Task<string> PostProcessTokensAsync(List<string> tokens)
        {
            var bytes = new List<byte>();
            
            foreach (var token in tokens)
            {
                // Try to get pre-stored bytes for this token
                var tokenId = _encoder.ContainsKey(token) ? _encoder[token] : -1;
                if (tokenId != -1 && _tokenBytes.TryGetValue(tokenId, out var tokenBytes))
                {
                    bytes.AddRange(tokenBytes);
                }
                else
                {
                    // Decode unicode representation back to bytes
                    foreach (char c in token)
                    {
                        if (c >= 0x100)
                        {
                            bytes.Add((byte)(c - 0x100));
                        }
                        else
                        {
                            bytes.Add((byte)c);
                        }
                    }
                }
            }
            
            // Convert bytes to string
            return await Task.FromResult(Encoding.UTF8.GetString(bytes.ToArray()));
        }

        /// <inheritdoc/>
        public override async Task<IReadOnlyList<string>> TokenizeAsync(string text)
        {
            return await TokenizeInternalAsync(text);
        }

        /// <summary>
        /// Encodes text with special handling for chat format
        /// </summary>
        public async Task<Vector<int>> EncodeChatAsync(List<Tuple<string, string>> messages)
        {
            var tokens = new List<int>();
            
            foreach (var tuple in messages)
            {
                var role = tuple.Item1;
                var content = tuple.Item2;
                // Add role marker
                if (_encoder.TryGetValue($"<|{role}|>", out var roleToken))
                {
                    tokens.Add(roleToken);
                }
                
                // Add content
                var contentTokens = await EncodeAsync(content, addSpecialTokens: false);
                for (int i = 0; i < contentTokens.Count; i++)
                {
                    tokens.Add(contentTokens[i]);
                }
                
                // Add end marker
                if (_encoder.TryGetValue("<|end|>", out var endToken))
                {
                    tokens.Add(endToken);
                }
            }
            
            return new Vector<int>(tokens.ToArray());
        }

        /// <summary>
        /// Counts tokens without full encoding (more efficient)
        /// </summary>
        public async Task<int> CountTokensAsync(string text)
        {
            var tokens = await TokenizeInternalAsync(text);
            return tokens.Count;
        }

        /// <summary>
        /// Splits text to fit within token limit
        /// </summary>
        public async Task<List<string>> SplitTextAsync(string text, int maxTokensPerChunk, int overlap = 0)
        {
            var chunks = new List<string>();
            var tokens = await TokenizeInternalAsync(text);
            
            for (int i = 0; i < tokens.Count; i += maxTokensPerChunk - overlap)
            {
                var chunkTokens = tokens.Skip(i).Take(maxTokensPerChunk).ToList();
                var chunkText = await PostProcessTokensAsync(chunkTokens);
                chunks.Add(chunkText);
                
                if (i + maxTokensPerChunk >= tokens.Count)
                {
                    break;
                }
            }
            
            return chunks;
        }
    }
}