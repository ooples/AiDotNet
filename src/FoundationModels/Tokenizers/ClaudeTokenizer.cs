using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.FoundationModels.Tokenizers
{
    /// <summary>
    /// Claude-style tokenizer with advanced Unicode handling and special token support.
    /// Optimized for multilingual support and reasoning tasks.
    /// </summary>
    public class ClaudeTokenizer : TokenizerBase
    {
        private readonly Dictionary<string, int> _encoder;
        private readonly Dictionary<int, string> _decoder;
        private readonly Dictionary<byte[], int> _byteEncoder;
        private readonly Regex _pattern;
        private readonly bool _useUnicodeNormalization;
        private readonly Dictionary<string, string> _tokenCache;
        private readonly int _cacheMaxSize;

        /// <inheritdoc/>
        public override int MaxSequenceLength => 100000; // Claude supports very long contexts

        /// <summary>
        /// Initializes a new instance of the ClaudeTokenizer class
        /// </summary>
        /// <param name="useUnicodeNormalization">Whether to normalize Unicode (NFC)</param>
        /// <param name="cacheMaxSize">Maximum size of token cache</param>
        public ClaudeTokenizer(bool useUnicodeNormalization = true, int cacheMaxSize = 10000)
        {
            _encoder = new Dictionary<string, int>();
            _decoder = new Dictionary<int, string>();
            _byteEncoder = new Dictionary<byte[], int>(new ByteArrayComparer());
            _useUnicodeNormalization = useUnicodeNormalization;
            _tokenCache = new Dictionary<string, string>();
            _cacheMaxSize = cacheMaxSize;

            // Claude uses a sophisticated pattern that handles:
            // - Multiple languages (including CJK)
            // - Code and mathematical notation
            // - Special formatting and whitespace
            _pattern = new Regex(
                @"<\|[a-z_]+\|>|" +                                    // Special tokens
                @"[\p{Lo}\p{Lm}]+|" +                                  // Unicode letters
                @"[\p{Nl}\p{Nd}]+|" +                                  // Numbers
                @"[\p{P}\p{S}]+|" +                                     // Punctuation and symbols
                @"\s+(?!\S)|" +                                         // Whitespace
                @"\s+",                                                 // Other whitespace
                RegexOptions.Compiled | RegexOptions.IgnoreCase
            );

            InitializeSpecialTokens();
        }

        /// <summary>
        /// Initializes special tokens for Claude
        /// </summary>
        protected override void InitializeSpecialTokens()
        {
            base.InitializeSpecialTokens();

            // Claude-specific special tokens
            var claudeSpecialTokens = new Dictionary<string, int>
            {
                ["<|begin_of_text|>"] = 100000,
                ["<|end_of_text|>"] = 100001,
                ["<|reserved_special_token_0|>"] = 100002,
                ["<|reserved_special_token_1|>"] = 100003,
                ["<|reserved_special_token_2|>"] = 100004,
                ["<|reserved_special_token_3|>"] = 100005,
                ["<|start_header_id|>"] = 100006,
                ["<|end_header_id|>"] = 100007,
                ["<|reserved_special_token_4|>"] = 100008,
                ["<|eot_id|>"] = 100009, // End of turn
                // Reasoning tokens
                ["<|thinking|>"] = 100010,
                ["<|/thinking|>"] = 100011,
                ["<|reflection|>"] = 100012,
                ["<|/reflection|>"] = 100013,
                ["<|planning|>"] = 100014,
                ["<|/planning|>"] = 100015,
                // Tool use tokens
                ["<|tool_use|>"] = 100016,
                ["<|/tool_use|>"] = 100017,
                ["<|tool_result|>"] = 100018,
                ["<|/tool_result|>"] = 100019,
                // Multimodal tokens
                ["<|image|>"] = 100020,
                ["<|/image|>"] = 100021,
                ["<|audio|>"] = 100022,
                ["<|/audio|>"] = 100023,
                ["<|video|>"] = 100024,
                ["<|/video|>"] = 100025,
            };

            foreach (var kvp in claudeSpecialTokens)
            {
                var token = kvp.Key;
                var id = kvp.Value;
                _specialTokens[token] = id;
            }
        }

        /// <summary>
        /// Loads the vocabulary
        /// </summary>
        protected override async Task LoadVocabularyAsync()
        {
            _encoder.Clear();
            _decoder.Clear();
            _byteEncoder.Clear();

            int tokenId = 0;

            // Add special tokens
            foreach (var kvp in _specialTokens)
            {
                var token = kvp.Key;
                var id = kvp.Value;
                _encoder[token] = id;
                _decoder[id] = token;
            }

            // Initialize with all possible bytes
            for (int i = 0; i < 256; i++)
            {
                var bytes = new byte[] { (byte)i };
                var token = BytesToToken(bytes);
                
                if (!_encoder.ContainsKey(token))
                {
                    _encoder[token] = tokenId;
                    _decoder[tokenId] = token;
                    _byteEncoder[bytes] = tokenId;
                    tokenId++;
                }
            }

            // Add common multi-byte sequences
            await AddAdvancedVocabulary(tokenId);
        }

        /// <summary>
        /// Adds advanced vocabulary including multilingual support
        /// </summary>
        private async Task AddAdvancedVocabulary(int startTokenId)
        {
            int tokenId = startTokenId;

            // Common Unicode sequences for various languages
            var multilingualSequences = new List<string>
            {
                // Common English words/subwords
                "the", "and", "ing", "tion", "ation", "ment", "able", "ness",
                
                // Common programming tokens
                "function", "return", "class", "import", "export", "const", "let", "var",
                "public", "private", "static", "void", "async", "await",
                
                // Mathematical symbols
                "∈", "∉", "∀", "∃", "∧", "∨", "¬", "⇒", "⇔", "∅", "∞",
                "α", "β", "γ", "δ", "ε", "θ", "λ", "μ", "π", "σ", "φ", "ω",
                
                // Common CJK characters
                "的", "是", "在", "我", "有", "他", "这", "了", "不", "们",
                "한", "국", "어", "있", "는", "이", "다", "를", "에", "의",
                "の", "は", "が", "を", "に", "と", "で", "た", "し", "い",
                
                // Common emojis
                "😀", "😃", "😄", "😁", "😅", "😂", "🤣", "😊", "😇", "🙂",
                "👍", "👎", "👌", "✌️", "🤞", "🤟", "🤘", "🤙", "💪", "🙏",
                
                // Whitespace variations
                "\n", "\r\n", "\t", "  ", "    ", "\u00A0", "\u2003", "\u2002"
            };

            foreach (var sequence in multilingualSequences)
            {
                if (tokenId >= 100000) break; // Stay below special token range
                
                var normalizedSequence = _useUnicodeNormalization 
                    ? sequence.Normalize(NormalizationForm.FormC) 
                    : sequence;
                
                if (!_encoder.ContainsKey(normalizedSequence))
                {
                    _encoder[normalizedSequence] = tokenId;
                    _decoder[tokenId] = normalizedSequence;
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
        /// Converts bytes to a token representation
        /// </summary>
        private string BytesToToken(byte[] bytes)
        {
            // Use a more sophisticated encoding that preserves all Unicode
            return Convert.ToBase64String(bytes);
        }

        /// <summary>
        /// Tokenizes text with advanced handling
        /// </summary>
        protected override async Task<List<string>> TokenizeInternalAsync(string text)
        {
            // Apply Unicode normalization if enabled
            if (_useUnicodeNormalization)
            {
                text = text.Normalize(NormalizationForm.FormC);
            }

            // Check cache first
            if (_tokenCache.TryGetValue(text, out var cachedTokens))
            {
                return cachedTokens.Split(' ').ToList();
            }

            var tokens = new List<string>();
            
            // Handle special tokens
            var specialTokenMatches = FindSpecialTokens(text);
            int lastEnd = 0;

            foreach (var tuple in specialTokenMatches)
            {
                var start = tuple.Item1;
                var end = tuple.Item2;
                var token = tuple.Item3;
                if (start > lastEnd)
                {
                    // Process text before special token
                    var segment = text.Substring(lastEnd, start - lastEnd);
                    tokens.AddRange(await TokenizeSegmentAdvanced(segment));
                }
                
                tokens.Add(token);
                lastEnd = end;
            }

            // Process remaining text
            if (lastEnd < text.Length)
            {
                var segment = text.Substring(lastEnd);
                tokens.AddRange(await TokenizeSegmentAdvanced(segment));
            }

            // Cache result if text is short enough
            if (text.Length <= 1000 && _tokenCache.Count < _cacheMaxSize)
            {
                _tokenCache[text] = string.Join(" ", tokens);
            }

            return tokens;
        }

        /// <summary>
        /// Advanced tokenization for text segments
        /// </summary>
        private async Task<List<string>> TokenizeSegmentAdvanced(string text)
        {
            var tokens = new List<string>();
            var matches = _pattern.Matches(text);

            foreach (Match match in matches)
            {
                var piece = match.Value;
                
                // Try to find in vocabulary first
                if (_encoder.ContainsKey(piece))
                {
                    tokens.Add(piece);
                    continue;
                }

                // Fall back to byte-level encoding
                var bytes = Encoding.UTF8.GetBytes(piece);
                
                // Apply a sliding window approach for unknown sequences
                int i = 0;
                while (i < bytes.Length)
                {
                    int maxLen = Math.Min(bytes.Length - i, 10); // Max token byte length
                    bool found = false;

                    // Try longest match first
                    for (int len = maxLen; len > 0; len--)
                    {
                        var subBytes = new byte[len];
                        Array.Copy(bytes, i, subBytes, 0, len);
                        
                        if (_byteEncoder.ContainsKey(subBytes))
                        {
                            tokens.Add(BytesToToken(subBytes));
                            i += len;
                            found = true;
                            break;
                        }
                    }

                    if (!found)
                    {
                        // Single byte fallback
                        tokens.Add(BytesToToken(new[] { bytes[i] }));
                        i++;
                    }
                }
            }

            return await Task.FromResult(tokens);
        }

        /// <summary>
        /// Finds special tokens in text
        /// </summary>
        private List<Tuple<int, int, string>> FindSpecialTokens(string text)
        {
            var matches = new List<Tuple<int, int, string>>();
            
            foreach (var token in _specialTokens.Keys)
            {
                int index = 0;
                while ((index = text.IndexOf(token, index, StringComparison.Ordinal)) != -1)
                {
                    matches.Add(Tuple.Create(index, index + token.Length, token));
                    index += token.Length;
                }
            }

            return matches.OrderBy(m => m.Item1).ToList();
        }

        /// <summary>
        /// Post-processes tokens back to text
        /// </summary>
        protected override async Task<string> PostProcessTokensAsync(List<string> tokens)
        {
            var result = new StringBuilder();
            
            foreach (var token in tokens)
            {
                // Handle base64 encoded tokens
                if (token.Length > 0 && IsBase64String(token))
                {
                    try
                    {
                        var bytes = Convert.FromBase64String(token);
                        result.Append(Encoding.UTF8.GetString(bytes));
                    }
                    catch
                    {
                        result.Append(token);
                    }
                }
                else
                {
                    result.Append(token);
                }
            }

            return await Task.FromResult(result.ToString());
        }

        /// <inheritdoc/>
        public override async Task<IReadOnlyList<string>> TokenizeAsync(string text)
        {
            return await TokenizeInternalAsync(text);
        }

        /// <summary>
        /// Checks if a string is valid base64
        /// </summary>
        private bool IsBase64String(string s)
        {
            s = s.Trim();
            return (s.Length % 4 == 0) && Regex.IsMatch(s, @"^[a-zA-Z0-9\+/]*={0,3}$", RegexOptions.None);
        }

        /// <summary>
        /// Encodes conversation with proper formatting
        /// </summary>
        public async Task<Vector<int>> EncodeConversationAsync(
            List<Tuple<string, string>> messages,
            bool addSystemPrompt = true)
        {
            var tokens = new List<int>();

            // Add begin of text
            if (_encoder.TryGetValue("<|begin_of_text|>", out var beginToken))
            {
                tokens.Add(beginToken);
            }

            // Add system prompt if needed
            if (addSystemPrompt && !messages.Any(m => m.Item1 == "system"))
            {
                messages.Insert(0, Tuple.Create("system", "You are Claude, a helpful AI assistant."));
            }

            foreach (var tuple in messages)
            {
                // Add header
                if (_encoder.TryGetValue("<|start_header_id|>", out var startHeader))
                {
                    tokens.Add(startHeader);
                }

                var role = tuple.Item1;
                var content = tuple.Item2;
                // Add role
                var roleTokens = await EncodeAsync(role, addSpecialTokens: false);
                tokens.AddRange(roleTokens.ToArray());

                // End header
                if (_encoder.TryGetValue("<|end_header_id|>", out var endHeader))
                {
                    tokens.Add(endHeader);
                }

                // Add content
                var contentTokens = await EncodeAsync(content, addSpecialTokens: false);
                tokens.AddRange(contentTokens.ToArray());

                // End of turn
                if (_encoder.TryGetValue("<|eot_id|>", out var eotToken))
                {
                    tokens.Add(eotToken);
                }
            }

            return new Vector<int>(tokens.ToArray());
        }

        /// <summary>
        /// Byte array comparer for dictionary keys
        /// </summary>
        private class ByteArrayComparer : IEqualityComparer<byte[]>
        {
            public bool Equals(byte[]? x, byte[]? y)
            {
                if (x == null || y == null) return x == y;
                return x.SequenceEqual(y);
            }

            public int GetHashCode(byte[] obj)
            {
                if (obj == null) return 0;
                int hash = 17;
                foreach (byte b in obj)
                {
                    hash = hash * 31 + b;
                }
                return hash;
            }
        }
    }
}