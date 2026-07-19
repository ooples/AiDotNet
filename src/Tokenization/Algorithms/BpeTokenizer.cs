using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using AiDotNet.Tokenization.Core;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.Tokenization.Models;
using AiDotNet.Validation;

namespace AiDotNet.Tokenization.Algorithms
{
    /// <summary>
    /// Byte-Pair Encoding (BPE) tokenizer implementation for subword tokenization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// BPE is a data compression algorithm adapted for NLP that learns to merge frequent
    /// character sequences into subword units. It's used by GPT, GPT-2, GPT-3, and many
    /// other modern language models.
    /// </para>
    /// <para><b>For Beginners:</b> BPE is like learning common letter combinations. Imagine
    /// you're creating shorthand notes:
    ///
    /// 1. Start with individual letters: "t", "h", "e", " ", "c", "a", "t"
    /// 2. Notice "th" appears often, so create a symbol for it: "th", "e", " ", ...
    /// 3. Notice "the" appears often, merge again: "the", " ", "cat"
    /// 4. Keep merging until you have a good vocabulary size
    ///
    /// This way, common words like "the" become single tokens, while rare words like
    /// "cryptocurrency" might be split into "crypt" + "ocurrency" or similar subwords.
    ///
    /// Benefits:
    /// - No out-of-vocabulary words (any text can be tokenized)
    /// - Common words are single tokens (efficient)
    /// - Rare words are split into meaningful subwords (handles new words)
    ///
    /// Example tokenization of "unhappiness":
    /// - Full word not in vocabulary, so split into subwords
    /// - Possible result: ["un", "happiness"] or ["un", "happy", "ness"]
    /// </para>
    /// </remarks>
    public class BpeTokenizer : TokenizerBase
    {
        private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(1);
        private readonly Dictionary<(string, string), int> _bpeMerges;
        private readonly Dictionary<string, List<string>> _cache;
        private readonly Regex _patternRegex;

        // Byte-level BPE (GPT-2 / Radford et al. 2019): the vocabulary and merges live in a "byte-mapped"
        // alphabet where every one of the 256 byte values is a single visible Unicode character (space is
        // 'Ġ', newline 'Ċ', …). When enabled, encoding UTF-8-encodes each pre-token, maps its bytes into this
        // alphabet before merging, and decoding maps the characters back to bytes and UTF-8-decodes. This is
        // what GPT-2/GPT-3/RoBERTa/CLIP and every GGUF/Hugging Face BPE checkpoint use, so a raw-character BPE
        // can never reproduce their token boundaries. Left off by default so corpus-trained char-level
        // tokenizers keep their existing behavior.
        private readonly bool _byteLevel;
        private readonly Dictionary<byte, char>? _byteEncoder;
        private readonly Dictionary<char, byte>? _byteDecoder;

        // Added/control tokens (e.g. GPT-2 "<|endoftext|>", ChatML "<|im_start|>") are matched as whole units
        // before byte-level BPE, exactly like Hugging Face's added-token handling: the text is split on them,
        // the token itself is emitted verbatim (it is its own vocabulary entry), and only the surrounding
        // spans run through BPE. Without this a control token would be shattered into byte pieces. Null when
        // the tokenizer has no added tokens.
        private readonly Regex? _specialSplitRegex;

        // Causal-LM special-token policy: GPT/Llama tokenizers optionally prepend a single BOS and never
        // append EOS or wrap the prompt in BERT [CLS]/[SEP]. Driven by the checkpoint's add_bos_token flag.
        private readonly bool _addBosToken;
        private readonly string? _bosToken;

        /// <summary>
        /// Creates a new BPE tokenizer with the specified vocabulary and merge rules.
        /// </summary>
        /// <param name="vocabulary">The vocabulary containing all valid tokens.</param>
        /// <param name="merges">The BPE merges (pairs of tokens to merge and their priority order).</param>
        /// <param name="specialTokens">The special tokens configuration. Defaults to GPT-style tokens.</param>
        /// <param name="pattern">The regex pattern for pre-tokenization. Defaults to GPT-2 pattern.</param>
        /// <remarks>
        /// <para><b>For Beginners:</b> Most users should use the Train method or load a pretrained
        /// tokenizer instead of calling this constructor directly. The merges dictionary contains
        /// rules like ("t", "h") -> 0 meaning "merge t and h first" (lower number = higher priority).
        /// </para>
        /// </remarks>
        /// <param name="byteLevel">
        /// When <c>true</c>, encode/decode use the GPT-2 byte-level alphabet (see <see cref="_byteLevel"/>).
        /// Required for any vocabulary whose tokens are byte-mapped (GPT-2/RoBERTa/CLIP/GGUF checkpoints).
        /// </param>
        /// <param name="specialTokenStrings">
        /// Added/control token strings to match as whole units before BPE (e.g. <c>&lt;|im_start|&gt;</c>).
        /// Must already be present in <paramref name="vocabulary"/>. Null/empty to disable.
        /// </param>
        public BpeTokenizer(
            IVocabulary vocabulary,
            Dictionary<(string, string), int> merges,
            SpecialTokens? specialTokens = null,
            string? pattern = null,
            bool byteLevel = false,
            IEnumerable<string>? specialTokenStrings = null,
            bool addBosToken = false,
            string? bosToken = null)
            : base(vocabulary, specialTokens ?? SpecialTokens.Gpt())
        {
            Guard.NotNull(merges);
            _bpeMerges = merges;
            _cache = new Dictionary<string, List<string>>();

            _byteLevel = byteLevel;
            _addBosToken = addBosToken;
            _bosToken = bosToken;

            // Build the added-token splitter, longest-first so a longer token wins over a prefix of it.
            var specials = specialTokenStrings?.Where(s => !string.IsNullOrEmpty(s)).Distinct().ToList();
            if (specials is { Count: > 0 })
            {
                var alternation = string.Join(
                    "|", specials.OrderByDescending(s => s.Length).Select(Regex.Escape));
                _specialSplitRegex = new Regex(alternation, RegexOptions.Compiled, RegexTimeout);
            }
            if (byteLevel)
            {
                _byteEncoder = BuildByteToUnicode();
                _byteDecoder = new Dictionary<char, byte>(_byteEncoder.Count);
                foreach (var kv in _byteEncoder)
                {
                    _byteDecoder[kv.Value] = kv.Key;
                }
            }

            // Default GPT-2 pattern for pre-tokenization
            pattern ??= @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
            _patternRegex = new Regex(pattern, RegexOptions.Compiled, RegexTimeout);
        }

        /// <summary>
        /// Builds the GPT-2 byte-to-Unicode table: a reversible map from all 256 byte values onto visible,
        /// non-whitespace Unicode code points, matching <c>bytes_to_unicode()</c> in the GPT-2 encoder and
        /// Hugging Face's ByteLevel pre-tokenizer exactly.
        /// </summary>
        private static Dictionary<byte, char> BuildByteToUnicode()
        {
            // The three "printable" byte ranges GPT-2's bytes_to_unicode keeps as-is:
            // '!'..'~' (0x21..0x7E), U+00A1..U+00AC, and U+00AE..U+00FF. Code points are written numerically
            // to keep this source file pure ASCII.
            var bs = new List<int>();
            for (int i = 0x21; i <= 0x7E; i++) bs.Add(i);
            for (int i = 0xA1; i <= 0xAC; i++) bs.Add(i);
            for (int i = 0xAE; i <= 0xFF; i++) bs.Add(i);

            var cs = new List<int>(bs);
            int n = 0;
            for (int b = 0; b < 256; b++)
            {
                if (!bs.Contains(b))
                {
                    bs.Add(b);
                    cs.Add(256 + n);
                    n++;
                }
            }

            var map = new Dictionary<byte, char>(256);
            for (int i = 0; i < bs.Count; i++)
            {
                map[(byte)bs[i]] = (char)cs[i];
            }

            return map;
        }

        // Maps a pre-token to the GPT-2 byte-level alphabet: UTF-8 bytes -> visible characters.
        private string ToByteLevel(string word)
        {
            var encoder = _byteEncoder
                ?? throw new InvalidOperationException("Byte-level encoder is not initialized.");
            return MapBytes(word, encoder);
        }

        // UTF-8-encodes a string and maps each byte through the GPT-2 byte-to-Unicode table.
        private static string MapBytes(string word, Dictionary<byte, char> encoder)
        {
            var bytes = Encoding.UTF8.GetBytes(word);
            var chars = new char[bytes.Length];
            for (int i = 0; i < bytes.Length; i++)
            {
                chars[i] = encoder[bytes[i]];
            }

            return new string(chars);
        }

        /// <summary>
        /// Trains a BPE tokenizer from a text corpus by learning merge rules.
        /// </summary>
        /// <param name="corpus">The training corpus - a collection of text strings.</param>
        /// <param name="vocabSize">The desired vocabulary size (number of unique tokens).</param>
        /// <param name="specialTokens">The special tokens configuration. Defaults to GPT-style tokens.</param>
        /// <param name="pattern">The regex pattern for pre-tokenization. Defaults to GPT-2 pattern.</param>
        /// <returns>A trained BPE tokenizer ready to tokenize text.</returns>
        /// <remarks>
        /// <para><b>For Beginners:</b> Training learns which letter combinations appear most
        /// frequently in your text. For example, if training on English text:
        ///
        /// 1. The algorithm starts with all individual characters as tokens
        /// 2. It counts all adjacent character pairs in the corpus
        /// 3. The most frequent pair (e.g., "t" + "h") becomes a new token "th"
        /// 4. This repeats until reaching the desired vocabulary size
        ///
        /// Larger vocabulary = longer sequences become single tokens = faster inference
        /// but more memory. Typical sizes: 30,000-50,000 tokens.
        /// </para>
        /// </remarks>
        public static BpeTokenizer Train(
            IEnumerable<string> corpus,
            int vocabSize,
            SpecialTokens? specialTokens = null,
            string? pattern = null,
            bool byteLevel = false)
        {
            if (corpus == null)
                throw new ArgumentNullException(nameof(corpus));
            if (vocabSize < 1)
                throw new ArgumentOutOfRangeException(nameof(vocabSize), "Vocabulary size must be at least 1.");

            var corpusList = corpus.ToList();
            specialTokens ??= SpecialTokens.Gpt();

            // Byte-level training learns merges over the GPT-2 byte alphabet, so the trained vocabulary is
            // byte-mapped and encode/decode round-trip any input exactly (as they do at inference).
            var byteEncoder = byteLevel ? BuildByteToUnicode() : null;

            // Step 1: Build character vocabulary
            var vocabulary = new Vocabulary.Vocabulary(specialTokens.UnkToken);

            // Add special tokens first
            foreach (var token in specialTokens.GetAllSpecialTokens())
            {
                vocabulary.AddToken(token);
            }

            // Handle empty corpus - return minimal tokenizer with only special tokens
            if (corpusList.Count == 0)
            {
                var emptyMerges = new Dictionary<(string, string), int>();
                pattern ??= @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
                return new BpeTokenizer(vocabulary, emptyMerges, specialTokens, pattern, byteLevel);
            }

            // Step 2: Pre-tokenize and get word frequencies
            pattern ??= @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
            var preTokenRegex = new Regex(pattern, RegexOptions.Compiled, RegexTimeout);

            var wordFreqs = new Dictionary<string, int>();
            foreach (var text in corpusList)
            {
                var words = preTokenRegex.Matches(text)
                    .Cast<Match>()
                    .Select(m => m.Value);

                foreach (var rawWord in words)
                {
                    var word = byteEncoder is null ? rawWord : MapBytes(rawWord, byteEncoder);
                    wordFreqs[word] = wordFreqs.GetValueOrDefault(word, 0) + 1;
                }
            }

            // Step 3: Initialize word representations as character sequences
            var splits = new Dictionary<string, List<string>>();
            foreach (var word in wordFreqs.Keys)
            {
                var charStrings = word.Select(c => c.ToString()).ToList();
                splits[word] = charStrings;

                // Add characters to vocabulary
                foreach (var charStr in charStrings)
                {
                    vocabulary.AddToken(charStr);
                }
            }

            // Step 4: Iteratively merge the most frequent pair
            var merges = new Dictionary<(string, string), int>();
            var mergeOrder = 0;

            while (vocabulary.Size < vocabSize)
            {
                // Count pairs
                var pairFreqs = new Dictionary<(string, string), int>();
                foreach (var (word, split) in splits)
                {
                    var freq = wordFreqs[word];
                    for (int i = 0; i < split.Count - 1; i++)
                    {
                        var pair = (split[i], split[i + 1]);
                        pairFreqs[pair] = pairFreqs.GetValueOrDefault(pair, 0) + freq;
                    }
                }

                if (pairFreqs.Count == 0)
                    break;

                // Find most frequent pair
                var bestPair = pairFreqs.OrderByDescending(p => p.Value).First().Key;

                // Add merge
                merges[bestPair] = mergeOrder++;

                // Add merged token to vocabulary
                var newToken = bestPair.Item1 + bestPair.Item2;
                vocabulary.AddToken(newToken);

                // Update splits
                var newSplits = new Dictionary<string, List<string>>();
                foreach (var (word, split) in splits)
                {
                    var newSplit = new List<string>();
                    int i = 0;
                    while (i < split.Count)
                    {
                        if (i < split.Count - 1 && split[i] == bestPair.Item1 && split[i + 1] == bestPair.Item2)
                        {
                            newSplit.Add(newToken);
                            i += 2;
                        }
                        else
                        {
                            newSplit.Add(split[i]);
                            i++;
                        }
                    }
                    newSplits[word] = newSplit;
                }
                splits = newSplits;
            }

            return new BpeTokenizer(vocabulary, merges, specialTokens, pattern, byteLevel);
        }

        /// <summary>
        /// Tokenizes text into BPE tokens.
        /// </summary>
        public override List<string> Tokenize(string text)
        {
            if (string.IsNullOrEmpty(text))
                return new List<string>();

            var tokens = new List<string>();

            // No added tokens: run the whole text through BPE.
            if (_specialSplitRegex is null)
            {
                TokenizeChunk(text, tokens);
                return tokens;
            }

            // Split on added/control tokens, emitting each verbatim and BPE-ing only the spans between them.
            int last = 0;
            foreach (Match m in _specialSplitRegex.Matches(text))
            {
                if (m.Index > last)
                {
                    TokenizeChunk(text.Substring(last, m.Index - last), tokens);
                }

                tokens.Add(m.Value);
                last = m.Index + m.Length;
            }

            if (last < text.Length)
            {
                TokenizeChunk(text.Substring(last), tokens);
            }

            return tokens;
        }

        // Runs one span of ordinary text (no added tokens) through pattern pre-tokenization and BPE.
        private void TokenizeChunk(string text, List<string> tokens)
        {
            var matches = _patternRegex.Matches(text);
            foreach (var rawWord in matches.Cast<Match>().Select(m => m.Value))
            {
                // Byte-level BPE maps each pre-token into the GPT-2 byte alphabet before merging, so the
                // symbols and merges share the same space as the vocabulary. Char-level BPE merges directly.
                var word = _byteLevel ? ToByteLevel(rawWord) : rawWord;

                // Check cache
                if (_cache.TryGetValue(word, out var cachedTokens))
                {
                    tokens.AddRange(cachedTokens);
                    continue;
                }

                // Apply BPE
                var bpeTokens = BpeEncode(word);
                _cache[word] = bpeTokens;
                tokens.AddRange(bpeTokens);
            }
        }

        /// <summary>
        /// Applies BPE encoding to a word.
        /// </summary>
        private List<string> BpeEncode(string word)
        {
            if (word.Length == 0)
                return new List<string>();

            // Start with character-level tokens
            var tokens = word.Select(c => c.ToString()).ToList();

            while (tokens.Count > 1)
            {
                // Find the best pair to merge
                var bestPair = ((string, string)?)null;
                var bestRank = int.MaxValue;

                for (int i = 0; i < tokens.Count - 1; i++)
                {
                    var pair = (tokens[i], tokens[i + 1]);
                    if (_bpeMerges.TryGetValue(pair, out var rank) && rank < bestRank)
                    {
                        bestPair = pair;
                        bestRank = rank;
                    }
                }

                if (bestPair == null)
                    break;

                // Merge the best pair
                var newTokens = new List<string>();
                int j = 0;
                while (j < tokens.Count)
                {
                    if (j < tokens.Count - 1 && tokens[j] == bestPair.Value.Item1 && tokens[j + 1] == bestPair.Value.Item2)
                    {
                        newTokens.Add(bestPair.Value.Item1 + bestPair.Value.Item2);
                        j += 2;
                    }
                    else
                    {
                        newTokens.Add(tokens[j]);
                        j++;
                    }
                }
                tokens = newTokens;
            }

            return tokens;
        }

        /// <summary>
        /// Causal-LM special-token policy: optionally prepend a single BOS, and never append EOS or add
        /// BERT-style [CLS]/[SEP]. This overrides the base BERT behavior, which would wrap a decoder prompt
        /// in out-of-vocabulary tokens and corrupt generation.
        /// </summary>
        protected override List<string> AddSpecialTokensToSequence(List<string> tokens)
        {
            if (_addBosToken && !string.IsNullOrEmpty(_bosToken))
            {
                var withBos = new List<string>(tokens.Count + 1) { _bosToken! };
                withBos.AddRange(tokens);
                return withBos;
            }

            // No BOS to add: defer to the base policy, which prepends/appends any configured Cls/Sep tokens.
            // Decoder tokenizers (e.g. from GGUF) leave those empty, so nothing is added; CLIP and similar
            // keep their start/end tokens.
            return base.AddSpecialTokensToSequence(tokens);
        }

        /// <summary>
        /// Cleans up tokens and converts them back to text.
        /// </summary>
        protected override string CleanupTokens(List<string> tokens)
        {
            if (tokens == null || tokens.Count == 0)
                return string.Empty;

            if (!_byteLevel)
            {
                return string.Join("", tokens);
            }

            // Byte-level: the concatenated token text is in the GPT-2 byte alphabet. Map every character back
            // to its byte and UTF-8-decode the result, so multi-byte characters that BPE split across tokens
            // are reassembled correctly (an exact inverse of ToByteLevel). Characters outside the alphabet
            // (e.g. an un-skipped special token) fall back to their own UTF-8 bytes.
            var decoder = _byteDecoder
                ?? throw new InvalidOperationException("Byte-level decoder is not initialized.");
            var bytes = new List<byte>(tokens.Sum(t => t.Length));
            foreach (var token in tokens)
            {
                foreach (var c in token)
                {
                    if (decoder.TryGetValue(c, out var b))
                    {
                        bytes.Add(b);
                    }
                    else
                    {
                        bytes.AddRange(Encoding.UTF8.GetBytes(c.ToString()));
                    }
                }
            }

            return Encoding.UTF8.GetString(bytes.ToArray());
        }
    }
}
