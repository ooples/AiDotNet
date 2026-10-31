using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Tokenization.Algorithms
{
    /// <summary>
    /// A Hugging Face <c>tokenizers</c> byte-level BPE tokenizer read from its <c>tokenizer.json</c> (the form GPT-2,
    /// Llama 3 and Qwen use): added tokens split out first (leftmost-longest, unnormalized), then the optional NFC
    /// normalizer, the <c>Split</c> pre-tokenizer's regular expression (isolated pieces) or ByteLevel's own, the GPT-2
    /// byte alphabet, and BPE by merge rank (with <c>ignore_merges</c>, a piece that is itself in the vocabulary is
    /// kept whole); with special tokens, the <c>TemplateProcessing</c> post-processor's single-sequence template.
    /// </summary>
    internal sealed class ByteLevelBpeTokenizer
    {
        private const string Gpt2Pattern = @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

        private readonly Dictionary<string, int> _vocabulary;
        private readonly Dictionary<string, int> _merges;
        private readonly Dictionary<string, int> _added = new(StringComparer.Ordinal);
        private readonly HashSet<string> _special = new(StringComparer.Ordinal);
        private readonly Dictionary<char, List<string>> _addedByFirst = new();
        private readonly Dictionary<int, string> _tokens = new();
        private readonly Regex _split;
        private readonly bool _nfc;
        private readonly bool _ignoreMerges;
        // Pre-tokenizers before the regular expression, in order: Digits (individual) and Punctuation, each isolating.
        private readonly string[] _isolators;
        private readonly Dictionary<byte, char> _byteToChar = BpeTokenizer.BuildByteToUnicode();
        private readonly Dictionary<char, byte> _charToByte;
        private readonly ConcurrentDictionary<string, int[]> _cache = new(StringComparer.Ordinal);
        private readonly int[] _templatePrefix;
        private readonly int[] _templateSuffix;

        public ByteLevelBpeTokenizer(Dictionary<string, int> vocabulary, Dictionary<string, int> merges,
            IEnumerable<(string Content, int Id, bool Special)> addedTokens, string? splitPattern, bool nfc, bool ignoreMerges,
            int[]? templatePrefix = null, int[]? templateSuffix = null, IEnumerable<string>? isolators = null)
        {
            _isolators = (isolators ?? Array.Empty<string>()).ToArray();
            _templatePrefix = templatePrefix ?? Array.Empty<int>();
            _templateSuffix = templateSuffix ?? Array.Empty<int>();
            _vocabulary = vocabulary ?? throw new ArgumentNullException(nameof(vocabulary));
            _merges = merges ?? throw new ArgumentNullException(nameof(merges));
            foreach (var pair in vocabulary) _tokens[pair.Value] = pair.Key;
            foreach (var (content, id, special) in addedTokens ?? Array.Empty<(string, int, bool)>())
            {
                _added[content] = id;
                _tokens[id] = content;
                if (special) _special.Add(content);
                if (content.Length == 0) continue;
                if (!_addedByFirst.TryGetValue(content[0], out var list)) _addedByFirst[content[0]] = list = new List<string>();
                list.Add(content);
            }
            foreach (var list in _addedByFirst.Values) list.Sort((a, b) => b.Length.CompareTo(a.Length));
            _split = new Regex(splitPattern ?? Gpt2Pattern, RegexOptions.CultureInvariant);
            _nfc = nfc;
            _ignoreMerges = ignoreMerges;
            _charToByte = _byteToChar.ToDictionary(p => p.Value, p => p.Key);
        }

        /// <summary>Reads a <c>tokenizer.json</c> with a byte-level BPE model.</summary>
        public static ByteLevelBpeTokenizer FromJson(string json)
        {
            var root = JObject.Parse(json ?? throw new ArgumentNullException(nameof(json)));
            var model = (JObject)(root["model"] ?? throw new InvalidDataException("The tokenizer has no model."));
            if ((string?)model["type"] != "BPE") throw new NotSupportedException($"Only BPE models are supported, not {(string?)model["type"]}.");
            var vocabulary = new Dictionary<string, int>(StringComparer.Ordinal);
            foreach (var pair in (JObject)model["vocab"]!) vocabulary[pair.Key] = (int)pair.Value!;
            var merges = new Dictionary<string, int>(StringComparer.Ordinal);
            int rank = 0;
            foreach (var merge in (JArray)model["merges"]!)
            {
                string left, right;
                if (merge is JArray pair)
                {
                    left = (string)pair[0]!;
                    right = (string)pair[1]!;
                }
                else
                {
                    string text = (string)merge!;
                    int space = text.IndexOf(' ');
                    left = text.Substring(0, space);
                    right = text.Substring(space + 1);
                }
                string key = left + "\u0001" + right;
                if (!merges.ContainsKey(key)) merges[key] = rank;
                rank++;
            }
            bool ignoreMerges = (bool?)model["ignore_merges"] ?? false;
            var isolators = new List<string>();
            bool nfc = false;
            var normalizer = root["normalizer"];
            if (normalizer is JObject n)
            {
                if ((string?)n["type"] == "NFC") nfc = true;
                else throw new NotSupportedException($"Normalizer {(string?)n["type"]} is not supported.");
            }
            var added = new List<(string, int, bool)>();
            foreach (var token in (JArray?)root["added_tokens"] ?? new JArray())
            {
                // A normalized added token is matched in the normalized text, which is the text itself without a normalizer.
                if (((bool?)token["normalized"] == true && nfc) || (bool?)token["lstrip"] == true || (bool?)token["rstrip"] == true || (bool?)token["single_word"] == true)
                    throw new NotSupportedException($"Added token '{(string?)token["content"]}' uses normalization or stripping, which this reader does not support.");
                added.Add(((string)token["content"]!, (int)token["id"]!, (bool?)token["special"] ?? false));
            }
            string? pattern = null;
            var pre = root["pre_tokenizer"] as JObject;
            var steps = (string?)pre?["type"] == "Sequence" ? ((JArray)pre!["pretokenizers"]!).OfType<JObject>().ToList() : pre is null ? new List<JObject>() : new List<JObject> { pre };
            foreach (var step in steps)
            {
                switch ((string?)step["type"])
                {
                    case "Split":
                        if ((string?)step["behavior"] != "Isolated" || (bool?)step["invert"] == true)
                            throw new NotSupportedException("Only an isolated, non-inverted Split is supported.");
                        pattern = (string?)step["pattern"]?["Regex"] ?? throw new NotSupportedException("Only a regular-expression Split is supported.");
                        break;
                    case "Digits":
                        if ((bool?)step["individual_digits"] != true) throw new NotSupportedException("Only Digits with individual_digits is supported.");
                        isolators.Add("Digits");
                        break;
                    case "Punctuation":
                        if (((string?)step["behavior"] ?? "Isolated") != "Isolated") throw new NotSupportedException("Only an isolated Punctuation split is supported.");
                        isolators.Add("Punctuation");
                        break;
                    case "ByteLevel":
                        if ((bool?)step["add_prefix_space"] == true) throw new NotSupportedException("ByteLevel's add_prefix_space is not supported.");
                        if ((bool?)step["use_regex"] != false) pattern ??= Gpt2Pattern;
                        break;
                    default:
                        throw new NotSupportedException($"Pre-tokenizer {(string?)step["type"]} is not supported.");
                }
            }
            // The post-processor: ByteLevel (offsets only) and TemplateProcessing's single-sequence template.
            var prefix = new List<int>();
            var suffix = new List<int>();
            var post = root["post_processor"] as JObject;
            var processors = (string?)post?["type"] == "Sequence" ? ((JArray)post!["processors"]!).OfType<JObject>().ToList() : post is null ? new List<JObject>() : new List<JObject> { post };
            foreach (var processor in processors)
            {
                switch ((string?)processor["type"])
                {
                    case "ByteLevel":
                        break;
                    case "TemplateProcessing":
                        bool sequenceSeen = false;
                        foreach (var item in (JArray)processor["single"]!)
                        {
                            if (item["Sequence"] is not null)
                            {
                                sequenceSeen = true;
                                continue;
                            }
                            string id = (string?)item["SpecialToken"]?["id"] ?? throw new NotSupportedException("Unknown template item.");
                            var ids = ((JArray)processor["special_tokens"]![id]!["ids"]!).Select(v => (int)v).ToArray();
                            (sequenceSeen ? suffix : prefix).AddRange(ids);
                        }
                        break;
                    default:
                        throw new NotSupportedException($"Post-processor {(string?)processor["type"]} is not supported.");
                }
            }
            return new ByteLevelBpeTokenizer(vocabulary, merges, added, pattern, nfc, ignoreMerges, prefix.ToArray(), suffix.ToArray(), isolators);
        }

        /// <summary>The largest token id plus one.</summary>
        public int VocabularySize => _tokens.Keys.Max() + 1;

        /// <summary>The id of a token (an added token or a vocabulary entry).</summary>
        public int TokenId(string token) =>
            _added.TryGetValue(token, out int id) ? id
            : _vocabulary.TryGetValue(token, out id) ? id
            : throw new KeyNotFoundException($"'{token}' is not a token.");

        /// <summary>The ids of <paramref name="text"/> with the post-processor's special tokens
        /// (<c>encode(text, add_special_tokens=True)</c>): the template's tokens around the text's.</summary>
        public int[] EncodeWithSpecialTokens(string text) => _templatePrefix.Concat(Encode(text)).Concat(_templateSuffix).ToArray();

        /// <summary>The ids of <paramref name="text"/> without the post-processor's special tokens
        /// (<c>encode(text, add_special_tokens=False)</c>).</summary>
        public int[] Encode(string text)
        {
            if (text is null) throw new ArgumentNullException(nameof(text));
            var ids = new List<int>();
            int start = 0, position = 0;
            while (position < text.Length)
            {
                string? found = null;
                if (_addedByFirst.TryGetValue(text[position], out var candidates))
                    foreach (var token in candidates)
                        if (string.CompareOrdinal(text, position, token, 0, token.Length) == 0)
                        {
                            found = token;
                            break;
                        }
                if (found is null)
                {
                    position++;
                    continue;
                }
                EncodeOrdinary(text.Substring(start, position - start), ids);
                ids.Add(_added[found]);
                position += found.Length;
                start = position;
            }
            EncodeOrdinary(text.Substring(start), ids);
            return ids.ToArray();
        }

        /// <summary>The text of ids, added tokens included unless <paramref name="skipSpecial"/>.</summary>
        public string Decode(IEnumerable<int> ids, bool skipSpecial = false)
        {
            var output = new StringBuilder();
            var bytes = new List<byte>();
            void Flush()
            {
                if (bytes.Count == 0) return;
                output.Append(Encoding.UTF8.GetString(bytes.ToArray()));
                bytes.Clear();
            }
            foreach (int id in ids)
            {
                if (!_tokens.TryGetValue(id, out var token)) throw new ArgumentOutOfRangeException(nameof(ids), id, "Unknown token id.");
                if (_added.ContainsKey(token))
                {
                    Flush();
                    if (!(skipSpecial && _special.Contains(token))) output.Append(token);
                    continue;
                }
                foreach (char c in token) bytes.Add(_charToByte[c]);
            }
            Flush();
            return output.ToString();
        }

        private void EncodeOrdinary(string segment, List<int> ids)
        {
            if (segment.Length == 0) return;
            string normalized = _nfc ? segment.Normalize(NormalizationForm.FormC) : segment;
            foreach (var piece in Isolate(normalized)) EncodeSplit(piece, ids);
        }

        // HF's Digits (individual) and Punctuation (Isolated) pre-tokenizers in order: each matching character becomes a piece
        // of its own. Digits are char::is_numeric (Nd, Nl, No); punctuation is ASCII punctuation or a Unicode P* category.
        private IEnumerable<string> Isolate(string text)
        {
            IEnumerable<string> pieces = new[] { text };
            foreach (var kind in _isolators)
            {
                var next = new List<string>();
                foreach (var piece in pieces)
                {
                    int start = 0;
                    for (int i = 0; i < piece.Length;)
                    {
                        int width = char.IsHighSurrogate(piece[i]) && i + 1 < piece.Length && char.IsLowSurrogate(piece[i + 1]) ? 2 : 1;
                        var category = System.Globalization.CharUnicodeInfo.GetUnicodeCategory(piece, i);
                        bool isolate = kind == "Digits"
                            ? category is System.Globalization.UnicodeCategory.DecimalDigitNumber or System.Globalization.UnicodeCategory.LetterNumber
                                or System.Globalization.UnicodeCategory.OtherNumber
                            : (piece[i] < 128 && "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~".IndexOf(piece[i]) >= 0) || category is
                                System.Globalization.UnicodeCategory.ConnectorPunctuation or System.Globalization.UnicodeCategory.DashPunctuation
                                or System.Globalization.UnicodeCategory.OpenPunctuation or System.Globalization.UnicodeCategory.ClosePunctuation
                                or System.Globalization.UnicodeCategory.InitialQuotePunctuation or System.Globalization.UnicodeCategory.FinalQuotePunctuation
                                or System.Globalization.UnicodeCategory.OtherPunctuation;
                        if (isolate)
                        {
                            if (i > start) next.Add(piece.Substring(start, i - start));
                            next.Add(piece.Substring(i, width));
                            start = i + width;
                        }
                        i += width;
                    }
                    if (start < piece.Length) next.Add(piece.Substring(start));
                }
                pieces = next;
            }
            return pieces;
        }

        private void EncodeSplit(string text, List<int> ids)
        {
            if (text.Length == 0) return;
            int position = 0;
            foreach (Match match in _split.Matches(text))
            {
                if (match.Length == 0) continue;
                if (match.Index > position) ids.AddRange(_cache.GetOrAdd(text.Substring(position, match.Index - position), Bpe));
                ids.AddRange(_cache.GetOrAdd(match.Value, Bpe));
                position = match.Index + match.Length;
            }
            if (position < text.Length) ids.AddRange(_cache.GetOrAdd(text.Substring(position), Bpe));
        }

        private int[] Bpe(string piece)
        {
            var symbols = Encoding.UTF8.GetBytes(piece).Select(b => _byteToChar[b].ToString()).ToList();
            if (_ignoreMerges && _vocabulary.TryGetValue(string.Concat(symbols), out int whole)) return new[] { whole };
            while (symbols.Count > 1)
            {
                int best = -1, bestRank = int.MaxValue;
                for (int i = 0; i + 1 < symbols.Count; i++)
                    if (_merges.TryGetValue(symbols[i] + "\u0001" + symbols[i + 1], out int rank) && rank < bestRank)
                    {
                        bestRank = rank;
                        best = i;
                    }
                if (best < 0) break;
                symbols[best] += symbols[best + 1];
                symbols.RemoveAt(best + 1);
            }
            return symbols.Select(s => _vocabulary.TryGetValue(s, out int id) ? id
                : throw new InvalidDataException($"The merges made '{s}', which the vocabulary lacks.")).ToArray();
        }
    }
}
