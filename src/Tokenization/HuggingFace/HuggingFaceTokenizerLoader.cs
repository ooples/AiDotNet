using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using AiDotNet.Tokenization.Algorithms;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.Tokenization.Models;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Tokenization.HuggingFace
{
    /// <summary>
    /// Loads HuggingFace pretrained tokenizers.
    /// </summary>
    public static class HuggingFaceTokenizerLoader
    {
        private static readonly HttpClient _httpClient = new HttpClient();
        private const string HuggingFaceHubUrl = "https://huggingface.co";
        private const string EndoTextMarker = "<|" + "endoftext|>";
        private const string EndOfTextMarker = "<|" + "end_of_text|>";
        private const string BeginOfTextMarker = "<|" + "begin_of_text|>";
        private const string StartOfTextMarker = "<|" + "startoftext|>";
        private const string FallbackCausalBpeRegex =
            @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

        /// <summary>
        /// Loads a HuggingFace tokenizer from a directory.
        /// </summary>
        /// <param name="modelPath">The path to the tokenizer directory.</param>
        /// <returns>The loaded tokenizer.</returns>
        public static ITokenizer LoadFromDirectory(string modelPath)
        {
            if (!Directory.Exists(modelPath))
                throw new DirectoryNotFoundException($"Tokenizer directory not found: {modelPath}");

            // Check for tokenizer.json first (modern format)
            var tokenizerJsonPath = GetSafePath(modelPath, "tokenizer.json");
            if (File.Exists(tokenizerJsonPath))
            {
                return LoadFromTokenizerJson(tokenizerJsonPath);
            }

            var configPath = GetSafePath(modelPath, "tokenizer_config.json");
            var vocabJsonPath = GetSafePath(modelPath, "vocab.json");
            var vocabTxtPath = GetSafePath(modelPath, "vocab.txt");
            var mergesPath = GetSafePath(modelPath, "merges.txt");

            // Determine which vocab file exists (BERT uses vocab.txt, GPT uses vocab.json)
            var vocabPath = File.Exists(vocabJsonPath) ? vocabJsonPath : vocabTxtPath;

            if (!File.Exists(configPath))
                throw new FileNotFoundException("tokenizer_config.json not found");

            // Load configuration resiliently (handles AddedToken objects and oversized model_max_length values)
            var configJson = File.ReadAllText(configPath);
            var config = ParseTokenizerConfig(configJson, out var addBosToken, out var decoderAddedTokens);

            // Determine tokenizer type and load
            var tokenizerClass = config.TokenizerClass?.ToLowerInvariant() ?? "";
            bool isBpe = tokenizerClass.Contains("gpt")
                || tokenizerClass.Contains("bpe")
                || tokenizerClass.Contains("roberta")
                || tokenizerClass.Contains("bart")
                || tokenizerClass.Contains("qwen")
                || tokenizerClass.Contains("llama")
                || File.Exists(mergesPath);

            if (isBpe)
            {
                var bpeSpecialTokens = CreateBpeSpecialTokens(config, decoderAddedTokens);
                return LoadBpeTokenizer(
                    vocabPath,
                    mergesPath,
                    bpeSpecialTokens,
                    tokenizerClass,
                    addBosToken ?? false,
                    decoderAddedTokens);
            }

            // Create special tokens with encoder defaults for WordPiece / SentencePiece
            var specialTokens = new SpecialTokens
            {
                UnkToken = config.UnkToken ?? "[UNK]",
                PadToken = config.PadToken ?? "[PAD]",
                ClsToken = config.ClsToken ?? "[CLS]",
                SepToken = config.SepToken ?? "[SEP]",
                MaskToken = config.MaskToken ?? "[MASK]",
                BosToken = config.BosToken ?? "[BOS]",
                EosToken = config.EosToken ?? "[EOS]",
                AdditionalSpecialTokens = config.AdditionalSpecialTokens ?? new List<string>()
            };

            if (tokenizerClass.Contains("bert") || tokenizerClass.Contains("wordpiece"))
            {
                return LoadWordPieceTokenizer(vocabPath, specialTokens);
            }
            else if (tokenizerClass.Contains("sentencepiece") || tokenizerClass.Contains("t5"))
            {
                return LoadSentencePieceTokenizer(vocabPath, specialTokens);
            }
            else
            {
                // Default to WordPiece if type is unknown
                return LoadWordPieceTokenizer(vocabPath, specialTokens);
            }
        }

        /// <summary>
        /// Loads a BPE tokenizer from HuggingFace format.
        /// </summary>
        private static BpeTokenizer LoadBpeTokenizer(
            string vocabPath,
            string mergesPath,
            SpecialTokens specialTokens,
            string tokenizerClass = "",
            bool addBosToken = false,
            List<(int Id, string Content, bool Special)>? addedTokens = null)
        {
            if (!File.Exists(vocabPath))
                throw new FileNotFoundException($"Vocabulary file not found: {vocabPath}");
            if (!File.Exists(mergesPath))
                throw new FileNotFoundException($"Merges file not found: {mergesPath}");

            // Load vocabulary
            var vocabJson = File.ReadAllText(vocabPath);
            var vocabDict = JsonConvert.DeserializeObject<Dictionary<string, int>>(vocabJson);

            if (vocabDict == null)
                throw new InvalidOperationException("Failed to parse vocabulary");

            var specialTokenStrings = new HashSet<string>(StringComparer.Ordinal);
            if (addedTokens != null)
            {
                foreach (var (id, content, _) in addedTokens)
                {
                    if (!string.IsNullOrEmpty(content))
                    {
                        if (!vocabDict.ContainsKey(content))
                            vocabDict[content] = id;
                        specialTokenStrings.Add(content);
                    }
                }
            }

            foreach (var st in specialTokens.GetAllSpecialTokens())
            {
                if (!string.IsNullOrEmpty(st) && vocabDict.ContainsKey(st))
                    specialTokenStrings.Add(st);
            }

            string unkToken = !string.IsNullOrEmpty(specialTokens.UnkToken)
                ? specialTokens.UnkToken
                : (!string.IsNullOrEmpty(specialTokens.EosToken) ? specialTokens.EosToken : "[UNK]");
            var vocabulary = new Vocabulary.Vocabulary(vocabDict, unkToken);

            // Load merges
            var merges = new Dictionary<(string, string), int>();
            var mergeLines = File.ReadAllLines(mergesPath);
            int order = 0;

            foreach (var line in mergeLines)
            {
                if (string.IsNullOrWhiteSpace(line) || line.StartsWith("#", StringComparison.Ordinal))
                    continue;

                int spaceIdx = line.IndexOf(' ');
                if (spaceIdx > 0 && spaceIdx < line.Length - 1)
                {
                    merges[(line.Substring(0, spaceIdx), line.Substring(spaceIdx + 1))] = order++;
                }
            }

            bool byteLevel = tokenizerClass.Contains("gpt")
                || tokenizerClass.Contains("roberta")
                || tokenizerClass.Contains("bart")
                || tokenizerClass.Contains("qwen")
                || tokenizerClass.Contains("llama")
                || vocabDict.Keys.Any(k => k.IndexOf('\u0120') >= 0);

            return new BpeTokenizer(
                vocabulary,
                merges,
                specialTokens,
                pattern: null,
                byteLevel: byteLevel,
                specialTokenStrings: specialTokenStrings,
                addBosToken: addBosToken,
                bosToken: specialTokens.BosToken);
        }

        /// <summary>
        /// Loads a WordPiece tokenizer from HuggingFace format.
        /// </summary>
        private static WordPieceTokenizer LoadWordPieceTokenizer(string vocabPath, SpecialTokens specialTokens)
        {
            if (!File.Exists(vocabPath))
                throw new FileNotFoundException($"Vocabulary file not found: {vocabPath}");

            // Try loading as JSON first
            var vocabJson = File.ReadAllText(vocabPath);
            Dictionary<string, int>? vocabDict = null;

            try
            {
                vocabDict = JsonConvert.DeserializeObject<Dictionary<string, int>>(vocabJson);
            }
            catch (JsonException)
            {
                // If JSON parsing fails, try as text file
                vocabDict = new Dictionary<string, int>();
                var lines = File.ReadAllLines(vocabPath);
                for (int i = 0; i < lines.Length; i++)
                {
                    if (!string.IsNullOrWhiteSpace(lines[i]))
                    {
                        vocabDict[lines[i].Trim()] = i;
                    }
                }
            }

            if (vocabDict == null || vocabDict.Count == 0)
                throw new InvalidOperationException("Failed to parse vocabulary");

            var vocabulary = new Vocabulary.Vocabulary(vocabDict, specialTokens.UnkToken);

            return new WordPieceTokenizer(vocabulary, specialTokens);
        }

        /// <summary>
        /// Loads a SentencePiece tokenizer from HuggingFace format.
        /// </summary>
        private static SentencePieceTokenizer LoadSentencePieceTokenizer(string vocabPath, SpecialTokens specialTokens)
        {
            if (!File.Exists(vocabPath))
                throw new FileNotFoundException($"Vocabulary file not found: {vocabPath}");

            // Load vocabulary
            var vocabJson = File.ReadAllText(vocabPath);
            var vocabDict = JsonConvert.DeserializeObject<Dictionary<string, int>>(vocabJson);

            if (vocabDict == null)
                throw new InvalidOperationException("Failed to parse vocabulary");

            var vocabulary = new Vocabulary.Vocabulary(vocabDict, specialTokens.UnkToken);

            // Create default scores based on token position as a proxy for frequency
            // Lower index = more common = higher score (less negative)
            // This approximates the log probability distribution of real tokenizers
            var pieceScores = new Dictionary<string, double>();
            int index = 0;
            int vocabSize = vocabDict.Count;
            foreach (var token in vocabDict.Keys)
            {
                // Use negative log of relative position to approximate frequency-based scores
                // Common tokens (lower index) get higher scores (closer to 0)
                // Rare tokens (higher index) get lower scores (more negative)
                double relativePosition = (double)(index + 1) / vocabSize;
                pieceScores[token] = Math.Log(1.0 / (relativePosition + 0.1));
                index++;
            }

            return new SentencePieceTokenizer(vocabulary, pieceScores, specialTokens);
        }

        /// <summary>
        /// Saves a tokenizer to HuggingFace format.
        /// </summary>
        /// <remarks>
        /// <para><b>Limitation:</b> This method saves vocabulary and configuration but does not save
        /// BPE merge rules. BPE tokenizers saved with this method will not fully round-trip -
        /// they will need to be retrained or loaded from a different source to recover merge information.</para>
        /// <para>For full BPE tokenizer serialization, consider using the original HuggingFace tokenizer files.</para>
        /// </remarks>
        /// <param name="tokenizer">The tokenizer to save.</param>
        /// <param name="outputPath">The output directory path.</param>
        public static void SaveToDirectory(ITokenizer tokenizer, string outputPath)
        {
            if (!Directory.Exists(outputPath))
                Directory.CreateDirectory(outputPath);

            // Save vocabulary
            var vocabPath = Path.Combine(outputPath, "vocab.json");
            var vocabDict = tokenizer.Vocabulary.TokenToId.ToDictionary(kv => kv.Key, kv => kv.Value);
            var vocabJson = JsonConvert.SerializeObject(vocabDict, Formatting.Indented);
            File.WriteAllText(vocabPath, vocabJson);

            // Save configuration
            var configPath = Path.Combine(outputPath, "tokenizer_config.json");
            var config = new TokenizerConfig
            {
                UnkToken = tokenizer.SpecialTokens.UnkToken,
                PadToken = tokenizer.SpecialTokens.PadToken,
                ClsToken = tokenizer.SpecialTokens.ClsToken,
                SepToken = tokenizer.SpecialTokens.SepToken,
                MaskToken = tokenizer.SpecialTokens.MaskToken,
                BosToken = tokenizer.SpecialTokens.BosToken,
                EosToken = tokenizer.SpecialTokens.EosToken,
                AdditionalSpecialTokens = tokenizer.SpecialTokens.AdditionalSpecialTokens
            };

            var configJson = JsonConvert.SerializeObject(config, Formatting.Indented);
            File.WriteAllText(configPath, configJson);
        }

        /// <summary>
        /// Loads a tokenizer from HuggingFace Hub by model name.
        /// </summary>
        /// <remarks>
        /// <para><b>Warning:</b> This method uses sync-over-async internally and may cause deadlocks
        /// in UI applications or ASP.NET contexts with synchronization contexts.
        /// Prefer using <see cref="LoadFromHubAsync"/> when possible.</para>
        /// <para>Files are cached locally, so subsequent calls will not make network requests.</para>
        /// </remarks>
        /// <param name="modelName">The model name (e.g., "bert-base-uncased", "gpt2").</param>
        /// <param name="cacheDir">Optional cache directory.</param>
        /// <returns>The loaded tokenizer.</returns>
        public static ITokenizer LoadFromHub(string modelName, string? cacheDir = null)
        {
            // Note: Using Task.Run to avoid deadlocks in contexts with synchronization contexts
            // This is a workaround for the sync-over-async pattern
            return Task.Run(() => LoadFromHubAsync(modelName, cacheDir)).GetAwaiter().GetResult();
        }

        /// <summary>
        /// Asynchronously loads a tokenizer from HuggingFace Hub.
        /// </summary>
        public static async Task<ITokenizer> LoadFromHubAsync(string modelName, string? cacheDir = null)
        {
            if (string.IsNullOrWhiteSpace(modelName))
                throw new ArgumentException("Model name cannot be empty", nameof(modelName));

            cacheDir ??= Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "huggingface", "tokenizers");
            var modelCacheDir = Path.Combine(cacheDir, modelName.Replace("/", "--"));

            if (!Directory.Exists(modelCacheDir))
                Directory.CreateDirectory(modelCacheDir);

            var filesToDownload = new[] { "tokenizer.json", "tokenizer_config.json", "vocab.json", "vocab.txt", "merges.txt" };

            foreach (var fileName in filesToDownload)
            {
                var localPath = Path.Combine(modelCacheDir, fileName);
                if (!File.Exists(localPath))
                {
                    await DownloadFileAsync(modelName, fileName, localPath);
                }
            }

            return LoadFromDirectory(modelCacheDir);
        }

        private static async Task DownloadFileAsync(string modelName, string fileName, string localPath)
        {
            var url = $"{HuggingFaceHubUrl}/{modelName}/resolve/main/{fileName}";
            try
            {
                var response = await _httpClient.GetAsync(url);
                if (response.IsSuccessStatusCode)
                {
                    var content = await response.Content.ReadAsByteArrayAsync();
                    File.WriteAllBytes(localPath, content);
                }
            }
            catch (HttpRequestException)
            {
                // Silently ignore - not all tokenizers have all files
            }
        }

        /// <summary>
        /// Loads a tokenizer from a tokenizer.json file.
        /// </summary>
        public static ITokenizer LoadFromTokenizerJson(string tokenizerJsonPath)
        {
            if (!File.Exists(tokenizerJsonPath))
                throw new FileNotFoundException($"tokenizer.json not found: {tokenizerJsonPath}");

            var json = File.ReadAllText(tokenizerJsonPath);
            var root = JObject.Parse(json);

            var model = root["model"] as JObject;
            if (model == null)
                throw new InvalidOperationException("Invalid tokenizer.json: missing 'model' section");

            TokenizerConfig? siblingConfig = null;
            bool? configAddBos = null;
            List<(int Id, string Content, bool Special)>? decoderAddedTokens = null;
            var parentDir = Path.GetDirectoryName(Path.GetFullPath(tokenizerJsonPath));
            if (!string.IsNullOrEmpty(parentDir))
            {
                var siblingConfigPath = GetSafePath(parentDir, "tokenizer_config.json");
                if (File.Exists(siblingConfigPath))
                {
                    try
                    {
                        siblingConfig = ParseTokenizerConfig(
                            File.ReadAllText(siblingConfigPath),
                            out configAddBos,
                            out decoderAddedTokens);
                    }
                    catch (JsonException)
                    {
                        // Ignore malformed optional sibling config; tokenizer.json remains authoritative.
                    }
                }
            }

            var modelType = model["type"]?.Value<string>()?.ToLowerInvariant() ?? "";
            bool isBpe = modelType == "bpe" || (modelType != "wordpiece" && modelType != "unigram" && model["merges"] != null);

            if (isBpe)
            {
                var allAddedTokens = ExtractAddedTokensList(root, decoderAddedTokens);
                var bpeSpecialTokens = CreateBpeSpecialTokens(siblingConfig, allAddedTokens);
                ExtractBpeOptionsFromJson(root, configAddBos, bpeSpecialTokens, out var pattern, out var byteLevel, out var addBosToken);
                return LoadBpeFromTokenizerJson(model, bpeSpecialTokens, allAddedTokens, pattern, byteLevel, addBosToken);
            }

            var specialTokens = ExtractSpecialTokensFromJson(root);
            if (modelType == "wordpiece")
                return LoadWordPieceFromTokenizerJson(model, specialTokens);
            else if (modelType == "unigram")
                return LoadUnigramFromTokenizerJson(model, specialTokens);
            else
                return LoadWordPieceFromTokenizerJson(model, specialTokens);
        }

        private static TokenizerConfig ParseTokenizerConfig(
            string configJson,
            out bool? addBosToken,
            out List<(int Id, string Content, bool Special)> decoderAddedTokens)
        {
            var root = JObject.Parse(configJson);
            addBosToken = root["add_bos_token"]?.Type == JTokenType.Boolean
                ? root["add_bos_token"]!.Value<bool>()
                : null;

            decoderAddedTokens = new List<(int Id, string Content, bool Special)>();
            if (root["added_tokens_decoder"] is JObject decoderObj)
            {
                foreach (var prop in decoderObj.Properties().Where(prop => int.TryParse(prop.Name, out _) && prop.Value is JObject))
                {
                    int id = int.Parse(prop.Name);
                    var entry = (JObject)prop.Value;
                    var content = entry["content"]?.Value<string>();
                    bool special = entry["special"]?.Value<bool>() ?? false;
                    if (!string.IsNullOrEmpty(content))
                        decoderAddedTokens.Add((id, content, special));
                }
            }

            List<string>? additionalSpecial = null;
            if (root["additional_special_tokens"] is JArray addArr)
            {
                additionalSpecial = new List<string>();
                foreach (var s in addArr.Select(ReadTokenString))
                {
                    if (!string.IsNullOrEmpty(s))
                        additionalSpecial.Add(s!);
                }
            }

            int? maxLen = null;
            var maxLenToken = root["model_max_length"];
            if (maxLenToken != null
                && (maxLenToken.Type == JTokenType.Integer || maxLenToken.Type == JTokenType.Float)
                && long.TryParse(
                    maxLenToken.ToString(),
                    System.Globalization.NumberStyles.Integer,
                    System.Globalization.CultureInfo.InvariantCulture,
                    out long raw)
                && raw > 0
                && raw <= int.MaxValue)
            {
                maxLen = (int)raw;
            }

            return new TokenizerConfig
            {
                TokenizerClass = root["tokenizer_class"]?.Value<string>(),
                ModelType = root["model_type"]?.Value<string>(),
                VocabFile = root["vocab_file"]?.Value<string>(),
                MergesFile = root["merges_file"]?.Value<string>(),
                UnkToken = ReadTokenString(root["unk_token"]),
                PadToken = ReadTokenString(root["pad_token"]),
                ClsToken = ReadTokenString(root["cls_token"]),
                SepToken = ReadTokenString(root["sep_token"]),
                MaskToken = ReadTokenString(root["mask_token"]),
                BosToken = ReadTokenString(root["bos_token"]),
                EosToken = ReadTokenString(root["eos_token"]),
                DoLowerCase = root["do_lower_case"]?.Type == JTokenType.Boolean && root["do_lower_case"]!.Value<bool>(),
                ModelMaxLength = maxLen,
                AdditionalSpecialTokens = additionalSpecial
            };
        }

        private static string? ReadTokenString(JToken? token)
        {
            if (token == null || token.Type == JTokenType.Null)
                return null;
            if (token.Type == JTokenType.String)
                return token.Value<string>();
            if (token is JObject obj)
                return obj["content"]?.Value<string>();
            return null;
        }

        private static List<(int Id, string Content, bool Special)> ExtractAddedTokensList(
            JObject root,
            List<(int Id, string Content, bool Special)>? decoderAddedTokens)
        {
            var result = new List<(int Id, string Content, bool Special)>();
            var seen = new HashSet<string>(StringComparer.Ordinal);

            if (root["added_tokens"] is JArray addedTokens)
            {
                foreach (var token in addedTokens)
                {
                    var content = token["content"]?.Value<string>();
                    int? id = token["id"]?.Type == JTokenType.Integer ? token["id"]!.Value<int>() : null;
                    bool special = token["special"]?.Value<bool>() ?? false;
                    if (!string.IsNullOrEmpty(content) && id.HasValue && seen.Add(content!))
                    {
                        result.Add((id.Value, content!, special));
                    }
                }
            }

            if (decoderAddedTokens != null)
            {
                foreach (var entry in decoderAddedTokens.Where(entry => !string.IsNullOrEmpty(entry.Content) && seen.Add(entry.Content)))
                {
                    result.Add(entry);
                }
            }

            return result;
        }

        private static SpecialTokens CreateBpeSpecialTokens(
            TokenizerConfig? config,
            List<(int Id, string Content, bool Special)>? addedTokens)
        {
            // Causal / BPE tokenizers must NOT inherit BERT's [CLS]/[SEP]/[MASK] defaults unless explicitly configured,
            // otherwise AddSpecialTokensToSequence wraps prompts in out-of-vocabulary [CLS] ... [SEP] markers.
            var specialTokens = new SpecialTokens
            {
                UnkToken = config?.UnkToken ?? string.Empty,
                PadToken = config?.PadToken ?? string.Empty,
                ClsToken = config?.ClsToken ?? string.Empty,
                SepToken = config?.SepToken ?? string.Empty,
                MaskToken = config?.MaskToken ?? string.Empty,
                BosToken = config?.BosToken ?? string.Empty,
                EosToken = config?.EosToken ?? string.Empty,
                AdditionalSpecialTokens = config?.AdditionalSpecialTokens != null
                    ? new List<string>(config.AdditionalSpecialTokens)
                    : new List<string>()
            };

            if (addedTokens != null)
            {
                foreach (var (_, content, special) in addedTokens)
                {
                    if (!special || string.IsNullOrEmpty(content))
                        continue;

                    var lower = content.ToLowerInvariant();
                    if (string.IsNullOrEmpty(specialTokens.UnkToken) && lower.Contains("unk"))
                        specialTokens.UnkToken = content;
                    else if (string.IsNullOrEmpty(specialTokens.PadToken) && lower.Contains("pad"))
                        specialTokens.PadToken = content;
                    else if (string.IsNullOrEmpty(specialTokens.ClsToken) && lower.Contains("cls"))
                        specialTokens.ClsToken = content;
                    else if (string.IsNullOrEmpty(specialTokens.SepToken) && lower.Contains("sep"))
                        specialTokens.SepToken = content;
                    else if (string.IsNullOrEmpty(specialTokens.MaskToken) && lower.Contains("mask"))
                        specialTokens.MaskToken = content;
                    else if (string.IsNullOrEmpty(specialTokens.BosToken) &&
                             (lower.Contains("bos") || lower == "<s>" || lower == BeginOfTextMarker || lower == StartOfTextMarker))
                        specialTokens.BosToken = content;
                    else if (string.IsNullOrEmpty(specialTokens.EosToken) &&
                             (lower.Contains("eos") || lower == "</s>" || lower == EndoTextMarker || lower == EndOfTextMarker))
                        specialTokens.EosToken = content;
                    else if (!specialTokens.AdditionalSpecialTokens.Contains(content))
                        specialTokens.AdditionalSpecialTokens.Add(content);
                }
            }

            if (string.IsNullOrEmpty(specialTokens.UnkToken))
            {
                specialTokens.UnkToken = !string.IsNullOrEmpty(specialTokens.EosToken)
                    ? specialTokens.EosToken
                    : "[UNK]";
            }

            return specialTokens;
        }

        private static void ExtractBpeOptionsFromJson(
            JObject root,
            bool? configAddBos,
            SpecialTokens specialTokens,
            out string? pattern,
            out bool byteLevel,
            out bool addBosToken)
        {
            pattern = null;
            byteLevel = false;
            bool individualDigits = false;

            var preNodes = new List<JObject>();
            CollectStageNodes(root["pre_tokenizer"], "pretokenizers", preNodes);
            foreach (var node in preNodes)
            {
                var type = node["type"]?.Value<string>();
                if (string.Equals(type, "ByteLevel", StringComparison.OrdinalIgnoreCase))
                {
                    byteLevel = true;
                }
                else if (string.Equals(type, "Split", StringComparison.OrdinalIgnoreCase))
                {
                    var regexStr = node["pattern"]?["Regex"]?.Value<string>();
                    if (!string.IsNullOrEmpty(regexStr) && pattern == null)
                        pattern = regexStr;
                }
                else if (string.Equals(type, "Digits", StringComparison.OrdinalIgnoreCase) &&
                         (node["individual_digits"]?.Value<bool>() ?? false))
                {
                    individualDigits = true;
                }
            }

            var decNodes = new List<JObject>();
            CollectStageNodes(root["decoder"], "decoders", decNodes);
            byteLevel |= decNodes.Any(node => string.Equals(node["type"]?.Value<string>(), "ByteLevel", StringComparison.OrdinalIgnoreCase));

            if (individualDigits && !string.IsNullOrEmpty(pattern))
            {
                pattern = pattern!.Replace(@"\p{N}+", @"\p{N}");
            }

            if (!string.IsNullOrEmpty(pattern))
            {
                try
                {
                    _ = new Regex(pattern!, RegexOptions.None, TimeSpan.FromSeconds(1));
                }
                catch (ArgumentException)
                {
                    pattern = FallbackCausalBpeRegex;
                }
            }

            addBosToken = configAddBos ?? DetectAddBosFromPostProcessor(root["post_processor"], specialTokens.BosToken);
        }

        private static void CollectStageNodes(JToken? token, string sequenceKey, List<JObject> output)
        {
            if (token is not JObject obj)
                return;

            if (string.Equals(obj["type"]?.Value<string>(), "Sequence", StringComparison.OrdinalIgnoreCase) &&
                obj[sequenceKey] is JArray children)
            {
                foreach (var child in children)
                    CollectStageNodes(child, sequenceKey, output);
            }
            else
            {
                output.Add(obj);
            }
        }

        private static bool DetectAddBosFromPostProcessor(JToken? postProcessor, string? bosToken)
        {
            if (postProcessor is not JObject obj || string.IsNullOrEmpty(bosToken))
                return false;

            if (string.Equals(obj["type"]?.Value<string>(), "TemplateProcessing", StringComparison.OrdinalIgnoreCase) &&
                obj["single"] is JArray single &&
                single.Count >= 2 &&
                single[0] is JObject first &&
                first["SpecialToken"] is JObject stObj)
            {
                var id = stObj["id"]?.Value<string>();
                bool hasTrailingSpecial = single[single.Count - 1] is JObject last && last["SpecialToken"] != null;
                if (!hasTrailingSpecial && string.Equals(id, bosToken, StringComparison.Ordinal))
                    return true;
            }

            return false;
        }

        private static SpecialTokens ExtractSpecialTokensFromJson(JObject root)
        {
            var specialTokens = new SpecialTokens();
            var addedTokens = root["added_tokens"] as JArray;

            if (addedTokens != null)
            {
                foreach (var token in addedTokens)
                {
                    var content = token["content"]?.Value<string>();
                    var special = token["special"]?.Value<bool>() ?? false;

                    if (content != null && special)
                    {
                        var lower = content.ToLowerInvariant();
                        if (lower.Contains("unk")) specialTokens.UnkToken = content;
                        else if (lower.Contains("pad")) specialTokens.PadToken = content;
                        else if (lower.Contains("cls")) specialTokens.ClsToken = content;
                        else if (lower.Contains("sep")) specialTokens.SepToken = content;
                        else if (lower.Contains("mask")) specialTokens.MaskToken = content;
                        else if (lower.Contains("bos") || lower == "<s>") specialTokens.BosToken = content;
                        else if (lower.Contains("eos") || lower == "</s>") specialTokens.EosToken = content;
                    }
                }
            }

            if (string.IsNullOrEmpty(specialTokens.UnkToken)) specialTokens.UnkToken = "[UNK]";
            return specialTokens;
        }

        private static BpeTokenizer LoadBpeFromTokenizerJson(
            JObject model,
            SpecialTokens specialTokens,
            List<(int Id, string Content, bool Special)> addedTokens,
            string? pattern,
            bool byteLevel,
            bool addBosToken)
        {
            var vocabObj = model["vocab"] as JObject;
            if (vocabObj == null)
                throw new InvalidOperationException("Invalid tokenizer.json: missing 'vocab'");

            var vocabDict = new Dictionary<string, int>(StringComparer.Ordinal);
            foreach (var prop in vocabObj.Properties())
                vocabDict[prop.Name] = prop.Value.Value<int>();

            var specialTokenStrings = new HashSet<string>(StringComparer.Ordinal);
            foreach (var (id, content, _) in addedTokens)
            {
                if (!string.IsNullOrEmpty(content))
                {
                    vocabDict[content] = id;
                    specialTokenStrings.Add(content);
                }
            }

            foreach (var st in specialTokens.GetAllSpecialTokens())
            {
                if (!string.IsNullOrEmpty(st) && vocabDict.ContainsKey(st))
                    specialTokenStrings.Add(st);
            }

            var vocabulary = new Vocabulary.Vocabulary(vocabDict, specialTokens.UnkToken);

            var merges = new Dictionary<(string, string), int>();
            var mergesArray = model["merges"] as JArray;
            if (mergesArray != null)
            {
                int order = 0;
                foreach (var entry in mergesArray)
                {
                    if (entry.Type == JTokenType.String)
                    {
                        var mergeStr = entry.Value<string>();
                        if (!string.IsNullOrEmpty(mergeStr))
                        {
                            int spaceIdx = mergeStr!.IndexOf(' ');
                            if (spaceIdx > 0 && spaceIdx < mergeStr.Length - 1)
                            {
                                merges[(mergeStr.Substring(0, spaceIdx), mergeStr.Substring(spaceIdx + 1))] = order++;
                            }
                        }
                    }
                    else if (entry is JArray pair && pair.Count >= 2)
                    {
                        var first = pair[0]?.Value<string>();
                        var second = pair[1]?.Value<string>();
                        if (first != null && second != null)
                        {
                            merges[(first, second)] = order++;
                        }
                    }
                }
            }

            return new BpeTokenizer(
                vocabulary,
                merges,
                specialTokens,
                pattern: pattern,
                byteLevel: byteLevel,
                specialTokenStrings: specialTokenStrings,
                addBosToken: addBosToken,
                bosToken: specialTokens.BosToken);
        }

        private static WordPieceTokenizer LoadWordPieceFromTokenizerJson(JObject model, SpecialTokens specialTokens)
        {
            var vocabObj = model["vocab"] as JObject;
            if (vocabObj == null)
                throw new InvalidOperationException("Invalid tokenizer.json: missing 'vocab'");

            var vocabDict = new Dictionary<string, int>();
            foreach (var prop in vocabObj.Properties())
                vocabDict[prop.Name] = prop.Value.Value<int>();

            var vocabulary = new Vocabulary.Vocabulary(vocabDict, specialTokens.UnkToken);
            return new WordPieceTokenizer(vocabulary, specialTokens);
        }

        private static UnigramTokenizer LoadUnigramFromTokenizerJson(JObject model, SpecialTokens specialTokens)
        {
            var vocabArray = model["vocab"] as JArray;
            if (vocabArray == null)
                throw new InvalidOperationException("Invalid tokenizer.json: missing 'vocab' in unigram model");

            var vocabDict = new Dictionary<string, int>();
            var tokenScores = new Dictionary<string, double>();
            int id = 0;

            var validPairs = vocabArray
                .OfType<JArray>()
                .Where(pair => pair.Count >= 2 && pair[0].Value<string>() != null);

            foreach (var pair in validPairs)
            {
                var token = pair[0].Value<string>();
                var score = pair[1].Value<double>();
                if (token != null)
                {
                    vocabDict[token] = id++;
                    tokenScores[token] = score;
                }
            }

            var vocabulary = new Vocabulary.Vocabulary(vocabDict, specialTokens.UnkToken);
            return new UnigramTokenizer(vocabulary, tokenScores, specialTokens);
        }

        /// <summary>
        /// Safely combines a base path with a filename, preventing path traversal attacks.
        /// </summary>
        /// <param name="basePath">The base directory path.</param>
        /// <param name="fileName">The filename to combine.</param>
        /// <returns>The combined path, guaranteed to be within basePath.</returns>
        private static string GetSafePath(string basePath, string fileName)
        {
            // Validate filename doesn't contain path traversal characters
            if (fileName.Contains("..") || fileName.Contains('/') || fileName.Contains('\\'))
            {
                throw new ArgumentException($"Invalid file path: {fileName} contains path traversal characters");
            }

            var normalizedBase = Path.GetFullPath(basePath);
            // Ensure base path ends with directory separator for proper concatenation
            if (!normalizedBase.EndsWith(Path.DirectorySeparatorChar.ToString()))
            {
                normalizedBase += Path.DirectorySeparatorChar;
            }

            var combined = Path.GetFullPath(normalizedBase + fileName);

            // Double-check the resulting path is within the base directory
            if (!combined.StartsWith(normalizedBase, StringComparison.OrdinalIgnoreCase))
            {
                throw new ArgumentException($"Invalid file path: {fileName} attempts path traversal");
            }

            return combined;
        }
    }
}
