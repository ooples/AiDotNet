using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Threading.Tasks;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using AiDotNet.Tokenization.Algorithms;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.Tokenization.Models;

namespace AiDotNet.Tokenization.HuggingFace
{
    /// <summary>
    /// Loads HuggingFace pretrained tokenizers.
    /// </summary>
    public static class HuggingFaceTokenizerLoader
    {
        private static readonly HttpClient _httpClient = new HttpClient();
        private const string HuggingFaceHubUrl = "https://huggingface.co";

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
            var tokenizerJsonPath = Path.Combine(modelPath, "tokenizer.json");
            if (File.Exists(tokenizerJsonPath))
            {
                return LoadFromTokenizerJson(tokenizerJsonPath);
            }

            var configPath = Path.Combine(modelPath, "tokenizer_config.json");
            var vocabJsonPath = Path.Combine(modelPath, "vocab.json");
            var vocabTxtPath = Path.Combine(modelPath, "vocab.txt");
            var mergesPath = Path.Combine(modelPath, "merges.txt");

            // Determine which vocab file exists (BERT uses vocab.txt, GPT uses vocab.json)
            var vocabPath = File.Exists(vocabJsonPath) ? vocabJsonPath : vocabTxtPath;

            if (!File.Exists(configPath))
                throw new FileNotFoundException("tokenizer_config.json not found");

            // Load configuration
            var configJson = File.ReadAllText(configPath);
            var config = JsonConvert.DeserializeObject<TokenizerConfig>(configJson);

            if (config == null)
                throw new InvalidOperationException("Failed to parse tokenizer configuration");

            // Create special tokens
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

            // Determine tokenizer type and load
            var tokenizerClass = config.TokenizerClass?.ToLowerInvariant() ?? "";

            if (tokenizerClass.Contains("gpt") || tokenizerClass.Contains("bpe") || File.Exists(mergesPath))
            {
                return LoadBpeTokenizer(vocabPath, mergesPath, specialTokens);
            }
            else if (tokenizerClass.Contains("bert") || tokenizerClass.Contains("wordpiece"))
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
        private static BpeTokenizer LoadBpeTokenizer(string vocabPath, string mergesPath, SpecialTokens specialTokens)
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

            var vocabulary = new Vocabulary.Vocabulary(vocabDict, specialTokens.UnkToken);

            // Load merges
            var merges = new Dictionary<(string, string), int>();
            var mergeLines = File.ReadAllLines(mergesPath);
            int order = 0;

            var validMergeLines = mergeLines
                .Where(line => !string.IsNullOrWhiteSpace(line) && !line.StartsWith("#"))
                .Select(line => line.Split(' '))
                .Where(parts => parts.Length >= 2);

            foreach (var parts in validMergeLines)
            {
                merges[(parts[0], parts[1])] = order++;
            }

            return new BpeTokenizer(vocabulary, merges, specialTokens);
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

            var modelType = model["type"]?.Value<string>()?.ToLowerInvariant() ?? "";
            var specialTokens = ExtractSpecialTokensFromJson(root);

            if (modelType == "bpe")
                return LoadBpeFromTokenizerJson(model, specialTokens);
            else if (modelType == "wordpiece")
                return LoadWordPieceFromTokenizerJson(model, specialTokens);
            else if (modelType == "unigram")
                return LoadUnigramFromTokenizerJson(model, specialTokens);
            else if (model["merges"] != null)
                return LoadBpeFromTokenizerJson(model, specialTokens);
            else
                return LoadWordPieceFromTokenizerJson(model, specialTokens);
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

        private static BpeTokenizer LoadBpeFromTokenizerJson(JObject model, SpecialTokens specialTokens)
        {
            var vocabObj = model["vocab"] as JObject;
            if (vocabObj == null)
                throw new InvalidOperationException("Invalid tokenizer.json: missing 'vocab'");

            var vocabDict = new Dictionary<string, int>();
            foreach (var prop in vocabObj.Properties())
                vocabDict[prop.Name] = prop.Value.Value<int>();

            var vocabulary = new Vocabulary.Vocabulary(vocabDict, specialTokens.UnkToken);

            var merges = new Dictionary<(string, string), int>();
            var mergesArray = model["merges"] as JArray;
            if (mergesArray != null)
            {
                int order = 0;
                foreach (var merge in mergesArray)
                {
                    var mergeStr = merge.Value<string>();
                    if (mergeStr != null)
                    {
                        var parts = mergeStr.Split(' ');
                        if (parts.Length >= 2)
                            merges[(parts[0], parts[1])] = order++;
                    }
                }
            }

            return new BpeTokenizer(vocabulary, merges, specialTokens);
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

            foreach (var item in vocabArray)
            {
                if (item is JArray pair && pair.Count >= 2)
                {
                    var token = pair[0].Value<string>();
                    var score = pair[1].Value<double>();
                    if (token != null)
                    {
                        vocabDict[token] = id++;
                        tokenScores[token] = score;
                    }
                }
            }

            var vocabulary = new Vocabulary.Vocabulary(vocabDict, specialTokens.UnkToken);
            return new UnigramTokenizer(vocabulary, tokenScores, specialTokens);
        }
    }
}
