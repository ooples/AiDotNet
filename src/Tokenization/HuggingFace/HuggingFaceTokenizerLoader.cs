using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json;
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
        /// <summary>
        /// Loads a HuggingFace tokenizer from a directory.
        /// </summary>
        /// <param name="modelPath">The path to the tokenizer directory.</param>
        /// <returns>The loaded tokenizer.</returns>
        public static ITokenizer LoadFromDirectory(string modelPath)
        {
            if (!Directory.Exists(modelPath))
                throw new DirectoryNotFoundException($"Tokenizer directory not found: {modelPath}");

            var configPath = Path.Combine(modelPath, "tokenizer_config.json");
            var vocabPath = Path.Combine(modelPath, "vocab.json");
            var mergesPath = Path.Combine(modelPath, "merges.txt");

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

            // Create default scores (would ideally load from model file)
            var pieceScores = new Dictionary<string, double>();
            foreach (var token in vocabDict.Keys)
            {
                pieceScores[token] = 0.0; // Default score
            }

            return new SentencePieceTokenizer(vocabulary, pieceScores, specialTokens);
        }

        /// <summary>
        /// Saves a tokenizer to HuggingFace format.
        /// </summary>
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
    }
}
