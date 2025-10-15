using System;
using AiDotNet.FoundationModels.Tokenizers;
using AiDotNet.Interfaces;

namespace AiDotNet.Factories
{
    /// <summary>
    /// Factory for creating tokenizer instances
    /// </summary>
    public static class TokenizerFactory
    {
        /// <summary>
        /// Creates a tokenizer based on the specified type
        /// </summary>
        /// <param name="tokenizerType">Type of tokenizer to create</param>
        /// <param name="vocabFile">Path to vocabulary file</param>
        /// <param name="mergesFile">Path to merges file (for BPE)</param>
        /// <param name="doLowerCase">Whether to lowercase input (for WordPiece)</param>
        /// <returns>Tokenizer instance</returns>
        public static ITokenizer CreateTokenizer(
            TokenizerType tokenizerType,
            string vocabFile = "",
            string mergesFile = "",
            bool doLowerCase = true)
        {
            return tokenizerType switch
            {
                TokenizerType.BPE => new BPETokenizer(
                    string.IsNullOrEmpty(vocabFile) ? "vocab.json" : vocabFile,
                    string.IsNullOrEmpty(mergesFile) ? "merges.txt" : mergesFile),
                
                TokenizerType.WordPiece => new WordPieceTokenizer(
                    string.IsNullOrEmpty(vocabFile) ? "vocab.txt" : vocabFile,
                    doLowerCase),
                
                TokenizerType.SentencePiece => new SentencePieceTokenizer(
                    string.IsNullOrEmpty(vocabFile) ? "spiece.model" : vocabFile),
                
                TokenizerType.CharacterLevel => new CharacterLevelTokenizer(
                    includePrintableAscii: true,
                    caseSensitive: !doLowerCase),
                
                TokenizerType.Unigram => new UnigramTokenizer(
                    vocabSize: 8000,
                    treatWhitespaceAsSuffix: true),
                
                TokenizerType.ByteLevelBPE => new ByteLevelBPETokenizer(
                    vocabSize: 50257),
                
                TokenizerType.TikToken => new TikTokenizer(
                    encodingName: "cl100k_base"),
                
                TokenizerType.Claude => new ClaudeTokenizer(
                    useUnicodeNormalization: true),
                
                _ => throw new ArgumentException($"Unknown tokenizer type: {tokenizerType}")
            };
        }

        /// <summary>
        /// Creates a tokenizer appropriate for the specified model type
        /// </summary>
        /// <param name="modelName">Name of the model (e.g., "bert-base", "gpt2", "t5-base")</param>
        /// <param name="vocabPath">Path to vocabulary files</param>
        /// <returns>Tokenizer instance</returns>
        public static ITokenizer CreateTokenizerForModel(string modelName, string vocabPath = "")
        {
            var lowerModelName = modelName.ToLower();
            
            // BERT family uses WordPiece
            if (lowerModelName.Contains("bert") || 
                lowerModelName.Contains("roberta") ||
                lowerModelName.Contains("electra"))
            {
                return new WordPieceTokenizer(
                    string.IsNullOrEmpty(vocabPath) ? $"{modelName}/vocab.txt" : vocabPath,
                    doLowerCase: !lowerModelName.Contains("uncased"));
            }
            
            // GPT-4 uses TikToken
            if (lowerModelName.Contains("gpt-4") || 
                lowerModelName.Contains("gpt4") ||
                lowerModelName.Contains("gpt-3.5") ||
                lowerModelName.Contains("chatgpt"))
            {
                return new TikTokenizer("cl100k_base");
            }
            
            // Claude models
            if (lowerModelName.Contains("claude"))
            {
                return new ClaudeTokenizer();
            }
            
            // GPT-2/GPT-3 uses Byte-level BPE
            if (lowerModelName.Contains("gpt-2") || 
                lowerModelName.Contains("gpt-3") ||
                lowerModelName.Contains("gpt2") ||
                lowerModelName.Contains("gpt3"))
            {
                return new ByteLevelBPETokenizer();
            }
            
            // Original GPT uses regular BPE
            if (lowerModelName.Contains("gpt") || 
                lowerModelName.Contains("codex"))
            {
                return new BPETokenizer(
                    string.IsNullOrEmpty(vocabPath) ? $"{modelName}/vocab.json" : $"{vocabPath}/vocab.json",
                    string.IsNullOrEmpty(vocabPath) ? $"{modelName}/merges.txt" : $"{vocabPath}/merges.txt");
            }
            
            // T5, mBART, XLNet use SentencePiece
            if (lowerModelName.Contains("t5") || 
                lowerModelName.Contains("mbart") ||
                lowerModelName.Contains("xlnet") ||
                lowerModelName.Contains("albert"))
            {
                return new SentencePieceTokenizer(
                    string.IsNullOrEmpty(vocabPath) ? $"{modelName}/spiece.model" : vocabPath);
            }
            
            // Default to WordPiece
            return new WordPieceTokenizer(
                string.IsNullOrEmpty(vocabPath) ? "vocab.txt" : vocabPath);
        }
    }

    /// <summary>
    /// Supported tokenizer types
    /// </summary>
    public enum TokenizerType
    {
        /// <summary>
        /// Byte Pair Encoding tokenizer (used by GPT models)
        /// </summary>
        BPE,
        
        /// <summary>
        /// WordPiece tokenizer (used by BERT models)
        /// </summary>
        WordPiece,
        
        /// <summary>
        /// SentencePiece tokenizer (used by T5, mBART)
        /// </summary>
        SentencePiece,
        
        /// <summary>
        /// Character-level tokenizer (for character-based models)
        /// </summary>
        CharacterLevel,
        
        /// <summary>
        /// Unigram tokenizer (SentencePiece algorithm)
        /// </summary>
        Unigram,
        
        /// <summary>
        /// Byte-level BPE tokenizer (used by GPT-2/GPT-3)
        /// </summary>
        ByteLevelBPE,
        
        /// <summary>
        /// TikToken tokenizer (used by GPT-3.5/GPT-4)
        /// </summary>
        TikToken,
        
        /// <summary>
        /// Claude tokenizer with advanced Unicode handling
        /// </summary>
        Claude
    }
}