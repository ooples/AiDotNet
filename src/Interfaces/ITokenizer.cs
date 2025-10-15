using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;

namespace AiDotNet.Interfaces
{
    /// <summary>
    /// Interface for text tokenization used by foundation models.
    /// Handles conversion between text and numerical representations.
    /// </summary>
    public interface ITokenizer
    {
        /// <summary>
        /// Gets the vocabulary size of the tokenizer
        /// </summary>
        int VocabularySize { get; }

        /// <summary>
        /// Gets the special tokens used by the tokenizer (e.g., [CLS], [SEP], [PAD])
        /// </summary>
        IReadOnlyDictionary<string, int> SpecialTokens { get; }

        /// <summary>
        /// Gets the maximum sequence length supported by the tokenizer
        /// </summary>
        int MaxSequenceLength { get; }

        /// <summary>
        /// Gets the padding token ID
        /// </summary>
        int PadTokenId { get; }

        /// <summary>
        /// Gets the unknown token ID
        /// </summary>
        int UnknownTokenId { get; }

        /// <summary>
        /// Gets the beginning of sequence token ID
        /// </summary>
        int BosTokenId { get; }

        /// <summary>
        /// Gets the end of sequence token ID
        /// </summary>
        int EosTokenId { get; }

        /// <summary>
        /// Tokenizes input text into token IDs
        /// </summary>
        /// <param name="text">Text to tokenize</param>
        /// <param name="addSpecialTokens">Whether to add special tokens</param>
        /// <returns>Vector of token IDs</returns>
        Task<Vector<int>> EncodeAsync(string text, bool addSpecialTokens = true);

        /// <summary>
        /// Tokenizes batch of texts into a structured output
        /// </summary>
        /// <param name="texts">Texts to tokenize</param>
        /// <param name="maxLength">Maximum sequence length (null uses model's max length)</param>
        /// <param name="padding">Whether to pad sequences to max length</param>
        /// <param name="truncation">Whether to truncate sequences exceeding max length</param>
        /// <returns>Tokenized output with input IDs and attention masks</returns>
        Task<TokenizerOutput> EncodeBatchAsync(
            IReadOnlyList<string> texts,
            int? maxLength = null,
            bool padding = true,
            bool truncation = true);

        /// <summary>
        /// Decodes token IDs back to text
        /// </summary>
        /// <param name="tokenIds">Vector of token IDs to decode</param>
        /// <param name="skipSpecialTokens">Whether to skip special tokens in output</param>
        /// <returns>Decoded text</returns>
        Task<string> DecodeAsync(Vector<int> tokenIds, bool skipSpecialTokens = true);

        /// <summary>
        /// Decodes batch of token IDs back to text
        /// </summary>
        /// <param name="tokenIdsBatch">Matrix of token IDs (batch_size x sequence_length)</param>
        /// <param name="skipSpecialTokens">Whether to skip special tokens in output</param>
        /// <returns>List of decoded texts</returns>
        Task<IReadOnlyList<string>> DecodeBatchAsync(Matrix<int> tokenIdsBatch, bool skipSpecialTokens = true);

        /// <summary>
        /// Gets the string tokens for a given text
        /// </summary>
        /// <param name="text">Input text</param>
        /// <returns>List of token strings</returns>
        Task<IReadOnlyList<string>> TokenizeAsync(string text);

        /// <summary>
        /// Gets token embeddings for given token IDs
        /// </summary>
        /// <param name="tokenIds">Vector of token IDs</param>
        /// <returns>Tensor of embeddings (sequence_length x embedding_dim)</returns>
        Task<Tensor<double>> GetTokenEmbeddingsAsync(Vector<int> tokenIds);

        /// <summary>
        /// Validates if the tokenizer is properly initialized
        /// </summary>
        /// <returns>True if tokenizer is ready for use</returns>
        bool IsInitialized { get; }

        /// <summary>
        /// Initializes the tokenizer asynchronously
        /// </summary>
        /// <returns>Task representing the initialization</returns>
        Task InitializeAsync();
    }
}