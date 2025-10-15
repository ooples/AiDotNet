using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.NeuralNetworks
{
    /// <summary>
    /// Simple text encoder implementation for demonstration purposes.
    /// In production, this would use a proper text embedding model like CLIP or BERT.
    /// </summary>
    public class SimpleTextEncoder : AiDotNet.Interfaces.ITextEncoder
    {
        private readonly int embeddingDim;
        private readonly int maxSequenceLength;
        private readonly Dictionary<string, int> vocabulary;
        private readonly Random random;

        /// <summary>
        /// Initializes a new instance of the SimpleTextEncoder class.
        /// </summary>
        /// <param name="embeddingDim">Dimension of text embeddings (default: 768, CLIP-like)</param>
        /// <param name="maxSequenceLength">Maximum sequence length (default: 77)</param>
        /// <param name="seed">Random seed for reproducibility</param>
        public SimpleTextEncoder(int embeddingDim = 768, int maxSequenceLength = 77, int? seed = null)
        {
            this.embeddingDim = embeddingDim;
            this.maxSequenceLength = maxSequenceLength;
            this.vocabulary = new Dictionary<string, int>();
            this.random = seed.HasValue ? new Random(seed.Value) : new Random();
            
            InitializeVocabulary();
        }

        /// <summary>
        /// Encodes text to embedding representation.
        /// </summary>
        /// <param name="text">Input text to encode</param>
        /// <returns>Text embedding tensor with shape [1, embeddingDim]</returns>
        public Tensor<double> Encode(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                throw new ArgumentException("Text cannot be null or empty", nameof(text));

            // Tokenize text (simplified tokenization)
            var tokens = TokenizeText(text);
            
            // Convert to embeddings (in production, this would use learned embeddings)
            var embeddings = new Tensor<double>(new[] { 1, embeddingDim });
            
            // Generate pseudo-embeddings based on text features
            // In a real implementation, this would use pre-trained embeddings
            GeneratePseudoEmbeddings(embeddings, tokens);

            return embeddings;
        }

        /// <summary>
        /// Simple tokenization method.
        /// </summary>
        private List<string> TokenizeText(string text)
        {
            // Convert to lowercase and split by whitespace and punctuation
            var cleanText = text.ToLowerInvariant();
            var tokens = cleanText.Split(new[] { ' ', '.', ',', '!', '?', ';', ':', '\n', '\r', '\t' }, 
                                       StringSplitOptions.RemoveEmptyEntries);
            
            // Limit to max sequence length
            if (tokens.Length > maxSequenceLength)
            {
                tokens = tokens.Take(maxSequenceLength).ToArray();
            }

            return tokens.ToList();
        }

        /// <summary>
        /// Initialize a basic vocabulary for demonstration.
        /// </summary>
        private void InitializeVocabulary()
        {
            // Add some common words for demonstration
            var commonWords = new[] 
            {
                "a", "the", "and", "or", "but", "in", "on", "at", "to", "for",
                "of", "with", "by", "from", "up", "about", "into", "through",
                "beautiful", "sunset", "mountains", "ocean", "forest", "city",
                "vibrant", "colors", "dramatic", "lighting", "peaceful", "serene"
            };

            for (int i = 0; i < commonWords.Length; i++)
            {
                vocabulary[commonWords[i]] = i;
            }
        }

        /// <summary>
        /// Generate pseudo-embeddings for demonstration.
        /// In production, this would use actual learned embeddings.
        /// </summary>
        private void GeneratePseudoEmbeddings(Tensor<double> embeddings, List<string> tokens)
        {
            // Create a deterministic but varied embedding based on the tokens
            var embedding = new double[embeddingDim];
            
            // Initialize with small random values
            for (int i = 0; i < embeddingDim; i++)
            {
                embedding[i] = (random.NextDouble() - 0.5) * 0.1;
            }

            // Add token-specific features
            foreach (var token in tokens)
            {
                if (vocabulary.ContainsKey(token))
                {
                    var tokenId = vocabulary[token];
                    // Create token-specific pattern
                    for (int i = 0; i < embeddingDim; i++)
                    {
                        var angle = (2.0 * Math.PI * tokenId * i) / embeddingDim;
                        embedding[i] += Math.Sin(angle) * 0.1 + Math.Cos(angle * 2) * 0.05;
                    }
                }
            }

            // Normalize the embedding
            var norm = Math.Sqrt(embedding.Sum(x => x * x));
            if (norm > 0)
            {
                for (int i = 0; i < embeddingDim; i++)
                {
                    embedding[i] /= norm;
                }
            }

            // Set the values in the tensor
            for (int i = 0; i < embeddingDim; i++)
            {
                embeddings[0, i] = embedding[i];
            }
        }
    }
}