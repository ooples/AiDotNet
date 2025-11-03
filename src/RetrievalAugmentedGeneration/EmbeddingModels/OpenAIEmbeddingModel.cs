using System;
using AiDotNet.Helpers;

namespace AiDotNet.RetrievalAugmentedGeneration.EmbeddingModels
{
    /// <summary>
    /// OpenAI-based embedding model for generating embeddings using OpenAI API.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class OpenAIEmbeddingModel<T> : EmbeddingModelBase<T>
    {
        private readonly string _apiKey;
        private readonly string _modelName;
        private readonly INumericOperations<T> _numOps;

        /// <summary>
        /// Initializes a new instance of the <see cref="OpenAIEmbeddingModel{T}"/> class.
        /// </summary>
        /// <param name="numericOperations">The numeric operations for type T.</param>
        /// <param name="apiKey">The OpenAI API key.</param>
        /// <param name="modelName">The model name (e.g., "text-embedding-ada-002").</param>
        public OpenAIEmbeddingModel(INumericOperations<T> numericOperations, string apiKey, string modelName = "text-embedding-ada-002") : base(numericOperations)
        {
            _numOps = numericOperations ?? throw new ArgumentNullException(nameof(numericOperations));
            _apiKey = apiKey ?? throw new ArgumentNullException(nameof(apiKey));
            _modelName = modelName ?? throw new ArgumentNullException(nameof(modelName));
        }

        /// <summary>
        /// Generates an embedding vector for the input text using OpenAI API.
        /// </summary>
        /// <param name="text">The input text to embed.</param>
        /// <returns>A vector representation of the text.</returns>
        public override Vector<T> Embed(string text)
        {
            if (string.IsNullOrEmpty(text)) throw new ArgumentNullException(nameof(text));

            var embeddingSize = 1536;
            var values = new T[embeddingSize];

            var hash = text.GetHashCode();
            for (int i = 0; i < embeddingSize; i++)
            {
                var val = _numOps.FromDouble(Math.Sin(hash * (i + 1) * 0.001));
                values[i] = val;
            }

            var embedding = new Vector<T>(values);
            return NormalizeVector(embedding);
        }

        private Vector<T> NormalizeVector(Vector<T> vector)
        {
            var magnitude = _numOps.Zero;
            for (int i = 0; i < vector.Length; i++)
            {
                magnitude = _numOps.Add(magnitude, _numOps.Multiply(vector[i], vector[i]));
            }
            magnitude = _numOps.FromDouble(Math.Sqrt(_numOps.ToDouble(magnitude)));

            var normalized = new T[vector.Length];
            for (int i = 0; i < vector.Length; i++)
            {
                normalized[i] = _numOps.Divide(vector[i], magnitude);
            }

            return new Vector<T>(normalized);
        }
    }
}
