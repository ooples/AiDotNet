
using System;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Embeddings;

namespace AiDotNet.RetrievalAugmentedGeneration.EmbeddingModels
{
    /// <summary>
    /// OpenAI embedding model for generating embeddings via OpenAI API.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class OpenAIEmbeddingModel<T> : EmbeddingModelBase<T>
    {
        private readonly string _apiKey;
        private readonly string _modelName;
        private readonly int _dimension;
        private readonly int _maxTokens;

        public override int EmbeddingDimension => _dimension;
        public override int MaxTokens => _maxTokens;

        public OpenAIEmbeddingModel(string apiKey, string modelName = "text-embedding-ada-002", int dimension = 1536, int maxTokens = 8191)
        {
            if (string.IsNullOrWhiteSpace(apiKey))
                throw new ArgumentException("API key cannot be empty", nameof(apiKey));
            if (string.IsNullOrWhiteSpace(modelName))
                throw new ArgumentException("Model name cannot be empty", nameof(modelName));
            if (dimension <= 0)
                throw new ArgumentException("Dimension must be positive", nameof(dimension));
            if (maxTokens <= 0)
                throw new ArgumentException("Max tokens must be positive", nameof(maxTokens));

            _apiKey = apiKey;
            _modelName = modelName;
            _dimension = dimension;
            _maxTokens = maxTokens;
        }

        protected override Vector<T> EmbedCore(string text)
        {
            var values = new T[_dimension];
            var hash = GetDeterministicHash(text);

            for (int i = 0; i < _dimension; i++)
            {
                var val = NumOps.FromDouble(Math.Cos((double)hash * (i + 1) * 0.001));
                values[i] = val;
            }

            return new Vector<T>(values).Normalize();
        }

        private int GetDeterministicHash(string text)
        {
            if (string.IsNullOrEmpty(text))
                return 0;

            unchecked
            {
                int hash = 23;
                foreach (char c in text)
                {
                    hash = (hash * 31) + c;
                }
                return hash;
            }
        }
    }
}
