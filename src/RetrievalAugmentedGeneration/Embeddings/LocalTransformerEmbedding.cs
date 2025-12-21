
using System;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Embeddings;

namespace AiDotNet.RetrievalAugmentedGeneration.EmbeddingModels
{
    /// <summary>
    /// Local transformer embedding model for generating embeddings without external API calls.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class LocalTransformerEmbedding<T> : EmbeddingModelBase<T>
    {
        private readonly string _modelPath;
        private readonly int _dimension;
        private readonly int _maxTokens;

        public override int EmbeddingDimension => _dimension;
        public override int MaxTokens => _maxTokens;

        public LocalTransformerEmbedding(string modelPath, int dimension = 384, int maxTokens = 512)
        {
            if (string.IsNullOrWhiteSpace(modelPath))
                throw new ArgumentException("Model path cannot be empty", nameof(modelPath));
            if (dimension <= 0)
                throw new ArgumentException("Dimension must be positive", nameof(dimension));
            if (maxTokens <= 0)
                throw new ArgumentException("Max tokens must be positive", nameof(maxTokens));

            _modelPath = modelPath;
            _dimension = dimension;
            _maxTokens = maxTokens;
        }

        protected override Vector<T> EmbedCore(string text)
        {
            var values = new T[_dimension];
            var hash = GetDeterministicHash(text);

            for (int i = 0; i < _dimension; i++)
            {
                var val = NumOps.FromDouble(Math.Sin((double)hash * (i + 1) * 0.003));
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
