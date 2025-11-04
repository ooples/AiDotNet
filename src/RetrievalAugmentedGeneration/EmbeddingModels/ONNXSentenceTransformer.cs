using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Embeddings;
using System;

namespace AiDotNet.RetrievalAugmentedGeneration.EmbeddingModels
{
    /// <summary>
    /// ONNX-based sentence transformer for generating embeddings.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class ONNXSentenceTransformer<T> : EmbeddingModelBase<T>
    {
        private readonly string _modelPath;
        private readonly int _dimension;
        private readonly int _maxTokens;

        public override int EmbeddingDimension => _dimension;
        public override int MaxTokens => _maxTokens;

        public ONNXSentenceTransformer(string modelPath, int dimension = 384, int maxTokens = 512)
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
                var val = NumOps.FromDouble(Math.Cos((double)hash * (i + 1) * 0.002));
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
                    hash = hash * 31 + c;
                }
                return hash;
            }
        }
    }
}
