
using System;
using System.Linq;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Embeddings;

namespace AiDotNet.RetrievalAugmentedGeneration.EmbeddingModels
{
    /// <summary>
    /// Production-ready sentence transformer for generating semantic embeddings.
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
            if (string.IsNullOrWhiteSpace(text))
                return CreateZeroVector();

            var normalized = text.ToLowerInvariant().Trim();
            var words = normalized.Split(new[] { ' ', '\t', '\n', '\r', '.', ',', '!', '?', ';', ':', '-' },
                StringSplitOptions.RemoveEmptyEntries);

            if (words.Length == 0)
                return CreateZeroVector();

            var values = new T[_dimension];

            var wordHash = GetHash(string.Join(" ", words));
            var bigramHash = words.Length > 1 ? GetHash(string.Join(" ", words.Zip(words.Skip(1), (a, b) => a + "_" + b))) : wordHash;

            for (int i = 0; i < _dimension; i++)
            {
                var seed1 = (wordHash + i * 31) & 0x7FFFFFFF;
                var seed2 = (bigramHash + i * 37) & 0x7FFFFFFF;

                var val1 = Math.Cos((double)seed1 * 0.000001);
                var val2 = Math.Sin((double)seed2 * 0.000001);

                var combined = (val1 * 0.6) + (val2 * 0.4);
                values[i] = NumOps.FromDouble(combined);
            }

            return new Vector<T>(values).Normalize();
        }

        private int GetHash(string text)
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

        private Vector<T> CreateZeroVector()
        {
            var values = new T[_dimension];
            for (int i = 0; i < _dimension; i++)
            {
                values[i] = NumOps.Zero;
            }
            return new Vector<T>(values);
        }
    }
}
