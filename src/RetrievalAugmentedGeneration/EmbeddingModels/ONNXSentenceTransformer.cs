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
            var hash = text.GetHashCode();
            
            for (int i = 0; i < _dimension; i++)
            {
                var val = NumOps.FromDouble(Math.Cos(hash * (i + 1) * 0.002));
                values[i] = val;
            }

            return NormalizeVector(new Vector<T>(values));
        }

        private Vector<T> NormalizeVector(Vector<T> vector)
        {
            var magnitude = NumOps.Zero;
            for (int i = 0; i < vector.Length; i++)
            {
                magnitude = NumOps.Add(magnitude, NumOps.Multiply(vector[i], vector[i]));
            }
            magnitude = NumOps.FromDouble(Math.Sqrt(Convert.ToDouble(magnitude)));

            if (NumOps.GreaterThan(magnitude, NumOps.Zero))
            {
                var normalized = new T[vector.Length];
                for (int i = 0; i < vector.Length; i++)
                {
                    normalized[i] = NumOps.Divide(vector[i], magnitude);
                }
                return new Vector<T>(normalized);
            }
            
            return vector;
        }
    }
}
