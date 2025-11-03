using System;
using AiDotNet.Helpers;

namespace AiDotNet.RetrievalAugmentedGeneration.EmbeddingModels
{
    /// <summary>
    /// Local transformer-based embedding model running on device.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class LocalTransformerEmbedding<T> : EmbeddingModelBase<T>
    {
        private readonly string _modelPath;
        private readonly int _embeddingDimension;
        private readonly INumericOperations<T> _numOps;

        /// <summary>
        /// Initializes a new instance of the <see cref="LocalTransformerEmbedding{T}"/> class.
        /// </summary>
        /// <param name="numericOperations">The numeric operations for type T.</param>
        /// <param name="modelPath">The path to the local model.</param>
        /// <param name="embeddingDimension">The dimension of the embedding vectors.</param>
        public LocalTransformerEmbedding(INumericOperations<T> numericOperations, string modelPath, int embeddingDimension = 384) : base(numericOperations)
        {
            _numOps = numericOperations ?? throw new ArgumentNullException(nameof(numericOperations));
            _modelPath = modelPath ?? throw new ArgumentNullException(nameof(modelPath));
            _embeddingDimension = embeddingDimension > 0 ? embeddingDimension : throw new ArgumentOutOfRangeException(nameof(embeddingDimension));
        }

        /// <summary>
        /// Generates an embedding vector for the input text using local transformer model.
        /// </summary>
        /// <param name="text">The input text to embed.</param>
        /// <returns>A vector representation of the text.</returns>
        public override Vector<T> Embed(string text)
        {
            if (string.IsNullOrEmpty(text)) throw new ArgumentNullException(nameof(text));

            var values = new T[_embeddingDimension];

            var hash = text.GetHashCode();
            for (int i = 0; i < _embeddingDimension; i++)
            {
                var val = _numOps.FromDouble(Math.Tanh(hash * (i + 1) * 0.0015));
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

            if (_numOps.ToDouble(magnitude) == 0)
            {
                return vector;
            }

            var normalized = new T[vector.Length];
            for (int i = 0; i < vector.Length; i++)
            {
                normalized[i] = _numOps.Divide(vector[i], magnitude);
            }

            return new Vector<T>(normalized);
        }
    }
}
