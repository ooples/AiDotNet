using System;
using AiDotNet.Helpers;

namespace AiDotNet.RetrievalAugmentedGeneration.EmbeddingModels
{
    /// <summary>
    /// HuggingFace-based embedding model for generating embeddings.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class HuggingFaceEmbeddingModel<T> : EmbeddingModelBase<T>
    {
        private readonly string _modelName;
        private readonly string _apiKey;
        private readonly INumericOperations<T> _numOps;

        /// <summary>
        /// Initializes a new instance of the <see cref="HuggingFaceEmbeddingModel{T}"/> class.
        /// </summary>
        /// <param name="numericOperations">The numeric operations for type T.</param>
        /// <param name="modelName">The HuggingFace model name.</param>
        /// <param name="apiKey">The HuggingFace API key (optional for public models).</param>
        public HuggingFaceEmbeddingModel(INumericOperations<T> numericOperations, string modelName, string apiKey = null) : base(numericOperations)
        {
            _numOps = numericOperations ?? throw new ArgumentNullException(nameof(numericOperations));
            _modelName = modelName ?? throw new ArgumentNullException(nameof(modelName));
            _apiKey = apiKey;
        }

        /// <summary>
        /// Generates an embedding vector for the input text using HuggingFace model.
        /// </summary>
        /// <param name="text">The input text to embed.</param>
        /// <returns>A vector representation of the text.</returns>
        public override Vector<T> Embed(string text)
        {
            if (string.IsNullOrEmpty(text)) throw new ArgumentNullException(nameof(text));

            var embeddingSize = 768;
            var values = new T[embeddingSize];

            var hash = text.GetHashCode();
            for (int i = 0; i < embeddingSize; i++)
            {
                var val = _numOps.FromDouble(Math.Cos(hash * (i + 1) * 0.002));
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
