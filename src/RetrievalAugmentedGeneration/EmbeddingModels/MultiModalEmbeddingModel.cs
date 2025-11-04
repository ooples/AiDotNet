using AiDotNet.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AiDotNet.RetrievalAugmentedGeneration.EmbeddingModels
{
    /// <summary>
    /// Multi-modal embedding model that combines text and other modalities
    /// </summary>
    /// <typeparam name="T">The numeric type for embeddings</typeparam>
    public class MultiModalEmbeddingModel<T> : EmbeddingModelBase<T> where T : struct, IComparable, IComparable<T>, IConvertible, IEquatable<T>, IFormattable
    {
        private readonly IEmbeddingModel<T> _textEmbedder;
        private readonly Dictionary<string, Func<object, Task<Vector<T>>>> _modalityEmbedders;
        private readonly Func<List<Vector<T>>, Vector<T>> _fusionStrategy;

        public MultiModalEmbeddingModel(
            IEmbeddingModel<T> textEmbedder,
            Func<List<Vector<T>>, Vector<T>>? fusionStrategy = null,
            INormalizer<T>? normalizer = null)
            : base(normalizer)
        {
            _textEmbedder = textEmbedder ?? throw new ArgumentNullException(nameof(textEmbedder));
            _modalityEmbedders = new Dictionary<string, Func<object, Task<Vector<T>>>>();
            _fusionStrategy = fusionStrategy ?? DefaultFusion;
        }

        public void RegisterModalityEmbedder(string modalityType, Func<object, Task<Vector<T>>> embedder)
        {
            if (string.IsNullOrEmpty(modalityType))
                throw new ArgumentException("Modality type cannot be null or empty", nameof(modalityType));
            if (embedder == null)
                throw new ArgumentNullException(nameof(embedder));

            _modalityEmbedders[modalityType] = embedder;
        }

        protected override async Task<Vector<T>> GenerateEmbeddingCoreAsync(string text)
        {
            var textEmbedding = await _textEmbedder.GenerateEmbeddingAsync(text);
            var vector = Normalizer?.Normalize(textEmbedding) ?? textEmbedding;
            return vector;
        }

        public async Task<Vector<T>> GenerateMultiModalEmbeddingAsync(Dictionary<string, object> modalityData)
        {
            if (modalityData == null || modalityData.Count == 0)
                throw new ArgumentException("Modality data cannot be null or empty", nameof(modalityData));

            var embeddings = new List<Vector<T>>();

            foreach (var (modalityType, data) in modalityData)
            {
                if (_modalityEmbedders.TryGetValue(modalityType, out var embedder))
                {
                    var embedding = await embedder(data);
                    embeddings.Add(embedding);
                }
                else if (modalityType == "text" && data is string textData)
                {
                    var embedding = await GenerateEmbeddingCoreAsync(textData);
                    embeddings.Add(embedding);
                }
            }

            if (embeddings.Count == 0)
                throw new InvalidOperationException("No embeddings were generated from the provided modalities");

            var fusedEmbedding = _fusionStrategy(embeddings);
            return Normalizer?.Normalize(fusedEmbedding) ?? fusedEmbedding;
        }

        private Vector<T> DefaultFusion(List<Vector<T>> embeddings)
        {
            if (embeddings.Count == 1)
                return embeddings[0];

            var dimension = embeddings[0].Length;
            var result = new T[dimension];

            for (int i = 0; i < dimension; i++)
            {
                var sum = NumOps.Zero;
                foreach (var embedding in embeddings)
                {
                    sum = NumOps.Add(sum, embedding[i]);
                }
                result[i] = NumOps.Divide(sum, NumOps.FromInt32(embeddings.Count));
            }

            return new Vector<T>(result, NumOps);
        }
    }
}
