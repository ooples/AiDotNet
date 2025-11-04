using AiDotNet.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AiDotNet.RetrievalAugmentedGeneration.EmbeddingModels
{
    /// <summary>
    /// Fine-tuning wrapper for sentence transformer models
    /// </summary>
    /// <typeparam name="T">The numeric type for embeddings</typeparam>
    public class SentenceTransformersFineTuner<T> : EmbeddingModelBase<T> where T : struct, IComparable, IComparable<T>, IConvertible, IEquatable<T>, IFormattable
    {
        private readonly IEmbeddingModel<T> _baseModel;
        private readonly Dictionary<string, Vector<T>> _finetuneCache;
        private bool _isFineTuned;

        public SentenceTransformersFineTuner(IEmbeddingModel<T> baseModel, INormalizer<T>? normalizer = null)
            : base(normalizer)
        {
            _baseModel = baseModel ?? throw new ArgumentNullException(nameof(baseModel));
            _finetuneCache = new Dictionary<string, Vector<T>>();
            _isFineTuned = false;
        }

        protected override async Task<Vector<T>> GenerateEmbeddingCoreAsync(string text)
        {
            if (_finetuneCache.TryGetValue(text, out var cachedEmbedding))
            {
                return Normalizer?.Normalize(cachedEmbedding) ?? cachedEmbedding;
            }

            var embedding = await _baseModel.GenerateEmbeddingAsync(text);
            
            if (_isFineTuned)
            {
                _finetuneCache[text] = embedding;
            }

            return Normalizer?.Normalize(embedding) ?? embedding;
        }

        public async Task FineTuneAsync(
            List<(string positive, string negative)> contrastivePairs,
            int epochs = 3,
            T learningRate = default)
        {
            if (contrastivePairs == null || contrastivePairs.Count == 0)
                throw new ArgumentException("Contrastive pairs cannot be null or empty", nameof(contrastivePairs));

            if (NumOps.Equals(learningRate, NumOps.Zero))
            {
                learningRate = NumOps.FromDouble(0.001);
            }

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                foreach (var (positive, negative) in contrastivePairs)
                {
                    var positiveEmbedding = await _baseModel.GenerateEmbeddingAsync(positive);
                    var negativeEmbedding = await _baseModel.GenerateEmbeddingAsync(negative);

                    var adjustedPositive = AdjustEmbedding(positiveEmbedding, learningRate, isPositive: true);
                    var adjustedNegative = AdjustEmbedding(negativeEmbedding, learningRate, isPositive: false);

                    _finetuneCache[positive] = adjustedPositive;
                    _finetuneCache[negative] = adjustedNegative;
                }
            }

            _isFineTuned = true;
        }

        private Vector<T> AdjustEmbedding(Vector<T> embedding, T learningRate, bool isPositive)
        {
            var adjustment = isPositive ? learningRate : NumOps.Negate(learningRate);
            var adjustedValues = new T[embedding.Length];

            for (int i = 0; i < embedding.Length; i++)
            {
                var delta = NumOps.Multiply(embedding[i], adjustment);
                adjustedValues[i] = NumOps.Add(embedding[i], delta);
            }

            return new Vector<T>(adjustedValues, NumOps);
        }

        public void ClearFineTuneCache()
        {
            _finetuneCache.Clear();
            _isFineTuned = false;
        }

        public int GetCacheSize() => _finetuneCache.Count;

        public bool IsFineTuned => _isFineTuned;
    }
}
