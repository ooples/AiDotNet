using System;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Embeddings;

namespace AiDotNet.RetrievalAugmentedGeneration.EmbeddingModels
{
    /// <summary>
    /// Local transformer embedding model for generating embeddings using ONNX Runtime without external API calls.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class LocalTransformerEmbedding<T> : EmbeddingModelBase<T>, IDisposable
    {
        private readonly ONNXSentenceTransformer<T> _onnxTransformer;

        public override int EmbeddingDimension => _onnxTransformer.EmbeddingDimension;
        public override int MaxTokens => _onnxTransformer.MaxTokens;

        public LocalTransformerEmbedding(string modelPath, int dimension = 384, int maxTokens = 512)
        {
            _onnxTransformer = new ONNXSentenceTransformer<T>(modelPath, dimension, maxTokens);
        }

        protected override Vector<T> EmbedCore(string text)
        {
            return _onnxTransformer.Embed(text);
        }

        public void Dispose()
        {
            _onnxTransformer.Dispose();
        }
    }
}
