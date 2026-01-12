using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Embeddings;
using AiDotNet.Tokenization.HuggingFace;
using AiDotNet.Tokenization.Interfaces;

namespace AiDotNet.RetrievalAugmentedGeneration.EmbeddingModels
{
    /// <summary>
    /// Production-ready sentence transformer for generating semantic embeddings using ONNX Runtime.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class ONNXSentenceTransformer<T> : EmbeddingModelBase<T>, IDisposable
    {
        private readonly string _modelPath;
        private readonly int _dimension;
        private readonly int _maxTokens;
        private readonly InferenceSession _session;
        private readonly ITokenizer _tokenizer;
        private bool _disposed;

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

            // Initialize ONNX Runtime session
            _session = new InferenceSession(_modelPath);

            // Initialize tokenizer - assuming tokenizer files are in the same directory as the model
            var modelDir = Path.GetDirectoryName(_modelPath) ?? ".";
            _tokenizer = AutoTokenizer.FromPretrained(modelDir);
        }

        protected override Vector<T> EmbedCore(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                return CreateZeroVector();

            // 1. Tokenize
            var tokenizationResult = _tokenizer.Encode(text, new AiDotNet.Tokenization.Models.EncodingOptions
            {
                MaxLength = _maxTokens,
                Truncation = true,
                Padding = true
            });

            var inputIds = tokenizationResult.TokenIds.Select(id => (long)id).ToArray();
            var attentionMask = tokenizationResult.AttentionMask.Select(m => (long)m).ToArray();
            var tokenTypeIds = tokenizationResult.TokenTypeIds.Select(t => (long)t).ToArray();

            var seqLength = inputIds.Length;
            var inputShape = new[] { 1, seqLength };

            // 2. Prepare inputs
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids", new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<long>(inputIds, inputShape)),
                NamedOnnxValue.CreateFromTensor("attention_mask", new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<long>(attentionMask, inputShape))
            };

            // Some models require token_type_ids
            if (_session.InputMetadata.ContainsKey("token_type_ids"))
            {
                inputs.Add(NamedOnnxValue.CreateFromTensor("token_type_ids", new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<long>(tokenTypeIds, inputShape)));
            }

            // 3. Run inference
            using var results = _session.Run(inputs);
            
            // 4. Process output (last_hidden_state is typical for sentence-transformers)
            var lastHiddenState = results.First(r => r.Name == "last_hidden_state").AsTensor<float>();
            
            // 5. Mean Pooling
            var embedding = ApplyMeanPooling(lastHiddenState, attentionMask);

            // 6. Convert to generic type T and normalize
            var values = new T[_dimension];
            for (int i = 0; i < Math.Min(_dimension, embedding.Length); i++)
            {
                values[i] = NumOps.FromDouble(embedding[i]);
            }

            return new Vector<T>(values).Normalize();
        }

        private float[] ApplyMeanPooling(Microsoft.ML.OnnxRuntime.Tensors.Tensor<float> lastHiddenState, long[] attentionMask)
        {
            int seqLength = (int)lastHiddenState.Dimensions[1];
            int hiddenDim = (int)lastHiddenState.Dimensions[2];
            var pooled = new float[hiddenDim];
            float sumMask = 0;

            for (int i = 0; i < seqLength; i++)
            {
                if (attentionMask[i] == 0) continue;

                sumMask += 1;
                for (int d = 0; d < hiddenDim; d++)
                {
                    pooled[d] += lastHiddenState[0, i, d];
                }
            }

            if (sumMask > 0)
            {
                for (int d = 0; d < hiddenDim; d++)
                {
                    pooled[d] /= sumMask;
                }
            }

            return pooled;
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

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    _session.Dispose();
                }
                _disposed = true;
            }
        }
    }
}