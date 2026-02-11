using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Embeddings;
using AiDotNet.Tokenization.HuggingFace;
using AiDotNet.Tokenization.Interfaces;
using Microsoft.ML.OnnxRuntime;

namespace AiDotNet.RetrievalAugmentedGeneration.EmbeddingModels
{
    /// <summary>
    /// Production-ready sentence transformer for generating semantic embeddings using ONNX Runtime.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class ONNXSentenceTransformer<T> : EmbeddingModelBase<T>
    {
        private readonly string _modelPath;
        private readonly int _dimension;
        private readonly int _maxTokens;
        private InferenceSession? _session;
        private ITokenizer? _tokenizer;
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
        }

        /// <summary>
        /// Ensures the ONNX model and tokenizer are loaded, loading lazily on first use.
        /// </summary>
        private void EnsureModelLoaded()
        {
            if (_session == null)
            {
                if (!File.Exists(_modelPath))
                {
                    throw new FileNotFoundException($"ONNX model file not found: {_modelPath}", _modelPath);
                }

                _session = new InferenceSession(_modelPath);

                var modelDir = Path.GetDirectoryName(_modelPath) ?? ".";
                _tokenizer = AutoTokenizer.FromPretrained(modelDir);
            }
        }

        /// <summary>
        /// Gets the inference session, ensuring it's loaded.
        /// </summary>
        private InferenceSession Session
        {
            get
            {
                EnsureModelLoaded();
                return _session ?? throw new InvalidOperationException("Failed to load ONNX session.");
            }
        }

        /// <summary>
        /// Gets the tokenizer, ensuring it's loaded.
        /// </summary>
        private ITokenizer Tokenizer
        {
            get
            {
                EnsureModelLoaded();
                return _tokenizer ?? throw new InvalidOperationException("Failed to load tokenizer.");
            }
        }

        protected override Vector<T> EmbedCore(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                return CreateZeroVector();

            if (!File.Exists(_modelPath))
            {
                throw new FileNotFoundException(
                    $"ONNX model file not found: '{_modelPath}'. " +
                    "Download the model file or provide a valid path.",
                    _modelPath);
            }

            // 1. Tokenize
            var tokenizationResult = Tokenizer.Encode(text, new AiDotNet.Tokenization.Models.EncodingOptions
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
            if (Session.InputMetadata.ContainsKey("token_type_ids"))
            {
                inputs.Add(NamedOnnxValue.CreateFromTensor("token_type_ids", new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<long>(tokenTypeIds, inputShape)));
            }

            // 3. Run inference
            using var results = Session.Run(inputs);

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
            int seqLength = lastHiddenState.Dimensions[1];
            int hiddenDim = lastHiddenState.Dimensions[2];
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

        /// <summary>
        /// Generates a deterministic fallback embedding based on the text hash.
        /// Used when the ONNX model file is not available (e.g., in unit tests).
        /// </summary>
        private Vector<T> GenerateFallbackEmbedding(string text)
        {
            var hash = text.ToLowerInvariant().GetHashCode();
            var values = new T[_dimension];
            for (int i = 0; i < _dimension; i++)
            {
                // Generate deterministic values based on text hash and position
                double val = Math.Sin(hash * 0.0001 + i * 0.1) * 0.5;
                values[i] = NumOps.FromDouble(val);
            }
            return new Vector<T>(values).Normalize();
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

        protected override void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    _session?.Dispose();
                    if (_tokenizer is IDisposable disposableTokenizer)
                    {
                        disposableTokenizer.Dispose();
                    }
                }

                _disposed = true;
            }

            base.Dispose(disposing);
        }
    }
}



