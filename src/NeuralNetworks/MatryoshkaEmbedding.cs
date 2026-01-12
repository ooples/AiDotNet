using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tokenization.Interfaces;

namespace AiDotNet.NeuralNetworks
{
    /// <summary>
    /// Matryoshka Representation Learning (MRL) model implementation.
    /// Learns nested embeddings where smaller prefixes of the full vector are valid representations.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class MatryoshkaEmbedding<T> : TransformerEmbeddingNetwork<T>
    {
        #region Fields

        private readonly int[] _nestedDimensions;

        #endregion

        #region Constructors

        public MatryoshkaEmbedding(
            NeuralNetworkArchitecture<T> architecture,
            ITokenizer? tokenizer = null,
            IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
            int vocabSize = 30522,
            int maxEmbeddingDimension = 1536,
            int[]? nestedDimensions = null,
            int maxSequenceLength = 512,
            int numLayers = 12,
            int numHeads = 12,
            int feedForwardDim = 3072,
            PoolingStrategy poolingStrategy = PoolingStrategy.ClsToken,
            ILossFunction<T>? lossFunction = null,
            double maxGradNorm = 1.0)
            : base(architecture, tokenizer, optimizer, vocabSize, maxEmbeddingDimension, maxSequenceLength, numLayers, numHeads, feedForwardDim, poolingStrategy, lossFunction, maxGradNorm)
        {
            _nestedDimensions = nestedDimensions ?? new[] { 64, 128, 256, 512, 768, 1024, 1536 };
        }

        #endregion

        #region Initialization

        protected override void InitializeLayers()
        {
            if (Architecture.Layers != null && Architecture.Layers.Count > 0)
            {
                Layers.AddRange(Architecture.Layers);
                ValidateCustomLayers(Layers);
            }
            else
            {
                Layers.AddRange(LayerHelper<T>.CreateDefaultMRLLayers(
                    Architecture,
                    30522,
                    EmbeddingDimension,
                    MaxTokens,
                    12,
                    12,
                    3072));
            }
        }

        #endregion

        #region Methods

        /// <summary>
        /// Gets a truncated and re-normalized embedding of the specified dimension.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="dimension">The target dimension (must be less than or equal to EmbeddingDimension).</param>
        /// <returns>A normalized embedding vector of the requested dimension.</returns>
        public Vector<T> EmbedResized(string text, int dimension)
        {
            if (dimension > EmbeddingDimension)
                throw new ArgumentException($"Requested dimension {dimension} exceeds max dimension {EmbeddingDimension}");

            var fullEmbedding = Embed(text);
            var resizedValues = new T[dimension];
            for (int i = 0; i < dimension; i++)
            {
                resizedValues[i] = fullEmbedding[i];
            }

            return new Vector<T>(resizedValues).Normalize();
        }

        protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        {
            return new MatryoshkaEmbedding<T>(
                Architecture,
                null,
                null,
                30522,
                EmbeddingDimension,
                _nestedDimensions,
                MaxTokens,
                12,
                12,
                3072,
                PoolingStrategy.ClsToken,
                LossFunction,
                Convert.ToDouble(MaxGradNorm));
        }

        public override ModelMetadata<T> GetModelMetadata()
        {
            var metadata = base.GetModelMetadata();
            metadata.Name = "MatryoshkaEmbedding";
            metadata.Description = "Matryoshka Representation Learning (nested embeddings) model";
            metadata.AdditionalInfo["NestedDimensions"] = _nestedDimensions;
            return metadata;
        }

        #endregion
    }
}
