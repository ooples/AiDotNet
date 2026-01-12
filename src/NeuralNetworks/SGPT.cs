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
    /// SGPT (Sentence GPT) embedding model implementation.
    /// Uses decoder-only transformer architectures for generating sentence embeddings.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class SGPT<T> : TransformerEmbeddingNetwork<T>
    {
        #region Constructors

        public SGPT(
            NeuralNetworkArchitecture<T> architecture,
            ITokenizer? tokenizer = null,
            IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
            int vocabSize = 50257,
            int embeddingDimension = 768,
            int maxSequenceLength = 1024,
            int numLayers = 12,
            int numHeads = 12,
            int feedForwardDim = 3072,
            PoolingStrategy poolingStrategy = PoolingStrategy.Mean,
            ILossFunction<T>? lossFunction = null,
            double maxGradNorm = 1.0)
            : base(architecture, tokenizer, optimizer, vocabSize, embeddingDimension, maxSequenceLength, numLayers, numHeads, feedForwardDim, poolingStrategy, lossFunction, maxGradNorm)
        {
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
                Layers.AddRange(LayerHelper<T>.CreateDefaultSGPTLayers(
                    Architecture,
                    50257,
                    EmbeddingDimension,
                    MaxTokens,
                    12,
                    12,
                    3072));
            }
        }

        #endregion

        #region Methods

        protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        {
            return new SGPT<T>(
                Architecture,
                null,
                null,
                50257,
                EmbeddingDimension,
                MaxTokens,
                12,
                12,
                3072,
                PoolingStrategy.Mean,
                LossFunction,
                Convert.ToDouble(MaxGradNorm));
        }

        public override ModelMetadata<T> GetModelMetadata()
        {
            var metadata = base.GetModelMetadata();
            metadata.Name = "SGPT";
            metadata.Description = "SGPT (Sentence GPT) decoder-based embedding model";
            return metadata;
        }

        #endregion
    }
}
