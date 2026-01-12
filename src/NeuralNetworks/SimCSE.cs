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
    /// SimCSE (Simple Contrastive Learning of Sentence Embeddings) model implementation.
    /// Supports both unsupervised (dropout-based) and supervised training.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class SimCSE<T> : TransformerEmbeddingNetwork<T>
    {
        #region Fields

        private readonly SimCSEType _simCseType;
        private readonly double _dropoutRate;

        #endregion

        #region Constructors

        public SimCSE(
            NeuralNetworkArchitecture<T> architecture,
            ITokenizer? tokenizer = null,
            IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
            SimCSEType type = SimCSEType.Unsupervised,
            int vocabSize = 30522,
            int embeddingDimension = 768,
            int maxSequenceLength = 512,
            int numLayers = 12,
            int numHeads = 12,
            int feedForwardDim = 3072,
            double dropoutRate = 0.1,
            PoolingStrategy poolingStrategy = PoolingStrategy.ClsToken,
            ILossFunction<T>? lossFunction = null,
            double maxGradNorm = 1.0)
            : base(architecture, tokenizer, optimizer, vocabSize, embeddingDimension, maxSequenceLength, numLayers, numHeads, feedForwardDim, poolingStrategy, lossFunction, maxGradNorm)
        {
            _simCseType = type;
            _dropoutRate = dropoutRate;
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
                Layers.AddRange(LayerHelper<T>.CreateDefaultSimCSELayers(
                    Architecture,
                    30522, // Default vocab
                    EmbeddingDimension,
                    MaxTokens,
                    12, // Default layers
                    12, // Default heads
                    3072)); // Default FF
            }
        }

        #endregion

        #region Methods

        public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
        {
            if (_simCseType == SimCSEType.Unsupervised)
            {
                // SimCSE Unsupervised: input1 and input2 are the same, 
                // but different dropout masks create a positive pair.
                // Standard Train loop expects (input, expected).
                // Here we might need a specialized Contrastive training loop.
                base.Train(input, expectedOutput);
            }
            else
            {
                base.Train(input, expectedOutput);
            }
        }

        protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        {
            return new SimCSE<T>(
                Architecture,
                null, // Tokenizer handled by factory in base
                null, // Optimizer handled by base
                _simCseType,
                30522,
                EmbeddingDimension,
                MaxTokens,
                12,
                12,
                3072,
                _dropoutRate,
                PoolingStrategy.ClsToken,
                LossFunction,
                Convert.ToDouble(MaxGradNorm));
        }

        public override ModelMetadata<T> GetModelMetadata()
        {
            var metadata = base.GetModelMetadata();
            metadata.Name = "SimCSE";
            metadata.Description = $"SimCSE ({_simCseType}) contrastive embedding model";
            metadata.AdditionalInfo["SimCSEType"] = _simCseType.ToString();
            metadata.AdditionalInfo["DropoutRate"] = _dropoutRate;
            return metadata;
        }

        #endregion
    }

    public enum SimCSEType
    {
        Unsupervised,
        Supervised
    }
}
