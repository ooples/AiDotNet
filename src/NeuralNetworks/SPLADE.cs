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
    /// SPLADE (Sparse Lexical and Expansion Model) embedding model implementation.
    /// Maps text to a sparse vector in the vocabulary space using max-pooling over token expansions.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class SPLADE<T> : TransformerEmbeddingNetwork<T>
    {
        #region Fields

        private readonly int _vocabSize;

        #endregion

        #region Constructors

        public SPLADE(
            NeuralNetworkArchitecture<T> architecture,
            ITokenizer? tokenizer = null,
            IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
            int vocabSize = 30522,
            int embeddingDimension = 768,
            int maxSequenceLength = 512,
            int numLayers = 12,
            int numHeads = 12,
            int feedForwardDim = 3072,
            ILossFunction<T>? lossFunction = null,
            double maxGradNorm = 1.0)
            : base(architecture, tokenizer, optimizer, vocabSize, embeddingDimension, maxSequenceLength, numLayers, numHeads, feedForwardDim, PoolingStrategy.Max, lossFunction, maxGradNorm)
        {
            _vocabSize = vocabSize;
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
                Layers.AddRange(LayerHelper<T>.CreateDefaultSPLADELayers(
                    Architecture,
                    _vocabSize,
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
        /// Encodes text into a sparse lexical representation.
        /// Result is a vector of size vocab_size where most elements are zero.
        /// </summary>
        public override Vector<T> Embed(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                return new Vector<T>(_vocabSize);

            // 1. Tokenize
            var tokenizer = Tokenization.LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.OPT);
            var tokenResult = tokenizer.Encode(text);
            var tokens = tokenResult.TokenIds.Take(MaxTokens).ToList();
            if (tokens.Count == 0) tokens.Add(0);

            // 2. Forward pass through full model
            var inputTensor = Tensor<T>.FromVector(new Vector<T>(tokens.Select(id => NumOps.FromDouble(id)).ToArray()), [1, tokens.Count]);
            
            // For SPLADE, the last layer is a DenseLayer projecting to vocabSize with ReLU
            var tokenExpansions = Predict(inputTensor); // [1, seqLen, vocabSize]

            // 3. SPLADE Pooling: log(1 + max_over_seq(ReLU(output)))
            var sparseVector = new Vector<T>(_vocabSize);
            for (int v = 0; v < _vocabSize; v++)
            {
                T maxVal = NumOps.Zero;
                for (int s = 0; s < tokenExpansions.Shape[1]; s++)
                {
                    T val = tokenExpansions[0, s, v];
                    if (NumOps.GreaterThan(val, maxVal))
                    {
                        maxVal = val;
                    }
                }

                // Apply log(1 + x) for sparsity and weight scaling
                sparseVector[v] = NumOps.FromDouble(Math.Log(1.0 + NumOps.ToDouble(maxVal)));
            }

            return sparseVector;
        }

        protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        {
            return new SPLADE<T>(
                Architecture,
                null,
                null,
                _vocabSize,
                EmbeddingDimension,
                MaxTokens,
                12,
                12,
                3072,
                LossFunction,
                Convert.ToDouble(MaxGradNorm));
        }

        public override ModelMetadata<T> GetModelMetadata()
        {
            var metadata = base.GetModelMetadata();
            metadata.Name = "SPLADE";
            metadata.Description = "SPLADE (Sparse Lexical and Expansion Model) embedding model";
            metadata.AdditionalInfo["VocabSize"] = _vocabSize;
            return metadata;
        }

        #endregion
    }
}
