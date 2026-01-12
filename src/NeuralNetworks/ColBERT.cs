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
    /// ColBERT (Contextualized Late Interaction over BERT) model implementation.
    /// Uses token-level representations for late interaction retrieval.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class ColBERT<T> : TransformerEmbeddingNetwork<T>
    {
        #region Fields

        private readonly int _outputDim; // Late interaction dimension (typically 128)

        #endregion

        #region Constructors

        public ColBERT(
            NeuralNetworkArchitecture<T> architecture,
            ITokenizer? tokenizer = null,
            IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
            int vocabSize = 30522,
            int outputDimension = 128,
            int maxSequenceLength = 512,
            int numLayers = 12,
            int numHeads = 12,
            int feedForwardDim = 3072,
            ILossFunction<T>? lossFunction = null,
            double maxGradNorm = 1.0)
            : base(architecture, tokenizer, optimizer, vocabSize, 768, maxSequenceLength, numLayers, numHeads, feedForwardDim, PoolingStrategy.Mean, lossFunction, maxGradNorm)
        {
            _outputDim = outputDimension;
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
                Layers.AddRange(LayerHelper<T>.CreateDefaultColBERTLayers(
                    Architecture,
                    30522,
                    _outputDim,
                    MaxTokens,
                    12,
                    12,
                    3072));
            }
        }

        #endregion

        #region Methods

        /// <summary>
        /// Encodes text into a multi-vector late interaction representation.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <returns>A matrix where each row is a token-level contextualized embedding.</returns>
        public Matrix<T> EmbedLateInteraction(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                return new Matrix<T>(0, _outputDim);

            // 1. Tokenize
            var tokenizer = Tokenization.LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.OPT);
            var tokenResult = tokenizer.Encode(text);
            var tokens = tokenResult.TokenIds.Take(MaxTokens).ToList();
            if (tokens.Count == 0) tokens.Add(0);

            // 2. Forward pass through full model
            var inputTensor = Tensor<T>.FromVector(new Vector<T>(tokens.Select(id => NumOps.FromDouble(id)).ToArray()), [1, tokens.Count]);
            var output = Predict(inputTensor); // [1, seqLen, outputDim]

            // 3. Extract matrix
            int seqLen = output.Shape[1];
            var result = new Matrix<T>(seqLen, _outputDim);
            for (int s = 0; s < seqLen; s++)
            {
                for (int d = 0; d < _outputDim; d++)
                {
                    result[s, d] = output[0, s, d];
                }
            }

            // ColBERT typically L2 normalizes token embeddings
            for (int s = 0; s < seqLen; s++)
            {
                var row = result.GetRow(s);
                result.SetRow(s, row.Normalize());
            }

            return result;
        }

        public override Vector<T> Embed(string text)
        {
            // Standard embedding model Embed returns a single vector.
            // For ColBERT, we return the mean of token embeddings as a fallback summary.
            var matrix = EmbedLateInteraction(text);
            if (matrix.Rows == 0) return new Vector<T>(_outputDim);

            var result = new Vector<T>(_outputDim);
            for (int d = 0; d < _outputDim; d++)
            {
                T sum = NumOps.Zero;
                for (int s = 0; s < matrix.Rows; s++)
                {
                    sum = NumOps.Add(sum, matrix[s, d]);
                }
                result[d] = NumOps.Divide(sum, NumOps.FromDouble(matrix.Rows));
            }

            return result.Normalize();
        }

        /// <summary>
        /// Computes the MaxSim (Late Interaction) similarity score between a query and a document.
        /// Formula: score = Σ_q max_d (cos_sim(E_q, E_d))
        /// </summary>
        public T LateInteractionScore(Matrix<T> queryEmbeddings, Matrix<T> docEmbeddings)
        {
            T totalScore = NumOps.Zero;

            for (int q = 0; q < queryEmbeddings.Rows; q++)
            {
                var queryVec = queryEmbeddings.GetRow(q);
                T maxSim = NumOps.FromDouble(double.MinValue);

                for (int d = 0; d < docEmbeddings.Rows; d++)
                {
                    var docVec = docEmbeddings.GetRow(d);
                    T sim = Engine.DotProduct(queryVec, docVec);
                    if (NumOps.GreaterThan(sim, maxSim))
                    {
                        maxSim = sim;
                    }
                }

                totalScore = NumOps.Add(totalScore, maxSim);
            }

            return totalScore;
        }

        protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        {
            return new ColBERT<T>(
                Architecture,
                null,
                null,
                30522,
                _outputDim,
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
            metadata.Name = "ColBERT";
            metadata.Description = "ColBERT late interaction embedding model";
            metadata.AdditionalInfo["OutputDimension"] = _outputDim;
            return metadata;
        }

        #endregion
    }
}
