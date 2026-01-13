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
    /// ColBERT (Contextualized Late Interaction over BERT) neural network implementation.
    /// Uses token-level representations for high-precision document retrieval.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
    /// <remarks>
    /// <para>
    /// ColBERT is a highly efficient and accurate retrieval model that keeps a separate vector for every 
    /// token in a sentence. It calculates the similarity between a query and a document using a 
    /// "Late Interaction" MaxSim operator, allowing it to capture fine-grained semantic matches.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Most AI search models are like people who read a whole book and then 
    /// try to summarize it in just one word. ColBERT is like a person who keeps detailed notes 
    /// on every single word. When you ask a question, ColBERT compares every word in your question 
    /// to every word in the document notes. This is much more accurate because no information 
    /// is "lost" during summarization.
    /// </para>
    /// </remarks>
    public class ColBERT<T> : TransformerEmbeddingNetwork<T>
    {
        #region Fields

        /// <summary>
        /// The dimension of each token-level embedding vector (typically 128).
        /// </summary>
        /// <remarks>
        /// <b>For Beginners:</b> This is the size of the "fingerprint" the model creates for 
        /// every single word. Because we keep many fingerprints per sentence, we make them 
        /// smaller (like 128 instead of 768) to save space while still being very precise.
        /// </remarks>
        private int _outputDim;

        private int _vocabSize;
        private int _numLayers;
        private int _numHeads;
        private int _feedForwardDim;
        private readonly ITokenizer _tokenizer;

        #endregion

        #region Constructors

        /// <summary>
        /// Initializes a new instance of the ColBERT model.
        /// </summary>
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
            _vocabSize = vocabSize;
            _numLayers = numLayers;
            _numHeads = numHeads;
            _feedForwardDim = feedForwardDim;
            _tokenizer = tokenizer ?? Tokenization.LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.OPT);

            InitializeLayersCore(false);
        }

        #endregion

        #region Initialization

        /// <summary>
        /// Sets up the transformer layers and the token-level projection head for ColBERT.
        /// </summary>
        protected override void InitializeLayers()
        {
            InitializeLayersCore(true);
        }

        private void InitializeLayersCore(bool useVirtualValidation)
        {
            if (Architecture.Layers != null && Architecture.Layers.Count > 0)
            {
                Layers.AddRange(Architecture.Layers);
                if (useVirtualValidation)
                {
                    ValidateCustomLayers(Layers);
                }
                else
                {
                    ValidateCustomLayersInternal(Layers);
                }
            }
            else
            {
                Layers.AddRange(LayerHelper<T>.CreateDefaultColBERTLayers(
                    Architecture,
                    _vocabSize,
                    _outputDim,
                    MaxTokens,
                    _numLayers,
                    _numHeads,
                    _feedForwardDim));
            }
        }

        #endregion

        #region Methods

        /// <summary>
        /// Encodes text into a multi-vector matrix where each row is a contextualized token embedding.
        /// </summary>
        public Matrix<T> EmbedLateInteraction(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                return new Matrix<T>(0, _outputDim);

            var tokenResult = _tokenizer.Encode(text);
            var tokens = tokenResult.TokenIds.Take(MaxTokens).ToList();
            if (tokens.Count == 0) tokens.Add(0);

            var inputTensor = Tensor<T>.FromVector(new Vector<T>(tokens.Select(id => NumOps.FromDouble(id)).ToArray()), [1, tokens.Count]);
            var output = Predict(inputTensor); // [1, seqLen, outputDim]

            int seqLen = output.Shape[1];
            var result = new Matrix<T>(seqLen, _outputDim);
            for (int s = 0; s < seqLen; s++)
            {
                for (int d = 0; d < _outputDim; d++)
                {
                    result[s, d] = output[0, s, d];
                }
            }

            // Normalization is critical for the MaxSim dot-product scoring
            for (int s = 0; s < seqLen; s++)
            {
                var row = result.GetRow(s);
                result.SetRow(s, row.SafeNormalize());
            }

            return result;
        }

        /// <summary>
        /// Fallback method that encodes a sentence into a single summary vector (mean-pooled).
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <returns>A summary vector for the input text.</returns>
        /// <remarks>
        /// <b>For Beginners:</b> This is a fallback option. While ColBERT works best when it keeps 
        /// all its notes (as a table), sometimes you just want one summary list of numbers. 
        /// This method averages all the word-level info into one overall summary.
        /// </remarks>
        public override Vector<T> Embed(string text)
        {
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

            return result.SafeNormalize();
        }

        /// <summary>
        /// Computes the similarity score between a query and document matrix using the MaxSim interaction.
        /// </summary>
        /// <param name="queryEmbeddings">The token-level embeddings for the query.</param>
        /// <param name="docEmbeddings">The token-level embeddings for the document.</param>
        /// <returns>A scalar interaction score.</returns>
        /// <remarks>
        /// <b>For Beginners:</b> This is how ColBERT compares a question to a document. It looks 
        /// at every word in your question and finds the absolute "best match" for it in the 
        /// entire document. It then combines all those best matches into one final score.
        /// </remarks>
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

        /// <inheritdoc/>
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

        /// <summary>
        /// Retrieves metadata about the ColBERT model.
        /// </summary>
        /// <returns>Metadata containing model type and output dimensionality information.</returns>
        public override ModelMetadata<T> GetModelMetadata()
        {
            var metadata = base.GetModelMetadata();
            metadata.Name = "ColBERT";
            metadata.Description = "ColBERT late interaction multi-vector embedding model";
            metadata.AdditionalInfo["OutputDimension"] = _outputDim;
            return metadata;
        }

        /// <inheritdoc/>
        protected override void SerializeNetworkSpecificData(BinaryWriter writer)
        {
            base.SerializeNetworkSpecificData(writer);
            writer.Write(_outputDim);
        }

        /// <inheritdoc/>
        protected override void DeserializeNetworkSpecificData(BinaryReader reader)
        {
            base.DeserializeNetworkSpecificData(reader);
            _outputDim = reader.ReadInt32();
        }

        #endregion
    }
}
