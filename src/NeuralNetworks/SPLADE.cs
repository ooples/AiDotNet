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
    /// SPLADE (Sparse Lexical and Expansion Model) neural network implementation.
    /// Maps text to a high-dimensional sparse vector in the vocabulary space.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
    /// <remarks>
    /// <para>
    /// SPLADE is a sparse retrieval model that learns to represent documents and queries as sparse 
    /// vectors over the vocabulary. It uses a log-saturation effect and sparsity regularization 
    /// (e.g., FLOPs or L1) to learn lexical expansion and term importance.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Imagine a dictionary with 30,000 words. For every sentence, SPLADE 
    /// creates a giant list of 30,000 numbers, but almost all of them are zero. It only puts 
    /// numbers next to the words that are actually important to the meaning. 
    /// </para>
    /// <para>
    /// The "Expansion" part is the most interesting: if you say "The smartphone is fast," SPLADE 
    /// might automatically put a number next to the word "Apple" or "Android" in its dictionary, 
    /// even if you didn't say them. This helps it find relevant documents that use different words.
    /// </para>
    /// </remarks>
    public class SPLADE<T> : TransformerEmbeddingNetwork<T>
    {
        #region Fields

        private int _vocabSize;
        private int _numLayers;
        private int _numHeads;
        private int _feedForwardDim;
        private readonly ITokenizer _tokenizer;

        #endregion

        #region Constructors

        /// <summary>
        /// Initializes a new instance of the SPLADE model.
        /// </summary>
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
            _numLayers = numLayers;
            _numHeads = numHeads;
            _feedForwardDim = feedForwardDim;
            _tokenizer = tokenizer ?? Tokenization.LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.OPT);

            InitializeLayers();
        }

        #endregion

        #region Initialization

        /// <summary>
        /// Configures the transformer backbone and the ReLU-based expansion head for SPLADE.
        /// </summary>
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
                    _numLayers,
                    _numHeads,
                    _feedForwardDim));
            }
        }

        #endregion

        #region Methods

        /// <summary>
        /// Encodes text into a high-dimensional sparse lexical representation.
        /// </summary>
        public override Vector<T> Embed(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                return new Vector<T>(_vocabSize);

            var tokenResult = _tokenizer.Encode(text);
            var tokens = tokenResult.TokenIds.Take(MaxTokens).ToList();
            if (tokens.Count == 0) tokens.Add(0);

            var inputTensor = Tensor<T>.FromVector(new Vector<T>(tokens.Select(id => NumOps.FromDouble(id)).ToArray()), [1, tokens.Count]);
            
            // For SPLADE, the last layer is a DenseLayer projecting to vocabSize with ReLU activation
            var tokenExpansions = Predict(inputTensor); // [1, seqLen, vocabSize]

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

                // Apply SPLADE log-saturation: log(1 + weight)
                sparseVector[v] = NumOps.FromDouble(Math.Log(1.0 + NumOps.ToDouble(maxVal)));
            }

            return sparseVector;
        }

        /// <inheritdoc/>
        protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        {
            return new SPLADE<T>(
                Architecture,
                null,
                null,
                _vocabSize,
                EmbeddingDimension,
                MaxTokens,
                _numLayers,
                _numHeads,
                _feedForwardDim,
                LossFunction,
                Convert.ToDouble(MaxGradNorm));
        }

        /// <summary>
        /// Retrieves detailed metadata about the SPLADE configuration.
        /// </summary>
        public override ModelMetadata<T> GetModelMetadata()
        {
            var metadata = base.GetModelMetadata();
            metadata.Name = "SPLADE";
            metadata.Description = "SPLADE (Sparse Lexical and Expansion) high-precision retrieval model";
            metadata.AdditionalInfo["VocabSize"] = _vocabSize;
            return metadata;
        }

        /// <inheritdoc/>
        protected override void SerializeNetworkSpecificData(BinaryWriter writer)
        {
            base.SerializeNetworkSpecificData(writer);
            writer.Write(_vocabSize);
            writer.Write(_numLayers);
            writer.Write(_numHeads);
            writer.Write(_feedForwardDim);
        }

        /// <inheritdoc/>
        protected override void DeserializeNetworkSpecificData(BinaryReader reader)
        {
            base.DeserializeNetworkSpecificData(reader);
            _vocabSize = reader.ReadInt32();
            _numLayers = reader.ReadInt32();
            _numHeads = reader.ReadInt32();
            _feedForwardDim = reader.ReadInt32();
        }

        /// <inheritdoc/>
        public override Task<Vector<T>> EmbedAsync(string text)
        {
            return Task.FromResult(Embed(text));
        }

        /// <inheritdoc/>
        public override Task<Matrix<T>> EmbedBatchAsync(IEnumerable<string> texts)
        {
            return Task.FromResult(EmbedBatch(texts));
        }

        #endregion
    }
}
