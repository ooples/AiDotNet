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
    /// Matryoshka Representation Learning (MRL) neural network implementation.
    /// Learns nested embeddings where smaller prefixes of the full vector are valid representations.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
    /// <remarks>
    /// <para>
    /// Matryoshka Representation Learning (MRL) is a technique that enables a single model to adapt 
    /// its embedding dimension to the requirements of the downstream task. It optimizes for multiple 
    /// dimensions simultaneously, ensuring high accuracy even when using truncated vector prefixes.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Imagine a Russian nesting doll (a Matryoshka). Inside the big doll is a smaller one, 
    /// and inside that is an even smaller one. MRL works the same way: it creates a long list of numbers 
    /// to describe a sentence, but it makes sure that the first few numbers are a "perfect miniature" 
    /// of the whole meaning. This lets you use a tiny list for a fast search and the full list when 
    /// you need total accuracy.
    /// </para>
    /// </remarks>
    public class MatryoshkaEmbedding<T> : TransformerEmbeddingNetwork<T>
    {
        #region Fields

        /// <summary>
        /// The set of dimensions optimized during the Matryoshka training process.
        /// </summary>
        /// <remarks>
        /// <b>For Beginners:</b> This is the list of "doll sizes" the model knows. For example, 
        /// [64, 128, 256]. The model has been trained so that each of these lengths is a valid 
        /// standalone summary of the text.
        /// </remarks>
        private readonly int[] _nestedDimensions;

        #endregion

        #region Constructors

        /// <summary>
        /// Initializes a new instance of the MatryoshkaEmbedding model.
        /// </summary>
        /// <param name="architecture">The configuration defining the model metadata.</param>
        /// <param name="tokenizer">Optional tokenizer for text processing.</param>
        /// <param name="optimizer">Optional optimizer for training.</param>
        /// <param name="vocabSize">The size of the vocabulary (default: 30522).</param>
        /// <param name="maxEmbeddingDimension">The maximum (full) embedding size (default: 1536).</param>
        /// <param name="nestedDimensions">The set of optimized nesting levels.</param>
        /// <param name="maxSequenceLength">The maximum length of input sequences (default: 512).</param>
        /// <param name="numLayers">The number of transformer layers (default: 12).</param>
        /// <param name="numHeads">The number of attention heads (default: 12).</param>
        /// <param name="feedForwardDim">The hidden dimension of feed-forward networks (default: 3072).</param>
        /// <param name="poolingStrategy">The aggregation strategy (default: ClsToken).</param>
        /// <param name="lossFunction">Optional loss function.</param>
        /// <param name="maxGradNorm">Maximum gradient norm for stability (default: 1.0).</param>
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

            InitializeLayers();
        }

        #endregion

        #region Initialization

        /// <summary>
        /// Configures the transformer encoder and projection layers for the MRL architecture.
        /// </summary>
        /// <remarks>
        /// <b>For Beginners:</b> This method builds the model's "nested organization center." 
        /// It sets up a deep brain that learns to sort information by importance, making sure 
        /// the absolute most important facts are always at the beginning of its numerical output.
        /// </remarks>
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
        /// Encodes text into a truncated and re-normalized embedding of the requested dimension.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="dimension">The target dimension (e.g., 64, 128, 256).</param>
        /// <returns>A normalized vector containing only the first 'dimension' elements.</returns>
        /// <remarks>
        /// <b>For Beginners:</b> This is like picking which doll you want to use. If you want a 
        /// lightning-fast search, you might only take the first 64 numbers. This method slices 
        /// the full list and makes sure it's mathematically consistent for comparison.
        /// </remarks>
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

        /// <inheritdoc/>
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

        /// <summary>
        /// Retrieves metadata about the Matryoshka configuration.
        /// </summary>
        /// <returns>Metadata containing model type and nested dimension information.</returns>
        public override ModelMetadata<T> GetModelMetadata()
        {
            var metadata = base.GetModelMetadata();
            metadata.Name = "MatryoshkaEmbedding";
            metadata.Description = "Matryoshka Representation Learning (nested dimension) model";
            metadata.AdditionalInfo["NestedDimensions"] = _nestedDimensions;
            return metadata;
        }

        /// <inheritdoc/>
        protected override void SerializeNetworkSpecificData(BinaryWriter writer)
        {
            base.SerializeNetworkSpecificData(writer);
            writer.Write(_nestedDimensions.Length);
            foreach (var dim in _nestedDimensions)
            {
                writer.Write(dim);
            }
        }

        /// <inheritdoc/>
        protected override void DeserializeNetworkSpecificData(BinaryReader reader)
        {
        }

        #endregion
    }
}