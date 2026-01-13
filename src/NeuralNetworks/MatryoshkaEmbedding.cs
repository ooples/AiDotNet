using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
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

        private int[] _nestedDimensions;
        private int _vocabSize;
        private int _numLayers;
        private int _numHeads;
        private int _feedForwardDim;

        #endregion

        #region Constructors

        /// <summary>
        /// Initializes a new instance of the MatryoshkaEmbedding model.
        /// </summary>
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
            _vocabSize = vocabSize;
            _numLayers = numLayers;
            _numHeads = numHeads;
            _feedForwardDim = feedForwardDim;
            _nestedDimensions = nestedDimensions ?? new[] { 64, 128, 256, 512, 768, 1024, 1536 };

            InitializeLayersCore(false);
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
                Layers.AddRange(LayerHelper<T>.CreateDefaultMRLLayers(
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

            return new Vector<T>(resizedValues).SafeNormalize();
        }

        /// <inheritdoc/>
        protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        {
            return new MatryoshkaEmbedding<T>(
                Architecture,
                null,
                null,
                _vocabSize,
                EmbeddingDimension,
                _nestedDimensions,
                MaxTokens,
                _numLayers,
                _numHeads,
                _feedForwardDim,
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
            writer.Write(_vocabSize);
            writer.Write(_numLayers);
            writer.Write(_numHeads);
            writer.Write(_feedForwardDim);
            writer.Write(_nestedDimensions.Length);
            foreach (var dim in _nestedDimensions)
            {
                writer.Write(dim);
            }
        }

        /// <inheritdoc/>
        protected override void DeserializeNetworkSpecificData(BinaryReader reader)
        {
            base.DeserializeNetworkSpecificData(reader);
            _vocabSize = reader.ReadInt32();
            _numLayers = reader.ReadInt32();
            _numHeads = reader.ReadInt32();
            _feedForwardDim = reader.ReadInt32();
            int count = reader.ReadInt32();
            _nestedDimensions = new int[count];
            for (int i = 0; i < count; i++)
            {
                _nestedDimensions[i] = reader.ReadInt32();
            }
        }

        /// <inheritdoc/>
        public override Vector<T> Embed(string text)
        {
            return base.Embed(text);
        }

        /// <inheritdoc/>
        public override Task<Vector<T>> EmbedAsync(string text)
        {
            return base.EmbedAsync(text);
        }

        /// <inheritdoc/>
        public override Task<Matrix<T>> EmbedBatchAsync(IEnumerable<string> texts)
        {
            return base.EmbedBatchAsync(texts);
        }

        #endregion
    }
}
