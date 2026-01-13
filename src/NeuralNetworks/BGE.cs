using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
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
    /// BGE (BAAI General Embedding) neural network implementation.
    /// A state-of-the-art retrieval model known for its high accuracy across diverse benchmarks.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
    /// <remarks>
    /// <para>
    /// BGE is a series of open-source embedding models from the Beijing Academy of Artificial Intelligence (BAAI). 
    /// These models are specifically optimized for retrieval tasks using a multi-stage training curriculum 
    /// that includes massive-scale pre-training and fine-grained instruction tuning.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> BGE is currently one of the "smartest" search engines in the world. It has been 
    /// trained like a student who went through elementary school (general reading), high school (specific facts), 
    /// and then a PhD program (answering hard questions). This makes it incredibly good at understanding 
    /// exactly what you're looking for, even if your query is phrased in a confusing way.
    /// </para>
    /// </remarks>
    public class BGE<T> : TransformerEmbeddingNetwork<T>
    {
        #region Fields

        private int _vocabSize;
        private int _numLayers;
        private int _numHeads;
        private int _feedForwardDim;

        #endregion

        #region Constructors

        /// <summary>
        /// Initializes a new instance of the BGE model.
        /// </summary>
        /// <param name="architecture">The configuration defining the model structure.</param>
        /// <param name="tokenizer">Optional tokenizer for text processing.</param>
        /// <param name="optimizer">Optional optimizer for training.</param>
        /// <param name="vocabSize">The size of the vocabulary (default: 30522).</param>
        /// <param name="embeddingDimension">The dimension of the embeddings (default: 768).</param>
        /// <param name="maxSequenceLength">The maximum length of input sequences (default: 512).</param>
        /// <param name="numLayers">The number of transformer layers (default: 12).</param>
        /// <param name="numHeads">The number of attention heads (default: 12).</param>
        /// <param name="feedForwardDim">The hidden dimension of feed-forward networks (default: 3072).</param>
        /// <param name="poolingStrategy">The strategy used to aggregate token outputs (default: ClsToken).</param>
        /// <param name="lossFunction">Optional loss function.</param>
        /// <param name="maxGradNorm">Maximum gradient norm for stability (default: 1.0).</param>
        public BGE(
            NeuralNetworkArchitecture<T> architecture,
            ITokenizer? tokenizer = null,
            IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
            int vocabSize = 30522,
            int embeddingDimension = 768,
            int maxSequenceLength = 512,
            int numLayers = 12,
            int numHeads = 12,
            int feedForwardDim = 3072,
            PoolingStrategy poolingStrategy = PoolingStrategy.ClsToken,
            ILossFunction<T>? lossFunction = null,
            double maxGradNorm = 1.0)
            : base(architecture, tokenizer, optimizer, vocabSize, embeddingDimension, maxSequenceLength, numLayers, numHeads, feedForwardDim, poolingStrategy, lossFunction, maxGradNorm)
        {
            _vocabSize = vocabSize;
            _numLayers = numLayers;
            _numHeads = numHeads;
            _feedForwardDim = feedForwardDim;

            InitializeLayersCore(false);
        }

        #endregion

        #region Initialization

        /// <summary>
        /// Configures the transformer layers for the BGE model using optimized retrieval defaults from LayerHelper.
        /// </summary>
        /// <remarks>
        /// <b>For Beginners:</b> This method builds the model's "library index." It sets up a 
        /// powerful transformer brain and a final precision checkpoint (layer normalization) 
        /// that makes sure every coordinate it creates is perfect for high-speed searching.
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
                Layers.AddRange(LayerHelper<T>.CreateDefaultBGELayers(
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

        /// <inheritdoc/>
        protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        {
            return new BGE<T>(
                Architecture,
                null,
                null,
                _vocabSize,
                EmbeddingDimension,
                MaxTokens,
                _numLayers,
                _numHeads,
                _feedForwardDim,
                PoolingStrategy.ClsToken,
                LossFunction,
                Convert.ToDouble(MaxGradNorm));
        }

        /// <summary>
        /// Retrieves metadata about the BGE model.
        /// </summary>
        /// <returns>Metadata containing model type and naming information.</returns>
        public override ModelMetadata<T> GetModelMetadata()
        {
            var metadata = base.GetModelMetadata();
            metadata.Name = "BGE";
            metadata.Description = "BGE (BAAI General Embedding) state-of-the-art retrieval model";
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
        public override Vector<T> Embed(string text)
        {
            return base.Embed(text);
        }

        /// <inheritdoc/>
        public override Task<Vector<T>> EmbedAsync(string text, CancellationToken cancellationToken = default)
        {
            return base.EmbedAsync(text, cancellationToken);
        }

        /// <inheritdoc/>
        public override Task<Matrix<T>> EmbedBatchAsync(IEnumerable<string> texts, CancellationToken cancellationToken = default)
        {
            return base.EmbedBatchAsync(texts, cancellationToken);
        }

        #endregion
    }
}
