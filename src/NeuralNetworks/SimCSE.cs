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
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Tokenization.Interfaces;

namespace AiDotNet.NeuralNetworks
{
    /// <summary>
    /// SimCSE (Simple Contrastive Learning of Sentence Embeddings) neural network implementation.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
    /// <remarks>
    /// <para>
    /// SimCSE is a state-of-the-art framework for learning sentence embeddings. It uses a contrastive learning 
    /// objective to pull semantically similar sentences together and push dissimilar ones apart. 
    /// Its most famous variant is unsupervised, using different dropout masks on the same sentence 
    /// as a minimal data augmentation.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Imagine you're trying to recognize a friend in a crowded room. Even if they 
    /// are wearing a hat, glasses, or a scarf (like "dropout" noise), they are still the same person. 
    /// SimCSE trains the model by showing it the same sentence twice with different "masks" and 
    /// telling it: "this is the same sentence." This helps the model learn the true, deep meaning 
    /// of the sentence that stays constant regardless of small changes.
    /// </para>
    /// </remarks>
    public class SimCSE<T> : TransformerEmbeddingNetwork<T>
    {
        private readonly SimCSEOptions _options;

        /// <inheritdoc/>
        public override ModelOptions GetOptions() => _options;

        #region Fields

        private SimCSEType _simCseType;
        private double _dropoutRate;
        private int _vocabSize;
        private int _numLayers;
        private int _numHeads;
        private int _feedForwardDim;

        #endregion

        #region Constructors

        /// <summary>
        /// Initializes a new instance of the SimCSE model.
        /// </summary>
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
            double maxGradNorm = 1.0,
            SimCSEOptions? options = null)
            : base(architecture, tokenizer, optimizer, vocabSize, embeddingDimension, maxSequenceLength, numLayers, numHeads, feedForwardDim, poolingStrategy, lossFunction, maxGradNorm)
        {
            _options = options ?? new SimCSEOptions();
            Options = _options;
            _simCseType = type;
            _dropoutRate = dropoutRate;
            _vocabSize = vocabSize;
            _numLayers = numLayers;
            _numHeads = numHeads;
            _feedForwardDim = feedForwardDim;

            InitializeLayersCore(false);
        }

        #endregion

        #region Initialization

        /// <summary>
        /// Configures the transformer encoder layers for SimCSE based on standard research patterns from LayerHelper.
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
                Layers.AddRange(LayerHelper<T>.CreateDefaultSimCSELayers(
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
            return new SimCSE<T>(
                Architecture,
                null,
                null,
                _simCseType,
                _vocabSize,
                EmbeddingDimension,
                MaxTokens,
                _numLayers,
                _numHeads,
                _feedForwardDim,
                _dropoutRate,
                PoolingStrategy.ClsToken,
                LossFunction,
                Convert.ToDouble(MaxGradNorm));
        }

        /// <summary>
        /// Retrieves detailed metadata about the SimCSE configuration.
        /// </summary>
        /// <returns>Metadata object with training mode and dropout details.</returns>
        public override ModelMetadata<T> GetModelMetadata()
        {
            var metadata = base.GetModelMetadata();
            metadata.Name = "SimCSE";
            metadata.Description = $"SimCSE ({_simCseType}) contrastive embedding model";
            metadata.AdditionalInfo["SimCSEType"] = _simCseType.ToString();
            metadata.AdditionalInfo["DropoutRate"] = _dropoutRate;
            return metadata;
        }

        /// <inheritdoc/>
        protected override void SerializeNetworkSpecificData(BinaryWriter writer)
        {
            base.SerializeNetworkSpecificData(writer);
            writer.Write((int)_simCseType);
            writer.Write(_dropoutRate);
            writer.Write(_vocabSize);
            writer.Write(_numLayers);
            writer.Write(_numHeads);
            writer.Write(_feedForwardDim);
        }

        /// <inheritdoc/>
        protected override void DeserializeNetworkSpecificData(BinaryReader reader)
        {
            base.DeserializeNetworkSpecificData(reader);
            _simCseType = (SimCSEType)reader.ReadInt32();
            _dropoutRate = reader.ReadDouble();
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


