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
        #region Fields

        /// <summary>
        /// The training strategy used (Unsupervised or Supervised).
        /// </summary>
        /// <remarks>
        /// <b>For Beginners:</b> 
        /// - <b>Unsupervised:</b> Learns entirely from raw text without any human help.
        /// - <b>Supervised:</b> Learns from human-labeled pairs (like "this sentence means the same as that one").
        /// </remarks>
        private readonly SimCSEType _simCseType;

        /// <summary>
        /// The percentage of internal connections to randomly ignore during training.
        /// </summary>
        /// <remarks>
        /// <b>For Beginners:</b> This is the "noise" we add to the model. By randomly ignoring some 
        /// information (e.g., 10%), we force the model to be more robust and find the most 
        /// important parts of every sentence.
        /// </remarks>
        private readonly double _dropoutRate;

        #endregion

        #region Constructors

        /// <summary>
        /// Initializes a new instance of the SimCSE model.
        /// </summary>
        /// <param name="architecture">The configuration defining the model's structural metadata.</param>
        /// <param name="tokenizer">Optional tokenizer for text processing.</param>
        /// <param name="optimizer">Optional optimizer for training.</param>
        /// <param name="type">The SimCSE learning mode (default: Unsupervised).</param>
        /// <param name="vocabSize">The size of the vocabulary (default: 30522).</param>
        /// <param name="embeddingDimension">The dimension of the sentence vectors (default: 768).</param>
        /// <param name="maxSequenceLength">The maximum length of input sequences (default: 512).</param>
        /// <param name="numLayers">The number of transformer layers (default: 12).</param>
        /// <param name="numHeads">The number of attention heads (default: 12).</param>
        /// <param name="feedForwardDim">The hidden dimension of feed-forward networks (default: 3072).</param>
        /// <param name="dropoutRate">The dropout probability (default: 0.1).</param>
        /// <param name="poolingStrategy">The strategy for creating a single vector (default: ClsToken).</param>
        /// <param name="lossFunction">Optional loss function.</param>
        /// <param name="maxGradNorm">Maximum gradient norm for stability (default: 1.0).</param>
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

            InitializeLayers();
        }

        #endregion

        #region Initialization

        /// <summary>
        /// Configures the transformer encoder layers for SimCSE based on standard research patterns from LayerHelper.
        /// </summary>
        /// <remarks>
        /// <b>For Beginners:</b> This method builds the model's "thinking engine." It sets up a deep 
        /// stack of layers that allow the model to look at every word in context and understand 
        /// complex grammar and meaning.
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
                Layers.AddRange(LayerHelper<T>.CreateDefaultSimCSELayers(
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
        /// Custom training implementation for the SimCSE contrastive objective.
        /// </summary>
        /// <param name="input">The input tokens.</param>
        /// <param name="expectedOutput">The target representations.</param>
        /// <remarks>
        /// <b>For Beginners:</b> This is where the "compare the same sentence twice" logic happens. 
        /// Even if the model gets the exact same words twice, the internal "dropout noise" makes 
        /// the numbers slightly different. The model's job during training is to learn how to 
        /// ignore that noise and find the identical meaning.
        /// </remarks>
        public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
        {
            if (_simCseType == SimCSEType.Unsupervised)
            {
                // Unsupervised training typically involves processing the same input batch twice 
                // within the same contrastive loss calculation.
                base.Train(input, expectedOutput);
            }
            else
            {
                base.Train(input, expectedOutput);
            }
        }

        /// <inheritdoc/>
        protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        {
            return new SimCSE<T>(
                Architecture,
                null,
                null,
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
        }

        /// <inheritdoc/>
        protected override void DeserializeNetworkSpecificData(BinaryReader reader)
        {
        }

        #endregion
    }
}