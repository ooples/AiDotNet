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
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Tokenization.Interfaces;

namespace AiDotNet.NeuralNetworks
{
    /// <summary>
    /// SGPT (Sentence GPT) neural network implementation using decoder-only transformer architectures.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
    /// <remarks>
    /// <para>
    /// SGPT leverages large-scale decoder-only models (like GPT-2 or GPT-Neo) to generate high-quality 
    /// sentence embeddings. By focusing on the last token of a sequence, the model utilizes the 
    /// unidirectional context to summarize the entire sentence's meaning.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Most AI models are like "readers" who read a whole sentence and then think about it. 
    /// SGPT is like a "writer." Because it's trained to write sentences one word at a time, it has a 
    /// very deep understanding of how sentences are built. When it finishes a sentence, the very 
    /// last word it would have written contains a "mental summary" of everything that came before it. 
    /// SGPT uses that summary as the coordinate (embedding) for the whole sentence.
    /// </para>
    /// </remarks>
    public class SGPT<T> : TransformerEmbeddingNetwork<T>
    {
        private readonly SGPTOptions _options;

        /// <inheritdoc/>
        public override ModelOptions GetOptions() => _options;

        #region Fields

        private int _vocabSize;
        private int _numLayers;
        private int _numHeads;
        private int _feedForwardDim;

        #endregion

        #region Constructors

        /// <summary>
        /// Initializes a new instance of the SGPT model.
        /// </summary>
        /// <param name="architecture">The configuration defining the model's structure.</param>
        /// <param name="tokenizer">Optional tokenizer for text processing.</param>
        /// <param name="optimizer">Optional optimizer for training.</param>
        /// <param name="vocabSize">The size of the vocabulary (default: 50257 for GPT-2).</param>
        /// <param name="embeddingDimension">The dimension of the embeddings (default: 768).</param>
        /// <param name="maxSequenceLength">The maximum length of input sequences (default: 1024).</param>
        /// <param name="numLayers">The number of transformer layers (default: 12).</param>
        /// <param name="numHeads">The number of attention heads (default: 12).</param>
        /// <param name="feedForwardDim">The hidden dimension of feed-forward networks (default: 3072).</param>
        /// <param name="poolingStrategy">The strategy used to aggregate outputs (default: Mean, though research often uses last token).</param>
        /// <param name="lossFunction">Optional loss function.</param>
        /// <param name="maxGradNorm">Maximum gradient norm for stability (default: 1.0).</param>
        public SGPT(
            NeuralNetworkArchitecture<T> architecture,
            ITokenizer? tokenizer = null,
            IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
            int vocabSize = 50257,
            int embeddingDimension = 768,
            int maxSequenceLength = 1024,
            int numLayers = 12,
            int numHeads = 12,
            int feedForwardDim = 3072,
            PoolingStrategy poolingStrategy = PoolingStrategy.Mean,
            ILossFunction<T>? lossFunction = null,
            double maxGradNorm = 1.0,
            SGPTOptions? options = null)
            : base(architecture, tokenizer, optimizer, vocabSize, embeddingDimension, maxSequenceLength, numLayers, numHeads, feedForwardDim, poolingStrategy, lossFunction, maxGradNorm)
        {
            _options = options ?? new SGPTOptions();
            Options = _options;
            _vocabSize = vocabSize;
            _numLayers = numLayers;
            _numHeads = numHeads;
            _feedForwardDim = feedForwardDim;

            InitializeLayersCore(false);
        }

        #endregion

        #region Initialization

        /// <summary>
        /// Configures the transformer layers for the SGPT model using decoder-only defaults from LayerHelper.
        /// </summary>
        /// <remarks>
        /// <b>For Beginners:</b> This method builds the "writer's brain." It sets up a large 
        /// transformer brain that is specifically tuned to understand the flow of information 
        /// from the start of a sentence to the very end.
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
                Layers.AddRange(LayerHelper<T>.CreateDefaultSGPTLayers(
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
            return new SGPT<T>(
                Architecture,
                null,
                null,
                _vocabSize,
                EmbeddingDimension,
                MaxTokens,
                _numLayers,
                _numHeads,
                _feedForwardDim,
                PoolingStrategy.Mean,
                LossFunction,
                Convert.ToDouble(MaxGradNorm));
        }

        /// <summary>
        /// Retrieves metadata about the SGPT model.
        /// </summary>
        /// <returns>Metadata containing model type and naming information.</returns>
        public override ModelMetadata<T> GetModelMetadata()
        {
            var metadata = base.GetModelMetadata();
            metadata.Name = "SGPT";
            metadata.Description = "SGPT (Sentence GPT) decoder-based high-quality embedding model";
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
