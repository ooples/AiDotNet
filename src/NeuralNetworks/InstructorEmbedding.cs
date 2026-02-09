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
    /// Instructor/E5 (Instruction-Tuned) embedding model implementation.
    /// Uses task-specific instructions to adapt embeddings for different use cases.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
    /// <remarks>
    /// <para>
    /// Instructor models are transformer-based encoders trained with instructions. By prepending a task 
    /// description (e.g., "Represent the Wikipedia sentence for retrieval:"), the model learns to 
    /// produce embeddings that are optimized for that specific task.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Most AI models read every sentence the same way. "Instructor" models are 
    /// like specialized scholars. If you tell them "read this like a doctor looking for a diagnosis," 
    /// they will focus on medical terms. If you tell them "read this like a poet," they will focus 
    /// on the mood. It makes the "coordinates" (embeddings) much more useful for your specific goal.
    /// </para>
    /// </remarks>
    public class InstructorEmbedding<T> : TransformerEmbeddingNetwork<T>
    {
        private readonly InstructorEmbeddingOptions _options;

        /// <inheritdoc/>
        public override ModelOptions GetOptions() => _options;

        #region Fields

        private string _defaultInstruction = "Represent this text for retrieval: ";
        private int _vocabSize;
        private int _numLayers;
        private int _numHeads;
        private int _feedForwardDim;
        private PoolingStrategy _poolingStrategy;

        #endregion

        #region Constructors

        /// <summary>
        /// Initializes a new instance of the InstructorEmbedding model.
        /// </summary>
        public InstructorEmbedding(
            NeuralNetworkArchitecture<T> architecture,
            ITokenizer? tokenizer = null,
            IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
            int vocabSize = 30522,
            int embeddingDimension = 768,
            int maxSequenceLength = 512,
            int numLayers = 12,
            int numHeads = 12,
            int feedForwardDim = 3072,
            PoolingStrategy poolingStrategy = PoolingStrategy.Mean,
            ILossFunction<T>? lossFunction = null,
            double maxGradNorm = 1.0,
            InstructorEmbeddingOptions? options = null)
            : base(architecture, tokenizer, optimizer, vocabSize, embeddingDimension, maxSequenceLength, numLayers, numHeads, feedForwardDim, poolingStrategy, lossFunction, maxGradNorm)
        {
            _options = options ?? new InstructorEmbeddingOptions();
            Options = _options;
            _vocabSize = vocabSize;
            _numLayers = numLayers;
            _numHeads = numHeads;
            _feedForwardDim = feedForwardDim;
            _poolingStrategy = poolingStrategy;

            InitializeLayersCore(false);
        }

        #endregion

        #region Initialization

        /// <summary>
        /// Configures the transformer encoder layers for the Instructor architecture.
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
                Layers.AddRange(LayerHelper<T>.CreateDefaultInstructorLayers(
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
        /// Sets the default instruction used for general embedding generation.
        /// </summary>
        public void SetDefaultInstruction(string instruction)
        {
            if (string.IsNullOrWhiteSpace(instruction))
                throw new ArgumentException("Instruction must be non-empty.", nameof(instruction));

            _defaultInstruction = instruction;
        }

        /// <summary>
        /// Encodes text into a normalized embedding vector using a task-specific instruction.
        /// </summary>
        public Vector<T> EmbedWithInstruction(string text, string? instruction = null)
        {
            if (text is null) throw new ArgumentNullException(nameof(text));
            string fullText = (instruction ?? _defaultInstruction) + text;
            return base.Embed(fullText);
        }

        /// <inheritdoc/>
        public override Vector<T> Embed(string text)
        {
            return EmbedWithInstruction(text);
        }

        /// <inheritdoc/>
        protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        {
            var instance = new InstructorEmbedding<T>(
                Architecture,
                null,
                null,
                _vocabSize,
                EmbeddingDimension,
                MaxTokens,
                _numLayers,
                _numHeads,
                _feedForwardDim,
                _poolingStrategy,
                LossFunction,
                Convert.ToDouble(MaxGradNorm));

            instance.SetDefaultInstruction(_defaultInstruction);
            return instance;
        }

        /// <summary>
        /// Retrieves metadata about the Instructor model, including its default instruction.
        /// </summary>
        public override ModelMetadata<T> GetModelMetadata()
        {
            var metadata = base.GetModelMetadata();
            metadata.Name = "InstructorEmbedding";
            metadata.Description = "Instruction-Tuned high-flexibility embedding model";

            if (metadata.AdditionalInfo == null)
                metadata.AdditionalInfo = new Dictionary<string, object>();

            metadata.AdditionalInfo["DefaultInstruction"] = _defaultInstruction;
            return metadata;
        }

        /// <inheritdoc/>
        protected override void SerializeNetworkSpecificData(BinaryWriter writer)
        {
            base.SerializeNetworkSpecificData(writer);
            writer.Write(_defaultInstruction);
            writer.Write(_vocabSize);
            writer.Write(_numLayers);
            writer.Write(_numHeads);
            writer.Write(_feedForwardDim);
            writer.Write((int)_poolingStrategy);
        }

        /// <inheritdoc/>
        protected override void DeserializeNetworkSpecificData(BinaryReader reader)
        {
            base.DeserializeNetworkSpecificData(reader);
            _defaultInstruction = reader.ReadString();
            _vocabSize = reader.ReadInt32();
            _numLayers = reader.ReadInt32();
            _numHeads = reader.ReadInt32();
            _feedForwardDim = reader.ReadInt32();
            _poolingStrategy = (PoolingStrategy)reader.ReadInt32();
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
