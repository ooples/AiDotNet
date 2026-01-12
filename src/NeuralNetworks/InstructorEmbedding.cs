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
        #region Fields

        /// <summary>
        /// The default instruction string prepended to inputs if no specific instruction is provided.
        /// </summary>
        /// <remarks>
        /// <b>For Beginners:</b> This is the "default lens" the model uses to read your text if 
        /// you don't give it any special instructions.
        /// </remarks>
        private string _defaultInstruction = "Represent this text for retrieval: ";

        #endregion

        #region Constructors

        /// <summary>
        /// Initializes a new instance of the InstructorEmbedding model.
        /// </summary>
        /// <param name="architecture">The configuration defining the model's metadata.</param>
        /// <param name="tokenizer">Optional tokenizer for text processing.</param>
        /// <param name="optimizer">Optional optimizer for training.</param>
        /// <param name="vocabSize">The size of the vocabulary (default: 30522).</param>
        /// <param name="embeddingDimension">The dimension of the embeddings (default: 768).</param>
        /// <param name="maxSequenceLength">The maximum length of input sequences (default: 512).</param>
        /// <param name="numLayers">The number of transformer layers (default: 12).</param>
        /// <param name="numHeads">The number of attention heads (default: 12).</param>
        /// <param name="feedForwardDim">The hidden dimension of feed-forward networks (default: 3072).</param>
        /// <param name="poolingStrategy">The strategy used to aggregate token outputs (default: Mean).</param>
        /// <param name="lossFunction">Optional loss function.</param>
        /// <param name="maxGradNorm">Maximum gradient norm for stability (default: 1.0).</param>
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
            double maxGradNorm = 1.0)
            : base(architecture, tokenizer, optimizer, vocabSize, embeddingDimension, maxSequenceLength, numLayers, numHeads, feedForwardDim, poolingStrategy, lossFunction, maxGradNorm)
        {
            InitializeLayers();
        }

        #endregion

        #region Initialization

        /// <summary>
        /// Configures the transformer encoder layers for the Instructor architecture.
        /// </summary>
        /// <remarks>
        /// <b>For Beginners:</b> This method builds the model's "listening center." It sets up a 
        /// deep brain that can understand both your specific instructions and the actual text 
        /// you want to analyze.
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
                Layers.AddRange(LayerHelper<T>.CreateDefaultInstructorLayers(
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
        /// Sets the default instruction used for general embedding generation.
        /// </summary>
        /// <param name="instruction">The task instruction (e.g., "Represent the product for recommendation: ").</param>
        public void SetDefaultInstruction(string instruction)
        {
            _defaultInstruction = instruction;
        }

        /// <summary>
        /// Encodes text into a normalized embedding vector using a task-specific instruction.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="instruction">The specific instruction for this text (overrides default).</param>
        /// <returns>A normalized embedding vector.</returns>
        /// <remarks>
        /// <b>For Beginners:</b> This is how you "ask" the model to read something. You give it 
        /// the text and a goal. The model combines them to find the best set of numbers to 
        /// describe the text for THAT specific goal.
        /// </remarks>
        public Vector<T> EmbedWithInstruction(string text, string? instruction = null)
        {
            string fullText = (instruction ?? _defaultInstruction) + text;
            return base.Embed(fullText);
        }

        /// <inheritdoc/>
        /// <remarks>
        /// <b>For Beginners:</b> This uses the default instruction to create a general-purpose embedding.
        /// </remarks>
        public override Vector<T> Embed(string text)
        {
            return EmbedWithInstruction(text);
        }

        /// <inheritdoc/>
        protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        {
            return new InstructorEmbedding<T>(
                Architecture,
                null,
                null,
                30522,
                EmbeddingDimension,
                MaxTokens,
                12,
                12,
                3072,
                PoolingStrategy.Mean,
                LossFunction,
                Convert.ToDouble(MaxGradNorm));
        }

        /// <summary>
        /// Retrieves metadata about the Instructor model, including its default instruction.
        /// </summary>
        /// <returns>Metadata object containing configuration and instruction info.</returns>
        public override ModelMetadata<T> GetModelMetadata()
        {
            var metadata = base.GetModelMetadata();
            metadata.Name = "InstructorEmbedding";
            metadata.Description = "Instruction-Tuned high-flexibility embedding model";
            metadata.AdditionalInfo["DefaultInstruction"] = _defaultInstruction;
            return metadata;
        }

        /// <inheritdoc/>
        protected override void SerializeNetworkSpecificData(BinaryWriter writer)
        {
            base.SerializeNetworkSpecificData(writer);
            writer.Write(_defaultInstruction);
        }

        /// <inheritdoc/>
        protected override void DeserializeNetworkSpecificData(BinaryReader reader)
        {
        }

        #endregion
    }
}
