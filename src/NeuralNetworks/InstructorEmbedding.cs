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
    /// Uses instructions to guide the generation of task-specific embeddings.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class InstructorEmbedding<T> : TransformerEmbeddingNetwork<T>
    {
        #region Fields

        private string _defaultInstruction = "Represent this text for retrieval: ";

        #endregion

        #region Constructors

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
        }

        #endregion

        #region Initialization

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
        /// Sets the default instruction to prepend to texts if none is provided.
        /// </summary>
        public void SetDefaultInstruction(string instruction)
        {
            _defaultInstruction = instruction;
        }

        /// <summary>
        /// Encodes text into an embedding vector using a task-specific instruction.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <param name="instruction">The task instruction (e.g., "Represent the medical query for retrieval:").</param>
        /// <returns>A normalized embedding vector.</returns>
        public Vector<T> EmbedWithInstruction(string text, string? instruction = null)
        {
            string fullText = (instruction ?? _defaultInstruction) + text;
            return base.Embed(fullText);
        }

        public override Vector<T> Embed(string text)
        {
            return EmbedWithInstruction(text);
        }

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

        public override ModelMetadata<T> GetModelMetadata()
        {
            var metadata = base.GetModelMetadata();
            metadata.Name = "InstructorEmbedding";
            metadata.Description = "Instruction-Tuned embedding model (Instructor/E5 style)";
            metadata.AdditionalInfo["DefaultInstruction"] = _defaultInstruction;
            return metadata;
        }

        #endregion
    }
}
