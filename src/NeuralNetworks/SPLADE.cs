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

        /// <summary>
        /// The size of the dictionary (vocabulary) used for expansion and sparse representation.
        /// </summary>
        /// <remarks>
        /// <b>For Beginners:</b> This is the number of possible words the model can choose to 
        /// "highlight" in its internal dictionary when trying to capture a sentence's meaning.
        /// </remarks>
        private readonly int _vocabSize;

        #endregion

        #region Constructors

        /// <summary>
        /// Initializes a new instance of the SPLADE model.
        /// </summary>
        /// <param name="architecture">The configuration defining the model structural metadata.</param>
        /// <param name="tokenizer">Optional tokenizer for text processing.</param>
        /// <param name="optimizer">Optional optimizer for training.</param>
        /// <param name="vocabSize">The size of the vocabulary (default: 30522).</param>
        /// <param name="embeddingDimension">The internal transformer dimension (default: 768).</param>
        /// <param name="maxSequenceLength">The maximum length of input sequences (default: 512).</param>
        /// <param name="numLayers">The number of transformer layers (default: 12).</param>
        /// <param name="numHeads">The number of attention heads (default: 12).</param>
        /// <param name="feedForwardDim">The hidden dimension of feed-forward networks (default: 3072).</param>
        /// <param name="lossFunction">Optional loss function.</param>
        /// <param name="maxGradNorm">Maximum gradient norm for stability (default: 1.0).</param>
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

            InitializeLayers();
        }

        #endregion

        #region Initialization

        /// <summary>
        /// Configures the transformer backbone and the ReLU-based expansion head for SPLADE.
        /// </summary>
        /// <remarks>
        /// <b>For Beginners:</b> This method builds the model's "expansion brain." It sets up 
        /// a transformer brain to understand context, followed by a special gate (ReLU) that 
        /// turns off unimportant dictionary words so that the final result is nice and sparse.
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
                Layers.AddRange(LayerHelper<T>.CreateDefaultSPLADELayers(
                    Architecture,
                    _vocabSize,
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
        /// Encodes text into a high-dimensional sparse lexical representation.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <returns>A sparse vector of size vocab_size representing semantic keyword weights.</returns>
        /// <remarks>
        /// <para>
        /// The embedding is calculated by taking the maximum activation across all token positions 
        /// for each vocabulary item, followed by a log-saturation effect: log(1 + ReLU(expansion)).
        /// </para>
        /// <para>
        /// <b>For Beginners:</b> This is the final step where your sentence is turned into a 
        /// "keyword map." It scans the sentence, thinks of related words, picks the strongest signals 
        /// from its dictionary, and creates a format that is incredibly good for finding exact 
        /// matches and related concepts simultaneously.
        /// </para>
        /// </remarks>
        public override Vector<T> Embed(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                return new Vector<T>(_vocabSize);

            var tokenizer = Tokenization.LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.OPT);
            var tokenResult = tokenizer.Encode(text);
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
                12,
                12,
                3072,
                LossFunction,
                Convert.ToDouble(MaxGradNorm));
        }

        /// <summary>
        /// Retrieves detailed metadata about the SPLADE configuration.
        /// </summary>
        /// <returns>Metadata object with model type and naming info.</returns>
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
        }

        /// <inheritdoc/>
        protected override void DeserializeNetworkSpecificData(BinaryReader reader)
        {
        }

        #endregion
    }
}
