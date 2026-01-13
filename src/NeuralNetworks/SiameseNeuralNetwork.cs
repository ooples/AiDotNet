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
    /// Siamese Neural Network implementation for dual-encoder comparison and similarity learning.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
    /// <remarks>
    /// <para>
    /// A Siamese Neural Network consists of two identical subnetworks (sharing the same parameters) 
    /// that process two different inputs. The outputs are typically compared using a distance metric 
    /// and optimized via contrastive or triplet loss.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> A Siamese Network is like having two identical twins who think exactly 
    /// the same way. You give a different photo to each twin, and they each describe what they see 
    /// using a list of numbers. Because the twins think the same way, if the photos are similar, 
    /// their descriptions will be almost identical. This is the most popular way to build face 
    /// recognition or "find similar" search systems.
    /// </para>
    /// </remarks>
    public class SiameseNeuralNetwork<T> : NeuralNetworkBase<T>, IEmbeddingModel<T>
    {
        #region Fields

        /// <summary>
        /// The tokenizer used to process text inputs into numerical token IDs.
        /// </summary>
        private readonly ITokenizer? _tokenizer;

        /// <summary>
        /// The dimensionality of the shared embedding space.
        /// </summary>
        private int _embeddingDimension;

        /// <summary>
        /// The maximum length of input sequences the model will process.
        /// </summary>
        private int _maxSequenceLength;

        /// <summary>
        /// The number of unique tokens the shared encoder can recognize.
        /// </summary>
        private int _vocabSize;

        /// <summary>
        /// The loss function used to evaluate similarity (defaults to ContrastiveLoss).
        /// </summary>
        private ILossFunction<T> _lossFunction;

        /// <summary>
        /// The optimization algorithm used to update the shared parameters of the dual encoders.
        /// </summary>
        private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

        #endregion

        #region Properties

        /// <inheritdoc/>
        public int EmbeddingDimension => _embeddingDimension;

        /// <inheritdoc/>
        public int MaxTokens => _maxSequenceLength;

        #endregion

        #region Constructors

        /// <summary>
        /// Initializes a new instance of the SiameseNeuralNetwork model.
        /// </summary>
        /// <param name="architecture">The configuration defining the model's structural metadata.</param>
        /// <param name="tokenizer">Optional tokenizer for text processing.</param>
        /// <param name="optimizer">Optional optimizer for training.</param>
        /// <param name="vocabSize">The size of the shared vocabulary (default: 30522).</param>
        /// <param name="embeddingDimension">The dimension of the shared embeddings (default: 768).</param>
        /// <param name="maxSequenceLength">The maximum allowed input length (default: 512).</param>
        /// <param name="lossFunction">Optional loss function. Defaults to Contrastive Loss.</param>
        /// <param name="maxGradNorm">Maximum gradient norm for stability (default: 1.0).</param>
        public SiameseNeuralNetwork(
            NeuralNetworkArchitecture<T> architecture,
            ITokenizer? tokenizer = null,
            IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
            int vocabSize = 30522,
            int embeddingDimension = 768,
            int maxSequenceLength = 512,
            ILossFunction<T>? lossFunction = null,
            double maxGradNorm = 1.0)
            : base(architecture, lossFunction ?? new ContrastiveLoss<T>(), maxGradNorm)
        {
            _tokenizer = tokenizer;
            _vocabSize = vocabSize;
            _embeddingDimension = embeddingDimension;
            _maxSequenceLength = maxSequenceLength;
            _lossFunction = lossFunction ?? new ContrastiveLoss<T>();
            _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

            InitializeLayers();
        }

        #endregion

        #region Initialization

        /// <summary>
        /// Sets up the shared encoder layers for the Siamese twins using defaults from LayerHelper.
        /// </summary>
        /// <remarks>
        /// <b>For Beginners:</b> This method builds the "twin's brain." It sets up a transformer 
        /// encoder that is shared by both sides of the network to ensure that identical inputs 
        /// always produce identical results.
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
                // Default Siamese architecture: Transformer-based encoder from LayerHelper
                Layers.AddRange(LayerHelper<T>.CreateDefaultSiameseLayers(
                    Architecture,
                    _vocabSize,
                    _embeddingDimension,
                    _maxSequenceLength));
            }
        }

        #endregion

        #region IEmbeddingModel Implementation

        /// <summary>
        /// Encodes a single string into a normalized embedding vector using the shared encoder brain.
        /// </summary>
        public Vector<T> Embed(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                return new Vector<T>(_embeddingDimension);

            var tokenizer = _tokenizer ?? Tokenization.LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.OPT);
            var tokenResult = tokenizer.Encode(text);
            var tokens = tokenResult.TokenIds.Take(_maxSequenceLength).ToList();
            if (tokens.Count == 0) tokens.Add(0);

            var inputTensor = Tensor<T>.FromVector(new Vector<T>(tokens.Select(id => NumOps.FromDouble(id)).ToArray()), [1, tokens.Count]);
            var output = Predict(inputTensor);

            // Standard mean pooling to get a single vector from token representations
            int seqLen = output.Shape[1];
            int dim = output.Shape[2];
            var result = new Vector<T>(dim);
            if (seqLen == 0) return result.SafeNormalize();

            for (int d = 0; d < dim; d++)
            {
                T sum = NumOps.Zero;
                for (int s = 0; s < seqLen; s++)
                {
                    sum = NumOps.Add(sum, output[0, s, d]);
                }
                result[d] = NumOps.Divide(sum, NumOps.FromDouble(seqLen));
            }

            return result.SafeNormalize();
        }

        /// <summary>
        /// Encodes a batch of strings into a matrix of embedding vectors.
        /// </summary>
        public Matrix<T> EmbedBatch(IEnumerable<string> texts)
        {
            var textList = texts.ToList();
            var matrix = new Matrix<T>(textList.Count, _embeddingDimension);
            for (int i = 0; i < textList.Count; i++)
            {
                var emb = Embed(textList[i]);
                for (int j = 0; j < _embeddingDimension; j++) matrix[i, j] = emb[j];
            }
            return matrix;
        }

        #endregion

        #region Methods

        /// <summary>
        /// Performs a forward pass through the shared encoder.
        /// </summary>
        public Tensor<T> Forward(Tensor<T> input)
        {
            if (TryForwardGpuOptimized(input, out var gpuResult))
                return gpuResult;

            Tensor<T> output = input;
            foreach (var layer in Layers)
            {
                output = layer.Forward(output);
            }

            return output;
        }

        /// <summary>
        /// Propagates error signal backward through the shared encoder layers.
        /// </summary>
        public Tensor<T> Backward(Tensor<T> outputGradient)
        {
            for (int i = Layers.Count - 1; i >= 0; i--)
            {
                outputGradient = Layers[i].Backward(outputGradient);
            }

            return outputGradient;
        }

        /// <summary>
        /// Updates the shared parameters of the dual encoders.
        /// </summary>
        public override void UpdateParameters(Vector<T> parameters)
        {
            int index = 0;
            foreach (var layer in Layers)
            {
                int layerParameterCount = layer.ParameterCount;
                if (layerParameterCount > 0)
                {
                    var layerParameters = parameters.Slice(index, layerParameterCount);
                    layer.UpdateParameters(layerParameters);
                    index += layerParameterCount;
                }
            }
        }

        /// <inheritdoc/>
        public override Tensor<T> Predict(Tensor<T> input)
        {
            return Forward(input);
        }

        /// <summary>
        /// Trains the model on pairs of inputs using a similarity learning objective.
        /// </summary>
        /// <param name="input">The input tensor containing a pair of sequences [2, seq_len].</param>
        /// <param name="expectedOutput">The target similarity label (1 for same, 0 for different) as a tensor of [1].</param>
        /// <remarks>
        /// <b>For Beginners:</b> This is where the twins learn. You show one twin the first input 
        /// and the other twin the second input. You then tell them if the inputs are the same 
        /// or different (the label). They adjust their shared brain to make similar inputs 
        /// have nearly identical coordinate summaries.
        /// </remarks>
        public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
        {
            if (input.Shape[0] < 2)
                throw new ArgumentException("Siamese training requires a pair of inputs in the first dimension.", nameof(input));

            // Split input into two separate sequences
            var input1 = input.Slice(0, 0, 1);
            var input2 = input.Slice(0, 1, 1);

            // Forward pass for both inputs through the shared encoder
            var prediction1 = Predict(input1);
            var prediction2 = Predict(input2);

            // Pool outputs to get embeddings
            var emb1 = PoolOutput(prediction1);
            var emb2 = PoolOutput(prediction2);

            T label = expectedOutput[0];

            if (_lossFunction is ContrastiveLoss<T> contrastiveLoss)
            {
                LastLoss = contrastiveLoss.CalculateLoss(emb1, emb2, label);
                var (grad1, grad2) = contrastiveLoss.CalculateDerivative(emb1, emb2, label);

                // Backpropagate through both branches (shared parameters)
                Backpropagate(new Tensor<T>(prediction1.Shape, grad1));
                Backpropagate(new Tensor<T>(prediction2.Shape, grad2));
            }
            else
            {
                // Fallback for other loss functions
                base.Train(input, expectedOutput);
            }

            _optimizer.UpdateParameters(Layers);
        }

        private Vector<T> PoolOutput(Tensor<T> output)
        {
            int seqLen = output.Shape[1];
            int dim = output.Shape[2];
            var result = new Vector<T>(dim);
            if (seqLen == 0) return result.SafeNormalize();

            for (int d = 0; d < dim; d++)
            {
                T sum = NumOps.Zero;
                for (int s = 0; s < seqLen; s++)
                {
                    sum = NumOps.Add(sum, output[0, s, d]);
                }
                result[d] = NumOps.Divide(sum, NumOps.FromDouble(seqLen));
            }

            return result.SafeNormalize();
        }

        /// <inheritdoc/>
        protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        {
            return new SiameseNeuralNetwork<T>(
                Architecture,
                _tokenizer,
                null, // Fresh optimizer for new instance
                _vocabSize,
                _embeddingDimension,
                _maxSequenceLength,
                _lossFunction,
                Convert.ToDouble(MaxGradNorm));
        }

        /// <summary>
        /// Retrieves metadata about the Siamese dual-encoder model.
        /// </summary>
        public override ModelMetadata<T> GetModelMetadata()
        {
            return new ModelMetadata<T>
            {
                Name = "SiameseNeuralNetwork",
                ModelType = ModelType.SiameseNetwork,
                Description = "Standardized Siamese dual-encoder high-performance network",
                Complexity = ParameterCount,
                AdditionalInfo = new Dictionary<string, object>
                {
                    { "EmbeddingDimension", _embeddingDimension },
                    { "VocabSize", _vocabSize },
                    { "MaxSequenceLength", _maxSequenceLength }
                }
            };
        }

        /// <inheritdoc/>
        protected override void SerializeNetworkSpecificData(BinaryWriter writer)
        {
            writer.Write(_vocabSize);
            writer.Write(_embeddingDimension);
            writer.Write(_maxSequenceLength);
        }

        /// <inheritdoc/>
        protected override void DeserializeNetworkSpecificData(BinaryReader reader)
        {
            _vocabSize = reader.ReadInt32();
            _embeddingDimension = reader.ReadInt32();
            _maxSequenceLength = reader.ReadInt32();
        }

        #endregion
    }
}
