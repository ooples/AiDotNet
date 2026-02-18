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
    /// A customizable Transformer-based embedding network.
    /// This serves as the high-performance foundation for modern sentence and document encoders.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
    /// <remarks>
    /// <para>
    /// This network provides a flexible implementation of the Transformer encoder architecture, 
    /// enabling the generation of high-quality semantic embeddings. It supports multiple 
    /// pooling strategies (Mean, Max, ClsToken) to aggregate token-level information.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is a "universal reading brain." Transformers are the most powerful 
    /// type of AI for understanding language because they can look at every word in a sentence 
    /// at the same time and see how they all relate. This customizable version lets you decide 
    /// how many layers of thinking the brain should have, and how it should summarize its 
    /// thoughts into a final list of numbers (the embedding).
    /// </para>
    /// </remarks>
    public class TransformerEmbeddingNetwork<T> : NeuralNetworkBase<T>, IEmbeddingModel<T>
    {
        private readonly TransformerEmbeddingOptions _options;

        /// <inheritdoc/>
        public override ModelOptions GetOptions() => _options;

        #region Fields

        /// <summary>
        /// The tokenizer used to translate raw text into numerical token IDs.
        /// </summary>
        private readonly ITokenizer? _tokenizer;

        /// <summary>
        /// The size of the output embedding vector.
        /// </summary>
        private int _embeddingDimension;

        /// <summary>
        /// The maximum sequence length allowed for input text.
        /// </summary>
        private int _maxSequenceLength;

        /// <summary>
        /// The size of the model's vocabulary (number of unique tokens).
        /// </summary>
        private int _vocabSize;

        /// <summary>
        /// The strategy used to pool token-level representations into a single vector.
        /// </summary>
        private PoolingStrategy _poolingStrategy;

        /// <summary>
        /// The total number of transformer encoder layers in the stack.
        /// </summary>
        private int _numLayers;

        /// <summary>
        /// The number of attention heads used in each multi-head attention layer.
        /// </summary>
        private int _numHeads;

        /// <summary>
        /// The dimensionality of the internal feed-forward hidden layers.
        /// </summary>
        private int _feedForwardDim;

        /// <summary>
        /// The loss function used during training.
        /// </summary>
        private ILossFunction<T> _lossFunction;

        /// <summary>
        /// The optimization algorithm used to refine the network's parameters.
        /// </summary>
        private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

        /// <summary>
        /// Cached fallback tokenizer to avoid per-call creation.
        /// </summary>
        private ITokenizer? _fallbackTokenizer;

        #endregion

        #region Properties

        /// <inheritdoc/>
        public int EmbeddingDimension => _embeddingDimension;

        /// <inheritdoc/>
        public int MaxTokens => _maxSequenceLength;

        /// <summary>
        /// Defines the available pooling strategies for creating a single sentence embedding.
        /// </summary>
        public enum PoolingStrategy
        {
            /// <summary>
            /// Averages all token representations across the sequence.
            /// </summary>
            Mean,

            /// <summary>
            /// Takes the maximum value across all sequence positions for each dimension.
            /// </summary>
            Max,

            /// <summary>
            /// Uses the representation of the first token (typically the [CLS] token).
            /// </summary>
            ClsToken
        }

        #endregion

        #region Constructors

        /// <summary>
        /// Initializes a new instance of the TransformerEmbeddingNetwork.
        /// </summary>
        /// <param name="architecture">The architecture metadata configuration.</param>
        /// <param name="tokenizer">Optional tokenizer for text processing.</param>
        /// <param name="optimizer">Optional optimizer for training.</param>
        /// <param name="vocabSize">The size of the vocabulary (default: 30522).</param>
        /// <param name="embeddingDimension">The size of the output vectors (default: 768).</param>
        /// <param name="maxSequenceLength">The maximum input sequence length (default: 512).</param>
        /// <param name="numLayers">The number of transformer layers (default: 12).</param>
        /// <param name="numHeads">The number of attention heads (default: 12).</param>
        /// <param name="feedForwardDim">The feed-forward hidden dimension (default: 3072).</param>
        /// <param name="poolingStrategy">The pooling method (default: Mean).</param>
        /// <param name="lossFunction">Optional loss function.</param>
        /// <param name="maxGradNorm">Maximum gradient norm for stability (default: 1.0).</param>
        public TransformerEmbeddingNetwork(
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
            TransformerEmbeddingOptions? options = null)
            : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), maxGradNorm)
        {
            _options = options ?? new TransformerEmbeddingOptions();
            Options = _options;

            _tokenizer = tokenizer;
            _vocabSize = vocabSize;
            _embeddingDimension = embeddingDimension;
            _maxSequenceLength = maxSequenceLength;
            _poolingStrategy = poolingStrategy;
            _numLayers = numLayers;
            _numHeads = numHeads;
            _feedForwardDim = feedForwardDim;
            _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
            _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

            InitializeLayersCore(false);
        }

        #endregion

        #region Initialization

        /// <summary>
        /// Sets up the layer stack for the transformer network, including embedding, positional encoding, and transformer blocks.
        /// </summary>
        /// <remarks>
        /// <b>For Beginners:</b> This method builds the brain's internal architecture. It sets up 
        /// the "translator" (embedding layer) to understand IDs, the "clock" (positional encoding) 
        /// to understand word order, and multiple "thinking centers" (transformer encoder layers) 
        /// to process complex context.
        /// </remarks>
        protected override void InitializeLayers()
        {
            InitializeLayersCore(true);
        }

        private void InitializeLayersCore(bool useVirtualValidation)
        {
            ClearLayers();

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
                Layers.AddRange(LayerHelper<T>.CreateTransformerEmbeddingLayers(
                    _vocabSize, _embeddingDimension, _maxSequenceLength,
                    _numLayers, _numHeads, _feedForwardDim));
            }
        }

        #endregion

        #region Methods

        /// <summary>
        /// Encodes a single string into a normalized summary vector.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <returns>A normalized embedding vector.</returns>
        /// <remarks>
        /// <b>For Beginners:</b> This is the main use case. You give the model a sentence, 
        /// it reads it with all its layers, summarizes the meaning based on your chosen 
        /// pooling strategy (like taking the average meaning), and returns one final 
        /// list of numbers.
        /// </remarks>
        public virtual Vector<T> Embed(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                return new Vector<T>(_embeddingDimension);

            var tokenizer = _tokenizer ?? (_fallbackTokenizer ??= Tokenization.LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.OPT));
            var tokenResult = tokenizer.Encode(text);
            var tokens = tokenResult.TokenIds.Take(_maxSequenceLength).ToList();

            if (tokens.Count == 0) tokens.Add(0);

            // OOV protection: ensure token IDs are within vocabulary bounds
            for (int i = 0; i < tokens.Count; i++)
            {
                if (tokens[i] < 0 || tokens[i] >= _vocabSize)
                {
                    tokens[i] = 0; // Fallback to unknown token ID
                }
            }

            var inputTensor = Tensor<T>.FromVector(new Vector<T>(tokens.Select(id => NumOps.FromDouble(id)).ToArray()), [1, tokens.Count]);

            var output = Predict(inputTensor);
            return PoolOutput(output);
        }

        /// <inheritdoc/>
        public virtual Task<Vector<T>> EmbedAsync(string text)
        {
            return Task.FromResult(Embed(text));
        }

        /// <summary>
        /// Encodes a collection of strings into a matrix of embeddings.
        /// </summary>
        /// <param name="texts">The texts to encode.</param>
        /// <returns>A matrix where each row is an embedding for the corresponding input string.</returns>
        public Matrix<T> EmbedBatch(IEnumerable<string> texts)
        {
            var textList = texts.ToList();
            if (textList.Count == 0) return new Matrix<T>(0, _embeddingDimension);

            var matrix = new Matrix<T>(textList.Count, _embeddingDimension);
            for (int i = 0; i < textList.Count; i++)
            {
                var embedding = Embed(textList[i]);
                for (int j = 0; j < _embeddingDimension; j++)
                {
                    matrix[i, j] = embedding[j];
                }
            }
            return matrix;
        }

        /// <inheritdoc/>
        public virtual Task<Matrix<T>> EmbedBatchAsync(IEnumerable<string> texts)
        {
            return Task.FromResult(EmbedBatch(texts));
        }

        /// <summary>
        /// Applies the configured pooling strategy to convert token-level outputs into a sentence representation.
        /// </summary>
        /// <param name="output">The 3D tensor of token representations [batch, seq, dim].</param>
        /// <returns>A single pooled vector for each sentence.</returns>
        protected virtual Vector<T> PoolOutput(Tensor<T> output)
        {
            if (output.Shape.Length != 2 && output.Shape.Length != 3)
                throw new ArgumentException($"PoolOutput expects Rank 2 [Seq, Dim] or Rank 3 [Batch, Seq, Dim] tensor, but got Rank {output.Shape.Length}.");

            int seqLen = output.Shape.Length == 3 ? output.Shape[1] : output.Shape[0];
            int dim = output.Shape.Length == 3 ? output.Shape[2] : output.Shape[1];
            var result = new Vector<T>(dim);

            // Helper to get value regardless of rank (assuming batch index 0 for rank 3)
            T GetVal(int s, int d) => output.Shape.Length == 3 ? output[0, s, d] : output[s, d];

            if (_poolingStrategy == PoolingStrategy.ClsToken)
            {
                for (int i = 0; i < dim; i++) result[i] = GetVal(0, i);
            }
            else if (_poolingStrategy == PoolingStrategy.Mean)
            {
                for (int d = 0; d < dim; d++)
                {
                    T sum = NumOps.Zero;
                    for (int s = 0; s < seqLen; s++)
                    {
                        sum = NumOps.Add(sum, GetVal(s, d));
                    }
                    result[d] = NumOps.Divide(sum, NumOps.FromDouble(seqLen));
                }
            }
            else if (_poolingStrategy == PoolingStrategy.Max)
            {
                for (int d = 0; d < dim; d++)
                {
                    T max = GetVal(0, d);
                    for (int s = 1; s < seqLen; s++)
                    {
                        T val = GetVal(s, d);
                        if (NumOps.GreaterThan(val, max)) max = val;
                    }
                    result[d] = max;
                }
            }

            return result.SafeNormalize();
        }

        /// <inheritdoc/>
        public override Tensor<T> Predict(Tensor<T> input)
        {
            if (TryForwardGpuOptimized(input, out var gpuResult))
                return gpuResult;

            Tensor<T> current = input;
            foreach (var layer in Layers)
            {
                current = layer.Forward(current);
            }
            return current;
        }

        /// <summary>
        /// Trains the transformer model on a single batch of data.
        /// </summary>
        public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
        {
            var prediction = Predict(input);
            LastLoss = _lossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());

            var outputGradient = _lossFunction.CalculateDerivative(prediction.ToVector(), expectedOutput.ToVector());
            var outputGradientTensor = new Tensor<T>(prediction.Shape, outputGradient);

            Backpropagate(outputGradientTensor);
            _optimizer.UpdateParameters(Layers);
        }

        /// <inheritdoc/>
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
        protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        {
            return new TransformerEmbeddingNetwork<T>(
                Architecture,
                _tokenizer,
                _optimizer,
                _vocabSize,
                _embeddingDimension,
                _maxSequenceLength,
                _numLayers,
                _numHeads,
                _feedForwardDim,
                _poolingStrategy,
                _lossFunction,
                Convert.ToDouble(MaxGradNorm));
        }

        /// <summary>
        /// Returns metadata about the transformer network configuration.
        /// </summary>
        public override ModelMetadata<T> GetModelMetadata()
        {
            return new ModelMetadata<T>
            {
                Name = "TransformerEmbeddingNetwork",
                ModelType = ModelType.Transformer,
                Description = "Customizable Transformer-based embedding foundation",
                Complexity = ParameterCount,
                AdditionalInfo = new Dictionary<string, object>
                {
                    { "EmbeddingDimension", _embeddingDimension },
                    { "NumLayers", _numLayers },
                    { "NumHeads", _numHeads },
                    { "PoolingStrategy", _poolingStrategy.ToString() }
                }
            };
        }

        /// <inheritdoc/>
        protected override void SerializeNetworkSpecificData(BinaryWriter writer)
        {
            writer.Write(_vocabSize);
            writer.Write(_embeddingDimension);
            writer.Write(_maxSequenceLength);
            writer.Write(_numLayers);
            writer.Write(_numHeads);
            writer.Write(_feedForwardDim);
            writer.Write((int)_poolingStrategy);
        }

        /// <inheritdoc/>
        protected override void DeserializeNetworkSpecificData(BinaryReader reader)
        {
            _vocabSize = reader.ReadInt32();
            _embeddingDimension = reader.ReadInt32();
            _maxSequenceLength = reader.ReadInt32();
            _numLayers = reader.ReadInt32();
            _numHeads = reader.ReadInt32();
            _feedForwardDim = reader.ReadInt32();
            _poolingStrategy = (PoolingStrategy)reader.ReadInt32();
        }

        #endregion

    }
}
