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
    /// A standardized Siamese Neural Network for dual-encoder embedding and comparison tasks.
    /// Follows the same pattern as other high-performance models in the library.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class SiameseNeuralNetwork<T> : NeuralNetworkBase<T>, IEmbeddingModel<T>
    {
        #region Fields

        private readonly ITokenizer? _tokenizer;
        private readonly int _embeddingDimension;
        private readonly int _maxSequenceLength;
        private readonly int _vocabSize;
        private readonly ILossFunction<T> _lossFunction;
        private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

        #endregion

        #region Properties

        public int EmbeddingDimension => _embeddingDimension;
        public int MaxTokens => _maxSequenceLength;

        #endregion

        #region Constructors

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

        protected override void InitializeLayers()
        {
            if (Architecture.Layers != null && Architecture.Layers.Count > 0)
            {
                Layers.AddRange(Architecture.Layers);
                ValidateCustomLayers(Layers);
            }
            else
            {
                // Default Siamese architecture: Transformer-based encoder
                Layers.AddRange(new ILayer<T>[]
                {
                    new EmbeddingLayer<T>(_vocabSize, _embeddingDimension),
                    new PositionalEncodingLayer<T>(_maxSequenceLength, _embeddingDimension),
                    new TransformerEncoderLayer<T>(_embeddingDimension, 12, 3072) // One default layer
                });
            }
        }

        #endregion

        #region IEmbeddingModel Implementation

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

            // Mean pooling
            var result = new Vector<T>(_embeddingDimension);
            for (int d = 0; d < _embeddingDimension; d++)
            {
                T sum = NumOps.Zero;
                for (int s = 0; s < output.Shape[1]; s++)
                {
                    sum = NumOps.Add(sum, output[0, s, d]);
                }
                result[d] = NumOps.Divide(sum, NumOps.FromDouble(output.Shape[1]));
            }

            return result.Normalize();
        }

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

        public override Tensor<T> Predict(Tensor<T> input)
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

        public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
        {
            // Siamese training typically involves comparing two inputs.
            // If input contains pairs [batch, 2, ...], we split and contrast.
            // Standard Train expects (input, expected).
            
            var prediction = Predict(input);
            LastLoss = _lossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());

            var outputGradient = _lossFunction.CalculateDerivative(prediction.ToVector(), expectedOutput.ToVector());
            var outputGradientTensor = new Tensor<T>(prediction.Shape, outputGradient);

            Backpropagate(outputGradientTensor);
            _optimizer.UpdateParameters(Layers);
        }

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

        protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        {
            return new SiameseNeuralNetwork<T>(
                Architecture,
                _tokenizer,
                _optimizer,
                _vocabSize,
                _embeddingDimension,
                _maxSequenceLength,
                _lossFunction,
                Convert.ToDouble(MaxGradNorm));
        }

        public override ModelMetadata<T> GetModelMetadata()
        {
            return new ModelMetadata<T>
            {
                Name = "SiameseNeuralNetwork",
                ModelType = ModelType.SiameseNetwork,
                Description = "Standardized Siamese dual-encoder network",
                Complexity = ParameterCount,
                AdditionalInfo = new Dictionary<string, object>
                {
                    { "EmbeddingDimension", _embeddingDimension },
                    { "VocabSize", _vocabSize }
                }
            };
        }

        protected override void SerializeNetworkSpecificData(BinaryWriter writer)
        {
            writer.Write(_vocabSize);
            writer.Write(_embeddingDimension);
            writer.Write(_maxSequenceLength);
        }

        protected override void DeserializeNetworkSpecificData(BinaryReader reader)
        {
        }

        #endregion
    }
}
