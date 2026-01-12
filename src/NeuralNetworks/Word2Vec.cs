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
    /// Word2Vec neural network implementation (Skip-Gram and CBOW architectures).
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class Word2Vec<T> : NeuralNetworkBase<T>, IEmbeddingModel<T>
    {
        #region Fields

        private readonly int _vocabSize;
        private readonly int _embeddingDimension;
        private readonly int _windowSize;
        private readonly Word2VecType _type;
        private readonly ITokenizer? _tokenizer;
        private readonly int _maxTokens;
        private readonly ILossFunction<T> _lossFunction;
        private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

        #endregion

        #region Properties

        public int EmbeddingDimension => _embeddingDimension;
        public int MaxTokens => _maxTokens;

        #endregion

        #region Constructors

        public Word2Vec(
            NeuralNetworkArchitecture<T> architecture,
            ITokenizer? tokenizer = null,
            IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
            int vocabSize = 10000,
            int embeddingDimension = 100,
            int windowSize = 5,
            int maxTokens = 512,
            Word2VecType type = Word2VecType.SkipGram,
            ILossFunction<T>? lossFunction = null,
            double maxGradNorm = 1.0)
            : base(architecture, lossFunction ?? new BinaryCrossEntropyLoss<T>(), maxGradNorm)
        {
            _tokenizer = tokenizer;
            _vocabSize = vocabSize;
            _embeddingDimension = embeddingDimension;
            _windowSize = windowSize;
            _maxTokens = maxTokens;
            _type = type;
            _lossFunction = lossFunction ?? new BinaryCrossEntropyLoss<T>();
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
                Layers.AddRange(LayerHelper<T>.CreateDefaultWord2VecLayers(
                    Architecture,
                    _vocabSize,
                    _embeddingDimension));
            }
        }

        #endregion

        #region Methods

        public override Tensor<T> Predict(Tensor<T> input)
        {
            return Forward(input);
        }

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

        public Tensor<T> Backward(Tensor<T> outputGradient)
        {
            for (int i = Layers.Count - 1; i >= 0; i--)
            {
                outputGradient = Layers[i].Backward(outputGradient);
            }

            return outputGradient;
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

        public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
        {
            var prediction = Predict(input);
            LastLoss = _lossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());

            var outputGradient = _lossFunction.CalculateDerivative(prediction.ToVector(), expectedOutput.ToVector());
            var outputGradientTensor = new Tensor<T>(prediction.Shape, outputGradient);

            var gradients = new List<Tensor<T>>();
            var currentGradient = outputGradientTensor;
            for (int i = Layers.Count - 1; i >= 0; i--)
            {
                currentGradient = Layers[i].Backward(currentGradient);
                gradients.Insert(0, currentGradient);
            }

            UpdateParameters(gradients);
        }

        private void UpdateParameters(List<Tensor<T>> gradients)
        {
            ClipGradients(gradients);
            _optimizer.UpdateParameters(Layers);
        }

        public Vector<T> Embed(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                return new Vector<T>(_embeddingDimension);

            var tokenizer = _tokenizer ?? Tokenization.LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.OPT);
            var tokenResult = tokenizer.Encode(text);
            var tokenIds = tokenResult.TokenIds.Take(_maxTokens).ToList();

            if (tokenIds.Count == 0)
                return new Vector<T>(_embeddingDimension);

            var inputVec = new Vector<T>(tokenIds.Select(id => NumOps.FromDouble(id)).ToArray());
            var inputTensor = Tensor<T>.FromVector(inputVec, [tokenIds.Count]);
            
            // First layer is always the target embedding layer (U matrix)
            var embeddings = Layers[0].Forward(inputTensor); // [seqLen, dim]
            
            var sumVector = new Vector<T>(_embeddingDimension);
            for (int s = 0; s < tokenIds.Count; s++)
            {
                for (int d = 0; d < _embeddingDimension; d++)
                {
                    sumVector[d] = NumOps.Add(sumVector[d], embeddings[s, d]);
                }
            }

            var meanVector = new Vector<T>(_embeddingDimension);
            for (int d = 0; d < _embeddingDimension; d++)
            {
                meanVector[d] = NumOps.Divide(sumVector[d], NumOps.FromDouble(tokenIds.Count));
            }

            return meanVector.Normalize();
        }

        public Matrix<T> EmbedBatch(IEnumerable<string> texts)
        {
            var textList = texts.ToList();
            var result = new Matrix<T>(textList.Count, _embeddingDimension);

            for (int i = 0; i < textList.Count; i++)
            {
                var embedding = Embed(textList[i]);
                for (int j = 0; j < _embeddingDimension; j++)
                {
                    result[i, j] = embedding[j];
                }
            }

            return result;
        }

        protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        {
            return new Word2Vec<T>(
                Architecture,
                _tokenizer,
                _optimizer,
                _vocabSize,
                _embeddingDimension,
                _windowSize,
                _maxTokens,
                _type,
                _lossFunction,
                Convert.ToDouble(MaxGradNorm));
        }

        public override ModelMetadata<T> GetModelMetadata()
        {
            return new ModelMetadata<T>
            {
                Name = "Word2Vec",
                ModelType = ModelType.NeuralNetwork,
                Description = $"Word2Vec ({_type}) embedding model",
                Complexity = ParameterCount,
                AdditionalInfo = new Dictionary<string, object>
                {
                    { "EmbeddingDimension", _embeddingDimension },
                    { "VocabSize", _vocabSize },
                    { "Type", _type.ToString() },
                    { "MaxTokens", _maxTokens }
                }
            };
        }

        protected override void SerializeNetworkSpecificData(BinaryWriter writer)
        {
            writer.Write(_vocabSize);
            writer.Write(_embeddingDimension);
            writer.Write(_windowSize);
            writer.Write(_maxTokens);
            writer.Write((int)_type);
        }

        protected override void DeserializeNetworkSpecificData(BinaryReader reader)
        {
        }

        #endregion
    }
}
