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
        private readonly ITokenizer _tokenizer;
        private readonly int _maxTokens;

        #endregion

        #region Properties

        public int EmbeddingDimension => _embeddingDimension;
        public int MaxTokens => _maxTokens;

        #endregion

        #region Constructors

        public Word2Vec(
            NeuralNetworkArchitecture<T> architecture,
            ITokenizer? tokenizer = null,
            int vocabSize = 10000,
            int embeddingDimension = 100,
            int windowSize = 5,
            int maxTokens = 512,
            Word2VecType type = Word2VecType.SkipGram,
            ILossFunction<T>? lossFunction = null)
            : base(architecture, lossFunction ?? new BinaryCrossEntropyLoss<T>())
        {
            _tokenizer = tokenizer ?? Tokenization.LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.OPT);
            _vocabSize = vocabSize;
            _embeddingDimension = embeddingDimension;
            _windowSize = windowSize;
            _maxTokens = maxTokens;
            _type = type;

            InitializeLayers();
        }

        #endregion

        #region Initialization

        protected override void InitializeLayers()
        {
            ClearLayers();
            
            if (Architecture.Layers != null && Architecture.Layers.Count > 0)
            {
                foreach (var layer in Architecture.Layers)
                {
                    AddLayerToCollection(layer);
                }
            }
            else
            {
                var defaultLayers = LayerHelper<T>.CreateDefaultWord2VecLayers(
                    _vocabSize,
                    _embeddingDimension);

                foreach (var layer in defaultLayers)
                {
                    AddLayerToCollection(layer);
                }
            }
        }

        #endregion

        #region Methods

        public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
        {
            // Standard backpropagation training
            var output = ForwardWithMemory(input);
            
            // Formula loss and gradient
            var outputVec = output.ToVector();
            var expectedVec = expectedOutput.ToVector();
            
            var gradVec = LossFunction.CalculateDerivative(outputVec, expectedVec);
            var grad = Tensor<T>.FromVector(gradVec, output.Shape);
            
            Backpropagate(grad);
        }

        public override Tensor<T> Predict(Tensor<T> input)
        {
            if (TryForwardGpuOptimized(input, out var gpuResult))
                return gpuResult;

            // Input is index tensor
            // Word2Vec prediction typically means getting the target word vector (first layer)
            return Layers[0].Forward(input);
        }

        public override void UpdateParameters(Vector<T> parameters)
        {
            int offset = 0;
            foreach (var layer in Layers)
            {
                int count = layer.ParameterCount;
                if (count > 0)
                {
                    var layerParams = parameters.SubVector(offset, count);
                    layer.SetParameters(layerParams);
                    offset += count;
                }
            }
        }

        public Vector<T> Embed(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                return new Vector<T>(_embeddingDimension);

            var tokenResult = _tokenizer.Encode(text);
            var tokenIds = tokenResult.TokenIds.Take(_maxTokens).ToList();

            if (tokenIds.Count == 0)
                return new Vector<T>(_embeddingDimension);

            // Forward through first layer (U matrix) for all tokens
            var inputVec = new Vector<T>(tokenIds.Select(id => NumOps.FromDouble(id)).ToArray());
            var inputTensor = Tensor<T>.FromVector(inputVec, [tokenIds.Count]);
            
            var embeddings = Layers[0].Forward(inputTensor); // [seqLen, dim]
            
            // Average word vectors (Standard sentence embedding approach for Word2Vec)
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
                _vocabSize,
                _embeddingDimension,
                _windowSize,
                _maxTokens,
                _type,
                LossFunction);
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
            // Fields are readonly, typically set via factory or constructor during reload
            reader.ReadInt32(); // vocab
            reader.ReadInt32(); // dim
            reader.ReadInt32(); // window
            reader.ReadInt32(); // maxTokens
            reader.ReadInt32(); // type
        }

        #endregion
    }
}