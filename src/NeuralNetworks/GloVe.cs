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
    /// GloVe (Global Vectors for Word Representation) model implementation.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class GloVe<T> : NeuralNetworkBase<T>, IEmbeddingModel<T>
    {
        #region Fields

        private readonly int _vocabSize;
        private readonly int _embeddingDimension;
        private readonly ITokenizer _tokenizer;
        private readonly int _maxTokens;

        #endregion

        #region Properties

        public int EmbeddingDimension => _embeddingDimension;
        public int MaxTokens => _maxTokens;

        #endregion

        #region Constructors

        public GloVe(
            NeuralNetworkArchitecture<T> architecture,
            ITokenizer? tokenizer = null,
            int vocabSize = 10000,
            int embeddingDimension = 100,
            int maxTokens = 512,
            ILossFunction<T>? lossFunction = null)
            : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>())
        {
            _tokenizer = tokenizer ?? Tokenization.LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.OPT);
            _vocabSize = vocabSize;
            _embeddingDimension = embeddingDimension;
            _maxTokens = maxTokens;

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
                var defaultLayers = LayerHelper<T>.CreateDefaultGloVeLayers(
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
            var output = ForwardWithMemory(input);
            var outputVec = output.ToVector();
            var expectedVec = expectedOutput.ToVector();
            
            var gradVec = LossFunction.CalculateDerivative(outputVec, expectedVec);
            var grad = Tensor<T>.FromVector(gradVec, output.Shape);
            
            Backpropagate(grad);
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

        public override Tensor<T> Predict(Tensor<T> input)
        {
            if (TryForwardGpuOptimized(input, out var gpuResult))
                return gpuResult;

            // Standard prediction uses main embedding layer (first layer)
            return Layers[0].Forward(input);
        }

        public Vector<T> Embed(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                return new Vector<T>(_embeddingDimension);

            var tokenResult = _tokenizer.Encode(text);
            var tokenIds = tokenResult.TokenIds.Take(_maxTokens).ToList();

            if (tokenIds.Count == 0)
                return new Vector<T>(_embeddingDimension);

            var inputVec = new Vector<T>(tokenIds.Select(id => NumOps.FromDouble(id)).ToArray());
            var inputTensor = Tensor<T>.FromVector(inputVec, [tokenIds.Count]);
            
            // In GloVe, final word vector is often W + W_tilde
            // W is layer 0, W_tilde is layer 1
            var w = Layers[0].Forward(inputTensor);
            var w_tilde = Layers[1].Forward(inputTensor);
            
            var sumVector = new Vector<T>(_embeddingDimension);
            for (int s = 0; s < tokenIds.Count; s++)
            {
                for (int d = 0; d < _embeddingDimension; d++)
                {
                    T val = NumOps.Add(w[s, d], w_tilde[s, d]);
                    sumVector[d] = NumOps.Add(sumVector[d], val);
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
            return new GloVe<T>(
                Architecture,
                _tokenizer,
                _vocabSize,
                _embeddingDimension,
                _maxTokens,
                LossFunction);
        }

        public override ModelMetadata<T> GetModelMetadata()
        {
            return new ModelMetadata<T>
            {
                Name = "GloVe",
                ModelType = ModelType.NeuralNetwork,
                Description = "GloVe (Global Vectors) embedding model",
                Complexity = ParameterCount,
                AdditionalInfo = new Dictionary<string, object>
                {
                    { "EmbeddingDimension", _embeddingDimension },
                    { "VocabSize", _vocabSize },
                    { "MaxTokens", _maxTokens }
                }
            };
        }

        protected override void SerializeNetworkSpecificData(BinaryWriter writer)
        {
            writer.Write(_vocabSize);
            writer.Write(_embeddingDimension);
            writer.Write(_maxTokens);
        }

        protected override void DeserializeNetworkSpecificData(BinaryReader reader)
        {
            reader.ReadInt32();
            reader.ReadInt32();
            reader.ReadInt32();
        }

        #endregion
    }
}
