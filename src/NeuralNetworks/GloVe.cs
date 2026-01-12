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

        #endregion

        #region Properties

        public int EmbeddingDimension => _embeddingDimension;
        public int MaxTokens => 10000;

        #endregion

        #region Constructors

        public GloVe(
            NeuralNetworkArchitecture<T> architecture,
            ITokenizer? tokenizer = null,
            int vocabSize = 10000,
            int embeddingDimension = 100,
            ILossFunction<T>? lossFunction = null)
            : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>())
        {
            _tokenizer = tokenizer ?? Tokenization.LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.BERT);
            _vocabSize = vocabSize;
            _embeddingDimension = embeddingDimension;

            InitializeLayers();
        }

        #endregion

        #region Initialization

        protected override void InitializeLayers()
        {
            ClearLayers();

            if (Architecture.Layers != null && Architecture.Layers.Count > 0)
            {
                // Use custom layers from architecture
                foreach (var layer in Architecture.Layers)
                {
                    AddLayerToCollection(layer);
                }
            }
            else
            {
                // Use default layers via LayerHelper
                var defaultLayers = LayerHelper<T>.CreateDefaultGloVeLayers(
                    Architecture,
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
            // GloVe training is non-standard (weighted least squares on co-occurrence)
            // Here we use standard backprop as a placeholder for the NN structure
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
                    layer.UpdateParameters(layerParams);
                    offset += count;
                }
            }
        }

        public override Tensor<T> Predict(Tensor<T> input)
        {
            // Standard prediction uses main embedding layer (first layer)
            return Layers[0].Forward(input);
        }

        public Vector<T> Embed(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                return new Vector<T>(_embeddingDimension);

            var tokenResult = _tokenizer.Encode(text);
            var tokenIds = tokenResult.TokenIds;

            if (tokenIds.Count == 0)
                return new Vector<T>(_embeddingDimension);

            var sumVector = new Vector<T>(_embeddingDimension);
            int count = 0;

            foreach (var id in tokenIds)
            {
                if (id >= 0 && id < _vocabSize)
                {
                    var inputVec = new Vector<T>(new[] { NumOps.FromDouble(id) });
                    var inputTensor = Tensor<T>.FromVector(inputVec);
                    
                    // Use main embedding layer (first layer)
                    var w = Layers[0].Forward(inputTensor);
                    
                    for (int i = 0; i < _embeddingDimension; i++)
                    {
                        sumVector[i] = NumOps.Add(sumVector[i], w[0, i]);
                    }
                    count++;
                }
            }

            if (count == 0) return new Vector<T>(_embeddingDimension);

            var meanVector = new Vector<T>(_embeddingDimension);
            for (int i = 0; i < _embeddingDimension; i++)
            {
                meanVector[i] = NumOps.Divide(sumVector[i], NumOps.FromDouble(count));
            }

            return NormalizeVector(meanVector);
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

        private Vector<T> NormalizeVector(Vector<T> vector)
        {
            T sumSquares = NumOps.Zero;
            for (int i = 0; i < vector.Length; i++)
            {
                sumSquares = NumOps.Add(sumSquares, NumOps.Multiply(vector[i], vector[i]));
            }

            T norm = NumOps.FromDouble(Math.Sqrt(NumOps.ToDouble(sumSquares)));
            if (NumOps.ToDouble(norm) < 1e-10)
                return vector;

            var normalized = new Vector<T>(vector.Length);
            for (int i = 0; i < vector.Length; i++)
            {
                normalized[i] = NumOps.Divide(vector[i], norm);
            }

            return normalized;
        }

        protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        {
            return new GloVe<T>(
                Architecture,
                _tokenizer,
                _vocabSize,
                _embeddingDimension,
                LossFunction);
        }

        public override ModelMetadata<T> GetModelMetadata()
        {
            return new ModelMetadata<T>
            {
                Name = "GloVe",
                ModelType = ModelType.NeuralNetwork,
                Description = "GloVe embedding model",
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
        }

        protected override void DeserializeNetworkSpecificData(BinaryReader reader)
        {
            reader.ReadInt32();
            reader.ReadInt32();
        }

        #endregion
    }
}
