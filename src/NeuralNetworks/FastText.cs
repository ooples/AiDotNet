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
    /// FastText embedding model implementation.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class FastText<T> : NeuralNetworkBase<T>, IEmbeddingModel<T>
    {
        #region Fields

        private readonly int _vocabSize;
        private readonly int _bucketSize; // Size of n-gram buckets
        private readonly int _embeddingDimension;
        private readonly ITokenizer _tokenizer;
        private readonly int _maxTokens;

        #endregion

        #region Properties

        public int EmbeddingDimension => _embeddingDimension;
        public int MaxTokens => _maxTokens;

        #endregion

        #region Constructors

        public FastText(
            NeuralNetworkArchitecture<T> architecture,
            ITokenizer? tokenizer = null,
            int vocabSize = 10000,
            int bucketSize = 2000000,
            int embeddingDimension = 100,
            int maxTokens = 512,
            ILossFunction<T>? lossFunction = null)
            : base(architecture, lossFunction ?? new BinaryCrossEntropyLoss<T>())
        {
            _tokenizer = tokenizer ?? Tokenization.LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.OPT);
            _vocabSize = vocabSize;
            _bucketSize = bucketSize;
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
                var defaultLayers = LayerHelper<T>.CreateDefaultFastTextLayers(
                    _vocabSize,
                    _bucketSize,
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

            // Simple prediction returns word embeddings
            return Layers[0].Forward(input);
        }

        public Vector<T> Embed(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                return new Vector<T>(_embeddingDimension);

            var tokenResult = _tokenizer.Encode(text);
            var tokenIds = tokenResult.TokenIds.Take(_maxTokens).ToList();
            if (tokenIds.Count == 0) return new Vector<T>(_embeddingDimension);

            var inputVec = new Vector<T>(tokenIds.Select(id => NumOps.FromDouble(id)).ToArray());
            var inputTensor = Tensor<T>.FromVector(inputVec, [tokenIds.Count]);
            
            // FastText word vector = word_embedding + sum(ngram_embeddings)
            // Layers[0] = Word Embeddings
            // Layers[1] = N-gram Embeddings
            var wordEmbeds = Layers[0].Forward(inputTensor);
            
            // Note: A true FastText implementation would generate n-grams from the text
            // and look them up in Layers[1]. Simplified here for structure.
            
            var sumVector = new Vector<T>(_embeddingDimension);
            for (int s = 0; s < tokenIds.Count; s++)
            {
                for (int d = 0; d < _embeddingDimension; d++)
                {
                    sumVector[d] = NumOps.Add(sumVector[d], wordEmbeds[s, d]);
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
            return new FastText<T>(
                Architecture,
                _tokenizer,
                _vocabSize,
                _bucketSize,
                _embeddingDimension,
                _maxTokens,
                LossFunction);
        }

        public override ModelMetadata<T> GetModelMetadata()
        {
            return new ModelMetadata<T>
            {
                Name = "FastText",
                ModelType = ModelType.NeuralNetwork,
                Description = "FastText embedding model",
                Complexity = ParameterCount,
                AdditionalInfo = new Dictionary<string, object>
                {
                    { "EmbeddingDimension", _embeddingDimension },
                    { "VocabSize", _vocabSize },
                    { "BucketSize", _bucketSize },
                    { "MaxTokens", _maxTokens }
                }
            };
        }

        protected override void SerializeNetworkSpecificData(BinaryWriter writer)
        {
            writer.Write(_vocabSize);
            writer.Write(_bucketSize);
            writer.Write(_embeddingDimension);
            writer.Write(_maxTokens);
        }

        protected override void DeserializeNetworkSpecificData(BinaryReader reader)
        {
            reader.ReadInt32();
            reader.ReadInt32();
            reader.ReadInt32();
            reader.ReadInt32();
        }

        #endregion
    }
}
