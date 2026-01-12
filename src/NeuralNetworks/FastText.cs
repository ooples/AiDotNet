using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.Enums;
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
        private readonly int _vocabSize;
        private readonly int _bucketSize; // Size of n-gram buckets
        private readonly int _embeddingDimension;
        private readonly ITokenizer _tokenizer;
        
        private EmbeddingLayer<T> _wordEmbeddings;
        private EmbeddingLayer<T> _ngramEmbeddings; // Subword embeddings

        public int EmbeddingDimension => _embeddingDimension;
        public int MaxTokens => 10000;

        public FastText(
            NeuralNetworkArchitecture<T> architecture,
            ITokenizer tokenizer,
            int vocabSize,
            int bucketSize = 2000000,
            int embeddingDimension = 100,
            ILossFunction<T>? lossFunction = null)
            : base(architecture, lossFunction ?? new BinaryCrossEntropyLoss<T>())
        {
            _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
            _vocabSize = vocabSize;
            _bucketSize = bucketSize;
            _embeddingDimension = embeddingDimension;

            _wordEmbeddings = null!;
            _ngramEmbeddings = null!;

            InitializeLayers();
        }

        protected override void InitializeLayers()
        {
            ClearLayers();
            // Word embeddings
            _wordEmbeddings = new EmbeddingLayer<T>(_vocabSize, _embeddingDimension);
            AddLayerToCollection(_wordEmbeddings);

            // N-gram embeddings (hashed)
            _ngramEmbeddings = new EmbeddingLayer<T>(_bucketSize, _embeddingDimension);
            AddLayerToCollection(_ngramEmbeddings);
        }

        public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
        {
            // Standard backpropagation
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
            // Input usually indices of words
            // But FastText needs n-grams too.
            // Simplified: return word embeddings
            return _wordEmbeddings.Forward(input);
        }

        public Vector<T> Embed(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                return new Vector<T>(_embeddingDimension);

            var tokenResult = _tokenizer.Encode(text);
            // In a real implementation, we would also generate n-grams here and hash them
            
            var tokenIds = tokenResult.TokenIds;
            if (tokenIds.Count == 0) return new Vector<T>(_embeddingDimension);

            var sumVector = new Vector<T>(_embeddingDimension);
            int count = 0;

            foreach (var id in tokenIds)
            {
                // Word vector
                var wordVec = GetWordVector(id);
                
                // Add to sum
                for (int i = 0; i < _embeddingDimension; i++)
                {
                    sumVector[i] = NumOps.Add(sumVector[i], wordVec[i]);
                }
                count++;
            }

            if (count == 0) return new Vector<T>(_embeddingDimension);

            // Compute mean
            var meanVector = new Vector<T>(_embeddingDimension);
            for (int i = 0; i < _embeddingDimension; i++)
            {
                meanVector[i] = NumOps.Divide(sumVector[i], NumOps.FromDouble(count));
            }

            return NormalizeVector(meanVector);
        }

        private Vector<T> GetWordVector(int wordId)
        {
            var inputVec = new Vector<T>(new[] { NumOps.FromDouble(wordId) });
            var inputTensor = Tensor<T>.FromVector(inputVec);
            var wordEmb = _wordEmbeddings.Forward(inputTensor);
            
            // In FastText, word vector is sum of word embedding + n-gram embeddings
            // Here we simplify to just word embedding for structure demonstration
            
            var vec = new Vector<T>(_embeddingDimension);
            for(int i = 0; i < _embeddingDimension; i++) vec[i] = wordEmb[0, i];
            return vec;
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
            return new FastText<T>(
                Architecture,
                _tokenizer,
                _vocabSize,
                _bucketSize,
                _embeddingDimension,
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
                    { "BucketSize", _bucketSize }
                }
            };
        }

        protected override void SerializeNetworkSpecificData(BinaryWriter writer)
        {
            writer.Write(_vocabSize);
            writer.Write(_bucketSize);
            writer.Write(_embeddingDimension);
        }

        protected override void DeserializeNetworkSpecificData(BinaryReader reader)
        {
            reader.ReadInt32();
            reader.ReadInt32();
            reader.ReadInt32();
        }
    }
}
