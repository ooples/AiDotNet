using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tokenization.Interfaces;

namespace AiDotNet.NeuralNetworks
{
    /// <summary>
    /// A customizable Transformer-based embedding network.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class TransformerEmbeddingNetwork<T> : NeuralNetworkBase<T>, IEmbeddingModel<T>
    {
        private readonly ITokenizer _tokenizer;
        private readonly int _embeddingDimension;
        private readonly int _maxSequenceLength;
        private readonly int _vocabSize;
        private readonly PoolingStrategy _poolingStrategy;
        private readonly int _numLayers;
        private readonly int _numHeads;
        private readonly int _feedForwardDim;

        public int EmbeddingDimension => _embeddingDimension;
        public int MaxTokens => _maxSequenceLength;

        public enum PoolingStrategy
        {
            Mean,
            Max,
            ClsToken
        }

        public TransformerEmbeddingNetwork(
            NeuralNetworkArchitecture<T> architecture,
            ITokenizer tokenizer,
            int vocabSize,
            int embeddingDimension = 768,
            int maxSequenceLength = 512,
            int numLayers = 12,
            int numHeads = 12,
            int feedForwardDim = 3072,
            PoolingStrategy poolingStrategy = PoolingStrategy.Mean,
            ILossFunction<T>? lossFunction = null)
            : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>())
        {
            _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
            _vocabSize = vocabSize;
            _embeddingDimension = embeddingDimension;
            _maxSequenceLength = maxSequenceLength;
            _poolingStrategy = poolingStrategy;
            _numLayers = numLayers;
            _numHeads = numHeads;
            _feedForwardDim = feedForwardDim;

            InitializeCustomLayers();
        }

        protected override void InitializeLayers()
        {
            // Base constructor calls this, but fields are not set yet.
            // We do initialization in InitializeCustomLayers called from constructor.
        }

        private void InitializeCustomLayers()
        {
            ClearLayers();

            // 1. Embedding Layer
            AddLayerToCollection(new EmbeddingLayer<T>(_vocabSize, _embeddingDimension));

            // 2. Positional Encoding
            AddLayerToCollection(new PositionalEncodingLayer<T>(_maxSequenceLength, _embeddingDimension));

            // 3. Transformer Encoder Layers
            for (int i = 0; i < _numLayers; i++)
            {
                AddLayerToCollection(new TransformerEncoderLayer<T>(_embeddingDimension, _numHeads, _feedForwardDim));
            }
        }

        public Vector<T> Embed(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                throw new ArgumentException("Text cannot be empty", nameof(text));

            var tokenResult = _tokenizer.Encode(text);
            var tokens = tokenResult.TokenIds.Take(_maxSequenceLength).ToList();
            
            // Pad or ensure at least one token
            if (tokens.Count == 0) tokens.Add(0); // Assuming 0 is pad/unknown, but we need something

            var inputTensor = new Tensor<T>(new int[] { 1, tokens.Count });
            for(int i = 0; i < tokens.Count; i++)
            {
                inputTensor[0, i] = NumOps.FromDouble(tokens[i]);
            }

            var output = Predict(inputTensor);
            return PoolOutput(output);
        }

        public Matrix<T> EmbedBatch(IEnumerable<string> texts)
        {
            var textList = texts.ToList();
            if (textList.Count == 0) return new Matrix<T>(0, _embeddingDimension);

            var embeddings = new List<Vector<T>>();
            foreach (var text in textList)
            {
                embeddings.Add(Embed(text));
            }

            var matrix = new Matrix<T>(embeddings.Count, _embeddingDimension);
            for (int i = 0; i < embeddings.Count; i++)
            {
                for (int j = 0; j < _embeddingDimension; j++)
                {
                    matrix[i, j] = embeddings[i][j];
                }
            }
            return matrix;
        }

        private Vector<T> PoolOutput(Tensor<T> output)
        {
            int seqLen = output.Shape[1];
            int dim = output.Shape[2];
            var result = new Vector<T>(dim);

            if (_poolingStrategy == PoolingStrategy.ClsToken)
            {
                for (int i = 0; i < dim; i++) result[i] = output[0, 0, i];
            }
            else if (_poolingStrategy == PoolingStrategy.Mean)
            {
                for (int d = 0; d < dim; d++)
                {
                    T sum = NumOps.Zero;
                    for (int s = 0; s < seqLen; s++)
                    {
                        sum = NumOps.Add(sum, output[0, s, d]);
                    }
                    result[d] = NumOps.Divide(sum, NumOps.FromDouble(seqLen));
                }
            }
            else if (_poolingStrategy == PoolingStrategy.Max)
            {
                for (int d = 0; d < dim; d++)
                {
                    T max = output[0, 0, d];
                    for (int s = 1; s < seqLen; s++)
                    {
                        T val = output[0, s, d];
                        if (NumOps.GreaterThan(val, max)) max = val;
                    }
                    result[d] = max;
                }
            }

            return result.Normalize();
        }

        public override Tensor<T> Predict(Tensor<T> input)
        {
            var current = input;
            foreach (var layer in Layers)
            {
                current = layer.Forward(current);
            }
            return current;
        }

        public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
        {
            // Standard backpropagation training
            var output = ForwardWithMemory(input);
            
            // Convert to vectors for loss function
            var outputVec = output.ToVector();
            var expectedVec = expectedOutput.ToVector();

            var loss = LossFunction.CalculateLoss(outputVec, expectedVec);
            var gradVec = LossFunction.CalculateDerivative(outputVec, expectedVec);
            
            // Convert gradient back to tensor with same shape as output
            var grad = Tensor<T>.FromVector(gradVec, output.Shape);
            
            Backpropagate(grad);
            // UpdateParameters should be called by the trainer/optimizer
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

        protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        {
            return new TransformerEmbeddingNetwork<T>(
                Architecture,
                _tokenizer,
                _vocabSize,
                _embeddingDimension,
                _maxSequenceLength,
                _numLayers,
                _numHeads,
                _feedForwardDim,
                _poolingStrategy,
                LossFunction);
        }

        public override ModelMetadata<T> GetModelMetadata()
        {
            return new ModelMetadata<T>
            {
                Name = "TransformerEmbeddingNetwork",
                ModelType = Enums.ModelType.Transformer,
                Description = "Customizable Transformer-based embedding network",
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

        protected override void DeserializeNetworkSpecificData(BinaryReader reader)
        {
            // Since this class is generic and depends on tokenizer which cannot be easily serialized/deserialized 
            // without more context, we might throw or implement a limited deserialization.
            // For now, we assume the constructor parameters are enough to rebuild the structure 
            // and we would load weights separately.
            // But if we are called from a factory that reads this data, we need to read it.
            // However, the fields are readonly.
            // Typically, Deserialize is used to set mutable state or check compatibility.
            
            int vocab = reader.ReadInt32();
            if (vocab != _vocabSize) throw new InvalidOperationException("Serialized vocabulary size mismatch");
            // ... check other params ...
            reader.ReadInt32(); // embeddingDim
            reader.ReadInt32(); // maxLen
            reader.ReadInt32(); // numLayers
            reader.ReadInt32(); // numHeads
            reader.ReadInt32(); // ffDim
            reader.ReadInt32(); // pooling
        }
    }
}
