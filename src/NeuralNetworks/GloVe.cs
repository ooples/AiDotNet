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
    /// GloVe (Global Vectors for Word Representation) model implementation.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class GloVe<T> : NeuralNetworkBase<T>, IEmbeddingModel<T>
    {
        private readonly int _vocabSize;
        private readonly int _embeddingDimension;
        private readonly ITokenizer _tokenizer;
        
        // GloVe uses two matrices W and W_tilde
        private EmbeddingLayer<T> _mainEmbeddings;
        private EmbeddingLayer<T> _contextEmbeddings;
        
        // And two bias vectors
        private EmbeddingLayer<T> _mainBiases;
        private EmbeddingLayer<T> _contextBiases;

        public int EmbeddingDimension => _embeddingDimension;
        public int MaxTokens => 10000;

        public GloVe(
            NeuralNetworkArchitecture<T> architecture,
            ITokenizer tokenizer,
            int vocabSize,
            int embeddingDimension = 100,
            ILossFunction<T>? lossFunction = null)
            : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>())
        {
            _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
            _vocabSize = vocabSize;
            _embeddingDimension = embeddingDimension;

            // Initialize to null!, they will be set in InitializeLayers called by base constructor or explicit call
            _mainEmbeddings = null!;
            _contextEmbeddings = null!;
            _mainBiases = null!;
            _contextBiases = null!;

            InitializeLayers();
        }

        protected override void InitializeLayers()
        {
            ClearLayers();
            // Main embeddings W
            _mainEmbeddings = new EmbeddingLayer<T>(_vocabSize, _embeddingDimension);
            AddLayerToCollection(_mainEmbeddings);

            // Context embeddings W_tilde
            _contextEmbeddings = new EmbeddingLayer<T>(_vocabSize, _embeddingDimension);
            AddLayerToCollection(_contextEmbeddings);

            // Biases (vocabSize x 1)
            _mainBiases = new EmbeddingLayer<T>(_vocabSize, 1);
            AddLayerToCollection(_mainBiases);

            _contextBiases = new EmbeddingLayer<T>(_vocabSize, 1);
            AddLayerToCollection(_contextBiases);
        }

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
            // Standard prediction uses main + context embeddings
            var w = _mainEmbeddings.Forward(input);
            var w_tilde = _contextEmbeddings.Forward(input);
            
            // Result is W + W_tilde
            // Simplified: return W + W_tilde
            var result = new Tensor<T>(w.Shape);
            
            // Assuming W and W_tilde have same shape [1, dim]
            for (int i = 0; i < w.Length; i++)
            {
                // Simple element-wise addition using flat indices
                T val = NumOps.Add(w.GetFlat(i), w_tilde.GetFlat(i));
                result.SetFlat(i, val);
            }
            
            return result; 
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
                    
                    // GloVe often uses W + W_tilde for final embeddings
                    var w = _mainEmbeddings.Forward(inputTensor);
                    var w_tilde = _contextEmbeddings.Forward(inputTensor);
                    
                    for (int i = 0; i < _embeddingDimension; i++)
                    {
                        var val = NumOps.Add(w[0, i], w_tilde[0, i]);
                        sumVector[i] = NumOps.Add(sumVector[i], val);
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
    }
}
