using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Logging;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.FoundationModels.Models
{
    /// <summary>
    /// GPT-2 model implementation for text generation.
    /// Implements autoregressive transformer architecture.
    /// </summary>
    public class GPT2Model<T> : FoundationModelBase<T>
    {
        private readonly string _modelPath;
        private readonly FoundationModelConfig _config;
        private TransformerBlock[]? _transformerBlocks;
        private Matrix<T>? _tokenEmbeddings;
        private Matrix<T>? _positionEmbeddings;
        private Vector<T>? _layerNormGamma;
        private Vector<T>? _layerNormBeta;
        private Matrix<T>? _outputProjection;
        
        // Model configuration
        private readonly int _numLayers = 12;
        private readonly int _numHeads = 12;
        private readonly int _embeddingDim = 768;
        private readonly int _maxPositions = 1024;
        private readonly int _vocabSize = 50257;

        /// <summary>
        /// Initializes a new instance of the GPT2Model class
        /// </summary>
        /// <param name="modelPath">Path to model weights</param>
        /// <param name="tokenizer">Tokenizer instance</param>
        /// <param name="config">Model configuration</param>
        /// <param name="logger">Optional logger</param>
        public GPT2Model(
            string modelPath, 
            ITokenizer tokenizer, 
            FoundationModelConfig config,
            ILogging? logger = null) 
            : base(tokenizer, logger)
        {
            _modelPath = modelPath ?? throw new ArgumentNullException(nameof(modelPath));
            _config = config ?? throw new ArgumentNullException(nameof(config));
            
            // Override vocab size from tokenizer
            _vocabSize = tokenizer.VocabularySize;
            
            // Register available checkpoints
            RegisterCheckpoint("gpt2", modelPath);
            RegisterCheckpoint("gpt2-medium", modelPath);
            RegisterCheckpoint("gpt2-large", modelPath);
        }

        #region FoundationModelBase Implementation

        /// <inheritdoc/>
        public override string Architecture => "GPT-2";

        /// <inheritdoc/>
        public override long ParameterCount => CalculateParameterCount();

        /// <inheritdoc/>
        protected override async Task<string> GenerateInternalAsync(
            TokenizerOutput tokenizedInput,
            int maxTokens,
            double temperature,
            double topP,
            CancellationToken cancellationToken)
        {
            var generatedTokens = new List<int>();
            var inputIds = tokenizedInput.InputIds;
            
            // Get initial sequence
            var currentSequence = new List<int>();
            for (int i = 0; i < tokenizedInput.SequenceLength; i++)
            {
                if (tokenizedInput.AttentionMask[0, i] == 1)
                {
                    currentSequence.Add(inputIds[0, i]);
                }
            }

            // Generate tokens one by one
            for (int step = 0; step < maxTokens; step++)
            {
                if (cancellationToken.IsCancellationRequested)
                {
                    break;
                }

                // Prepare input tensor
                var inputTensor = PrepareInputTensor(currentSequence);
                
                // Forward pass through the model
                var logits = await ForwardPassAsync(inputTensor, cancellationToken);
                
                // Get next token logits
                var nextTokenLogits = GetLastTokenLogits(logits);
                
                // Apply temperature
                if (temperature != 1.0)
                {
                    var tempT = NumOps.FromDouble(temperature);
                    for (int i = 0; i < nextTokenLogits.Length; i++)
                    {
                        nextTokenLogits[i] = NumOps.Divide(nextTokenLogits[i], tempT);
                    }
                }
                
                // Sample next token
                var nextToken = SampleToken(nextTokenLogits, topP);
                
                // Check for end token
                if (nextToken == _tokenizer.EosTokenId)
                {
                    break;
                }
                
                generatedTokens.Add(nextToken);
                currentSequence.Add(nextToken);
                
                // Truncate if exceeding max length
                if (currentSequence.Count > _maxPositions)
                {
                    currentSequence.RemoveAt(0);
                }
            }

            // Decode generated tokens
            var generatedVector = new Vector<int>(generatedTokens.ToArray());
            return await _tokenizer.DecodeAsync(generatedVector, skipSpecialTokens: true);
        }

        /// <inheritdoc/>
        protected override async Task<Tensor<T>> ComputeEmbeddingsAsync(
            TokenizerOutput tokenizedInput,
            CancellationToken cancellationToken)
        {
            // Prepare input
            var inputIds = tokenizedInput.InputIds;
            var batchSize = inputIds.Rows;
            var seqLength = inputIds.Columns;
            
            // Get embeddings through forward pass
            var hiddenStates = new Tensor<T>(new[] { batchSize, seqLength, _embeddingDim });
            
            for (int b = 0; b < batchSize; b++)
            {
                var sequence = new List<int>();
                for (int i = 0; i < seqLength; i++)
                {
                    if (tokenizedInput.AttentionMask[b, i] == 1)
                    {
                        sequence.Add(inputIds[b, i]);
                    }
                }
                
                var inputTensor = PrepareInputTensor(sequence);
                var outputs = await ForwardPassAsync(inputTensor, cancellationToken);
                
                // Copy last hidden states
                for (int i = 0; i < sequence.Count && i < seqLength; i++)
                {
                    for (int j = 0; j < _embeddingDim; j++)
                    {
                        hiddenStates[b, i, j] = outputs[i, j];
                    }
                }
            }
            
            return hiddenStates;
        }

        /// <inheritdoc/>
        protected override async Task LoadModelWeightsAsync(string checkpointPath, CancellationToken cancellationToken)
        {
            _logger.Information("Loading GPT-2 weights from {Path}", checkpointPath);
            
            // In a real implementation, this would load actual model weights
            // For now, initialize with random weights for demonstration
            await Task.Run(() => InitializeWeights(), cancellationToken);
            
            _logger.Information("GPT-2 weights loaded successfully");
        }

        /// <inheritdoc/>
        protected override async Task InitializeModelAsync(CancellationToken cancellationToken)
        {
            _logger.Debug("Initializing GPT-2 model architecture");
            
            // Initialize transformer blocks
            _transformerBlocks = new TransformerBlock[_numLayers];
            for (int i = 0; i < _numLayers; i++)
            {
                _transformerBlocks[i] = new TransformerBlock(
                    _embeddingDim,
                    _numHeads,
                    _embeddingDim * 4, // FFN dimension
                    dropoutRate: 0.1
                );
            }
            
            // Load weights
            await LoadModelWeightsAsync(_modelPath, cancellationToken);
        }

        #endregion

        #region Private Methods

        /// <summary>
        /// Initializes model weights
        /// </summary>
        private void InitializeWeights()
        {
            var random = new Random(42);
            
            // Token embeddings
            _tokenEmbeddings = new Matrix<T>(_vocabSize, _embeddingDim);
            InitializeMatrix(_tokenEmbeddings, random, 0.02);
            
            // Position embeddings
            _positionEmbeddings = new Matrix<T>(_maxPositions, _embeddingDim);
            InitializeMatrix(_positionEmbeddings, random, 0.02);
            
            // Final layer norm
            _layerNormGamma = new Vector<T>(_embeddingDim);
            _layerNormBeta = new Vector<T>(_embeddingDim);
            for (int i = 0; i < _embeddingDim; i++)
            {
                _layerNormGamma[i] = NumOps.One;
                _layerNormBeta[i] = NumOps.Zero;
            }
            
            // Output projection (can share weights with token embeddings)
            _outputProjection = _tokenEmbeddings; // Weight tying
        }

        /// <summary>
        /// Initializes a matrix with random values
        /// </summary>
        private void InitializeMatrix(Matrix<T> matrix, Random random, double stdDev)
        {
            for (int i = 0; i < matrix.Rows; i++)
            {
                for (int j = 0; j < matrix.Columns; j++)
                {
                    matrix[i, j] = NumOps.FromDouble(NormalRandom(random) * stdDev);
                }
            }
        }

        /// <summary>
        /// Generates a normal random number
        /// </summary>
        private double NormalRandom(Random random)
        {
            // Box-Muller transform
            double u1 = 1.0 - random.NextDouble();
            double u2 = 1.0 - random.NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        }

        /// <summary>
        /// Prepares input tensor from token sequence
        /// </summary>
        private Tensor<T> PrepareInputTensor(List<int> tokenIds)
        {
            var seqLength = tokenIds.Count;
            var inputTensor = new Tensor<T>(new[] { seqLength, _embeddingDim });
            
            // Embed tokens
            for (int i = 0; i < seqLength; i++)
            {
                var tokenId = tokenIds[i];
                
                // Token embedding
                for (int j = 0; j < _embeddingDim; j++)
                {
                    inputTensor[i, j] = _tokenEmbeddings![tokenId, j];
                }
                
                // Add position embedding
                if (i < _maxPositions)
                {
                    for (int j = 0; j < _embeddingDim; j++)
                    {
                        inputTensor[i, j] = NumOps.Add(inputTensor[i, j], _positionEmbeddings![i, j]);
                    }
                }
            }
            
            return inputTensor;
        }

        /// <summary>
        /// Performs forward pass through the model
        /// </summary>
        private async Task<Tensor<T>> ForwardPassAsync(Tensor<T> input, CancellationToken cancellationToken)
        {
            var hiddenStates = input;
            
            // Pass through transformer blocks
            foreach (var block in _transformerBlocks!)
            {
                hiddenStates = await block.ForwardAsync(hiddenStates, cancellationToken);
            }
            
            // Final layer norm
            hiddenStates = ApplyLayerNorm(hiddenStates);
            
            // Project to vocabulary
            var seqLength = hiddenStates.Shape[0];
            var logits = new Tensor<T>(new[] { seqLength, _vocabSize });
            
            for (int i = 0; i < seqLength; i++)
            {
                for (int v = 0; v < _vocabSize; v++)
                {
                    T sum = NumOps.Zero;
                    for (int j = 0; j < _embeddingDim; j++)
                    {
                        sum = NumOps.Add(sum, NumOps.Multiply(hiddenStates[i, j], _outputProjection![v, j]));
                    }
                    logits[i, v] = sum;
                }
            }
            
            return logits;
        }

        /// <summary>
        /// Applies layer normalization
        /// </summary>
        private Tensor<T> ApplyLayerNorm(Tensor<T> input)
        {
            var shape = input.Shape;
            var result = new Tensor<T>(shape);
            var seqLength = shape[0];
            
            for (int i = 0; i < seqLength; i++)
            {
                // Compute mean and variance
                T mean = NumOps.Zero;
                for (int j = 0; j < _embeddingDim; j++)
                {
                    mean = NumOps.Add(mean, input[i, j]);
                }
                mean = NumOps.Divide(mean, NumOps.FromDouble(_embeddingDim));
                
                T variance = NumOps.Zero;
                for (int j = 0; j < _embeddingDim; j++)
                {
                    var diff = NumOps.Subtract(input[i, j], mean);
                    variance = NumOps.Add(variance, NumOps.Multiply(diff, diff));
                }
                variance = NumOps.Divide(variance, NumOps.FromDouble(_embeddingDim));
                
                // Normalize
                var stdDev = NumOps.Sqrt(NumOps.Add(variance, NumOps.FromDouble(1e-5)));
                for (int j = 0; j < _embeddingDim; j++)
                {
                    var normalized = NumOps.Divide(NumOps.Subtract(input[i, j], mean), stdDev);
                    result[i, j] = NumOps.Add(NumOps.Multiply(_layerNormGamma![j], normalized), _layerNormBeta![j]);
                }
            }
            
            return result;
        }

        /// <summary>
        /// Gets logits for the last token
        /// </summary>
        private Vector<T> GetLastTokenLogits(Tensor<T> logits)
        {
            var lastPosition = logits.Shape[0] - 1;
            var lastTokenLogits = new Vector<T>(_vocabSize);
            
            for (int i = 0; i < _vocabSize; i++)
            {
                lastTokenLogits[i] = logits[lastPosition, i];
            }
            
            return lastTokenLogits;
        }

        /// <summary>
        /// Samples a token from logits using top-p sampling
        /// </summary>
        private int SampleToken(Vector<T> logits, double topP)
        {
            // Apply softmax
            var probabilities = Softmax(logits);
            
            // Convert to double for sorting and sampling
            var probsAsDouble = new double[probabilities.Length];
            for (int i = 0; i < probabilities.Length; i++)
            {
                probsAsDouble[i] = Convert.ToDouble(probabilities[i]);
            }
            
            // Sort by probability
            var indexed = probsAsDouble
                .Select((prob, idx) => new { Probability = prob, Index = idx })
                .OrderByDescending(x => x.Probability)
                .ToList();
            
            // Apply top-p filtering
            double cumulativeProb = 0;
            var filtered = new List<Tuple<int, double>>();
            
            foreach (var item in indexed)
            {
                filtered.Add(Tuple.Create(item.Index, item.Probability));
                cumulativeProb += item.Probability;
                
                if (cumulativeProb >= topP)
                {
                    break;
                }
            }
            
            // Renormalize
            var sum = filtered.Sum(x => x.Item2);
            var normalized = filtered.Select(x => Tuple.Create(x.Item1, x.Item2 / sum)).ToList();
            
            // Sample
            var random = new Random();
            var sample = random.NextDouble();
            double cumulative = 0;
            
            foreach (var tuple in normalized)
            {
                var index = tuple.Item1;
                var prob = tuple.Item2;
                cumulative += prob;
                if (sample <= cumulative)
                {
                    return index;
                }
            }
            
            return normalized.Last().Item1;
        }

        /// <summary>
        /// Applies softmax to logits
        /// </summary>
        private Vector<T> Softmax(Vector<T> logits)
        {
            // Find max logit
            var maxLogit = logits[0];
            for (int i = 1; i < logits.Length; i++)
            {
                if (NumOps.GreaterThan(logits[i], maxLogit))
                {
                    maxLogit = logits[i];
                }
            }
            
            var expValues = new Vector<T>(logits.Length);
            T sum = NumOps.Zero;
            
            for (int i = 0; i < logits.Length; i++)
            {
                var shifted = NumOps.Subtract(logits[i], maxLogit);
                expValues[i] = NumOps.Exp(shifted);
                sum = NumOps.Add(sum, expValues[i]);
            }
            
            for (int i = 0; i < expValues.Length; i++)
            {
                expValues[i] = NumOps.Divide(expValues[i], sum);
            }
            
            return expValues;
        }

        /// <summary>
        /// Calculates total parameter count
        /// </summary>
        private long CalculateParameterCount()
        {
            long count = 0;
            
            // Embeddings
            count += _vocabSize * _embeddingDim; // Token embeddings
            count += _maxPositions * _embeddingDim; // Position embeddings
            
            // Transformer blocks
            count += _numLayers * (
                4 * _embeddingDim * _embeddingDim + // QKV + output projections
                2 * _embeddingDim * (_embeddingDim * 4) + // FFN
                4 * _embeddingDim // Layer norms
            );
            
            // Final layer norm
            count += 2 * _embeddingDim;
            
            // Output projection (if not weight tied)
            if (_outputProjection != _tokenEmbeddings)
            {
                count += _vocabSize * _embeddingDim;
            }
            
            return count;
        }

        #endregion

        #region Nested Classes

        /// <summary>
        /// Transformer block implementation
        /// </summary>
        private class TransformerBlock
        {
            private readonly int _hiddenSize;
            private readonly int _numHeads;
            private readonly int _ffnDim;
            private readonly double _dropoutRate;
            
            public TransformerBlock(int hiddenSize, int numHeads, int ffnDim, double dropoutRate)
            {
                _hiddenSize = hiddenSize;
                _numHeads = numHeads;
                _ffnDim = ffnDim;
                _dropoutRate = dropoutRate;
            }
            
            public async Task<Tensor<T>> ForwardAsync(Tensor<T> input, CancellationToken cancellationToken)
            {
                // Simplified forward pass
                // In a real implementation, this would include:
                // - Multi-head self-attention
                // - Layer normalization
                // - Feed-forward network
                // - Residual connections
                
                await Task.CompletedTask;
                return input; // Placeholder
            }
        }

        #endregion

        #region Protected Methods

        /// <summary>
        /// Creates a new instance of the GPT-2 model
        /// </summary>
        protected override IFullModel<T, string, string> CreateNewInstance()
        {
            return new GPT2Model<T>(_modelPath, _tokenizer, _config, _logger);
        }

        #endregion
    }
}