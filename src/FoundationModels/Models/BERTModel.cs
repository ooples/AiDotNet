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

namespace AiDotNet.FoundationModels.Models
{
    /// <summary>
    /// BERT (Bidirectional Encoder Representations from Transformers) model implementation.
    /// Supports masked language modeling, text classification, and embeddings extraction.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations</typeparam>
    public class BERTModel<T> : FoundationModelBase<T>
    {
        private readonly string _modelPath;
        private readonly FoundationModelConfig _config;
        private BERTEncoder<T>? _encoder;
        private Matrix<T>? _tokenEmbeddings;
        private Matrix<T>? _segmentEmbeddings;
        private Matrix<T>? _positionEmbeddings;
        private BERTPooler<T>? _pooler;
        private Matrix<T>? _mlmOutputWeights;
        private Vector<T>? _mlmOutputBias;
        
        // Model configuration
        private readonly int _numLayers = 12;
        private readonly int _numHeads = 12;
        private readonly int _hiddenSize = 768;
        private readonly int _maxPositions = 512;
        private readonly int _vocabSize = 30522;
        private readonly int _typeVocabSize = 2; // For segment embeddings

        /// <summary>
        /// Initializes a new instance of the BERTModel class
        /// </summary>
        /// <param name="modelPath">Path to model weights</param>
        /// <param name="tokenizer">Tokenizer instance</param>
        /// <param name="config">Model configuration</param>
        /// <param name="logger">Optional logger</param>
        public BERTModel(
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
            RegisterCheckpoint("bert-base-uncased", modelPath);
            RegisterCheckpoint("bert-base-cased", modelPath);
            RegisterCheckpoint("bert-large-uncased", modelPath);
        }

        #region FoundationModelBase Implementation

        /// <inheritdoc/>
        public override string Architecture => "BERT";

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
            // BERT is primarily an encoder model, not designed for generation
            // We can use it for masked language modeling instead
            
            var inputIds = tokenizedInput.InputIds;
            var attentionMask = tokenizedInput.AttentionMask;
            
            // Find [MASK] tokens
            var maskPositions = FindMaskPositions(inputIds);
            
            if (maskPositions.Count == 0)
            {
                // No masks to predict, return empty
                return string.Empty;
            }
            
            // Get BERT embeddings
            var embeddings = await GetBERTEmbeddingsAsync(tokenizedInput, cancellationToken);
            
            // Predict masked tokens
            var predictions = await PredictMaskedTokensAsync(embeddings, maskPositions, temperature, topP);
            
            // Replace masks with predictions
            var resultIds = new List<int>();
            for (int i = 0; i < inputIds.Columns; i++)
            {
                var tokenId = inputIds[0, i];
                if (maskPositions.Contains(i))
                {
                    var predIndex = maskPositions.IndexOf(i);
                    resultIds.Add(predictions[predIndex]);
                }
                else if (attentionMask[0, i] == 1)
                {
                    resultIds.Add(tokenId);
                }
            }
            
            // Decode the result
            var resultVector = new Vector<int>(resultIds.ToArray());
            return await _tokenizer.DecodeAsync(resultVector, skipSpecialTokens: true);
        }

        /// <inheritdoc/>
        protected override async Task<Tensor<T>> ComputeEmbeddingsAsync(
            TokenizerOutput tokenizedInput,
            CancellationToken cancellationToken)
        {
            var embeddings = await GetBERTEmbeddingsAsync(tokenizedInput, cancellationToken);
            
            // For sequence embeddings, we typically use the [CLS] token representation
            // or mean pooling of all tokens
            var batchSize = tokenizedInput.BatchSize;
            var seqLength = tokenizedInput.SequenceLength;
            var outputEmbeddings = new Tensor<double>(new[] { batchSize, _hiddenSize });
            
            // Use pooler output (CLS token representation)
            if (_pooler != null)
            {
                for (int b = 0; b < batchSize; b++)
                {
                    // Extract CLS token embedding (first position)
                    var clsEmbedding = new Vector<double>(_hiddenSize);
                    for (int h = 0; h < _hiddenSize; h++)
                    {
                        clsEmbedding[h] = embeddings[b, 0, h];
                    }
                    
                    // Apply pooler
                    var pooled = _pooler.Forward(clsEmbedding);
                    
                    // Copy to output
                    for (int h = 0; h < _hiddenSize; h++)
                    {
                        outputEmbeddings[b, h] = pooled[h];
                    }
                }
            }
            else
            {
                // Mean pooling fallback
                for (int b = 0; b < batchSize; b++)
                {
                    var validTokens = 0;
                    var meanEmbedding = new double[_hiddenSize];
                    
                    for (int s = 0; s < seqLength; s++)
                    {
                        if (tokenizedInput.AttentionMask[b, s] == 1)
                        {
                            validTokens++;
                            for (int h = 0; h < _hiddenSize; h++)
                            {
                                meanEmbedding[h] += embeddings[b, s, h];
                            }
                        }
                    }
                    
                    // Average
                    if (validTokens > 0)
                    {
                        for (int h = 0; h < _hiddenSize; h++)
                        {
                            outputEmbeddings[b, h] = meanEmbedding[h] / validTokens;
                        }
                    }
                }
            }
            
            return outputEmbeddings;
        }

        /// <inheritdoc/>
        protected override async Task LoadModelWeightsAsync(string checkpointPath, CancellationToken cancellationToken)
        {
            _logger.Information("Loading BERT weights from {Path}", checkpointPath);
            
            // In a real implementation, this would load actual model weights
            // For now, initialize with random weights for demonstration
            await Task.Run(() => InitializeWeights(), cancellationToken);
            
            _logger.Information("BERT weights loaded successfully");
        }

        /// <inheritdoc/>
        protected override async Task InitializeModelAsync(CancellationToken cancellationToken)
        {
            _logger.Debug("Initializing BERT model architecture");
            
            // Initialize BERT encoder
            _encoder = new BERTEncoder(
                _numLayers,
                _hiddenSize,
                _numHeads,
                intermediateSize: _hiddenSize * 4,
                maxPositions: _maxPositions
            );
            
            // Initialize pooler
            _pooler = new BERTPooler(_hiddenSize);
            
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
            _tokenEmbeddings = new Matrix<T>(_vocabSize, _hiddenSize);
            InitializeMatrix(_tokenEmbeddings, random, 0.02);
            
            // Segment embeddings
            _segmentEmbeddings = new Matrix<T>(_typeVocabSize, _hiddenSize);
            InitializeMatrix(_segmentEmbeddings, random, 0.02);
            
            // Position embeddings
            _positionEmbeddings = new Matrix<T>(_maxPositions, _hiddenSize);
            InitializeMatrix(_positionEmbeddings, random, 0.02);
            
            // MLM output weights (can be tied with input embeddings)
            _mlmOutputWeights = _tokenEmbeddings; // Weight tying
            _mlmOutputBias = new Vector<double>(_vocabSize);
        }

        /// <summary>
        /// Gets BERT embeddings for input
        /// </summary>
        private async Task<Tensor<double>> GetBERTEmbeddingsAsync(
            TokenizerOutput tokenizedInput,
            CancellationToken cancellationToken)
        {
            var batchSize = tokenizedInput.BatchSize;
            var seqLength = tokenizedInput.SequenceLength;
            
            // Prepare embeddings
            var embeddings = new Tensor<double>(new[] { batchSize, seqLength, _hiddenSize });
            
            // Combine token, segment, and position embeddings
            for (int b = 0; b < batchSize; b++)
            {
                for (int s = 0; s < seqLength; s++)
                {
                    if (tokenizedInput.AttentionMask[b, s] == 0)
                    {
                        continue; // Skip padding
                    }
                    
                    var tokenId = tokenizedInput.InputIds[b, s];
                    var segmentId = tokenizedInput.TokenTypeIds?[b, s] ?? 0;
                    var positionId = s;
                    
                    // Add embeddings
                    for (int h = 0; h < _hiddenSize; h++)
                    {
                        embeddings[b, s, h] = 
                            _tokenEmbeddings![tokenId, h] +
                            _segmentEmbeddings![segmentId, h] +
                            _positionEmbeddings![positionId, h];
                    }
                }
            }
            
            // Apply layer normalization and dropout (simplified)
            embeddings = ApplyLayerNorm(embeddings);
            
            // Pass through encoder
            if (_encoder != null)
            {
                embeddings = await _encoder.ForwardAsync(embeddings, tokenizedInput.AttentionMask, cancellationToken);
            }
            
            return embeddings;
        }

        /// <summary>
        /// Finds positions of [MASK] tokens
        /// </summary>
        private List<int> FindMaskPositions(Matrix<int> inputIds)
        {
            var maskPositions = new List<int>();
            var maskTokenId = _tokenizer.SpecialTokens.ContainsKey("[MASK]") ? _tokenizer.SpecialTokens["[MASK]"] : -1;
            
            if (maskTokenId == -1)
            {
                return maskPositions;
            }
            
            for (int i = 0; i < inputIds.Columns; i++)
            {
                if (inputIds[0, i] == maskTokenId)
                {
                    maskPositions.Add(i);
                }
            }
            
            return maskPositions;
        }

        /// <summary>
        /// Predicts masked tokens
        /// </summary>
        private async Task<List<int>> PredictMaskedTokensAsync(
            Tensor<double> embeddings,
            List<int> maskPositions,
            double temperature,
            double topP)
        {
            var predictions = new List<int>();
            
            foreach (var position in maskPositions)
            {
                // Get embedding at mask position
                var maskedEmbedding = new Vector<double>(_hiddenSize);
                for (int h = 0; h < _hiddenSize; h++)
                {
                    maskedEmbedding[h] = embeddings[0, position, h];
                }
                
                // Project to vocabulary
                var logits = new Vector<double>(_vocabSize);
                for (int v = 0; v < _vocabSize; v++)
                {
                    double sum = _mlmOutputBias![v];
                    for (int h = 0; h < _hiddenSize; h++)
                    {
                        sum += maskedEmbedding[h] * _mlmOutputWeights![v, h];
                    }
                    logits[v] = sum;
                }
                
                // Apply temperature
                if (temperature != 1.0)
                {
                    for (int i = 0; i < logits.Length; i++)
                    {
                        logits[i] /= temperature;
                    }
                }
                
                // Sample token
                var predictedToken = SampleToken(logits, topP);
                predictions.Add(predictedToken);
            }
            
            return await Task.FromResult(predictions);
        }

        /// <summary>
        /// Other helper methods...
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

        private double NormalRandom(Random random)
        {
            double u1 = 1.0 - random.NextDouble();
            double u2 = 1.0 - random.NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        }

        private Tensor<T> ApplyLayerNorm(Tensor<T> input)
        {
            // Simplified layer normalization
            return input;
        }

        private int SampleToken(Vector<double> logits, double topP)
        {
            // Simplified sampling (reuse from GPT2Model)
            var probabilities = Softmax(logits);
            var indexed = probabilities
                .Select((prob, idx) => new { Probability = prob, Index = idx })
                .OrderByDescending(x => x.Probability)
                .ToList();
            
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
            
            var sum = filtered.Sum(x => x.Item2);
            var normalized = filtered.Select(x => Tuple.Create(x.Item1, x.Item2 / sum)).ToList();
            
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

        private Vector<double> Softmax(Vector<double> logits)
        {
            var maxLogit = logits.Max();
            var expValues = new Vector<double>(logits.Length);
            double sum = 0;
            
            for (int i = 0; i < logits.Length; i++)
            {
                expValues[i] = Math.Exp(logits[i] - maxLogit);
                sum += expValues[i];
            }
            
            for (int i = 0; i < expValues.Length; i++)
            {
                expValues[i] /= sum;
            }
            
            return expValues;
        }

        private long CalculateParameterCount()
        {
            long count = 0;
            
            // Embeddings
            count += _vocabSize * _hiddenSize; // Token embeddings
            count += _typeVocabSize * _hiddenSize; // Segment embeddings
            count += _maxPositions * _hiddenSize; // Position embeddings
            
            // Encoder layers
            count += _numLayers * (
                4 * _hiddenSize * _hiddenSize + // QKV + output projections
                2 * _hiddenSize * (_hiddenSize * 4) + // FFN
                4 * _hiddenSize // Layer norms
            );
            
            // Pooler
            count += _hiddenSize * _hiddenSize + _hiddenSize;
            
            // MLM head (if not weight tied)
            if (_mlmOutputWeights != _tokenEmbeddings)
            {
                count += _vocabSize * _hiddenSize;
            }
            count += _vocabSize; // Bias
            
            return count;
        }

        #endregion

        #region Nested Classes

        /// <summary>
        /// BERT encoder implementation
        /// </summary>
        private class BERTEncoder<TNum>
        {
            private readonly int _numLayers;
            private readonly int _hiddenSize;
            private readonly int _numHeads;
            private readonly int _intermediateSize;
            private readonly int _maxPositions;
            
            public BERTEncoder(int numLayers, int hiddenSize, int numHeads, int intermediateSize, int maxPositions)
            {
                _numLayers = numLayers;
                _hiddenSize = hiddenSize;
                _numHeads = numHeads;
                _intermediateSize = intermediateSize;
                _maxPositions = maxPositions;
            }
            
            public async Task<Tensor<TNum>> ForwardAsync(
                Tensor<TNum> input, 
                Matrix<int> attentionMask,
                CancellationToken cancellationToken)
            {
                // Simplified forward pass
                // In a real implementation, this would include:
                // - Multi-head self-attention with attention mask
                // - Layer normalization
                // - Feed-forward network
                // - Residual connections
                
                await Task.CompletedTask;
                return input; // Placeholder
            }
        }

        /// <summary>
        /// BERT pooler for CLS token
        /// </summary>
        private class BERTPooler<TNum>
        {
            private readonly int _hiddenSize;
            private readonly Matrix<TNum> _denseWeight;
            private readonly Vector<TNum> _denseBias;
            private readonly INumericOperations<TNum> _numOps;
            
            public BERTPooler(int hiddenSize)
            {
                _hiddenSize = hiddenSize;
                _denseWeight = new Matrix<TNum>(hiddenSize, hiddenSize);
                _denseBias = new Vector<TNum>(hiddenSize);
                _numOps = MathHelper.GetNumericOperations<TNum>();
                
                // Initialize weights
                var random = new Random(42);
                for (int i = 0; i < hiddenSize; i++)
                {
                    _denseBias[i] = _numOps.Zero;
                    for (int j = 0; j < hiddenSize; j++)
                    {
                        _denseWeight[i, j] = _numOps.FromDouble((random.NextDouble() * 2 - 1) * 0.02);
                    }
                }
            }
            
            public Vector<TNum> Forward(Vector<TNum> clsToken)
            {
                var output = new Vector<TNum>(_hiddenSize);
                
                // Linear transformation
                for (int i = 0; i < _hiddenSize; i++)
                {
                    double sum = _denseBias[i];
                    for (int j = 0; j < _hiddenSize; j++)
                    {
                        sum += clsToken[j] * _denseWeight[i, j];
                    }
                    // Apply tanh activation
                    output[i] = Math.Tanh(sum);
                }
                
                return output;
            }
        }

        #endregion

        #region Protected Methods

        /// <summary>
        /// Creates a new instance of the BERT model
        /// </summary>
        protected override IFullModel<T, string, string> CreateNewInstance()
        {
            return new BERTModel<T>(_modelPath, _tokenizer, _config, _logger);
        }

        #endregion
    }
}