using System.Threading.Tasks;
using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Interpretability;
using AiDotNet.Enums;
using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks
{
    /// <summary>
    /// Adapter that bridges tokenizers with neural network models to enable text processing.
    /// Allows using traditional neural networks (like Transformer) for NLP tasks.
    /// </summary>
    /// <typeparam name="T">Numeric type for the neural network</typeparam>
    public class TokenizerAdapter<T> : IFullModel<T, string, string> 
        where T : unmanaged, INumericOperations<T>
    {
        private readonly INeuralNetworkModel<T> _neuralNetwork;
        private readonly ITokenizer _tokenizer;
        private readonly int _embeddingDim;
        private readonly bool _usePositionalEncoding;
        private Matrix<T>? _embeddingMatrix;
        private Matrix<T>? _positionalEncodings;

        /// <summary>
        /// Initializes a new instance of the TokenizerAdapter class
        /// </summary>
        /// <param name="neuralNetwork">The neural network model to adapt</param>
        /// <param name="tokenizer">The tokenizer to use for text processing</param>
        /// <param name="embeddingDim">Dimension of token embeddings</param>
        /// <param name="usePositionalEncoding">Whether to add positional encodings</param>
        public TokenizerAdapter(
            INeuralNetworkModel<T> neuralNetwork,
            ITokenizer tokenizer,
            int embeddingDim = 512,
            bool usePositionalEncoding = true)
        {
            _neuralNetwork = neuralNetwork ?? throw new ArgumentNullException(nameof(neuralNetwork));
            _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
            _embeddingDim = embeddingDim;
            _usePositionalEncoding = usePositionalEncoding;
            
            InitializeEmbeddings();
        }

        /// <summary>
        /// Initializes embedding matrix and positional encodings
        /// </summary>
        private void InitializeEmbeddings()
        {
            // Initialize embedding matrix with random values
            var vocabSize = _tokenizer.VocabularySize;
            _embeddingMatrix = new Matrix<T>(vocabSize, _embeddingDim);
            
            // Xavier initialization
            var scale = Math.Sqrt(2.0 / _embeddingDim);
            var random = new Random();
            
            for (int i = 0; i < vocabSize; i++)
            {
                for (int j = 0; j < _embeddingDim; j++)
                {
                    double value = (random.NextDouble() * 2 - 1) * scale;
                    _embeddingMatrix[i, j] = T.CreateChecked(value);
                }
            }

            // Initialize positional encodings if needed
            if (_usePositionalEncoding)
            {
                InitializePositionalEncodings();
            }
        }

        /// <summary>
        /// Initializes sinusoidal positional encodings
        /// </summary>
        private void InitializePositionalEncodings()
        {
            int maxLength = _tokenizer.MaxSequenceLength;
            _positionalEncodings = new Matrix<T>(maxLength, _embeddingDim);
            
            for (int pos = 0; pos < maxLength; pos++)
            {
                for (int i = 0; i < _embeddingDim; i++)
                {
                    double angle = pos / Math.Pow(10000, (2.0 * i) / _embeddingDim);
                    
                    if (i % 2 == 0)
                    {
                        _positionalEncodings[pos, i] = T.CreateChecked(Math.Sin(angle));
                    }
                    else
                    {
                        _positionalEncodings[pos, i] = T.CreateChecked(Math.Cos(angle));
                    }
                }
            }
        }

        /// <summary>
        /// Trains the model on text input/output pairs
        /// </summary>
        public void Train(string input, string expectedOutput)
        {
            // Initialize tokenizer synchronously if needed
            if (!_tokenizer.IsInitialized)
            {
                _tokenizer.InitializeAsync().GetAwaiter().GetResult();
            }

            // Tokenize input and output synchronously
            var inputTokens = _tokenizer.EncodeAsync(input).GetAwaiter().GetResult();
            var outputTokens = _tokenizer.EncodeAsync(expectedOutput).GetAwaiter().GetResult();
            
            // Convert to embeddings
            var inputEmbeddings = GetEmbeddings(inputTokens);
            var outputEmbeddings = GetEmbeddings(outputTokens);
            
            // Add positional encodings if enabled
            if (_usePositionalEncoding)
            {
                AddPositionalEncodings(inputEmbeddings);
                AddPositionalEncodings(outputEmbeddings);
            }
            
            // Convert to tensors for neural network
            var inputTensor = ConvertToTensor(inputEmbeddings);
            var outputTensor = ConvertToTensor(outputEmbeddings);
            
            // Train the neural network
            _neuralNetwork.Train(inputTensor, outputTensor);
        }

        /// <summary>
        /// Predicts output text given input text
        /// </summary>
        public string Predict(string input)
        {
            // Initialize tokenizer synchronously if needed
            if (!_tokenizer.IsInitialized)
            {
                _tokenizer.InitializeAsync().GetAwaiter().GetResult();
            }

            // Tokenize input synchronously
            var inputTokens = _tokenizer.EncodeAsync(input).GetAwaiter().GetResult();
            
            // Convert to embeddings
            var inputEmbeddings = GetEmbeddings(inputTokens);
            
            // Add positional encodings
            if (_usePositionalEncoding)
            {
                AddPositionalEncodings(inputEmbeddings);
            }
            
            // Convert to tensor
            var inputTensor = ConvertToTensor(inputEmbeddings);
            
            // Get neural network prediction
            var outputTensor = _neuralNetwork.Predict(inputTensor);
            
            // Convert output tensor back to tokens
            var outputTokens = ConvertTensorToTokens(outputTensor);
            
            // Decode tokens to text synchronously
            var outputText = _tokenizer.DecodeAsync(outputTokens).GetAwaiter().GetResult();
            
            return outputText;
        }

        /// <summary>
        /// Gets embeddings for token IDs
        /// </summary>
        private Matrix<T> GetEmbeddings(Vector<int> tokenIds)
        {
            var seqLength = tokenIds.Count;
            var embeddings = new Matrix<T>(seqLength, _embeddingDim);
            
            for (int i = 0; i < seqLength; i++)
            {
                int tokenId = tokenIds[i];
                for (int j = 0; j < _embeddingDim; j++)
                {
                    embeddings[i, j] = _embeddingMatrix![tokenId, j];
                }
            }
            
            return embeddings;
        }

        /// <summary>
        /// Adds positional encodings to embeddings
        /// </summary>
        private void AddPositionalEncodings(Matrix<T> embeddings)
        {
            int seqLength = embeddings.Rows;
            
            for (int i = 0; i < seqLength; i++)
            {
                for (int j = 0; j < _embeddingDim; j++)
                {
                    embeddings[i, j] = T.Add(embeddings[i, j], _positionalEncodings![i, j]);
                }
            }
        }

        /// <summary>
        /// Converts embedding matrix to tensor
        /// </summary>
        private Tensor<T> ConvertToTensor(Matrix<T> embeddings)
        {
            // Create a 3D tensor: [batch_size=1, sequence_length, embedding_dim]
            var tensor = new Tensor<T>(new[] { 1, embeddings.Rows, embeddings.Columns });
            
            for (int i = 0; i < embeddings.Rows; i++)
            {
                for (int j = 0; j < embeddings.Columns; j++)
                {
                    tensor[0, i, j] = embeddings[i, j];
                }
            }
            
            return tensor;
        }

        /// <summary>
        /// Converts output tensor to token IDs
        /// </summary>
        private Vector<int> ConvertTensorToTokens(Tensor<T> outputTensor)
        {
            // Assuming the output tensor contains logits over vocabulary
            // Shape: [batch_size, sequence_length, vocab_size]
            
            var seqLength = outputTensor.Shape[1];
            var vocabSize = _tokenizer.VocabularySize;
            var tokens = new int[seqLength];
            
            for (int i = 0; i < seqLength; i++)
            {
                // Find argmax for each position
                T maxValue = outputTensor[0, i, 0];
                int maxIndex = 0;
                
                for (int j = 1; j < vocabSize && j < outputTensor.Shape[2]; j++)
                {
                    if (T.IsGreaterThan(outputTensor[0, i, j], maxValue))
                    {
                        maxValue = outputTensor[0, i, j];
                        maxIndex = j;
                    }
                }
                
                tokens[i] = maxIndex;
            }
            
            return new Vector<int>(tokens);
        }

        #region IFullModel Implementation

        public ModelMetadata<T> GetModelMetadata()
        {
            return _neuralNetwork.GetModelMetadata();
        }

        public byte[] Serialize()
        {
            // Serialize both the neural network and embedding matrix
            // In a real implementation, this would be more sophisticated
            return _neuralNetwork.Serialize();
        }

        public void Deserialize(byte[] data)
        {
            _neuralNetwork.Deserialize(data);
        }

        public Vector<T> GetParameters()
        {
            // Combine neural network parameters with embedding parameters
            var nnParams = _neuralNetwork.GetParameters();
            var embeddingParams = _embeddingMatrix!.Data;
            
            var combined = new T[nnParams.Count + embeddingParams.Length];
            for (int i = 0; i < nnParams.Count; i++)
            {
                combined[i] = nnParams[i];
            }
            for (int i = 0; i < embeddingParams.Length; i++)
            {
                combined[nnParams.Count + i] = embeddingParams[i];
            }
            
            return new Vector<T>(combined);
        }

        public void SetParameters(Vector<T> parameters)
        {
            // Split parameters between neural network and embeddings
            var nnParamCount = _neuralNetwork.GetParameters().Count;
            var nnParams = new T[nnParamCount];
            
            for (int i = 0; i < nnParamCount; i++)
            {
                nnParams[i] = parameters[i];
            }
            
            _neuralNetwork.SetParameters(new Vector<T>(nnParams));
            
            // Update embedding matrix
            int embeddingIndex = 0;
            for (int i = 0; i < _embeddingMatrix!.Rows; i++)
            {
                for (int j = 0; j < _embeddingMatrix.Columns; j++)
                {
                    if (nnParamCount + embeddingIndex < parameters.Count)
                    {
                        _embeddingMatrix[i, j] = parameters[nnParamCount + embeddingIndex];
                        embeddingIndex++;
                    }
                }
            }
        }

        public IFullModel<T, string, string> WithParameters(Vector<T> parameters)
        {
            var clone = Clone();
            clone.SetParameters(parameters);
            return clone;
        }

        public IEnumerable<int> GetActiveFeatureIndices()
        {
            return _neuralNetwork.GetActiveFeatureIndices();
        }

        public bool IsFeatureUsed(int featureIndex)
        {
            return _neuralNetwork.IsFeatureUsed(featureIndex);
        }

        public void SetActiveFeatureIndices(IEnumerable<int> indices)
        {
            _neuralNetwork.SetActiveFeatureIndices(indices);
        }

        public IFullModel<T, string, string> Clone()
        {
            var clonedNN = (INeuralNetworkModel<T>)_neuralNetwork.Clone();
            var adapter = new TokenizerAdapter<T>(clonedNN, _tokenizer, _embeddingDim, _usePositionalEncoding);
            
            // Copy embedding matrix
            adapter._embeddingMatrix = _embeddingMatrix!.Clone();
            
            return adapter;
        }

        public IFullModel<T, string, string> DeepCopy()
        {
            // Deep copy the neural network
            var clonedNN = (INeuralNetworkModel<T>)_neuralNetwork.DeepCopy();
            
            // Create a new adapter with the cloned neural network
            var adapter = new TokenizerAdapter<T>(clonedNN, _tokenizer, _embeddingDim, _usePositionalEncoding);
            
            // Deep copy embedding matrix
            adapter._embeddingMatrix = _embeddingMatrix!.Clone();
            
            // Deep copy positional encodings if they exist
            if (_positionalEncodings != null)
            {
                adapter._positionalEncodings = _positionalEncodings.Clone();
            }
            
            return adapter;
        }

        #endregion

        /// <summary>
        /// Creates a text generation function using the adapted model
        /// </summary>
        public string GenerateText(string prompt, int maxLength = 100, double temperature = 1.0)
        {
            var generatedTokens = new List<int>();
            var promptTokens = _tokenizer.EncodeAsync(prompt, addSpecialTokens: false).GetAwaiter().GetResult();
            
            // Start with prompt tokens
            for (int i = 0; i < promptTokens.Count; i++)
            {
                generatedTokens.Add(promptTokens[i]);
            }
            
            // Generate tokens one by one
            for (int i = 0; i < maxLength; i++)
            {
                // Create input from generated tokens
                var inputTokens = new Vector<int>(generatedTokens.ToArray());
                var embeddings = GetEmbeddings(inputTokens);
                
                if (_usePositionalEncoding)
                {
                    AddPositionalEncodings(embeddings);
                }
                
                var inputTensor = ConvertToTensor(embeddings);
                var outputTensor = _neuralNetwork.Predict(inputTensor);
                
                // Get the last token's logits
                var lastPos = outputTensor.Shape[1] - 1;
                var vocabSize = Math.Min(_tokenizer.VocabularySize, outputTensor.Shape[2]);
                
                // Apply temperature
                var logits = new double[vocabSize];
                for (int j = 0; j < vocabSize; j++)
                {
                    logits[j] = Convert.ToDouble(outputTensor[0, lastPos, j]) / temperature;
                }
                
                // Softmax
                var maxLogit = logits.Max();
                var expValues = logits.Select(l => Math.Exp(l - maxLogit)).ToArray();
                var sumExp = expValues.Sum();
                var probs = expValues.Select(e => e / sumExp).ToArray();
                
                // Sample from distribution
                var random = new Random();
                var sample = random.NextDouble();
                var cumSum = 0.0;
                int nextToken = 0;
                
                for (int j = 0; j < probs.Length; j++)
                {
                    cumSum += probs[j];
                    if (sample < cumSum)
                    {
                        nextToken = j;
                        break;
                    }
                }
                
                // Check for end token
                if (nextToken == _tokenizer.EosTokenId)
                {
                    break;
                }
                
                generatedTokens.Add(nextToken);
            }
            
            // Decode all tokens
            var allTokens = new Vector<int>(generatedTokens.ToArray());
            return _tokenizer.DecodeAsync(allTokens).GetAwaiter().GetResult();
        }

        #region IInterpretableModel Implementation

        protected readonly HashSet<InterpretationMethod> _enabledMethods = new();
        protected Vector<int> _sensitiveFeatures;
        protected readonly List<FairnessMetric> _fairnessMetrics = new();
        protected IModel<string, string, ModelMetadata<T>> _baseModel;

        /// <summary>
        /// Gets the global feature importance across all predictions.
        /// </summary>
        public virtual async Task<Dictionary<int, T>> GetGlobalFeatureImportanceAsync()
        {
        return await InterpretableModelHelper.GetGlobalFeatureImportanceAsync(this, _enabledMethods);
        }

        /// <summary>
        /// Gets the local feature importance for a specific input.
        /// </summary>
        public virtual async Task<Dictionary<int, T>> GetLocalFeatureImportanceAsync(string input)
        {
        return await InterpretableModelHelper.GetLocalFeatureImportanceAsync(this, _enabledMethods, input);
        }

        /// <summary>
        /// Gets SHAP values for the given inputs.
        /// </summary>
        public virtual async Task<Matrix<T>> GetShapValuesAsync(string inputs)
        {
        return await InterpretableModelHelper.GetShapValuesAsync(this, _enabledMethods);
        }

        /// <summary>
        /// Gets LIME explanation for a specific input.
        /// </summary>
        public virtual async Task<LimeExplanation<T>> GetLimeExplanationAsync(string input, int numFeatures = 10)
        {
        return await InterpretableModelHelper.GetLimeExplanationAsync<T>(_enabledMethods, numFeatures);
        }

        /// <summary>
        /// Gets partial dependence data for specified features.
        /// </summary>
        public virtual async Task<PartialDependenceData<T>> GetPartialDependenceAsync(Vector<int> featureIndices, int gridResolution = 20)
        {
        return await InterpretableModelHelper.GetPartialDependenceAsync<T>(_enabledMethods, featureIndices, gridResolution);
        }

        /// <summary>
        /// Gets counterfactual explanation for a given input and desired output.
        /// </summary>
        public virtual async Task<CounterfactualExplanation<T>> GetCounterfactualAsync(string input, string desiredOutput, int maxChanges = 5)
        {
        return await InterpretableModelHelper.GetCounterfactualAsync<T>(_enabledMethods, maxChanges);
        }

        /// <summary>
        /// Gets model-specific interpretability information.
        /// </summary>
        public virtual async Task<Dictionary<string, object>> GetModelSpecificInterpretabilityAsync()
        {
        return await InterpretableModelHelper.GetModelSpecificInterpretabilityAsync(this);
        }

        /// <summary>
        /// Generates a text explanation for a prediction.
        /// </summary>
        public virtual async Task<string> GenerateTextExplanationAsync(string input, string prediction)
        {
        return await InterpretableModelHelper.GenerateTextExplanationAsync(this, input, prediction);
        }

        /// <summary>
        /// Gets feature interaction effects between two features.
        /// </summary>
        public virtual async Task<T> GetFeatureInteractionAsync(int feature1Index, int feature2Index)
        {
        return await InterpretableModelHelper.GetFeatureInteractionAsync<T>(_enabledMethods, feature1Index, feature2Index);
        }

        /// <summary>
        /// Validates fairness metrics for the given inputs.
        /// </summary>
        public virtual async Task<FairnessMetrics<T>> ValidateFairnessAsync(string inputs, int sensitiveFeatureIndex)
        {
        return await InterpretableModelHelper.ValidateFairnessAsync<T>(_fairnessMetrics);
        }

        /// <summary>
        /// Gets anchor explanation for a given input.
        /// </summary>
        public virtual async Task<AnchorExplanation<T>> GetAnchorExplanationAsync(string input, T threshold)
        {
        return await InterpretableModelHelper.GetAnchorExplanationAsync(_enabledMethods, threshold);
        }

        /// <summary>
        /// Sets the base model for interpretability analysis.
        /// </summary>
        public virtual void SetBaseModel(IModel<string, string, ModelMetadata<T>> model)
        {
        _baseModel = model ?? throw new ArgumentNullException(nameof(model));
        }

        /// <summary>
        /// Enables specific interpretation methods.
        /// </summary>
        public virtual void EnableMethod(params InterpretationMethod[] methods)
        {
        foreach (var method in methods)
        {
            _enabledMethods.Add(method);
        }
        }

        /// <summary>
        /// Configures fairness evaluation settings.
        /// </summary>
        public virtual void ConfigureFairness(Vector<int> sensitiveFeatures, params FairnessMetric[] fairnessMetrics)
        {
        _sensitiveFeatures = sensitiveFeatures ?? throw new ArgumentNullException(nameof(sensitiveFeatures));
        _fairnessMetrics.Clear();
        _fairnessMetrics.AddRange(fairnessMetrics);
        }

        #endregion
    }
}