using System.Threading.Tasks;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Statistics;
using AiDotNet.Enums;
using AiDotNet.MultimodalAI.Encoders;
using AiDotNet.Models;
using AiDotNet.Models.Inputs;
using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Interpretability;
using AiDotNet.Helpers;

namespace AiDotNet.MultimodalAI
{
    /// <summary>
    /// Base class for multimodal AI models
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations</typeparam>
    public abstract class MultimodalModelBase<T> : IMultimodalModel<T>
    {
        protected readonly Dictionary<string, IModalityEncoder<T>> _modalityEncoders;
        protected readonly string _fusionStrategy;
        protected Matrix<T>? _crossModalityAttention;
        protected bool _isTrained;
        protected int _fusedDimension;
        protected readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

        /// <summary>
        /// Gets the supported modalities
        /// </summary>
        public IReadOnlyList<string> SupportedModalities => [.. _modalityEncoders.Keys];

        /// <summary>
        /// Gets the fusion strategy used by the model
        /// </summary>
        public string FusionStrategy => _fusionStrategy;

        /// <summary>
        /// Gets whether the model is trained
        /// </summary>
        public bool IsTrained => _isTrained;

        /// <summary>
        /// Gets or sets the name of the model
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        /// Initializes a new instance of MultimodalModelBase
        /// </summary>
        /// <param name="fusionStrategy">The fusion strategy to use</param>
        /// <param name="fusedDimension">Dimension of the fused representation</param>
        /// <param name="numericOps">Numeric operations for type T</param>
        protected MultimodalModelBase(string fusionStrategy, int fusedDimension)
        {
            _fusionStrategy = fusionStrategy;
            _fusedDimension = fusedDimension;
            _modalityEncoders = [];
            _isTrained = false;
            Name = GetType().Name;
        }

        /// <summary>
        /// Processes multimodal input data
        /// </summary>
        /// <param name="modalityData">Dictionary mapping modality names to their data</param>
        /// <returns>Fused representation</returns>
        public abstract Vector<T> ProcessMultimodal(Dictionary<string, object> modalityData);

        /// <summary>
        /// Adds a modality encoder to the model
        /// </summary>
        /// <param name="modalityName">Name of the modality</param>
        /// <param name="encoder">The encoder for this modality</param>
        public virtual void AddModalityEncoder(string modalityName, IModalityEncoder<T> encoder)
        {
            if (string.IsNullOrWhiteSpace(modalityName))
                throw new ArgumentException("Modality name cannot be null or empty", nameof(modalityName));

            if (encoder == null)
                throw new ArgumentNullException(nameof(encoder));

            _modalityEncoders[modalityName] = encoder;
        }

        /// <summary>
        /// Gets the encoder for a specific modality
        /// </summary>
        /// <param name="modalityName">Name of the modality</param>
        /// <returns>The modality encoder</returns>
        public IModalityEncoder<T> GetModalityEncoder(string modalityName)
        {
            if (_modalityEncoders.TryGetValue(modalityName, out var encoder))
                return encoder;

            throw new KeyNotFoundException($"No encoder found for modality: {modalityName}");
        }

        /// <summary>
        /// Sets the attention weights between modalities
        /// </summary>
        /// <param name="weights">Matrix of attention weights</param>
        public virtual void SetCrossModalityAttention(Matrix<T> weights)
        {
            _crossModalityAttention = weights;
        }

        /// <summary>
        /// Trains the model with the given data
        /// </summary>
        /// <param name="inputs">Training inputs</param>
        /// <param name="targets">Target outputs</param>
        public virtual void Train(Matrix<T> inputs, Vector<T> targets)
        {
            // Base implementation - derived classes should override
            _isTrained = true;
        }

        /// <summary>
        /// Makes predictions for the given inputs
        /// </summary>
        /// <param name="inputs">Input data</param>
        /// <returns>Predictions</returns>
        public virtual Vector<T> Predict(Matrix<T> inputs)
        {
            if (!_isTrained)
                throw new InvalidOperationException("Model must be trained before making predictions");

            // Base implementation - derived classes should override
            return new Vector<T>(inputs.Rows);
        }

        /// <summary>
        /// Evaluates the model on test data
        /// </summary>
        /// <param name="testInputs">Test inputs</param>
        /// <param name="testTargets">Test targets</param>
        /// <returns>Model statistics</returns>
        public virtual ModelStats<T, Matrix<T>, Vector<T>> Evaluate(Matrix<T> testInputs, Vector<T> testTargets)
        {
            var predictions = Predict(testInputs);
            
            // Create proper ModelStatsInputs
            var statsInputs = new ModelStatsInputs<T, Matrix<T>, Vector<T>>
            {
                Model = null,  // This model doesn't match the expected interface
                XMatrix = testInputs,
                Actual = testTargets,
                Predicted = predictions,
                FeatureCount = testInputs.Columns
            };
            
            return new ModelStats<T, Matrix<T>, Vector<T>>(statsInputs, ModelType.CustomEnsemble);
        }

        /// <summary>
        /// Saves the model to a file
        /// </summary>
        /// <param name="path">Path to save the model</param>
        public virtual void Save(string path)
        {
            // Implementation would serialize the model
            throw new NotImplementedException("Model serialization not implemented");
        }

        /// <summary>
        /// Loads the model from a file
        /// </summary>
        /// <param name="path">Path to load the model from</param>
        public virtual void Load(string path)
        {
            // Implementation would deserialize the model
            throw new NotImplementedException("Model deserialization not implemented");
        }

        /// <summary>
        /// Gets parameters of the model
        /// </summary>
        /// <returns>Dictionary of parameters</returns>
        public virtual Dictionary<string, object> GetParametersDictionary()
        {
            return new Dictionary<string, object>
            {
                ["FusionStrategy"] = _fusionStrategy,
                ["FusedDimension"] = _fusedDimension,
                ["NumModalities"] = _modalityEncoders.Count,
                ["Modalities"] = string.Join(", ", SupportedModalities)
            };
        }

        /// <summary>
        /// Sets parameters of the model
        /// </summary>
        /// <param name="parameters">Dictionary of parameters</param>
        public virtual void SetParameters(Dictionary<string, object> parameters)
        {
            // Base implementation - derived classes should override
        }

        /// <summary>
        /// Creates a copy of the model
        /// </summary>
        /// <returns>A copy of the model</returns>
        public abstract IFullModel<T, Dictionary<string, object>, Vector<T>> Clone();

        /// <summary>
        /// Validates that all required modalities are present in the input
        /// </summary>
        /// <param name="modalityData">Input modality data</param>
        /// <param name="requiredModalities">List of required modalities</param>
        protected void ValidateModalityData(Dictionary<string, object> modalityData, IEnumerable<string>? requiredModalities = null)
        {
            if (modalityData == null || modalityData.Count == 0)
                throw new ArgumentException("Modality data cannot be null or empty");

            var required = requiredModalities ?? _modalityEncoders.Keys;

            foreach (var modality in required)
            {
                if (!modalityData.ContainsKey(modality))
                    throw new KeyNotFoundException($"Required modality '{modality}' not found in input data");

                if (!_modalityEncoders.ContainsKey(modality))
                    throw new InvalidOperationException($"No encoder registered for modality '{modality}'");
            }
        }

        /// <summary>
        /// Encodes data from a specific modality
        /// </summary>
        /// <param name="modalityName">Name of the modality</param>
        /// <param name="data">Data to encode</param>
        /// <returns>Encoded vector</returns>
        protected Vector<T> EncodeModality(string modalityName, object data)
        {
            var encoder = GetModalityEncoder(modalityName);
            return encoder.Encode(data);
        }

        /// <summary>
        /// Applies normalization to the fused representation
        /// </summary>
        /// <param name="fused">Fused vector</param>
        /// <returns>Normalized vector</returns>
        protected virtual Vector<T> NormalizeFused(Vector<T> fused)
        {
            // L2 normalization
            var magnitude = fused.Magnitude();
            if (NumOps.GreaterThan(magnitude, NumOps.Zero))
            {
                return fused / magnitude;
            }
            return fused;
        }

        /// <summary>
        /// Projects the fused representation to the target dimension
        /// </summary>
        /// <param name="fused">Fused vector</param>
        /// <param name="targetDimension">Target dimension</param>
        /// <returns>Projected vector</returns>
        protected virtual Vector<T> ProjectToTargetDimension(Vector<T> fused, int targetDimension)
        {
            if (fused.Length == targetDimension)
                return fused;

            // Simple linear projection (would use learned projection in real implementation)
            var projectionMatrix = CreateProjectionMatrix(fused.Length, targetDimension);
            return projectionMatrix * fused;
        }

        /// <summary>
        /// Creates a random projection matrix
        /// </summary>
        private Matrix<T> CreateProjectionMatrix(int inputDim, int outputDim)
        {
            var random = new Random(42);
            var matrix = new Matrix<T>(outputDim, inputDim);
            
            double scale = Math.Sqrt(2.0 / (inputDim + outputDim));
            
            for (int i = 0; i < outputDim; i++)
            {
                for (int j = 0; j < inputDim; j++)
                {
                    matrix[i, j] = NumOps.FromDouble((random.NextDouble() * 2 - 1) * scale);
                }
            }
            
            return matrix;
        }

        /// <summary>
        /// Gets a string representation of the model
        /// </summary>
        /// <returns>String representation</returns>
        public override string ToString()
        {
            return $"{Name}(fusion={_fusionStrategy}, modalities={_modalityEncoders.Count})";
        }

        #region IFullModel Implementation

        /// <summary>
        /// Trains the model using Dictionary input
        /// </summary>
        public void Train(Dictionary<string, object> inputs, Vector<T> outputs)
        {
            // Convert Dictionary to Matrix for compatibility
            var inputMatrix = ConvertDictionaryToMatrix(inputs);
            Train(inputMatrix, outputs);
        }

        /// <summary>
        /// Predicts using Dictionary input
        /// </summary>
        public Vector<T> Predict(Dictionary<string, object> inputs)
        {
            return ProcessMultimodal(inputs);
        }

        /// <summary>
        /// Gets model metadata
        /// </summary>
        public ModelMetadata<T> GetModelMetadata()
        {
            return new ModelMetadata<T>
            {
                ModelType = Enums.ModelType.CustomEnsemble,
                FeatureCount = _modalityEncoders.Sum(e => e.Value.OutputDimension),
                Complexity = _modalityEncoders.Count,
                Description = $"Multimodal {_fusionStrategy} fusion model",
                AdditionalInfo = new Dictionary<string, object>
                {
                    ["FusionStrategy"] = _fusionStrategy,
                    ["Modalities"] = _modalityEncoders.Keys.ToList(),
                    ["FusedDimension"] = _fusedDimension
                }
            };
        }

        /// <summary>
        /// Serializes the model
        /// </summary>
        public virtual byte[] Serialize()
        {
            throw new NotImplementedException("Model serialization not implemented");
        }

        /// <summary>
        /// Deserializes the model
        /// </summary>
        public virtual void Deserialize(byte[] data)
        {
            throw new NotImplementedException("Model deserialization not implemented");
        }

        /// <summary>
        /// Gets model parameters
        /// </summary>
        public virtual Vector<T> GetParameters()
        {
            // Return empty vector - derived classes should override
            return new Vector<T>(0);
        }


        /// <summary>
        /// Sets model parameters
        /// </summary>
        public virtual void SetParameters(Vector<T> parameters)
        {
            // Base implementation - derived classes should override
        }

        /// <summary>
        /// Creates a new model with the given parameters
        /// </summary>
        public virtual IFullModel<T, Dictionary<string, object>, Vector<T>> WithParameters(Vector<T> parameters)
        {
            var clone = Clone();
            clone.SetParameters(parameters);
            return clone;
        }

        /// <summary>
        /// Gets active feature indices
        /// </summary>
        public virtual IEnumerable<int> GetActiveFeatureIndices()
        {
            var totalFeatures = _modalityEncoders.Sum(e => e.Value.OutputDimension);
            return Enumerable.Range(0, totalFeatures);
        }

        /// <summary>
        /// Checks if a feature is used
        /// </summary>
        public virtual bool IsFeatureUsed(int featureIndex)
        {
            var totalFeatures = _modalityEncoders.Sum(e => e.Value.OutputDimension);
            return featureIndex >= 0 && featureIndex < totalFeatures;
        }

        /// <summary>
        /// Sets active feature indices
        /// </summary>
        public virtual void SetActiveFeatureIndices(IEnumerable<int> indices)
        {
            // Base implementation - derived classes should override
        }

        /// <summary>
        /// Creates a deep copy of the model
        /// </summary>
        public virtual IFullModel<T, Dictionary<string, object>, Vector<T>> DeepCopy()
        {
            return Clone();
        }


        /// <summary>
        /// Converts Dictionary input to Matrix for compatibility
        /// </summary>
        private Matrix<T> ConvertDictionaryToMatrix(Dictionary<string, object> inputs)
        {
            // This is a simplified conversion - in practice would need proper handling
            var vectors = new List<Vector<T>>();
            foreach (var kvp in inputs)
            {
                if (kvp.Value is Vector<T> vec)
                {
                    vectors.Add(vec);
                }
            }
            
            if (vectors.Count == 0)
                return new Matrix<T>(0, 0);
                
            var rows = vectors.Count;
            var cols = vectors[0].Length;
            var matrix = new Matrix<T>(rows, cols);
            
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    matrix[i, j] = vectors[i][j];
                }
            }
            
            return matrix;
        }

        /// <summary>
        /// Adds a new modality to the model with the specified dimension
        /// </summary>
        /// <param name="modalityName">Name of the modality to add. Must be one of the supported types: "Text", "Image", "Audio", or "Numerical"</param>
        /// <param name="dimension">Expected dimension of the modality input</param>
        /// <exception cref="ArgumentException">Thrown when modalityName is invalid or dimension is non-positive</exception>
        /// <exception cref="InvalidOperationException">Thrown when modality already exists</exception>
        /// <remarks>
        /// This method creates a default encoder based on the modality type. The supported modality names are:
        /// - "Text" or names containing "text"/"language": Creates a TextModalityEncoder
        /// - "Image" or names containing "image"/"visual"/"vision": Creates an ImageModalityEncoder
        /// - "Audio" or names containing "audio"/"sound"/"speech": Creates an AudioModalityEncoder
        /// - Any other name: Creates a NumericalModalityEncoder
        /// 
        /// Note: The actual encoder will use a predefined modality name (e.g., "Text", "Image") regardless of the exact modalityName provided.
        /// For custom modality names, use AddModalityEncoder directly with a custom encoder instance.
        /// </remarks>
        public void AddModality(string modalityName, int dimension)
        {
            // Validate input parameters
            if (string.IsNullOrWhiteSpace(modalityName))
            {
                throw new ArgumentException("Modality name cannot be null or empty.", nameof(modalityName));
            }

            if (dimension <= 0)
            {
                throw new ArgumentException($"Modality dimension must be positive. Received: {dimension}", nameof(dimension));
            }

            // Thread-safe check and add
            lock (_modalityEncoders)
            {
                // Check if modality already exists
                if (_modalityEncoders.ContainsKey(modalityName))
                {
                    throw new InvalidOperationException($"Modality '{modalityName}' already exists in the model.");
                }

                // Create appropriate encoder based on modality name conventions
                IModalityEncoder<T> encoder = CreateDefaultEncoder(modalityName, dimension);
                
                // Add the encoder with the requested modalityName as the key
                // Note: The encoder itself may have a different internal modality name
                _modalityEncoders[modalityName] = encoder;
                
                // Update cross-modality attention matrix if it exists
                if (_crossModalityAttention != null)
                {
                    UpdateCrossModalityAttentionForNewModality();
                }
                
                // Mark model as requiring retraining
                _isTrained = false;
            }
        }

        /// <summary>
        /// Sets the fusion strategy for the multimodal model
        /// </summary>
        /// <param name="strategy">The fusion strategy to use (e.g., "early", "late", "cross-attention")</param>
        /// <exception cref="ArgumentException">Thrown when strategy is invalid</exception>
        /// <exception cref="InvalidOperationException">Thrown when changing strategy on a trained model</exception>
        public void SetFusionStrategy(string strategy)
        {
            // Validate input
            if (string.IsNullOrWhiteSpace(strategy))
            {
                throw new ArgumentException("Fusion strategy cannot be null or empty.", nameof(strategy));
            }

            // Normalize strategy name
            string normalizedStrategy = strategy.ToLowerInvariant().Trim();
            
            // Validate strategy is supported
            var supportedStrategies = new HashSet<string> 
            { 
                "early", "late", "cross-attention", "hierarchical", 
                "adaptive", "gated", "tensor", "bilinear" 
            };
            
            if (!supportedStrategies.Contains(normalizedStrategy))
            {
                throw new ArgumentException(
                    $"Unsupported fusion strategy: '{strategy}'. " +
                    $"Supported strategies are: {string.Join(", ", supportedStrategies)}", 
                    nameof(strategy));
            }

            // Check if we can change the strategy
            if (_isTrained)
            {
                throw new InvalidOperationException(
                    "Cannot change fusion strategy on a trained model. " +
                    "Create a new model instance or reset the current model first.");
            }

            // Since this is a base class and fusion strategy is readonly,
            // we need to handle this differently. In a production system,
            // this would either:
            // 1. Be handled by a factory pattern that creates the appropriate model type
            // 2. Use a mutable internal field with proper state management
            // 3. Trigger a model reconstruction
            
            // For now, we'll store the intended strategy change for derived classes to handle
            var parameters = GetParametersDictionary();
            parameters["RequestedFusionStrategy"] = normalizedStrategy;
            SetParameters(parameters);
        }

        /// <summary>
        /// Creates a default encoder for the given modality name
        /// </summary>
        private IModalityEncoder<T> CreateDefaultEncoder(string modalityName, int dimension)
        {
            // Use naming conventions to determine encoder type
            string lowerName = modalityName.ToLowerInvariant();
            
            if (lowerName.Contains("text") || lowerName.Contains("language"))
            {
                return new TextModalityEncoder<T>(dimension);
            }
            else if (lowerName.Contains("image") || lowerName.Contains("visual") || lowerName.Contains("vision"))
            {
                return new ImageModalityEncoder<T>(dimension);
            }
            else if (lowerName.Contains("audio") || lowerName.Contains("sound") || lowerName.Contains("speech"))
            {
                return new AudioModalityEncoder<T>(dimension);
            }
            else
            {
                // Default to numerical encoder for unknown modalities
                return new NumericalModalityEncoder<T>(dimension);
            }
        }

        /// <summary>
        /// Updates the cross-modality attention matrix when a new modality is added
        /// </summary>
        private void UpdateCrossModalityAttentionForNewModality()
        {
            int oldSize = _crossModalityAttention!.Rows;
            int newSize = _modalityEncoders.Count;
            
            // Create new attention matrix
            var newAttention = new Matrix<T>(newSize, newSize);
            
            // Copy existing attention weights
            for (int i = 0; i < oldSize; i++)
            {
                for (int j = 0; j < oldSize; j++)
                {
                    newAttention[i, j] = _crossModalityAttention[i, j];
                }
            }
            
            // Initialize new modality attention weights
            // Use small random values to break symmetry
            var random = new Random();
            for (int i = 0; i < newSize; i++)
            {
                if (i >= oldSize)
                {
                    // New row
                    for (int j = 0; j < newSize; j++)
                    {
                        newAttention[i, j] = NumOps.FromDouble(0.1 + random.NextDouble() * 0.1);
                    }
                }
                else
                {
                    // New column in existing row
                    for (int j = oldSize; j < newSize; j++)
                    {
                        newAttention[i, j] = NumOps.FromDouble(0.1 + random.NextDouble() * 0.1);
                    }
                }
            }
            
            // Normalize attention weights
            for (int i = 0; i < newSize; i++)
            {
                T rowSum = NumOps.Zero;
                for (int j = 0; j < newSize; j++)
                {
                    rowSum = NumOps.Add(rowSum, newAttention[i, j]);
                }
                
                if (NumOps.GreaterThan(rowSum, NumOps.Zero))
                {
                    for (int j = 0; j < newSize; j++)
                    {
                        newAttention[i, j] = NumOps.Divide(newAttention[i, j], rowSum);
                    }
                }
            }
            
            _crossModalityAttention = newAttention;
        }

        #endregion

        #region IInterpretableModel Implementation

        protected readonly HashSet<InterpretationMethod> _enabledMethods = new();
        protected Vector<int> _sensitiveFeatures;
        protected readonly List<FairnessMetric> _fairnessMetrics = new();
        protected IModel<Dictionary<string, object>, Vector<T>, ModelMetadata<T>> _baseModel;

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
        public virtual async Task<Dictionary<int, T>> GetLocalFeatureImportanceAsync(Dictionary<string, object> input)
        {
        return await InterpretableModelHelper.GetLocalFeatureImportanceAsync(this, _enabledMethods, input);
        }

        /// <summary>
        /// Gets SHAP values for the given inputs.
        /// </summary>
        public virtual async Task<Matrix<T>> GetShapValuesAsync(Dictionary<string, object> inputs)
        {
        return await InterpretableModelHelper.GetShapValuesAsync(this, _enabledMethods);
        }

        /// <summary>
        /// Gets LIME explanation for a specific input.
        /// </summary>
        public virtual async Task<LimeExplanation<T>> GetLimeExplanationAsync(Dictionary<string, object> input, int numFeatures = 10)
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
        public virtual async Task<CounterfactualExplanation<T>> GetCounterfactualAsync(Dictionary<string, object> input, Vector<T> desiredOutput, int maxChanges = 5)
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
        public virtual async Task<string> GenerateTextExplanationAsync(Dictionary<string, object> input, Vector<T> prediction)
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
        public virtual async Task<FairnessMetrics<T>> ValidateFairnessAsync(Dictionary<string, object> inputs, int sensitiveFeatureIndex)
        {
        return await InterpretableModelHelper.ValidateFairnessAsync<T>(_fairnessMetrics);
        }

        /// <summary>
        /// Gets anchor explanation for a given input.
        /// </summary>
        public virtual async Task<AnchorExplanation<T>> GetAnchorExplanationAsync(Dictionary<string, object> input, T threshold)
        {
        return await InterpretableModelHelper.GetAnchorExplanationAsync(_enabledMethods, threshold);
        }

        /// <summary>
        /// Sets the base model for interpretability analysis.
        /// </summary>
        public virtual void SetBaseModel(IModel<Dictionary<string, object>, Vector<T>, ModelMetadata<T>> model)
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