using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Statistics;
using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.MultimodalAI
{
    /// <summary>
    /// Base class for multimodal AI models
    /// </summary>
    public abstract class MultimodalModelBase : IMultimodalModel<double, Dictionary<string, object>, Vector<double>>
    {
        protected readonly Dictionary<string, IModalityEncoder> _modalityEncoders;
        protected readonly string _fusionStrategy;
        protected Matrix<double> _crossModalityAttention = default!;
        protected bool _isTrained;
        protected int _fusedDimension;

        /// <summary>
        /// Gets the supported modalities
        /// </summary>
        public IReadOnlyList<string> SupportedModalities => _modalityEncoders.Keys.ToList();

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
        protected MultimodalModelBase(string fusionStrategy, int fusedDimension)
        {
            _fusionStrategy = fusionStrategy;
            _fusedDimension = fusedDimension;
            _modalityEncoders = new Dictionary<string, IModalityEncoder>();
            _isTrained = false;
            Name = GetType().Name;
        }

        /// <summary>
        /// Processes multimodal input data
        /// </summary>
        /// <param name="modalityData">Dictionary mapping modality names to their data</param>
        /// <returns>Fused representation</returns>
        public abstract Vector<double> ProcessMultimodal(Dictionary<string, object> modalityData);

        /// <summary>
        /// Adds a modality encoder to the model
        /// </summary>
        /// <param name="modalityName">Name of the modality</param>
        /// <param name="encoder">The encoder for this modality</param>
        public virtual void AddModalityEncoder(string modalityName, IModalityEncoder encoder)
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
        public IModalityEncoder GetModalityEncoder(string modalityName)
        {
            if (_modalityEncoders.TryGetValue(modalityName, out var encoder))
                return encoder;

            throw new KeyNotFoundException($"No encoder found for modality: {modalityName}");
        }

        /// <summary>
        /// Sets the attention weights between modalities
        /// </summary>
        /// <param name="weights">Matrix<double> of attention weights</param>
        public virtual void SetCrossModalityAttention(Matrix<double> weights)
        {
            _crossModalityAttention = weights;
        }

        /// <summary>
        /// Trains the model with the given data
        /// </summary>
        /// <param name="inputs">Training inputs</param>
        /// <param name="targets">Target outputs</param>
        public virtual void Train(Matrix<double> inputs, Vector<double> targets)
        {
            // Base implementation - derived classes should override
            _isTrained = true;
        }

        /// <summary>
        /// Makes predictions for the given inputs
        /// </summary>
        /// <param name="inputs">Input data</param>
        /// <returns>Predictions</returns>
        public virtual Vector<double> Predict(Matrix<double> inputs)
        {
            if (!_isTrained)
                throw new InvalidOperationException("Model must be trained before making predictions");

            // Base implementation - derived classes should override
            return new Vector<double>(inputs.Rows);
        }

        /// <summary>
        /// Evaluates the model on test data
        /// </summary>
        /// <param name="testInputs">Test inputs</param>
        /// <param name="testTargets">Test targets</param>
        /// <returns>Model statistics</returns>
        public virtual ModelStats<double, Matrix<double>, Vector<double>> Evaluate(Matrix<double> testInputs, Vector<double> testTargets)
        {
            var predictions = Predict(testInputs);

            // Create ModelStatsInputs with the required data
            var inputs = new ModelStatsInputs<double, Matrix<double>, Vector<double>>
            {
                Actual = testTargets,
                Predicted = predictions,
                XMatrix = testInputs,
                FeatureCount = testInputs.Columns,
                Coefficients = Vector<double>.Empty(),
                Model = null
            };

            return new ModelStats<double, Matrix<double>, Vector<double>>(inputs, ModelType.None);
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
        public virtual Dictionary<string, object> GetParameters()
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
        public abstract IFullModel<double, Dictionary<string, object>, Vector<double>> Clone();

        /// <summary>
        /// Validates that all required modalities are present in the input
        /// </summary>
        /// <param name="modalityData">Input modality data</param>
        /// <param name="requiredModalities">List of required modalities</param>
        protected void ValidateModalityData(Dictionary<string, object> modalityData, IEnumerable<string> requiredModalities = null!)
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
        protected Vector<double> EncodeModality(string modalityName, object data)
        {
            var encoder = GetModalityEncoder(modalityName);
            return encoder.Encode(data);
        }

        /// <summary>
        /// Applies normalization to the fused representation
        /// </summary>
        /// <param name="fused">Fused vector</param>
        /// <returns>Normalized vector</returns>
        protected virtual Vector<double> NormalizeFused(Vector<double> fused)
        {
            // L2 normalization
            var magnitude = fused.Magnitude();
            if (magnitude > 0)
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
        protected virtual Vector<double> ProjectToTargetDimension(Vector<double> fused, int targetDimension)
        {
            if (fused.Dimension == targetDimension)
                return fused;

            // Simple linear projection (would use learned projection in real implementation)
            var projectionMatrix = CreateProjectionMatrix(fused.Dimension, targetDimension);
            return projectionMatrix * fused;
        }

        /// <summary>
        /// Creates a random projection matrix
        /// </summary>
        private Matrix<double> CreateProjectionMatrix(int inputDim, int outputDim)
        {
            var random = new Random(42);
            var matrix = new Matrix<double>(outputDim, inputDim);
            
            double scale = Math.Sqrt(2.0 / (inputDim + outputDim));
            
            for (int i = 0; i < outputDim; i++)
            {
                for (int j = 0; j < inputDim; j++)
                {
                    matrix[i, j] = (random.NextDouble() * 2 - 1) * scale;
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
    }
}