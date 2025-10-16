using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.Logging;
using AiDotNet.Enums;

namespace AiDotNet.FoundationModels
{
    /// <summary>
    /// Adapts foundation models to work as predictive models in the AiDotNet framework.
    /// Bridges the gap between text generation models and the standard prediction interface.
    /// </summary>
    /// <typeparam name="T">The numeric type for computations</typeparam>
    public class FoundationModelAdapter<T> : Interpretability.InterpretableModelBase<T, Matrix<T>, Vector<T>>, IFullModel<T, Matrix<T>, Vector<T>>
    {
        private readonly IFoundationModel<T> _foundationModel;
        private readonly ILogging _logger;
        private PredictionType _predictionType;
        private ModelConfiguration _configuration;
        private readonly Dictionary<string, object> _metadata;
        private T[]? _adapterParameters;
        private T[]? _featureImportance;
        private HashSet<int>? _activeFeatureIndices;

        /// <summary>
        /// Initializes a new instance of the FoundationModelAdapter class
        /// </summary>
        /// <param name="foundationModel">The foundation model to adapt</param>
        /// <param name="predictionType">Type of prediction task</param>
        /// <param name="configuration">Model configuration</param>
        /// <param name="logger">Optional logger</param>
        public FoundationModelAdapter(
            IFoundationModel<T> foundationModel,
            PredictionType predictionType,
            ModelConfiguration? configuration = null,
            ILogging? logger = null)
        {
            _foundationModel = foundationModel ?? throw new ArgumentNullException(nameof(foundationModel));
            _predictionType = predictionType;
            _configuration = configuration ?? new ModelConfiguration();
            _logger = logger ?? new AiDotNetLogger();
            _metadata = new Dictionary<string, object>();
            
            InitializeMetadata();
        }

        #region IPredictiveModel Implementation

        /// <inheritdoc/>
        public override Vector<T> Predict(Matrix<T> input)
        {
            if (input == null || input.Rows == 0)
            {
                throw new ArgumentException("Input cannot be null or empty", nameof(input));
            }

            _logger.Debug("Processing {Rows} samples for prediction", input.Rows);

            var predictions = new T[input.Rows];
            
            // Process each row as a separate prediction
            for (int i = 0; i < input.Rows; i++)
            {
                var inputText = ConvertRowToText(input, i);
                var prediction = PredictSingleAsync(inputText).GetAwaiter().GetResult();
                predictions[i] = ConvertPredictionToNumeric(prediction);
            }

            return new Vector<T>(predictions);
        }

        /// <inheritdoc/>
        public override void Train(Matrix<T> input, Vector<T> output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (input.Rows != output.Length)
            {
                throw new ArgumentException("Input rows must match output length", nameof(input));
            }

            _logger.Information("Training foundation model adapter with {Rows} samples", input.Rows);

            // Convert training data to foundation model format
            var trainingExamples = new List<TrainingExample>();
            
            for (int i = 0; i < input.Rows; i++)
            {
                var inputText = ConvertRowToText(input, i);
                var outputValue = output[i].ToString() ?? "";
                
                // Format based on prediction type
                var outputText = _predictionType switch
                {
                    PredictionType.Classification => FormatClassLabel(outputValue),
                    PredictionType.Regression => outputValue,
                    _ => outputValue
                };

                trainingExamples.Add(new TrainingExample
                {
                    Input = inputText,
                    Output = outputText,
                    Weight = 1.0
                });
            }

            // Use foundation model's fine-tuning capabilities
            var config = new FineTuningConfig
            {
                LearningRate = _configuration.LearningRate ?? 0.0001,
                BatchSize = _configuration.BatchSize ?? 32,
                Epochs = _configuration.Epochs ?? 3,
                ValidationSplit = _configuration.ValidationSplit ?? 0.1
            };

            // Split data for validation
            var splitIndex = (int)(trainingExamples.Count * (1 - config.ValidationSplit));
            var trainingData = trainingExamples.Take(splitIndex).ToList();
            var validationData = trainingExamples.Skip(splitIndex).ToList();

            // Fine-tune synchronously (blocking call)
            var fineTunedModel = _foundationModel.FineTuneAsync(
                trainingData,
                validationData,
                config,
                progress => _logger.Debug("Fine-tuning progress: {Progress}%", progress.PercentComplete),
                CancellationToken.None).GetAwaiter().GetResult();

            // Update metadata
            _metadata["training_samples"] = input.Rows;
            _metadata["last_trained"] = DateTime.UtcNow;
            _metadata["fine_tuning_epochs"] = config.Epochs;
            
            _logger.Information("Foundation model adapter training completed");
        }

        /// <inheritdoc/>
        public override ModelMetadata<T> GetModelMetadata()
        {
            var metadata = new ModelMetadata<T>
            {
                ModelType = ModelType.FoundationModel,
                FeatureCount = _configuration.FeatureNames?.Count ?? 0,
                Complexity = (int)Math.Log10(_foundationModel.ParameterCount + 1), // Log scale of parameters
                Description = $"{_foundationModel.Architecture} foundation model adapted for {_predictionType}",
                AdditionalInfo = new Dictionary<string, object>(_metadata),
                ModelData = SerializeModel()
            };

            // Add feature importance if available
            if (_configuration.FeatureNames != null && _featureImportance != null)
            {
                for (int i = 0; i < _configuration.FeatureNames.Count && i < _featureImportance.Count; i++)
                {
                    metadata.FeatureImportance[_configuration.FeatureNames[i]] = _featureImportance[i];
                }
            }

            return metadata;
        }

        /// <inheritdoc/>
        public byte[] Serialize()
        {
            return SerializeModel();
        }

        /// <inheritdoc/>
        public void Deserialize(byte[] data)
        {
            if (data == null || data.Length == 0)
            {
                throw new ArgumentException("Data cannot be null or empty", nameof(data));
            }

            using var stream = new MemoryStream(data);
            using var reader = new BinaryReader(stream);
            
            // Read and verify magic number
            var magic = reader.ReadInt32();
            if (magic != 0x464D4144) // "FMAD"
            {
                throw new InvalidOperationException("Invalid serialization format");
            }
            
            // Read version
            var version = reader.ReadInt32();
            if (version != 1)
            {
                throw new InvalidOperationException($"Unsupported version: {version}");
            }
            
            // Read JSON data
            var jsonLength = reader.ReadInt32();
            var jsonBytes = reader.ReadBytes(jsonLength);
            var json = System.Text.Encoding.UTF8.GetString(jsonBytes);
            
            var serializationData = System.Text.Json.JsonSerializer.Deserialize<FoundationModelSerializationData>(json);
            if (serializationData == null)
            {
                throw new InvalidOperationException("Failed to deserialize data");
            }
            
            // Update configuration and metadata
            _predictionType = serializationData.PredictionType;
            _configuration = serializationData.Configuration;
            _metadata.Clear();
            foreach (var kvp in serializationData.Metadata)
            {
                _metadata[kvp.Key] = kvp.Value;
            }
            
            // Note: The foundation model itself needs to be reconstructed based on architecture
            // This would typically involve loading from a checkpoint
            _logger.Information("Deserialized foundation model adapter configuration");
        }

        /// <inheritdoc/>
        public Vector<T> GetParameters()
        {
            // Foundation models typically have billions of parameters
            // Return a representative subset or throw an exception
            _logger.Warning("GetParameters called on foundation model - returning empty vector as full parameters are too large");
            
            // Return adapter parameters if using parameter-efficient fine-tuning
            if (_adapterParameters != null)
            {
                return new Vector<T>(_adapterParameters);
            }
            
            // Return empty vector for base foundation models
            return new Vector<T>(Array.Empty<T>());
        }

        /// <inheritdoc/>
        public void SetParameters(Vector<T> parameters)
        {
            if (parameters == null) throw new ArgumentNullException(nameof(parameters));
            
            _logger.Information("Setting {Count} adapter parameters", parameters.Length);
            
            // Store as adapter parameters (for parameter-efficient fine-tuning)
            _adapterParameters = parameters.ToArray();
            
            // Update metadata
            _metadata["adapter_parameter_count"] = parameters.Length;
            _metadata["parameters_updated"] = DateTime.UtcNow;
        }

        /// <inheritdoc/>
        public IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
        {
            var newAdapter = new FoundationModelAdapter<T>(
                _foundationModel,
                _predictionType,
                _configuration,
                _logger);
            
            newAdapter.SetParameters(parameters);
            
            // Copy metadata
            foreach (var kvp in _metadata)
            {
                newAdapter._metadata[kvp.Key] = kvp.Value;
            }
            
            // Copy feature importance
            if (_featureImportance != null)
            {
                newAdapter._featureImportance = _featureImportance.ToArray();
            }
            
            // Copy active features
            if (_activeFeatureIndices != null)
            {
                newAdapter._activeFeatureIndices = new HashSet<int>(_activeFeatureIndices);
            }
            
            return newAdapter;
        }

        /// <inheritdoc/>
        public IEnumerable<int> GetActiveFeatureIndices()
        {
            return _activeFeatureIndices ?? Enumerable.Range(0, _configuration.FeatureNames?.Count ?? 0);
        }

        /// <inheritdoc/>
        public bool IsFeatureUsed(int featureIndex)
        {
            if (_activeFeatureIndices == null)
            {
                // All features are active by default
                return featureIndex >= 0 && featureIndex < (_configuration.FeatureNames?.Count ?? int.MaxValue);
            }
            
            return _activeFeatureIndices.Contains(featureIndex);
        }

        /// <inheritdoc/>
        public void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
        {
            if (featureIndices == null) throw new ArgumentNullException(nameof(featureIndices));
            
            _activeFeatureIndices = new HashSet<int>(featureIndices);
            
            _logger.Information("Set {Count} active feature indices", _activeFeatureIndices.Count);
            
            // Update metadata
            _metadata["active_feature_count"] = _activeFeatureIndices.Count;
        }


        /// <inheritdoc/>
        public IFullModel<T, Matrix<T>, Vector<T>> DeepCopy()
        {
            // Create new instance with same configuration
            var copy = new FoundationModelAdapter<T>(
                _foundationModel, // Note: Foundation model is shared (not deep copied)
                _predictionType,
                new ModelConfiguration
                {
                    PromptTemplate = _configuration.PromptTemplate,
                    FeatureNames = _configuration.FeatureNames?.ToList(),
                    MaxTokens = _configuration.MaxTokens,
                    Temperature = _configuration.Temperature,
                    TopP = _configuration.TopP,
                    MaxConcurrency = _configuration.MaxConcurrency,
                    LearningRate = _configuration.LearningRate,
                    BatchSize = _configuration.BatchSize,
                    Epochs = _configuration.Epochs,
                    ValidationSplit = _configuration.ValidationSplit
                },
                _logger);
            
            // Deep copy metadata
            foreach (var kvp in _metadata)
            {
                copy._metadata[kvp.Key] = kvp.Value;
            }
            
            // Deep copy adapter parameters
            if (_adapterParameters != null)
            {
                copy._adapterParameters = _adapterParameters.ToArray();
            }
            
            // Deep copy feature importance
            if (_featureImportance != null)
            {
                copy._featureImportance = _featureImportance.ToArray();
            }
            
            // Deep copy active features
            if (_activeFeatureIndices != null)
            {
                copy._activeFeatureIndices = new HashSet<int>(_activeFeatureIndices);
            }
            
            return copy;
        }

        /// <inheritdoc/>
        public IFullModel<T, Matrix<T>, Vector<T>> Clone()
        {
            // Shallow copy - shares the foundation model instance
            var clone = new FoundationModelAdapter<T>(
                _foundationModel, // Shared reference
                _predictionType,
                _configuration, // Shared reference
                _logger);
            
            // Share metadata reference
            clone._metadata = _metadata;
            
            // Share array references
            clone._adapterParameters = _adapterParameters;
            clone._featureImportance = _featureImportance;
            clone._activeFeatureIndices = _activeFeatureIndices;
            
            return clone;
        }

        /// <inheritdoc/>
        public byte[] SerializeModel()
        {
            // Foundation models typically aren't serialized directly
            // Instead, we serialize the configuration and model reference
            var serializationData = new FoundationModelSerializationData
            {
                ModelArchitecture = _foundationModel.Architecture,
                PredictionType = _predictionType,
                Configuration = _configuration,
                Metadata = _metadata,
                AvailableCheckpoints = _foundationModel.GetAvailableCheckpoints()
            };

            using var stream = new MemoryStream();
            using var writer = new BinaryWriter(stream);
            
            // Write magic number and version
            writer.Write(0x464D4144); // "FMAD" in hex
            writer.Write(1); // Version
            
            // Serialize the data
            var json = System.Text.Json.JsonSerializer.Serialize(serializationData);
            var bytes = System.Text.Encoding.UTF8.GetBytes(json);
            writer.Write(bytes.Length);
            writer.Write(bytes);
            
            return stream.ToArray();
        }

        /// <inheritdoc/>
        public Task<byte[]> SerializeModelAsync()
        {
            return Task.FromResult(SerializeModel());
        }

        #endregion

        #region Public Methods

        /// <summary>
        /// Makes a prediction asynchronously
        /// </summary>
        /// <param name="input">Input matrix</param>
        /// <returns>Prediction vector</returns>
        public override async Task<Vector<T>> PredictAsync(Matrix<T> input)
        {
            if (input == null || input.Rows == 0)
            {
                throw new ArgumentException("Input cannot be null or empty", nameof(input));
            }

            var predictions = new T[input.Rows];
            var tasks = new Task<T>[input.Rows];

            // Process predictions in parallel with controlled concurrency
            var maxConcurrency = _configuration.MaxConcurrency ?? Environment.ProcessorCount;
            using var semaphore = new System.Threading.SemaphoreSlim(maxConcurrency);

            for (int i = 0; i < input.Rows; i++)
            {
                var index = i;
                tasks[i] = Task.Run(async () =>
                {
                    await semaphore.WaitAsync();
                    try
                    {
                        var inputText = ConvertRowToText(input, index);
                        var prediction = await PredictSingleAsync(inputText);
                        return ConvertPredictionToNumeric(prediction);
                    }
                    finally
                    {
                        semaphore.Release();
                    }
                });
            }

            var results = await Task.WhenAll(tasks);
            return new Vector<T>(results);
        }

        /// <summary>
        /// Gets the underlying foundation model
        /// </summary>
        public IFoundationModel<T> GetFoundationModel() => _foundationModel;

        /// <summary>
        /// Updates the model configuration
        /// </summary>
        public void UpdateConfiguration(ModelConfiguration configuration)
        {
            _configuration.UpdateFrom(configuration);
        }

        #endregion

        #region Private Methods

        private void InitializeMetadata()
        {
            _metadata["adapter_version"] = "1.0";
            _metadata["foundation_model_architecture"] = _foundationModel.Architecture;
            _metadata["foundation_model_parameters"] = _foundationModel.ParameterCount;
            _metadata["prediction_type"] = _predictionType.ToString();
            _metadata["max_context_length"] = _foundationModel.MaxContextLength;
            _metadata["vocabulary_size"] = _foundationModel.VocabularySize;
        }

        private string ConvertRowToText(Matrix<T> input, int rowIndex)
        {
            // Convert matrix row to text based on configuration
            var values = new List<string>();
            
            for (int j = 0; j < input.Columns; j++)
            {
                var value = input[rowIndex, j];
                
                if (_configuration.FeatureNames != null && j < _configuration.FeatureNames.Count)
                {
                    values.Add($"{_configuration.FeatureNames[j]}: {value}");
                }
                else
                {
                    values.Add(value.ToString() ?? "");
                }
            }

            // Apply prompt template if configured
            if (!string.IsNullOrEmpty(_configuration.PromptTemplate))
            {
                var variables = new Dictionary<string, string>
                {
                    ["features"] = string.Join(", ", values),
                    ["task"] = _predictionType.ToString()
                };
                
                return _foundationModel.ApplyPromptTemplate(_configuration.PromptTemplate, variables);
            }

            // Default formatting
            return _predictionType switch
            {
                PredictionType.Classification => $"Classify the following: {string.Join(", ", values)}",
                PredictionType.Regression => $"Predict the value for: {string.Join(", ", values)}",
                PredictionType.Clustering => $"Analyze the cluster for: {string.Join(", ", values)}",
                _ => string.Join(", ", values)
            };
        }

        private async Task<string> PredictSingleAsync(string inputText)
        {
            var result = await _foundationModel.GenerateAsync(
                inputText,
                maxTokens: _configuration.MaxTokens ?? 50,
                temperature: _configuration.Temperature ?? 0.1,
                topP: _configuration.TopP ?? 0.95);

            return result.Trim();
        }

        private T ConvertPredictionToNumeric(string prediction)
        {
            try
            {
                // Extract numeric value from the prediction
                var numericString = ExtractNumericValue(prediction);
                
                // Convert to the target type
                if (typeof(T) == typeof(double))
                {
                    return (T)(object)double.Parse(numericString);
                }
                else if (typeof(T) == typeof(float))
                {
                    return (T)(object)float.Parse(numericString);
                }
                else if (typeof(T) == typeof(int))
                {
                    return (T)(object)int.Parse(numericString);
                }
                else
                {
                    return (T)Convert.ChangeType(numericString, typeof(T));
                }
            }
            catch (Exception ex)
            {
                _logger.Warning("Failed to parse prediction '{Prediction}': {Error}", prediction, ex.Message);
                
                // Return default value for the type
                return default(T);
            }
        }

        private string ExtractNumericValue(string text)
        {
            // Try to extract numeric value from text
            // This is a simple implementation - can be enhanced with regex or more sophisticated parsing
            
            var parts = text.Split(new[] { ' ', '\n', '\t', ',', ':', '=' }, StringSplitOptions.RemoveEmptyEntries);
            
            foreach (var part in parts)
            {
                if (double.TryParse(part, out _))
                {
                    return part;
                }
            }

            // If no numeric value found, try to map text to numeric for classification
            if (_predictionType == PredictionType.Classification)
            {
                return MapClassLabelToNumeric(text);
            }

            return "0";
        }

        private string MapClassLabelToNumeric(string label)
        {
            // Map common class labels to numeric values
            var normalizedLabel = label.ToLowerInvariant().Trim();
            
            return normalizedLabel switch
            {
                "yes" or "true" or "positive" or "1" => "1",
                "no" or "false" or "negative" or "0" => "0",
                _ => "0" // Default to 0 for unknown labels
            };
        }

        private string FormatClassLabel(string value)
        {
            // Format numeric class label to text representation
            var normalizedValue = value.Trim();
            
            // Common binary classification mappings
            return normalizedValue switch
            {
                "1" or "1.0" => "positive",
                "0" or "0.0" => "negative",
                _ => $"class_{normalizedValue}"
            };
        }
        
        public override async Task TrainAsync(Matrix<T> inputs, Vector<T> targets)
        {
            // Use the synchronous Train method for now
            await Task.Run(() => Train(inputs, targets));
        }
        
        public override void SetModelMetadata(ModelMetadata<T> metadata)
        {
            // Update relevant fields from the provided metadata
            if (metadata.AdditionalInfo != null)
            {
                foreach (var kvp in metadata.AdditionalInfo)
                {
                    _metadata[kvp.Key] = kvp.Value;
                }
            }
        }
        
        public override void Save(string filepath)
        {
            // Foundation models typically save their state to checkpoints
            // For the adapter, we save the configuration and metadata
            var data = SerializeModel();
            File.WriteAllBytes(filepath, data);
            _logger.Information("Saved foundation model adapter to {FilePath}", filepath);
        }
        
        public override void Load(string filepath)
        {
            // Load the configuration and metadata
            var data = File.ReadAllBytes(filepath);
            Deserialize(data);
            _logger.Information("Loaded foundation model adapter from {FilePath}", filepath);
        }
        
        public override void Dispose()
        {
            // Dispose of the foundation model if it's disposable
            if (_foundationModel is IDisposable disposable)
            {
                disposable.Dispose();
            }
            _metadata.Clear();
            _logger.Information("Foundation model adapter disposed");
        }
        
        // Override interpretability methods with foundation model-specific implementations
        public override async Task<Dictionary<int, T>> GetGlobalFeatureImportanceAsync()
        {
            _logger.Warning("Traditional feature importance not applicable to foundation models. Returning uniform importance.");
            var importance = new Dictionary<int, T>();
            var featureCount = _configuration.FeatureNames?.Count ?? 0;
            var uniformValue = _ops.Divide(_ops.One, _ops.FromDouble(Math.Max(1, featureCount)));
            
            for (int i = 0; i < featureCount; i++)
            {
                importance[i] = uniformValue;
            }
            
            return importance;
        }
        
        public override async Task<LimeExplanation<T>> GetLimeExplanationAsync(Matrix<T> input, int numFeatures = 10)
        {
            _logger.Warning("LIME explanations not directly applicable to foundation models. Consider using attention weights instead.");
            return new LimeExplanation<T>();
        }
        
        public override async Task<Matrix<T>> GetShapValuesAsync(Matrix<T> inputs)
        {
            _logger.Warning("SHAP values not directly applicable to foundation models. Consider using gradient-based attribution methods.");
            return new Matrix<T>(inputs.Rows, inputs.Columns);
        }
        
        public override async Task<string> GenerateTextExplanationAsync(Matrix<T> input, Vector<T> prediction)
        {
            // Foundation models can provide natural language explanations
            var inputText = ConvertRowToText(input, 0);
            var predictionValue = prediction[0].ToString() ?? "unknown";
            
            var explanationPrompt = $"Explain why the input '{inputText}' resulted in the prediction '{predictionValue}'.";
            
            try
            {
                var response = await _foundationModel.GenerateAsync(
                    explanationPrompt,
                    new GenerationConfig { MaxTokens = 150, Temperature = 0.7 });
                    
                return response.Text;
            }
            catch (Exception ex)
            {
                _logger.Error(ex, "Failed to generate text explanation");
                return $"Foundation model predicted {predictionValue} based on the input features.";
            }
        }

        #endregion

        #region Nested Classes

        /// <summary>
        /// Configuration for the model adapter
        /// </summary>
        public class ModelConfiguration
        {
            public string? PromptTemplate { get; set; }
            public List<string>? FeatureNames { get; set; }
            public int? MaxTokens { get; set; }
            public double? Temperature { get; set; }
            public double? TopP { get; set; }
            public int? MaxConcurrency { get; set; }
            public double? LearningRate { get; set; }
            public int? BatchSize { get; set; }
            public int? Epochs { get; set; }
            public double? ValidationSplit { get; set; }
            
            public void UpdateFrom(ModelConfiguration other)
            {
                if (other == null) return;
                
                PromptTemplate = other.PromptTemplate ?? PromptTemplate;
                FeatureNames = other.FeatureNames ?? FeatureNames;
                MaxTokens = other.MaxTokens ?? MaxTokens;
                Temperature = other.Temperature ?? Temperature;
                TopP = other.TopP ?? TopP;
                MaxConcurrency = other.MaxConcurrency ?? MaxConcurrency;
                LearningRate = other.LearningRate ?? LearningRate;
                BatchSize = other.BatchSize ?? BatchSize;
                Epochs = other.Epochs ?? Epochs;
                ValidationSplit = other.ValidationSplit ?? ValidationSplit;
            }
        }

        /// <summary>
        /// Data structure for serialization
        /// </summary>
        private class FoundationModelSerializationData
        {
            public string ModelArchitecture { get; set; } = string.Empty;
            public PredictionType PredictionType { get; set; }
            public ModelConfiguration Configuration { get; set; } = new();
            public Dictionary<string, object> Metadata { get; set; } = new();
            public List<string> AvailableCheckpoints { get; set; } = new();
        }

        #endregion
    }
}