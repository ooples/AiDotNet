using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Logging;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.Enums;

namespace AiDotNet.FoundationModels;

/// <summary>
/// Abstract base class for foundation model implementations.
/// Provides common functionality and structure for all foundation models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations</typeparam>
public abstract class FoundationModelBase<T> : Interpretability.InterpretableModelBase<T, string, string>, IFoundationModel<T>
{
    protected readonly ITokenizer _tokenizer;
    protected readonly ILogging _logger;
    protected readonly Dictionary<string, string> _availableCheckpoints;
    protected bool _isInitialized;
    protected readonly SemaphoreSlim _initSemaphore = new SemaphoreSlim(1, 1);

    /// <summary>
    /// Initializes a new instance of the FoundationModelBase class
    /// </summary>
    /// <param name="tokenizer">Tokenizer to use for text processing</param>
    /// <param name="logger">Optional logger instance</param>
    protected FoundationModelBase(ITokenizer tokenizer, ILogging? logger = null)
    {
        _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
        _logger = logger ?? new AiDotNetLogger(logger);
        _availableCheckpoints = new Dictionary<string, string>();
        _isInitialized = false;
    }

    #region Abstract Properties and Methods

    /// <inheritdoc/>
    public abstract string Architecture { get; }

    /// <inheritdoc/>
    public abstract long ParameterCount { get; }

    /// <summary>
    /// Performs the actual text generation logic
    /// </summary>
    protected abstract Task<string> GenerateInternalAsync(
        TokenizerOutput tokenizedInput,
        int maxTokens,
        double temperature,
        double topP,
        CancellationToken cancellationToken);

    /// <summary>
    /// Computes embeddings for the given token IDs
    /// </summary>
    protected abstract Task<Tensor<T>> ComputeEmbeddingsAsync(
        TokenizerOutput tokenizedInput,
        CancellationToken cancellationToken);

    /// <summary>
    /// Loads model weights from a checkpoint
    /// </summary>
    protected abstract Task LoadModelWeightsAsync(string checkpointPath, CancellationToken cancellationToken);

    /// <summary>
    /// Initializes model-specific components
    /// </summary>
    protected abstract Task InitializeModelAsync(CancellationToken cancellationToken);

    #endregion

    #region IFoundationModel Implementation

    /// <inheritdoc/>
    public virtual int VocabularySize => _tokenizer.VocabularySize;

    /// <inheritdoc/>
    public virtual int MaxContextLength => _tokenizer.MaxSequenceLength;

    /// <inheritdoc/>
    public virtual async Task<string> GenerateAsync(
        string prompt,
        int maxTokens = 100,
        double temperature = 1.0,
        double topP = 1.0,
        CancellationToken cancellationToken = default)
    {
        await EnsureInitializedAsync(cancellationToken);
        ValidateGenerationParameters(maxTokens, temperature, topP);

        _logger.Debug("Generating text with max_tokens={MaxTokens}, temperature={Temperature}, top_p={TopP}", 
            maxTokens, temperature, topP);

        // Tokenize input
        var tokenizedInput = await _tokenizer.EncodeBatchAsync(
            new[] { prompt },
            maxLength: MaxContextLength,
            padding: false,
            truncation: true);

        // Generate text
        var generatedText = await GenerateInternalAsync(
            tokenizedInput,
            maxTokens,
            temperature,
            topP,
            cancellationToken);

        _logger.Debug("Generated {Length} characters of text", generatedText.Length);
        return generatedText;
    }

    /// <inheritdoc/>
    public virtual async Task<T[]> GetEmbeddingAsync(string text)
    {
        await EnsureInitializedAsync();

        var tokenizedInput = await _tokenizer.EncodeBatchAsync(
            new[] { text },
            maxLength: MaxContextLength,
            padding: false,
            truncation: true);

        var embeddings = await ComputeEmbeddingsAsync(tokenizedInput, CancellationToken.None);
        
        // Average pooling over sequence dimension
        var averaged = new T[embeddings.Shape[embeddings.Rank - 1]];
        var sequenceLength = embeddings.Shape[0];
        var numOps = MathHelper.GetNumericOperations<T>();
        
        for (int i = 0; i < averaged.Length; i++)
        {
            T sum = numOps.Zero;
            for (int j = 0; j < sequenceLength; j++)
            {
                sum = numOps.Add(sum, embeddings[j, i]);
            }
            averaged[i] = numOps.Divide(sum, numOps.FromInt(sequenceLength));
        }

        return averaged;
    }

    /// <inheritdoc/>
    public virtual async Task<int[]> TokenizeAsync(string text)
    {
        var tokens = await _tokenizer.EncodeAsync(text, addSpecialTokens: true);
        return tokens.ToArray();
    }

    /// <inheritdoc/>
    public virtual async Task<string> DecodeAsync(int[] tokenIds)
    {
        var tokenVector = new Vector<int>(tokenIds);
        return await _tokenizer.DecodeAsync(tokenVector, skipSpecialTokens: true);
    }

    /// <inheritdoc/>
    public virtual async Task<IFoundationModel<T>> FineTuneAsync(
        List<TrainingExample> trainingData,
        List<TrainingExample> validationData,
        FineTuningConfig config,
        Action<FineTuningProgress>? progressCallback = null,
        CancellationToken cancellationToken = default)
    {
        await EnsureInitializedAsync(cancellationToken);

        if (trainingData == null || trainingData.Count == 0)
        {
            throw new ArgumentException("Training data cannot be null or empty", nameof(trainingData));
        }

        _logger.Information("Starting fine-tuning with {TrainCount} training examples and {ValCount} validation examples",
            trainingData.Count, validationData?.Count ?? 0);

        // This is a placeholder - actual implementation would perform real fine-tuning
        throw new NotImplementedException(
            "Fine-tuning functionality requires specific model implementation. " +
            "Override this method in derived classes to provide actual fine-tuning logic.");
    }

    /// <inheritdoc/>
    public virtual async Task<string> FewShotAsync(List<FewShotExample> examples, string query)
    {
        if (examples == null || examples.Count == 0)
        {
            throw new ArgumentException("Examples cannot be null or empty", nameof(examples));
        }

        // Construct prompt with examples
        var promptBuilder = new System.Text.StringBuilder();
        
        foreach (var example in examples)
        {
            promptBuilder.AppendLine($"Input: {example.Input}");
            promptBuilder.AppendLine($"Output: {example.Output}");
            
            if (!string.IsNullOrEmpty(example.Explanation))
            {
                promptBuilder.AppendLine($"Explanation: {example.Explanation}");
            }
            
            promptBuilder.AppendLine();
        }
        
        promptBuilder.AppendLine($"Input: {query}");
        promptBuilder.Append("Output: ");

        var prompt = promptBuilder.ToString();
        return await GenerateAsync(prompt, maxTokens: 100);
    }

    /// <inheritdoc/>
    public virtual string ApplyPromptTemplate(string template, Dictionary<string, string> variables)
    {
        if (string.IsNullOrEmpty(template))
        {
            throw new ArgumentException("Template cannot be null or empty", nameof(template));
        }

        var result = template;
        
        if (variables != null)
        {
            foreach (var kvp in variables)
            {
                result = result.Replace($"{{{kvp.Key}}}", kvp.Value ?? string.Empty);
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public virtual Task<AttentionWeights> GetAttentionWeightsAsync(string text)
    {
        // This would require model-specific implementation
        throw new NotImplementedException(
            "Attention weight extraction requires specific model implementation. " +
            "Override this method in derived classes.");
    }

    /// <inheritdoc/>
    public virtual async Task<ChainOfThoughtResult> ChainOfThoughtAsync(string problem)
    {
        // Basic implementation using prompting
        var cot_prompt = $"Problem: {problem}\n\nLet's solve this step by step.\n\nStep 1:";
        var response = await GenerateAsync(cot_prompt, maxTokens: 500);

        // Parse response into steps (simplified version)
        var steps = response.Split(new[] { "Step " }, StringSplitOptions.RemoveEmptyEntries)
            .Select(s => s.Trim())
            .Where(s => !string.IsNullOrEmpty(s))
            .ToList();

        var finalAnswer = steps.LastOrDefault()?.Contains("Final Answer:") == true
            ? steps.Last().Substring(steps.Last().IndexOf("Final Answer:") + 13).Trim()
            : "Unable to determine final answer";

        return new ChainOfThoughtResult
        {
            ReasoningSteps = steps,
            FinalAnswer = finalAnswer,
            Confidence = 0.75, // Placeholder
            Metadata = new Dictionary<string, object>
            {
                ["model"] = Architecture,
                ["prompt_tokens"] = problem.Length / 4 // Rough estimate
            }
        };
    }

    /// <inheritdoc/>
    public virtual Task<BenchmarkResults> EvaluateBenchmarkAsync(IBenchmarkDataset benchmark)
    {
        throw new NotImplementedException(
            "Benchmark evaluation requires specific implementation. " +
            "Override this method in derived classes.");
    }

    /// <inheritdoc/>
    public virtual void ApplyAdapter(IModelAdapter adapter)
    {
        if (adapter == null)
        {
            throw new ArgumentNullException(nameof(adapter));
        }

        _logger.Information("Applying {AdapterType} adapter with {Parameters} parameters",
            adapter.AdapterType, adapter.AdapterParameters);

        adapter.Apply(this);
    }

    /// <inheritdoc/>
    public virtual List<string> GetAvailableCheckpoints()
    {
        return _availableCheckpoints.Keys.ToList();
    }

    /// <inheritdoc/>
    public virtual async Task LoadCheckpointAsync(string checkpointName)
    {
        if (!_availableCheckpoints.TryGetValue(checkpointName, out var checkpointPath))
        {
            throw new ArgumentException($"Unknown checkpoint: {checkpointName}", nameof(checkpointName));
        }

        _logger.Information("Loading checkpoint: {Checkpoint}", checkpointName);
        await LoadModelWeightsAsync(checkpointPath, CancellationToken.None);
        _logger.Information("Successfully loaded checkpoint: {Checkpoint}", checkpointName);
    }

    #endregion

    #region Protected Helper Methods

    /// <summary>
    /// Ensures the model is initialized before use
    /// </summary>
    protected async Task EnsureInitializedAsync(CancellationToken cancellationToken = default)
    {
        if (_isInitialized) return;

        await _initSemaphore.WaitAsync(cancellationToken);
        try
        {
            if (_isInitialized) return;

            _logger.Information("Initializing {Architecture} model", Architecture);
            
            // Initialize tokenizer
            if (!_tokenizer.IsInitialized)
            {
                await _tokenizer.InitializeAsync();
            }

            // Initialize model
            await InitializeModelAsync(cancellationToken);
            
            _isInitialized = true;
            _logger.Information("{Architecture} model initialized successfully", Architecture);
        }
        finally
        {
            _initSemaphore.Release();
        }
    }

    /// <summary>
    /// Validates generation parameters
    /// </summary>
    protected void ValidateGenerationParameters(int maxTokens, double temperature, double topP)
    {
        if (maxTokens <= 0 || maxTokens > MaxContextLength)
        {
            throw new ArgumentOutOfRangeException(nameof(maxTokens), 
                $"Max tokens must be between 1 and {MaxContextLength}");
        }

        if (temperature <= 0 || temperature > 2.0)
        {
            throw new ArgumentOutOfRangeException(nameof(temperature), 
                "Temperature must be between 0 (exclusive) and 2.0");
        }

        if (topP <= 0 || topP > 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(topP), 
                "Top-p must be between 0 (exclusive) and 1.0");
        }
    }

    /// <summary>
    /// Registers an available checkpoint
    /// </summary>
    protected void RegisterCheckpoint(string name, string path)
    {
        _availableCheckpoints[name] = path;
    }

    #endregion

    #region IFullModel Implementation

    // Fields for IFullModel implementation
    protected readonly HashSet<int> _activeFeatureIndices = new HashSet<int>();
    protected Vector<T> _parameters = new Vector<T>(0);
    protected ModelType _modelType = ModelType.Transformer;
    protected readonly Dictionary<string, double> _featureImportance = new Dictionary<string, double>();

    /// <summary>
    /// Trains the model with input-output pairs
    /// </summary>
    public override void Train(string input, string expectedOutput)
    {
        // Foundation models typically use fine-tuning rather than single-example training
        // This is a simplified implementation - real training would use FineTuneAsync
        _logger.Warning("Single-example training is not typical for foundation models. Consider using FineTuneAsync for proper training.");
        
        var trainingExample = new TrainingExample
        {
            Input = input,
            Output = expectedOutput,
            Metadata = new Dictionary<string, object>()
        };

        // In a real implementation, this would accumulate examples and trigger batch training
        // For now, we just log the training request
        _logger.Debug("Received training example: {Input} -> {Output}", input, expectedOutput);
    }

    /// <summary>
    /// Makes a prediction for the given input
    /// </summary>
    public override string Predict(string input)
    {
        // Use the async generation method synchronously for the IModel interface
        try
        {
            return GenerateAsync(input, maxTokens: 100).GetAwaiter().GetResult();
        }
        catch (Exception ex)
        {
            _logger.Error(ex, "Error during prediction");
            throw;
        }
    }

    /// <summary>
    /// Gets model metadata
    /// </summary>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = _modelType,
            FeatureCount = VocabularySize, // For foundation models, vocabulary size represents feature space
            Complexity = (int)Math.Log10(ParameterCount), // Log scale of parameters as complexity measure
            Description = $"{Architecture} foundation model with {ParameterCount:N0} parameters",
            AdditionalInfo = new Dictionary<string, object>
            {
                ["Architecture"] = Architecture,
                ["ParameterCount"] = ParameterCount,
                ["MaxContextLength"] = MaxContextLength,
                ["VocabularySize"] = VocabularySize,
                ["IsInitialized"] = _isInitialized,
                ["AvailableCheckpoints"] = GetAvailableCheckpoints()
            },
            ModelData = Serialize(),
            FeatureImportance = _featureImportance
        };
    }

    /// <summary>
    /// Serializes the model to binary format
    /// </summary>
    public virtual byte[] Serialize()
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);

        // Write model metadata
        writer.Write(Architecture);
        writer.Write(ParameterCount);
        writer.Write(VocabularySize);
        writer.Write(MaxContextLength);
        writer.Write(_isInitialized);

        // Write checkpoints
        writer.Write(_availableCheckpoints.Count);
        foreach (var checkpoint in _availableCheckpoints)
        {
            writer.Write(checkpoint.Key);
            writer.Write(checkpoint.Value);
        }

        // Write parameters
        writer.Write(_parameters.Length);
        for (int i = 0; i < _parameters.Length; i++)
        {
            writer.Write(Convert.ToDouble(_parameters[i]));
        }

        // Write active features
        writer.Write(_activeFeatureIndices.Count);
        foreach (var index in _activeFeatureIndices)
        {
            writer.Write(index);
        }

        // Write feature importance
        writer.Write(_featureImportance.Count);
        foreach (var kvp in _featureImportance)
        {
            writer.Write(kvp.Key);
            writer.Write(kvp.Value);
        }

        // Derived classes should override to add model-specific data
        SerializeModelSpecificData(writer);

        return stream.ToArray();
    }

    /// <summary>
    /// Deserializes the model from binary format
    /// </summary>
    public virtual void Deserialize(byte[] data)
    {
        using var stream = new MemoryStream(data);
        using var reader = new BinaryReader(stream);

        // Read and validate architecture
        var architecture = reader.ReadString();
        if (architecture != Architecture)
        {
            throw new InvalidOperationException($"Architecture mismatch: expected {Architecture}, got {architecture}");
        }

        // Read metadata
        var paramCount = reader.ReadInt64();
        var vocabSize = reader.ReadInt32();
        var maxContext = reader.ReadInt32();
        _isInitialized = reader.ReadBoolean();

        // Read checkpoints
        _availableCheckpoints.Clear();
        var checkpointCount = reader.ReadInt32();
        for (int i = 0; i < checkpointCount; i++)
        {
            var name = reader.ReadString();
            var path = reader.ReadString();
            _availableCheckpoints[name] = path;
        }

        // Read parameters
        var parameterCount = reader.ReadInt32();
        var parameters = new T[parameterCount];
        for (int i = 0; i < parameterCount; i++)
        {
            parameters[i] = NumOps.FromDouble(reader.ReadDouble());
        }
        _parameters = new Vector<T>(parameters);

        // Read active features
        _activeFeatureIndices.Clear();
        var activeFeatureCount = reader.ReadInt32();
        for (int i = 0; i < activeFeatureCount; i++)
        {
            _activeFeatureIndices.Add(reader.ReadInt32());
        }

        // Read feature importance
        _featureImportance.Clear();
        var featureImportanceCount = reader.ReadInt32();
        for (int i = 0; i < featureImportanceCount; i++)
        {
            var key = reader.ReadString();
            var value = reader.ReadDouble();
            _featureImportance[key] = value;
        }

        // Derived classes should override to read model-specific data
        DeserializeModelSpecificData(reader);
    }

    /// <summary>
    /// Gets the model parameters
    /// </summary>
    public virtual Vector<T> GetParameters()
    {
        return (Vector<T>)_parameters.Clone();
    }

    /// <summary>
    /// Sets the model parameters
    /// </summary>
    public virtual void SetParameters(Vector<T> parameters)
    {
        if (parameters == null)
        {
            throw new ArgumentNullException(nameof(parameters));
        }

        _parameters = (Vector<T>)parameters.Clone();
        _logger.Debug("Updated model parameters. Count: {Count}", parameters.Length);

        // In a real implementation, this would update the actual model weights
        OnParametersUpdated();
    }

    /// <summary>
    /// Creates a new instance with the specified parameters
    /// </summary>
    public virtual IFullModel<T, string, string> WithParameters(Vector<T> parameters)
    {
        // This requires creating a new instance of the derived class
        // Since we can't instantiate abstract classes, derived classes must override this
        throw new NotImplementedException(
            "WithParameters must be implemented by derived classes to create new instances.");
    }

    /// <summary>
    /// Gets the indices of active features
    /// </summary>
    public virtual IEnumerable<int> GetActiveFeatureIndices()
    {
        return _activeFeatureIndices.ToList();
    }

    /// <summary>
    /// Checks if a feature is used
    /// </summary>
    public virtual bool IsFeatureUsed(int featureIndex)
    {
        return _activeFeatureIndices.Contains(featureIndex);
    }

    /// <summary>
    /// Sets the active feature indices
    /// </summary>
    public virtual void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
        if (featureIndices == null)
        {
            throw new ArgumentNullException(nameof(featureIndices));
        }

        _activeFeatureIndices.Clear();
        foreach (var index in featureIndices)
        {
            if (index >= 0 && index < VocabularySize)
            {
                _activeFeatureIndices.Add(index);
            }
        }

        _logger.Debug("Set {Count} active feature indices", _activeFeatureIndices.Count);
    }

    /// <summary>
    /// Creates a deep copy of the model
    /// </summary>
    public virtual IFullModel<T, string, string> DeepCopy()
    {
        // Serialize and deserialize to create a deep copy
        // Derived classes should override for more efficient implementation
        var serialized = Serialize();
        var copy = CreateNewInstance();
        copy.Deserialize(serialized);
        return copy;
    }

    /// <summary>
    /// Creates a shallow copy of the model
    /// </summary>
    public virtual IFullModel<T, string, string> Clone()
    {
        // For foundation models, clone typically means deep copy due to complexity
        return DeepCopy();
    }

    // Additional abstract method implementations
    public override async Task<string> PredictAsync(string input)
    {
        return await GenerateAsync(input, maxTokens: 100);
    }
    
    public override async Task TrainAsync(string input, string expectedOutput)
    {
        // Use the synchronous Train method for now
        await Task.Run(() => Train(input, expectedOutput));
    }
    
    public override void SetModelMetadata(ModelMetadata<T> metadata)
    {
        // Update relevant fields from the provided metadata
        if (metadata.ModelType != ModelType.Unknown)
        {
            _modelType = metadata.ModelType;
        }
        
        if (metadata.FeatureImportance != null)
        {
            _featureImportance.Clear();
            foreach (var kvp in metadata.FeatureImportance)
            {
                _featureImportance[kvp.Key] = Convert.ToDouble(kvp.Value);
            }
        }
    }
    
    public override void Save(string filepath)
    {
        var data = Serialize();
        File.WriteAllBytes(filepath, data);
        _logger.Information("Saved foundation model to {FilePath}", filepath);
    }
    
    public override void Load(string filepath)
    {
        var data = File.ReadAllBytes(filepath);
        Deserialize(data);
        _logger.Information("Loaded foundation model from {FilePath}", filepath);
    }
    
    public override void Dispose()
    {
        _initSemaphore?.Dispose();
        _activeFeatureIndices.Clear();
        _availableCheckpoints.Clear();
        _featureImportance.Clear();
        _logger.Information("Foundation model disposed");
    }
    
    // Override interpretability methods with foundation model-specific implementations
    public override async Task<Dictionary<int, T>> GetGlobalFeatureImportanceAsync()
    {
        _logger.Warning("Traditional feature importance not applicable to foundation models. Consider using attention analysis.");
        var importance = new Dictionary<int, T>();
        // Return empty dictionary as foundation models don't have traditional feature importance
        return importance;
    }
    
    public override async Task<string> GenerateTextExplanationAsync(string input, string prediction)
    {
        // Foundation models excel at generating natural language explanations
        var explanationPrompt = $"Explain why the input '{input}' resulted in the output '{prediction}'.";
        
        try
        {
            return await GenerateAsync(explanationPrompt, maxTokens: 150);
        }
        catch (Exception ex)
        {
            _logger.Error(ex, "Failed to generate text explanation");
            return $"The model generated '{prediction}' based on the input '{input}'.";
        }
    }

    #endregion

    #region Protected Virtual Methods

    /// <summary>
    /// Serializes model-specific data. Override in derived classes.
    /// </summary>
    protected virtual void SerializeModelSpecificData(BinaryWriter writer)
    {
        // Derived classes should override to serialize their specific data
    }

    /// <summary>
    /// Deserializes model-specific data. Override in derived classes.
    /// </summary>
    protected virtual void DeserializeModelSpecificData(BinaryReader reader)
    {
        // Derived classes should override to deserialize their specific data
    }

    /// <summary>
    /// Called when parameters are updated. Override to handle parameter changes.
    /// </summary>
    protected virtual void OnParametersUpdated()
    {
        // Derived classes can override to handle parameter updates
    }

    /// <summary>
    /// Creates a new instance of the model. Must be implemented by derived classes.
    /// </summary>
    protected abstract IFullModel<T, string, string> CreateNewInstance();

    #endregion
}