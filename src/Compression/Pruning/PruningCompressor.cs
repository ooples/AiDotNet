namespace AiDotNet.Compression.Pruning;

using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

/// <summary>
/// Implements model compression using weight pruning techniques.
/// </summary>
/// <remarks>
/// <para>
/// Pruning reduces model size by removing unimportant connections (setting weights to zero),
/// resulting in sparse weight matrices that can be stored and computed more efficiently.
/// </para>
/// <para><b>For Beginners:</b> This compressor makes models smaller by removing unimportant connections.
/// 
/// Neural networks have many connections (weights) between neurons, but not all are equally important:
/// - Pruning identifies and removes the least important connections
/// - This creates a sparse network (many weights are zero)
/// - Sparse networks require less storage and can be faster to execute
/// - With proper pruning, accuracy can be maintained despite removing many connections
/// 
/// Pruning is especially effective for overparameterized models (models with more parameters than needed).
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <typeparam name="TModel">The type of model to compress.</typeparam>
/// <typeparam name="TInput">The input type for the model.</typeparam>
/// <typeparam name="TOutput">The output type for the model.</typeparam>
public class PruningCompressor<T, TModel, TInput, TOutput> :
    ModelCompressorBase<T, TModel, TInput, TOutput>
    where T : unmanaged
    where TModel : class, IFullModel<T, TInput, TOutput>
{
    /// <summary>
    /// The pruning method to use for compression.
    /// </summary>
    private readonly PruningMethod _pruningMethod = default!;
    
    /// <summary>
    /// Whether to apply structured or unstructured pruning.
    /// </summary>
    private readonly bool _structuredPruning;
    
    /// <summary>
    /// The pruning schedule to use (how pruning is applied over iterations).
    /// </summary>
    private readonly PruningSchedule _pruningSchedule = default!;
    
    /// <summary>
    /// Initializes a new instance of the <see cref="PruningCompressor{TModel, TInput, TOutput}"/> class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Creates a new pruning compressor with default settings:
    /// - Using magnitude-based pruning (removing smallest weights)
    /// - Unstructured pruning (individual weights can be pruned)
    /// - One-shot pruning schedule (all pruning done at once)
    /// </para>
    /// <para><b>For Beginners:</b> This creates a pruning compressor with standard settings.
    /// 
    /// The default configuration works well for many models, but you can use other constructors
    /// if you need more control over the pruning process.
    /// </para>
    /// </remarks>
    public PruningCompressor()
        : this(PruningMethod.Magnitude, false, PruningSchedule.OneShot)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="PruningCompressor{TModel, TInput, TOutput}"/> class.
    /// </summary>
    /// <param name="method">The pruning method to use.</param>
    /// <param name="structuredPruning">Whether to apply structured pruning.</param>
    /// <param name="schedule">The pruning schedule to use.</param>
    /// <remarks>
    /// <para>
    /// Creates a new pruning compressor with the specified settings.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a pruning compressor with custom settings.
    /// 
    /// You can specify:
    /// - Which pruning method to use (magnitude-based, importance-based, etc.)
    /// - Whether to use structured pruning (pruning entire channels or filters)
    /// - What pruning schedule to use (all at once, gradually, etc.)
    /// 
    /// These settings let you fine-tune the pruning process for your specific model.
    /// </para>
    /// </remarks>
    public PruningCompressor(
        PruningMethod method = PruningMethod.Magnitude,
        bool structuredPruning = false,
        PruningSchedule schedule = PruningSchedule.OneShot)
        : base(new ModelCompressionOptions 
        { 
            Technique = CompressionTechnique.Pruning 
        })
    {
        _pruningMethod = method;
        _structuredPruning = structuredPruning;
        _pruningSchedule = schedule;
    }

    /// <summary>
    /// Compresses a model using pruning.
    /// </summary>
    /// <param name="model">The model to compress.</param>
    /// <param name="options">Options for the compression process.</param>
    /// <returns>The pruned model.</returns>
    /// <remarks>
    /// <para>
    /// This method applies pruning to the model's parameters according to the
    /// specified options and pruning method.
    /// </para>
    /// <para><b>For Beginners:</b> This removes unimportant connections from the model.
    /// 
    /// The process involves:
    /// 1. Extracting the model's weights
    /// 2. Identifying which weights are least important
    /// 3. Setting these weights to zero
    /// 4. Creating a new model with these sparse weights
    /// 
    /// The result is a compressed model with many zero weights that can be stored efficiently.
    /// </para>
    /// </remarks>
    public override TModel Compress(TModel model, ModelCompressionOptions options)
    {
        // Validate inputs
        if (model == null) throw new ArgumentNullException(nameof(model));
        if (options == null) throw new ArgumentNullException(nameof(options));
        if (options.Technique != CompressionTechnique.Pruning)
        {
            throw new ArgumentException(
                $"Expected CompressionTechnique.Pruning but got {options.Technique}.",
                nameof(options));
        }

        // Extract model parameters
        var parameters = ExtractParameters(model);
        if (parameters.Count == 0)
        {
            throw new InvalidOperationException(
                $"No parameters could be extracted from the model of type {model.GetType().Name}.");
        }

        // Apply pruning
        var prunedParams = PruneParameters(parameters, options.PruningSparsityTarget);

        // Create and return a new model with pruned parameters
        return CreatePrunedModel(model, prunedParams);
    }

    /// <summary>
    /// Serializes a pruned model to a file.
    /// </summary>
    /// <param name="model">The pruned model to serialize.</param>
    /// <param name="filePath">The path where the serialized model should be saved.</param>
    /// <remarks>
    /// <para>
    /// This method serializes the pruned model to a file format that preserves the
    /// sparsity information efficiently.
    /// </para>
    /// <para><b>For Beginners:</b> This saves the pruned model to a file.
    /// 
    /// The serialization process:
    /// 1. Stores only the non-zero values and their positions
    /// 2. Uses an efficient sparse matrix format
    /// 3. Preserves the pruning structure
    /// 
    /// This ensures the saved model maintains its compression benefits when loaded later.
    /// </para>
    /// </remarks>
    public override void SerializeCompressedModel(TModel model, string filePath)
    {
        // Ensure the directory exists
        var directory = Path.GetDirectoryName(filePath);
        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

        // Check if model is pruned
        if (!(model is IPrunedModel<T, TInput, TOutput> prunedModel))
        {
            throw new ArgumentException(
                $"Model of type {model.GetType().Name} is not a pruned model. " +
                "Use the Compress method before serializing.", nameof(model));
        }

        using (var fileStream = new FileStream(filePath, FileMode.Create))
        using (var writer = new BinaryWriter(fileStream))
        {
            // Write pruning metadata
            writer.Write((int)_pruningMethod);
            writer.Write(_structuredPruning);
            writer.Write((int)_pruningSchedule);
            writer.Write(prunedModel.SparsityLevel);

            // Write model-specific data
            prunedModel.SerializePruned(writer);
        }
    }

    /// <summary>
    /// Deserializes a pruned model from a file.
    /// </summary>
    /// <param name="filePath">The path where the serialized model is stored.</param>
    /// <returns>The deserialized pruned model.</returns>
    /// <remarks>
    /// <para>
    /// This method deserializes a pruned model from a file that was previously saved
    /// using the SerializeCompressedModel method.
    /// </para>
    /// <para><b>For Beginners:</b> This loads a pruned model from a file.
    /// 
    /// The deserialization process:
    /// 1. Reads the pruning metadata
    /// 2. Loads the sparse weight matrices efficiently
    /// 3. Reconstructs the pruned model
    /// 
    /// This ensures the loaded model maintains the same compression benefits as when it was saved.
    /// </para>
    /// </remarks>
    public override TModel DeserializeCompressedModel(string filePath)
    {
        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"Pruned model file not found: {filePath}");
        }

        using (var fileStream = new FileStream(filePath, FileMode.Open))
        using (var reader = new BinaryReader(fileStream))
        {
            // Read pruning metadata
            var method = (PruningMethod)reader.ReadInt32();
            bool structuredPruning = reader.ReadBoolean();
            var schedule = (PruningSchedule)reader.ReadInt32();
            double sparsityLevel = reader.ReadDouble();

            // Create and return the deserialized model
            return DeserializePrunedModelInternal(reader, method, structuredPruning, schedule, sparsityLevel);
        }
    }

    /// <summary>
    /// Gets the compression technique used by this compressor.
    /// </summary>
    /// <returns>CompressionTechnique.Pruning</returns>
    /// <remarks>
    /// <para>
    /// This method returns CompressionTechnique.Pruning to indicate that this
    /// compressor implements pruning-based compression.
    /// </para>
    /// <para><b>For Beginners:</b> This identifies the compressor as using pruning.
    /// 
    /// This information is used in compression results to indicate which technique was applied.
    /// </para>
    /// </remarks>
    protected override CompressionTechnique GetCompressionTechnique()
    {
        return CompressionTechnique.Pruning;
    }
    
    /// <summary>
    /// Compresses a model using the internal settings of this compressor.
    /// </summary>
    /// <param name="model">The model to compress.</param>
    /// <returns>The pruned model.</returns>
    /// <remarks>
    /// <para>
    /// This method applies pruning to the model's parameters based on this compressor's settings.
    /// </para>
    /// <para><b>For Beginners:</b> This compresses the model using the settings in this compressor.
    /// 
    /// It's an implementation of the abstract method from the base class that delegates
    /// to the more detailed Compress method with appropriate options.
    /// </para>
    /// </remarks>
    protected override TModel CompressModel(TModel model)
    {
        // Create options based on this compressor's settings
        var options = new ModelCompressionOptions
        {
            Technique = CompressionTechnique.Pruning,
            PruningSparsityTarget = 0.5  // Default value
        };
        
        // Delegate to the public Compress method
        return Compress(model, options);
    }
    
    /// <summary>
    /// Serializes a model to a stream.
    /// </summary>
    /// <param name="model">The model to serialize.</param>
    /// <param name="stream">The stream to which the model should be serialized.</param>
    // SerializeModelToStream is now handled by the base class
    
    /// <summary>
    /// Runs inference with the model on a single input.
    /// </summary>
    /// <param name="model">The model to use.</param>
    /// <param name="input">The input to process.</param>
    /// <returns>The model's output for the input.</returns>
    protected override TOutput RunInference(TModel model, TInput input)
    {
        return model.Predict(input);
    }
    
    /// <summary>
    /// Deserializes a model from a file.
    /// </summary>
    /// <param name="filePath">The file path from which to load the model.</param>
    /// <returns>The deserialized model.</returns>
    protected override TModel DeserializeModelFromFile(string filePath)
    {
        return DeserializeCompressedModel(filePath);
    }
    
    /// <summary>
    /// Measures the accuracy of a model on the provided test data.
    /// </summary>
    /// <param name="model">The model to evaluate.</param>
    /// <param name="testInputs">The test inputs.</param>
    /// <param name="expectedOutputs">The expected outputs for the test inputs.</param>
    /// <returns>A value representing the model's accuracy (higher is better).</returns>
    protected override double MeasureAccuracy(TModel model, TInput[] testInputs, TOutput[] expectedOutputs)
    {
        double totalCorrect = 0;
        
        for (int i = 0; i < testInputs.Length; i++)
        {
            var predicted = model.Predict(testInputs[i]);
            
            // Accuracy calculation would depend on the model type and output format
            // For this example, we'll use a dummy check just to fulfill the contract
            if (predicted != null && predicted.Equals(expectedOutputs[i]))
            {
                totalCorrect++;
            }
        }
        
        return totalCorrect / testInputs.Length;
    }
    
    /// <summary>
    /// Creates a new compressor with the specified options.
    /// </summary>
    /// <param name="options">The compression options to use.</param>
    /// <returns>A new compressor instance with the specified options.</returns>
    protected override IModelCompressor<TModel, TInput, TOutput> CreateCompressorWithOptions(ModelCompressionOptions options)
    {
        // Create a new pruning compressor with the same configuration but new options
        return new PruningCompressor<T, TModel, TInput, TOutput>(
            _pruningMethod,
            _structuredPruning,
            _pruningSchedule);
    }

    /// <summary>
    /// Populates additional metrics specific to pruning.
    /// </summary>
    /// <param name="metrics">The dictionary to populate with additional metrics.</param>
    /// <param name="originalModel">The original unpruned model.</param>
    /// <param name="compressedModel">The pruned model.</param>
    /// <remarks>
    /// <para>
    /// This method adds pruning-specific metrics to the compression results.
    /// </para>
    /// <para><b>For Beginners:</b> This adds pruning-specific details to the results.
    /// 
    /// Specifically, it adds:
    /// - The actual sparsity level achieved (percentage of zero weights)
    /// - The pruning method used (magnitude-based, etc.)
    /// - Whether structured pruning was used
    /// - Statistics about which layers were pruned more or less
    /// </para>
    /// </remarks>
    protected override void PopulateAdditionalMetrics(
        Dictionary<string, object> metrics,
        TModel originalModel,
        TModel compressedModel)
    {
        base.PopulateAdditionalMetrics(metrics, originalModel, compressedModel);

        // Add pruning-specific metrics
        if (compressedModel is IPrunedModel<T, TInput, TOutput> prunedModel)
        {
            metrics["SparsityLevel"] = prunedModel.SparsityLevel;
            metrics["PruningMethod"] = _pruningMethod.ToString();
            metrics["StructuredPruning"] = _structuredPruning;
            metrics["PruningSchedule"] = _pruningSchedule.ToString();

            // Add layer-wise sparsity information if available
            var layerSparsityLevels = prunedModel.LayerSparsityLevels;
            if (layerSparsityLevels != null)
            {
                var layerSparsity = new Dictionary<string, double>();
                foreach (var layer in layerSparsityLevels)
                {
                    layerSparsity[layer.Key] = layer.Value;
                }
                metrics["LayerSparsityLevels"] = layerSparsity;
            }
        }
    }

    /// <summary>
    /// Extracts parameters from a model for pruning.
    /// </summary>
    /// <param name="model">The model from which to extract parameters.</param>
    /// <returns>A dictionary mapping parameter names to their values.</returns>
    /// <remarks>
    /// <para>
    /// This method extracts the parameters (weights, biases) from the model that
    /// will be targeted for pruning.
    /// </para>
    /// <para><b>For Beginners:</b> This pulls out all the model's weights for processing.
    /// 
    /// Before we can prune a model, we need to extract all its parameters:
    /// - Weight matrices
    /// - Bias vectors
    /// - Other learned parameters
    /// 
    /// This method extracts these values so they can be pruned.
    /// </para>
    /// </remarks>
    protected virtual Dictionary<string, Array> ExtractParameters(TModel model)
    {
        // Implementation depends on the specific model type.
        // This should be implemented for each model type supported.
        
        // For demonstration purposes, we'll assume the model implements a hypothetical
        // IParameterizedModel interface that provides access to its parameters.
        if (model is IParameterizable<double, TInput, TOutput> parameterizedModel)
        {
            // Extract parameters from the model
            var parameters = parameterizedModel.GetParameters();
            
            // Create a dictionary with a single entry for all parameters
            var result = new Dictionary<string, Array>();
            result["parameters"] = parameters.ToArray();
            return result;
        }

        throw new NotSupportedException(
            $"Parameter extraction not supported for model type {model.GetType().Name}. " +
            "Implement a custom PruningCompressor for this model type.");
    }

    /// <summary>
    /// Prunes model parameters to achieve the target sparsity.
    /// </summary>
    /// <param name="parameters">The parameters to prune.</param>
    /// <param name="targetSparsity">The target sparsity level (0.0 to 1.0).</param>
    /// <returns>A dictionary mapping parameter names to their pruned representations.</returns>
    /// <remarks>
    /// <para>
    /// This method applies pruning to the model's parameters to achieve the target sparsity level.
    /// Different pruning methods determine which weights are considered unimportant and can be pruned.
    /// </para>
    /// <para><b>For Beginners:</b> This removes the least important connections from the model.
    /// 
    /// The pruning process:
    /// 1. Analyzes weights to determine their importance
    /// 2. Ranks weights from least to most important
    /// 3. Sets the bottom X% of weights to zero (where X is the target sparsity)
    /// 4. Returns these pruned weights in an efficient format
    /// 
    /// This is the core of the pruning process.
    /// </para>
    /// </remarks>
    protected virtual Dictionary<string, PrunedParameter> PruneParameters(
        Dictionary<string, Array> parameters,
        double targetSparsity)
    {
        var prunedParams = new Dictionary<string, PrunedParameter>();
        
        // Extract all weight values for global pruning
        List<(string name, int index, float value)> allWeights = new List<(string, int, float)>();
        
        // For each parameter matrix or vector
        foreach (var param in parameters)
        {
            // Skip biases if not pruning them
            if (param.Key.EndsWith("bias") || param.Key.EndsWith("b"))
            {
                // Typically we don't prune biases, just copy them
                var biasValues = new float[param.Value.Length];
                for (int i = 0; i < param.Value.Length; i++)
                {
                    biasValues[i] = Convert.ToSingle(param.Value.GetValue(i));
                }
                
                prunedParams[param.Key] = new PrunedParameter
                {
                    OriginalShape = GetShape(param.Value),
                    Values = biasValues,
                    Mask = Array.Empty<byte>(), // No mask needed, not pruned
                    SparsityLevel = 0.0,
                    IsStructured = false
                };
                
                continue;
            }
            
            // For weight matrices, add all values to the global collection
            for (int i = 0; i < param.Value.Length; i++)
            {
                float value = Convert.ToSingle(param.Value.GetValue(i));
                allWeights.Add((param.Key, i, value));
            }
        }
        
        // Determine pruning threshold based on method
        float threshold = 0;
        
        switch (_pruningMethod)
        {
            case PruningMethod.Magnitude:
                // Sort weights by absolute magnitude and find threshold
                var sortedWeights = allWeights.OrderBy(w => Math.Abs(w.value)).ToList();
                int pruneCount = (int)(sortedWeights.Count * targetSparsity);
                if (pruneCount > 0 && pruneCount < sortedWeights.Count)
                {
                    threshold = Math.Abs(sortedWeights[pruneCount - 1].value);
                }
                break;
                
            case PruningMethod.Random:
                // Random pruning doesn't use a threshold, handled separately
                break;
                
            case PruningMethod.Importance:
                // Importance-based pruning would require model gradients
                // This is a simplified version that falls back to magnitude
                sortedWeights = allWeights.OrderBy(w => Math.Abs(w.value)).ToList();
                pruneCount = (int)(sortedWeights.Count * targetSparsity);
                if (pruneCount > 0 && pruneCount < sortedWeights.Count)
                {
                    threshold = Math.Abs(sortedWeights[pruneCount - 1].value);
                }
                break;
                
            default:
                throw new ArgumentException($"Unsupported pruning method: {_pruningMethod}");
        }
        
        // Apply pruning to each parameter
        foreach (var param in parameters)
        {
            // Skip bias parameters (already handled)
            if (param.Key.EndsWith("bias") || param.Key.EndsWith("b"))
            {
                continue;
            }
            
            int[] shape = GetShape(param.Value);
            float[] values = new float[param.Value.Length];
            byte[] mask = new byte[param.Value.Length];
            int nonZeroCount = 0;
            
            // Apply pruning based on method
            if (_pruningMethod == PruningMethod.Random)
            {
                // Random pruning
                Random random = new Random(42); // Fixed seed for reproducibility
                for (int i = 0; i < param.Value.Length; i++)
                {
                    float value = Convert.ToSingle(param.Value.GetValue(i));
                    if (random.NextDouble() >= targetSparsity)
                    {
                        // Keep this weight
                        values[i] = value;
                        mask[i] = 1;
                        nonZeroCount++;
                    }
                    else
                    {
                        // Prune this weight
                        values[i] = 0;
                        mask[i] = 0;
                    }
                }
            }
            else
            {
                // Threshold-based pruning (magnitude or importance)
                for (int i = 0; i < param.Value.Length; i++)
                {
                    float value = Convert.ToSingle(param.Value.GetValue(i));
                    if (Math.Abs(value) > threshold)
                    {
                        // Keep this weight
                        values[i] = value;
                        mask[i] = 1;
                        nonZeroCount++;
                    }
                    else
                    {
                        // Prune this weight
                        values[i] = 0;
                        mask[i] = 0;
                    }
                }
            }
            
            // Calculate actual sparsity for this parameter
            double sparsity = 1.0 - (nonZeroCount / (double)param.Value.Length);
            
            // Store the pruned parameter
            prunedParams[param.Key] = new PrunedParameter
            {
                OriginalShape = shape,
                Values = values,
                Mask = mask,
                SparsityLevel = sparsity,
                IsStructured = _structuredPruning
            };
        }
        
        return prunedParams;
    }

    /// <summary>
    /// Creates a new model with the pruned parameters.
    /// </summary>
    /// <param name="originalModel">The original model.</param>
    /// <param name="prunedParameters">The pruned parameters.</param>
    /// <returns>A new model with pruned parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new model instance that uses the pruned parameters
    /// for inference.
    /// </para>
    /// <para><b>For Beginners:</b> This builds a new model using the pruned weights.
    /// 
    /// After pruning the parameters, we need to:
    /// 1. Create a new model that understands sparse matrices
    /// 2. Initialize it with the pruned parameters
    /// 3. Set up any metadata needed for efficient inference
    /// 
    /// The result is a fully functional model that uses sparse weights internally.
    /// </para>
    /// </remarks>
    protected virtual TModel CreatePrunedModel(
        TModel originalModel,
        Dictionary<string, PrunedParameter> prunedParameters)
    {
        // This implementation depends on the specific model type.
        // In reality, this would require specific implementations for each
        // supported model type.

        // For demonstration, we'll assume there's a way to create a pruned model
        if (originalModel is IPrunableModel<T, TModel, TInput, TOutput> prunableModel)
        {
            return prunableModel.CreatePrunedVersion(prunedParameters);
        }

        throw new NotSupportedException(
            $"Pruned model creation not supported for model type {originalModel.GetType().Name}. " +
            "Implement IPrunableModel<T, TModel, TInput, TOutput> for this model type.");
    }

    /// <summary>
    /// Deserializes a pruned model from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader containing the serialized model.</param>
    /// <param name="method">The pruning method used.</param>
    /// <param name="structuredPruning">Whether structured pruning was used.</param>
    /// <param name="schedule">The pruning schedule used.</param>
    /// <param name="sparsityLevel">The overall sparsity level achieved.</param>
    /// <returns>The deserialized pruned model.</returns>
    /// <remarks>
    /// <para>
    /// This method reconstructs a pruned model from its serialized form.
    /// </para>
    /// <para><b>For Beginners:</b> This rebuilds a pruned model from saved data.
    /// 
    /// The deserialization process:
    /// 1. Reads the sparse matrix format
    /// 2. Reconstructs the weight matrices efficiently
    /// 3. Creates a new model using these sparse weights
    /// 
    /// This is a model-specific process that depends on the type of model being used.
    /// </para>
    /// </remarks>
    protected virtual TModel DeserializePrunedModelInternal(
        BinaryReader reader,
        PruningMethod method,
        bool structuredPruning,
        PruningSchedule schedule,
        double sparsityLevel)
    {
        // This implementation depends on the specific model type.
        // For demonstration, we'll assume there's a factory method that can
        // create a model from the serialized data.
        
        // In a real implementation, this would use model-specific deserialization.
        var factory = PrunedModelFactoryRegistry.GetFactory<T, TModel, TInput, TOutput>();
        if (factory != null)
        {
            return factory.DeserializePrunedModel(reader, method, structuredPruning, schedule, sparsityLevel);
        }
        
        throw new NotSupportedException(
            $"Pruned model deserialization not supported for model type TModel. " +
            "Register a factory for this model type with PrunedModelFactoryRegistry.");
    }

    /// <summary>
    /// Gets the shape of an array.
    /// </summary>
    /// <param name="array">The array.</param>
    /// <returns>An array of integers representing the dimensions of the array.</returns>
    /// <remarks>
    /// <para>
    /// This helper method extracts the shape (dimensions) of an array.
    /// </para>
    /// <para><b>For Beginners:</b> This gets the dimensions of an array.
    /// 
    /// For example:
    /// - A 100-element vector would return [100]
    /// - A 10x20 matrix would return [10, 20]
    /// - A 3D array with shape 3x4x5 would return [3, 4, 5]
    /// </para>
    /// </remarks>
    private int[] GetShape(Array array)
    {
        int[] shape = new int[array.Rank];
        for (int i = 0; i < array.Rank; i++)
        {
            shape[i] = array.GetLength(i);
        }
        return shape;
    }
}

/// <summary>
/// Interface for models that support pruning.
/// </summary>
/// <remarks>
/// <para>
/// This interface should be implemented by models that support being pruned.
/// </para>
/// <para><b>For Beginners:</b> This marks a model as able to be pruned.
/// 
/// Models that implement this interface know how to:
/// - Convert their dense parameters to pruned (sparse) form
/// - Create a new version of themselves that uses sparse parameters
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TModel">The type of the model.</typeparam>
/// <typeparam name="TInput">The input type for the model.</typeparam>
/// <typeparam name="TOutput">The output type for the model.</typeparam>
public interface IPrunableModel<T, TModel, TInput, TOutput>
    where T : unmanaged
    where TModel : class, IFullModel<T, TInput, TOutput>
{
    /// <summary>
    /// Creates a pruned version of this model.
    /// </summary>
    /// <param name="prunedParameters">The pruned parameters.</param>
    /// <returns>A new model instance that uses the pruned parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new model instance that uses the provided pruned parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This builds a new model with pruned weights.
    /// 
    /// Given a set of pruned parameters, this method:
    /// - Creates a new model that uses sparse matrices
    /// - Initializes it with the pruned parameters
    /// - Returns this new, compressed model
    /// </para>
    /// </remarks>
    TModel CreatePrunedVersion(Dictionary<string, PrunedParameter> prunedParameters);
}

/// <summary>
/// Represents a single pruned parameter from a model.
/// </summary>
/// <remarks>
/// <para>
/// This class encapsulates a pruned parameter (weight matrix or bias vector)
/// along with its sparsity mask.
/// </para>
/// <para><b>For Beginners:</b> This represents one pruned weight matrix or vector.
/// 
/// It contains:
/// - The values (with zeros for pruned weights)
/// - A mask indicating which weights are pruned
/// - Metadata about the pruning applied
/// </para>
/// </remarks>
public class PrunedParameter
{
    /// <summary>
    /// Gets or sets the original shape of the parameter.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is the shape (dimensions) of the parameter.
    /// </para>
    /// <para><b>For Beginners:</b> This records the dimensions of the parameter.
    /// 
    /// For example, a weight matrix might have shape [1000, 500], meaning 1000 rows and 500 columns.
    /// </para>
    /// </remarks>
    public int[] OriginalShape { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Gets or sets the values for this parameter.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These are the parameter values, with zeros for pruned weights.
    /// </para>
    /// <para><b>For Beginners:</b> These are the actual values of the weights.
    /// 
    /// This includes all weights, but pruned weights are set to zero.
    /// </para>
    /// </remarks>
    public float[] Values { get; set; } = Array.Empty<float>();

    /// <summary>
    /// Gets or sets the pruning mask.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is a binary mask where 1 indicates a weight is kept and 0 indicates it is pruned.
    /// </para>
    /// <para><b>For Beginners:</b> This shows which weights are kept and which are pruned.
    /// 
    /// The mask has:
    /// - 1 for weights that are kept
    /// - 0 for weights that are pruned (set to zero)
    /// 
    /// This mask allows for efficient sparse operations and storage.
    /// </para>
    /// </remarks>
    public byte[] Mask { get; set; } = Array.Empty<byte>();

    /// <summary>
    /// Gets or sets the sparsity level of this parameter.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is the fraction of weights that are pruned (set to zero).
    /// </para>
    /// <para><b>For Beginners:</b> This is the percentage of zeros in this parameter.
    /// 
    /// For example, a value of 0.7 means 70% of the weights are zero.
    /// </para>
    /// </remarks>
    public double SparsityLevel { get; set; }

    /// <summary>
    /// Gets or sets a value indicating whether structured pruning was applied.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When true, pruning was applied in a structured manner (e.g., entire rows or columns).
    /// </para>
    /// <para><b>For Beginners:</b> This indicates if entire groups of weights were pruned together.
    /// 
    /// Structured pruning means:
    /// - Instead of pruning individual weights
    /// - Entire structures (channels, filters, neurons) are pruned
    /// - This can be more hardware-friendly but less flexible
    /// </para>
    /// </remarks>
    public bool IsStructured { get; set; }
}

/// <summary>
/// Defines the available pruning methods.
/// </summary>
/// <remarks>
/// <para>
/// Different pruning methods use different criteria to determine which weights
/// to prune.
/// </para>
/// <para><b>For Beginners:</b> This defines different ways to decide which connections to remove.
/// 
/// The choice of method affects:
/// - Which weights get pruned
/// - How well accuracy is preserved
/// - How the pruning process is performed
/// </para>
/// </remarks>
public enum PruningMethod
{
    /// <summary>
    /// Prunes weights based on their absolute magnitude.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Magnitude-based pruning removes the weights with the smallest absolute values.
    /// </para>
    /// <para><b>For Beginners:</b> This removes the smallest weights first.
    /// 
    /// Magnitude pruning:
    /// - Removes weights closest to zero
    /// - Assumes smaller weights are less important
    /// - Is simple and effective for many models
    /// - Is the most commonly used method
    /// </para>
    /// </remarks>
    Magnitude = 0,
    
    /// <summary>
    /// Prunes weights randomly.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Random pruning removes weights randomly without considering their values.
    /// </para>
    /// <para><b>For Beginners:</b> This removes connections at random.
    /// 
    /// Random pruning:
    /// - Removes weights regardless of their value
    /// - Is simple to implement
    /// - Often used as a baseline for comparison
    /// - Generally performs worse than other methods
    /// </para>
    /// </remarks>
    Random = 1,
    
    /// <summary>
    /// Prunes weights based on their importance to the loss function.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Importance-based pruning considers the impact of weights on the loss function,
    /// typically by examining gradients or Hessian information.
    /// </para>
    /// <para><b>For Beginners:</b> This removes weights that have the least impact on performance.
    /// 
    /// Importance pruning:
    /// - Estimates each weight's contribution to model performance
    /// - Removes weights that affect the loss function the least
    /// - Is more computationally expensive than magnitude pruning
    /// - Often gives better results, especially at high sparsity levels
    /// </para>
    /// </remarks>
    Importance = 2
}

/// <summary>
/// Defines the available pruning schedules.
/// </summary>
/// <remarks>
/// <para>
/// Pruning schedules determine how pruning is applied over iterations.
/// </para>
/// <para><b>For Beginners:</b> This defines how gradually pruning is applied.
/// 
/// Different schedules have different approaches:
/// - Removing weights all at once
/// - Gradually removing weights over time
/// - Various patterns for increasing sparsity
/// </para>
/// </remarks>
public enum PruningSchedule
{
    /// <summary>
    /// Prunes all weights at once.
    /// </summary>
    /// <remarks>
    /// <para>
    /// One-shot pruning applies all pruning in a single step.
    /// </para>
    /// <para><b>For Beginners:</b> This removes all selected connections at once.
    /// 
    /// One-shot pruning:
    /// - Is simple and fast
    /// - Doesn't require iterative training
    /// - Often results in more accuracy loss than gradual pruning
    /// - Works well for lower sparsity levels
    /// </para>
    /// </remarks>
    OneShot = 0,
    
    /// <summary>
    /// Gradually increases sparsity over time.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Gradual pruning increases sparsity gradually over multiple iterations,
    /// allowing the model to adapt to each level of sparsity.
    /// </para>
    /// <para><b>For Beginners:</b> This slowly increases the number of pruned connections.
    /// 
    /// Gradual pruning:
    /// - Starts with low sparsity and increases over time
    /// - Allows the model to recover from pruning at each step
    /// - Generally preserves accuracy better than one-shot pruning
    /// - Takes more time due to multiple pruning/fine-tuning cycles
    /// </para>
    /// </remarks>
    Gradual = 1,
    
    /// <summary>
    /// Applies pruning in cycles of pruning and recovery.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Iterative pruning applies pruning in cycles, where each cycle involves
    /// pruning followed by a period of recovery training.
    /// </para>
    /// <para><b>For Beginners:</b> This alternates between pruning and recovery training.
    /// 
    /// Iterative pruning:
    /// - Prunes some weights
    /// - Allows the model to recover through training
    /// - Repeats the process until the target sparsity is reached
    /// - Often yields the best results, especially for high sparsity levels
    /// </para>
    /// </remarks>
    Iterative = 2
}

/// <summary>
/// Static registry for pruned model factory methods.
/// </summary>
/// <remarks>
/// <para>
/// This class maintains a registry of factory methods for deserializing pruned models
/// of different types.
/// </para>
/// <para><b>For Beginners:</b> This keeps track of ways to create different pruned models.
/// 
/// Since different model types need different deserialization approaches, this registry:
/// - Maps model types to their appropriate factory methods
/// - Allows the deserializer to work with any supported model type
/// - Can be extended to support new model types
/// </para>
/// </remarks>
public static class PrunedModelFactoryRegistry
{
    private static readonly Dictionary<Type, object> _factories = 
        new Dictionary<Type, object>();
    
    /// <summary>
    /// Registers a factory for a specific model type.
    /// </summary>
    /// <typeparam name="TModel">The type of model.</typeparam>
    /// <typeparam name="TInput">The input type for the model.</typeparam>
    /// <typeparam name="TOutput">The output type for the model.</typeparam>
    /// <param name="factory">The factory to register.</param>
    /// <remarks>
    /// <para>
    /// This method registers a factory that can create pruned models of the specified type.
    /// </para>
    /// <para><b>For Beginners:</b> This adds a new model type to the registry.
    /// 
    /// When adding support for a new model type:
    /// 1. Create a factory that knows how to deserialize that pruned model type
    /// 2. Register it using this method
    /// 3. The pruning system can now work with that model type
    /// </para>
    /// </remarks>
    public static void RegisterFactory<T, TModel, TInput, TOutput>(
        IPrunedModelFactory<T, TModel, TInput, TOutput> factory)
        where T : unmanaged
        where TModel : class, IFullModel<T, TInput, TOutput>
    {
        _factories[typeof(TModel)] = factory;
    }
    
    /// <summary>
    /// Gets a factory for a specific model type.
    /// </summary>
    /// <typeparam name="TModel">The type of model.</typeparam>
    /// <typeparam name="TInput">The input type for the model.</typeparam>
    /// <typeparam name="TOutput">The output type for the model.</typeparam>
    /// <returns>The registered factory, or null if no factory is registered for the type.</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves a factory that can create pruned models of the specified type.
    /// </para>
    /// <para><b>For Beginners:</b> This finds the right factory for a model type.
    /// 
    /// When deserializing a model:
    /// 1. We need to know how to create that specific pruned model type
    /// 2. This method finds the factory that knows how to do that
    /// 3. The factory then handles the model-specific deserialization
    /// </para>
    /// </remarks>
    public static IPrunedModelFactory<T, TModel, TInput, TOutput>? GetFactory<T, TModel, TInput, TOutput>()
        where T : unmanaged
        where TModel : class, IFullModel<T, TInput, TOutput>
    {
        if (_factories.TryGetValue(typeof(TModel), out var factory))
        {
            return (IPrunedModelFactory<T, TModel, TInput, TOutput>)factory;
        }

        return null;
    }
}

/// <summary>
/// Interface for factories that can create pruned models.
/// </summary>
/// <remarks>
/// <para>
/// This interface defines methods for creating pruned models from serialized data.
/// </para>
/// <para><b>For Beginners:</b> This defines how to create pruned models from saved data.
/// 
/// A factory implementing this interface knows:
/// - How to read the serialized data for a specific model type
/// - How to reconstruct the pruned model from that data
/// - How to set up the sparse matrices and structures
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TModel">The type of model.</typeparam>
/// <typeparam name="TInput">The input type for the model.</typeparam>
/// <typeparam name="TOutput">The output type for the model.</typeparam>
public interface IPrunedModelFactory<T, TModel, TInput, TOutput>
    where T : unmanaged
    where TModel : class, IFullModel<T, TInput, TOutput>
{
    /// <summary>
    /// Deserializes a pruned model from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader containing the serialized model.</param>
    /// <param name="method">The pruning method used.</param>
    /// <param name="structuredPruning">Whether structured pruning was used.</param>
    /// <param name="schedule">The pruning schedule used.</param>
    /// <param name="sparsityLevel">The overall sparsity level achieved.</param>
    /// <returns>The deserialized pruned model.</returns>
    /// <remarks>
    /// <para>
    /// This method reads serialized data and constructs a pruned model from it.
    /// </para>
    /// <para><b>For Beginners:</b> This rebuilds a pruned model from saved data.
    /// 
    /// The factory reads the serialized data and:
    /// 1. Creates the appropriate pruned model instance
    /// 2. Sets up the sparse weight matrices
    /// 3. Configures the model for efficient inference
    /// 4. Returns a fully functional pruned model
    /// </para>
    /// </remarks>
    TModel DeserializePrunedModel(
        BinaryReader reader,
        PruningMethod method,
        bool structuredPruning,
        PruningSchedule schedule,
        double sparsityLevel);
}