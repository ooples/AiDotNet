using AiDotNet.Compression.Logging;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using System;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;

namespace AiDotNet.Compression;

/// <summary>
/// Base class for all model compressors providing common functionality.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <typeparam name="TModel">The type of model to compress.</typeparam>
/// <typeparam name="TInput">The input type for the model.</typeparam>
/// <typeparam name="TOutput">The output type for the model.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This is the foundation class for all model compression techniques.
/// It handles common tasks like measuring model size, tracking compression progress,
/// evaluating accuracy impact, and logging important information throughout the process.
/// </para>
/// </remarks>
public abstract class ModelCompressorBase<T, TModel, TInput, TOutput> : IModelCompressor<TModel, TInput, TOutput>
    where TModel : class, IFullModel<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the compression options that configure the compression behavior.
    /// </summary>
    protected ModelCompressionOptions Options { get; }
    
    /// <summary>
    /// Gets the logger for recording compression activities and metrics.
    /// </summary>
    protected CompressionLogger Logger { get; }
    
    /// <summary>
    /// Initializes a new instance of the ModelCompressorBase class.
    /// </summary>
    /// <param name="options">The options that configure compression behavior.</param>
    protected ModelCompressorBase(ModelCompressionOptions options)
    {
        Options = options ?? throw new ArgumentNullException(nameof(options));
        
        // Initialize the logger
        Logger = new CompressionLogger(
            options.LoggingOptions ?? new LoggingOptions { IsEnabled = false },
            GetModelName(typeof(TModel)),
            GetCompressionTechniqueName());
    }
    
    /// <summary>
    /// Compresses the specified model according to the configured options.
    /// </summary>
    /// <param name="model">The model to compress.</param>
    /// <returns>The compressed model.</returns>
    public TModel Compress(TModel model)
    {
        // Measure the original model size
        long originalSize = MeasureModelSize(model);
        
        // Log the start of compression
        Logger.LogCompressionStart(originalSize, Options.TargetCompressionRatio);
        
        try
        {
            // Perform the actual compression (implemented by derived classes)
            var stopwatch = Stopwatch.StartNew();
            var compressedModel = CompressModel(model);
            stopwatch.Stop();
            
            // Measure the compressed model size
            long compressedSize = MeasureModelSize(compressedModel);
            double compressionRatio = (double)originalSize / compressedSize;
            
            // Log compression metrics
            Logger.LogCompressionMetric("CompressionTimeMs", stopwatch.ElapsedMilliseconds);
            
            // Log completion
            Logger.LogCompressionComplete(
                compressedSize,
                compressionRatio);
            
            return compressedModel;
        }
        catch (Exception ex)
        {
            Logger.LogError(ex, "Error occurred during model compression");
            throw;
        }
    }
    
    /// <summary>
    /// Compresses the specified model according to the provided options.
    /// </summary>
    /// <param name="model">The model to compress.</param>
    /// <param name="options">Options for the compression process.</param>
    /// <returns>The compressed model.</returns>
    public virtual TModel Compress(TModel model, ModelCompressionOptions options)
    {
        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }
        
        // Create a new compressor with the provided options
        var compressor = CreateCompressorWithOptions(options);
        
        // Use the new compressor to compress the model
        return compressor.Compress(model, options);
    }
    
    /// <summary>
    /// Compresses the specified model according to the provided options.
    /// </summary>
    /// <param name="model">The model to compress.</param>
    /// <param name="options">Options for the compression process.</param>
    /// <returns>The compressed model.</returns>
    public virtual TModel Compress(TModel model, object options)
    {
        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }
        
        if (options is ModelCompressionOptions compressionOptions)
        {
            return Compress(model, compressionOptions);
        }
        
        throw new ArgumentException($"Options must be of type {nameof(ModelCompressionOptions)}", nameof(options));
    }
    
    /// <summary>
    /// Creates a new compressor with the specified options.
    /// </summary>
    /// <param name="options">The compression options to use.</param>
    /// <returns>A new compressor instance with the specified options.</returns>
    protected abstract IModelCompressor<TModel, TInput, TOutput> CreateCompressorWithOptions(ModelCompressionOptions options);
    
    /// <summary>
    /// Asynchronously compresses the specified model according to the configured options.
    /// </summary>
    /// <param name="model">The model to compress.</param>
    /// <returns>A task representing the asynchronous operation, containing the compressed model when complete.</returns>
    public async Task<TModel> CompressAsync(TModel model)
    {
        // Measure the original model size
        long originalSize = MeasureModelSize(model);
        
        // Log the start of compression
        Logger.LogCompressionStart(originalSize, Options.TargetCompressionRatio);
        
        try
        {
            // Perform the actual compression (implemented by derived classes)
            var stopwatch = Stopwatch.StartNew();
            var compressedModel = await CompressModelAsync(model);
            stopwatch.Stop();
            
            // Measure the compressed model size
            long compressedSize = MeasureModelSize(compressedModel);
            double compressionRatio = (double)originalSize / compressedSize;
            
            // Log compression metrics
            Logger.LogCompressionMetric("CompressionTimeMs", stopwatch.ElapsedMilliseconds);
            
            // Log completion
            Logger.LogCompressionComplete(
                compressedSize,
                compressionRatio);
            
            return compressedModel;
        }
        catch (Exception ex)
        {
            Logger.LogError(ex, "Error occurred during asynchronous model compression");
            throw;
        }
    }
    
    /// <summary>
    /// Asynchronously compresses the specified model according to the provided options.
    /// </summary>
    /// <param name="model">The model to compress.</param>
    /// <param name="options">Options for the compression process.</param>
    /// <returns>A task representing the asynchronous operation, containing the compressed model when complete.</returns>
    public async Task<TModel> CompressAsync(TModel model, ModelCompressionOptions options)
    {
        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }
        
        // Create a new compressor with the provided options
        var compressor = CreateCompressorWithOptions(options);
        
        // Use the new compressor to compress the model asynchronously
        return await compressor.CompressAsync(model, options);
    }
    
    /// <summary>
    /// Asynchronously compresses the specified model according to the provided options.
    /// </summary>
    /// <param name="model">The model to compress.</param>
    /// <param name="options">Options for the compression process.</param>
    /// <returns>A task representing the asynchronous operation, containing the compressed model when complete.</returns>
    public async Task<TModel> CompressAsync(TModel model, object options)
    {
        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }
        
        if (options is ModelCompressionOptions compressionOptions)
        {
            return await CompressAsync(model, compressionOptions);
        }
        
        throw new ArgumentException($"Options must be of type {nameof(ModelCompressionOptions)}", nameof(options));
    }
    
    /// <summary>
    /// Evaluates the impact of compression by comparing the original and compressed models.
    /// </summary>
    /// <param name="originalModel">The original uncompressed model.</param>
    /// <param name="compressedModel">The compressed model.</param>
    /// <param name="testInputs">Test inputs to evaluate model performance.</param>
    /// <param name="expectedOutputs">Expected outputs for the test inputs.</param>
    /// <returns>A result containing metrics about the compression.</returns>
    public ModelCompressionResult EvaluateCompression(
        TModel originalModel, 
        TModel compressedModel, 
        TInput[] testInputs, 
        TOutput[] expectedOutputs)
    {
        if (testInputs == null || expectedOutputs == null)
        {
            throw new ArgumentNullException(
                testInputs == null ? nameof(testInputs) : nameof(expectedOutputs),
                "Test data is required to evaluate compression impact");
        }
        
        if (testInputs.Length != expectedOutputs.Length)
        {
            throw new ArgumentException("The number of test inputs must match the number of expected outputs");
        }
        
        Logger.LogCompressionProgress("Evaluation", 0.0, "Starting compression evaluation");
        
        try
        {
            // Measure sizes
            long originalSize = MeasureModelSize(originalModel);
            long compressedSize = MeasureModelSize(compressedModel);
            double compressionRatio = (double)originalSize / compressedSize;
            
            Logger.LogCompressionProgress("Evaluation", 0.2, "Size measurements complete");
            
            // Measure accuracy impact
            double originalAccuracy = MeasureAccuracy(originalModel, testInputs, expectedOutputs);
            double compressedAccuracy = MeasureAccuracy(compressedModel, testInputs, expectedOutputs);
            double accuracyImpact = compressedAccuracy - originalAccuracy;
            
            Logger.LogCompressionProgress("Evaluation", 0.5, "Accuracy measurements complete");
            
            // Measure inference speed
            double originalSpeed = MeasureInferenceSpeed(originalModel, testInputs);
            double compressedSpeed = MeasureInferenceSpeed(compressedModel, testInputs);
            double speedupFactor = compressedSpeed / originalSpeed;
            
            Logger.LogCompressionProgress("Evaluation", 0.8, "Speed measurements complete");
            
            // Measure memory usage
            double originalMemory = MeasureMemoryUsage(originalModel);
            double compressedMemory = MeasureMemoryUsage(compressedModel);
            double memoryReduction = 1 - (compressedMemory / originalMemory);
            
            Logger.LogCompressionProgress("Evaluation", 1.0, "Evaluation complete");
            
            // Create and return the result
            var result = new ModelCompressionResult
            {
                OriginalModelSizeBytes = originalSize,
                CompressedModelSizeBytes = compressedSize,
                CompressionRatio = compressionRatio,
                OriginalAccuracy = originalAccuracy,
                CompressedAccuracy = compressedAccuracy,
                AccuracyImpact = accuracyImpact,
                InferenceSpeedupFactor = speedupFactor,
                MemoryReduction = memoryReduction,
                CompressionTechniqueName = GetCompressionTechniqueName(),
                Technique = Options.Technique,
                OriginalInferenceTimeMs = originalSpeed > 0 ? 1000.0 / originalSpeed : 0,
                CompressedInferenceTimeMs = compressedSpeed > 0 ? 1000.0 / compressedSpeed : 0,
                CompressionTimeMs = 0, // This is set elsewhere
                CompressionDevice = "CPU" // Default value
            };
            
            // Log the evaluation result
            Logger.LogCompressionComplete(
                compressedSize,
                compressionRatio,
                accuracyImpact,
                speedupFactor);
                
            return result;
        }
        catch (Exception ex)
        {
            Logger.LogError(ex, "Error during compression evaluation");
            throw;
        }
    }
    
    /// <summary>
    /// Serializes a compressed model to the specified file path.
    /// </summary>
    /// <param name="model">The compressed model to serialize.</param>
    /// <param name="filePath">The file path where the model should be saved.</param>
    public virtual void SerializeCompressedModel(TModel model, string filePath)
    {
        if (model == null)
        {
            throw new ArgumentNullException(nameof(model));
        }
        
        if (string.IsNullOrEmpty(filePath))
        {
            throw new ArgumentException("File path cannot be null or empty", nameof(filePath));
        }
        
        try
        {
            // Ensure the directory exists
            Directory.CreateDirectory(Path.GetDirectoryName(filePath) ?? string.Empty);
            
            Logger.LogCompressionProgress("Serialization", 0.0, $"Serializing compressed model to {filePath}");
            
            // Perform serialization (specific to each compression technique)
            SerializeModelToFile(model, filePath);
            
            Logger.LogCompressionProgress("Serialization", 1.0, "Serialization complete");
        }
        catch (Exception ex)
        {
            Logger.LogError(ex, "Error serializing compressed model to {FilePath}", filePath);
            throw;
        }
    }
    
    /// <summary>
    /// Deserializes a compressed model from the specified file path.
    /// </summary>
    /// <param name="filePath">The file path from which to load the model.</param>
    /// <returns>The deserialized compressed model.</returns>
    public virtual TModel DeserializeCompressedModel(string filePath)
    {
        if (string.IsNullOrEmpty(filePath))
        {
            throw new ArgumentException("File path cannot be null or empty", nameof(filePath));
        }
        
        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException("Compressed model file not found", filePath);
        }
        
        try
        {
            Logger.LogCompressionProgress("Deserialization", 0.0, $"Deserializing compressed model from {filePath}");
            
            // Perform deserialization (specific to each compression technique)
            var model = DeserializeModelFromFile(filePath);
            
            Logger.LogCompressionProgress("Deserialization", 1.0, "Deserialization complete");
            
            return model;
        }
        catch (Exception ex)
        {
            Logger.LogError(ex, "Error deserializing compressed model from {FilePath}", filePath);
            throw;
        }
    }
    
    /// <summary>
    /// Implemented by derived classes to perform the actual model compression.
    /// </summary>
    /// <param name="model">The model to compress.</param>
    /// <returns>The compressed model.</returns>
    protected abstract TModel CompressModel(TModel model);
    
    /// <summary>
    /// Implemented by derived classes to perform the actual model compression asynchronously.
    /// </summary>
    /// <param name="model">The model to compress.</param>
    /// <returns>A task representing the asynchronous operation, containing the compressed model when complete.</returns>
    protected virtual Task<TModel> CompressModelAsync(TModel model)
    {
        // Default implementation calls the synchronous version
        return Task.FromResult(CompressModel(model));
    }
    
    /// <summary>
    /// Measures the size of a model in bytes.
    /// </summary>
    /// <param name="model">The model to measure.</param>
    /// <returns>The size of the model in bytes.</returns>
    protected virtual long MeasureModelSize(TModel model)
    {
        // Default implementation uses serialization to measure size
        using var memoryStream = new MemoryStream();
        SerializeModelToStream(model, memoryStream);
        return memoryStream.Length;
    }
    
    /// <summary>
    /// Measures the accuracy of a model on the provided test data.
    /// </summary>
    /// <param name="model">The model to evaluate.</param>
    /// <param name="testInputs">The test inputs.</param>
    /// <param name="expectedOutputs">The expected outputs for the test inputs.</param>
    /// <returns>A value representing the model's accuracy (higher is better).</returns>
    protected abstract double MeasureAccuracy(TModel model, TInput[] testInputs, TOutput[] expectedOutputs);
    
    /// <summary>
    /// Measures the inference speed of a model in samples per second.
    /// </summary>
    /// <param name="model">The model to measure.</param>
    /// <param name="testInputs">Test inputs to use for measurement.</param>
    /// <returns>The inference speed in samples per second.</returns>
    protected virtual double MeasureInferenceSpeed(TModel model, TInput[] testInputs)
    {
        const int warmupRuns = 5;
        const int measurementRuns = 20;
        
        // Warmup runs to eliminate JIT compilation effects
        for (int i = 0; i < warmupRuns; i++)
        {
            RunInference(model, testInputs[i % testInputs.Length]);
        }
        
        // Measurement runs
        var stopwatch = Stopwatch.StartNew();
        
        for (int i = 0; i < measurementRuns; i++)
        {
            RunInference(model, testInputs[i % testInputs.Length]);
        }
        
        stopwatch.Stop();
        
        // Calculate samples per second
        double samplesPerSecond = (double)measurementRuns / stopwatch.Elapsed.TotalSeconds;
        
        Logger.LogCompressionMetric("InferenceSamplesPerSecond", samplesPerSecond);
        
        return samplesPerSecond;
    }
    
    /// <summary>
    /// Runs inference with the model on a single input.
    /// </summary>
    /// <param name="model">The model to use.</param>
    /// <param name="input">The input to process.</param>
    /// <returns>The model's output for the input.</returns>
    protected abstract TOutput RunInference(TModel model, TInput input);
    
    /// <summary>
    /// Measures the approximate memory usage of a model during inference.
    /// </summary>
    /// <param name="model">The model to measure.</param>
    /// <returns>The memory usage in bytes.</returns>
    protected virtual double MeasureMemoryUsage(TModel model)
    {
        // Default implementation just returns the model size
        // (real implementations would measure actual runtime memory usage)
        return MeasureModelSize(model);
    }
    
    /// <summary>
    /// Serializes a model to a stream.
    /// </summary>
    /// <param name="model">The model to serialize.</param>
    /// <param name="stream">The stream to which the model should be serialized.</param>
    protected virtual void SerializeModelToStream(TModel model, Stream stream)
    {
        // First, try IModelSerializer interface
        if (model is IModelSerializer serializer)
        {
            var data = serializer.Serialize();
            stream.Write(data, 0, data.Length);
            return;
        }
        
        // Try to use reflection to find a Serialize method
        var serializeMethod = model.GetType().GetMethod("Serialize", Type.EmptyTypes);
        if (serializeMethod != null && serializeMethod.ReturnType == typeof(byte[]))
        {
            var data = (byte[]?)serializeMethod.Invoke(model, null);
            if (data != null)
            {
                stream.Write(data, 0, data.Length);
            }
            return;
        }
        
        // For models with parameters, try to serialize those
        using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, true))
        {
            // Write model type name for deserialization
            writer.Write(model.GetType().AssemblyQualifiedName ?? model.GetType().FullName ?? "Unknown");
            
            var getParamsMethod = model.GetType().GetMethod("GetParameters", Type.EmptyTypes);
            if (getParamsMethod != null)
            {
                var parameters = getParamsMethod.Invoke(model, null);
                if (parameters != null)
                {
                    var paramSerializeMethod = parameters.GetType().GetMethod("Serialize", Type.EmptyTypes);
                    if (paramSerializeMethod != null && paramSerializeMethod.ReturnType == typeof(byte[]))
                    {
                        var paramData = (byte[]?)paramSerializeMethod.Invoke(parameters, null);
                        if (paramData != null)
                        {
                            writer.Write(true); // Has parameters
                            writer.Write(paramData.Length);
                            writer.Write(paramData);
                        }
                        return;
                    }
                }
            }
            
            // If we can't serialize, throw exception
            writer.Write(false); // No parameters
            throw new NotSupportedException($"Model type {model.GetType().Name} does not support serialization. " +
                "The model must either implement IModelSerializer interface or have a Serialize() method.");
        }
    }
    
    /// <summary>
    /// Serializes a model to a file.
    /// </summary>
    /// <param name="model">The model to serialize.</param>
    /// <param name="filePath">The file path where the model should be saved.</param>
    protected virtual void SerializeModelToFile(TModel model, string filePath)
    {
        using var fileStream = new FileStream(filePath, FileMode.Create);
        SerializeModelToStream(model, fileStream);
    }
    
    /// <summary>
    /// Deserializes a model from a file.
    /// </summary>
    /// <param name="filePath">The file path from which to load the model.</param>
    /// <returns>The deserialized model.</returns>
    protected abstract TModel DeserializeModelFromFile(string filePath);
    
    /// <summary>
    /// Gets the name of the model type.
    /// </summary>
    /// <param name="modelType">The model type.</param>
    /// <returns>A string representing the model name.</returns>
    protected virtual string GetModelName(Type modelType)
    {
        return modelType.Name;
    }
    
    /// <summary>
    /// Gets the compression technique being used.
    /// </summary>
    /// <returns>An enum value representing the compression technique.</returns>
    protected abstract Enums.CompressionTechnique GetCompressionTechnique();
    
    /// <summary>
    /// Gets the name of the compression technique being used.
    /// </summary>
    /// <returns>A string representing the compression technique.</returns>
    protected virtual string GetCompressionTechniqueName()
    {
        return GetCompressionTechnique().ToString();
    }
    
    /// <summary>
    /// Populates additional metrics specific to the compression technique.
    /// </summary>
    /// <param name="metrics">The dictionary to populate with additional metrics.</param>
    /// <param name="originalModel">The original uncompressed model.</param>
    /// <param name="compressedModel">The compressed model.</param>
    /// <remarks>
    /// <para>
    /// This method allows derived classes to add compression-specific metrics to the results.
    /// </para>
    /// <para><b>For Beginners:</b> This adds technique-specific details to the results.
    /// 
    /// Different compression techniques (pruning, quantization, etc.) have different
    /// metrics that are important. This method lets each technique add its specific metrics.
    /// </para>
    /// </remarks>
    protected virtual void PopulateAdditionalMetrics(
        System.Collections.Generic.Dictionary<string, object> metrics,
        TModel originalModel,
        TModel compressedModel)
    {
        // Base implementation adds common metrics
        metrics["CompressionTechnique"] = GetCompressionTechnique().ToString();
    }
    
    /// <summary>
    /// Reports progress of the compression operation.
    /// </summary>
    /// <param name="stage">The current compression stage.</param>
    /// <param name="progressPercentage">The percentage of completion for the current stage.</param>
    /// <param name="message">An optional message describing the current progress.</param>
    protected void ReportProgress(string stage, double progressPercentage, string? message = null)
    {
        Logger.LogCompressionProgress(stage, progressPercentage, message ?? string.Empty);
    }
}