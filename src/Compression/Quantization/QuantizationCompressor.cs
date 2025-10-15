namespace AiDotNet.Compression.Quantization;

using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

/// <summary>
/// Implements model compression using weight quantization techniques.
/// </summary>
/// <remarks>
/// <para>
/// Quantization reduces the precision of model parameters (e.g., from 32-bit floating-point
/// to 8-bit integers) to decrease model size and potentially improve inference speed.
/// </para>
/// <para><b>For Beginners:</b> This compressor makes models smaller by using fewer bits per number.
/// 
/// Neural networks store millions of numbers (weights) that are typically 32-bit floating-point values.
/// Quantization converts these to lower precision (like 8-bit integers):
/// - This makes the model up to 4x smaller
/// - It can make inference faster, especially on hardware with integer acceleration
/// - Usually maintains good accuracy if done carefully
/// 
/// This is one of the most widely used compression techniques due to its simplicity and effectiveness.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <typeparam name="TModel">The type of model to compress.</typeparam>
/// <typeparam name="TInput">The input type for the model.</typeparam>
/// <typeparam name="TOutput">The output type for the model.</typeparam>
public class QuantizationCompressor<T, TModel, TInput, TOutput> :
    ModelCompressorBase<T, TModel, TInput, TOutput>
    where T : unmanaged
    where TModel : class, IFullModel<T, TInput, TOutput>
{
    private readonly QuantizationMethod _method = default!;
    private readonly bool _useCalibration;
    private readonly double _calibrationPercentile;

    /// <summary>
    /// Initializes a new instance of the <see cref="QuantizationCompressor{TModel, TInput, TOutput}"/> class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Creates a new quantization compressor with default settings:
    /// - Using symmetric quantization
    /// - No calibration
    /// </para>
    /// <para><b>For Beginners:</b> This creates a quantization compressor with standard settings.
    /// 
    /// The default configuration works well for most models, but you can use other constructors
    /// if you need more control over the quantization process.
    /// </para>
    /// </remarks>
    public QuantizationCompressor()
        : this(QuantizationMethod.Symmetric, false, 99.99)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="QuantizationCompressor{TModel, TInput, TOutput}"/> class.
    /// </summary>
    /// <param name="method">The quantization method to use.</param>
    /// <param name="useCalibration">Whether to use calibration data to optimize quantization.</param>
    /// <param name="calibrationPercentile">The percentile to use for outlier clipping in calibration.</param>
    /// <remarks>
    /// <para>
    /// Creates a new quantization compressor with the specified settings.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a quantization compressor with custom settings.
    /// 
    /// You can specify:
    /// - Which quantization method to use (symmetric, asymmetric, etc.)
    /// - Whether to use calibration to handle outlier values
    /// - What percentile to use for clipping outliers
    /// 
    /// These settings let you fine-tune the quantization process for your specific model.
    /// </para>
    /// </remarks>
    public QuantizationCompressor(
        QuantizationMethod method = QuantizationMethod.Symmetric,
        bool useCalibration = false,
        double calibrationPercentile = 99.99)
        : base(new ModelCompressionOptions 
        { 
            Technique = CompressionTechnique.Quantization 
        })
    {
        _method = method;
        _useCalibration = useCalibration;
        _calibrationPercentile = calibrationPercentile;
    }

    /// <summary>
    /// Compresses a model using quantization.
    /// </summary>
    /// <param name="model">The model to compress.</param>
    /// <param name="options">Options for the compression process.</param>
    /// <returns>The quantized model.</returns>
    /// <remarks>
    /// <para>
    /// This method applies quantization to the model's parameters according to the
    /// specified options and quantization method.
    /// </para>
    /// <para><b>For Beginners:</b> This converts the model's weights to lower precision.
    /// 
    /// The process involves:
    /// 1. Extracting the model's weights
    /// 2. Analyzing their distribution
    /// 3. Converting them to the target precision (e.g., 8-bit)
    /// 4. Creating a new model with these quantized weights
    /// 
    /// The result is a compressed model that requires less storage and potentially runs faster.
    /// </para>
    /// </remarks>
    public override TModel Compress(TModel model, ModelCompressionOptions options)
    {
        // Validate inputs
        if (model == null) throw new ArgumentNullException(nameof(model));
        if (options == null) throw new ArgumentNullException(nameof(options));
        if (options.Technique != CompressionTechnique.Quantization)
        {
            throw new ArgumentException(
                $"Expected CompressionTechnique.Quantization but got {options.Technique}.",
                nameof(options));
        }

        // Extract model parameters
        var parameters = ExtractParameters(model);
        if (parameters.Count == 0)
        {
            throw new InvalidOperationException(
                $"No parameters could be extracted from the model of type {model.GetType().Name}.");
        }

        // Apply quantization
        var quantizedParams = QuantizeParameters(parameters, options.QuantizationPrecision, options.UseMixedPrecision);

        // Create and return a new model with quantized parameters
        return CreateQuantizedModel(model, quantizedParams);
    }

    /// <summary>
    /// Serializes a quantized model to a file.
    /// </summary>
    /// <param name="model">The quantized model to serialize.</param>
    /// <param name="filePath">The path where the serialized model should be saved.</param>
    /// <remarks>
    /// <para>
    /// This method serializes the quantized model to a file format that preserves the
    /// quantization information.
    /// </para>
    /// <para><b>For Beginners:</b> This saves the quantized model to a file.
    /// 
    /// The serialization process:
    /// 1. Stores the quantized values efficiently
    /// 2. Preserves quantization parameters (scales, zero points)
    /// 3. Uses a format optimized for quantized models
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

        // Check if model is quantized
        if (!(model is IQuantizedModel<T, TInput, TOutput> quantizedModel))
        {
            throw new ArgumentException(
                $"Model of type {model.GetType().Name} is not a quantized model. " +
                "Use the Compress method before serializing.", nameof(model));
        }

        using (var fileStream = new FileStream(filePath, FileMode.Create))
        using (var writer = new BinaryWriter(fileStream))
        {
            // Write quantization metadata
            writer.Write((int)_method);
            writer.Write(quantizedModel.QuantizationBitWidth);
            writer.Write(quantizedModel.UseMixedPrecision);

            // Write model-specific data
            quantizedModel.SerializeQuantized(writer);
        }
    }

    /// <summary>
    /// Deserializes a quantized model from a file.
    /// </summary>
    /// <param name="filePath">The path where the serialized model is stored.</param>
    /// <returns>The deserialized quantized model.</returns>
    /// <remarks>
    /// <para>
    /// This method deserializes a quantized model from a file that was previously saved
    /// using the SerializeCompressedModel method.
    /// </para>
    /// <para><b>For Beginners:</b> This loads a quantized model from a file.
    /// 
    /// The deserialization process:
    /// 1. Reads the quantization metadata
    /// 2. Loads the quantized values
    /// 3. Reconstructs the quantized model
    /// 
    /// This ensures the loaded model maintains the same compression benefits as when it was saved.
    /// </para>
    /// </remarks>
    public override TModel DeserializeCompressedModel(string filePath)
    {
        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"Quantized model file not found: {filePath}");
        }

        using (var fileStream = new FileStream(filePath, FileMode.Open))
        using (var reader = new BinaryReader(fileStream))
        {
            // Read quantization metadata
            var method = (QuantizationMethod)reader.ReadInt32();
            int bitWidth = reader.ReadInt32();
            bool useMixedPrecision = reader.ReadBoolean();

            // Create and return the deserialized model
            return DeserializeQuantizedModelInternal(reader, method, bitWidth, useMixedPrecision);
        }
    }

    /// <summary>
    /// Gets the compression technique used by this compressor.
    /// </summary>
    /// <returns>CompressionTechnique.Quantization</returns>
    /// <remarks>
    /// <para>
    /// This method returns CompressionTechnique.Quantization to indicate that this
    /// compressor implements quantization-based compression.
    /// </para>
    /// <para><b>For Beginners:</b> This identifies the compressor as using quantization.
    /// 
    /// This information is used in compression results to indicate which technique was applied.
    /// </para>
    /// </remarks>
    protected override CompressionTechnique GetCompressionTechnique()
    {
        return CompressionTechnique.Quantization;
    }
    
    /// <summary>
    /// Compresses a model using the internal settings of this compressor.
    /// </summary>
    /// <param name="model">The model to compress.</param>
    /// <returns>The quantized model.</returns>
    /// <remarks>
    /// <para>
    /// This method applies quantization to the model's parameters based on this compressor's settings.
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
            Technique = CompressionTechnique.Quantization,
            QuantizationPrecision = 8, // Default
            UseMixedPrecision = false  // Default
        };
        
        // Delegate to the public Compress method
        return Compress(model, options);
    }
    
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
        // Create a new quantization compressor with the same configuration but new options
        return new QuantizationCompressor<T, TModel, TInput, TOutput>(
            _method,
            _useCalibration,
            _calibrationPercentile);
    }

    /// <summary>
    /// Populates additional metrics specific to quantization.
    /// </summary>
    /// <param name="metrics">The dictionary to populate with additional metrics.</param>
    /// <param name="originalModel">The original uncompressed model.</param>
    /// <param name="compressedModel">The quantized model.</param>
    /// <remarks>
    /// <para>
    /// This method adds quantization-specific metrics to the compression results.
    /// </para>
    /// <para><b>For Beginners:</b> This adds quantization-specific details to the results.
    /// 
    /// Specifically, it adds:
    /// - The actual bit width used for quantization
    /// - The quantization method used (symmetric, asymmetric)
    /// - Whether mixed precision was used
    /// - Statistics about weight distribution before and after quantization
    /// </para>
    /// </remarks>
    protected override void PopulateAdditionalMetrics(
        Dictionary<string, object> metrics,
        TModel originalModel,
        TModel compressedModel)
    {
        base.PopulateAdditionalMetrics(metrics, originalModel, compressedModel);

        // Add quantization-specific metrics
        if (compressedModel is IQuantizedModel<T, TInput, TOutput> quantizedModel)
        {
            metrics["QuantizationBitWidth"] = quantizedModel.QuantizationBitWidth;
            metrics["QuantizationMethod"] = _method.ToString();
            metrics["UseMixedPrecision"] = quantizedModel.UseMixedPrecision;

            // Add weight distribution statistics if available
            var weightStats = quantizedModel.WeightDistributionStatistics;
            if (weightStats != null)
            {
                metrics["OriginalWeightRange"] = weightStats.OriginalRange;
                metrics["QuantizedWeightRange"] = weightStats.QuantizedRange;
                metrics["ClippedOutliersPercentage"] = weightStats.ClippedOutliersPercentage;
            }
        }
    }

    /// <summary>
    /// Extracts parameters from a model for quantization.
    /// </summary>
    /// <param name="model">The model from which to extract parameters.</param>
    /// <returns>A dictionary mapping parameter names to their values.</returns>
    /// <remarks>
    /// <para>
    /// This method extracts the parameters (weights, biases) from the model that
    /// will be targeted for quantization.
    /// </para>
    /// <para><b>For Beginners:</b> This pulls out all the model's weights for processing.
    /// 
    /// Before we can quantize a model, we need to extract all its parameters:
    /// - Weight matrices
    /// - Bias vectors
    /// - Other learned parameters
    /// 
    /// This method extracts these values so they can be quantized.
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
            "Implement a custom QuantizationCompressor for this model type.");
    }

    /// <summary>
    /// Quantizes model parameters to the specified bit width.
    /// </summary>
    /// <param name="parameters">The parameters to quantize.</param>
    /// <param name="bitWidth">The target bit width for quantization.</param>
    /// <param name="useMixedPrecision">Whether to use mixed precision quantization.</param>
    /// <returns>A dictionary mapping parameter names to their quantized representations.</returns>
    /// <remarks>
    /// <para>
    /// This method converts the model's parameters from their original precision (typically
    /// 32-bit floating-point) to the target bit width using the specified quantization method.
    /// </para>
    /// <para><b>For Beginners:</b> This converts the weights to lower precision.
    /// 
    /// The quantization process:
    /// 1. Determines the range of values for each parameter
    /// 2. Calculates scaling factors to map to the lower precision
    /// 3. Converts each value to the quantized representation
    /// 4. Returns these quantized values along with the quantization parameters
    /// 
    /// This is the core of the quantization process.
    /// </para>
    /// </remarks>
    protected virtual Dictionary<string, QuantizedParameter> QuantizeParameters(
        Dictionary<string, Array> parameters,
        int bitWidth,
        bool useMixedPrecision)
    {
        var quantizedParams = new Dictionary<string, QuantizedParameter>();
        
        foreach (var param in parameters)
        {
            // Determine actual bit width for this parameter if using mixed precision
            int actualBitWidth = bitWidth;
            if (useMixedPrecision)
            {
                // Example logic for mixed precision:
                // - Use higher precision for small, sensitive layers
                // - Use lower precision for large, redundant layers
                // This would be customized based on model architecture
                actualBitWidth = DetermineMixedPrecisionBitWidth(param.Key, param.Value, bitWidth);
            }
            
            // Apply the appropriate quantization based on method
            QuantizedParameter quantizedParam;
            switch (_method)
            {
                case QuantizationMethod.Symmetric:
                    quantizedParam = QuantizeSymmetric(param.Value, actualBitWidth);
                    break;
                case QuantizationMethod.Asymmetric:
                    quantizedParam = QuantizeAsymmetric(param.Value, actualBitWidth);
                    break;
                case QuantizationMethod.PerChannel:
                    quantizedParam = QuantizePerChannel(param.Value, actualBitWidth);
                    break;
                default:
                    throw new ArgumentException($"Unsupported quantization method: {_method}");
            }
            
            quantizedParams[param.Key] = quantizedParam;
        }
        
        return quantizedParams;
    }

    /// <summary>
    /// Determines the appropriate bit width for a parameter when using mixed precision.
    /// </summary>
    /// <param name="paramName">The name of the parameter.</param>
    /// <param name="paramValue">The parameter value array.</param>
    /// <param name="defaultBitWidth">The default bit width specified in options.</param>
    /// <returns>The bit width to use for this specific parameter.</returns>
    /// <remarks>
    /// <para>
    /// This method implements heuristics for determining the appropriate bit width
    /// for each parameter when using mixed precision quantization.
    /// </para>
    /// <para><b>For Beginners:</b> This decides how many bits each parameter should use.
    /// 
    /// When using mixed precision:
    /// - Not all parameters need the same precision
    /// - Critical parameters get higher precision (more bits)
    /// - Less important parameters get lower precision (fewer bits)
    /// 
    /// This can give better accuracy than using the same precision for everything.
    /// </para>
    /// </remarks>
    protected virtual int DetermineMixedPrecisionBitWidth(
        string paramName,
        Array paramValue,
        int defaultBitWidth)
    {
        // Example heuristic for mixed precision:
        // - First layer and output layer use higher precision
        // - Large matrices use lower precision
        // - Attention-related parameters use higher precision
        
        // For first layer or output layer, use higher precision
        if (paramName.Contains("input_layer") || paramName.Contains("output_layer"))
        {
            return Math.Min(defaultBitWidth + 4, 16); // Higher precision, up to 16 bits
        }
        
        // For attention mechanisms, use higher precision
        if (paramName.Contains("attention") || paramName.Contains("query") || 
            paramName.Contains("key") || paramName.Contains("value"))
        {
            return Math.Min(defaultBitWidth + 2, 12); // Higher precision
        }
        
        // For very large matrices, use lower precision
        if (paramValue.Length > 1_000_000)
        {
            return Math.Max(defaultBitWidth - 2, 4); // Lower precision, minimum 4 bits
        }
        
        // Default to the specified bit width
        return defaultBitWidth;
    }

    /// <summary>
    /// Applies symmetric quantization to a parameter.
    /// </summary>
    /// <param name="parameter">The parameter to quantize.</param>
    /// <param name="bitWidth">The target bit width.</param>
    /// <returns>The quantized parameter.</returns>
    /// <remarks>
    /// <para>
    /// Symmetric quantization uses a symmetric range around zero, with a single scale factor.
    /// </para>
    /// <para><b>For Beginners:</b> This converts values using a symmetric range.
    /// 
    /// Symmetric quantization:
    /// - Uses the same scale for positive and negative values
    /// - Is centered around zero
    /// - Works well for weights, which are often distributed around zero
    /// - Is simpler and more hardware-friendly than asymmetric quantization
    /// </para>
    /// </remarks>
    protected virtual QuantizedParameter QuantizeSymmetric(Array parameter, int bitWidth)
    {
        // Implementation would convert floating-point values to quantized values
        // using symmetric quantization. This is a simplified version.
        
        // Find the maximum absolute value (for symmetric range)
        float maxAbs = 0;
        foreach (float val in parameter)
        {
            maxAbs = Math.Max(maxAbs, Math.Abs(val));
        }
        
        // Calculate scale factor
        int maxQuantized = (1 << (bitWidth - 1)) - 1; // 2^(bits-1) - 1 for signed values
        float scale = maxAbs / maxQuantized;
        
        // Allocate array for quantized values (use sbyte for 8-bit, short for 16-bit, etc.)
        Array quantizedValues;
        if (bitWidth <= 8)
        {
            quantizedValues = new sbyte[parameter.Length];
        }
        else if (bitWidth <= 16)
        {
            quantizedValues = new short[parameter.Length];
        }
        else
        {
            quantizedValues = new int[parameter.Length];
        }
        
        // Quantize each value
        for (int i = 0; i < parameter.Length; i++)
        {
            float originalValue = (float)(parameter.GetValue(i) ?? 0f);
            int quantizedValue = (int)Math.Round(originalValue / scale);
            
            // Clamp to representable range
            quantizedValue = Math.Max(-maxQuantized - 1, Math.Min(maxQuantized, quantizedValue));
            
            // Store the quantized value
            if (bitWidth <= 8)
            {
                ((sbyte[])quantizedValues)[i] = (sbyte)quantizedValue;
            }
            else if (bitWidth <= 16)
            {
                ((short[])quantizedValues)[i] = (short)quantizedValue;
            }
            else
            {
                ((int[])quantizedValues)[i] = quantizedValue;
            }
        }
        
        return new QuantizedParameter
        {
            OriginalShape = GetShape(parameter),
            QuantizedValues = quantizedValues,
            Scale = scale,
            ZeroPoint = 0, // Symmetric quantization has zero point at 0
            BitWidth = bitWidth,
            Method = QuantizationMethod.Symmetric
        };
    }

    /// <summary>
    /// Applies asymmetric quantization to a parameter.
    /// </summary>
    /// <param name="parameter">The parameter to quantize.</param>
    /// <param name="bitWidth">The target bit width.</param>
    /// <returns>The quantized parameter.</returns>
    /// <remarks>
    /// <para>
    /// Asymmetric quantization uses a range that can be offset from zero, with a scale
    /// factor and a zero point.
    /// </para>
    /// <para><b>For Beginners:</b> This converts values using an asymmetric range.
    /// 
    /// Asymmetric quantization:
    /// - Uses a range that doesn't have to be centered at zero
    /// - Works better for activations or biases that might be mostly positive or negative
    /// - Can represent more values in the actual range of the data
    /// - Requires both a scale and a zero point for dequantization
    /// </para>
    /// </remarks>
    protected virtual QuantizedParameter QuantizeAsymmetric(Array parameter, int bitWidth)
    {
        // Implementation would convert floating-point values to quantized values
        // using asymmetric quantization. This is a simplified version.
        
        // Find the minimum and maximum values
        float minVal = float.MaxValue;
        float maxVal = float.MinValue;
        foreach (float val in parameter)
        {
            minVal = Math.Min(minVal, val);
            maxVal = Math.Max(maxVal, val);
        }
        
        // Calculate quantization parameters
        int qmin = 0;
        int qmax = (1 << bitWidth) - 1; // 2^bits - 1
        float scale = (maxVal - minVal) / (qmax - qmin);
        int zeroPoint = (int)Math.Round(qmin - minVal / scale);
        
        // Allocate array for quantized values
        Array quantizedValues;
        if (bitWidth <= 8)
        {
            quantizedValues = new byte[parameter.Length];
        }
        else if (bitWidth <= 16)
        {
            quantizedValues = new ushort[parameter.Length];
        }
        else
        {
            quantizedValues = new uint[parameter.Length];
        }
        
        // Quantize each value
        for (int i = 0; i < parameter.Length; i++)
        {
            float originalValue = (float)(parameter.GetValue(i) ?? 0f);
            int quantizedValue = (int)Math.Round(originalValue / scale) + zeroPoint;
            
            // Clamp to representable range
            quantizedValue = Math.Max(qmin, Math.Min(qmax, quantizedValue));
            
            // Store the quantized value
            if (bitWidth <= 8)
            {
                ((byte[])quantizedValues)[i] = (byte)quantizedValue;
            }
            else if (bitWidth <= 16)
            {
                ((ushort[])quantizedValues)[i] = (ushort)quantizedValue;
            }
            else
            {
                ((uint[])quantizedValues)[i] = (uint)quantizedValue;
            }
        }
        
        return new QuantizedParameter
        {
            OriginalShape = GetShape(parameter),
            QuantizedValues = quantizedValues,
            Scale = scale,
            ZeroPoint = zeroPoint,
            BitWidth = bitWidth,
            Method = QuantizationMethod.Asymmetric
        };
    }

    /// <summary>
    /// Applies per-channel quantization to a parameter.
    /// </summary>
    /// <param name="parameter">The parameter to quantize.</param>
    /// <param name="bitWidth">The target bit width.</param>
    /// <returns>The quantized parameter.</returns>
    /// <remarks>
    /// <para>
    /// Per-channel quantization applies separate quantization parameters to each channel
    /// (e.g., each output feature in a convolutional layer).
    /// </para>
    /// <para><b>For Beginners:</b> This quantizes each channel separately.
    /// 
    /// Per-channel quantization:
    /// - Uses different scales for different channels
    /// - Provides better accuracy than per-tensor quantization
    /// - Works especially well for convolutional layers
    /// - Is more complex but often worth it for accuracy-critical applications
    /// </para>
    /// </remarks>
    protected virtual QuantizedParameter QuantizePerChannel(Array parameter, int bitWidth)
    {
        // This would implement per-channel quantization, which is more complex.
        // In a real implementation, we would:
        // 1. Determine the channel dimension based on parameter shape and layer type
        // 2. Calculate separate scale/zeroPoint for each channel
        // 3. Apply quantization separately to each channel's weights
        
        // This is a complex implementation that depends on the specific model architecture.
        // For this example, we'll return a placeholder that indicates the method but 
        // defers to asymmetric quantization for the actual implementation.
        var quantizedParam = QuantizeAsymmetric(parameter, bitWidth);
        quantizedParam.Method = QuantizationMethod.PerChannel;
        quantizedParam.ChannelScales = new[] { quantizedParam.Scale }; // Placeholder
        
        return quantizedParam;
    }

    /// <summary>
    /// Creates a new model with the quantized parameters.
    /// </summary>
    /// <param name="originalModel">The original model.</param>
    /// <param name="quantizedParameters">The quantized parameters.</param>
    /// <returns>A new model with quantized parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new model instance that uses the quantized parameters
    /// for inference.
    /// </para>
    /// <para><b>For Beginners:</b> This builds a new model using the quantized weights.
    /// 
    /// After quantizing the parameters, we need to:
    /// 1. Create a new model that uses these quantized parameters
    /// 2. Ensure it knows how to dequantize during inference
    /// 3. Set up any additional metadata or structures needed
    /// 
    /// The result is a fully functional model that uses quantized weights internally.
    /// </para>
    /// </remarks>
    protected virtual TModel CreateQuantizedModel(
        TModel originalModel,
        Dictionary<string, QuantizedParameter> quantizedParameters)
    {
        // This implementation depends on the specific model type.
        // Typically, you would:
        // 1. Create a new model of a type that supports quantization
        // 2. Initialize it with the quantized parameters
        // 3. Copy any non-parameter state from the original model
        
        // For demonstration, we'll assume there's a way to create a quantized version
        // of the model. In reality, this would require specific implementations for each
        // supported model type.
        
        if (originalModel is IQuantizableModel<T, TModel, TInput, TOutput> quantizableModel)
        {
            return quantizableModel.CreateQuantizedVersion(quantizedParameters);
        }

        throw new NotSupportedException(
            $"Quantized model creation not supported for model type {originalModel.GetType().Name}. " +
            "Implement IQuantizableModel<T, TModel, TInput, TOutput> for this model type.");
    }

    /// <summary>
    /// Deserializes a quantized model from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader containing the serialized model.</param>
    /// <param name="method">The quantization method used.</param>
    /// <param name="bitWidth">The bit width used for quantization.</param>
    /// <param name="useMixedPrecision">Whether mixed precision was used.</param>
    /// <returns>The deserialized quantized model.</returns>
    /// <remarks>
    /// <para>
    /// This method reconstructs a quantized model from its serialized form.
    /// </para>
    /// <para><b>For Beginners:</b> This rebuilds a quantized model from saved data.
    /// 
    /// The deserialization process:
    /// 1. Reads the serialized quantized parameters
    /// 2. Reconstructs the quantization metadata
    /// 3. Creates a new model using this information
    /// 
    /// This is a model-specific process that depends on the type of model being used.
    /// </para>
    /// </remarks>
    protected virtual TModel DeserializeQuantizedModelInternal(
        BinaryReader reader,
        QuantizationMethod method,
        int bitWidth,
        bool useMixedPrecision)
    {
        // This implementation depends on the specific model type.
        // For demonstration, we'll assume there's a factory method or constructor
        // that can create a model from the serialized data.

        // In a real implementation, this would use model-specific deserialization.
        var factory = QuantizedModelFactoryRegistry.GetFactory<T, TModel, TInput, TOutput>();
        if (factory != null)
        {
            return factory.DeserializeQuantizedModel(reader, method, bitWidth, useMixedPrecision);
        }

        throw new NotSupportedException(
            $"Quantized model deserialization not supported for model type TModel. " +
            "Register a factory for this model type with QuantizedModelFactoryRegistry.");
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
/// Interface for models that support quantization.
/// </summary>
/// <remarks>
/// <para>
/// This interface should be implemented by models that support being quantized.
/// </para>
/// <para><b>For Beginners:</b> This marks a model as able to be quantized.
/// 
/// Models that implement this interface know how to:
/// - Convert their parameters to a quantized form
/// - Create a new version of themselves that uses quantized parameters
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TModel">The type of the model.</typeparam>
/// <typeparam name="TInput">The input type for the model.</typeparam>
/// <typeparam name="TOutput">The output type for the model.</typeparam>
public interface IQuantizableModel<T, TModel, TInput, TOutput>
    where T : unmanaged
    where TModel : class, IFullModel<T, TInput, TOutput>
{
    /// <summary>
    /// Creates a quantized version of this model.
    /// </summary>
    /// <param name="quantizedParameters">The quantized parameters.</param>
    /// <returns>A new model instance that uses the quantized parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new model instance that uses the provided quantized parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This builds a new model with quantized weights.
    /// 
    /// Given a set of quantized parameters, this method:
    /// - Creates a new model that uses these quantized parameters
    /// - Sets up the dequantization process for inference
    /// - Returns this new, compressed model
    /// </para>
    /// </remarks>
    TModel CreateQuantizedVersion(Dictionary<string, QuantizedParameter> quantizedParameters);
}

/// <summary>
/// Contains statistics about weight distribution before and after quantization.
/// </summary>
/// <remarks>
/// <para>
/// This class provides metrics about how quantization affected the distribution of
/// weight values in the model.
/// </para>
/// <para><b>For Beginners:</b> This shows how quantization changed the model's weights.
/// 
/// It helps you understand:
/// - How much the value range was compressed
/// - What percentage of outliers had to be clipped
/// - How well the quantization captured the original distribution
/// </para>
/// </remarks>
public class WeightDistributionStatistics
{
    /// <summary>
    /// Gets or sets the range of weight values in the original model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is the difference between the maximum and minimum weight values before quantization.
    /// </para>
    /// <para><b>For Beginners:</b> This is the range of values before quantization.
    /// 
    /// For example, if weights ranged from -2.5 to 1.5, the range would be 4.0.
    /// </para>
    /// </remarks>
    public double OriginalRange { get; set; }

    /// <summary>
    /// Gets or sets the range of weight values in the quantized model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is the effective range of values that can be represented after quantization.
    /// </para>
    /// <para><b>For Beginners:</b> This is the range of values after quantization.
    /// 
    /// For example, if 8-bit quantization maps to a range of -2.5 to 1.5, this would be 4.0.
    /// </para>
    /// </remarks>
    public double QuantizedRange { get; set; }

    /// <summary>
    /// Gets or sets the percentage of weight values that were clipped as outliers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When using calibration, some outlier values may be clipped to improve the
    /// representation of the most common values.
    /// </para>
    /// <para><b>For Beginners:</b> This is the percentage of extreme values that got clipped.
    /// 
    /// For example, a value of 0.01 means that 1% of the weights were outside the
    /// quantization range and had to be clipped to fit.
    /// </para>
    /// </remarks>
    public double ClippedOutliersPercentage { get; set; }
}

/// <summary>
/// Represents a single quantized parameter from a model.
/// </summary>
/// <remarks>
/// <para>
/// This class encapsulates a quantized parameter along with the metadata needed
/// to dequantize it.
/// </para>
/// <para><b>For Beginners:</b> This represents one quantized weight matrix or vector.
/// 
/// It contains:
/// - The quantized values themselves (in low precision)
/// - The information needed to convert back to original values
/// - Metadata about the quantization process used
/// </para>
/// </remarks>
public class QuantizedParameter
{
    /// <summary>
    /// Gets or sets the original shape of the parameter.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is the shape (dimensions) of the parameter before quantization.
    /// </para>
    /// <para><b>For Beginners:</b> This records the original dimensions of the parameter.
    /// 
    /// For example, a weight matrix might have shape [1000, 500], meaning 1000 rows and 500 columns.
    /// </para>
    /// </remarks>
    public int[] OriginalShape { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Gets or sets the quantized values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These are the low-precision values that represent the original parameter.
    /// </para>
    /// <para><b>For Beginners:</b> These are the actual compressed values.
    /// 
    /// For example, if using 8-bit quantization, this would be an array of bytes
    /// instead of an array of 32-bit floats.
    /// </para>
    /// </remarks>
    public Array QuantizedValues { get; set; } = Array.Empty<byte>();

    /// <summary>
    /// Gets or sets the scale factor for dequantization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This scale factor is used to convert quantized values back to their original range.
    /// </para>
    /// <para><b>For Beginners:</b> This is the multiplier to convert back to original values.
    /// 
    /// During dequantization:
    /// original_value ≈ (quantized_value - zero_point) * scale
    /// </para>
    /// </remarks>
    public float Scale { get; set; }

    /// <summary>
    /// Gets or sets the zero point for asymmetric quantization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For asymmetric quantization, this is the quantized value that corresponds to 0
    /// in the original parameter.
    /// </para>
    /// <para><b>For Beginners:</b> This is the value that represents zero in the quantized space.
    /// 
    /// For asymmetric quantization, we need to know which quantized value maps to zero
    /// in the original space. This zero point is subtracted before scaling during dequantization.
    /// </para>
    /// </remarks>
    public int ZeroPoint { get; set; }

    /// <summary>
    /// Gets or sets the bit width used for quantization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is the number of bits used to represent each quantized value.
    /// </para>
    /// <para><b>For Beginners:</b> This is how many bits each quantized value uses.
    /// 
    /// Common values are 8 bits or 4 bits, compared to the original 32 bits used by floating-point values.
    /// </para>
    /// </remarks>
    public int BitWidth { get; set; }

    /// <summary>
    /// Gets or sets the quantization method used.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This specifies whether symmetric, asymmetric, or per-channel quantization was used.
    /// </para>
    /// <para><b>For Beginners:</b> This tells which quantization approach was used.
    /// 
    /// Different methods have different tradeoffs and require different dequantization approaches.
    /// </para>
    /// </remarks>
    public QuantizationMethod Method { get; set; }

    /// <summary>
    /// Gets or sets the per-channel scale factors for per-channel quantization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For per-channel quantization, each channel has its own scale factor.
    /// </para>
    /// <para><b>For Beginners:</b> These are the multipliers for each channel.
    /// 
    /// When using per-channel quantization:
    /// - Each output channel has its own scaling factor
    /// - This array contains one scale value per channel
    /// - Dequantization uses the appropriate scale for each channel
    /// </para>
    /// </remarks>
    public float[] ChannelScales { get; set; } = Array.Empty<float>();
}

/// <summary>
/// Defines the available quantization methods.
/// </summary>
/// <remarks>
/// <para>
/// Different quantization methods offer different trade-offs between accuracy, complexity,
/// and hardware compatibility.
/// </para>
/// <para><b>For Beginners:</b> This defines different ways to convert to lower precision.
/// 
/// The choice of method affects:
/// - How accurately the original values can be represented
/// - How complex the dequantization process is
/// - How well it works on different types of hardware
/// </para>
/// </remarks>
public enum QuantizationMethod
{
    /// <summary>
    /// Uses a symmetric range around zero for quantization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Symmetric quantization uses the same scale for positive and negative values,
    /// and represents zero exactly.
    /// </para>
    /// <para><b>For Beginners:</b> This uses a range that's balanced around zero.
    /// 
    /// For example, with 8-bit signed integers:
    /// - Values range from -127 to 127
    /// - Zero is represented exactly
    /// - Only a scale factor is needed for dequantization
    /// 
    /// This is simpler and often works well for weights.
    /// </para>
    /// </remarks>
    Symmetric = 0,
    
    /// <summary>
    /// Uses a potentially asymmetric range for quantization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Asymmetric quantization can handle ranges that aren't centered around zero
    /// by using both a scale and a zero point.
    /// </para>
    /// <para><b>For Beginners:</b> This uses a range that can be shifted from zero.
    /// 
    /// For example, with 8-bit unsigned integers:
    /// - Values range from 0 to 255
    /// - A "zero point" specifies which of these values represents zero
    /// - Both a scale and zero point are needed for dequantization
    /// 
    /// This works better for activations that might be mostly positive or negative.
    /// </para>
    /// </remarks>
    Asymmetric = 1,
    
    /// <summary>
    /// Applies separate quantization parameters to each channel.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Per-channel quantization quantizes each output channel separately,
    /// allowing for better accuracy when different channels have different ranges.
    /// </para>
    /// <para><b>For Beginners:</b> This quantizes each channel with its own parameters.
    /// 
    /// Instead of using the same scale for the whole tensor:
    /// - Each output channel gets its own scale (and optionally zero point)
    /// - This better preserves the relative differences between channels
    /// - It's especially useful for convolutional layers
    /// 
    /// This is more complex but typically gives better accuracy.
    /// </para>
    /// </remarks>
    PerChannel = 2
}

/// <summary>
/// Static registry for quantized model factory methods.
/// </summary>
/// <remarks>
/// <para>
/// This class maintains a registry of factory methods for deserializing quantized models
/// of different types.
/// </para>
/// <para><b>For Beginners:</b> This keeps track of ways to create different quantized models.
/// 
/// Since different model types need different deserialization approaches, this registry:
/// - Maps model types to their appropriate factory methods
/// - Allows the deserializer to work with any supported model type
/// - Can be extended to support new model types
/// </para>
/// </remarks>
public static class QuantizedModelFactoryRegistry
{
    private static readonly Dictionary<Type, object> _factories = 
        new Dictionary<Type, object>();
    
    /// <summary>
    /// Registers a factory for a specific model type.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
    /// <typeparam name="TModel">The type of model.</typeparam>
    /// <typeparam name="TInput">The input type for the model.</typeparam>
    /// <typeparam name="TOutput">The output type for the model.</typeparam>
    /// <param name="factory">The factory to register.</param>
    /// <remarks>
    /// <para>
    /// This method registers a factory that can create quantized models of the specified type.
    /// </para>
    /// <para><b>For Beginners:</b> This adds a new model type to the registry.
    ///
    /// When adding support for a new model type:
    /// 1. Create a factory that knows how to deserialize that model type
    /// 2. Register it using this method
    /// 3. The quantization system can now work with that model type
    /// </para>
    /// </remarks>
    public static void RegisterFactory<T, TModel, TInput, TOutput>(
        IQuantizedModelFactory<T, TModel, TInput, TOutput> factory)
        where T : unmanaged
        where TModel : class, IFullModel<T, TInput, TOutput>
    {
        _factories[typeof(TModel)] = factory;
    }
    
    /// <summary>
    /// Gets a factory for a specific model type.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
    /// <typeparam name="TModel">The type of model.</typeparam>
    /// <typeparam name="TInput">The input type for the model.</typeparam>
    /// <typeparam name="TOutput">The output type for the model.</typeparam>
    /// <returns>The registered factory, or null if no factory is registered for the type.</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves a factory that can create quantized models of the specified type.
    /// </para>
    /// <para><b>For Beginners:</b> This finds the right factory for a model type.
    ///
    /// When deserializing a model:
    /// 1. We need to know how to create that specific model type
    /// 2. This method finds the factory that knows how to do that
    /// 3. The factory then handles the model-specific deserialization
    /// </para>
    /// </remarks>
    public static IQuantizedModelFactory<T, TModel, TInput, TOutput>? GetFactory<T, TModel, TInput, TOutput>()
        where T : unmanaged
        where TModel : class, IFullModel<T, TInput, TOutput>
    {
        if (_factories.TryGetValue(typeof(TModel), out var factory))
        {
            return (IQuantizedModelFactory<T, TModel, TInput, TOutput>)factory;
        }
        
        return null;
    }
}

/// <summary>
/// Interface for factories that can create quantized models.
/// </summary>
/// <remarks>
/// <para>
/// This interface defines methods for creating quantized models from serialized data.
/// </para>
/// <para><b>For Beginners:</b> This defines how to create quantized models from saved data.
/// 
/// A factory implementing this interface knows:
/// - How to read the serialized data for a specific model type
/// - How to reconstruct the quantized model from that data
/// - How to set up any model-specific structures or parameters
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TModel">The type of model.</typeparam>
/// <typeparam name="TInput">The input type for the model.</typeparam>
/// <typeparam name="TOutput">The output type for the model.</typeparam>
public interface IQuantizedModelFactory<T, TModel, TInput, TOutput>
    where T : unmanaged
    where TModel : class, IFullModel<T, TInput, TOutput>
{
    /// <summary>
    /// Deserializes a quantized model from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader containing the serialized model.</param>
    /// <param name="method">The quantization method used.</param>
    /// <param name="bitWidth">The bit width used for quantization.</param>
    /// <param name="useMixedPrecision">Whether mixed precision was used.</param>
    /// <returns>The deserialized quantized model.</returns>
    /// <remarks>
    /// <para>
    /// This method reads serialized data and constructs a quantized model from it.
    /// </para>
    /// <para><b>For Beginners:</b> This rebuilds a quantized model from saved data.
    /// 
    /// The factory reads the serialized data and:
    /// 1. Creates the appropriate model instance
    /// 2. Sets up the quantized parameters
    /// 3. Configures any model-specific structures
    /// 4. Returns a fully functional quantized model
    /// </para>
    /// </remarks>
    TModel DeserializeQuantizedModel(
        BinaryReader reader,
        QuantizationMethod method,
        int bitWidth,
        bool useMixedPrecision);
}