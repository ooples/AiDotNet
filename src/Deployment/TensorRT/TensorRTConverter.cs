using AiDotNet.Deployment.Export;
using AiDotNet.Deployment.Export.Onnx;
using AiDotNet.Interfaces;

namespace AiDotNet.Deployment.TensorRT;

/// <summary>
/// Converts models to TensorRT optimized format for NVIDIA GPU deployment.
/// Properly integrates with IFullModel architecture.
/// </summary>
/// <typeparam name="T">The numeric type used in the model</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
public class TensorRTConverter<T, TInput, TOutput>
{
    private readonly OnnxModelExporter<T, TInput, TOutput> _onnxExporter;

    public TensorRTConverter()
    {
        _onnxExporter = new OnnxModelExporter<T, TInput, TOutput>();
    }

    /// <summary>
    /// Converts a model to TensorRT format.
    /// </summary>
    /// <param name="model">The model to convert</param>
    /// <param name="outputPath">Output path for the TensorRT engine</param>
    /// <param name="config">TensorRT conversion configuration</param>
    public void ConvertToTensorRT(IFullModel<T, TInput, TOutput> model, string outputPath, TensorRTConfiguration config)
    {
        if (model == null)
            throw new ArgumentNullException(nameof(model));
        if (string.IsNullOrWhiteSpace(outputPath))
            throw new ArgumentException("Output path cannot be null or empty", nameof(outputPath));
        if (config == null)
            throw new ArgumentNullException(nameof(config));

        // Step 1: Export to ONNX first
        var onnxPath = Path.ChangeExtension(outputPath, ".onnx");
        var exportConfig = ExportConfiguration.ForTensorRT(config.MaxBatchSize, config.Precision == TensorRTPrecision.FP16);
        _onnxExporter.Export(model, onnxPath, exportConfig);

        // Step 2: Build TensorRT engine from ONNX
        BuildTensorRTEngine(onnxPath, outputPath, config);

        // Step 3: Clean up intermediate ONNX file if requested
        if (config.CleanupIntermediateFiles && File.Exists(onnxPath))
        {
            File.Delete(onnxPath);
        }
    }

    /// <summary>
    /// Converts a model to TensorRT format and returns as byte array.
    /// </summary>
    public byte[] ConvertToTensorRTBytes(IFullModel<T, TInput, TOutput> model, TensorRTConfiguration config)
    {
        var tempPath = Path.Combine(Path.GetTempPath(), $"tensorrt_{Guid.NewGuid()}.engine");

        try
        {
            ConvertToTensorRT(model, tempPath, config);
            return File.ReadAllBytes(tempPath);
        }
        finally
        {
            if (File.Exists(tempPath))
            {
                File.Delete(tempPath);
            }
        }
    }

    private void BuildTensorRTEngine(string onnxPath, string enginePath, TensorRTConfiguration config)
    {
        // Create TensorRT builder metadata
        var builder = new TensorRTEngineBuilder
        {
            MaxBatchSize = config.MaxBatchSize,
            MaxWorkspaceSize = config.MaxWorkspaceSize,
            Precision = config.Precision,
            StrictTypeConstraints = config.StrictTypeConstraints,
            EnableDynamicShapes = config.EnableDynamicShapes
        };

        // Build optimization profiles
        if (config.EnableDynamicShapes)
        {
            builder.OptimizationProfiles = BuildOptimizationProfiles(config);
        }

        // Configure device properties
        builder.DeviceId = config.DeviceId;
        builder.DLACore = config.DLACore;

        // Serialize engine configuration
        var engineData = SerializeTensorRTEngine(builder, onnxPath, config);

        // Write to output
        File.WriteAllBytes(enginePath, engineData);
    }

    private List<OptimizationProfile> BuildOptimizationProfiles(TensorRTConfiguration config)
    {
        return config.OptimizationProfiles.Select(profileConfig => new OptimizationProfile
        {
            MinShape = profileConfig.MinShape,
            OptimalShape = profileConfig.OptimalShape,
            MaxShape = profileConfig.MaxShape,
            InputName = profileConfig.InputName
        }).ToList();
    }

    /// <summary>
    /// Serializes TensorRT engine configuration for use with ONNX Runtime TensorRT execution provider.
    /// This creates a configuration file that tells the inference engine how to use TensorRT via ONNX Runtime.
    /// The actual TensorRT engine building is handled by ONNX Runtime's TensorRT EP at runtime.
    /// </summary>
    private byte[] SerializeTensorRTEngine(TensorRTEngineBuilder builder, string onnxPath, TensorRTConfiguration config)
    {
        // Create a TensorRT configuration package containing:
        // 1. ONNX model
        // 2. TensorRT execution provider settings
        // This allows the inference engine to use real TensorRT acceleration via ONNX Runtime

        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);

        // Write TensorRT engine header (version 2 = ONNX Runtime TensorRT EP)
        writer.Write("TRTENGINE".ToCharArray());
        writer.Write(2); // Version 2 indicates ONNX Runtime TensorRT EP approach

        // Write TensorRT configuration
        writer.Write(builder.MaxBatchSize);
        writer.Write(builder.MaxWorkspaceSize);
        writer.Write((int)builder.Precision);
        writer.Write(builder.StrictTypeConstraints);
        writer.Write(builder.EnableDynamicShapes);
        writer.Write(builder.DeviceId);
        writer.Write(builder.DLACore);

        // Embed the ONNX model data (allows self-contained engine file)
        var onnxData = File.ReadAllBytes(onnxPath);
        writer.Write(onnxData.Length);
        writer.Write(onnxData);

        // Write optimization profiles
        writer.Write(builder.OptimizationProfiles?.Count ?? 0);
        if (builder.OptimizationProfiles != null)
        {
            foreach (var profile in builder.OptimizationProfiles)
            {
                writer.Write(profile.InputName ?? "input");
                WriteShape(writer, profile.MinShape);
                WriteShape(writer, profile.OptimalShape);
                WriteShape(writer, profile.MaxShape);
            }
        }

        // Write custom operator plugins
        writer.Write(config.CustomPluginPaths.Count);
        foreach (var pluginPath in config.CustomPluginPaths)
        {
            writer.Write(pluginPath);
        }

        // Write calibration data path if INT8 is used
        if (builder.Precision == TensorRTPrecision.INT8)
        {
            writer.Write(config.CalibrationDataPath ?? string.Empty);
        }

        return stream.ToArray();
    }

    private void WriteShape(BinaryWriter writer, int[]? shape)
    {
        if (shape == null)
        {
            writer.Write(0);
        }
        else
        {
            writer.Write(shape.Length);
            foreach (var dim in shape)
            {
                writer.Write(dim);
            }
        }
    }
}
