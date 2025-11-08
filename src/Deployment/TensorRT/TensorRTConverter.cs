using AiDotNet.Deployment.Export;
using AiDotNet.Deployment.Export.Onnx;

namespace AiDotNet.Deployment.TensorRT;

/// <summary>
/// Converts models to TensorRT optimized format for NVIDIA GPU deployment.
/// </summary>
/// <typeparam name="T">The numeric type used in the model</typeparam>
public class TensorRTConverter<T> where T : struct
{
    private readonly OnnxModelExporter<T> _onnxExporter;

    public TensorRTConverter()
    {
        _onnxExporter = new OnnxModelExporter<T>();
    }

    /// <summary>
    /// Converts a model to TensorRT format.
    /// </summary>
    /// <param name="model">The model to convert</param>
    /// <param name="outputPath">Output path for the TensorRT engine</param>
    /// <param name="config">TensorRT conversion configuration</param>
    public void ConvertToTensorRT(object model, string outputPath, TensorRTConfiguration config)
    {
        if (model == null)
            throw new ArgumentNullException(nameof(model));
        if (string.IsNullOrWhiteSpace(outputPath))
            throw new ArgumentException("Output path cannot be null or empty", nameof(outputPath));
        if (config == null)
            throw new ArgumentNullException(nameof(config));

        // Step 1: Export to ONNX first
        var onnxPath = Path.ChangeExtension(outputPath, ".onnx");
        var exportConfig = ExportConfiguration.ForTensorRT(config.MaxBatchSize, config.UseFp16);
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
    public byte[] ConvertToTensorRTBytes(object model, TensorRTConfiguration config)
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
            UseFp16 = config.UseFp16,
            UseInt8 = config.UseInt8,
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
        builder.DlaCore = config.DlaCore;

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
    /// Placeholder implementation for TensorRT engine serialization.
    /// CRITICAL: This is not a real TensorRT engine and cannot be used for actual inference.
    /// In production, this would call the NVIDIA TensorRT C++ library to build and serialize a real engine.
    /// For now, creates a metadata file that describes the engine configuration only.
    /// </summary>
    private byte[] SerializeTensorRTEngine(TensorRTEngineBuilder builder, string onnxPath, TensorRTConfiguration config)
    {
        // This is a placeholder for actual TensorRT engine serialization
        // In production, this would interface with NVIDIA TensorRT C++ library
        // For now, we'll create a metadata file that describes the engine

        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);

        // Write TensorRT engine header
        writer.Write("TRTENGINE".ToCharArray());
        writer.Write(1); // Version
        writer.Write(builder.MaxBatchSize);
        writer.Write(builder.MaxWorkspaceSize);
        writer.Write(builder.UseFp16);
        writer.Write(builder.UseInt8);
        writer.Write(builder.StrictTypeConstraints);
        writer.Write(builder.EnableDynamicShapes);
        writer.Write(builder.DeviceId);
        writer.Write(builder.DlaCore ?? -1);

        // Write ONNX model path reference
        writer.Write(onnxPath);

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
        if (builder.UseInt8)
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

/// <summary>
/// Internal class for building TensorRT engines.
/// </summary>
internal class TensorRTEngineBuilder
{
    public int MaxBatchSize { get; set; }
    public long MaxWorkspaceSize { get; set; }
    public bool UseFp16 { get; set; }
    public bool UseInt8 { get; set; }
    public bool StrictTypeConstraints { get; set; }
    public bool EnableDynamicShapes { get; set; }
    public int DeviceId { get; set; }
    public int? DlaCore { get; set; }
    public List<OptimizationProfile>? OptimizationProfiles { get; set; }
}

/// <summary>
/// Represents a TensorRT optimization profile for dynamic shapes.
/// </summary>
public class OptimizationProfile
{
    public string? InputName { get; set; }
    public int[]? MinShape { get; set; }
    public int[]? OptimalShape { get; set; }
    public int[]? MaxShape { get; set; }
}
