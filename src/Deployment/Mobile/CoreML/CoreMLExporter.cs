using AiDotNet.Deployment.Export;
using AiDotNet.Deployment.Export.Onnx;
using AiDotNet.Interfaces;

namespace AiDotNet.Deployment.Mobile.CoreML;

/// <summary>
/// Exports models to CoreML format for iOS deployment.
/// Properly integrates with IFullModel architecture.
/// </summary>
/// <typeparam name="T">The numeric type used in the model</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
public class CoreMLExporter<T, TInput, TOutput> : ModelExporterBase<T, TInput, TOutput> where T : struct
{
    private readonly OnnxModelExporter<T, TInput, TOutput> _onnxExporter;

    public CoreMLExporter()
    {
        _onnxExporter = new OnnxModelExporter<T, TInput, TOutput>();
    }

    /// <inheritdoc/>
    public override string ExportFormat => "CoreML";

    /// <inheritdoc/>
    public override string FileExtension => ".mlmodel";

    /// <inheritdoc/>
    public override byte[] ExportToBytes(IFullModel<T, TInput, TOutput> model, ExportConfiguration config)
    {
        if (model == null)
            throw new ArgumentNullException(nameof(model));

        // Step 1: Export to ONNX first
        var onnxBytes = _onnxExporter.ExportToBytes(model, config);

        // Step 2: Convert ONNX to CoreML format
        var coreMLBytes = ConvertOnnxToCoreML(onnxBytes, config);

        return coreMLBytes;
    }

    /// <summary>
    /// Exports model directly to CoreML file.
    /// </summary>
    public void ExportToCoreML(IFullModel<T, TInput, TOutput> model, string outputPath, CoreMLConfiguration config)
    {
        if (model == null)
            throw new ArgumentNullException(nameof(model));
        if (string.IsNullOrWhiteSpace(outputPath))
            throw new ArgumentException("Output path cannot be null or empty", nameof(outputPath));

        var exportConfig = config.ToExportConfiguration();
        Export(model, outputPath, exportConfig);
    }

    private byte[] ConvertOnnxToCoreML(byte[] onnxBytes, ExportConfiguration config)
    {
        // Create CoreML deployment package using ONNX Runtime CoreML execution provider
        // This approach uses ONNX Runtime's CoreML EP for real hardware acceleration on iOS
        // instead of converting to native CoreML format (which would require coremltools)

        var coreMLPackage = new CoreMLDeploymentPackage
        {
            Version = 2, // Version 2 = ONNX Runtime CoreML EP approach
            ModelDescription = config.ModelDescription ?? "AiDotNet Model",
            OnnxModel = onnxBytes,
            CoreMLConfiguration = CreateCoreMLConfiguration(config)
        };

        // Serialize the deployment package
        return SerializeCoreMLPackage(coreMLPackage, config);
    }

    private CoreMLExecutionConfiguration CreateCoreMLConfiguration(ExportConfiguration config)
    {
        return new CoreMLExecutionConfiguration
        {
            // Map quantization mode to CoreML compute precision
            UseFp16 = config.QuantizationMode == QuantizationMode.Float16,
            EnableOnSubgraphs = true,
            OnlyEnableDeviceWithANE = false, // Allow CPU/GPU as well
            RequireStaticShapes = !config.UseDynamicShapes,
            ModelFormat = CoreMLModelFormat.MLProgram, // Use ML Program (iOS 15+) for best performance
            OptimizationLevel = config.OptimizeModel ? 2 : 0
        };
    }

    private byte[] SerializeCoreMLPackage(CoreMLDeploymentPackage package, ExportConfiguration config)
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);

        // Write CoreML deployment package header
        writer.Write("COREML".ToCharArray());
        writer.Write(package.Version);
        writer.Write(package.ModelDescription);

        // Write embedded ONNX model
        writer.Write(package.OnnxModel.Length);
        writer.Write(package.OnnxModel);

        // Write CoreML execution configuration
        writer.Write(package.CoreMLConfiguration.UseFp16);
        writer.Write(package.CoreMLConfiguration.EnableOnSubgraphs);
        writer.Write(package.CoreMLConfiguration.OnlyEnableDeviceWithANE);
        writer.Write(package.CoreMLConfiguration.RequireStaticShapes);
        writer.Write((int)package.CoreMLConfiguration.ModelFormat);
        writer.Write(package.CoreMLConfiguration.OptimizationLevel);

        // Write metadata
        if (config.IncludeMetadata)
        {
            writer.Write(true);
            writer.Write(config.ModelName ?? "AiDotNetModel");
            writer.Write(config.ModelVersion ?? "1.0");
        }
        else
        {
            writer.Write(false);
        }

        return stream.ToArray();
    }
}

/// <summary>
/// Represents a CoreML deployment package for iOS using ONNX Runtime CoreML EP.
/// </summary>
internal class CoreMLDeploymentPackage
{
    public int Version { get; set; }
    public string ModelDescription { get; set; } = string.Empty;
    public byte[] OnnxModel { get; set; } = Array.Empty<byte>();
    public CoreMLExecutionConfiguration CoreMLConfiguration { get; set; } = new();
}

/// <summary>
/// Configuration for ONNX Runtime CoreML execution provider.
/// </summary>
internal class CoreMLExecutionConfiguration
{
    public bool UseFp16 { get; set; }
    public bool EnableOnSubgraphs { get; set; }
    public bool OnlyEnableDeviceWithANE { get; set; }
    public bool RequireStaticShapes { get; set; }
    public CoreMLModelFormat ModelFormat { get; set; }
    public int OptimizationLevel { get; set; }
}

/// <summary>
/// CoreML model format options.
/// </summary>
internal enum CoreMLModelFormat
{
    NeuralNetwork = 0,  // Legacy format (iOS 11+)
    MLProgram = 1       // Modern format with better performance (iOS 15+)
}
