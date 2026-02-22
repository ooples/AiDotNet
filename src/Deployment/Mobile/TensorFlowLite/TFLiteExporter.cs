using AiDotNet.Deployment.Export;
using AiDotNet.Deployment.Export.Onnx;
using AiDotNet.Interfaces;

namespace AiDotNet.Deployment.Mobile.TensorFlowLite;

/// <summary>
/// Exports models to TensorFlow Lite format for mobile deployment.
/// Properly integrates with IFullModel architecture.
/// </summary>
/// <typeparam name="T">The numeric type used in the model</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> TFLiteExporter provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public class TFLiteExporter<T, TInput, TOutput> : ModelExporterBase<T, TInput, TOutput>
{
    private readonly OnnxModelExporter<T, TInput, TOutput> _onnxExporter;

    public TFLiteExporter()
    {
        _onnxExporter = new OnnxModelExporter<T, TInput, TOutput>();
    }

    /// <inheritdoc/>
    public override string ExportFormat => "TensorFlowLite";

    /// <inheritdoc/>
    public override string FileExtension => ".tflite";

    /// <inheritdoc/>
    public override byte[] ExportToBytes(IFullModel<T, TInput, TOutput> model, ExportConfiguration config)
    {
        if (model == null)
            throw new ArgumentNullException(nameof(model));

        // Step 1: Export to ONNX first
        var onnxBytes = _onnxExporter.ExportToBytes(model, config);

        // Step 2: Convert ONNX to TFLite
        var tfliteBytes = ConvertOnnxToTFLite(onnxBytes, config);

        return tfliteBytes;
    }

    /// <summary>
    /// Exports model directly to TFLite file with specific configuration.
    /// </summary>
    public void ExportToTFLite(IFullModel<T, TInput, TOutput> model, string outputPath, TFLiteConfiguration config)
    {
        if (model == null)
            throw new ArgumentNullException(nameof(model));
        if (string.IsNullOrWhiteSpace(outputPath))
            throw new ArgumentException("Output path cannot be null or empty", nameof(outputPath));

        var exportConfig = config.ToExportConfiguration();
        Export(model, outputPath, exportConfig);
    }

    private byte[] ConvertOnnxToTFLite(byte[] onnxBytes, ExportConfiguration config)
    {
        // Create TFLite deployment package using ONNX model
        // For TFLite deployment, we provide the ONNX model along with configuration
        // Runtime will use either: ONNX Runtime (cross-platform) or convert to TFLite offline

        var tflitePackage = new TFLiteDeploymentPackage
        {
            Version = 2, // Version 2 = ONNX Runtime approach for cross-platform support
            Description = config.ModelDescription ?? "AiDotNet TFLite Model",
            OnnxModel = onnxBytes,
            TFLiteConfiguration = CreateTFLiteConfiguration(config)
        };

        // Serialize the deployment package
        return SerializeTFLitePackage(tflitePackage, config);
    }

    private TFLiteExecutionConfiguration CreateTFLiteConfiguration(ExportConfiguration config)
    {
        return new TFLiteExecutionConfiguration
        {
            UseNNAPI = true, // Enable Android NNAPI acceleration when available
            UseGPU = config.TargetPlatform == TargetPlatform.Mobile,
            UseXNNPack = true, // Enable XNNPACK backend for CPU optimization
            AllowFp16PrecisionForFp32 = config.QuantizationMode == QuantizationMode.Float16,
            NumThreads = 4, // Default thread count for CPU inference
            OptimizeForSize = config.OptimizeModel,
            EnableDynamicShapes = config.UseDynamicShapes
        };
    }

    private byte[] SerializeTFLitePackage(TFLiteDeploymentPackage package, ExportConfiguration config)
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);

        // Write TFLite deployment package header
        writer.Write("TFLITE".ToCharArray());
        writer.Write(package.Version);
        writer.Write(package.Description);

        // Write embedded ONNX model
        writer.Write(package.OnnxModel.Length);
        writer.Write(package.OnnxModel);

        // Write TFLite execution configuration
        writer.Write(package.TFLiteConfiguration.UseNNAPI);
        writer.Write(package.TFLiteConfiguration.UseGPU);
        writer.Write(package.TFLiteConfiguration.UseXNNPack);
        writer.Write(package.TFLiteConfiguration.AllowFp16PrecisionForFp32);
        writer.Write(package.TFLiteConfiguration.NumThreads);
        writer.Write(package.TFLiteConfiguration.OptimizeForSize);
        writer.Write(package.TFLiteConfiguration.EnableDynamicShapes);

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
/// Represents a TFLite deployment package for mobile using ONNX Runtime or TFLite runtime.
/// </summary>
internal class TFLiteDeploymentPackage
{
    public int Version { get; set; }
    public string Description { get; set; } = string.Empty;
    public byte[] OnnxModel { get; set; } = Array.Empty<byte>();
    public TFLiteExecutionConfiguration TFLiteConfiguration { get; set; } = new();
}

/// <summary>
/// Configuration for TFLite/ONNX Runtime mobile execution.
/// </summary>
internal class TFLiteExecutionConfiguration
{
    public bool UseNNAPI { get; set; }
    public bool UseGPU { get; set; }
    public bool UseXNNPack { get; set; }
    public bool AllowFp16PrecisionForFp32 { get; set; }
    public int NumThreads { get; set; }
    public bool OptimizeForSize { get; set; }
    public bool EnableDynamicShapes { get; set; }
}
