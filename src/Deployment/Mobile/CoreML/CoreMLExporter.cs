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
public class CoreMLExporter<T, TInput, TOutput> : ModelExporterBase<T, TInput, TOutput>
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
        if (config == null)
            throw new ArgumentNullException(nameof(config));
        if (string.IsNullOrWhiteSpace(outputPath))
            throw new ArgumentException("Output path cannot be null or empty", nameof(outputPath));

        var exportConfig = config.ToExportConfiguration();

        // Preserve CoreML-specific configuration in PlatformSpecificOptions
        exportConfig.PlatformSpecificOptions["CoreMLConfiguration"] = config;

        Export(model, outputPath, exportConfig);
    }

    private byte[] ConvertOnnxToCoreML(byte[] onnxBytes, ExportConfiguration config)
    {
        if (config == null)
            throw new ArgumentNullException(nameof(config));

        // Try to retrieve preserved CoreML configuration from PlatformSpecificOptions
        var coreMLConfig = config.PlatformSpecificOptions.TryGetValue("CoreMLConfiguration", out var configObj) &&
            configObj is CoreMLConfiguration preservedConfig
            ? preservedConfig
            : new CoreMLConfiguration
            {
                ModelName = config.ModelName,
                ModelDescription = config.ModelDescription,
                OptimizeForSize = true,
                QuantizationBits = config.QuantizationMode switch
                {
                    QuantizationMode.Int8 => 8,
                    QuantizationMode.Float16 => 16,
                    _ => 32
                }
            };

        // Perform ONNX→CoreML conversion using production-ready converter
        var coreMLModel = OnnxToCoreMLConverter.ConvertOnnxToCoreML(onnxBytes, coreMLConfig);

        // Serialize to Apple CoreML protobuf format
        return CoreMLProto.CreateModelProto(coreMLModel);
    }
}
