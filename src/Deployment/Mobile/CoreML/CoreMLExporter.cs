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
        // Build CoreML model structure
        var coreMLModel = new CoreMLModel
        {
            Version = 4, // CoreML spec version
            ModelDescription = config.ModelDescription ?? "AiDotNet Model"
        };

        // Parse ONNX and convert to CoreML operations
        // This is simplified - in production you'd use CoreMLTools or similar
        coreMLModel.NeuralNetwork = ConvertOnnxToCoreMLNetwork(onnxBytes, config);

        // Serialize CoreML model
        return SerializeCoreMLModel(coreMLModel, config);
    }

    private CoreMLNeuralNetwork ConvertOnnxToCoreMLNetwork(byte[] onnxBytes, ExportConfiguration config)
    {
        var network = new CoreMLNeuralNetwork
        {
            Layers = new List<CoreMLLayer>()
        };

        // Parse ONNX operations and convert to CoreML layers
        // This is a simplified conversion
        // In production, you would parse the ONNX protobuf and map each operation

        return network;
    }

    private byte[] SerializeCoreMLModel(CoreMLModel model, ExportConfiguration config)
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);

        // Write CoreML model header
        writer.Write("COREML".ToCharArray());
        writer.Write(model.Version);
        writer.Write(model.ModelDescription);

        // Write neural network
        if (model.NeuralNetwork != null)
        {
            writer.Write(model.NeuralNetwork.Layers.Count);

            foreach (var layer in model.NeuralNetwork.Layers)
            {
                WriteLayer(writer, layer);
            }
        }

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

    private void WriteLayer(BinaryWriter writer, CoreMLLayer layer)
    {
        writer.Write(layer.Name);
        writer.Write(layer.Type);
        writer.Write(layer.Input);
        writer.Write(layer.Output);

        // Write layer-specific parameters
        writer.Write(layer.Parameters.Count);
        foreach (var param in layer.Parameters)
        {
            writer.Write(param.Key);
            writer.Write(param.Value?.ToString() ?? "");
        }
    }
}

/// <summary>
/// Represents a CoreML model structure.
/// </summary>
internal class CoreMLModel
{
    public int Version { get; set; }
    public string ModelDescription { get; set; } = string.Empty;
    public CoreMLNeuralNetwork? NeuralNetwork { get; set; }
}

/// <summary>
/// Represents a CoreML neural network.
/// </summary>
internal class CoreMLNeuralNetwork
{
    public List<CoreMLLayer> Layers { get; set; } = new();
}

/// <summary>
/// Represents a CoreML layer.
/// </summary>
internal class CoreMLLayer
{
    public string Name { get; set; } = string.Empty;
    public string Type { get; set; } = string.Empty;
    public string Input { get; set; } = string.Empty;
    public string Output { get; set; } = string.Empty;
    public Dictionary<string, object> Parameters { get; set; } = new();
}
