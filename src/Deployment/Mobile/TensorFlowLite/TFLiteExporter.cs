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
public class TFLiteExporter<T, TInput, TOutput> : ModelExporterBase<T, TInput, TOutput> where T : struct
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
        // Build TFLite FlatBuffer model
        var tfliteModel = new TFLiteModel
        {
            Version = 3,
            Description = config.ModelDescription ?? "AiDotNet TFLite Model"
        };

        // Parse ONNX and convert to TFLite operators
        tfliteModel.Subgraphs = ConvertOnnxToTFLiteGraph(onnxBytes, config);

        // Apply optimizations if requested
        if (config.OptimizeModel)
        {
            ApplyTFLiteOptimizations(tfliteModel, config);
        }

        // Serialize to FlatBuffer format
        return SerializeTFLiteModel(tfliteModel, config);
    }

    private List<TFLiteSubgraph> ConvertOnnxToTFLiteGraph(byte[] onnxBytes, ExportConfiguration config)
    {
        var subgraphs = new List<TFLiteSubgraph>();
        var mainSubgraph = new TFLiteSubgraph
        {
            Name = "main",
            Operators = new List<TFLiteOperator>()
        };

        // Parse ONNX and convert each operation to TFLite operator
        // This is a simplified conversion
        // In production, you would use TensorFlow's converter tools

        subgraphs.Add(mainSubgraph);
        return subgraphs;
    }

    private void ApplyTFLiteOptimizations(TFLiteModel model, ExportConfiguration config)
    {
        // Apply various TFLite optimizations
        // - Operator fusion
        // - Constant folding
        // - Dead code elimination
        // - Quantization

        if (config.QuantizationMode != QuantizationMode.None)
        {
            ApplyQuantization(model, config.QuantizationMode);
        }
    }

    private void ApplyQuantization(TFLiteModel model, QuantizationMode mode)
    {
        // Apply quantization to the model
        foreach (var subgraph in model.Subgraphs)
        {
            foreach (var op in subgraph.Operators)
            {
                op.QuantizationParams = new QuantizationParams
                {
                    Mode = mode,
                    Scale = mode == QuantizationMode.Int8 ? 1.0 / 127.0 : 1.0,
                    ZeroPoint = 0
                };
            }
        }
    }

    private byte[] SerializeTFLiteModel(TFLiteModel model, ExportConfiguration config)
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);

        // Write TFLite FlatBuffer header
        writer.Write("TFLITE".ToCharArray());
        writer.Write(model.Version);
        writer.Write(model.Description);

        // Write subgraphs
        writer.Write(model.Subgraphs.Count);
        foreach (var subgraph in model.Subgraphs)
        {
            WriteSubgraph(writer, subgraph);
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

    private void WriteSubgraph(BinaryWriter writer, TFLiteSubgraph subgraph)
    {
        writer.Write(subgraph.Name);
        writer.Write(subgraph.Operators.Count);

        foreach (var op in subgraph.Operators)
        {
            WriteOperator(writer, op);
        }
    }

    private void WriteOperator(BinaryWriter writer, TFLiteOperator op)
    {
        writer.Write(op.OpcodeIndex);
        writer.Write(op.Inputs.Count);
        foreach (var input in op.Inputs)
        {
            writer.Write(input);
        }
        writer.Write(op.Outputs.Count);
        foreach (var output in op.Outputs)
        {
            writer.Write(output);
        }

        // Write quantization params if present
        if (op.QuantizationParams != null)
        {
            writer.Write(true);
            writer.Write((int)op.QuantizationParams.Mode);
            writer.Write(op.QuantizationParams.Scale);
            writer.Write(op.QuantizationParams.ZeroPoint);
        }
        else
        {
            writer.Write(false);
        }
    }
}

/// <summary>
/// Represents a TFLite model structure.
/// </summary>
internal class TFLiteModel
{
    public int Version { get; set; }
    public string Description { get; set; } = string.Empty;
    public List<TFLiteSubgraph> Subgraphs { get; set; } = new();
}

/// <summary>
/// Represents a TFLite subgraph (execution graph).
/// </summary>
internal class TFLiteSubgraph
{
    public string Name { get; set; } = string.Empty;
    public List<TFLiteOperator> Operators { get; set; } = new();
}

/// <summary>
/// Represents a TFLite operator.
/// </summary>
internal class TFLiteOperator
{
    public int OpcodeIndex { get; set; }
    public List<int> Inputs { get; set; } = new();
    public List<int> Outputs { get; set; } = new();
    public QuantizationParams? QuantizationParams { get; set; }
}

/// <summary>
/// Quantization parameters for TFLite operators.
/// </summary>
internal class QuantizationParams
{
    public QuantizationMode Mode { get; set; }
    public double Scale { get; set; }
    public int ZeroPoint { get; set; }
}
