using AiDotNet.Models.Results;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Onnx;

/// <summary>
/// Public-facing ONNX export API on <see cref="AiModelResult{T,TInput,TOutput}"/>.
/// These are the methods user code calls; everything in <c>src/Onnx/</c> below
/// this surface is implementation detail.
///
/// Usage:
/// <code>
///   var result = builder.Build(trainingData, validationData);
///   result.ExportToOnnx("model.onnx");
/// </code>
///
/// v0.1 supports sequential models composed of: DenseLayer, ActivationLayer
/// (ReLU/Sigmoid/Tanh/Softmax/Identity), BatchNormalizationLayer, DropoutLayer.
/// Other layer types throw <see cref="OnnxExportUnsupportedException"/> with
/// the unsupported layer's type name.
/// </summary>
public static class AiModelResultOnnxExtensions
{
    /// <summary>
    /// Exports a trained model to an ONNX file at <paramref name="filePath"/>.
    /// Throws <see cref="OnnxExportUnsupportedException"/> if the model contains
    /// a layer that does not yet have a <c>ConvertToOnnx</c> override.
    /// </summary>
    public static void ExportToOnnx<T, TInput, TOutput>(
        this AiModelResult<T, TInput, TOutput> result,
        string filePath,
        OnnxExportOptions? options = null)
    {
        if (result is null) throw new ArgumentNullException(nameof(result));
        if (string.IsNullOrWhiteSpace(filePath)) throw new ArgumentNullException(nameof(filePath));

        var directory = Path.GetDirectoryName(filePath);
        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

        using var fs = new FileStream(filePath, FileMode.Create, FileAccess.Write);
        ExportToOnnx(result, fs, options);
    }

    /// <summary>Exports to a stream — useful for uploading directly to cloud storage.</summary>
    public static void ExportToOnnx<T, TInput, TOutput>(
        this AiModelResult<T, TInput, TOutput> result,
        Stream stream,
        OnnxExportOptions? options = null)
    {
        if (result is null) throw new ArgumentNullException(nameof(result));
        if (stream is null) throw new ArgumentNullException(nameof(stream));

        options ??= new OnnxExportOptions();
        var builder = new OnnxGraphBuilder(options);

        var layers = GetLayers(result);
        if (layers.Count == 0)
        {
            throw new InvalidOperationException(
                "Cannot export to ONNX: model has no layers. Make sure the model is built and trained first.");
        }

        // Infer input/output shapes from the first/last layer. Symbolic batch dim.
        int? inputSize = TryGetInputSize(layers[0]);
        int? outputSize = TryGetOutputSize(layers[^1]);
        if (inputSize is null || outputSize is null)
        {
            throw new InvalidOperationException(
                "Cannot export to ONNX: could not infer input/output sizes from the first/last layers. " +
                "The model's first layer must expose InputSize and the last must expose OutputSize.");
        }

        var inputName = options.InputNames?.Count > 0 ? options.InputNames[0] : "input";
        var outputName = options.OutputNames?.Count > 0 ? options.OutputNames[0] : "output";

        builder.AddFloatInput(inputName, new[] { -1, inputSize.Value });

        var currentInputs = new OnnxLayerInputs(inputName);
        OnnxLayerOutputs lastOutputs = currentInputs is null
            ? throw new InvalidOperationException("unreachable")
            : new OnnxLayerOutputs(inputName);

        foreach (var layer in layers)
        {
            var convertMethod = layer.GetType().GetMethod("ConvertToOnnx");
            if (convertMethod is null)
            {
                throw new OnnxExportUnsupportedException(
                    layer.GetType().Name,
                    "Layer does not expose ConvertToOnnx — it is not a LayerBase-derived layer.");
            }
            lastOutputs = (OnnxLayerOutputs)convertMethod.Invoke(layer, new object[] { builder, currentInputs })!;
            currentInputs = new OnnxLayerInputs(lastOutputs.Primary);
        }

        // Re-name the final tensor to the configured outputName by wiring it as
        // a graph output identity. Simpler: just declare the graph output as the
        // final layer's emitted tensor name.
        builder.AddFloatOutput(lastOutputs.Primary, new[] { -1, outputSize.Value });

        builder.WriteTo(stream);
    }

    /// <summary>
    /// Returns true if every layer in the model has an ONNX converter today.
    /// Lets callers gate UX (e.g., disable the Export button) without raising.
    /// </summary>
    public static bool CanExportToOnnx<T, TInput, TOutput>(
        this AiModelResult<T, TInput, TOutput> result)
    {
        if (result is null) throw new ArgumentNullException(nameof(result));

        var layers = GetLayers(result);
        if (layers.Count == 0) return false;

        // Check that every layer has overridden ConvertToOnnx (vs. the default
        // that throws). We probe by invoking with a dummy builder — if it throws
        // OnnxExportUnsupportedException, the layer doesn't support export.
        var dummyBuilder = new OnnxGraphBuilder(new OnnxExportOptions());
        var dummyInputs = new OnnxLayerInputs("probe");

        foreach (var layer in layers)
        {
            var convertMethod = layer.GetType().GetMethod("ConvertToOnnx");
            if (convertMethod is null) return false;
            try
            {
                _ = convertMethod.Invoke(layer, new object[] { dummyBuilder, dummyInputs });
            }
            catch (System.Reflection.TargetInvocationException tie)
                when (tie.InnerException is OnnxExportUnsupportedException)
            {
                return false;
            }
            catch
            {
                // Other failures (e.g., uninitialised weights) don't disqualify
                // the layer from supporting export — only an explicit
                // OnnxExportUnsupportedException means "not supported".
            }
        }
        return true;
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    private static List<object> GetLayers<T, TInput, TOutput>(AiModelResult<T, TInput, TOutput> result)
    {
        var layers = new List<object>();

        // AiModelResult is itself an IFullModel; its Model property is the
        // underlying NeuralNetworkBase (or similar) which exposes Layers.
        // Fall back to reflecting Layers off result itself if a direct Model
        // property isn't found (covers different model topologies).
        object[] roots = new object[] { result };
        var modelProp = result.GetType().GetProperty("Model");
        if (modelProp is not null)
        {
            var inner = modelProp.GetValue(result);
            if (inner is not null) roots = new[] { inner, result };
        }

        foreach (var root in roots)
        {
            var layersProperty = root.GetType().GetProperty("Layers");
            if (layersProperty is null) continue;
            var layersList = layersProperty.GetValue(root);
            if (layersList is System.Collections.IEnumerable enumerable)
            {
                foreach (var layer in enumerable) layers.Add(layer);
                if (layers.Count > 0) break;
            }
        }
        return layers;
    }

    private static int? TryGetInputSize(object layer)
    {
        var inputSizeProp = layer.GetType().GetProperty("InputSize");
        if (inputSizeProp is not null) return (int?)inputSizeProp.GetValue(layer);

        var inputShapeProp = layer.GetType().GetProperty("InputShape");
        if (inputShapeProp?.GetValue(layer) is int[] shape && shape.Length > 0)
        {
            return shape[0];
        }
        return null;
    }

    private static int? TryGetOutputSize(object layer)
    {
        var outputSizeProp = layer.GetType().GetProperty("OutputSize");
        if (outputSizeProp is not null) return (int?)outputSizeProp.GetValue(layer);

        var outputShapeProp = layer.GetType().GetProperty("OutputShape");
        if (outputShapeProp?.GetValue(layer) is int[] shape && shape.Length > 0)
        {
            return shape[0];
        }
        return null;
    }
}
