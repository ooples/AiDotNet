namespace AiDotNet.Onnx;

/// <summary>
/// Named tensors flowing INTO a layer's ONNX node(s). The names refer to tensors that
/// have already been added to the <see cref="OnnxGraphBuilder"/> by an upstream layer
/// (or the model input). A layer converter consumes these names as the `inputs` field
/// of the ONNX nodes it emits.
/// </summary>
public sealed record OnnxLayerInputs(IReadOnlyList<string> TensorNames)
{
    /// <summary>Convenience constructor for the common single-input case.</summary>
    public OnnxLayerInputs(string tensorName) : this(new[] { tensorName }) { }

    /// <summary>The primary input tensor name (the first entry). Most layers have exactly one input.</summary>
    public string Primary => TensorNames[0];

    /// <summary>True if this layer has more than one input tensor (e.g., a concat or residual add).</summary>
    public bool HasMultiple => TensorNames.Count > 1;
}
