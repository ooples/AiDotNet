namespace AiDotNet.Onnx;

/// <summary>
/// Named tensors flowing OUT of a layer's ONNX node(s). The names are the tensors a
/// layer's ONNX nodes write to; downstream layers will reference them as inputs.
/// The graph builder uses these names to wire the next layer's <see cref="OnnxLayerInputs"/>.
/// </summary>
public sealed record OnnxLayerOutputs(IReadOnlyList<string> TensorNames)
{
    /// <summary>Convenience constructor for the common single-output case.</summary>
    public OnnxLayerOutputs(string tensorName) : this(new[] { tensorName }) { }

    /// <summary>The primary output tensor name (the first entry). Most layers have exactly one output.</summary>
    public string Primary => TensorNames[0];

    /// <summary>True if this layer produces more than one output tensor (e.g., a split or a BN training-mode emit).</summary>
    public bool HasMultiple => TensorNames.Count > 1;
}
