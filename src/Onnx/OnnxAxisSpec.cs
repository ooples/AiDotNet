namespace AiDotNet.Onnx;

/// <summary>
/// Per-axis shape descriptor for an ONNX <c>TensorShapeProto.Dimension</c>.
/// Encodes either a concrete <c>dim_value</c> (e.g. channel count = 3) or a
/// symbolic <c>dim_param</c> (e.g. <c>"batch"</c>, <c>"H"</c>, <c>"W"</c>).
/// </summary>
/// <remarks>
/// <para>
/// Symbolic axes (issue #1211) let a single exported ONNX file run at any
/// (batch, height, width) the downstream runtime feeds it. ONNX Runtime,
/// OpenVINO and TensorRT all expect symbolic axes for production deployment;
/// PyTorch surfaces them via <c>torch.onnx.export(..., dynamic_axes=...)</c>.
/// </para>
/// <para>
/// Construct with the static <see cref="Fixed(int)"/> or
/// <see cref="Symbolic(string)"/> factory methods.
/// </para>
/// </remarks>
public readonly struct OnnxAxisSpec
{
    /// <summary>Concrete dimension size when <see cref="SymbolicName"/> is null.</summary>
    public int FixedDim { get; }

    /// <summary>Symbolic name (e.g. "batch", "H", "W") when this axis is dynamic.</summary>
    public string? SymbolicName { get; }

    private OnnxAxisSpec(int fixedDim, string? symbolicName)
    {
        FixedDim = fixedDim;
        SymbolicName = symbolicName;
    }

    /// <summary>Concrete-size axis (encoded as ONNX <c>dim_value</c>).</summary>
    public static OnnxAxisSpec Fixed(int dim) => new OnnxAxisSpec(dim, null);

    /// <summary>Symbolic axis (encoded as ONNX <c>dim_param</c>).</summary>
    public static OnnxAxisSpec Symbolic(string name)
    {
        if (string.IsNullOrWhiteSpace(name))
            throw new ArgumentException("Symbolic axis name must be non-empty.", nameof(name));
        return new OnnxAxisSpec(0, name);
    }
}
