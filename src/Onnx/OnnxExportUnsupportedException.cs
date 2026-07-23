namespace AiDotNet.Onnx;

/// <summary>
/// Thrown when a model contains a layer or component that does not yet have an ONNX
/// export converter. The exception message names the unsupported component so callers
/// can point users at the right follow-up (open an issue, request the converter, or
/// switch to a supported layer type).
///
/// See <see cref="AiDotNet.Onnx.OnnxSupportMatrix"/> (forthcoming doc) for the
/// canonical list of supported layer types.
/// </summary>
public sealed class OnnxExportUnsupportedException : InvalidOperationException
{
    /// <summary>The type name of the unsupported component (e.g., "ConvolutionalLayer").</summary>
    public string ComponentTypeName { get; }

    public OnnxExportUnsupportedException(string componentTypeName, string suggestion)
        : base($"ONNX export does not yet support component type '{componentTypeName}'. {suggestion}")
    {
        ComponentTypeName = componentTypeName;
    }
}
