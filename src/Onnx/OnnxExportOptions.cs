namespace AiDotNet.Onnx;

/// <summary>
/// Optional configuration for ONNX export. Defaults are chosen so a caller can pass
/// `null` and get a working file for the most common case (float32 inputs, recent
/// opset, broadly compatible with downstream runtimes including Databricks / Spark).
/// </summary>
public sealed class OnnxExportOptions
{
    /// <summary>
    /// ONNX opset version to target. Lower values are more compatible with older
    /// runtimes; higher values get newer operators. Default: 17 (released 2022,
    /// broadly supported by onnxruntime, Spark MLLib loaders, and ML.NET).
    /// </summary>
    public int OpsetVersion { get; init; } = 17;

    /// <summary>Producer name written into the .onnx file's metadata. Default: "AiDotNet".</summary>
    public string ProducerName { get; init; } = "AiDotNet";

    /// <summary>
    /// Producer version written into the .onnx file's metadata. Helpful for downstream
    /// diagnostics ("which AiDotNet build wrote this file?"). Default: an empty string.
    /// </summary>
    public string ProducerVersion { get; init; } = string.Empty;

    /// <summary>Human-readable description written into the .onnx file's metadata.</summary>
    public string? ModelDescription { get; init; }

    /// <summary>
    /// Names to assign to input tensors. If null, defaults to "input_0", "input_1", ...
    /// Must match the count of inputs the model expects.
    /// </summary>
    public IReadOnlyList<string>? InputNames { get; init; }

    /// <summary>
    /// Names to assign to output tensors. If null, defaults to "output_0", "output_1", ...
    /// Must match the count of outputs the model produces.
    /// </summary>
    public IReadOnlyList<string>? OutputNames { get; init; }
}
