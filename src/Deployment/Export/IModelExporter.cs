using System.Collections.Generic;

namespace AiDotNet.Deployment.Export;

/// <summary>
/// Base interface for model exporters that convert AiDotNet models to various deployment formats.
/// </summary>
/// <typeparam name="T">The numeric type used in the model (e.g., float, double)</typeparam>
public interface IModelExporter<T> where T : struct
{
    /// <summary>
    /// Gets the target export format (e.g., "ONNX", "TensorFlowLite", "CoreML", "TensorRT")
    /// </summary>
    string ExportFormat { get; }

    /// <summary>
    /// Gets the file extension for the exported model (e.g., ".onnx", ".tflite", ".mlmodel")
    /// </summary>
    string FileExtension { get; }

    /// <summary>
    /// Exports the model to the specified path with the given configuration.
    /// </summary>
    /// <param name="model">The model to export</param>
    /// <param name="outputPath">The output file path</param>
    /// <param name="config">Export configuration options</param>
    void Export(object model, string outputPath, ExportConfiguration config);

    /// <summary>
    /// Exports the model to a byte array with the given configuration.
    /// </summary>
    /// <param name="model">The model to export</param>
    /// <param name="config">Export configuration options</param>
    /// <returns>The exported model as a byte array</returns>
    byte[] ExportToBytes(object model, ExportConfiguration config);

    /// <summary>
    /// Validates that the model can be exported to this format.
    /// </summary>
    /// <param name="model">The model to validate</param>
    /// <returns>True if the model can be exported; otherwise, false</returns>
    bool CanExport(object model);

    /// <summary>
    /// Gets validation errors if the model cannot be exported.
    /// </summary>
    /// <param name="model">The model to validate</param>
    /// <returns>List of validation error messages, or empty if valid</returns>
    IReadOnlyList<string> GetValidationErrors(object model);
}
