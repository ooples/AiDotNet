using System.Collections.Generic;
using AiDotNet.Interfaces;

namespace AiDotNet.Deployment.Export;

/// <summary>
/// Base interface for model exporters that convert AiDotNet models to various deployment formats.
/// Properly integrates with IFullModel architecture.
/// </summary>
/// <typeparam name="T">The numeric type used in the model (e.g., float, double)</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
[AiDotNet.Configuration.YamlConfigurable("ModelExporter")]
public interface IModelExporter<T, TInput, TOutput>
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
    void Export(IFullModel<T, TInput, TOutput> model, string outputPath, ExportConfiguration config);

    /// <summary>
    /// Exports the model to a byte array with the given configuration.
    /// </summary>
    /// <param name="model">The model to export</param>
    /// <param name="config">Export configuration options</param>
    /// <returns>The exported model as a byte array</returns>
    byte[] ExportToBytes(IFullModel<T, TInput, TOutput> model, ExportConfiguration config);

    /// <summary>
    /// Validates that the model can be exported to this format.
    /// </summary>
    /// <param name="model">The model to validate</param>
    /// <returns>True if the model can be exported; otherwise, false</returns>
    bool CanExport(IFullModel<T, TInput, TOutput> model);

    /// <summary>
    /// Gets validation errors if the model cannot be exported.
    /// </summary>
    /// <param name="model">The model to validate</param>
    /// <returns>List of validation error messages, or empty if valid</returns>
    IReadOnlyList<string> GetValidationErrors(IFullModel<T, TInput, TOutput> model);
}
