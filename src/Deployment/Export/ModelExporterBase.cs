using AiDotNet.Interfaces;

namespace AiDotNet.Deployment.Export;

/// <summary>
/// Abstract base class for model exporters that provides common functionality.
/// Properly integrates with IFullModel architecture.
/// </summary>
/// <typeparam name="T">The numeric type used in the model</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> for provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public abstract class ModelExporterBase<T, TInput, TOutput> : IModelExporter<T, TInput, TOutput>
{
    /// <inheritdoc/>
    public abstract string ExportFormat { get; }

    /// <inheritdoc/>
    public abstract string FileExtension { get; }

    /// <inheritdoc/>
    public virtual void Export(IFullModel<T, TInput, TOutput> model, string outputPath, ExportConfiguration config)
    {
        if (model == null)
            throw new ArgumentNullException(nameof(model));
        if (string.IsNullOrWhiteSpace(outputPath))
            throw new ArgumentException("Output path cannot be null or empty.", nameof(outputPath));
        if (config == null)
            throw new ArgumentNullException(nameof(config));

        // Validate the model can be exported
        if (!CanExport(model))
        {
            var errors = GetValidationErrors(model);
            throw new InvalidOperationException(
                $"Model cannot be exported to {ExportFormat}. Errors: {string.Join(", ", errors)}");
        }

        // Ensure output directory exists
        // Path.GetDirectoryName can return null for root paths or relative filenames
        var directory = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrWhiteSpace(directory) && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }
        // If directory is null/empty, file will be written to current working directory

        // Perform the export
        var exportedBytes = ExportToBytes(model, config);
        File.WriteAllBytes(outputPath, exportedBytes);

        // Validate after export if requested
        if (config.ValidateAfterExport)
        {
            ValidateExportedModel(outputPath, config);
        }
    }

    /// <inheritdoc/>
    public abstract byte[] ExportToBytes(IFullModel<T, TInput, TOutput> model, ExportConfiguration config);

    /// <inheritdoc/>
    public virtual bool CanExport(IFullModel<T, TInput, TOutput> model)
    {
        return GetValidationErrors(model).Count == 0;
    }

    /// <inheritdoc/>
    public virtual IReadOnlyList<string> GetValidationErrors(IFullModel<T, TInput, TOutput> model)
    {
        var errors = new List<string>();

        if (model == null)
        {
            errors.Add("Model is null");
            return errors;
        }

        // IFullModel already extends IModelSerializer, so no need to check
        // All models using this exporter are guaranteed to be serializable

        return errors;
    }

    /// <summary>
    /// Validates the exported model file.
    /// </summary>
    /// <param name="exportedPath">Path to the exported model</param>
    /// <param name="config">Export configuration</param>
    protected virtual void ValidateExportedModel(string exportedPath, ExportConfiguration config)
    {
        if (!File.Exists(exportedPath))
        {
            throw new FileNotFoundException($"Exported model file not found: {exportedPath}");
        }

        var fileInfo = new FileInfo(exportedPath);
        if (fileInfo.Length == 0)
        {
            throw new InvalidOperationException($"Exported model file is empty: {exportedPath}");
        }
    }

    /// <summary>
    /// Gets the input shape from the model or configuration.
    /// </summary>
    /// <remarks>
    /// <para>When the model implements <see cref="ILayeredModel{T}"/>, the input shape can
    /// be automatically inferred from the first layer's input shape, eliminating the need
    /// to specify it manually in the export configuration.</para>
    /// </remarks>
    protected int[] GetInputShape(IFullModel<T, TInput, TOutput> model, ExportConfiguration config)
    {
        if (config.InputShape != null && config.InputShape.Length > 0)
        {
            return config.InputShape;
        }

        // Try to infer from ILayeredModel if available
        if (model is ILayeredModel<T> layeredModel && layeredModel.LayerCount > 0)
        {
            var firstLayerInput = layeredModel.Layers[0].GetInputShape();
            if (firstLayerInput.Length > 0)
            {
                return firstLayerInput;
            }
        }

        throw new InvalidOperationException(
            "Could not determine input shape. Please specify InputShape in ExportConfiguration.");
    }

    /// <summary>
    /// Gets the output shape from the model or configuration.
    /// </summary>
    /// <remarks>
    /// <para>When the model implements <see cref="ILayeredModel{T}"/>, the output shape can
    /// be automatically inferred from the last layer's output shape.</para>
    /// </remarks>
    protected int[] GetOutputShape(IFullModel<T, TInput, TOutput> model, ExportConfiguration config)
    {
        if (config.OutputShape != null && config.OutputShape.Length > 0)
        {
            return config.OutputShape;
        }

        // Try to infer from ILayeredModel if available
        if (model is ILayeredModel<T> layeredModel && layeredModel.LayerCount > 0)
        {
            var lastLayerOutput = layeredModel.Layers[layeredModel.LayerCount - 1].GetOutputShape();
            if (lastLayerOutput.Length > 0)
            {
                return lastLayerOutput;
            }
        }

        throw new InvalidOperationException(
            "Could not determine output shape. Please specify OutputShape in ExportConfiguration.");
    }

    /// <summary>
    /// Gets a summary of all layers for export metadata, using <see cref="ILayeredModel{T}"/>
    /// when available.
    /// </summary>
    /// <param name="model">The model to summarize.</param>
    /// <returns>A list of layer descriptions, or an empty list if layer info is unavailable.</returns>
    protected IReadOnlyList<LayerInfo<T>> GetLayerSummary(IFullModel<T, TInput, TOutput> model)
    {
        if (model is ILayeredModel<T> layeredModel)
        {
            return layeredModel.GetAllLayerInfo();
        }

        return Array.Empty<LayerInfo<T>>();
    }
}
