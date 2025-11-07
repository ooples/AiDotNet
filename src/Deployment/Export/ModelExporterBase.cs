using AiDotNet.Interfaces;

namespace AiDotNet.Deployment.Export;

/// <summary>
/// Abstract base class for model exporters that provides common functionality.
/// </summary>
/// <typeparam name="T">The numeric type used in the model</typeparam>
public abstract class ModelExporterBase<T> : IModelExporter<T> where T : struct
{
    /// <inheritdoc/>
    public abstract string ExportFormat { get; }

    /// <inheritdoc/>
    public abstract string FileExtension { get; }

    /// <inheritdoc/>
    public virtual void Export(object model, string outputPath, ExportConfiguration config)
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

        // Ensure directory exists
        var directory = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

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
    public abstract byte[] ExportToBytes(object model, ExportConfiguration config);

    /// <inheritdoc/>
    public virtual bool CanExport(object model)
    {
        return GetValidationErrors(model).Count == 0;
    }

    /// <inheritdoc/>
    public virtual IReadOnlyList<string> GetValidationErrors(object model)
    {
        var errors = new List<string>();

        if (model == null)
        {
            errors.Add("Model is null");
            return errors;
        }

        // Check if model implements IModelSerializer
        if (model is not IModelSerializer)
        {
            errors.Add($"Model does not implement IModelSerializer interface");
        }

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
    protected int[] GetInputShape(object model, ExportConfiguration config)
    {
        if (config.InputShape != null && config.InputShape.Length > 0)
        {
            return config.InputShape;
        }

        // Try to infer from model
        if (model is IParameterizable<T> paramModel)
        {
            var parameters = paramModel.GetParameters();
            if (parameters != null && parameters.Length > 0)
            {
                // Assume first parameter represents input shape
                return new[] { parameters.Length };
            }
        }

        throw new InvalidOperationException(
            "Could not determine input shape. Please specify InputShape in ExportConfiguration.");
    }
}
