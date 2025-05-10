using AiDotNet.Serialization;

/// <summary>
/// Extension methods to help with debugging PredictionModelResult serialization
/// </summary>
public static class PredictionModelResultExtensions
{
    /// <summary>
    /// Serializes a model to JSON for debugging purposes
    /// </summary>
    public static string SerializeToJsonString<T, TInput, TOutput>(
        this PredictionModelResult<T, TInput, TOutput> model)
    {
        try
        {
            var settings = new JsonSerializerSettings
            {
                TypeNameHandling = TypeNameHandling.All,
                Formatting = Formatting.Indented,
                Converters =
                [
                    new MatrixJsonConverter<T>(),
                    new VectorJsonConverter<T>(),
                    new InterfaceJsonConverter()
                ]
            };

            return JsonConvert.SerializeObject(model, settings);
        }
        catch (Exception ex)
        {
            return $"Error serializing to JSON: {ex.Message}";
        }
    }

    /// <summary>
    /// Saves diagnostics for model debugging
    /// </summary>
    public static void SaveDiagnostics<T, TInput, TOutput>(
        this PredictionModelResult<T, TInput, TOutput> model,
        string filePath)
    {
        try
        {
            var diagnostics = new StringBuilder();

            // Get basic info about the model
            diagnostics.AppendLine($"Model Diagnostics - {DateTime.Now}");
            diagnostics.AppendLine("---------------------------------------------");

            diagnostics.AppendLine($"Model is null: {model.Model == null}");
            if (model.Model != null)
            {
                diagnostics.AppendLine($"Model Type: {model.Model.GetType().FullName}");
                diagnostics.AppendLine($"Model Assembly: {model.Model.GetType().Assembly.GetName().Name}");

                // Get model metadata
                try
                {
                    var metadata = model.Model.GetModelMetaData();
                    diagnostics.AppendLine($"Model Metadata Type: {metadata.ModelType}");
                }
                catch (Exception ex)
                {
                    diagnostics.AppendLine($"Error getting model metadata: {ex.Message}");
                }
            }

            diagnostics.AppendLine();
            diagnostics.AppendLine("Normalization Info:");
            diagnostics.AppendLine($"Normalizer is null: {model.NormalizationInfo.Normalizer == null}");

            diagnostics.AppendLine();
            diagnostics.AppendLine("Optimization Result:");
            diagnostics.AppendLine($"Best Solution is null: {model.OptimizationResult.BestSolution == null}");

            // Add serialized JSON
            diagnostics.AppendLine();
            diagnostics.AppendLine("Serialized JSON:");
            diagnostics.AppendLine("---------------------------------------------");
            diagnostics.AppendLine(model.SerializeToJsonString());

            // Save to file
            File.WriteAllText(filePath, diagnostics.ToString());
        }
        catch (Exception ex)
        {
            File.WriteAllText(filePath, $"Error saving diagnostics: {ex.Message}\n{ex.StackTrace}");
        }
    }
}