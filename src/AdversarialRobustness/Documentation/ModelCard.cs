using System.Text;

namespace AiDotNet.AdversarialRobustness.Documentation;

/// <summary>
/// Represents a Model Card for documenting AI model characteristics and performance.
/// </summary>
/// <remarks>
/// <para>
/// Model Cards provide transparent documentation about machine learning models,
/// including their intended use, limitations, performance metrics, and ethical considerations.
/// </para>
/// <para><b>For Beginners:</b> A Model Card is like a nutrition label for AI models.
/// Just as food labels tell you what's in your food and its nutritional value, Model Cards
/// tell you what the AI model is for, how well it works, what its limitations are, and
/// any ethical considerations you should know about.</para>
/// <para>
/// Based on "Model Cards for Model Reporting" by Mitchell et al. (2019)
/// </para>
/// </remarks>
public class ModelCard
{
    public ModelCard()
    {
        Date = DateTime.UtcNow;
    }

    /// <summary>
    /// Gets or sets the model name and version.
    /// </summary>
    public string ModelName { get; set; } = "Unnamed Model";

    /// <summary>
    /// Gets or sets the model version identifier.
    /// </summary>
    public string Version { get; set; } = "1.0.0";

    /// <summary>
    /// Gets or sets the date the model was created or last updated.
    /// </summary>
    public DateTime Date { get; set; }

    /// <summary>
    /// Gets or sets the model developers or organization.
    /// </summary>
    public string Developers { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the model type (e.g., "Classification", "Regression", "LLM").
    /// </summary>
    public string ModelType { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the intended use cases for the model.
    /// </summary>
    public List<string> IntendedUses { get; set; } = new();

    /// <summary>
    /// Gets or sets the out-of-scope use cases (what the model should NOT be used for).
    /// </summary>
    public List<string> OutOfScopeUses { get; set; } = new();

    /// <summary>
    /// Gets or sets the training data description.
    /// </summary>
    public string TrainingData { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets performance metrics on different datasets.
    /// </summary>
    public Dictionary<string, Dictionary<string, double>> PerformanceMetrics { get; set; } = new();

    /// <summary>
    /// Gets or sets known limitations of the model.
    /// </summary>
    public List<string> Limitations { get; set; } = new();

    /// <summary>
    /// Gets or sets ethical considerations and potential biases.
    /// </summary>
    public List<string> EthicalConsiderations { get; set; } = new();

    /// <summary>
    /// Gets or sets fairness metrics across different demographic groups.
    /// </summary>
    public Dictionary<string, Dictionary<string, double>> FairnessMetrics { get; set; } = new();

    /// <summary>
    /// Gets or sets robustness evaluation results.
    /// </summary>
    public Dictionary<string, double> RobustnessMetrics { get; set; } = new();

    /// <summary>
    /// Gets or sets recommendations for responsible use.
    /// </summary>
    public List<string> Recommendations { get; set; } = new();

    /// <summary>
    /// Gets or sets caveats and additional warnings.
    /// </summary>
    public List<string> Caveats { get; set; } = new();

    /// <summary>
    /// Generates a formatted Model Card document.
    /// </summary>
    /// <returns>A string containing the formatted Model Card.</returns>
    public string Generate()
    {
        var sb = new StringBuilder();

        sb.AppendLine("# Model Card");
        sb.AppendLine();
        sb.AppendLine($"**Model:** {ModelName}");
        sb.AppendLine($"**Version:** {Version}");
        sb.AppendLine($"**Date:** {Date:yyyy-MM-dd}");
        sb.AppendLine($"**Developers:** {Developers}");
        sb.AppendLine($"**Model Type:** {ModelType}");
        sb.AppendLine();

        // Intended Uses
        sb.AppendLine("## Intended Uses");
        if (IntendedUses.Count > 0)
        {
            foreach (var use in IntendedUses)
            {
                sb.AppendLine($"- {use}");
            }
        }
        else
        {
            sb.AppendLine("Not specified");
        }
        sb.AppendLine();

        // Out-of-Scope Uses
        sb.AppendLine("## Out-of-Scope Uses");
        if (OutOfScopeUses.Count > 0)
        {
            foreach (var use in OutOfScopeUses)
            {
                sb.AppendLine($"- {use}");
            }
        }
        else
        {
            sb.AppendLine("Not specified");
        }
        sb.AppendLine();

        // Training Data
        sb.AppendLine("## Training Data");
        sb.AppendLine(string.IsNullOrEmpty(TrainingData) ? "Not specified" : TrainingData);
        sb.AppendLine();

        // Performance Metrics
        sb.AppendLine("## Performance Metrics");
        if (PerformanceMetrics.Count > 0)
        {
            foreach (var dataset in PerformanceMetrics)
            {
                sb.AppendLine($"### {dataset.Key}");
                foreach (var metric in dataset.Value)
                {
                    sb.AppendLine($"- **{metric.Key}:** {metric.Value:F4}");
                }
                sb.AppendLine();
            }
        }
        else
        {
            sb.AppendLine("Not specified");
            sb.AppendLine();
        }

        // Robustness Metrics
        if (RobustnessMetrics.Count > 0)
        {
            sb.AppendLine("## Robustness Metrics");
            foreach (var metric in RobustnessMetrics)
            {
                sb.AppendLine($"- **{metric.Key}:** {metric.Value:F4}");
            }
            sb.AppendLine();
        }

        // Fairness Metrics
        if (FairnessMetrics.Count > 0)
        {
            sb.AppendLine("## Fairness Metrics");
            foreach (var group in FairnessMetrics)
            {
                sb.AppendLine($"### {group.Key}");
                foreach (var metric in group.Value)
                {
                    sb.AppendLine($"- **{metric.Key}:** {metric.Value:F4}");
                }
                sb.AppendLine();
            }
        }

        // Limitations
        sb.AppendLine("## Limitations");
        if (Limitations.Count > 0)
        {
            foreach (var limitation in Limitations)
            {
                sb.AppendLine($"- {limitation}");
            }
        }
        else
        {
            sb.AppendLine("Not specified");
        }
        sb.AppendLine();

        // Ethical Considerations
        sb.AppendLine("## Ethical Considerations");
        if (EthicalConsiderations.Count > 0)
        {
            foreach (var consideration in EthicalConsiderations)
            {
                sb.AppendLine($"- {consideration}");
            }
        }
        else
        {
            sb.AppendLine("Not specified");
        }
        sb.AppendLine();

        // Recommendations
        sb.AppendLine("## Recommendations");
        if (Recommendations.Count > 0)
        {
            foreach (var recommendation in Recommendations)
            {
                sb.AppendLine($"- {recommendation}");
            }
        }
        else
        {
            sb.AppendLine("Not specified");
        }
        sb.AppendLine();

        // Caveats
        if (Caveats.Count > 0)
        {
            sb.AppendLine("## Caveats and Warnings");
            foreach (var caveat in Caveats)
            {
                sb.AppendLine($"- {caveat}");
            }
            sb.AppendLine();
        }

        return sb.ToString();
    }

    /// <summary>
    /// Saves the Model Card to a markdown file.
    /// </summary>
    /// <param name="filePath">The path where the Model Card should be saved.</param>
    /// <exception cref="ArgumentException">Thrown when file path is null or empty.</exception>
    public void SaveToFile(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));
        }

        var fullPath = Path.GetFullPath(filePath);
        var directory = Path.GetDirectoryName(fullPath);
        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

        var content = Generate();
        File.WriteAllText(fullPath, content);
    }

    /// <summary>
    /// Creates a Model Card from evaluation results.
    /// </summary>
    /// <param name="modelName">The name of the model.</param>
    /// <param name="modelType">The type of the model (e.g., "Classification", "Regression").</param>
    /// <param name="performanceMetrics">Performance metrics for the model. Can be null.</param>
    /// <param name="robustnessMetrics">Robustness metrics for the model. Can be null.</param>
    /// <returns>A new ModelCard instance populated with the provided evaluation results.</returns>
    /// <exception cref="ArgumentException">Thrown when modelName is null or empty.</exception>
    public static ModelCard CreateFromEvaluation(
        string modelName,
        string modelType,
        Dictionary<string, double>? performanceMetrics,
        Dictionary<string, double>? robustnessMetrics)
    {
        if (string.IsNullOrWhiteSpace(modelName))
        {
            throw new ArgumentException("Model name cannot be null or empty.", nameof(modelName));
        }

        var card = new ModelCard
        {
            ModelName = modelName,
            ModelType = modelType ?? string.Empty,
            Date = DateTime.UtcNow
        };

        // Add performance metrics (defensive copy to prevent external mutation)
        if (performanceMetrics != null && performanceMetrics.Count > 0)
        {
            card.PerformanceMetrics["Overall"] = new Dictionary<string, double>(performanceMetrics);
        }

        // Add robustness metrics (defensive copy to prevent external mutation)
        if (robustnessMetrics != null && robustnessMetrics.Count > 0)
        {
            card.RobustnessMetrics = new Dictionary<string, double>(robustnessMetrics);
        }

        // Add standard recommendations
        card.Recommendations.Add("Continuously monitor model performance in production");
        card.Recommendations.Add("Regularly update the model with new data");
        card.Recommendations.Add("Implement safety filters for sensitive applications");
        card.Recommendations.Add("Conduct fairness audits across demographic groups");

        return card;
    }
}
