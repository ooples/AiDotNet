using System.Text;

namespace AiDotNet.Models;

/// <summary>
/// Contains the results of applying agent-recommended hyperparameters to a model's options.
/// </summary>
/// <remarks>
/// <para>
/// This class tracks which hyperparameters were successfully applied, which were skipped
/// (no matching property found), which failed (type conversion or other errors), and any
/// warnings generated during the process (e.g., out-of-range values).
/// </para>
/// <para><b>For Beginners:</b> When the AI agent recommends hyperparameters, this class tells you
/// exactly what happened when those recommendations were applied to your model:
/// - **Applied**: Parameters that were successfully set on the model
/// - **Skipped**: Parameters the agent recommended but the model doesn't support
/// - **Failed**: Parameters that couldn't be set due to errors (e.g., wrong data type)
/// - **Warnings**: Issues that didn't prevent application but may need attention (e.g., values outside typical ranges)
/// </para>
/// </remarks>
public class HyperparameterApplicationResult
{
    /// <summary>
    /// Gets the dictionary of successfully applied hyperparameters (parameter name -> value set).
    /// </summary>
    public Dictionary<string, object> Applied { get; } = new();

    /// <summary>
    /// Gets the dictionary of skipped hyperparameters (parameter name -> value, no matching property found).
    /// </summary>
    public Dictionary<string, object> Skipped { get; } = new();

    /// <summary>
    /// Gets the dictionary of failed hyperparameters (parameter name -> error message).
    /// </summary>
    public Dictionary<string, string> Failed { get; } = new();

    /// <summary>
    /// Gets the list of warning messages generated during hyperparameter application.
    /// </summary>
    public List<string> Warnings { get; } = new();

    /// <summary>
    /// Gets a value indicating whether any parameters were successfully applied.
    /// </summary>
    public bool HasAppliedParameters => Applied.Count > 0;

    /// <summary>
    /// Gets a human-readable summary of the hyperparameter application results.
    /// </summary>
    /// <returns>A formatted string summarizing applied, skipped, and failed parameters.</returns>
    public string GetSummary()
    {
        var sb = new StringBuilder();
        sb.AppendLine("=== Hyperparameter Application Results ===");

        if (Applied.Count > 0)
        {
            sb.AppendLine($"\nApplied ({Applied.Count}):");
            foreach (var kvp in Applied)
            {
                sb.AppendLine($"  {kvp.Key} = {kvp.Value}");
            }
        }

        if (Skipped.Count > 0)
        {
            sb.AppendLine($"\nSkipped ({Skipped.Count}):");
            foreach (var kvp in Skipped)
            {
                sb.AppendLine($"  {kvp.Key} = {kvp.Value} (no matching property)");
            }
        }

        if (Failed.Count > 0)
        {
            sb.AppendLine($"\nFailed ({Failed.Count}):");
            foreach (var kvp in Failed)
            {
                sb.AppendLine($"  {kvp.Key}: {kvp.Value}");
            }
        }

        if (Warnings.Count > 0)
        {
            sb.AppendLine($"\nWarnings ({Warnings.Count}):");
            foreach (var warning in Warnings)
            {
                sb.AppendLine($"  {warning}");
            }
        }

        if (Applied.Count == 0 && Skipped.Count == 0 && Failed.Count == 0)
        {
            sb.AppendLine("\nNo hyperparameters were processed.");
        }

        sb.AppendLine("==========================================");
        return sb.ToString();
    }
}
