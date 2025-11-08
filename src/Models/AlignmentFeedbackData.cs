namespace AiDotNet.Models;

/// <summary>
/// Contains human feedback data for AI alignment.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class AlignmentFeedbackData<T>
{
    /// <summary>
    /// Gets or sets the input prompts or examples.
    /// </summary>
    public T[][] Inputs { get; set; } = Array.Empty<T[]>();

    /// <summary>
    /// Gets or sets the model outputs for each input.
    /// </summary>
    public T[][] Outputs { get; set; } = Array.Empty<T[]>();

    /// <summary>
    /// Gets or sets human preference comparisons.
    /// </summary>
    /// <remarks>
    /// Each element is a pair of indices into Outputs, with the first being preferred.
    /// </remarks>
    public (int preferred, int notPreferred)[] Preferences { get; set; } = Array.Empty<(int, int)>();

    /// <summary>
    /// Gets or sets numerical ratings for each output (optional).
    /// </summary>
    public double[] Ratings { get; set; } = Array.Empty<double>();

    /// <summary>
    /// Gets or sets textual feedback for outputs (optional).
    /// </summary>
    public string[] TextualFeedback { get; set; } = Array.Empty<string>();

    /// <summary>
    /// Gets or sets reward labels for reinforcement learning.
    /// </summary>
    public double[] Rewards { get; set; } = Array.Empty<double>();
}
