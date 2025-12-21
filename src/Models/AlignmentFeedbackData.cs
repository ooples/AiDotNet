namespace AiDotNet.Models;

using AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// Contains human feedback data for AI alignment.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class AlignmentFeedbackData<T>
{
    /// <summary>
    /// Gets or sets the input prompts or examples.
    /// </summary>
    public Matrix<T> Inputs { get; set; } = Matrix<T>.Empty();

    /// <summary>
    /// Gets or sets the model outputs for each input.
    /// </summary>
    public Matrix<T> Outputs { get; set; } = Matrix<T>.Empty();

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

    /// <summary>
    /// Validates that preference indices are within valid bounds.
    /// </summary>
    /// <returns>True if all preference indices are valid, false otherwise.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method checks that all the preference pairs
    /// reference valid output indices. Since preferences are pairs of indices pointing
    /// to specific outputs, they must all be within the range of available outputs.</para>
    /// </remarks>
    public bool ValidatePreferences()
    {
        if (Preferences == null || Preferences.Length == 0)
        {
            return true; // No preferences to validate
        }

        int maxIndex = Outputs.Rows - 1;
        if (maxIndex < 0)
        {
            return Preferences.Length == 0; // No outputs means no valid preferences
        }

        foreach (var (preferred, notPreferred) in Preferences)
        {
            if (preferred < 0 || preferred > maxIndex)
            {
                return false;
            }
            if (notPreferred < 0 || notPreferred > maxIndex)
            {
                return false;
            }
            if (preferred == notPreferred)
            {
                return false; // Can't prefer something over itself
            }
        }

        return true;
    }

    /// <summary>
    /// Validates preference indices and throws if invalid.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown when preference indices are out of bounds.</exception>
    public void EnsurePreferencesValid()
    {
        if (Preferences == null || Preferences.Length == 0)
        {
            return;
        }

        int maxIndex = Outputs.Rows - 1;
        if (maxIndex < 0 && Preferences.Length > 0)
        {
            throw new InvalidOperationException(
                "Preferences exist but Outputs is empty. Cannot have preferences without outputs.");
        }

        for (int i = 0; i < Preferences.Length; i++)
        {
            var (preferred, notPreferred) = Preferences[i];

            if (preferred < 0 || preferred > maxIndex)
            {
                throw new InvalidOperationException(
                    $"Preference[{i}].preferred index {preferred} is out of bounds [0, {maxIndex}].");
            }
            if (notPreferred < 0 || notPreferred > maxIndex)
            {
                throw new InvalidOperationException(
                    $"Preference[{i}].notPreferred index {notPreferred} is out of bounds [0, {maxIndex}].");
            }
            if (preferred == notPreferred)
            {
                throw new InvalidOperationException(
                    $"Preference[{i}] has identical preferred and notPreferred indices ({preferred}).");
            }
        }
    }
}
