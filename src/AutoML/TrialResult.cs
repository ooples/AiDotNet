using AiDotNet.Enums;

namespace AiDotNet.AutoML;

/// <summary>
/// Represents the result of a single trial during AutoML search.
/// </summary>
/// <remarks>
/// <para>
/// This type is used by internal AutoML implementations to track trial execution outcomes. When accessed via public
/// APIs (for example, through <see cref="AiDotNet.Interfaces.IAutoMLModel{T,TInput,TOutput}.GetTrialHistory"/>),
/// sensitive fields like raw hyperparameter values must be redacted to align with the AiDotNet facade/IP goals.
/// </para>
/// <para><b>For Beginners:</b> AutoML runs many small experiments called "trials". Each trial:
/// <list type="bullet">
/// <item><description>Tries one model family with one set of settings.</description></item>
/// <item><description>Trains the model and scores it.</description></item>
/// <item><description>Records whether it succeeded and how well it did.</description></item>
/// </list>
/// This class stores the outcome of one of those experiments.
/// </para>
/// </remarks>
public sealed class TrialResult
{
    /// <summary>
    /// Gets or sets the unique identifier for the trial.
    /// </summary>
    public int TrialId { get; set; }

    /// <summary>
    /// Gets or sets the candidate model family used for the trial, when known.
    /// </summary>
    /// <remarks>
    /// This value is safe to expose in redacted summaries and helps users understand which high-level model family was tried
    /// without exposing proprietary hyperparameter values or internal configuration details.
    /// </remarks>
    public ModelType? CandidateModelType { get; set; }

    /// <summary>
    /// Gets or sets the hyperparameters used in this trial.
    /// </summary>
    /// <remarks>
    /// This dictionary is considered sensitive for IP protection. Public APIs should return a redacted copy that does not expose
    /// raw values (and may omit keys entirely), while internal AutoML implementations may keep the full dictionary for search logic.
    /// </remarks>
    public Dictionary<string, object> Parameters { get; set; } = new(StringComparer.Ordinal);

    /// <summary>
    /// Gets or sets the score achieved by this trial.
    /// </summary>
    public double Score { get; set; }

    /// <summary>
    /// Gets or sets the duration of the trial.
    /// </summary>
    public TimeSpan Duration { get; set; }

    /// <summary>
    /// Gets or sets the UTC timestamp when the trial was completed.
    /// </summary>
    public DateTime Timestamp { get; set; }

    /// <summary>
    /// Gets or sets additional metadata about the trial.
    /// </summary>
    /// <remarks>
    /// Treat this as sensitive unless explicitly documented otherwise. Public APIs should avoid returning raw metadata by default.
    /// </remarks>
    public Dictionary<string, object>? Metadata { get; set; }

    /// <summary>
    /// Gets or sets a value indicating whether the trial completed successfully.
    /// </summary>
    public bool Success { get; set; } = true;

    /// <summary>
    /// Gets or sets an error message if the trial failed.
    /// </summary>
    public string? ErrorMessage { get; set; }

    /// <summary>
    /// Creates a deep copy of this trial result including sensitive parameters.
    /// </summary>
    public TrialResult Clone()
    {
        return new TrialResult
        {
            TrialId = TrialId,
            CandidateModelType = CandidateModelType,
            Parameters = DeepCopyDictionary(Parameters),
            Score = Score,
            Duration = Duration,
            Timestamp = Timestamp,
            Metadata = Metadata != null ? DeepCopyDictionary(Metadata) : null,
            Success = Success,
            ErrorMessage = ErrorMessage
        };
    }

    private static Dictionary<string, object> DeepCopyDictionary(Dictionary<string, object> source)
    {
        var copy = new Dictionary<string, object>(source.Count, StringComparer.Ordinal);
        foreach (var (key, value) in source)
        {
            copy[key] = DeepCopyValue(value);
        }

        return copy;
    }

    private static object DeepCopyValue(object value)
    {
        if (value is null)
        {
            return null!;
        }

        if (value is Dictionary<string, object> dictionary)
        {
            return DeepCopyDictionary(dictionary);
        }

        if (value is Array array)
        {
            return DeepCopyArray(array);
        }

        if (value is ICloneable cloneable)
        {
            return cloneable.Clone() ?? value;
        }

        // TODO: Consider deep-copying common collection types (e.g., List<T>/IList) when used in trial metadata.
        return value;
    }

    private static Array DeepCopyArray(Array source)
    {
        var elementType = source.GetType().GetElementType() ?? typeof(object);
        var lengths = new int[source.Rank];
        for (int i = 0; i < source.Rank; i++)
        {
            lengths[i] = source.GetLength(i);
        }

        var copy = Array.CreateInstance(elementType, lengths);
        DeepCopyArrayRecursive(source, copy, dimension: 0, new int[source.Rank]);
        return copy;
    }

    private static void DeepCopyArrayRecursive(Array source, Array destination, int dimension, int[] indices)
    {
        if (dimension == source.Rank)
        {
            var element = source.GetValue(indices);
            destination.SetValue(element is null ? null : DeepCopyValue(element), indices);
            return;
        }

        int length = source.GetLength(dimension);
        for (int i = 0; i < length; i++)
        {
            indices[dimension] = i;
            DeepCopyArrayRecursive(source, destination, dimension + 1, indices);
        }
    }

    /// <summary>
    /// Creates a deep copy of this trial result with sensitive fields redacted.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method exists to support the AiDotNet facade/IP policy: users should be able to understand what happened during
    /// AutoML without being able to reconstruct proprietary models from raw hyperparameter values.
    /// </para>
    /// </remarks>
    public TrialResult CloneRedacted()
    {
        return new TrialResult
        {
            TrialId = TrialId,
            CandidateModelType = CandidateModelType,
            Parameters = new Dictionary<string, object>(StringComparer.Ordinal),
            Score = Score,
            Duration = Duration,
            Timestamp = Timestamp,
            Metadata = null,
            Success = Success,
            ErrorMessage = ErrorMessage
        };
    }
}
