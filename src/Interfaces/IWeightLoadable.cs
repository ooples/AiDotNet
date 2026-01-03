namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for models that support loading weights by name.
/// </summary>
/// <typeparam name="T">The numeric type used for computations.</typeparam>
/// <remarks>
/// <para>
/// This interface enables loading pretrained weights from external sources like
/// SafeTensors, HuggingFace, and ONNX files into AiDotNet models.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of this as a way to "transplant" knowledge from
/// pretrained models. Each weight has a name (like "encoder.conv1.weight") and
/// this interface lets us set those weights by their names.
///
/// Example:
/// ```csharp
/// // Load pretrained weights
/// var weights = safeTensorsLoader.Load("model.safetensors");
///
/// // Apply to model
/// if (model is IWeightLoadable&lt;float&gt; loadable)
/// {
///     loadable.SetParameter("encoder.conv1.weight", weights["encoder.conv1.weight"]);
/// }
/// ```
/// </para>
/// </remarks>
public interface IWeightLoadable<T>
{
    /// <summary>
    /// Gets all parameter names in this model.
    /// </summary>
    /// <returns>A collection of all parameter names.</returns>
    /// <remarks>
    /// <para>
    /// Parameter names follow a hierarchical convention like:
    /// - "encoder.down0.res0.conv1.weight"
    /// - "encoder.down0.res0.conv1.bias"
    /// - "decoder.up3.norm.gamma"
    /// </para>
    /// </remarks>
    IEnumerable<string> GetParameterNames();

    /// <summary>
    /// Tries to get a parameter tensor by name.
    /// </summary>
    /// <param name="name">The parameter name.</param>
    /// <param name="tensor">The parameter tensor if found.</param>
    /// <returns>True if the parameter was found, false otherwise.</returns>
    bool TryGetParameter(string name, out Tensor<T>? tensor);

    /// <summary>
    /// Sets a parameter tensor by name.
    /// </summary>
    /// <param name="name">The parameter name.</param>
    /// <param name="value">The tensor value to set.</param>
    /// <returns>True if the parameter was set successfully, false if the name was not found.</returns>
    /// <exception cref="ArgumentException">Thrown when the tensor shape doesn't match expected shape.</exception>
    bool SetParameter(string name, Tensor<T> value);

    /// <summary>
    /// Gets the expected shape for a parameter.
    /// </summary>
    /// <param name="name">The parameter name.</param>
    /// <returns>The expected shape, or null if the parameter doesn't exist.</returns>
    int[]? GetParameterShape(string name);

    /// <summary>
    /// Gets the total number of named parameters.
    /// </summary>
    int NamedParameterCount { get; }

    /// <summary>
    /// Validates that a set of weight names can be loaded into this model.
    /// </summary>
    /// <param name="weightNames">Names of weights to validate.</param>
    /// <param name="mapping">Optional weight name mapping.</param>
    /// <returns>Validation result with matched and unmatched names.</returns>
    WeightLoadValidation ValidateWeights(IEnumerable<string> weightNames, Func<string, string?>? mapping = null);

    /// <summary>
    /// Loads weights from a dictionary of tensors using optional name mapping.
    /// </summary>
    /// <param name="weights">Dictionary of weight name to tensor.</param>
    /// <param name="mapping">Optional function to map source names to target names.</param>
    /// <param name="strict">If true, throws exception when any mapped weight fails to load.</param>
    /// <returns>Load result with statistics.</returns>
    WeightLoadResult LoadWeights(Dictionary<string, Tensor<T>> weights, Func<string, string?>? mapping = null, bool strict = false);
}

/// <summary>
/// Result of weight validation.
/// </summary>
public class WeightLoadValidation
{
    /// <summary>
    /// Parameter names in the model that matched weights.
    /// </summary>
    public List<string> Matched { get; set; } = new();

    /// <summary>
    /// Weight names that could not be mapped to any model parameter.
    /// </summary>
    public List<string> UnmatchedWeights { get; set; } = new();

    /// <summary>
    /// Model parameters that have no corresponding weight.
    /// </summary>
    public List<string> MissingParameters { get; set; } = new();

    /// <summary>
    /// Weights where shape doesn't match the model parameter.
    /// </summary>
    public List<(string Name, int[] Expected, int[] Actual)> ShapeMismatches { get; set; } = new();

    /// <summary>
    /// Whether all model parameters have matching weights.
    /// </summary>
    public bool IsComplete => MissingParameters.Count == 0;

    /// <summary>
    /// Whether validation passed (all matched weights have correct shapes).
    /// </summary>
    public bool IsValid => ShapeMismatches.Count == 0;

    /// <inheritdoc />
    public override string ToString()
    {
        return $"Matched: {Matched.Count}, Unmatched weights: {UnmatchedWeights.Count}, " +
               $"Missing params: {MissingParameters.Count}, Shape errors: {ShapeMismatches.Count}";
    }
}

/// <summary>
/// Result of weight loading operation.
/// </summary>
public class WeightLoadResult
{
    /// <summary>
    /// Whether loading succeeded.
    /// </summary>
    public bool Success { get; set; }

    /// <summary>
    /// Number of weights successfully loaded.
    /// </summary>
    public int LoadedCount { get; set; }

    /// <summary>
    /// Number of weights that failed to load.
    /// </summary>
    public int FailedCount { get; set; }

    /// <summary>
    /// Number of weights skipped (no mapping).
    /// </summary>
    public int SkippedCount { get; set; }

    /// <summary>
    /// Names of successfully loaded parameters.
    /// </summary>
    public List<string> LoadedParameters { get; set; } = new();

    /// <summary>
    /// Names and errors for failed parameters.
    /// </summary>
    public List<(string Name, string Error)> FailedParameters { get; set; } = new();

    /// <summary>
    /// Error message if loading failed.
    /// </summary>
    public string? ErrorMessage { get; set; }

    /// <inheritdoc />
    public override string ToString()
    {
        if (!Success && !string.IsNullOrEmpty(ErrorMessage))
        {
            return $"Load failed: {ErrorMessage}";
        }

        return $"Loaded: {LoadedCount}, Failed: {FailedCount}, Skipped: {SkippedCount}";
    }
}
