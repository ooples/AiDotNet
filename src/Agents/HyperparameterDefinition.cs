namespace AiDotNet.Agents;

/// <summary>
/// Defines a hyperparameter mapping between LLM-recommended names and C# property names,
/// including type information and valid ranges for validation.
/// </summary>
/// <remarks>
/// <para>
/// This class bridges the gap between how an LLM refers to hyperparameters (e.g., "n_estimators",
/// "learning_rate") and the actual C# property names on options classes (e.g., "NumberOfTrees",
/// "LearningRate"). It also provides validation metadata such as expected type and valid ranges.
/// </para>
/// <para><b>For Beginners:</b> When an AI agent recommends hyperparameters, it uses common ML names
/// like "n_estimators" or "learning_rate". But the C# code uses different names like "NumberOfTrees"
/// or "LearningRate". This class maps between those two naming conventions and also knows what
/// valid values look like (e.g., learning rate should be between 0.00001 and 1.0).
/// </para>
/// </remarks>
internal class HyperparameterDefinition
{
    /// <summary>
    /// Gets or sets the C# property name on the options class.
    /// </summary>
    public string PropertyName { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the list of aliases (LLM parameter names) that map to this property.
    /// </summary>
    /// <remarks>
    /// Examples: "n_estimators", "num_trees", "ntrees" all map to "NumberOfTrees".
    /// </remarks>
    public List<string> Aliases { get; set; } = new();

    /// <summary>
    /// Gets or sets the expected value type for this hyperparameter.
    /// </summary>
    public Type ValueType { get; set; } = typeof(double);

    /// <summary>
    /// Gets or sets the minimum valid value (inclusive), or null if no minimum.
    /// </summary>
    public double? MinValue { get; set; }

    /// <summary>
    /// Gets or sets the maximum valid value (inclusive), or null if no maximum.
    /// </summary>
    public double? MaxValue { get; set; }

    private HashSet<string> _normalizedAliases = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>
    /// Checks whether the given normalized name matches any alias of this definition.
    /// </summary>
    internal bool MatchesAlias(string normalizedName)
    {
        return _normalizedAliases.Contains(normalizedName);
    }

    /// <summary>
    /// Initializes the normalized aliases from the Aliases list. Call after setting Aliases.
    /// </summary>
    public void BuildNormalizedAliases()
    {
        _normalizedAliases.Clear();

        // Always include the property name itself (normalized)
        _normalizedAliases.Add(NormalizeName(PropertyName));

        foreach (var alias in Aliases)
        {
            _normalizedAliases.Add(NormalizeName(alias));
        }
    }

    /// <summary>
    /// Normalizes a parameter name by converting to lowercase and removing separators.
    /// </summary>
    internal static string NormalizeName(string name)
    {
        return name.ToLowerInvariant()
            .Replace("_", "")
            .Replace("-", "")
            .Replace(" ", "");
    }
}
