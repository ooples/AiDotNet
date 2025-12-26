namespace AiDotNet.Augmentation;

/// <summary>
/// Specifies the type of hyperparameter.
/// </summary>
public enum HyperparameterType
{
    /// <summary>
    /// A continuous floating-point parameter.
    /// </summary>
    Continuous,

    /// <summary>
    /// A discrete integer parameter.
    /// </summary>
    Integer,

    /// <summary>
    /// A categorical parameter with fixed choices.
    /// </summary>
    Categorical,

    /// <summary>
    /// A boolean parameter.
    /// </summary>
    Boolean,

    /// <summary>
    /// A log-scale continuous parameter.
    /// </summary>
    LogScale
}

/// <summary>
/// Represents a hyperparameter definition for AutoML search.
/// </summary>
public class HyperparameterDefinition
{
    /// <summary>
    /// Gets or sets the parameter name.
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the parameter type.
    /// </summary>
    public HyperparameterType Type { get; set; }

    /// <summary>
    /// Gets or sets the minimum value (for continuous/integer/log-scale).
    /// </summary>
    public double? MinValue { get; set; }

    /// <summary>
    /// Gets or sets the maximum value (for continuous/integer/log-scale).
    /// </summary>
    public double? MaxValue { get; set; }

    /// <summary>
    /// Gets or sets the default value.
    /// </summary>
    public object? DefaultValue { get; set; }

    /// <summary>
    /// Gets or sets the categorical choices (for categorical type).
    /// </summary>
    public IList<object>? Choices { get; set; }

    /// <summary>
    /// Gets or sets whether this parameter is required.
    /// </summary>
    public bool IsRequired { get; set; } = true;

    /// <summary>
    /// Gets or sets the description of this parameter.
    /// </summary>
    public string? Description { get; set; }

    /// <summary>
    /// Gets or sets conditional dependencies (e.g., only active if another param has specific value).
    /// </summary>
    public IDictionary<string, object>? Conditions { get; set; }
}

/// <summary>
/// Represents the search space for augmentation hyperparameters.
/// </summary>
public class AugmentationSearchSpace
{
    /// <summary>
    /// Gets or sets the augmentation type name.
    /// </summary>
    public string AugmentationType { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets whether to include this augmentation in the search.
    /// </summary>
    public bool IncludeInSearch { get; set; } = true;

    /// <summary>
    /// Gets or sets the probability search range.
    /// </summary>
    public (double min, double max) ProbabilityRange { get; set; } = (0.0, 1.0);

    /// <summary>
    /// Gets or sets the hyperparameter definitions.
    /// </summary>
    public IList<HyperparameterDefinition> Hyperparameters { get; set; } = new List<HyperparameterDefinition>();

    /// <summary>
    /// Gets or sets incompatible augmentations (cannot be used together).
    /// </summary>
    public IList<string>? IncompatibleWith { get; set; }

    /// <summary>
    /// Gets or sets required augmentations (must be used together).
    /// </summary>
    public IList<string>? RequiredWith { get; set; }
}

/// <summary>
/// Represents a complete search space for augmentation policies.
/// </summary>
public class PolicySearchSpace
{
    /// <summary>
    /// Gets or sets the policy name.
    /// </summary>
    public string PolicyName { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the augmentation search spaces.
    /// </summary>
    public IList<AugmentationSearchSpace> AugmentationSpaces { get; set; } = new List<AugmentationSearchSpace>();

    /// <summary>
    /// Gets or sets the minimum number of augmentations to include.
    /// </summary>
    public int MinAugmentations { get; set; } = 1;

    /// <summary>
    /// Gets or sets the maximum number of augmentations to include.
    /// </summary>
    public int MaxAugmentations { get; set; } = 10;

    /// <summary>
    /// Gets or sets global constraints on the policy.
    /// </summary>
    public IDictionary<string, object>? GlobalConstraints { get; set; }
}

/// <summary>
/// Represents a sampled configuration from the search space.
/// </summary>
public class SampledConfiguration
{
    /// <summary>
    /// Gets or sets the augmentation type.
    /// </summary>
    public string AugmentationType { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets whether this augmentation is enabled.
    /// </summary>
    public bool IsEnabled { get; set; }

    /// <summary>
    /// Gets or sets the sampled probability.
    /// </summary>
    public double Probability { get; set; }

    /// <summary>
    /// Gets or sets the sampled hyperparameter values.
    /// </summary>
    public IDictionary<string, object> Parameters { get; set; } = new Dictionary<string, object>();
}

/// <summary>
/// Interface for augmentations that expose their hyperparameter search space.
/// </summary>
/// <remarks>
/// <para>
/// This interface enables AutoML systems to automatically tune augmentation
/// hyperparameters during neural architecture search or hyperparameter optimization.
/// </para>
/// <para><b>For Beginners:</b> AutoML systems can automatically find the best
/// augmentation settings (like rotation angle ranges or color adjustment strength)
/// by searching through possible configurations and measuring their effect on
/// model performance.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <typeparam name="TData">The data type being augmented.</typeparam>
public interface IAutoMLAugmentation<T, TData> : IAugmentation<T, TData>
{
    /// <summary>
    /// Gets the hyperparameter search space for this augmentation.
    /// </summary>
    /// <returns>The search space definition.</returns>
    AugmentationSearchSpace GetSearchSpace();

    /// <summary>
    /// Creates a new instance with the specified hyperparameters.
    /// </summary>
    /// <param name="parameters">The hyperparameter values.</param>
    /// <returns>A new augmentation instance with the specified configuration.</returns>
    IAugmentation<T, TData> CreateWithParameters(IDictionary<string, object> parameters);

    /// <summary>
    /// Validates hyperparameter values.
    /// </summary>
    /// <param name="parameters">The parameters to validate.</param>
    /// <returns>True if parameters are valid.</returns>
    bool ValidateParameters(IDictionary<string, object> parameters);
}

/// <summary>
/// Interface for policies that expose their complete search space.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <typeparam name="TData">The data type being augmented.</typeparam>
public interface IAutoMLPolicy<T, TData> : IAugmentationPolicy<T, TData>
{
    /// <summary>
    /// Gets the complete policy search space.
    /// </summary>
    /// <returns>The policy search space.</returns>
    PolicySearchSpace GetPolicySearchSpace();

    /// <summary>
    /// Creates a new policy from sampled configurations.
    /// </summary>
    /// <param name="configurations">The sampled configurations.</param>
    /// <returns>A new policy with the specified augmentations.</returns>
    IAugmentationPolicy<T, TData> CreateFromConfigurations(IList<SampledConfiguration> configurations);

    /// <summary>
    /// Samples a random configuration from the search space.
    /// </summary>
    /// <param name="random">The random number generator.</param>
    /// <returns>A list of sampled configurations.</returns>
    IList<SampledConfiguration> SampleConfiguration(Random random);
}

/// <summary>
/// Interface for AutoML search algorithms over augmentation spaces.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <typeparam name="TData">The data type being augmented.</typeparam>
public interface IAugmentationSearcher<T, TData>
{
    /// <summary>
    /// Gets the search space being explored.
    /// </summary>
    PolicySearchSpace SearchSpace { get; }

    /// <summary>
    /// Suggests the next configuration to evaluate.
    /// </summary>
    /// <returns>The next configuration to try.</returns>
    IList<SampledConfiguration> SuggestNext();

    /// <summary>
    /// Reports the result of evaluating a configuration.
    /// </summary>
    /// <param name="configurations">The evaluated configurations.</param>
    /// <param name="score">The evaluation score (higher is better).</param>
    void ReportResult(IList<SampledConfiguration> configurations, double score);

    /// <summary>
    /// Gets the best configuration found so far.
    /// </summary>
    /// <returns>The best configurations and their score.</returns>
    (IList<SampledConfiguration> configurations, double score) GetBest();

    /// <summary>
    /// Gets the number of evaluations performed.
    /// </summary>
    int EvaluationCount { get; }
}

/// <summary>
/// Factory for creating augmentations from configuration.
/// </summary>
/// <remarks>
/// Enables dynamic augmentation creation from serialized configurations.
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <typeparam name="TData">The data type being augmented.</typeparam>
public interface IAugmentationFactory<T, TData>
{
    /// <summary>
    /// Gets all registered augmentation types.
    /// </summary>
    IReadOnlyList<string> RegisteredTypes { get; }

    /// <summary>
    /// Creates an augmentation from type name and parameters.
    /// </summary>
    /// <param name="typeName">The augmentation type name.</param>
    /// <param name="parameters">The parameters for the augmentation.</param>
    /// <returns>The created augmentation.</returns>
    IAugmentation<T, TData> Create(string typeName, IDictionary<string, object> parameters);

    /// <summary>
    /// Gets the search space for an augmentation type.
    /// </summary>
    /// <param name="typeName">The augmentation type name.</param>
    /// <returns>The search space definition.</returns>
    AugmentationSearchSpace GetSearchSpace(string typeName);

    /// <summary>
    /// Registers a custom augmentation type.
    /// </summary>
    /// <param name="typeName">The type name.</param>
    /// <param name="factory">The factory function.</param>
    void Register(string typeName, Func<IDictionary<string, object>, IAugmentation<T, TData>> factory);
}
