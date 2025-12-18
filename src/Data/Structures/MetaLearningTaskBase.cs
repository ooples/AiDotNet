using AiDotNet.Interfaces;

namespace AiDotNet.Data.Structures;

/// <summary>
/// Abstract base class for meta-learning tasks, providing common functionality and validation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// This base class implements the <see cref="IMetaLearningTask{T, TInput, TOutput}"/> interface
/// and provides common functionality for all meta-learning task implementations. It includes
/// validation, default values, and helper methods to reduce code duplication across concrete
/// implementations.
/// </para>
/// <para>
/// Concrete implementations should inherit from this class and provide any additional
/// functionality specific to their meta-learning scenario (e.g., episode tracking,
/// hierarchical relationships, continual learning).
/// </para>
/// </remarks>
public abstract class MetaLearningTaskBase<T, TInput, TOutput> : IMetaLearningTask<T, TInput, TOutput>
{
    private TInput? _supportInput;
    private TOutput? _supportOutput;
    private TInput? _queryInput;
    private TOutput? _queryOutput;
    private string? _name;
    private Dictionary<string, object>? _metadata;

    /// <summary>
    /// Initializes a new instance of the MetaLearningTaskBase class.
    /// </summary>
    /// <param name="name">Optional name for the task.</param>
    /// <param name="metadata">Optional metadata dictionary.</param>
    protected MetaLearningTaskBase(string? name = null, Dictionary<string, object>? metadata = null)
    {
        _name = name;
        _metadata = metadata ?? new Dictionary<string, object>();
    }

    /// <summary>
    /// Gets or sets the input features for the support set.
    /// </summary>
    /// <value>
    /// Input data containing examples for task adaptation.
    /// </value>
    /// <exception cref="ArgumentNullException">Thrown when trying to set null value.</exception>
    public virtual TInput SupportInput
    {
        get => _supportInput;
        set
        {
            if (value == null)
                throw new ArgumentNullException(nameof(SupportInput));
            _supportInput = value;
        }
    }

    /// <summary>
    /// Gets or sets the target labels for the support set.
    /// </summary>
    /// <value>
    /// Output data containing labels corresponding to SupportInput.
    /// </value>
    /// <exception cref="ArgumentNullException">Thrown when trying to set null value.</exception>
    public virtual TOutput SupportOutput
    {
        get => _supportOutput;
        set
        {
            if (value == null)
                throw new ArgumentNullException(nameof(SupportOutput));
            _supportOutput = value;
        }
    }

    /// <summary>
    /// Gets or sets the input features for the query set.
    /// </summary>
    /// <value>
    /// Input data for evaluating adaptation performance.
    /// </value>
    /// <exception cref="ArgumentNullException">Thrown when trying to set null value.</exception>
    public virtual TInput QueryInput
    {
        get => _queryInput;
        set
        {
            if (value == null)
                throw new ArgumentNullException(nameof(QueryInput));
            _queryInput = value;
        }
    }

    /// <summary>
    /// Gets or sets the target labels for the query set.
    /// </summary>
    /// <value>
    /// True labels for evaluating query set performance.
    /// </value>
    /// <exception cref="ArgumentNullException">Thrown when trying to set null value.</exception>
    public virtual TOutput QueryOutput
    {
        get => _queryOutput;
        set
        {
            if (value == null)
                throw new ArgumentNullException(nameof(QueryOutput));
            _queryOutput = value;
        }
    }

    /// <summary>
    /// Gets or sets an optional name or identifier for the task.
    /// </summary>
    /// <value>
    /// Human-readable task name or null if not specified.
    /// </value>
    public virtual string? Name
    {
        get => _name;
        set => _name = value;
    }

    /// <summary>
    /// Gets the additional metadata about the task.
    /// </summary>
    /// <value>
    /// Dictionary containing task-specific information.
    /// </value>
    public virtual Dictionary<string, object>? Metadata => _metadata;

    /// <summary>
    /// Adds or updates a metadata entry.
    /// </summary>
    /// <param name="key">The metadata key.</param>
    /// <param name="value">The metadata value.</param>
    /// <exception cref="ArgumentNullException">Thrown when key is null.</exception>
    public void SetMetadata(string key, object value)
    {
        if (key == null)
            throw new ArgumentNullException(nameof(key));

        _metadata ??= new Dictionary<string, object>();
        _metadata[key] = value;
    }

    /// <summary>
    /// Gets a metadata value if it exists.
    /// </summary>
    /// <typeparam name="TValue">The type of value to retrieve.</typeparam>
    /// <param name="key">The metadata key.</param>
    /// <param name="value">Output for the value if found.</param>
    /// <returns>True if the value was found and successfully cast to TValue.</returns>
    public bool TryGetMetadata<TValue>(string key, out TValue? value)
    {
        if (_metadata != null && _metadata.TryGetValue(key, out var obj) && obj is TValue)
        {
            value = (TValue)obj;
            return true;
        }
        value = default;
        return false;
    }

    /// <summary>
    /// Validates that the task has all required data populated.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown when required data is missing.</exception>
    public virtual void Validate()
    {
        if (SupportInput == null)
            throw new InvalidOperationException("SupportInput is required but is null.");
        if (SupportOutput == null)
            throw new InvalidOperationException("SupportOutput is required but is null.");
        if (QueryInput == null)
            throw new InvalidOperationException("QueryInput is required but is null.");
        if (QueryOutput == null)
            throw new InvalidOperationException("QueryOutput is required but is null.");
    }

    /// <summary>
    /// Creates a string representation of the task.
    /// </summary>
    /// <returns>String containing task name and type information.</returns>
    public override string ToString()
    {
        var name = string.IsNullOrEmpty(Name) ? GetType().Name : Name;
        return $"{name} (Support: {SupportInput?.GetType().Name}, Query: {QueryInput?.GetType().Name})";
    }
}