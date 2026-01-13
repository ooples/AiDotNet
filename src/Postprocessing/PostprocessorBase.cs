using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Postprocessing;

/// <summary>
/// Abstract base class for all postprocessors providing common functionality.
/// </summary>
/// <remarks>
/// <para>
/// This class provides the template method pattern for postprocessing.
/// Derived classes implement the core processing logic while this base class
/// handles validation, configuration, and common operations.
/// </para>
/// <para><b>For Beginners:</b> This is the foundation that all postprocessors build on.
/// It provides common features like:
/// - Configuration management
/// - Batch processing support
/// - Error handling
///
/// When creating a new postprocessor, you extend this class and implement the abstract methods.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (model output).</typeparam>
/// <typeparam name="TOutput">The output data type after postprocessing.</typeparam>
public abstract class PostprocessorBase<T, TInput, TOutput> : IPostprocessor<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the numeric operations helper for type T.
    /// </summary>
    protected INumericOperations<T> NumOps { get; }

    /// <summary>
    /// Gets whether this postprocessor is configured and ready to use.
    /// </summary>
    public bool IsConfigured { get; protected set; }

    /// <summary>
    /// Gets whether this postprocessor supports inverse transformation.
    /// </summary>
    public abstract bool SupportsInverse { get; }

    /// <summary>
    /// Gets the configuration settings for this postprocessor.
    /// </summary>
    protected Dictionary<string, object> Settings { get; } = new();

    /// <summary>
    /// Creates a new instance of the postprocessor.
    /// </summary>
    protected PostprocessorBase()
    {
        NumOps = MathHelper.GetNumericOperations<T>();
        // Most postprocessors are stateless and don't require configuration
        IsConfigured = true;
    }

    /// <summary>
    /// Configures the postprocessor with optional settings.
    /// </summary>
    /// <param name="settings">Optional configuration dictionary.</param>
    public virtual void Configure(Dictionary<string, object>? settings = null)
    {
        Settings.Clear();

        if (settings != null)
        {
            foreach (var kvp in settings)
            {
                Settings[kvp.Key] = kvp.Value;
            }
        }

        ConfigureCore(settings);
        IsConfigured = true;
    }

    /// <summary>
    /// Transforms model output into the final result format.
    /// </summary>
    /// <param name="input">The model output to process.</param>
    /// <returns>The postprocessed result.</returns>
    /// <exception cref="InvalidOperationException">Thrown if not configured.</exception>
    public TOutput Process(TInput input)
    {
        EnsureConfigured();
        ValidateInput(input);
        return ProcessCore(input);
    }

    /// <summary>
    /// Transforms a batch of model outputs.
    /// </summary>
    /// <param name="inputs">The model outputs to process.</param>
    /// <returns>The postprocessed results.</returns>
    public virtual IList<TOutput> ProcessBatch(IEnumerable<TInput> inputs)
    {
        EnsureConfigured();
        return inputs.Select(input =>
        {
            ValidateInput(input);
            return ProcessCore(input);
        }).ToList();
    }

    /// <summary>
    /// Reverses the postprocessing (if supported).
    /// </summary>
    /// <param name="output">The postprocessed result.</param>
    /// <returns>The original model output format.</returns>
    /// <exception cref="NotSupportedException">Thrown if inverse is not supported.</exception>
    public TInput Inverse(TOutput output)
    {
        if (!SupportsInverse)
        {
            throw new NotSupportedException(
                $"{GetType().Name} does not support inverse transformation.");
        }

        EnsureConfigured();
        return InverseCore(output);
    }

    /// <summary>
    /// Core configuration implementation. Override to handle specific settings.
    /// </summary>
    /// <param name="settings">The configuration settings.</param>
    protected virtual void ConfigureCore(Dictionary<string, object>? settings)
    {
        // Default implementation does nothing
    }

    /// <summary>
    /// Core processing implementation. Override this in derived classes.
    /// </summary>
    /// <param name="input">The input to process.</param>
    /// <returns>The processed output.</returns>
    protected abstract TOutput ProcessCore(TInput input);

    /// <summary>
    /// Core inverse transformation implementation. Override this in derived classes.
    /// </summary>
    /// <param name="output">The output to invert.</param>
    /// <returns>The inverted input.</returns>
    protected virtual TInput InverseCore(TOutput output)
    {
        throw new NotSupportedException(
            $"{GetType().Name} does not support inverse transformation.");
    }

    /// <summary>
    /// Validates input before processing.
    /// </summary>
    /// <param name="input">The input to validate.</param>
    /// <exception cref="ArgumentNullException">Thrown if input is null.</exception>
    protected virtual void ValidateInput(TInput input)
    {
        if (input is null)
        {
            throw new ArgumentNullException(nameof(input), "Input cannot be null.");
        }
    }

    /// <summary>
    /// Ensures the postprocessor is configured before use.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown if not configured.</exception>
    protected void EnsureConfigured()
    {
        if (!IsConfigured)
        {
            throw new InvalidOperationException(
                $"{GetType().Name} has not been configured. Call Configure() first.");
        }
    }

    /// <summary>
    /// Gets a setting value with type conversion.
    /// </summary>
    /// <typeparam name="TSetting">The expected setting type.</typeparam>
    /// <param name="key">The setting key.</param>
    /// <param name="defaultValue">Default value if not found.</param>
    /// <returns>The setting value or default.</returns>
    protected TSetting GetSetting<TSetting>(string key, TSetting defaultValue)
    {
        if (Settings.TryGetValue(key, out var value) && value is TSetting typedValue)
        {
            return typedValue;
        }
        return defaultValue;
    }
}
