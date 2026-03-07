using AiDotNet.Interfaces;

namespace AiDotNet.Preprocessing;

/// <summary>
/// Global registry for the preprocessing pipeline.
/// </summary>
/// <remarks>
/// <para>
/// <b>Deprecated:</b> This static registry causes race conditions when multiple
/// <c>AiModelBuilder</c> instances build models concurrently, because they overwrite
/// each other's pipeline. Use instance-based preprocessing via
/// <c>AiModelBuilder.ConfigurePreprocessing()</c> instead, which stores the pipeline
/// per-builder and flows it to <c>AiModelResult</c> via <c>PreprocessingInfo</c>.
/// </para>
/// <para>
/// PreprocessingRegistry provides a singleton pattern for managing the active preprocessing pipeline.
/// By default, a standard pipeline with imputation and scaling is used. Users can configure
/// custom preprocessing via AiModelBuilder.ConfigurePreprocessing().
/// </para>
/// <para><b>For Beginners:</b> This is like a global settings panel for data preprocessing.
/// You don't need to interact with this directly - just use AiModelBuilder:
///
/// <code>
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigurePreprocessing(pipeline => pipeline
///         .Add(new SimpleImputer&lt;double&gt;(ImputationStrategy.Mean))
///         .Add(new StandardScaler&lt;double&gt;()))
///     .ConfigureModel(new LogisticRegression&lt;double&gt;())
///     .Build(X, y);
/// </code>
///
/// The configured preprocessing is automatically applied to all models.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
[Obsolete("Use instance-based preprocessing via AiModelBuilder.ConfigurePreprocessing() instead. " +
    "This static registry causes race conditions in concurrent model building and will be removed in a future version.")]
public static class PreprocessingRegistry<T, TInput>
{
    private static IDataTransformer<T, TInput, TInput>? _current;
    private static readonly object _lock = new();

    /// <summary>
    /// Gets or sets the current preprocessing pipeline.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Deprecated:</b> This property is no longer set by <c>AiModelBuilder.ConfigurePreprocessing()</c>.
    /// The builder now stores the pipeline per-instance and flows it to <c>AiModelResult</c>
    /// via <c>PreprocessingInfo</c>. This property remains only for backward compatibility
    /// with external code that sets it directly.
    /// </para>
    /// </remarks>
    public static IDataTransformer<T, TInput, TInput>? Current
    {
        get
        {
            lock (_lock)
            {
                return _current;
            }
        }
        set
        {
            lock (_lock)
            {
                _current = value;
            }
        }
    }

    /// <summary>
    /// Gets whether a preprocessing pipeline is currently configured.
    /// </summary>
    public static bool IsConfigured
    {
        get
        {
            lock (_lock)
            {
                return _current != null;
            }
        }
    }

    /// <summary>
    /// Transforms input data using the current preprocessing pipeline.
    /// </summary>
    /// <param name="input">The input data to preprocess.</param>
    /// <returns>The preprocessed data, or the original input if no pipeline is configured.</returns>
    /// <remarks>
    /// <para>
    /// If no preprocessing pipeline has been configured, this method returns the input unchanged.
    /// This allows models to safely call this method without checking if preprocessing is configured.
    /// </para>
    /// </remarks>
    public static TInput Transform(TInput input)
    {
        var pipeline = Current;
        if (pipeline != null && pipeline.IsFitted)
        {
            return pipeline.Transform(input);
        }
        return input;
    }

    /// <summary>
    /// Fits the current preprocessing pipeline to data and transforms it.
    /// </summary>
    /// <param name="input">The data to fit and transform.</param>
    /// <returns>The preprocessed data, or the original input if no pipeline is configured.</returns>
    public static TInput FitTransform(TInput input)
    {
        var pipeline = Current;
        if (pipeline != null)
        {
            return pipeline.FitTransform(input);
        }
        return input;
    }

    /// <summary>
    /// Clears the current preprocessing pipeline.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This resets the preprocessing to no-op (pass-through) behavior.
    /// </para>
    /// </remarks>
    public static void Clear()
    {
        lock (_lock)
        {
            _current = null;
        }
    }
}
