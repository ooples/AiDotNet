using AiDotNet.Interfaces;

namespace AiDotNet.Preprocessing;

/// <summary>
/// Global registry for the preprocessing pipeline.
/// </summary>
/// <remarks>
/// <para>
/// PreprocessingRegistry provides a static accessor for the active preprocessing pipeline.
/// It is kept in sync by <c>AiModelBuilder.ConfigurePreprocessing()</c> so that components
/// outside the builder (e.g., <c>DocumentNeuralNetworkBase</c>) can access the current pipeline
/// without holding a direct reference.
/// </para>
/// <para>
/// <b>Thread safety:</b> The static <see cref="Current"/> property is guarded by a lock, but
/// concurrent builders will overwrite each other's pipeline. For concurrent builds, each builder
/// stores its own <c>_preprocessingPipeline</c> instance and passes it directly to
/// <c>PreprocessingInfo</c>; the registry is a convenience accessor, not the authoritative source.
/// </para>
/// <para><b>For Beginners:</b> You don't need to interact with this directly — just use AiModelBuilder:
/// <code>
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigurePreprocessing(pipeline => pipeline
///         .Add(new SimpleImputer&lt;double&gt;(ImputationStrategy.Mean))
///         .Add(new StandardScaler&lt;double&gt;()))
///     .ConfigureModel(new LogisticRegression&lt;double&gt;())
///     .Build(X, y);
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
public static class PreprocessingRegistry<T, TInput>
{
    private static IDataTransformer<T, TInput, TInput>? _current;
    private static readonly object _lock = new();

    /// <summary>
    /// Gets or sets the current preprocessing pipeline.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is set automatically when you call AiModelBuilder.ConfigurePreprocessing().
    /// The pipeline is global and thread-safe.
    /// </para>
    /// <para><b>For Beginners:</b> You typically don't set this directly.
    /// Use AiModelBuilder.ConfigurePreprocessing() instead.
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
