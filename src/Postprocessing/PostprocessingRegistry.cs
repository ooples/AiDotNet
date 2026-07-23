using AiDotNet.Interfaces;

namespace AiDotNet.Postprocessing;

/// <summary>
/// Global registry for the postprocessing pipeline.
/// </summary>
/// <remarks>
/// <para>
/// <b>Deprecated:</b> This static registry causes race conditions when multiple
/// <c>AiModelBuilder</c> instances build models concurrently, because they overwrite
/// each other's pipeline. Use instance-based postprocessing via
/// <c>AiModelBuilder.ConfigurePostprocessing()</c> instead, which stores the pipeline
/// per-builder and flows it to <c>AiModelResult</c>.
/// </para>
/// <para><b>For Beginners:</b> You don't need to interact with this directly — just use AiModelBuilder:
/// <code>
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigurePostprocessing(pipeline => pipeline
///         .Add(new SoftmaxTransformer&lt;double&gt;())
///         .Add(new LabelDecoder&lt;double&gt;(labels)))
///     .ConfigureModel(new LogisticRegression&lt;double&gt;())
///     .Build(X, y);
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
[Obsolete("Use instance-based postprocessing via AiModelBuilder.ConfigurePostprocessing() instead. " +
    "This static registry causes race conditions in concurrent model building and will be removed in a future version.")]
public static class PostprocessingRegistry<T, TOutput>
{
    private static IDataTransformer<T, TOutput, TOutput>? _current;
    private static readonly object _lock = new();

    /// <summary>
    /// Gets or sets the current postprocessing pipeline.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Deprecated:</b> This property is no longer set by <c>AiModelBuilder.ConfigurePostprocessing()</c>.
    /// The builder now stores the pipeline per-instance and flows it to <c>AiModelResult</c>.
    /// This property remains only for backward compatibility with external code that sets it directly.
    /// </para>
    /// </remarks>
    public static IDataTransformer<T, TOutput, TOutput>? Current
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
    /// Gets whether a postprocessing pipeline is currently configured.
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
    /// Transforms output data using the current postprocessing pipeline.
    /// </summary>
    /// <param name="output">The model output to postprocess.</param>
    /// <returns>The postprocessed output, or the original output if no pipeline is configured.</returns>
    /// <remarks>
    /// <para>
    /// If no postprocessing pipeline has been configured, this method returns the output unchanged.
    /// This allows models to safely call this method without checking if postprocessing is configured.
    /// </para>
    /// </remarks>
    public static TOutput Transform(TOutput output)
    {
        var pipeline = Current;
        if (pipeline != null && pipeline.IsFitted)
        {
            return pipeline.Transform(output);
        }
        return output;
    }

    /// <summary>
    /// Fits the current postprocessing pipeline to data and transforms it.
    /// </summary>
    /// <param name="output">The data to fit and transform.</param>
    /// <returns>The postprocessed data, or the original output if no pipeline is configured.</returns>
    public static TOutput FitTransform(TOutput output)
    {
        var pipeline = Current;
        if (pipeline != null)
        {
            return pipeline.FitTransform(output);
        }
        return output;
    }

    /// <summary>
    /// Clears the current postprocessing pipeline.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This resets the postprocessing to no-op (pass-through) behavior.
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
