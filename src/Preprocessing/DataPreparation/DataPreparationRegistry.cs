using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation;

/// <summary>
/// Global registry for the data preparation pipeline.
/// </summary>
/// <remarks>
/// <para>
/// DataPreparationRegistry provides a singleton pattern for managing the active data preparation
/// pipeline. This handles row-changing operations like outlier removal and data augmentation.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is an internal component that stores your data preparation settings.
/// You don't need to interact with this directly - just use AiModelBuilder:
///
/// <code>
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigurePreprocessing(options => options
///         .RemoveOutliers(new IsolationForest&lt;double&gt;())
///         .ApplySmote())
///     .ConfigureModel(new LogisticRegression&lt;double&gt;())
///     .Build(X, y);
/// </code>
///
/// The configured data preparation is automatically applied during training.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public static class DataPreparationRegistry<T>
{
    private static DataPreparationPipeline<T>? _current;
    private static readonly object _lock = new();

    /// <summary>
    /// Gets or sets the current data preparation pipeline.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is set automatically when you call AiModelBuilder.ConfigurePreprocessing().
    /// The pipeline is global and thread-safe.
    /// </para>
    /// </remarks>
    public static DataPreparationPipeline<T>? Current
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
    /// Gets whether a data preparation pipeline is currently configured.
    /// </summary>
    public static bool IsConfigured
    {
        get
        {
            lock (_lock)
            {
                return _current != null && _current.Count > 0;
            }
        }
    }

    /// <summary>
    /// Applies the current data preparation pipeline to data.
    /// </summary>
    /// <param name="X">The feature matrix.</param>
    /// <param name="y">The label vector.</param>
    /// <returns>The prepared (X, y) data, or original data if no pipeline is configured.</returns>
    /// <remarks>
    /// <para>
    /// If no data preparation pipeline has been configured, this method returns the input unchanged.
    /// This allows models to safely call this method without checking if preparation is configured.
    /// </para>
    /// </remarks>
    public static (Matrix<T> X, Vector<T> y) FitResample(Matrix<T> X, Vector<T> y)
    {
        var pipeline = Current;
        if (pipeline != null && pipeline.Count > 0)
        {
            return pipeline.FitResample(X, y);
        }
        return (X, y);
    }

    /// <summary>
    /// Clears the current data preparation pipeline.
    /// </summary>
    /// <remarks>
    /// This resets the data preparation to no-op (pass-through) behavior.
    /// </remarks>
    public static void Clear()
    {
        lock (_lock)
        {
            _current = null;
        }
    }
}
