namespace AiDotNet.Preprocessing;

/// <summary>
/// Stores the fitted preprocessing pipeline for inference.
/// </summary>
/// <remarks>
/// <para>
/// This class encapsulates the preprocessing state needed to transform new data
/// during inference. It stores the fitted feature pipeline and optionally a target
/// pipeline for inverse transformation of predictions.
/// </para>
/// <para><b>For Beginners:</b> After training, your preprocessing pipeline has "learned"
/// things like the mean and standard deviation for scaling. This class stores all that
/// learned information so you can apply the same transformations to new data during
/// predictions.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type (typically Matrix&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (typically Vector&lt;T&gt;).</typeparam>
public class PreprocessingInfo<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the fitted feature preprocessing pipeline.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This pipeline transforms input features (X) before they are passed to the model.
    /// It must be fitted before use during inference.
    /// </para>
    /// <para><b>For Beginners:</b> This is the pipeline that transforms your input data
    /// (like scaling, encoding, and imputation) before making predictions.
    /// </para>
    /// </remarks>
    public PreprocessingPipeline<T, TInput, TInput>? Pipeline { get; set; }

    /// <summary>
    /// Gets or sets the fitted target preprocessing pipeline.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This optional pipeline transforms targets (y) during training and can perform
    /// inverse transformation on predictions during inference.
    /// </para>
    /// <para><b>For Beginners:</b> If you scaled your target values during training
    /// (like log-transforming prices), this pipeline knows how to "unscale" the
    /// predictions back to the original range.
    /// </para>
    /// </remarks>
    public PreprocessingPipeline<T, TOutput, TOutput>? TargetPipeline { get; set; }

    /// <summary>
    /// Gets whether the feature pipeline has been fitted to data.
    /// </summary>
    public bool IsFitted => Pipeline?.IsFitted ?? false;

    /// <summary>
    /// Gets whether the target pipeline has been fitted to data.
    /// </summary>
    public bool IsTargetFitted => TargetPipeline?.IsFitted ?? false;

    /// <summary>
    /// Creates a new instance of <see cref="PreprocessingInfo{T, TInput, TOutput}"/>.
    /// </summary>
    public PreprocessingInfo()
    {
    }

    /// <summary>
    /// Creates a new instance with the specified fitted pipeline.
    /// </summary>
    /// <param name="pipeline">The fitted feature preprocessing pipeline.</param>
    public PreprocessingInfo(PreprocessingPipeline<T, TInput, TInput> pipeline)
    {
        Pipeline = pipeline;
    }

    /// <summary>
    /// Creates a new instance with both feature and target pipelines.
    /// </summary>
    /// <param name="pipeline">The fitted feature preprocessing pipeline.</param>
    /// <param name="targetPipeline">The fitted target preprocessing pipeline.</param>
    public PreprocessingInfo(
        PreprocessingPipeline<T, TInput, TInput> pipeline,
        PreprocessingPipeline<T, TOutput, TOutput>? targetPipeline)
    {
        Pipeline = pipeline;
        TargetPipeline = targetPipeline;
    }

    /// <summary>
    /// Transforms input features using the fitted pipeline.
    /// </summary>
    /// <param name="input">The input features to transform.</param>
    /// <returns>The transformed features.</returns>
    /// <exception cref="InvalidOperationException">Thrown if the pipeline is not fitted.</exception>
    public TInput TransformFeatures(TInput input)
    {
        if (Pipeline is null)
        {
            throw new InvalidOperationException("Feature preprocessing pipeline is not configured.");
        }

        if (!Pipeline.IsFitted)
        {
            throw new InvalidOperationException("Feature preprocessing pipeline has not been fitted. Call FitTransform during training first.");
        }

        return Pipeline.Transform(input);
    }

    /// <summary>
    /// Inverse transforms predictions using the target pipeline.
    /// </summary>
    /// <param name="predictions">The predictions to inverse transform.</param>
    /// <returns>The inverse transformed predictions in original scale.</returns>
    /// <exception cref="InvalidOperationException">Thrown if the target pipeline is not configured or doesn't support inverse transform.</exception>
    public TOutput InverseTransformPredictions(TOutput predictions)
    {
        if (TargetPipeline is null)
        {
            // No target pipeline means no transformation was applied to targets
            return predictions;
        }

        if (!TargetPipeline.IsFitted)
        {
            throw new InvalidOperationException("Target preprocessing pipeline has not been fitted.");
        }

        if (!TargetPipeline.SupportsInverseTransform)
        {
            throw new InvalidOperationException("Target preprocessing pipeline does not support inverse transformation.");
        }

        return TargetPipeline.InverseTransform(predictions);
    }
}
