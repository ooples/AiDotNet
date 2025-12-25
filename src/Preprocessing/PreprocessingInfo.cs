using AiDotNet.Interfaces;
using AiDotNet.Models;

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

    /// <summary>
    /// Creates a PreprocessingInfo from a legacy NormalizationInfo object.
    /// </summary>
    /// <param name="normInfo">The legacy normalization info.</param>
    /// <returns>A new PreprocessingInfo wrapping the legacy normalizer.</returns>
    /// <remarks>
    /// This method provides backward compatibility with code using the old INormalizer system.
    /// The legacy normalizer is wrapped in an adapter that implements IDataTransformer.
    /// </remarks>
    public static PreprocessingInfo<T, TInput, TOutput> FromNormalizationInfo(
        NormalizationInfo<T, TInput, TOutput> normInfo)
    {
        if (normInfo is null)
        {
            throw new ArgumentNullException(nameof(normInfo));
        }

        var result = new PreprocessingInfo<T, TInput, TOutput>();

        if (normInfo.Normalizer is not null)
        {
            // Create a pipeline containing the legacy normalizer adapter
            var featurePipeline = new PreprocessingPipeline<T, TInput, TInput>();
            featurePipeline.Add("legacy_normalizer", new LegacyNormalizerAdapter<T, TInput, TOutput>(
                normInfo.Normalizer, normInfo.XParams));
            result.Pipeline = featurePipeline;

            // Create a target pipeline for inverse transformation
            var targetPipeline = new PreprocessingPipeline<T, TOutput, TOutput>();
            targetPipeline.Add("legacy_target", new LegacyTargetNormalizerAdapter<T, TInput, TOutput>(
                normInfo.Normalizer, normInfo.YParams));
            result.TargetPipeline = targetPipeline;
        }

        return result;
    }
}

/// <summary>
/// Adapter that wraps a legacy INormalizer for use as an IDataTransformer for features.
/// </summary>
internal class LegacyNormalizerAdapter<T, TInput, TOutput> : IDataTransformer<T, TInput, TInput>
{
    private readonly INormalizer<T, TInput, TOutput> _normalizer;
    private readonly List<NormalizationParameters<T>> _xParams;
    private bool _isFitted;

    public LegacyNormalizerAdapter(
        INormalizer<T, TInput, TOutput> normalizer,
        List<NormalizationParameters<T>> xParams)
    {
        _normalizer = normalizer ?? throw new ArgumentNullException(nameof(normalizer));
        _xParams = xParams ?? new List<NormalizationParameters<T>>();
        _isFitted = _xParams.Count > 0;
    }

    public bool IsFitted => _isFitted;
    public int[]? ColumnIndices => null;
    public bool SupportsInverseTransform => false;

    public void Fit(TInput data)
    {
        var (_, xParams) = _normalizer.NormalizeInput(data);
        _xParams.Clear();
        _xParams.AddRange(xParams);
        _isFitted = true;
    }

    public TInput Transform(TInput data)
    {
        if (!_isFitted)
        {
            throw new InvalidOperationException("Normalizer has not been fitted.");
        }

        var (normalized, _) = _normalizer.NormalizeInput(data);
        return normalized;
    }

    public TInput FitTransform(TInput data)
    {
        var (normalized, xParams) = _normalizer.NormalizeInput(data);
        _xParams.Clear();
        _xParams.AddRange(xParams);
        _isFitted = true;
        return normalized;
    }

    public TInput InverseTransform(TInput data)
    {
        throw new NotSupportedException("Legacy normalizer adapter does not support inverse transform for features.");
    }

    public string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        return inputFeatureNames ?? Array.Empty<string>();
    }
}

/// <summary>
/// Adapter that wraps a legacy INormalizer for use as an IDataTransformer for targets.
/// </summary>
internal class LegacyTargetNormalizerAdapter<T, TInput, TOutput> : IDataTransformer<T, TOutput, TOutput>
{
    private readonly INormalizer<T, TInput, TOutput> _normalizer;
    private readonly NormalizationParameters<T> _yParams;
    private bool _isFitted;

    public LegacyTargetNormalizerAdapter(
        INormalizer<T, TInput, TOutput> normalizer,
        NormalizationParameters<T> yParams)
    {
        _normalizer = normalizer;
        _yParams = yParams;
        _isFitted = true;
    }

    public bool IsFitted => _isFitted;
    public int[]? ColumnIndices => null;
    public bool SupportsInverseTransform => true;

    public void Fit(TOutput data)
    {
        _isFitted = true;
    }

    public TOutput Transform(TOutput data)
    {
        return data;
    }

    public TOutput FitTransform(TOutput data)
    {
        _isFitted = true;
        return data;
    }

    public TOutput InverseTransform(TOutput data)
    {
        if (_normalizer is null)
        {
            return data;
        }

        return _normalizer.Denormalize(data, _yParams);
    }

    public string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        return inputFeatureNames ?? Array.Empty<string>();
    }
}
