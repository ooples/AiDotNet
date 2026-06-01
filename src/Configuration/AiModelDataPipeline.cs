using System;
using AiDotNet.Augmentation;
using AiDotNet.Postprocessing;
using AiDotNet.Preprocessing;
using AiDotNet.Preprocessing.DataPreparation;
using AiDotNet.Preprocessing.Imputers;
using AiDotNet.Preprocessing.Scalers;

namespace AiDotNet.Configuration;

/// <summary>
/// Default implementation of <see cref="IAiModelDataPipeline{T,TInput,TOutput}"/>. Mirrors the
/// preprocessing / postprocessing / data-loading / data-preparation / augmentation logic that
/// previously lived inline in <c>AiModelBuilder</c>, with no behavioural change. The facade
/// continues to be the supported entry point; this class is the audit-2026-05 phase-2a internal
/// reorganisation that makes the data-pipeline surface testable in isolation and replaceable
/// without touching the god class.
/// </summary>
/// <typeparam name="T">Element numeric type (e.g. <c>double</c> / <c>float</c>).</typeparam>
/// <typeparam name="TInput">Model input tensor type.</typeparam>
/// <typeparam name="TOutput">Model output tensor type.</typeparam>
public class AiModelDataPipeline<T, TInput, TOutput> : IAiModelDataPipeline<T, TInput, TOutput>
{
    /// <inheritdoc/>
    public PreprocessingPipeline<T, TInput, TInput>? PreprocessingPipeline { get; private set; }

    /// <inheritdoc/>
    public PostprocessingPipeline<T, TOutput, TOutput>? PostprocessingPipeline { get; private set; }

    /// <inheritdoc/>
    public int? PostprocessingFitMaxRows { get; private set; }

    /// <inheritdoc/>
    public IDataLoader<T>? DataLoader { get; private set; }

    /// <inheritdoc/>
    public DataPreparationPipeline<T>? DataPreparationPipeline { get; private set; }

    /// <inheritdoc/>
    public AugmentationConfig? AugmentationConfig { get; private set; }

    /// <inheritdoc/>
    public void ConfigurePreprocessing(Action<PreprocessingPipeline<T, TInput, TInput>>? pipelineBuilder)
    {
        var pipeline = new PreprocessingPipeline<T, TInput, TInput>();
        if (pipelineBuilder is not null)
        {
            pipelineBuilder(pipeline);
        }
        else
        {
            ApplyAutoMlPreprocessingDefaults(pipeline);
        }
        PreprocessingPipeline = pipeline;
    }

    /// <inheritdoc/>
    public void ConfigurePreprocessing(IDataTransformer<T, TInput, TInput>? transformer)
    {
        var pipeline = new PreprocessingPipeline<T, TInput, TInput>();
        if (transformer is not null)
        {
            pipeline.Add(transformer);
        }
        else
        {
            ApplyAutoMlPreprocessingDefaults(pipeline);
        }
        PreprocessingPipeline = pipeline;
    }

    /// <inheritdoc/>
    public void ConfigurePreprocessing(PreprocessingPipeline<T, TInput, TInput>? pipeline)
    {
        if (pipeline is not null)
        {
            PreprocessingPipeline = pipeline;
        }
        else
        {
            var fresh = new PreprocessingPipeline<T, TInput, TInput>();
            ApplyAutoMlPreprocessingDefaults(fresh);
            PreprocessingPipeline = fresh;
        }
    }

    /// <inheritdoc/>
    public void ConfigurePostprocessing(Action<PostprocessingPipeline<T, TOutput, TOutput>>? pipelineBuilder)
    {
        var pipeline = new PostprocessingPipeline<T, TOutput, TOutput>();
        // Note: Unlike preprocessing, postprocessing has no universal defaults — the appropriate
        // postprocessing depends heavily on the model type (classification vs regression vs
        // generation), so the empty pipeline is the right starting point when no builder is
        // supplied.
        pipelineBuilder?.Invoke(pipeline);
        PostprocessingPipeline = pipeline;
    }

    /// <inheritdoc/>
    public void ConfigurePostprocessing(IDataTransformer<T, TOutput, TOutput>? transformer)
    {
        var pipeline = new PostprocessingPipeline<T, TOutput, TOutput>();
        if (transformer is not null) pipeline.Add(transformer);
        PostprocessingPipeline = pipeline;
    }

    /// <inheritdoc/>
    public void ConfigurePostprocessing(PostprocessingPipeline<T, TOutput, TOutput>? pipeline)
    {
        PostprocessingPipeline = pipeline ?? new PostprocessingPipeline<T, TOutput, TOutput>();
    }

    /// <inheritdoc/>
    public void SetPostprocessingFitMaxRows(int? maxRows)
    {
        PostprocessingFitMaxRows = maxRows is int v && v > 0 ? v : null;
    }

    /// <inheritdoc/>
    public void ConfigureDataLoader(IDataLoader<T> dataLoader)
    {
        if (dataLoader is null) throw new ArgumentNullException(nameof(dataLoader));
        DataLoader = dataLoader;
    }

    /// <inheritdoc/>
    public void ConfigureDataPreparation(Action<DataPreparationPipeline<T>> pipelineBuilder)
    {
        if (pipelineBuilder is null) throw new ArgumentNullException(nameof(pipelineBuilder));
        var pipeline = new DataPreparationPipeline<T>();
        pipelineBuilder(pipeline);
        DataPreparationPipeline = pipeline;

        // Side-effect preserved verbatim from the inline AiModelBuilder implementation —
        // DataPreparationRegistry is a process-global lookup that BuildAsync's worker
        // threads consult before applying row-changing operations. Subsequent slices may
        // refactor this into an explicit dependency on the component instead of a static
        // registry, but during slice 1 we keep the exact pre-refactor behaviour.
        DataPreparationRegistry<T>.Current = pipeline;
    }

    /// <inheritdoc/>
    public void ConfigureAugmentation(AugmentationConfig? config)
    {
        AugmentationConfig = config ?? CreateDefaultAugmentationConfig();
    }

    /// <inheritdoc/>
    public void ConfigureAugmentation(AugmentationConfig<T, TInput>? config)
        => ConfigureAugmentation((AugmentationConfig?)config);

    private AugmentationConfig CreateDefaultAugmentationConfig()
    {
        var config = new AugmentationConfig();
        var modality = DataModalityDetector.Detect<TInput>();
        switch (modality)
        {
            case DataModality.Image:   config.ImageSettings   = new ImageAugmentationSettings();   break;
            case DataModality.Tabular: config.TabularSettings = new TabularAugmentationSettings(); break;
            case DataModality.Audio:   config.AudioSettings   = new AudioAugmentationSettings();   break;
            case DataModality.Text:    config.TextSettings    = new TextAugmentationSettings();    break;
            case DataModality.Video:   config.VideoSettings   = new VideoAugmentationSettings();   break;
            default: break;
        }
        return config;
    }

    private static void ApplyAutoMlPreprocessingDefaults(PreprocessingPipeline<T, TInput, TInput> pipeline)
    {
        // Industry-standard AutoML defaults preserved verbatim from inline AiModelBuilder
        // implementation: handle missing values via mean imputation, then z-score normalise.
        pipeline.Add((IDataTransformer<T, TInput, TInput>)(object)new SimpleImputer<T>(ImputationStrategy.Mean));
        pipeline.Add((IDataTransformer<T, TInput, TInput>)(object)new StandardScaler<T>());
    }
}
