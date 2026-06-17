using System;
using AiDotNet.Augmentation;
using AiDotNet.Postprocessing;
using AiDotNet.Preprocessing;
using AiDotNet.Preprocessing.DataPreparation;

namespace AiDotNet.Configuration;

/// <summary>
/// Component that owns the data-pipeline configuration for an AI model build: preprocessing,
/// postprocessing, data loading, data preparation, and augmentation. Extracted from
/// <c>AiModelBuilder</c> as slice 1 of the audit-2026-05 phase-2a DI refactor (see
/// <c>docs/internal/audit-2026-05-phase2a-aimodelbuilder-refactor.md</c>).
/// </summary>
/// <typeparam name="T">Element numeric type (e.g. <c>double</c> / <c>float</c>).</typeparam>
/// <typeparam name="TInput">Model input tensor type (e.g. <c>Matrix&lt;double&gt;</c>).</typeparam>
/// <typeparam name="TOutput">Model output tensor type (e.g. <c>Vector&lt;double&gt;</c>).</typeparam>
/// <remarks>
/// <para>
/// The interface exposes the configured state via get-only properties so that the rest of the
/// build pipeline (<c>AiModelBuilder.BuildAsync</c>, the partial-class siblings, and any
/// alternative composition root such as a future YAML loader) can consume it without depending
/// on the god class. The <c>Configure*</c> methods mutate the underlying component instance and
/// return <c>void</c> — the fluent chaining returns happen at the <c>AiModelBuilder</c> facade
/// layer, not here, so the component stays usable from non-fluent contexts (e.g. tests).
/// </para>
/// <para>
/// This is the public extension seam: third-party packages or downstream consumers can implement
/// <see cref="IAiModelDataPipeline{T,TInput,TOutput}"/> with custom defaults (e.g. a domain-
/// specific preprocessing pipeline that ships with the audit-2026-05 federal-use SDK) and inject
/// it into <c>AiModelBuilder</c> via the ctor parameter added in the same slice.
/// </para>
/// </remarks>
public interface IAiModelDataPipeline<T, TInput, TOutput>
{
    /// <summary>The configured preprocessing pipeline, or <c>null</c> if <c>ConfigurePreprocessing</c> hasn't been called.</summary>
    PreprocessingPipeline<T, TInput, TInput>? PreprocessingPipeline { get; }

    /// <summary>The configured postprocessing pipeline, or <c>null</c> if <c>ConfigurePostprocessing</c> hasn't been called.</summary>
    PostprocessingPipeline<T, TOutput, TOutput>? PostprocessingPipeline { get; }

    /// <summary>
    /// Optional cap on the number of training rows fed into the post-train pipeline-fit
    /// <c>Predict</c> call. <c>null</c> = no cap = full <c>XTrain</c>. Set via
    /// <see cref="SetPostprocessingFitMaxRows"/> when pipeline transformers stabilise on a
    /// subsample and the doubled build-time inference cost matters (review #1368 C7HAu).
    /// </summary>
    int? PostprocessingFitMaxRows { get; }

    /// <summary>The configured data loader, or <c>null</c> if <c>ConfigureDataLoader</c> hasn't been called.</summary>
    IDataLoader<T>? DataLoader { get; }

    /// <summary>
    /// The configured row-changing data-preparation pipeline (outlier removal, SMOTE
    /// augmentation, etc.), or <c>null</c> if <c>ConfigureDataPreparation</c> hasn't been called.
    /// </summary>
    DataPreparationPipeline<T>? DataPreparationPipeline { get; }

    /// <summary>The configured (non-row-changing) augmentation configuration, or <c>null</c> if <c>ConfigureAugmentation</c> hasn't been called.</summary>
    AugmentationConfig? AugmentationConfig { get; }

    /// <summary>Configures preprocessing using a pipeline-builder callback. <c>null</c> applies AutoML defaults (mean imputation + standard scaling).</summary>
    void ConfigurePreprocessing(Action<PreprocessingPipeline<T, TInput, TInput>>? pipelineBuilder);

    /// <summary>Configures preprocessing with a single transformer. <c>null</c> applies AutoML defaults.</summary>
    void ConfigurePreprocessing(IDataTransformer<T, TInput, TInput>? transformer);

    /// <summary>Configures preprocessing from a pre-built pipeline. <c>null</c> applies AutoML defaults.</summary>
    void ConfigurePreprocessing(PreprocessingPipeline<T, TInput, TInput>? pipeline);

    /// <summary>Configures postprocessing using a pipeline-builder callback. <c>null</c> leaves the pipeline empty.</summary>
    void ConfigurePostprocessing(Action<PostprocessingPipeline<T, TOutput, TOutput>>? pipelineBuilder);

    /// <summary>Configures postprocessing with a single transformer. <c>null</c> leaves the pipeline empty.</summary>
    void ConfigurePostprocessing(IDataTransformer<T, TOutput, TOutput>? transformer);

    /// <summary>Configures postprocessing from a pre-built pipeline. <c>null</c> leaves the pipeline empty.</summary>
    void ConfigurePostprocessing(PostprocessingPipeline<T, TOutput, TOutput>? pipeline);

    /// <summary>Caps the number of training rows fed into the post-train pipeline-fit <c>Predict</c>. Non-positive clears the cap.</summary>
    void SetPostprocessingFitMaxRows(int? maxRows);

    /// <summary>Configures the data loader. <c>null</c> throws <see cref="ArgumentNullException"/>.</summary>
    void ConfigureDataLoader(IDataLoader<T> dataLoader);

    /// <summary>Configures the row-changing data-preparation pipeline. <c>null</c> throws <see cref="ArgumentNullException"/>.</summary>
    void ConfigureDataPreparation(Action<DataPreparationPipeline<T>> pipelineBuilder);

    /// <summary>Configures augmentation with an explicit (possibly null) config. <c>null</c> applies modality-auto-detected defaults.</summary>
    void ConfigureAugmentation(AugmentationConfig? config);

    /// <summary>Strongly-typed overload of <see cref="ConfigureAugmentation(AugmentationConfig?)"/> with IDE-discoverable typed <c>Augmenter</c> slot.</summary>
    void ConfigureAugmentation(AugmentationConfig<T, TInput>? config);
}
