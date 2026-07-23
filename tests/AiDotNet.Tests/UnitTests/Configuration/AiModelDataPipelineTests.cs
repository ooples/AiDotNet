using System;
using System.Threading.Tasks;
using AiDotNet.Augmentation;
using AiDotNet.Configuration;
using AiDotNet.Interfaces;
using AiDotNet.Postprocessing;
using AiDotNet.Preprocessing;
using AiDotNet.Preprocessing.DataPreparation;
using AiDotNet.Preprocessing.Imputers;
using AiDotNet.Preprocessing.Scalers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Configuration;

/// <summary>
/// Unit tests for <see cref="AiModelDataPipeline{T,TInput,TOutput}"/> — the slice-1
/// extraction of the audit-2026-05 phase-2a <c>AiModelBuilder</c> DI refactor (see
/// <c>docs/internal/audit-2026-05-phase2a-aimodelbuilder-refactor.md</c>). These
/// tests exercise the component in isolation, without instantiating the rest of
/// <c>AiModelBuilder</c>, to prove that the data-pipeline concern can be used by
/// alternative composition roots (e.g. future YAML loaders, federal-use SDK builds
/// that ship their own defaults).
/// </summary>
public class AiModelDataPipelineTests
{
    [Fact(Timeout = 30000)]
    public async Task InitialState_AllSlotsAreNull()
    {
        await Task.Yield();
        var pipeline = new AiModelDataPipeline<double, Matrix<double>, Vector<double>>();

        Assert.Null(pipeline.PreprocessingPipeline);
        Assert.Null(pipeline.PostprocessingPipeline);
        Assert.Null(pipeline.PostprocessingFitMaxRows);
        Assert.Null(pipeline.DataLoader);
        Assert.Null(pipeline.DataPreparationPipeline);
        Assert.Null(pipeline.AugmentationConfig);
    }

    [Fact(Timeout = 30000)]
    public async Task ConfigurePreprocessing_NullAction_AppliesAutoMlDefaults()
    {
        await Task.Yield();
        var pipeline = new AiModelDataPipeline<double, Matrix<double>, Vector<double>>();

        pipeline.ConfigurePreprocessing((Action<PreprocessingPipeline<double, Matrix<double>, Matrix<double>>>?)null);

        Assert.NotNull(pipeline.PreprocessingPipeline);
        // AutoML defaults: 2 transformers — SimpleImputer(Mean) + StandardScaler.
        Assert.Equal(2, pipeline.PreprocessingPipeline!.Count);
    }

    [Fact(Timeout = 30000)]
    public async Task ConfigurePreprocessing_NullTransformer_AppliesAutoMlDefaults()
    {
        await Task.Yield();
        var pipeline = new AiModelDataPipeline<double, Matrix<double>, Vector<double>>();

        pipeline.ConfigurePreprocessing((IDataTransformer<double, Matrix<double>, Matrix<double>>?)null);

        Assert.NotNull(pipeline.PreprocessingPipeline);
        Assert.Equal(2, pipeline.PreprocessingPipeline!.Count);
    }

    [Fact(Timeout = 30000)]
    public async Task ConfigurePreprocessing_NullPipeline_AppliesAutoMlDefaults()
    {
        await Task.Yield();
        var pipeline = new AiModelDataPipeline<double, Matrix<double>, Vector<double>>();

        pipeline.ConfigurePreprocessing((PreprocessingPipeline<double, Matrix<double>, Matrix<double>>?)null);

        Assert.NotNull(pipeline.PreprocessingPipeline);
        Assert.Equal(2, pipeline.PreprocessingPipeline!.Count);
    }

    [Fact(Timeout = 30000)]
    public async Task ConfigurePreprocessing_ExplicitPipeline_UsesExactInstance()
    {
        await Task.Yield();
        var pipeline = new AiModelDataPipeline<double, Matrix<double>, Vector<double>>();
        var explicitPipeline = new PreprocessingPipeline<double, Matrix<double>, Matrix<double>>();
        explicitPipeline.Add(new StandardScaler<double>() as IDataTransformer<double, Matrix<double>, Matrix<double>>
            ?? throw new InvalidOperationException("StandardScaler<double> not assignable to IDataTransformer<double, Matrix<double>, Matrix<double>>"));

        pipeline.ConfigurePreprocessing(explicitPipeline);

        Assert.Same(explicitPipeline, pipeline.PreprocessingPipeline);
        Assert.Equal(1, pipeline.PreprocessingPipeline!.Count);
    }

    [Fact(Timeout = 30000)]
    public async Task ConfigurePostprocessing_NullAction_LeavesPipelineEmpty()
    {
        await Task.Yield();
        var pipeline = new AiModelDataPipeline<double, Matrix<double>, Vector<double>>();

        pipeline.ConfigurePostprocessing((Action<PostprocessingPipeline<double, Vector<double>, Vector<double>>>?)null);

        Assert.NotNull(pipeline.PostprocessingPipeline);
        // Postprocessing has no universal defaults — empty pipeline is correct.
        Assert.Equal(0, pipeline.PostprocessingPipeline!.Count);
    }

    [Fact(Timeout = 30000)]
    public async Task SetPostprocessingFitMaxRows_PositiveValue_Stored()
    {
        await Task.Yield();
        var pipeline = new AiModelDataPipeline<double, Matrix<double>, Vector<double>>();

        pipeline.SetPostprocessingFitMaxRows(500);

        Assert.Equal(500, pipeline.PostprocessingFitMaxRows);
    }

    [Fact(Timeout = 30000)]
    public async Task SetPostprocessingFitMaxRows_ZeroOrNegative_ClearsToNull()
    {
        await Task.Yield();
        var pipeline = new AiModelDataPipeline<double, Matrix<double>, Vector<double>>();

        pipeline.SetPostprocessingFitMaxRows(100);
        Assert.Equal(100, pipeline.PostprocessingFitMaxRows);

        pipeline.SetPostprocessingFitMaxRows(0);
        Assert.Null(pipeline.PostprocessingFitMaxRows);

        pipeline.SetPostprocessingFitMaxRows(-50);
        Assert.Null(pipeline.PostprocessingFitMaxRows);

        pipeline.SetPostprocessingFitMaxRows(null);
        Assert.Null(pipeline.PostprocessingFitMaxRows);
    }

    [Fact(Timeout = 30000)]
    public async Task ConfigureDataLoader_Null_Throws()
    {
        await Task.Yield();
        var pipeline = new AiModelDataPipeline<double, Matrix<double>, Vector<double>>();

        Assert.Throws<ArgumentNullException>(() => pipeline.ConfigureDataLoader(null!));
    }

    [Fact(Timeout = 30000)]
    public async Task ConfigureDataPreparation_Null_Throws()
    {
        await Task.Yield();
        var pipeline = new AiModelDataPipeline<double, Matrix<double>, Vector<double>>();

        Assert.Throws<ArgumentNullException>(() => pipeline.ConfigureDataPreparation(null!));
    }

    [Fact(Timeout = 30000)]
    public async Task ConfigureDataPreparation_BuilderInvokedAndPipelineStored()
    {
        await Task.Yield();
        var pipeline = new AiModelDataPipeline<double, Matrix<double>, Vector<double>>();
        DataPreparationPipeline<double>? builderReceived = null;

        pipeline.ConfigureDataPreparation(p => builderReceived = p);

        Assert.NotNull(pipeline.DataPreparationPipeline);
        Assert.Same(pipeline.DataPreparationPipeline, builderReceived);
        Assert.Same(pipeline.DataPreparationPipeline, DataPreparationRegistry<double>.Current);
    }

    [Fact(Timeout = 30000)]
    public async Task ConfigureAugmentation_Null_ResolvesModalityDefault()
    {
        await Task.Yield();
        var pipeline = new AiModelDataPipeline<double, Matrix<double>, Vector<double>>();

        pipeline.ConfigureAugmentation((AugmentationConfig?)null);

        Assert.NotNull(pipeline.AugmentationConfig);
        // Matrix<double> auto-detects as Tabular modality, so TabularSettings should be populated.
        Assert.NotNull(pipeline.AugmentationConfig!.TabularSettings);
    }

    [Fact(Timeout = 30000)]
    public async Task ConfigureAugmentation_ExplicitConfig_UsedAsIs()
    {
        await Task.Yield();
        var pipeline = new AiModelDataPipeline<double, Matrix<double>, Vector<double>>();
        var explicitConfig = new AugmentationConfig
        {
            ImageSettings = new ImageAugmentationSettings(),
        };

        pipeline.ConfigureAugmentation(explicitConfig);

        Assert.Same(explicitConfig, pipeline.AugmentationConfig);
        Assert.NotNull(pipeline.AugmentationConfig!.ImageSettings);
    }

    [Fact(Timeout = 30000)]
    public async Task Interface_IsImplementedByDefaultComponent()
    {
        await Task.Yield();
        // Proves callers can hold the component via the interface.
        IAiModelDataPipeline<double, Matrix<double>, Vector<double>> component
            = new AiModelDataPipeline<double, Matrix<double>, Vector<double>>();

        component.SetPostprocessingFitMaxRows(42);
        Assert.Equal(42, component.PostprocessingFitMaxRows);
    }
}
