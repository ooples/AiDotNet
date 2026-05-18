using AiDotNet.Interfaces;
using AiDotNet.Postprocessing;
using AiDotNet.Preprocessing;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.ConfigureMethodCoverage;

/// <summary>
/// Bucket 6 — Configure* methods that wire data-processing pipelines.
/// These tests use RECORDING transformers that increment a call counter
/// on every Fit/Transform invocation, so the assertion proves the
/// configured transformer was actually invoked (not just stored).
/// </summary>
/// <remarks>
/// Methods covered (3 overloads each = 6 unique entry points):
/// <list type="bullet">
///   <item>ConfigurePreprocessing(Action&lt;PreprocessingPipeline&gt;)</item>
///   <item>ConfigurePreprocessing(IDataTransformer)</item>
///   <item>ConfigurePreprocessing(PreprocessingPipeline)</item>
///   <item>ConfigurePostprocessing(Action&lt;PostprocessingPipeline&gt;)</item>
///   <item>ConfigurePostprocessing(IDataTransformer)</item>
///   <item>ConfigurePostprocessing(PostprocessingPipeline)</item>
/// </list>
/// </remarks>
[Collection("ConfigureMethodCoverage")]
public class Bucket6_PrePostProcessingTests : ConfigureMethodTestBase
{
    private readonly ITestOutputHelper _output;
    public Bucket6_PrePostProcessingTests(ITestOutputHelper output) { _output = output; }

    /// <summary>
    /// ConfigurePreprocessing (Action overload) — verifies that a transformer
    /// added via the builder action is actually invoked during BuildAsync.
    /// A stored-but-not-consumed regression would leave FitCalls + TransformCalls
    /// at 0.
    /// </summary>
    [Fact]
    [Trait("category", "integration-configure-method")]
    public async Task ConfigurePreprocessing_ActionOverload_ActuallyInvokesTransformer()
    {
        var recorder = new RecordingTensorTransformer();
        var (features, labels) = MakeMemorizationSet();
        var loader = MakeCanaryLoader(features, labels);
        var model = MakeCanaryModel();

        await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureDataLoader(loader)
            .ConfigurePreprocessing(p => p.Add(recorder))
            .BuildAsync();

        // BuildSupervisedInternalAsync calls _preprocessingPipeline
        // .FitTransform(XTrain) at AiModelBuilder.cs:2711 when both the
        // pipeline and a data loader are configured. The recording
        // transformer's counter increments on every Fit / Transform call.
        Assert.True(recorder.FitCalls + recorder.FitTransformCalls > 0,
            $"ConfigurePreprocessing(Action) wired the transformer but BuildAsync never invoked it (Fit={recorder.FitCalls}, FitTransform={recorder.FitTransformCalls}, Transform={recorder.TransformCalls}). Stored-but-not-consumed regression.");
    }

    /// <summary>
    /// ConfigurePreprocessing (transformer overload) — same check via the
    /// single-transformer overload.
    /// </summary>
    [Fact]
    [Trait("category", "integration-configure-method")]
    public async Task ConfigurePreprocessing_TransformerOverload_ActuallyInvokesTransformer()
    {
        var recorder = new RecordingTensorTransformer();
        var (features, labels) = MakeMemorizationSet();
        var loader = MakeCanaryLoader(features, labels);
        var model = MakeCanaryModel();

        await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureDataLoader(loader)
            .ConfigurePreprocessing(recorder)
            .BuildAsync();

        Assert.True(recorder.FitCalls + recorder.FitTransformCalls > 0,
            $"ConfigurePreprocessing(transformer) wired the transformer but BuildAsync never invoked it (Fit={recorder.FitCalls}, FitTransform={recorder.FitTransformCalls}, Transform={recorder.TransformCalls}). Stored-but-not-consumed regression.");
    }

    /// <summary>
    /// ConfigurePreprocessing (prebuilt-pipeline overload) — same check via
    /// the prebuilt-pipeline overload.
    /// </summary>
    [Fact]
    [Trait("category", "integration-configure-method")]
    public async Task ConfigurePreprocessing_PipelineOverload_ActuallyInvokesTransformer()
    {
        var recorder = new RecordingTensorTransformer();
        var pipeline = new PreprocessingPipeline<float, Tensor<float>, Tensor<float>>();
        pipeline.Add(recorder);

        var (features, labels) = MakeMemorizationSet();
        var loader = MakeCanaryLoader(features, labels);
        var model = MakeCanaryModel();

        await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureDataLoader(loader)
            .ConfigurePreprocessing(pipeline)
            .BuildAsync();

        Assert.True(recorder.FitCalls + recorder.FitTransformCalls > 0,
            $"ConfigurePreprocessing(prebuilt) wired the transformer but BuildAsync never invoked it (Fit={recorder.FitCalls}, FitTransform={recorder.FitTransformCalls}, Transform={recorder.TransformCalls}). Stored-but-not-consumed regression.");
    }

    /// <summary>
    /// ConfigurePostprocessing (Action overload) — postprocessing only fires
    /// on Predict via the resulting AiModelResult. The test asserts the
    /// pipeline is observable on the post-build builder; if needed, follow
    /// up by calling Predict and verifying the recorder's counter moved.
    /// </summary>
    [Fact]
    [Trait("category", "integration-configure-method")]
    public async Task ConfigurePostprocessing_ActionOverload_PipelineSurvivesBuild()
    {
        var recorder = new RecordingTensorTransformer();
        var (features, labels) = MakeMemorizationSet();
        var loader = MakeCanaryLoader(features, labels);
        var model = MakeCanaryModel();

        var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureDataLoader(loader)
            .ConfigurePostprocessing(p => p.Add(recorder))
            .BuildAsync();

        // Postprocessing is applied to the prediction output, not during
        // training. Trigger a prediction and confirm the recorder saw
        // it. A stored-but-not-consumed regression would leave the
        // counters at 0 even after Predict.
        var probe = new Tensor<float>([1, CanaryCtxLen]);
        for (int s = 0; s < CanaryCtxLen; s++) probe[0, s] = features[0, s];
        _ = result.Predict(probe);

        // Postprocessing's contract is that .Transform runs on prediction
        // outputs; if it doesn't fire here, the wiring is broken.
        Assert.True(recorder.TransformCalls > 0,
            $"ConfigurePostprocessing(Action) wired the transformer but result.Predict never invoked it (Transform={recorder.TransformCalls}). Stored-but-not-consumed regression on the postprocessing path.");
    }

    /// <summary>
    /// ConfigurePostprocessing (transformer overload) — same check via the
    /// single-transformer overload.
    /// </summary>
    [Fact]
    [Trait("category", "integration-configure-method")]
    public async Task ConfigurePostprocessing_TransformerOverload_PipelineSurvivesBuild()
    {
        var recorder = new RecordingTensorTransformer();
        var (features, labels) = MakeMemorizationSet();
        var loader = MakeCanaryLoader(features, labels);
        var model = MakeCanaryModel();

        var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureDataLoader(loader)
            .ConfigurePostprocessing(recorder)
            .BuildAsync();

        var probe = new Tensor<float>([1, CanaryCtxLen]);
        for (int s = 0; s < CanaryCtxLen; s++) probe[0, s] = features[0, s];
        _ = result.Predict(probe);

        Assert.True(recorder.TransformCalls > 0,
            $"ConfigurePostprocessing(transformer) wired the transformer but result.Predict never invoked it (Transform={recorder.TransformCalls}).");
    }

    /// <summary>
    /// ConfigurePostprocessing (prebuilt-pipeline overload) — same check via
    /// the prebuilt-pipeline overload.
    /// </summary>
    [Fact]
    [Trait("category", "integration-configure-method")]
    public async Task ConfigurePostprocessing_PipelineOverload_PipelineSurvivesBuild()
    {
        var recorder = new RecordingTensorTransformer();
        var pipeline = new PostprocessingPipeline<float, Tensor<float>, Tensor<float>>();
        pipeline.Add(recorder);

        var (features, labels) = MakeMemorizationSet();
        var loader = MakeCanaryLoader(features, labels);
        var model = MakeCanaryModel();

        var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureDataLoader(loader)
            .ConfigurePostprocessing(pipeline)
            .BuildAsync();

        var probe = new Tensor<float>([1, CanaryCtxLen]);
        for (int s = 0; s < CanaryCtxLen; s++) probe[0, s] = features[0, s];
        _ = result.Predict(probe);

        Assert.True(recorder.TransformCalls > 0,
            $"ConfigurePostprocessing(prebuilt) wired the transformer but result.Predict never invoked it (Transform={recorder.TransformCalls}).");
    }

    /// <summary>
    /// Identity transformer that records every Fit / Transform /
    /// FitTransform call so the test can assert the configure → build path
    /// actually invoked it. The transform is a no-op (returns input as-is)
    /// so the model's training trajectory is undisturbed — the test
    /// screens for wiring, not for transform behavior.
    /// </summary>
    private sealed class RecordingTensorTransformer : IDataTransformer<float, Tensor<float>, Tensor<float>>
    {
        // Counters are written under Interlocked so the recorder is safe
        // to reuse from concurrent Predict paths (e.g. if a future test
        // exercises parallel inference). Without this the counters could
        // race and undercount.
        private int _fitCalls;
        private int _transformCalls;
        private int _fitTransformCalls;
        public int FitCalls => _fitCalls;
        public int TransformCalls => _transformCalls;
        public int FitTransformCalls => _fitTransformCalls;
        public bool IsFitted { get; private set; }
        public int[]? ColumnIndices => null;
        public bool SupportsInverseTransform => false;

        public void Fit(Tensor<float> data) { System.Threading.Interlocked.Increment(ref _fitCalls); IsFitted = true; }

        public Tensor<float> Transform(Tensor<float> data)
        {
            System.Threading.Interlocked.Increment(ref _transformCalls);
            return data;
        }

        public Tensor<float> FitTransform(Tensor<float> data)
        {
            System.Threading.Interlocked.Increment(ref _fitTransformCalls);
            IsFitted = true;
            return data;
        }

        public Tensor<float> InverseTransform(Tensor<float> data) => data;
        public string[] GetFeatureNamesOut(string[]? inputFeatureNames = null) => inputFeatureNames ?? System.Array.Empty<string>();
    }
}
