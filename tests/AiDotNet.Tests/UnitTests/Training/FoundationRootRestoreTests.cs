using AiDotNet.Interfaces;
using AiDotNet.Models.Parameters;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
namespace AiDotNet.Tests.UnitTests.Training;

public class FoundationRootRestoreTests
{
    private sealed class RestoreAwareNetwork : AiDotNet.NeuralNetworks.NeuralNetwork<float>
    {
        public int Restored;
        public RestoreAwareNetwork() : base(new AiDotNet.NeuralNetworks.NeuralNetworkArchitecture<float>(
            AiDotNet.Enums.InputType.OneDimensional, AiDotNet.Enums.NeuralNetworkTaskType.Regression,
            inputSize: 2, outputSize: 1, layers: new List<ILayer<float>> { new AiDotNet.NeuralNetworks.Layers.DenseLayer<float>(1) })) { }
        protected override void OnParametersRestored() => Restored++;
    }

    [Fact]
    public void NeuralNetworkChunkRestore_InvokesHookOnlyAfterSuccessfulWrites()
    {
        using var model = new RestoreAwareNetwork();
        var chunks = model.GetParameterChunks().ToArray();
        Assert.NotEmpty(chunks);
        var replacement = chunks.Select(t => new Tensor<float>(Enumerable.Repeat(7f, t.Length).ToArray(), t.Shape.ToArray())).ToArray();
        int before = model.Restored;
        model.SetParameterChunks(replacement);
        Assert.Equal(before + 1, model.Restored);
        Assert.All(model.GetParameterChunks(), t => Assert.All(t.ToArray(), v => Assert.Equal(7f, v)));
        Assert.Throws<ArgumentException>(() => model.SetParameterChunks(Array.Empty<Tensor<float>>()));
        Assert.Equal(before + 1, model.Restored);
    }
    private class TimeSeries : AiDotNet.TimeSeries.TimeSeriesModelBase<float>
    {
        public TimeSeries() : base(new AiDotNet.Models.Options.TimeSeriesRegressionOptions<float>()) { }
        public Tensor<float> Weight { get; } = new(new[] { 2 });
        public int Restored;
        public Vector<float> Packed => ModelParameters;
        protected override void RegisterComponents() => RegisterParameterComponent("weight", new TensorFieldParameterSource<float>(() => Weight));
        protected override void OnParametersRestored() => Restored++;
        public override float PredictSingle(Vector<float> x) => 0;
        public override AiDotNet.Models.ModelMetadata<float> GetModelMetadata() => new();
        protected override IFullModel<float, Matrix<float>, Vector<float>> CreateInstance() => new TimeSeries();
    }
    private sealed class CustomTimeSeries : TimeSeries
    {
        public int Writes;
        public override void SetParameters(Vector<float> values) { Writes++; base.SetParameters(values); }
    }
    [Fact]
    public void TimeSeriesChunkRestore_PreservesCustomSetterAndPackedView()
    {
        var model = new CustomTimeSeries();
        model.SetParameterChunks(new[] { new Tensor<float>(new float[] { 3, 7 }, new[] { 2 }) });
        Assert.Equal(1, model.Writes); Assert.Equal(1, model.Restored);
        Assert.Equal(new float[] { 3, 7 }, model.Packed.ToArray());
    }
    [Fact, Trait("Category", "GPU")]
    public void TimeSeriesChunkRestore_DefersPackedViewWithoutDownloadingDeviceWeights()
    {
        using var backend = new AiDotNet.Tensors.Engines.DirectGpu.OpenCL.OpenClBackend();
        Assert.True(backend.IsAvailable);
        using var direct = new AiDotNet.Tensors.Engines.DirectGpu.DirectGpuEngine(backend);
        using var gpu = new AiDotNet.Tensors.Engines.DirectGpuTensorEngine(direct);
        var previous = AiDotNetEngine.Current; AiDotNetEngine.Current = gpu;
        try
        {
            var model = new TimeSeries();
            var source = gpu.TensorAddScalar(new Tensor<float>(new[] { 2 }), 7f);
            using (var scope = AiDotNet.Tensors.Engines.Diagnostics.GpuResidencyScope.Begin(captureOperations: true))
            {
                model.SetParameterChunks(new[] { source });
                Assert.Equal(0, scope.Downloads);
            }
            Assert.Equal(1, model.Restored);
            Assert.Same(model.Weight, Assert.Single(model.GetParameterChunks()));
            Assert.Equal(new float[] { 7, 7 }, model.Packed.ToArray());
            source = gpu.TensorAddScalar(source, 2f);
            model.SetParameterChunks(new[] { source });
            Assert.Equal(new float[] { 9, 9 }, model.Packed.ToArray());
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    private interface IFixture
    {
        Tensor<double> Weight { get; }
        int Restored { get; }
        IEnumerable<Tensor<double>> GetParameterChunks();
        void SetParameterChunks(IEnumerable<Tensor<double>> chunks);
    }
    [Theory]
    [InlineData("classifier")]
    [InlineData("multilabel")]
    [InlineData("online")]
    [InlineData("clustering")]
    [InlineData("survival")]
    [InlineData("causal")]
    public void IndependentRoots_RestoreLiveTensorsAndInvokeHooks(string family)
    {
        IFixture model = family switch
        {
            "classifier" => new Classifier(), "multilabel" => new MultiLabel(),
            "online" => new Online(), "clustering" => new Clustering(),
            "survival" => new Survival(), "causal" => new Causal(), _ => throw new ArgumentException()
        };
        Assert.Same(model.Weight, Assert.Single(model.GetParameterChunks()));
        var replacement = new Tensor<double>(new[] { 2 }); replacement[0] = 8; replacement[1] = 3;
        model.SetParameterChunks(new[] { replacement });
        Assert.Equal(new double[] { 8, 3 }, model.Weight.ToArray());
        Assert.Same(model.Weight, Assert.Single(model.GetParameterChunks()));
        Assert.Equal(1, model.Restored);
    }
    private class Classifier : AiDotNet.Classification.ClassifierBase<double>, IFixture
    {
        public Tensor<double> Weight { get; } = new(new[] { 2 });
        public int Restored { get; private set; }
        protected override void RegisterComponents() => RegisterParameterComponent("weight", new TensorFieldParameterSource<double>(() => Weight));
        protected override void OnParametersRestored() => Restored++;
        public override void Train(Matrix<double> x, Vector<double> y) { }
        public override Vector<double> Predict(Matrix<double> x) => new(x.Rows);
    }
    private class ExplicitStateClassifier : Classifier
    {
        public override Vector<double> GetParameters() => base.GetParameters();
        public override IEnumerable<ParameterChunk<double>> GetParameterStateChunks()
            => base.GetParameterStateChunks();
    }
    private sealed class FlatOnlyDerivedClassifier : ExplicitStateClassifier
    {
        public override Vector<double> GetParameters() => new(new double[] { Weight[1], Weight[0] });
    }
    [Fact]
    public void ExplicitStateOverride_PreservesLiveTrainableTensors()
    {
        var model = new ExplicitStateClassifier();
        var chunk = Assert.Single(model.GetParameterStateChunks());
        Assert.True(chunk.IsWritableInPlace);
        Assert.Same(model.Weight, chunk.Tensor);
        chunk.Tensor[0] = 7;
        Assert.Equal(7, model.GetParameters()[0]);
    }
    [Fact]
    public void MoreDerivedFlatOnlyOverride_PreservesCustomOrderAndSnapshotIsolation()
    {
        var model = new FlatOnlyDerivedClassifier(); model.Weight[0] = 3; model.Weight[1] = 7;
        var chunk = Assert.Single(model.GetParameterStateChunks());
        Assert.False(chunk.IsWritableInPlace);
        Assert.Equal(new double[] { 7, 3 }, chunk.Tensor.ToArray());
        chunk.Tensor[0] = 9;
        Assert.Equal(7, model.Weight[1]);
    }

    private sealed class MultiLabel : AiDotNet.Classification.MultiLabel.MultiLabelClassifierBase<double>, IFixture
    {
        public Tensor<double> Weight { get; } = new(new[] { 2 });
        public int Restored { get; private set; }
        protected override void RegisterComponents() => RegisterParameterComponent("weight", new TensorFieldParameterSource<double>(() => Weight));
        protected override void OnParametersRestored() => Restored++;
        protected override void TrainMultiLabelCore(Matrix<double> x, Matrix<double> y) { }
        public override Matrix<double> PredictMultiLabelProbabilities(Matrix<double> x) => new(x.Rows, 1);
    }
    private sealed class Online : AiDotNet.OnlineLearning.OnlineLearningModelBase<double>, IFixture
    {
        public Tensor<double> Weight { get; } = new(new[] { 2 });
        public int Restored { get; private set; }
        protected override void RegisterComponents() => RegisterParameterComponent("weight", new TensorFieldParameterSource<double>(() => Weight));
        protected override void OnParametersRestored() => Restored++;
        public override void PartialFit(Vector<double> x, double y) { }
        public override double PredictSingle(Vector<double> x) => 0;
        public override IFullModel<double, Matrix<double>, Vector<double>> WithParameters(Vector<double> p) => throw new NotSupportedException();
    }
    private sealed class Clustering : AiDotNet.Clustering.Base.ClusteringBase<double>, IFixture
    {
        public Tensor<double> Weight { get; } = new(new[] { 2 });
        public int Restored { get; private set; }
        protected override void RegisterComponents() => RegisterParameterComponent("weight", new TensorFieldParameterSource<double>(() => Weight));
        protected override void OnParametersRestored() => Restored++;
        public override void Train(Matrix<double> x, Vector<double> y) { }
        public override Vector<double> Predict(Matrix<double> x) => new(x.Rows);
        public override IFullModel<double, Matrix<double>, Vector<double>> WithParameters(Vector<double> p) => throw new NotSupportedException();
    }
    private sealed class Survival : AiDotNet.SurvivalAnalysis.SurvivalModelBase<double>, IFixture
    {
        public Tensor<double> Weight { get; } = new(new[] { 2 });
        public int Restored { get; private set; }
        protected override void RegisterComponents() => RegisterParameterComponent("weight", new TensorFieldParameterSource<double>(() => Weight));
        protected override void OnParametersRestored() => Restored++;
        protected override void FitSurvivalCore(Matrix<double> x, Vector<double> t, Vector<int> e) { }
        public override Matrix<double> PredictSurvivalProbability(Matrix<double> x, Vector<double> t) => new(x.Rows,t.Length);
        public override Vector<double> PredictHazardRatio(Matrix<double> x) => new(x.Rows);
        public override Vector<double> GetBaselineSurvival(Vector<double> t) => new(t.Length);
        public override Vector<double> Predict(Matrix<double> x) => new(x.Rows);
        public override IFullModel<double, Matrix<double>, Vector<double>> WithParameters(Vector<double> p) => throw new NotSupportedException();
    }
    private sealed class Causal : AiDotNet.CausalInference.CausalModelBase<double>, IFixture
    {
        public Tensor<double> Weight { get; } = new(new[] { 2 });
        public int Restored { get; private set; }
        protected override void RegisterComponents() => RegisterParameterComponent("weight", new TensorFieldParameterSource<double>(() => Weight));
        protected override void OnParametersRestored() => Restored++;
        public override void Fit(Matrix<double> x, Vector<double> t, Vector<double> y) { }
        public override Vector<double> EstimateTreatmentEffect(Matrix<double> x) => new(x.Rows);
        public override Vector<double> PredictTreated(Matrix<double> x) => new(x.Rows);
        public override Vector<double> PredictControl(Matrix<double> x) => new(x.Rows);
        public override Vector<double> PredictTreatmentEffect(Matrix<double> x) => new(x.Rows);
        public override Vector<double> Predict(Matrix<double> x) => new(x.Rows);
        protected override Vector<double> EstimatePropensityScoresCore(Matrix<double> x) => new(x.Rows);
        public override (double estimate, double standardError) EstimateATE(Matrix<double> x, Vector<int> t, Vector<double> y) => (0,0);
        public override (double estimate, double standardError) EstimateATT(Matrix<double> x, Vector<int> t, Vector<double> y) => (0,0);
        public override Vector<double> EstimateCATEPerIndividual(Matrix<double> x, Vector<int> t, Vector<double> y) => new(x.Rows);
        public override IFullModel<double, Matrix<double>, Vector<double>> WithParameters(Vector<double> p) => throw new NotSupportedException();
    }
    private sealed class StreamingClassifier : Classifier
    {
        public int Reads, Writes;
        public override Vector<double> GetParameters() => throw new InvalidOperationException("Unexpected flattening");
        public override void SetParameters(Vector<double> p) => throw new InvalidOperationException("Unexpected flattening");
        public override IEnumerable<Tensor<double>> GetParameterChunks() { Reads++; return new[] { Weight }; }
        public override void SetParameterChunks(IEnumerable<Tensor<double>> chunks)
        { Writes++; AiDotNet.Training.TensorCheckpoint<double>.CopyValues(chunks.Single(), Weight); }
    }
    private sealed class Wrapper : AiDotNet.Models.ModelWrapperBase<double, Matrix<double>, Vector<double>>
    {
        public Wrapper(IFullModel<double, Matrix<double>, Vector<double>> model) : base(model) { }
        public override Vector<double> Predict(Matrix<double> x) => new(x.Rows);
        public override IFullModel<double, Matrix<double>, Vector<double>> WithParameters(Vector<double> p) => throw new NotSupportedException();
    }
    private sealed class AutoML : AiDotNet.AutoML.AutoMLModelBase<double, Matrix<double>, Vector<double>>
    {
        public AutoML(IFullModel<double, Matrix<double>, Vector<double>> model) { BestModel = model; }
        public override Task<IFullModel<double, Matrix<double>, Vector<double>>> SearchAsync(Matrix<double> x, Vector<double> y,
            Matrix<double> vx, Vector<double> vy, TimeSpan limit, CancellationToken cancellationToken = default) => throw new NotSupportedException();
        public override Task<Dictionary<string, object>> SuggestNextTrialAsync() => throw new NotSupportedException();
        protected override Task<IFullModel<double, Matrix<double>, Vector<double>>> CreateModelAsync(Type type, Dictionary<string, object> p) => throw new NotSupportedException();
        protected override Dictionary<string, AiDotNet.AutoML.ParameterRange> GetDefaultSearchSpace(Type type) => new();
    }
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Wrappers_ForwardStreamingContractsWithoutFlattening(bool autoML)
    {
        var child = new StreamingClassifier();
        IParameterizable<double, Matrix<double>, Vector<double>> wrapper = autoML ? new AutoML(child) : new Wrapper(child);
        var chunks = autoML ? ((AutoML)wrapper).GetParameterChunks() : ((Wrapper)wrapper).GetParameterChunks();
        Assert.Same(child.Weight, Assert.Single(chunks));
        var replacement = new Tensor<double>(new[] { 2 }); replacement[0] = 6;
        if (autoML) ((AutoML)wrapper).SetParameterChunks(new[] { replacement });
        else ((Wrapper)wrapper).SetParameterChunks(new[] { replacement });
        Assert.Equal(6, child.Weight[0]);
        Assert.Equal(1, child.Reads); Assert.Equal(1, child.Writes);
    }
    private sealed class CustomSetterClassifier : Classifier
    {
        public int SetterCalls;
        public override void SetParameters(Vector<double> values) { SetterCalls++; base.SetParameters(values); }
    }
    [Fact]
    public void BestModelTransfer_PreservesMoreSpecificSetter()
    {
        var source = new Classifier(); source.Weight[0] = 12;
        var destination = new CustomSetterClassifier();
        ParameterTransfer.Copy<double, Matrix<double>, Vector<double>>(source, destination);
        Assert.Equal(1, destination.SetterCalls);
        Assert.Equal(12, destination.Weight[0]);
    }
    [Fact]
    public void BestModelTransfer_UsesExplicitStreamingImplementation()
    {
        var source = new StreamingClassifier(); source.Weight[0] = 12;
        var destination = new StreamingClassifier();
        ParameterTransfer.Copy<double, Matrix<double>, Vector<double>>(source, destination);
        Assert.Equal(1, source.Reads); Assert.Equal(1, destination.Writes);
        Assert.Equal(12, destination.Weight[0]);
    }

    [Fact]
    public void ModelCheckpoint_ReusesSnapshotAndPreservesCustomSetter()
    {
        var model = new CustomSetterClassifier(); model.Weight[0] = 12;
        var checkpoint = new AiDotNet.Training.ModelParameterCheckpoint<double, Matrix<double>, Vector<double>>(model);
        checkpoint.Capture(); model.Weight[0] = 3; checkpoint.Restore();
        Assert.Equal(12, model.Weight[0]); Assert.Equal(1, model.SetterCalls);
        model.Weight[0] = 8; checkpoint.Capture(); model.Weight[0] = 1; checkpoint.Restore();
        Assert.Equal(8, model.Weight[0]); Assert.Equal(2, model.SetterCalls);
    }

    [Fact]
    public void AdaptedPrediction_RestoresStreamingModelAfterException()
    {
        var model = new ThrowingClassifier(); model.Weight[0] = 12;
        var adapted = new AiDotNet.MetaLearning.Algorithms.AdaptedMetaModel<double, Matrix<double>, Vector<double>>(
            model, new Vector<double>(new double[] { 4, 5 }));
        Assert.Throws<InvalidOperationException>(() => adapted.Predict(new Matrix<double>(1, 1)));
        Assert.Equal(12, model.Weight[0]);
    }
    private sealed class ThrowingClassifier : Classifier
    {
        public override Vector<double> Predict(Matrix<double> x) => throw new InvalidOperationException("prediction failed");
    }

    private sealed class LayerClassifier : AiDotNet.Classification.ClassifierBase<double>
    {
        internal readonly AiDotNet.NeuralNetworks.Layers.DenseLayer<double> Layer = new(2);
        public LayerClassifier() => Layer.Forward(new Tensor<double>(new[] { 1, 3 }));
        internal int Restored;
        protected override void RegisterComponents() => RegisterParameterComponent("layer", Layer);
        protected override void OnParametersRestored() => Restored++;
        public override void Train(Matrix<double> x, Vector<double> y) { }
        public override Vector<double> Predict(Matrix<double> x) => new(x.Rows);
    }

    [Fact]
    public void RegisteredLayerModel_CheckpointRestoresLiveStateAndInvokesModelHook()
    {
        var model = new LayerClassifier();
        var chunks = model.GetParameterChunks().ToArray();
        Assert.NotEmpty(chunks); chunks[0][0] = 12;
        var checkpoint = new AiDotNet.Training.ModelParameterCheckpoint<double, Matrix<double>, Vector<double>>(model);
        checkpoint.Capture(); chunks[0][0] = 8; checkpoint.Restore();
        Assert.Same(chunks[0], model.GetParameterChunks().First());
        Assert.Equal(12, chunks[0][0]); Assert.Equal(1, model.Restored);
    }

    [Theory]
    [InlineData("classifier")] [InlineData("multilabel")] [InlineData("online")]
    [InlineData("clustering")] [InlineData("survival")] [InlineData("causal")]
    public void NestedModelRestore_UsesChildChunkContractAndRunsChildHook(string family)
    {
        IFixture child = family switch
        {
            "classifier" => new Classifier(), "multilabel" => new MultiLabel(), "online" => new Online(),
            "clustering" => new Clustering(), "survival" => new Survival(), "causal" => new Causal(),
            _ => throw new ArgumentException()
        };
        var registry = new ParameterComponentRegistry<double>();
        registry.Register("children", new ComponentCollectionParameterSource<double>(
            () => new[] { (IParameterSource<double>)child }));
        var replacement = new Tensor<double>(new[] { 2 }); replacement[0] = 7;
        Assert.True(registry.TrySetParameterChunks(new[] { replacement }));
        Assert.Equal(7, child.Weight[0]); Assert.Equal(1, child.Restored);
    }

    [Fact]
    public void NestedRestore_SnapshotsAliasesBeforeAnyChildWrite()
    {
        var own = new Tensor<double>(new[] { 2 }); own[0] = 3;
        var child = new Classifier(); child.Weight[0] = 7;
        var registry = new ParameterComponentRegistry<double>();
        registry.Register("a", new TensorFieldParameterSource<double>(() => own));
        registry.Register("child", child);
        Assert.False(registry.TrySetParameterChunks(new[] { child.Weight, new Tensor<double>(new[] { 3 }) }));
        Assert.Equal(3, own[0]); Assert.Equal(0, child.Restored);
        Assert.True(registry.TrySetParameterChunks(new[] { child.Weight, own }));
        Assert.Equal(7, own[0]); Assert.Equal(3, child.Weight[0]); Assert.Equal(1, child.Restored);
    }

    [Fact]
    public void NestedRestore_DoesNotBypassMoreSpecificChildSetter()
    {
        var child = new CustomSetterClassifier();
        var registry = new ParameterComponentRegistry<double>(); registry.Register("child", child);
        Assert.False(registry.TrySetParameterChunks(new[] { child.Weight }));
        registry.SetParameters(new Vector<double>(new double[] { 8, 9 }));
        Assert.Equal(1, child.SetterCalls); Assert.Equal(8, child.Weight[0]);
    }
}
