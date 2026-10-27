using AiDotNet.Interfaces;
using AiDotNet.Models.Parameters;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
namespace AiDotNet.Tests.UnitTests.Training;

public class FoundationRootRestoreTests
{
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
        Assert.Same(child.Weight, Assert.Single(wrapper.GetParameterChunks()));
        var replacement = new Tensor<double>(new[] { 2 }); replacement[0] = 6;
        wrapper.SetParameterChunks(new[] { replacement });
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
}
