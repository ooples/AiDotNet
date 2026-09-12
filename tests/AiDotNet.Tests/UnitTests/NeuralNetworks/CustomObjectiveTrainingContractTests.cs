using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

public sealed class CustomObjectiveTrainingContractTests
{
    public CustomObjectiveTrainingContractTests() => TestModuleInitializer.EnsureInitialized();

    [Fact]
    public void FirstObjectiveMaterializesAndUpdatesAnAdditionalLazyBranch()
    {
        using var model = new ObjectiveNetwork();
        var optimizer = new ObservingSgd(model);
        Assert.False(model.Extra.IsShapeResolved);
        model.Step(Input(), Target(), optimizer);
        Assert.True(model.Extra.IsShapeResolved);
        Assert.True(optimizer.ParameterElements >= 5);
        Assert.True(optimizer.NonzeroGradients > 0);
        Assert.True(optimizer.ParameterChanged);
        Assert.Equal(1, model.ForwardCalls);
        Assert.False(model.IsTrainingMode);
    }

    [Fact]
    public void OptimizerReevaluationExecutesTheRealObjectiveAtCurrentWeights()
    {
        using var model = new ObjectiveNetwork();
        var optimizer = new ObservingSgd(model) { Reevaluate = true };
        model.Step(Input(), Target(), optimizer);
        Assert.Equal(2, model.ForwardCalls);
        Assert.True(optimizer.ReevaluationSupported);
        Assert.NotEqual(optimizer.InitialLoss, optimizer.ReevaluatedLoss);
        Assert.True(optimizer.ParameterChanged);
        Assert.False(model.IsTrainingMode);
    }

    [Fact]
    public void ReentrantObjectiveFailsBeforeNestedForwardAndReleasesItsSentinel()
    {
        using var model = new ObjectiveNetwork();
        var optimizer = new ObservingSgd(model);
        model.BeforeForward = () => model.Step(Input(), Target(), optimizer);
        Assert.Throws<InvalidOperationException>(() => model.Step(Input(), Target(), optimizer));
        Assert.False(model.IsTrainingMode);
        Assert.Equal(0, optimizer.Steps);
        model.BeforeForward = null;
        model.Step(Input(), Target(), optimizer);
        Assert.Equal(1, optimizer.Steps);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(2)]
    public void InvalidObjectiveShapeDoesNotMutateWeightsOrLeaveTrainingMode(int length)
    {
        using var model = new ObjectiveNetwork();
        var optimizer = new ObservingSgd(model);
        model.InvalidLossLength = length;
        Assert.Throws<InvalidOperationException>(() => model.Step(Input(), Target(), optimizer));
        Assert.Equal(0, optimizer.Steps);
        Assert.False(model.IsTrainingMode);
        model.InvalidLossLength = null;
        model.Step(Input(), Target(), optimizer);
        Assert.Equal(1, optimizer.Steps);
    }

    [Fact]
    public void SeededAdditionalDropoutBranchHasAnIdenticalTrainingTrajectory()
    {
        using var first = new ObjectiveNetwork(dropout: 0.5);
        using var second = new ObjectiveNetwork(dropout: 0.5);
        _ = first.Extra.Forward(Input());
        _ = second.Extra.Forward(Input());
        second.Extra.SetParameters(first.Extra.GetParameters());
        var firstOptimizer = new ObservingSgd(first);
        var secondOptimizer = new ObservingSgd(second);
        for (int step = 0; step < 3; step++)
        {
            double firstLoss = first.Step(Input(), Target(), firstOptimizer);
            double secondLoss = second.Step(Input(), Target(), secondOptimizer);
            Assert.Equal(firstLoss, secondLoss);
            Assert.Equal(first.Extra.GetParameters().ToArray(), second.Extra.GetParameters().ToArray());
        }
        Assert.NotNull(first.Dropout.RandomSeed);
        Assert.Equal(first.Dropout.RandomSeed, second.Dropout.RandomSeed);
        Assert.Equal(3, firstOptimizer.Steps);
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void NonfiniteObjectiveIsRejectedBeforeOptimizerMutation(double invalid)
    {
        using var model = new ObjectiveNetwork();
        var optimizer = new ObservingSgd(model);
        var target = Target();
        target[0] = invalid;
        Assert.Throws<InvalidOperationException>(() => model.Step(Input(), target, optimizer));
        Assert.Equal(0, optimizer.Steps);
        Assert.False(model.IsTrainingMode);
    }

    [Fact]
    public void AdditionalCompositeBranchTrainsEveryLazyChildExactlyOnce()
    {
        using var model = new CompositeObjectiveNetwork();
        var optimizer = new ObservingSgd(model);
        Assert.Empty(model.Branch.GetTrainableParameters());
        Assert.False(model.Branch.Dense.IsShapeResolved);
        model.Step(Input(), Target(), optimizer);
        Assert.True(model.Branch.Dense.IsShapeResolved);
        Assert.Equal(5, optimizer.ParameterElements);
        Assert.Equal(2, optimizer.ParameterTensors);
        Assert.True(optimizer.NonzeroGradients > 0);
        Assert.True(optimizer.ParameterChanged);
        Assert.False(model.Branch.TrainingMode);
        Assert.False(model.Branch.Dropout.TrainingMode);
    }

    [Fact]
    public void TrainingModeReachesAdditionalCompositeRootsAndTheirChildren()
    {
        using var model = new CompositeObjectiveNetwork();
        model.SetTrainingMode(false);
        Assert.False(model.IsTrainingMode);
        Assert.False(model.Branch.TrainingMode);
        Assert.False(model.Branch.Dropout.TrainingMode);
        var probe = Tensor<double>.CreateDefault(new[] { 2, 4 }, 1.0);
        Assert.Equal(probe.AsSpan().ToArray(), model.Branch.Dropout.Forward(probe).AsSpan().ToArray());
        model.SetTrainingMode(true);
        Assert.True(model.Branch.TrainingMode);
        Assert.True(model.Branch.Dropout.TrainingMode);
    }

    [Fact]
    public void MainLayerAndExtraCompositeAliasHaveOneOptimizerEntryPerTensor()
    {
        using var model = new CompositeObjectiveNetwork();
        model.ShareDenseInMainLayers();
        var optimizer = new ObservingSgd(model);
        model.Step(Input(), Target(), optimizer);
        Assert.Equal(5, optimizer.ParameterElements);
        Assert.Equal(2, optimizer.ParameterTensors);
    }

    [Fact]
    public void CustomParameterSelectionStillFreezesCompositeWeights()
    {
        using var model = new CompositeObjectiveNetwork { TrainBiasOnly = true };
        _ = model.Branch.Forward(Input());
        var weights = model.Branch.Dense.GetTrainableParameters().Single(parameter => parameter.Rank == 2);
        var before = weights.AsSpan().ToArray();
        var optimizer = new ObservingSgd(model);
        model.Step(Input(), Target(), optimizer);
        Assert.Equal(1, optimizer.ParameterElements);
        Assert.Equal(1, optimizer.ParameterTensors);
        Assert.Equal(before, weights.AsSpan().ToArray());
        Assert.True(optimizer.ParameterChanged);
    }

    private static Tensor<double> Input()
    {
        var input = new Tensor<double>(new[] { 2, 4 });
        for (int i = 0; i < input.Length; i++) input[i] = 0.2 + 0.1 * i;
        return input;
    }

    private static Tensor<double> Target() => Tensor<double>.CreateDefault(new[] { 2, 1 }, 0.4);

    private sealed class ObjectiveNetwork : NeuralNetworkBase<double>
    {
        public override bool SupportsTraining => true;
        internal FullyConnectedLayer<double> Extra { get; } = new(1, (AiDotNet.Interfaces.IActivationFunction<double>)new IdentityActivation<double>());
        internal DropoutLayer<double> Dropout { get; }
        internal int ForwardCalls { get; private set; }
        internal Action? BeforeForward { get; set; }
        internal int? InvalidLossLength { get; set; }

        internal ObjectiveNetwork(double dropout = 0) : base(new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputSize: 4, outputSize: 1) { RandomSeed = 9281 }, new MeanSquaredErrorLoss<double>())
        {
            Dropout = new DropoutLayer<double>(dropout);
        }

        protected override void InitializeLayers() { }
        public override AiDotNet.Models.ModelMetadata<double> GetModelMetadata() => new() { Name = nameof(ObjectiveNetwork) };
        protected override IEnumerable<LayerBase<double>?> GetExtraTrainableLayers()
        {
            foreach (var layer in base.GetExtraTrainableLayers()) yield return layer;
            yield return Extra;
            yield return Dropout;
        }

        internal double Step(Tensor<double> input, Tensor<double> target, ObservingSgd optimizer)
            => TrainWithCustomObjective(input, target, (currentInput, currentTarget) =>
            {
                BeforeForward?.Invoke();
                ForwardCalls++;
                var predicted = Dropout.Forward(Extra.Forward(currentInput));
                return InvalidLossLength is int invalid ? new Tensor<double>(new[] { invalid })
                    : new MeanSquaredErrorLoss<double>().ComputeTapeLoss(predicted, currentTarget);
            }, optimizer);
    }

    private sealed class CompositeObjectiveNetwork : NeuralNetworkBase<double>
    {
        internal CompositeBranch Branch { get; } = new();
        internal bool TrainBiasOnly { get; set; }
        public override bool SupportsTraining => true;

        internal CompositeObjectiveNetwork() : base(new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputSize: 4, outputSize: 1) { RandomSeed = 9281 }, new MeanSquaredErrorLoss<double>()) { }

        protected override void InitializeLayers() { }
        internal void ShareDenseInMainLayers() => Layers.Add(Branch.Dense);
        public override AiDotNet.Models.ModelMetadata<double> GetModelMetadata() => new() { Name = nameof(CompositeObjectiveNetwork) };
        protected override IReadOnlyList<Tensor<double>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<double>> parameters)
            => TrainBiasOnly ? parameters.Where(parameter => parameter.Rank == 1).ToArray() : parameters;
        protected override IEnumerable<LayerBase<double>?> GetExtraTrainableLayers()
        {
            foreach (var layer in base.GetExtraTrainableLayers()) yield return layer;
            yield return Branch;
            yield return Branch; // Shared branch identity must not duplicate optimizer updates.
        }

        internal double Step(Tensor<double> input, Tensor<double> target, ObservingSgd optimizer)
            => TrainWithCustomObjective(input, target, (currentInput, currentTarget)
                => new MeanSquaredErrorLoss<double>().ComputeTapeLoss(Branch.Forward(currentInput), currentTarget), optimizer);
    }

    private sealed class CompositeBranch : LayerBase<double>
    {
        internal FullyConnectedLayer<double> Dense { get; } = new(1,
            (AiDotNet.Interfaces.IActivationFunction<double>)new IdentityActivation<double>());
        internal ObservedDropout Dropout { get; } = new();
        internal bool TrainingMode => IsTrainingMode;
        public override bool SupportsTraining => true;

        internal CompositeBranch() : base(new[] { 4 }, new[] { 1 })
        {
            RegisterSubLayer(Dense);
            RegisterSubLayer(Dropout);
        }

        protected override Tensor<double> ForwardTraced(Tensor<double> input) => Dropout.Forward(Dense.Forward(input));
        public override void ResetState() { Dense.ResetState(); Dropout.ResetState(); }
    }

    private sealed class ObservedDropout : DropoutLayer<double>
    {
        internal ObservedDropout() : base(0.25) { }
        internal bool TrainingMode => IsTrainingMode;
    }

    private sealed class ObservingSgd : StochasticGradientDescentOptimizer<double, Tensor<double>, Tensor<double>>
    {
        internal int Steps { get; private set; }
        internal int ParameterElements { get; private set; }
        internal int ParameterTensors { get; private set; }
        internal int NonzeroGradients { get; private set; }
        internal bool ParameterChanged { get; private set; }
        internal bool Reevaluate { get; set; }
        internal bool ReevaluationSupported { get; private set; }
        internal double InitialLoss { get; private set; }
        internal double ReevaluatedLoss { get; private set; }
        internal ObservingSgd(NeuralNetworkBase<double> model) : base(model) { }

        // PR2130 predates the StepCore/no-grad optimizer wrapper on PR2136. Observe its
        // public Step boundary directly; reevaluation still records the real objective.
        public override void Step(TapeStepContext<double> context)
        {
            Steps++;
            ParameterElements = context.Parameters.Sum(parameter => parameter.Length);
            ParameterTensors = context.Parameters.Count;
            NonzeroGradients = context.Gradients.Values.Sum(gradient => gradient.AsSpan().ToArray().Count(value => value != 0));
            Assert.NotEmpty(context.Parameters);
            var first = context.Parameters[0];
            var before = first.AsSpan().ToArray();
            if (Reevaluate)
            {
                ReevaluationSupported = context.SupportsReevaluation;
                InitialLoss = context.Loss;
                first[0] += 0.5;
                ReevaluatedLoss = context.Reevaluate();
            }
            base.Step(context);
            ParameterChanged = !before.SequenceEqual(first.AsSpan().ToArray());
        }
    }
}
