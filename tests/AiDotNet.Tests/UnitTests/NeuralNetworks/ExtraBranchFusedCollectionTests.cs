using System.Reflection;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

public sealed class ExtraBranchFusedCollectionTests
{
    public ExtraBranchFusedCollectionTests() => TestModuleInitializer.EnsureInitialized();

    [Fact]
    public void FusedExtraCollectorIncludesRegisteredChildrenExactlyOnce()
    {
        using var model = new ExtraNetwork();
        var method = typeof(NeuralNetworkBase<double>).GetMethod("CollectFusedExtraTrainableTensors",
            BindingFlags.Instance | BindingFlags.NonPublic);
        if (method is null) throw new Xunit.Sdk.XunitException("The production fused extra collector was not found.");
        var parameters = Assert.IsAssignableFrom<IReadOnlyList<Tensor<double>>>(method.Invoke(model, null));
        Assert.Equal(2, parameters.Count);
        Assert.Equal(5, parameters.Sum(parameter => parameter.Length));
        Assert.All(parameters, parameter => Assert.Contains(model.Branch.Dense.GetTrainableParameters(), candidate => ReferenceEquals(candidate, parameter)));
    }

    // [Batch, Features] in and out: one dense 4 -> 1 branch over a single feature axis.
    [TensorLayout(TensorAxis.Batch, TensorAxis.Features,
        BatchOptional = true, Direction = TensorLayoutDirection.Input)]
    [TensorLayout(TensorAxis.Batch, TensorAxis.Features,
        BatchOptional = true, Direction = TensorLayoutDirection.Output)]
    private sealed class ExtraNetwork : NeuralNetworkBase<double>
    {
        internal CompositeBranch Branch { get; } = new();
        internal ExtraNetwork() : base(new NeuralNetworkArchitecture<double>(inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression, inputSize: 4, outputSize: 1), new MeanSquaredErrorLoss<double>()) { }
        protected override void InitializeLayers() { }
        public override AiDotNet.Models.ModelMetadata<double> GetModelMetadata() => new() { Name = nameof(ExtraNetwork) };
        protected override IEnumerable<LayerBase<double>?> GetExtraTrainableLayers()
        {
            foreach (var layer in base.GetExtraTrainableLayers()) yield return layer;
            yield return Branch;
            yield return Branch;
        }
    }

    // 4 -> 1, so not shape-preserving: the axis roles are declared here and the output width comes from
    // OutputShape, the way FullyConnectedLayer states the same relation.
    [TensorLayout(TensorAxis.Batch, TensorAxis.Features,
        BatchOptional = true, Direction = TensorLayoutDirection.Input)]
    [TensorLayout(TensorAxis.Batch, TensorAxis.Features,
        BatchOptional = true, Direction = TensorLayoutDirection.Output)]
    private sealed class CompositeBranch : LayerBase<double>, IShapeContract
    {
        internal FullyConnectedLayer<double> Dense { get; } = new(4, 1, new IdentityActivation<double>());
        internal CompositeBranch() : base(new[] { 4 }, new[] { 1 }) => RegisterSubLayer(Dense);
        public override bool SupportsTraining => true;
        protected override Tensor<double> ForwardTraced(Tensor<double> input) => Dense.Forward(input);
        public override void ResetState() => Dense.ResetState();

        /// <inheritdoc />
        public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
        {
            int outputSize = OutputShape.Length > 0 ? OutputShape[0] : -1;
            if (outputSize <= 0) return null;

            var features = new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(outputSize));
            return inputRank switch
            {
                1 => new[] { features },
                2 => new[] { new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)), features },
                _ => null,
            };
        }
    }
}
