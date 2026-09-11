using System.Reflection;
using AiDotNet.ActivationFunctions;
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

    private sealed class CompositeBranch : LayerBase<double>
    {
        internal FullyConnectedLayer<double> Dense { get; } = new(4, 1, new IdentityActivation<double>());
        internal CompositeBranch() : base(new[] { 4 }, new[] { 1 }) => RegisterSubLayer(Dense);
        public override bool SupportsTraining => true;
        protected override Tensor<double> ForwardTraced(Tensor<double> input) => Dense.Forward(input);
        public override void ResetState() => Dense.ResetState();
    }
}
