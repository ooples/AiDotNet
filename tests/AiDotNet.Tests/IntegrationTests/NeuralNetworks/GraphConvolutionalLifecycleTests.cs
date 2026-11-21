using AiDotNet.ActivationFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

[Collection("NonParallelIntegration")]
public sealed class GraphConvolutionalLifecycleTests
{
    [Fact]
    public void TiledAdjacencyReleasesReplacedAndDisposedRegistrationsOnTheirOwningEngines()
    {
        var previous = AiDotNetEngine.Current;
        var first = new TrackingEngine();
        var second = new TrackingEngine();
        try
        {
            AiDotNetEngine.Current = first;
            using var layer = new GraphConvolutionalLayer<double>(2, 2, new IdentityActivation<double>());
            var adjacency = new Tensor<double>(new[] { 3, 3 });
            for (int i = 0; i < 3; i++) adjacency[i, i] = 1;
            layer.SetAdjacencyMatrix(adjacency);
            var input = new Tensor<double>(new[] { 2, 3, 2 });
            layer.Forward(input);
            Assert.Single(first.Constants);
            object old = first.Constants.Single();
            adjacency[0, 1] = 1;
            layer.Forward(input);
            Assert.Single(first.Constants);
            Assert.DoesNotContain(old, first.Constants);
            AiDotNetEngine.Current = second;
            layer.Forward(input);
            Assert.Empty(first.Constants);
            Assert.Single(second.Constants);
            AiDotNetEngine.Current = first;
            layer.Dispose();
            layer.Dispose();
            Assert.Empty(second.Constants);
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    private sealed class TrackingEngine : CpuEngine
    {
        internal HashSet<object> Constants { get; } = new();
        public override void RegisterPersistentTensor<T>(Tensor<T> tensor, PersistentTensorRole role)
        {
            base.RegisterPersistentTensor(tensor, role);
            if (role == PersistentTensorRole.Constant) Constants.Add(tensor);
        }
        public override void UnregisterPersistentTensor<T>(Tensor<T> tensor)
        {
            Constants.Remove(tensor);
            base.UnregisterPersistentTensor(tensor);
        }
    }
}
