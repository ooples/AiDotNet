using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

public class LossCompatibleTargetProjectionTests
{
    [Fact]
    public void CrossEntropyTargets_AreOneHotAlongTheClassAxis()
    {
        var probe = new ProjectionProbe();
        using var network = probe.CreateNetwork(new CrossEntropyWithLogitsLoss<float>());
        using var target = new Tensor<float>([2, 3]);

        using var projected = probe.Project(network, target);

        for (int row = 0; row < 2; row++)
        {
            var values = Enumerable.Range(0, 3).Select(column => projected[row, column]).ToArray();
            Assert.Equal(1.0f, values.Sum(), 6);
            Assert.Single(values, value => value == 1.0f);
            Assert.All(values, value => Assert.True(value is 0.0f or 1.0f));
        }
    }

    [Fact]
    public void BinaryCrossEntropyTargets_AreClampedToTheProbabilityDomain()
    {
        var probe = new ProjectionProbe();
        using var network = probe.CreateNetwork(new BinaryCrossEntropyWithLogitsLoss<float>());
        using var target = new Tensor<float>([3]);
        target[0] = -2.0f;
        target[1] = 0.25f;
        target[2] = 2.0f;

        using var projected = probe.Project(network, target);

        Assert.Equal(new[] { 0.0f, 0.25f, 1.0f }, projected.ToArray());
    }

    [Fact]
    public void BornRuleTargets_AreProjectedOntoTheProbabilitySimplex()
    {
        var probe = new ProjectionProbe();
        using var network = probe.CreateBornRuleNetwork();
        using var target = new Tensor<float>([3]);
        target[0] = -2.0f;
        target[1] = 1.0f;
        target[2] = -1.0f;

        using var projected = probe.Project(network, target);

        Assert.Equal(0.5f, projected[0], 6);
        Assert.Equal(0.25f, projected[1], 6);
        Assert.Equal(0.25f, projected[2], 6);
        Assert.Equal(1.0f, projected.ToArray().Sum(), 6);
    }

    [Fact]
    public void ZeroBornRuleTarget_UsesAUniformDistribution()
    {
        var probe = new ProjectionProbe();
        using var network = probe.CreateBornRuleNetwork();
        using var target = new Tensor<float>([4]);

        using var projected = probe.Project(network, target);

        Assert.All(projected.ToArray(), value => Assert.Equal(0.25f, value, 6));
    }

    [Fact]
    public void MaximumMagnitudeBornRuleTargets_RemainAProbabilityDistribution()
    {
        var probe = new ProjectionProbe();
        using var network = probe.CreateBornRuleNetwork();
        using var target = new Tensor<float>([3]);
        target[0] = float.MaxValue;
        target[1] = -float.MaxValue;

        using var projected = probe.Project(network, target);

        Assert.Equal(0.5f, projected[0], 6);
        Assert.Equal(0.5f, projected[1], 6);
        Assert.Equal(0.0f, projected[2], 6);
        Assert.Equal(1.0f, projected.ToArray().Sum(), 6);
    }

    private sealed class ProjectionProbe : NeuralNetworkModelTestBase<float>
    {
        // This harness deliberately exercises a dense categorical public target even though its
        // minimal architecture is otherwise tagged as regression. The shared base now distinguishes
        // public target encoding from an internal loss, so state the test's target contract explicitly.
        protected override ExternalTargetEncodingKind ExternalTargetEncoding
            => ExternalTargetEncodingKind.DenseClassProbabilities;

        protected override INeuralNetworkModel<float> CreateNetwork()
            => CreateBornRuleNetwork();

        public INeuralNetworkModel<float> CreateBornRuleNetwork()
            => CreateNetwork(new BornRuleMseLoss<float>());

        public INeuralNetworkModel<float> CreateNetwork(ILossFunction<float> lossFunction)
        {
            var architecture = new NeuralNetworkArchitecture<float>(
                inputType: InputType.OneDimensional,
                taskType: NeuralNetworkTaskType.Regression,
                inputSize: 3,
                outputSize: 3);
            return new NeuralNetwork<float>(architecture, lossFunction: lossFunction);
        }

        public Tensor<float> Project(INeuralNetworkModel<float> network, Tensor<float> target)
            => MakeTargetWellPosedForLoss(network, target, new Random(1));
    }
}
