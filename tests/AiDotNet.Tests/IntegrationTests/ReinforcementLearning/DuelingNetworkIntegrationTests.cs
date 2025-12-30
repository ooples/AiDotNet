using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Agents.DuelingDQN;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ReinforcementLearning;

[Collection("NonParallelIntegration")]
public class DuelingNetworkIntegrationTests
{
    [Fact]
    public void DuelingNetwork_ForwardBackwardAndSerialization_Work()
    {
        var numOps = MathHelper.GetNumericOperations<double>();
        var network = new DuelingNetwork<double>(
            stateSize: 2,
            actionSize: 3,
            sharedLayerSizes: new[] { 4 },
            valueLayerSizes: new[] { 4 },
            advantageLayerSizes: new[] { 4 },
            numOps);

        var state = new Vector<double>(2);
        state[0] = 0.1;
        state[1] = 0.2;

        var qValues = network.Forward(state);

        Assert.Equal(3, qValues.Length);
        for (int i = 0; i < qValues.Length; i++)
        {
            Assert.False(double.IsNaN(qValues[i]));
            Assert.False(double.IsInfinity(qValues[i]));
        }

        var gradients = new Vector<double>(3);
        gradients[0] = 0.1;
        gradients[1] = -0.2;
        gradients[2] = 0.05;

        network.Backward(state, gradients);
        network.UpdateWeights(0.01);

        var parameters = network.GetFlattenedParameters();
        network.SetFlattenedParameters(parameters);

        var data = network.Serialize();
        network.Deserialize(data);
    }
}
