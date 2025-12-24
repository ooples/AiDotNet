using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.PhysicsInformed.NeuralOperators;
using Xunit;

namespace AiDotNet.Tests.UnitTests.PhysicsInformed.NeuralOperators;

public class FourierNeuralOperatorTests
{
    [Fact]
    public void FourierNeuralOperator_ForwardPreservesSpatialShape()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 2,
            outputSize: 3);

        var model = new FourierNeuralOperator<double>(
            architecture,
            modes: 2,
            width: 4,
            spatialDimensions: new[] { 4, 4 },
            numLayers: 1);

        var input = new Tensor<double>(new[] { 1, 2, 4, 4 });
        var output = model.Forward(input);

        Assert.Equal(new[] { 1, 3, 4, 4 }, output.Shape);
    }
}
