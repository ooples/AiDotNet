using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.PhysicsInformed.NeuralOperators;
using AiDotNet.PhysicsInformed.Options;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.UnitTests.PhysicsInformed.NeuralOperators;

public class FourierNeuralOperatorTests
{
    [Fact(Timeout = 60000)]
    public async Task FourierNeuralOperator_ForwardPreservesSpatialShape()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 2,
            outputSize: 3);

        var model = new FourierNeuralOperator<double>(
            architecture,
            spatialDimensions: new[] { 4, 4 },
            options: new FourierNeuralOperatorOptions { Modes = 2, Width = 4, NumLayers = 1 });

        var input = new Tensor<double>(new[] { 1, 2, 4, 4 });
        var output = model.Forward(input);

        Assert.Equal(new[] { 1, 3, 4, 4 }, output.Shape.ToArray());
    }
}
