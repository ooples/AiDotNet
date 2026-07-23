#nullable disable
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Motion;
using System.IO;
using System.Text;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Video;

public class RAPIDFlowReviewRegressionIntegrationTests
{
    [Fact]
    public void UpdateParameters_WithPartialVector_ThrowsBeforeMutatingLayers()
    {
        var model = CreateModel();
        var original = model.GetParameters();
        var partial = new Vector<double>(original.Length - 1);

        var ex = Assert.Throws<ArgumentException>(() => model.UpdateParameters(partial));

        Assert.Contains("Expected", ex.Message);
        var after = model.GetParameters();
        Assert.Equal(original.Length, after.Length);
        for (int i = 0; i < original.Length; i++)
        {
            Assert.Equal(original[i], after[i]);
        }
    }

    [Fact]
    public void DeserializeNetworkSpecificData_WithLayerCountMismatch_Throws()
    {
        var model = CreateModel();

        var ex = Assert.Throws<InvalidDataException>(() =>
            model.InvokeDeserializeNetworkSpecificData(numRefinementIterations: 2));

        Assert.Contains("RAPIDFlow layers", ex.Message);
    }

    [Fact]
    public void DeserializeNetworkSpecificData_WithLayerTypeMismatch_Throws()
    {
        var model = CreateModel();
        model.Layers[0] = model.Layers[^1];

        var ex = Assert.Throws<InvalidDataException>(() =>
            model.InvokeDeserializeNetworkSpecificData(numRefinementIterations: 1));

        Assert.Contains("Layer 0", ex.Message);
    }

    private static RAPIDFlowProbe CreateModel()
    {
        return new RAPIDFlowProbe(
            new NeuralNetworkArchitecture<double>(
                inputType: InputType.ThreeDimensional,
                taskType: NeuralNetworkTaskType.Regression,
                inputHeight: 32,
                inputWidth: 32,
                inputDepth: 3,
                outputSize: 2),
            numRefinementIterations: 1);
    }

    private sealed class RAPIDFlowProbe : RAPIDFlow<double>
    {
        public RAPIDFlowProbe(
            NeuralNetworkArchitecture<double> architecture,
            int numRefinementIterations)
            : base(architecture, numRefinementIterations)
        {
        }

        public void InvokeDeserializeNetworkSpecificData(int numRefinementIterations)
        {
            using var stream = new MemoryStream();
            using (var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true))
            {
                writer.Write(numRefinementIterations);
            }

            stream.Position = 0;
            using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: false);
            DeserializeNetworkSpecificData(reader);
        }
    }
}
