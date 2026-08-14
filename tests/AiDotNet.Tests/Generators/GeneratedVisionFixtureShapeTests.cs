using AiDotNet.Generators;
using Xunit;

namespace AiDotNet.Tests.Generators;

public sealed class GeneratedVisionFixtureShapeTests
{
    [Fact]
    public void MedClipSmokeConstructor_UsesBoundedArchitectureAsFixtureContract()
    {
        const string expression = """
            new AiDotNet.VisionLanguage.Encoders.MedCLIP<double>(
                new AiDotNet.NeuralNetworks.NeuralNetworkArchitecture<double>(
                inputType: AiDotNet.Enums.InputType.ThreeDimensional,
                inputHeight: 32,
                inputWidth: 32,
                inputDepth: 3,
                outputSize: 16),
                new AiDotNet.VisionLanguage.Encoders.MedCLIPOptions
                {
                    ImageSize = 32,
                    PatchSize = 2,
                    VisionEmbeddingDim = 32,
                    ProjectionDim = 16
                })
            """;

        bool found = GeneratedVisionFixtureShape.TryGetExplicitArchitectureSpatialSize(
            expression,
            out int height,
            out int width);

        Assert.True(found);
        Assert.Equal(32, height);
        Assert.Equal(32, width);
    }

    [Fact]
    public void RectangularArchitecture_PreservesIndependentDimensionsAndArgumentOrder()
    {
        const string expression =
            "new Example<double>(new Architecture(inputWidth: 96, inputHeight: 48, outputSize: 4))";

        bool found = GeneratedVisionFixtureShape.TryGetExplicitArchitectureSpatialSize(
            expression,
            out int height,
            out int width);

        Assert.True(found);
        Assert.Equal(48, height);
        Assert.Equal(96, width);
    }

    [Theory]
    [InlineData("")]
    [InlineData("new Example<double>(inputSize: 128)")]
    [InlineData("new Example<double>(inputHeight: 32)")]
    [InlineData("new Example<double>(inputHeight: 0, inputWidth: 32)")]
    public void MissingOrInvalidArchitectureSize_IsNotTreatedAsExplicit(string expression)
    {
        bool found = GeneratedVisionFixtureShape.TryGetExplicitArchitectureSpatialSize(
            expression,
            out _,
            out _);

        Assert.False(found);
    }
}
