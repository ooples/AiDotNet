using AiDotNet.Generators;
using System.Reflection;
using Xunit;

namespace AiDotNet.Tests.Generators;

public sealed class GeneratedVisionFixtureContractTests
{
    [Fact]
    public void BoundedMedClipArchitecture_DrivesItsFixtureDimensions()
    {
        const string expression = """
            new AiDotNet.VisionLanguage.Encoders.MedCLIP<double>(
                new AiDotNet.NeuralNetworks.NeuralNetworkArchitecture<double>(
                    inputType: AiDotNet.Enums.InputType.ThreeDimensional,
                    taskType: AiDotNet.Enums.NeuralNetworkTaskType.Embedding,
                    inputHeight: 32,
                    inputWidth: 32,
                    inputDepth: 3,
                    outputSize: 16),
                new AiDotNet.VisionLanguage.Encoders.MedCLIPOptions
                {
                    ImageSize = 32,
                    VisionEmbeddingDim = 32,
                    ProjectionDim = 16
                })
            """;

        bool found = GeneratedVisionFixtureContract.TryGetArchitectureSpatialSize(
            expression,
            out int height,
            out int width);

        Assert.True(found);
        Assert.Equal(32, height);
        Assert.Equal(32, width);
    }

    [Fact]
    public void RectangularArchitecture_PreservesNamedDimensionOrder()
    {
        const string expression =
            "new Model(new NeuralNetworkArchitecture<double>(inputWidth: 96, inputHeight: 48))";

        bool found = GeneratedVisionFixtureContract.TryGetArchitectureSpatialSize(
            expression,
            out int height,
            out int width);

        Assert.True(found);
        Assert.Equal(48, height);
        Assert.Equal(96, width);
    }

    [Fact]
    public void UnrelatedNamedArguments_DoNotOverrideTheArchitectureContract()
    {
        const string expression = """
            new Wrapper(
                new OtherOptions(inputHeight: 999, inputWidth: 999),
                new NeuralNetworkArchitecture<double>(inputHeight: 24, inputWidth: 40))
            """;

        bool found = GeneratedVisionFixtureContract.TryGetArchitectureSpatialSize(
            expression,
            out int height,
            out int width);

        Assert.True(found);
        Assert.Equal(24, height);
        Assert.Equal(40, width);
    }

    [Theory]
    [InlineData("")]
    [InlineData("new Model(inputHeight: 32, inputWidth: 32)")]
    [InlineData("new Model(new NeuralNetworkArchitecture<double>(inputHeight: 32))")]
    [InlineData("new Model(new NeuralNetworkArchitecture<double>(inputHeight: size, inputWidth: size))")]
    [InlineData("new Model(new NeuralNetworkArchitecture<double>(inputHeight: 0, inputWidth: 32))")]
    public void MissingOrNonLiteralArchitectureSize_UsesTheFamilyFallback(string expression)
    {
        bool found = GeneratedVisionFixtureContract.TryGetArchitectureSpatialSize(
            expression,
            out _,
            out _);

        Assert.False(found);
    }

    [Fact]
    public void TwoDimensionalDeclaration_PreservesFixtureChannelAxis()
    {
        int[] conformed = GeneratedVisionFixtureContract.ConformToDeclaredShape(
            new[] { 3, 128, 128 },
            new[] { 32, 256 });

        Assert.Equal(new[] { 3, 32, 256 }, conformed);
    }

    [Fact]
    public void ThreeDimensionalDeclaration_PreservesFixtureBatchAxis()
    {
        int[] conformed = GeneratedVisionFixtureContract.ConformToDeclaredShape(
            new[] { 1, 3, 256, 256 },
            new[] { 3, 224, 224 });

        Assert.Equal(new[] { 1, 3, 224, 224 }, conformed);
    }

    [Fact]
    public void LargerDeclaration_DoesNotExpandBoundedFixture()
    {
        int[] conformed = GeneratedVisionFixtureContract.ConformToDeclaredShape(
            new[] { 3, 128, 128 },
            new[] { 3, 640, 640 });

        Assert.Equal(new[] { 3, 128, 128 }, conformed);
    }

    [Theory]
    [InlineData(new[] { 0, 32 })]
    [InlineData(new[] { 3, 32, 32, 2 })]
    public void InvalidOrHigherRankDeclaration_UsesIndependentFallback(int[] declared)
    {
        int[] fallback = { 3, 128, 128 };

        int[] conformed = GeneratedVisionFixtureContract.ConformToDeclaredShape(fallback, declared);
        Assert.Equal(fallback, conformed);
        Assert.NotSame(fallback, conformed);

        conformed[0] = 99;

        Assert.Equal(new[] { 3, 128, 128 }, fallback);
    }

    [Fact]
    public void ParameterlessVisionModel_UsesItsRuntimeDeclaredGeometry()
    {
        var fixture = new ModelFamilyTests.Generated.SVTRTests();

        Assert.Equal(new[] { 3, 32, 256 }, ReadInputShape(fixture));
    }

    [Fact]
    public void ParameterlessSquareVisionModels_RemainBounded()
    {
        Assert.Equal(
            new[] { 3, 128, 128 },
            ReadInputShape(new ModelFamilyTests.Generated.PSENetTests()));
    }

    private static int[] ReadInputShape(object fixture)
    {
        PropertyInfo? property = fixture.GetType().GetProperty(
            "InputShape",
            BindingFlags.Instance | BindingFlags.NonPublic);

        Assert.NotNull(property);
        object? value = property.GetValue(fixture);
        return Assert.IsType<int[]>(value);
    }
}
