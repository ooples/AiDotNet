using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for 3D diffusion models. Inherits latent diffusion invariants
/// and adds 3D-specific: output dimensionality and finite vertex positions.
/// </summary>
public abstract class ThreeDDiffusionTestBase : LatentDiffusionTestBase
{
    [Fact]
    public void Output_ShouldBeNonEmpty()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var input = CreateRandomTensor(InputShape, rng);
        var output = model.Predict(input);
        Assert.True(output.Length > 0, "3D diffusion model produced empty output — no geometry generated.");
    }

    [Fact]
    public void VertexPositions_ShouldBeBounded()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var input = CreateRandomTensor(InputShape, rng);
        var output = model.Predict(input);

        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(output[i]),
                $"3D output[{i}] is NaN — invalid geometry.");
            Assert.True(Math.Abs(output[i]) < 1e6,
                $"3D output[{i}] = {output[i]:E4} is unbounded — unrealistic vertex position.");
        }
    }
}
