using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for GAN (Generative Adversarial Network) models.
/// Inherits all neural network invariant tests and adds GAN-specific invariants:
/// generator output shape, mode diversity, parameter count, and output range.
/// </summary>
public abstract class GANModelTestBase : NeuralNetworkModelTestBase
{
    // =====================================================
    // GAN INVARIANT: Generator Output Shape
    // The generator should produce output matching the declared OutputShape.
    // =====================================================

    [Fact]
    public void GeneratorOutput_ShouldHaveCorrectShape()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var output = network.Predict(input);

        int expectedLength = 1;
        foreach (var dim in OutputShape)
            expectedLength *= dim;

        Assert.Equal(expectedLength, output.Length);
    }

    // =====================================================
    // GAN INVARIANT: Mode Diversity
    // Different latent inputs should produce different outputs.
    // A GAN that produces the same output regardless of input has
    // mode-collapsed — a fundamental GAN failure mode.
    // =====================================================

    [Fact]
    public void DifferentLatentInputs_ProduceDifferentOutputs()
    {
        var network = CreateNetwork();

        var input1 = CreateRandomTensor(InputShape, ModelTestHelpers.CreateSeededRandom(1));
        var input2 = CreateRandomTensor(InputShape, ModelTestHelpers.CreateSeededRandom(2));

        var output1 = network.Predict(input1);
        var output2 = network.Predict(input2);

        bool anyDifferent = false;
        int minLen = Math.Min(output1.Length, output2.Length);
        for (int i = 0; i < minLen; i++)
        {
            if (Math.Abs(output1[i] - output2[i]) > 1e-12)
            {
                anyDifferent = true;
                break;
            }
        }
        Assert.True(anyDifferent,
            "GAN produces identical output for different latent inputs — possible mode collapse.");
    }

    // =====================================================
    // GAN INVARIANT: Parameter Count
    // Real GANs should have a substantial number of parameters
    // (generator + discriminator). A model with very few parameters
    // is likely misconfigured.
    // =====================================================

    [Fact]
    public void ParameterCount_ShouldBeSubstantial()
    {
        var network = CreateNetwork();
        var parameters = network.GetParameters();
        Assert.True(parameters.Length > 100,
            $"GAN has only {parameters.Length} parameters — expected >100 for a real GAN architecture.");
    }

    // =====================================================
    // GAN INVARIANT: Output Values in Reasonable Range
    // Generated output values should not be extreme (no exploding values).
    // =====================================================

    [Fact]
    public void OutputValues_ShouldBeInReasonableRange()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var output = network.Predict(input);

        for (int i = 0; i < output.Length; i++)
        {
            Assert.True(Math.Abs(output[i]) < 1e6,
                $"Output[{i}] = {output[i]:E4} is out of reasonable range [-1e6, 1e6]. " +
                "Generator may have exploding values.");
        }
    }
}
