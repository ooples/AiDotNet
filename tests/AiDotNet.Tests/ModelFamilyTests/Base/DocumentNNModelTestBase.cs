using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for document neural network models (OCR, layout, document understanding).
/// Inherits all NN invariant tests and adds document-specific invariants:
/// empty input handling, output consistency, structural sensitivity, and scaling robustness.
/// </summary>
public abstract class DocumentNNModelTestBase : NeuralNetworkModelTestBase
{
    // =====================================================
    // DOCUMENT INVARIANT: Empty Input Should Not Crash
    // A blank/empty document is a valid edge case. The model
    // should produce finite output, not NaN or exceptions.
    // =====================================================

    [Fact]
    public void EmptyInput_ShouldNotCrash()
    {
        var network = CreateNetwork();
        var emptyInput = CreateConstantTensor(InputShape, 0.0);

        var output = network.Predict(emptyInput);
        Assert.True(output.Length > 0, "Output should not be empty for blank document input.");
        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(output[i]),
                $"Output[{i}] is NaN for blank document — model should handle empty input.");
            Assert.False(double.IsInfinity(output[i]),
                $"Output[{i}] is Infinity for blank document.");
        }
    }

    // =====================================================
    // DOCUMENT INVARIANT: Output Dimensionality Consistency
    // Same input shape should always produce same output shape.
    // =====================================================

    [Fact]
    public void OutputDimensionality_Consistent()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input1 = CreateRandomTensor(InputShape, rng);
        var input2 = CreateRandomTensor(InputShape, ModelTestHelpers.CreateSeededRandom(99));

        var output1 = network.Predict(input1);
        var output2 = network.Predict(input2);

        Assert.Equal(output1.Length, output2.Length);
    }

    // =====================================================
    // DOCUMENT INVARIANT: Different Documents → Different Outputs
    // Structurally different document inputs should produce
    // different representations. A model ignoring content is broken.
    // =====================================================

    [Fact]
    public void DifferentDocuments_DifferentOutputs()
    {
        var network = CreateNetwork();

        var doc1 = CreateConstantTensor(InputShape, 0.2);
        var doc2 = CreateConstantTensor(InputShape, 0.8);

        var output1 = network.Predict(doc1);
        var output2 = network.Predict(doc2);

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
            "Document model produces identical output for different inputs — content is being ignored.");
    }

    // =====================================================
    // DOCUMENT INVARIANT: Larger Input Should Not Explode
    // Doubling input values should not cause overflow.
    // =====================================================

    [Fact]
    public void LargerInput_ShouldNotExplode()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();

        var input = CreateRandomTensor(InputShape, rng);
        var largeInput = new Tensor<double>(InputShape);
        for (int i = 0; i < input.Length; i++)
            largeInput[i] = input[i] * 2.0;

        var output = network.Predict(largeInput);
        for (int i = 0; i < output.Length; i++)
        {
            Assert.True(!double.IsNaN(output[i]) && !double.IsInfinity(output[i]),
                $"Output[{i}] is not finite for 2x scaled input — numerical instability.");
        }
    }
}
