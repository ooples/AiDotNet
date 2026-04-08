using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for code/program synthesis neural network models (CodeBERT, CodeT5, etc.).
/// Inherits all NN invariant tests and adds code-specific invariants:
/// output finiteness, different inputs produce different outputs, and empty input handling.
/// </summary>
public abstract class CodeModelTestBase : NeuralNetworkModelTestBase
{
    // =====================================================
    // CODE INVARIANT: Output Should Be Finite
    // Code model outputs (embeddings, token probabilities) must be finite.
    // =====================================================

    [Fact]
    public void CodeOutput_ShouldBeFinite()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var output = network.Predict(input);
        Assert.True(output.Length > 0, "Code model produced empty output.");
        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(output[i]),
                $"Code output[{i}] is NaN — broken code understanding model.");
            Assert.False(double.IsInfinity(output[i]),
                $"Code output[{i}] is Infinity — overflow in code processing.");
        }
    }

    // =====================================================
    // CODE INVARIANT: Different Code → Different Representations
    // Different code inputs should produce different embeddings/outputs.
    // =====================================================

    [Fact]
    public void DifferentCode_DifferentOutputs()
    {
        var network = CreateNetwork();

        var code1 = CreateConstantTensor(InputShape, 0.1);
        var code2 = CreateConstantTensor(InputShape, 0.9);

        var out1 = network.Predict(code1);
        var out2 = network.Predict(code2);

        bool anyDifferent = false;
        int minLen = Math.Min(out1.Length, out2.Length);
        for (int i = 0; i < minLen; i++)
        {
            if (Math.Abs(out1[i] - out2[i]) > 1e-12)
            {
                anyDifferent = true;
                break;
            }
        }
        Assert.True(anyDifferent,
            "Code model produces identical output for different code inputs.");
    }

    // =====================================================
    // CODE INVARIANT: Empty Input Should Not Crash
    // An empty code file is a valid edge case.
    // =====================================================

    [Fact]
    public void EmptyInput_ShouldNotCrash()
    {
        var network = CreateNetwork();
        var emptyInput = CreateConstantTensor(InputShape, 0.0);

        var output = network.Predict(emptyInput);
        Assert.True(output.Length > 0, "Code model produced empty output for empty input.");
        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(output[i]),
                $"Code output[{i}] is NaN for empty code input.");
        }
    }
}
