using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for Named Entity Recognition (NER) neural network models.
/// Inherits all NN invariant tests and adds NER-specific invariants:
/// output sequence length, valid label indices, empty input handling,
/// and different inputs produce different labels.
/// </summary>
public abstract class NERModelTestBase : NeuralNetworkModelTestBase
{
    // =====================================================
    // NER INVARIANT: Output Length Related to Input
    // NER models produce one label per token. Output length
    // should be related to input length (same or proportional).
    // =====================================================

    [Fact]
    public void OutputLength_RelatedToInput()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var output = network.Predict(input);
        Assert.True(output.Length > 0, "NER model produced empty label sequence.");
        // Output should be proportional to input (one label per token at minimum)
        Assert.True(output.Length <= input.Length * 2,
            $"NER output length ({output.Length}) is much larger than input ({input.Length}). " +
            "Label sequence should be proportional to token count.");
    }

    // =====================================================
    // NER INVARIANT: Label Values Should Be Non-Negative
    // NER labels are class indices (O, B-PER, I-PER, etc.).
    // Negative label indices are invalid.
    // =====================================================

    [Fact]
    public void LabelValues_ShouldBeNonNegative()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var output = network.Predict(input);
        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(output[i]),
                $"NER label[{i}] is NaN — broken classification head.");
            Assert.True(output[i] >= -1e-10,
                $"NER label[{i}] = {output[i]:F4} is negative — invalid entity label index.");
        }
    }

    // =====================================================
    // NER INVARIANT: Empty Input Should Not Crash
    // An empty/padding-only input is a valid edge case.
    // =====================================================

    [Fact]
    public void EmptyInput_ShouldNotCrash()
    {
        var network = CreateNetwork();
        var emptyInput = CreateConstantTensor(InputShape, 0.0);

        var output = network.Predict(emptyInput);
        Assert.True(output.Length > 0, "NER model produced empty output for zero input.");
        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(output[i]),
                $"NER label[{i}] is NaN for empty input.");
        }
    }

    // =====================================================
    // NER INVARIANT: Different Inputs → Different Labels
    // Structurally different inputs should potentially produce
    // different entity labels. A model labeling everything
    // the same is degenerate.
    // =====================================================

    [Fact]
    public void DifferentInputs_DifferentLabels()
    {
        var network = CreateNetwork();

        var input1 = CreateConstantTensor(InputShape, 0.1);
        var input2 = CreateConstantTensor(InputShape, 0.9);

        var labels1 = network.Predict(input1);
        var labels2 = network.Predict(input2);

        bool anyDifferent = false;
        int minLen = Math.Min(labels1.Length, labels2.Length);
        for (int i = 0; i < minLen; i++)
        {
            if (Math.Abs(labels1[i] - labels2[i]) > 1e-12)
            {
                anyDifferent = true;
                break;
            }
        }
        Assert.True(anyDifferent,
            "NER model produces identical labels for very different inputs — model may be degenerate.");
    }
}
