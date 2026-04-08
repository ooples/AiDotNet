using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for transformer-based NER models (BERT-NER, RoBERTa-NER, etc.).
/// Inherits NER invariants and adds transformer-specific: contextual sensitivity
/// and attention-based output variation.
/// </summary>
public abstract class TransformerNERTestBase : NERModelTestBase
{
    [Fact]
    public void ContextualSensitivity_DifferentContext_DifferentLabels()
    {
        var network = CreateNetwork();
        var input1 = CreateConstantTensor(InputShape, 0.3);
        var input2 = CreateConstantTensor(InputShape, 0.7);

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
            "Transformer NER produces identical labels for different contexts — attention may be broken.");
    }

    [Fact]
    public void Output_ShouldBeFiniteSequence()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var output = network.Predict(input);

        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(output[i]), $"Transformer NER output[{i}] is NaN.");
            Assert.False(double.IsInfinity(output[i]), $"Transformer NER output[{i}] is Infinity.");
        }
    }
}
