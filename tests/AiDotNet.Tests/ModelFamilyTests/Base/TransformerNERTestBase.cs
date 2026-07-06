using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for transformer-based NER models (BERT-NER, RoBERTa-NER, etc.).
/// Inherits NER invariants and adds transformer-specific: contextual sensitivity
/// and attention-based output variation.
/// </summary>
public abstract class TransformerNERTestBase<T> : NERModelTestBase<T>
{
    [Fact(Timeout = 120000)]
    public virtual async Task ContextualSensitivity_DifferentContext_DifferentLabels()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var network = CreateNetwork();
        // Two CONTENT-distinct contexts. The previous probe used two SPATIALLY
        // CONSTANT inputs (0.3 vs 0.7) that differ only by a global scale, but a
        // BERT-class NER encoder is LayerNorm-first and therefore invariant to a
        // uniform scale/shift of its input — it correctly maps both constants to
        // the same output, so the test was a false positive for "attention is
        // broken". Use two inputs that differ in their PER-POSITION pattern (which
        // survives LayerNorm) so the test exercises genuine contextual sensitivity
        // and still fails loudly if the encoder truly ignores its input.
        //
        // The pattern is driven from BOTH the sequence axis (token position) and the
        // feature axis. A feature-only pattern (i % lastDim) gives every token the
        // SAME row vector, so a model that ignores cross-token attention — reacting
        // only to a single repeated embedding — could still pass. Varying per token
        // index makes each position distinct, so the probe genuinely requires the
        // encoder to attend across the sequence.
        int lastDim = InputShape[InputShape.Length - 1];
        int seqDim = InputShape.Length > 1 ? InputShape[InputShape.Length - 2] : 1;
        var input1 = new Tensor<T>(InputShape);
        var input2 = new Tensor<T>(InputShape);
        for (int i = 0; i < input1.Length; i++)
        {
            int featureIndex = i % lastDim;
            int tokenIndex = (i / lastDim) % seqDim;
            double featurePhase = featureIndex / (double)lastDim;
            double tokenPhase = tokenIndex / (double)System.Math.Max(1, seqDim - 1);
            input1[i] = NumOps.FromDouble(0.2 + 0.3 * tokenPhase + 0.3 * featurePhase);
            input2[i] = NumOps.FromDouble(0.8 - 0.3 * tokenPhase - 0.3 * featurePhase);
        }

        var labels1 = network.Predict(input1);
        var labels2 = network.Predict(input2);

        bool anyDifferent = false;
        int minLen = Math.Min(labels1.Length, labels2.Length);
        for (int i = 0; i < minLen; i++)
        {
            if (Math.Abs(ConvertToDouble(labels1[i]) - ConvertToDouble(labels2[i])) > 1e-12)
            {
                anyDifferent = true;
                break;
            }
        }
        Assert.True(anyDifferent,
            "Transformer NER produces identical labels for different contexts — attention may be broken.");
    }

    [Fact(Timeout = 120000)]
    public async Task Output_ShouldBeFiniteSequence()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var output = network.Predict(input);

        for (int i = 0; i < output.Length; i++)
        {
            double v = ConvertToDouble(output[i]);
            Assert.False(double.IsNaN(v), $"Transformer NER output[{i}] is NaN.");
            Assert.False(double.IsInfinity(v), $"Transformer NER output[{i}] is Infinity.");
        }
    }
}

/// <summary>Double-precision default for <see cref="TransformerNERTestBase{T}"/>.</summary>
public abstract class TransformerNERTestBase : TransformerNERTestBase<double> { }
