using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for Named Entity Recognition (NER) neural network models.
/// Inherits all NN invariant tests and adds NER-specific invariants:
/// output sequence length, valid label indices, empty input handling,
/// and different inputs produce different labels.
/// </summary>
public abstract class NERModelTestBase : NeuralNetworkModelTestBase
{
    /// <summary>
    /// NER labels are categorical class indices (O, B-PER, I-PER, ...) per the
    /// CoNLL-2003 / OntoNotes convention used by the BERT-NER family (Devlin
    /// et al. 2019 §3, Sanh et al. 2019 DistilBERT §3, Jiao et al. 2020
    /// TinyBERT §3). CrossEntropyLoss expects targets to be either one-hot
    /// or integer class indices; the default continuous-uniform target from
    /// <see cref="NeuralNetworkModelTestBase.CreateRandomTargetTensor"/> hits
    /// "Target value 0.6476... is not an integer class index". Generate
    /// rank-1 [seq] integer class IDs in [0, numLabels-1] (numLabels inferred
    /// from the predicted shape's last axis when the warmup succeeds; falls
    /// back to a small default range otherwise).
    /// </summary>
    protected override Tensor<double> CreateRandomTargetTensor(int[] shape, Random rng)
    {
        // For NER the prediction is rank-2 [seq, numLabels]; target should be
        // rank-1 [seq] of integer class indices. If the requested shape's last
        // axis matches the predicted "numLabels" axis (i.e. EffectiveOutputShape
        // came directly from a successful warmup), drop that axis and emit
        // [seq] integer indices. Otherwise emit [shape[0]] integers as a
        // best-effort default.
        int seqLen = shape[0];
        int numLabels = shape.Length >= 2 ? shape[shape.Length - 1] : 9;
        if (numLabels < 2) numLabels = 9;  // sanity fallback

        var tensor = new Tensor<double>([seqLen]);
        for (int i = 0; i < seqLen; i++)
            tensor[i] = rng.Next(numLabels);
        return tensor;
    }

    // =====================================================
    // NER INVARIANT: Output Length Related to Input
    // NER models produce one label per token. Output length
    // should be related to input length (same or proportional).
    // =====================================================

    [Fact(Timeout = 120000)]
    public async Task OutputLength_RelatedToInput()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
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

    [Fact(Timeout = 120000)]
    public async Task LabelValues_ShouldBeNonNegative()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
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

    [Fact(Timeout = 120000)]
    public async Task EmptyInput_ShouldNotCrash()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
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

    [Fact(Timeout = 120000)]
    public virtual async Task DifferentInputs_DifferentLabels()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
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
