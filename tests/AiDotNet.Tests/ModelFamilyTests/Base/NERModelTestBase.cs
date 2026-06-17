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
        AssertModelCanLearnToDifferentiateInputs();
    }

    /// <summary>
    /// CRF-aware replacement for the base continuous-output dispersion check. A CRF's
    /// <c>Predict</c> returns a DISCRETE Viterbi argmax label sequence, which legitimately
    /// collapses to the same labels for distinct inputs — an untrained model's argmax is
    /// scale-insensitive, and a model trained on a single random target collapses to the
    /// majority label — even when the underlying network is perfectly healthy (its emissions
    /// ARE input-sensitive). So the generic "distinct inputs -&gt; distinct decoded labels"
    /// assertion gives a false positive on discrete-output sequence labelers.
    ///
    /// The real invariant the base test guards (#1208/#1221: gradient flow to the
    /// embedding/recurrent layers is broken, so the model is input-INSENSITIVE) is preserved
    /// here by verifying the model can LEARN to map two clearly-distinct inputs to two distinct
    /// label sequences. A network with broken input-side gradient flow cannot — it stays
    /// degenerate regardless of training — so this still fails loudly on the real bug.
    /// </summary>
    protected void AssertModelCanLearnToDifferentiateInputs()
    {
        using var _arena = TensorArena.Create();
        using var network = CreateNetwork();
        if (TrainingInvariantsNotApplicable(network)) return;

        var input1 = CreateConstantTensor(InputShape, 0.1);
        var input2 = CreateConstantTensor(InputShape, 0.9);
        // Distinct, valid integer-label targets (label 0 vs label 1 — every NER label space
        // has at least these two: 'O' and a 'B-' tag). Train both mappings so a healthy model
        // learns input1 -> all-0 and input2 -> all-1 and decodes them differently.
        var target0 = CreateConstantTensor(EffectiveOutputShape, 0.0);
        var target1 = CreateConstantTensor(EffectiveOutputShape, 1.0);

        // Train up to a generous budget, succeeding as soon as the model has learned
        // to decode the two inputs differently. CreateNetwork() initialises weights from
        // a non-deterministic RNG, and a pure bidirectional CRF (2x the recurrent
        // parameters of a unidirectional one, with no CNN front-end to sharpen features)
        // sits right at the convergence boundary at a fixed-10 budget — so a fixed tiny
        // count is init-flaky. Early-exit-on-success keeps the probe fair for every
        // healthy model while still failing loudly for the real bug it guards
        // (#1208/#1221: broken input-side gradient flow never differentiates, so it
        // runs the full budget and still asserts false).
        int maxLearnIterations = Math.Max(TrainingIterations, 50);
        bool anyDifferent = false;
        for (int iter = 0; iter < maxLearnIterations && !anyDifferent; iter++)
        {
            network.Train(input1, target0);
            network.Train(input2, target1);

            var labels1 = network.Predict(input1);
            var labels2 = network.Predict(input2);
            int minLen = Math.Min(labels1.Length, labels2.Length);
            for (int i = 0; i < minLen; i++)
            {
                if (Math.Abs(ConvertToDouble(labels1[i]) - ConvertToDouble(labels2[i])) > 1e-12)
                {
                    anyDifferent = true;
                    break;
                }
            }
        }

        Assert.True(anyDifferent,
            "NER model could not learn to map two clearly-distinct inputs to distinct label " +
            "sequences after training — gradient flow to the embedding / recurrent layers is " +
            "likely broken (#1208/#1221), leaving the model input-insensitive.");
    }

    /// <summary>
    /// CRF-aware override: see <see cref="AssertModelCanLearnToDifferentiateInputs"/> for why the
    /// base discrete-label L2 dispersion check is a false positive on sequence labelers.
    /// </summary>
    public override async Task DifferentInputs_AfterTraining_ShouldProduceDifferentOutputs()
    {
        await Task.Yield();
        AssertModelCanLearnToDifferentiateInputs();
    }
}
