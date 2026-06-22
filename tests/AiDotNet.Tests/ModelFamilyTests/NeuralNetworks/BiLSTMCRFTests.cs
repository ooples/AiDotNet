using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NER.Options;
using AiDotNet.NER.SequenceLabeling;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;
using System;
using System.Threading.Tasks;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Manual scaffold for the BiLSTM-CRF sequence-labeling model (Lample et al. 2016,
/// "Neural Architectures for Named Entity Recognition"). Suppresses the auto-generated
/// stub so we can add the paper-fidelity invariant the POC reported as broken in
/// AiDotNet 0.213.3: <b>training diverges</b>.
/// </summary>
/// <remarks>
/// <para>
/// The inherited family invariants below reproduce exactly what the auto-generated
/// SequenceLabelingNER scaffold emits (InputShape <c>[8,100]</c>, a
/// <see cref="CreateRandomTargetTensor"/> that probes the model's own label count, and a
/// widened <see cref="TrainingLossReductionTolerance"/> because the CRF objective is
/// <c>−log P(gold_path)</c>, not the argmax-MSE those generic tests measure). Keeping them
/// here means suppressing the stub costs no coverage.
/// </para>
/// <para>
/// On top of that, <see cref="Training_DoesNotDiverge_AndDrivesCrfNllDown"/> and
/// <see cref="Parameters_StayFinite_AfterTraining"/> are the divergence regression. Two
/// correctness fixes underpin them: (1) the CRF negative-log-likelihood is computed by a
/// tape-tracked forward algorithm so gradients flow into the transition / start / end
/// scores, and (2) the default optimizer is the paper's clipped SGD with the
/// <c>lr_t = lr_0 / (1 + 0.05 t)</c> decay (Lample §4) rather than an adaptive optimizer
/// whose steps oscillate on the CRF objective. They run on a small, self-contained,
/// linearly-separable task so the assertions are fast and deterministic.
/// </para>
/// </remarks>
public class BiLSTMCRFTests : SequenceLabelingNERTestBase
{
    // ---- Family-invariant configuration (mirrors the auto-generated stub) ----

    // LSTM-CRF family defaults to EmbeddingDimension = 100.
    protected override int[] InputShape => [8, 100];

    // The CRF NLL objective (−log P(gold_path)) is correlated with but not equal to the
    // argmax-MSE the generic Training_ShouldReduceLoss measures; combined with 0.5 dropout
    // and SGD warm-up on a random-target one-sample probe, per-step MSE can transiently
    // rise. 5.0 is well above stochastic noise yet far below catastrophic divergence
    // (which spirals to 1e3+ within steps). Matches the scaffold's emitted value.
    protected override double TrainingLossReductionTolerance => 5.0;

    /// <summary>
    /// Sequence-labeling CRF models consume INTEGER label indices, not arbitrary floats.
    /// The base default yields random doubles in [0,1) which the CRF NLL path rounds to a
    /// degenerate two-class target. Probe the model's actual <c>NumLabels</c> and emit
    /// legal integer indices in <c>[0, NumLabels)</c> — a test-data adaptation to the
    /// family's output type, not an assertion weakening. Mirrors the auto-generated stub.
    /// </summary>
    protected override Tensor<double> CreateRandomTargetTensor(int[] shape, Random rng)
    {
        using var probe = CreateNetwork();
        if (probe is not AiDotNet.NER.Interfaces.INERModel<double> ner)
        {
            string actualType = probe?.GetType().FullName ?? "null";
            throw new InvalidOperationException(
                $"SequenceLabelingNER scaffold expected an INERModel<double>, got {actualType}.");
        }
        int numLabels = ner.NumLabels;
        if (numLabels <= 0)
            throw new InvalidOperationException(
                $"INERModel<double>.NumLabels returned {numLabels}; expected a positive label count.");
        var tensor = new Tensor<double>(shape);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = rng.Next(0, numLabels);
        return tensor;
    }

    protected override INeuralNetworkModel<double> CreateNetwork() =>
        new BiLSTMCRF<double>(
            new NeuralNetworkArchitecture<double>(
                inputType: InputType.OneDimensional,
                taskType: NeuralNetworkTaskType.Regression,
                inputSize: 100,
                outputSize: 9));

    // ---- Divergence regression (the POC-reported defect) ----

    private const int Embed = 16;
    private const int Hidden = 16;
    private const int RegressionLabels = 4;
    private const int SeqLen = 8;   // == MaxSequenceLength so no pad/truncate noise

    private static BiLSTMCRF<double> CreateSmallModel() =>
        new BiLSTMCRF<double>(
            new NeuralNetworkArchitecture<double>(
                inputType: InputType.OneDimensional,
                taskType: NeuralNetworkTaskType.Regression,
                inputSize: Embed,
                outputSize: RegressionLabels),
            new BiLSTMCRFOptions
            {
                EmbeddingDimension = Embed,
                HiddenDimension = Hidden,
                NumLSTMLayers = 1,
                NumLabels = RegressionLabels,
                MaxSequenceLength = SeqLen,
                UseCRF = true,
                LabelNames = new[] { "O", "B-PER", "I-PER", "B-LOC" },
            });

    /// <summary>
    /// Builds one linearly-separable training example: each token's gold label is encoded
    /// in its embedding (a one-hot block of width <c>Embed / RegressionLabels</c> plus
    /// small deterministic jitter), so a correctly-wired BiLSTM-CRF can drive the NLL down.
    /// The jitter is seeded per (sentence, token) so the dataset is identical across runs.
    /// </summary>
    private static (Tensor<double> input, Tensor<double> labels) MakeExample(int sentence)
    {
        int block = Embed / RegressionLabels;
        var input = new Tensor<double>([SeqLen, Embed]);
        var labels = new Tensor<double>([SeqLen]);
        for (int t = 0; t < SeqLen; t++)
        {
            int label = (t + sentence) % RegressionLabels;
            labels[t] = label;
            for (int e = 0; e < Embed; e++)
            {
                bool hot = e >= label * block && e < (label + 1) * block;
                double jitter = 0.05 * (((t * 7 + e * 13 + sentence * 17) % 5) - 2);
                input[t, e] = (hot ? 1.0 : 0.0) + jitter;
            }
        }
        return (input, labels);
    }

    /// <summary>
    /// THE divergence regression: train the BiLSTM-CRF on a small separable dataset and
    /// assert the CRF NLL stays finite every epoch, is always ≥ 0 (NLL = logZ − goldScore,
    /// and logZ ≥ goldScore by construction), and ends well below where it started — i.e.
    /// the model learns instead of diverging. A divergent run shows up here as NaN/Inf loss
    /// or a final loss that has grown rather than shrunk.
    /// </summary>
    [Fact(Timeout = 180000)]
    public async Task Training_DoesNotDiverge_AndDrivesCrfNllDown()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        using var model = CreateSmallModel();

        var examples = new (Tensor<double> input, Tensor<double> labels)[3];
        for (int s = 0; s < examples.Length; s++)
            examples[s] = MakeExample(s);

        const int epochs = 40;
        double firstLoss = double.NaN;
        double lastLoss = double.NaN;

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double epochLoss = 0.0;
            foreach (var (input, labels) in examples)
            {
                model.Train(input, labels);
                double loss = Convert.ToDouble(model.GetLastLoss());

                Assert.False(double.IsNaN(loss),
                    $"CRF NLL is NaN at epoch {epoch} — training diverged.");
                Assert.False(double.IsInfinity(loss),
                    $"CRF NLL is Infinity at epoch {epoch} — training diverged.");
                // NLL = logZ − goldScore ≥ 0; a negative value means the forward
                // log-partition is below the gold-path score, which is impossible.
                Assert.True(loss >= -1e-6,
                    $"CRF NLL = {loss:E3} is negative at epoch {epoch} — logZ/goldScore inconsistent.");

                epochLoss += loss;
            }
            epochLoss /= examples.Length;

            if (epoch == 0) firstLoss = epochLoss;
            lastLoss = epochLoss;
        }

        Assert.True(lastLoss < firstLoss,
            $"CRF NLL did not decrease over {epochs} epochs (first={firstLoss:F4}, " +
            $"last={lastLoss:F4}). BiLSTM-CRF training is not converging.");
    }

    /// <summary>
    /// Parameters must stay finite after a full training run. Divergence in the original
    /// 0.213.3 report manifested as the weights blowing up; this guards that directly.
    /// </summary>
    [Fact(Timeout = 180000)]
    public async Task Parameters_StayFinite_AfterTraining()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        using var model = CreateSmallModel();

        var (input, labels) = MakeExample(0);
        for (int i = 0; i < 30; i++)
            model.Train(input, labels);

        var p = model.GetParameters();
        for (int i = 0; i < p.Length; i++)
        {
            Assert.False(double.IsNaN(p[i]), $"Parameter[{i}] is NaN after training — diverged.");
            Assert.False(double.IsInfinity(p[i]), $"Parameter[{i}] is Infinity after training — diverged.");
        }
    }
}
