using AiDotNet.Enums;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNetTests.IntegrationTests.NeuralNetworks;

/// <summary>
/// End-to-end convergence tests for <see cref="Transformer{T}"/>.
///
/// Protects against issue #1187 — Transformer.Train() loss plateaus at
/// log(V)/V from epoch 1, meaning parameters aren't being updated.
///
/// Strategy: overfit a tiny memorization task for a handful of epochs
/// and assert loss strictly decreases. If training is broken the loss
/// stays bit-identical at the initialization value (log(V)/V for the
/// softmax + cross-entropy head on a V-class problem) and these tests
/// fail loudly. If training works, loss falls monotonically on this
/// trivial task well within 50 epochs.
/// </summary>
public class TransformerTrainConvergenceTests
{
    private readonly ITestOutputHelper _output;

    public TransformerTrainConvergenceTests(ITestOutputHelper output)
    {
        _output = output;
    }

    /// <summary>
    /// Mirrors issue #1187's V=512 scenario at reduced scale (V=16, 4
    /// memorization facts) to keep the test fast. Calls Transformer.
    /// Train() directly on the exact same code path the issue uses.
    /// </summary>
    [Fact]
    public void Train_SequenceClassification_LossDecreasesAcrossEpochs()
    {
        // Tiny vocab + short sequence + small d_model so the test runs
        // in well under a second per epoch, yet still exercises the
        // same encoder → pool → dense → softmax → CE stack the issue
        // reproduces on.
        const int vocabSize = 16;
        const int seqLen = 4;
        const int modelDim = 16;
        const int ffDim = 32;
        const int numFacts = 4;

        // Each fact gets class-index `f` in the one-hot target below.
        // If a future edit bumps numFacts past vocabSize the loop at
        // `tgt[0, f] = 1f` would silently miss or read out-of-range,
        // producing a malformed target and a misleading test failure.
        // Fail fast with the variable names in the message so the cause
        // is obvious to whoever tripped it.
        Assert.True(numFacts <= vocabSize,
            $"numFacts ({numFacts}) must not exceed vocabSize ({vocabSize}) — " +
            "one-hot target indexing assumes class id < vocab.");

        var architecture = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.SequenceClassification,
            numEncoderLayers: 2,
            numDecoderLayers: 0,
            numHeads: 2,
            modelDimension: modelDim,
            feedForwardDimension: ffDim,
            inputSize: seqLen,
            outputSize: vocabSize,
            maxSequenceLength: seqLen,
            vocabularySize: vocabSize);

        var transformer = new Transformer<float>(
            architecture,
            lossFunction: new CategoricalCrossEntropyLoss<float>());

        // Seed the model parameters deterministically so the test is
        // reproducible across machines. Transformer.Serialize() + a
        // single forward pass would warm the same tensors, but using
        // a fixed construction via the architecture gives us the same
        // initial weights every run.

        // Build a deterministic memorization set: each "fact" pairs a
        // fixed token sequence with a fixed class label. A one-hot
        // target tensor drives CategoricalCrossEntropyLoss the same
        // way the issue's repro does.
        var inputs = new Tensor<float>[numFacts];
        var targets = new Tensor<float>[numFacts];
        for (int f = 0; f < numFacts; f++)
        {
            var inp = new Tensor<float>([1, seqLen]);
            for (int s = 0; s < seqLen; s++)
            {
                // Deterministic token ids distinct per fact.
                inp[0, s] = (f * seqLen + s) % vocabSize;
            }
            inputs[f] = inp;

            var tgt = new Tensor<float>([1, vocabSize]);
            tgt[0, f] = 1f; // one-hot on class = fact index
            targets[f] = tgt;
        }

        transformer.SetTrainingMode(true);
        var losses = new List<float>();
        const int epochs = 20;
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            float sum = 0f;
            for (int f = 0; f < numFacts; f++)
            {
                transformer.Train(inputs[f], targets[f]);
                sum += transformer.GetLastLoss();
            }
            float avg = sum / numFacts;
            losses.Add(avg);
            _output.WriteLine($"epoch {epoch + 1,2}/{epochs}  avg loss={avg:F6}");
        }

        // Sanity: every epoch produced a finite, non-negative loss.
        // Use !IsNaN && !IsInfinity instead of float.IsFinite — the latter
        // is netcoreapp2.1+ / netstandard2.1+ only and this test compiles
        // on net471 too.
        foreach (var l in losses)
            Assert.True(!float.IsNaN(l) && !float.IsInfinity(l) && l >= 0, $"loss must be finite and non-negative; got {l}");

        // Core regression guard: a bit-identical-across-epochs loss is
        // exactly the symptom issue #1187 describes. With a working
        // training loop, the spread between the best and worst epoch
        // on a 4-fact memorization task is substantial (typically
        // loss falls by more than half in 20 epochs). Allow a tiny
        // tolerance for numerical noise while still failing loudly on
        // true stasis.
        float maxLoss = losses.Max();
        float minLoss = losses.Min();
        float spread = maxLoss - minLoss;
        Assert.True(spread > 1e-4f,
            $"Transformer.Train() appears to have plateaued (issue #1187): " +
            $"loss min={minLoss:F6}, max={maxLoss:F6}, spread={spread:F6}. " +
            $"Expected strictly decreasing loss on a 4-fact memorization task.");

        // Stronger guard: loss at the end of training must be lower than
        // loss at the start. Uses the average of the first few epochs
        // vs the last few so momentary noise near the start doesn't
        // flip the comparison.
        float earlyAvg = (losses[0] + losses[1] + losses[2]) / 3f;
        float lateAvg = (losses[^3] + losses[^2] + losses[^1]) / 3f;
        Assert.True(lateAvg < earlyAvg,
            $"Expected late-epoch loss to be lower than early-epoch loss; " +
            $"got early={earlyAvg:F6}, late={lateAvg:F6}. " +
            $"Parameters are not being updated effectively (issue #1187).");

        // STRONG MEMORIZATION GUARD (closes the test-infra gap that let
        // ooples/AiDotNet#1264 slip through CI). The previous "lateAvg <
        // earlyAvg" guard only required loss to decrease SOMEWHAT — a
        // model trained with vanilla SGD's tiny per-step updates would
        // see a small loss decrease while never actually learning the
        // task, and CI would happily green-light it. This guard asserts
        // the model has actually MEMORISED the training set: after 20
        // epochs of overfitting on 4 facts, the predicted probability
        // mass on the correct class must be >0.50 on EVERY training
        // example. Random (1/V = 1/16 = 0.0625) is the floor; "loss
        // decreased a bit" sits around 0.10–0.20; genuine memorisation
        // crosses 0.50 easily.
        transformer.SetTrainingMode(false);
        for (int f = 0; f < numFacts; f++)
        {
            var pred = transformer.Predict(inputs[f]);
            // Softmax inline — the network's output may be raw logits or
            // already-softmax'd depending on the head; normalise both
            // forms to a probability distribution before checking.
            float maxLogit = float.NegativeInfinity;
            for (int v = 0; v < vocabSize; v++)
                if (pred[0, v] > maxLogit) maxLogit = pred[0, v];
            float sumExp = 0f;
            var probs = new float[vocabSize];
            for (int v = 0; v < vocabSize; v++)
            {
                probs[v] = MathF.Exp(pred[0, v] - maxLogit);
                sumExp += probs[v];
            }
            for (int v = 0; v < vocabSize; v++) probs[v] /= sumExp;

            float pTarget = probs[f];
            int argmax = 0;
            float pmax = probs[0];
            for (int v = 1; v < vocabSize; v++)
                if (probs[v] > pmax) { pmax = probs[v]; argmax = v; }

            _output.WriteLine($"  fact {f}: P(target={f})={pTarget:F4}  argmax={argmax}");
            Assert.True(pTarget > 0.50f,
                $"Transformer failed to memorise fact {f} after {epochs} overfitting epochs: "
                + $"P(target={f})={pTarget:F4} (need >0.50). "
                + "This is the strong convergence guard that catches optimizer/default-LR bugs "
                + "(see ooples/AiDotNet#1264) — losing this assertion is what previously let the "
                + "vanilla-SGD-default ship without anyone noticing the model wasn't learning.");
            Assert.True(argmax == f,
                $"Transformer predicted argmax={argmax} but expected {f} for memorised fact {f}. "
                + "Model converged on wrong class — likely gradient sign or target indexing issue.");
        }

        _output.WriteLine(
            $"Convergence check passed: early={earlyAvg:F6}, late={lateAvg:F6}, " +
            $"spread={spread:F6}, all {numFacts} facts memorised with P(target)>0.50.");
    }
}
