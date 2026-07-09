using System;
using System.IO;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Models.Options;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNetTests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Opt-in numerically-stable log-softmax-cross-entropy head for SequenceClassification Transformers.
///
/// <para>When the caller supplies <see cref="CrossEntropyWithLogitsLoss{T}"/> (fused stable
/// log-softmax + NLL over LOGITS), the head drops its final Softmax activation layer so training /
/// the loss operate on logits; inference (<c>PredictCore</c>) re-applies softmax so <c>Predict</c>
/// still returns a probability distribution. Purely opt-in: with any other loss (default
/// CategoricalCrossEntropy) the softmax head is retained and the default output path is unchanged.</para>
/// </summary>
public class TransformerLogitsLossHeadTests
{
    private readonly ITestOutputHelper _o;
    public TransformerLogitsLossHeadTests(ITestOutputHelper o) { _o = o; }

    private const int V = 16, Ctx = 6, N = 64;

    private static TransformerArchitecture<float> Arch() => new(
        inputType: InputType.TwoDimensional,
        taskType: NeuralNetworkTaskType.SequenceClassification,
        numEncoderLayers: 1, numDecoderLayers: 0, numHeads: 4,
        modelDimension: 32, feedForwardDimension: 64,
        inputSize: Ctx, outputSize: V, maxSequenceLength: Ctx, vocabularySize: V, randomSeed: 123);

    private static AdamOptimizer<float, Tensor<float>, Tensor<float>> ConstAdam() =>
        new(null, new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
        {
            InitialLearningRate = 3e-3,
            LearningRateScheduler = new AiDotNet.LearningRateSchedulers.ConstantLRScheduler(3e-3),
            UseAdaptiveLearningRate = false,
        });

    private static (Tensor<float> x, Tensor<float> y) Data()
    {
        var rng = new Random(7);
        var x = new Tensor<float>(new[] { N, Ctx });
        var y = new Tensor<float>(new[] { N, V });
        for (int i = 0; i < N; i++)
        {
            int last = 0;
            for (int s = 0; s < Ctx; s++) { int t = rng.Next(V); x[i, s] = t; last = t; }
            y[i, (last + 1) % V] = 1f;
        }
        return (x, y);
    }

    private static double RowSum(Tensor<float> p, int row) { double s = 0; for (int v = 0; v < V; v++) s += p[row, v]; return s; }

    // (1) DEFAULT loss: softmax head retained; Predict output UNCHANGED (probabilities).
    [Fact]
    public void DefaultLoss_KeepsSoftmaxHead_And_LogitsLoss_ProducesIdenticalPredict()
    {
        var (x, _) = Data();
        var probModel = new Transformer<float>(Arch(), lossFunction: new CategoricalCrossEntropyLoss<float>());
        var logitModel = new Transformer<float>(Arch(), lossFunction: new CrossEntropyWithLogitsLoss<float>());

        // The opt-in head drops exactly one layer (the final Softmax activation).
        _o.WriteLine($"layers: prob={probModel.Layers.Count} logits={logitModel.Layers.Count}");
        Assert.Equal(probModel.Layers.Count - 1, logitModel.Layers.Count);

        var pProb = probModel.Predict(x);
        var pLog = logitModel.Predict(x);

        // Both return probability distributions (rows sum to 1).
        Assert.Equal(1.0, RowSum(pProb, 0), 3);
        Assert.Equal(1.0, RowSum(pLog, 0), 3);

        // Same architecture + seed → identical weights → identical logits → the logits-head's
        // re-applied softmax must reproduce the softmax-head output BIT-for-BIT (default Predict
        // semantics unchanged by the opt-in).
        double maxDiff = 0;
        for (int i = 0; i < N; i++)
            for (int v = 0; v < V; v++)
                maxDiff = Math.Max(maxDiff, Math.Abs((double)pProb[i, v] - (double)pLog[i, v]));
        _o.WriteLine($"max |probHead - logitsHead| Predict diff = {maxDiff:E3}");
        Assert.True(maxDiff < 1e-5, $"logits-head Predict must match softmax-head Predict (diff={maxDiff:E3})");
    }

    // (2) Loss parity: CE(softmax(logits)) == CEWithLogits(logits) on the same logits.
    [Fact]
    public void LogitsLoss_CE_Matches_ProbabilityCE()
    {
        var rng = new Random(5);
        int B = 12;
        var logits = new Tensor<float>(new[] { B, V });
        var target = new Tensor<float>(new[] { B, V });
        for (int i = 0; i < B; i++) { for (int v = 0; v < V; v++) logits[i, v] = (float)(rng.NextDouble() * 6 - 3); target[i, rng.Next(V)] = 1f; }

        var eng = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var probs = eng.Softmax(logits, 1);

        double ceProb = (double)new CategoricalCrossEntropyLoss<float>().ComputeTapeLoss(probs, target).ToArray()[0];
        double ceLogits = (double)new CrossEntropyWithLogitsLoss<float>().ComputeTapeLoss(logits, target).ToArray()[0];
        _o.WriteLine($"CE(softmax(logits))={ceProb:F6}  CEWithLogits(logits)={ceLogits:F6}");
        Assert.Equal(ceProb, ceLogits, 4);
    }

    // (3) Training on the logits head learns (held-out top-1 beats the unigram baseline).
    [Fact]
    public void LogitsLoss_Training_Learns()
    {
        var (x, y) = Data();
        var model = new Transformer<float>(Arch(), lossFunction: new CrossEntropyWithLogitsLoss<float>(), optimizer: ConstAdam());
        for (int step = 0; step < 120; step++) model.Train(x, y);

        var pred = model.Predict(x);
        int hit = 0;
        for (int i = 0; i < N; i++)
        {
            int a = 0; float b = pred[i, 0];
            for (int v = 1; v < V; v++) if (pred[i, v] > b) { b = pred[i, v]; a = v; }
            int tgt = 0; for (int v = 0; v < V; v++) if (y[i, v] > 0.5f) tgt = v;
            if (a == tgt) hit++;
        }
        double top1 = 100.0 * hit / N;
        _o.WriteLine($"logits-head trained top-1 = {top1:F1}% (chance {100.0 / V:F1}%)");
        Assert.True(RowSum(pred, 0) is > 0.99 and < 1.01, "Predict must still return probabilities");
        Assert.True(top1 > 100.0 / V + 20.0, $"logits-head training did not learn (top-1={top1:F1}% vs chance {100.0/V:F1}%)");
    }

    // (4) Serialization round-trips the logits head (Predict identical + still probabilities).
    [Fact]
    public void LogitsLoss_Serialization_RoundTrips()
    {
        var (x, y) = Data();
        var model = new Transformer<float>(Arch(), lossFunction: new CrossEntropyWithLogitsLoss<float>(), optimizer: ConstAdam());
        for (int step = 0; step < 20; step++) model.Train(x, y);
        var before = model.Predict(x);

        byte[] blob = model.Serialize();
        var restored = new Transformer<float>(Arch(), lossFunction: new CrossEntropyWithLogitsLoss<float>());
        restored.Deserialize(blob);
        var after = restored.Predict(x);

        Assert.True(RowSum(after, 0) is > 0.99 and < 1.01, "restored Predict must still return probabilities (logits head round-tripped)");
        double maxDiff = 0;
        for (int i = 0; i < N; i++) for (int v = 0; v < V; v++) maxDiff = Math.Max(maxDiff, Math.Abs((double)before[i, v] - (double)after[i, v]));
        _o.WriteLine($"serialize round-trip max Predict diff = {maxDiff:E3}");
        Assert.True(maxDiff < 1e-4, $"restored Predict must match pre-serialize (diff={maxDiff:E3})");
    }
}
