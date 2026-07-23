using AiDotNet;
using AiDotNet.ActivationFunctions;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Repro + regression test for #1625 — AiModelResult.Predict and AiModelBuilder.Predict
/// returned all-exact-zero output for every input after a successful training run.
///
/// <para>Root cause: NeuralNetworkBase.TryForwardGpuOptimized used
/// <c>using var gpuResult = ForwardGpu(input)</c>, which disposed the returned GPU
/// tensor before the caller (NeuralNetwork.PredictCore → AiModelResult.Predict →
/// AiModelBuilder.Predict) could read its values. The caller received a reference
/// to a disposed tensor whose backing memory read as zeros. Every facade-routed
/// Predict call on a model whose layers all support GPU execution and whose engine
/// resolved to DirectGpuTensorEngine returned the all-zero output, even though the
/// underlying forward computation had produced correct probabilities.</para>
///
/// <para>Fix: drop the <c>using</c> so the returned tensor's lifetime belongs to
/// the caller, matching the contract of every other Predict path in the codebase
/// (the result IS the model's output, not an intermediate that can be safely
/// freed).</para>
///
/// <para>Observed externally: the Ivory Cloud W9B capstone (Loan Approval Predictor
/// with per-group fairness audit) and every example in the AiDotNet facade
/// walkthrough showed "high" accuracy that was actually the test set's majority
/// class rate (i.e. accuracy was being computed against an all-zero prediction
/// vector that happened to match all the negative-class labels).</para>
/// </summary>
public class Issue1625_PredictReturnsAllZeroTests
{
    private readonly ITestOutputHelper _output;
    public Issue1625_PredictReturnsAllZeroTests(ITestOutputHelper output) => _output = output;

    /// <summary>
    /// End-to-end facade build/train/predict for a binary classifier. After
    /// training, Predict MUST return output with non-trivial variance — not all
    /// exact zeros. Pre-fix this test fails with mean=0, min=0, max=0.
    /// </summary>
    [Fact]
    public async System.Threading.Tasks.Task Facade_Predict_BinaryClassification_ReturnsNonZeroProbabilities()
    {
        // ── Reproducible synthetic data ──
        // 60 train + 15 test, 8 features, label = sign of sum of first 3 features.
        // Mirrors the AiDotNet facade walkthrough's Customer Churn example (the
        // canonical "smallest facade-based binary classifier" scenario in the
        // repo) so a regression in either the facade or NeuralNetworkBase shows
        // up here first.
        var (trainX, trainY, testX, testY) = MakeBinaryData(samples: 60, features: 8, seed: 42);

        // ── Dense MLP via facade-supported architecture pattern ──
        var layers = new List<ILayer<double>>
        {
            new DenseLayer<double>(outputSize: 16, activationFunction: new ReLUActivation<double>()),
            new DenseLayer<double>(outputSize: 8,  activationFunction: new ReLUActivation<double>()),
            new DenseLayer<double>(outputSize: 1,  activationFunction: new SigmoidActivation<double>()),
        };
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.BinaryClassification,
            complexity: NetworkComplexity.Simple,
            inputSize: 8, outputSize: 1, layers: layers);
        var nn = new NeuralNetwork<double>(architecture);

        // ── Facade build + train ──
        var builder = new AiModelBuilder<double, Tensor<double>, Tensor<double>>();
        var optimizer = new AdamOptimizer<double, Tensor<double>, Tensor<double>>(
            null, new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>
            { InitialLearningRate = 0.05, MaxIterations = 30 });

        var model = await builder
            .ConfigureModel(nn)
            .ConfigureOptimizer(optimizer)
            .ConfigureLossFunction(new BinaryCrossEntropyLoss<double>())
            .ConfigureDataLoader(new InMemoryDataLoader<double, Tensor<double>, Tensor<double>>(trainX, trainY))
            .BuildAsync();

        // ── Facade-only inference ──
        // No SetTrainingMode, no ForwardForTraining, no internal NN APIs — this is
        // what end users are supposed to call.
        var preds = builder.Predict(testX, model);

        // ── Distribution checks ──
        int n = preds.Shape[0];
        Assert.Equal(testX.Shape[0], n);

        double min = double.MaxValue, max = double.MinValue, sum = 0;
        int exactZeroCount = 0;
        for (int i = 0; i < n; i++)
        {
            double v = preds[i, 0];
            if (v < min) min = v;
            if (v > max) max = v;
            sum += v;
            if (v == 0.0) exactZeroCount++;
        }
        double mean = sum / n;

        _output.WriteLine($"Predict output: min={min:F6}, max={max:F6}, mean={mean:F6}, exactZeroCount={exactZeroCount}/{n}");

        // The bug produced exactly 0.0 for every sample. With probabilities from
        // a sigmoid head, no prediction should be exactly 0.0 — sigmoid is in
        // (0, 1). Allow exactly one prediction to be ~0 just in case of extreme
        // saturation (the bug produced n out of n exact zeros).
        Assert.True(exactZeroCount < n,
            $"Pre-fix bug: Predict returned exactly 0.0 for all {n} samples. " +
            "After fix, the sigmoid head should produce non-zero probabilities in (0, 1).");

        // Variance check — a trained-then-Predict pipeline should produce SOME
        // spread across samples, not a degenerate constant. The pre-fix bug
        // returned the same exact 0.0 for every sample (variance = 0).
        Assert.True(max - min > 1e-9,
            $"Pre-fix bug: Predict output had zero variance (min == max == {min:F6}). " +
            "After fix, predictions for different inputs should differ.");
    }

    private static (Tensor<double> trainX, Tensor<double> trainY, Tensor<double> testX, Tensor<double> testY)
        MakeBinaryData(int samples, int features, int seed)
    {
        int testN = samples / 4;
        int trainN = samples - testN;
        var (tx, ty) = MakeSplit(trainN, features, seed);
        var (ex, ey) = MakeSplit(testN, features, seed + 7);
        return (tx, ty, ex, ey);
    }

    private static (Tensor<double> X, Tensor<double> Y) MakeSplit(int n, int features, int seed)
    {
        var rng = new System.Random(seed);
        var X = new Tensor<double>(new[] { n, features });
        var Y = new Tensor<double>(new[] { n, 1 });
        for (int i = 0; i < n; i++)
        {
            double risk = 0;
            for (int f = 0; f < features; f++)
            {
                var v = rng.NextDouble() * 2 - 1;
                X[i, f] = v;
                if (f < 3) risk += v;
            }
            Y[i, 0] = risk > 0 ? 1.0 : 0.0;
        }
        return (X, Y);
    }
}
