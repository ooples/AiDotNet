using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet;
using AiDotNet.ActivationFunctions;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.CreditAssignment;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Verifies the pluggable credit-assignment ("learning rule") system: training a dense MLP through
/// <c>AiModelBuilder.ConfigureCreditRule(...).BuildAsync()</c> with the non-backprop feedback-alignment
/// family (Feedback Alignment, Direct Feedback Alignment, Sign-Symmetric) produces <b>real learning</b>,
/// measured the correct way — held-out TOP-1 accuracy vs chance (NOT the streamed optimizer loss, and NOT a
/// double-softmaxed perplexity). Also verifies the default (back-prop) path is left unchanged, and that the
/// alternative rules' gradients positively align with true back-propagation on a fixed net (the feedback
/// matrices self-align, per Lillicrap et al. 2016 / Nøkland 2016).
/// </summary>
public class CreditRuleFacadeTrainingTests
{
    private const int Dim = 8;
    private const int Classes = 3;
    private const int Hidden = 16;

    /// <summary>Linearly-separable Gaussian blobs — a genuinely learnable multi-class task.</summary>
    private static (Tensor<double> x, Tensor<double> y, int[] labels) MakeBlobs(int samples, int seed)
    {
        var rng = new Random(seed);
        // Fixed, well-separated class centres.
        var centres = new double[Classes][];
        var centreRng = new Random(20260706);
        for (int c = 0; c < Classes; c++)
        {
            centres[c] = new double[Dim];
            for (int d = 0; d < Dim; d++)
                centres[c][d] = (centreRng.NextDouble() * 2.0 - 1.0) * 4.0;
        }

        var x = new Tensor<double>(new[] { samples, Dim });
        var y = new Tensor<double>(new[] { samples, Classes });
        var labels = new int[samples];
        for (int n = 0; n < samples; n++)
        {
            int c = rng.Next(Classes);
            labels[n] = c;
            for (int d = 0; d < Dim; d++)
                x[n, d] = centres[c][d] + NextGaussian(rng) * 0.6;
            for (int k = 0; k < Classes; k++)
                y[n, k] = k == c ? 1.0 : 0.0;
        }
        return (x, y, labels);
    }

    private static double NextGaussian(Random r)
    {
        double u1 = 1.0 - r.NextDouble();
        double u2 = 1.0 - r.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    private static NeuralNetwork<double> BuildMlp()
    {
        var layers = new List<ILayer<double>>
        {
            new FullyConnectedLayer<double>(Dim, Hidden, new ReLUActivation<double>()),
            new FullyConnectedLayer<double>(Hidden, Classes, new SoftmaxActivation<double>()),
        };
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            complexity: NetworkComplexity.Simple,
            inputSize: Dim,
            outputSize: Classes,
            layers: layers);
        return new NeuralNetwork<double>(architecture);
    }

    private static double HeldOutAccuracy(IFullModelPredict predict, Tensor<double> testX, int[] testLabels)
    {
        var preds = predict.Predict(testX);
        int correct = 0;
        for (int n = 0; n < testLabels.Length; n++)
        {
            int argmax = 0;
            double best = double.NegativeInfinity;
            for (int c = 0; c < Classes; c++)
            {
                double p = preds[n, c];
                if (p > best) { best = p; argmax = c; }
            }
            if (argmax == testLabels[n]) correct++;
        }
        return (double)correct / testLabels.Length;
    }

    // Minimal predict-only surface so the helper works for both the facade result and a raw network.
    private interface IFullModelPredict { Tensor<double> Predict(Tensor<double> x); }

    private sealed class ResultPredict : IFullModelPredict
    {
        private readonly Func<Tensor<double>, Tensor<double>> _f;
        public ResultPredict(Func<Tensor<double>, Tensor<double>> f) => _f = f;
        public Tensor<double> Predict(Tensor<double> x) => _f(x);
    }

    [Theory]
    [InlineData(CreditRule.FeedbackAlignment, 0.65)]
    [InlineData(CreditRule.DirectFeedbackAlignment, 0.65)]
    [InlineData(CreditRule.SignSymmetric, 0.65)]
    [InlineData(CreditRule.Backprop, 0.75)]
    public async Task ConfigureCreditRule_TrainsMlp_HeldOutAccuracyBeatsChance(CreditRule rule, double minAccuracy)
    {
        var (trainX, trainY, _) = MakeBlobs(300, seed: 1);
        var (testX, _, testLabels) = MakeBlobs(120, seed: 999);

        var mlp = BuildMlp();

        // Accuracy from random init should be around chance (1/3).
        double beforeAcc = HeldOutAccuracy(new ResultPredict(mlp.Predict), testX, testLabels);

        var adam = new AdamOptimizer<double, Tensor<double>, Tensor<double>>(
            null,
            new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>
            {
                InitialLearningRate = 0.03,
                MaxIterations = 80,
                BatchSize = 32,
            });

        var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureModel(mlp)
            .ConfigureOptimizer(adam)
            .ConfigureCreditRule(rule, seed: 42)
            .ConfigureLossFunction(new CategoricalCrossEntropyLoss<double>())
            .ConfigureDataLoader(new InMemoryDataLoader<double, Tensor<double>, Tensor<double>>(trainX, trainY))
            .BuildAsync();

        Assert.NotNull(result);
        Assert.NotNull(result.Model);

        double afterAcc = HeldOutAccuracy(new ResultPredict(result.Predict), testX, testLabels);

        // Real learning: held-out top-1 accuracy well above chance AND above the untrained baseline.
        Assert.True(afterAcc >= minAccuracy,
            $"{rule}: held-out accuracy {afterAcc:F3} did not reach {minAccuracy:F2} (chance={1.0 / Classes:F3}, before={beforeAcc:F3}).");
        Assert.True(afterAcc > beforeAcc + 0.10,
            $"{rule}: accuracy did not improve enough (before={beforeAcc:F3}, after={afterAcc:F3}).");
    }

    [Fact]
    public void ConfigureCreditRule_Backprop_LeavesDefaultPathUnchanged()
    {
        // Selecting Backprop is identical to not configuring a rule: no ICreditRule is attached.
        var mlp = BuildMlp();
        mlp.SetCreditRule(CreditRuleFactory<double>.Create(CreditRule.Backprop));
        Assert.Null(mlp.ActiveCreditRule);

        Assert.NotNull(CreditRuleFactory<double>.Create(CreditRule.FeedbackAlignment));
        Assert.NotNull(CreditRuleFactory<double>.Create(CreditRule.DirectFeedbackAlignment));
        Assert.NotNull(CreditRuleFactory<double>.Create(CreditRule.SignSymmetric));
    }

    [Fact]
    public void CreditRuleGradients_PositivelyAlignWithBackprop_OnFixedNet()
    {
        // On a fixed net + batch, the credit-rule path's own back-prop must closely match the tape gradient,
        // and FA/DFA/Sign-Symmetric must be positively aligned with true back-prop (they self-align).
        var (x, y, _) = MakeBlobs(64, seed: 7);

        var mlp = BuildMlp();
        // Force shape resolution + identical starting weights for every measurement.
        _ = mlp.Predict(x);

        // True back-prop gradient (default tape path).
        mlp.SetCreditRule(null);
        var tapeGrad = mlp.ComputeGradients(x, y);

        // Credit-rule engine's own back-prop reproduction (internal rule, visible via InternalsVisibleTo).
        var engineBackprop = ComputeViaEngine(mlp, x, y, new BackpropCreditRule<double>());

        double cosEngineVsTape = Cosine(tapeGrad, engineBackprop);
        Assert.True(cosEngineVsTape > 0.99,
            $"Credit-rule back-prop should reproduce the tape gradient direction (cosine={cosEngineVsTape:F4}).");

        foreach (var rule in new[] { CreditRule.FeedbackAlignment, CreditRule.DirectFeedbackAlignment, CreditRule.SignSymmetric })
        {
            var grad = ComputeViaEngine(mlp, x, y, CreditRuleFactory<double>.Create(rule)!);
            double cos = Cosine(tapeGrad, grad);
            Assert.True(cos > 0.0,
                $"{rule} gradient should be positively aligned with back-prop (cosine={cos:F4}).");
        }
    }

    private static Vector<double> ComputeViaEngine(NeuralNetwork<double> mlp, Tensor<double> x, Tensor<double> y, ICreditRule<double> rule)
    {
        mlp.SetCreditRule(rule, seed: 123);
        var g = mlp.ComputeGradients(x, y);
        mlp.SetCreditRule(null);
        return g;
    }

    private static double Cosine(Vector<double> a, Vector<double> b)
    {
        double dot = 0, na = 0, nb = 0;
        int n = Math.Min(a.Length, b.Length);
        for (int i = 0; i < n; i++)
        {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        if (na == 0 || nb == 0) return 0;
        return dot / (Math.Sqrt(na) * Math.Sqrt(nb));
    }
}
