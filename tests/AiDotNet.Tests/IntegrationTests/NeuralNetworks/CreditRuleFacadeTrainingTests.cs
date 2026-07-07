using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet;
using AiDotNet.ActivationFunctions;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.Engines;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.CreditAssignment;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Verifies the pluggable credit-assignment ("learning rule") system end to end through
/// <c>AiModelBuilder.ConfigureCreditRule(...).BuildAsync()</c>: the non-backprop rules (Feedback Alignment,
/// Direct Feedback Alignment, Sign-Symmetric) produce <b>real learning</b> on both a dense MLP and — the key
/// deliverable — an actual <see cref="Transformer{T}"/> (Direct Feedback Alignment scales to attention, per
/// Launay et al. 2020). Learning is measured the correct way: held-out TOP-1 accuracy vs chance (NOT the streamed
/// optimizer loss, NOT a double-softmaxed perplexity).
/// </summary>
public class CreditRuleFacadeTrainingTests
{
    // ---- shared metric ------------------------------------------------------------------------

    private static double Accuracy<T>(Func<Tensor<T>, Tensor<T>> predict, Tensor<T> testX, int[] labels, int numClasses)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var preds = predict(testX);
        int correct = 0;
        for (int n = 0; n < labels.Length; n++)
        {
            int argmax = 0;
            double best = double.NegativeInfinity;
            for (int c = 0; c < numClasses; c++)
            {
                double p = ops.ToDouble(preds[n, c]);
                if (p > best) { best = p; argmax = c; }
            }
            if (argmax == labels[n]) correct++;
        }
        return (double)correct / labels.Length;
    }

    // ===========================================================================================
    // Dense MLP
    // ===========================================================================================

    private const int MlpDim = 8;
    private const int MlpClasses = 3;
    private const int MlpHidden = 16;

    private static (Tensor<double> x, Tensor<double> y, int[] labels) MakeBlobs(int samples, int seed)
    {
        var rng = new Random(seed);
        var centres = new double[MlpClasses][];
        var centreRng = new Random(20260706);
        for (int c = 0; c < MlpClasses; c++)
        {
            centres[c] = new double[MlpDim];
            for (int d = 0; d < MlpDim; d++)
                centres[c][d] = (centreRng.NextDouble() * 2.0 - 1.0) * 4.0;
        }

        var x = new Tensor<double>(new[] { samples, MlpDim });
        var y = new Tensor<double>(new[] { samples, MlpClasses });
        var labels = new int[samples];
        for (int n = 0; n < samples; n++)
        {
            int c = rng.Next(MlpClasses);
            labels[n] = c;
            for (int d = 0; d < MlpDim; d++)
                x[n, d] = centres[c][d] + NextGaussian(rng) * 0.6;
            for (int k = 0; k < MlpClasses; k++)
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
            new FullyConnectedLayer<double>(MlpDim, MlpHidden, new ReLUActivation<double>()),
            new FullyConnectedLayer<double>(MlpHidden, MlpClasses, new SoftmaxActivation<double>()),
        };
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            complexity: NetworkComplexity.Simple,
            inputSize: MlpDim,
            outputSize: MlpClasses,
            layers: layers);
        return new NeuralNetwork<double>(architecture);
    }

    [Theory]
    [InlineData(CreditRule.FeedbackAlignment, 0.60)]
    [InlineData(CreditRule.DirectFeedbackAlignment, 0.60)]
    [InlineData(CreditRule.SignSymmetric, 0.60)]
    public async Task ConfigureCreditRule_TrainsMlp_HeldOutAccuracyBeatsChance(CreditRule rule, double minAccuracy)
    {
        var (trainX, trainY, _) = MakeBlobs(300, seed: 1);
        var (testX, _, testLabels) = MakeBlobs(120, seed: 999);

        var mlp = BuildMlp();
        double beforeAcc = Accuracy<double>(mlp.Predict, testX, testLabels, MlpClasses);

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

        double afterAcc = Accuracy<double>(result.Predict, testX, testLabels, MlpClasses);

        Assert.True(afterAcc >= minAccuracy,
            $"{rule}: held-out accuracy {afterAcc:F3} did not reach {minAccuracy:F2} (chance={1.0 / MlpClasses:F3}, before={beforeAcc:F3}).");
        Assert.True(afterAcc > beforeAcc + 0.10,
            $"{rule}: accuracy did not improve enough (before={beforeAcc:F3}, after={afterAcc:F3}).");
    }

    // ===========================================================================================
    // Transformer (the key deliverable: DFA scales to attention)
    // ===========================================================================================

    private const int SeqLen = 6;
    private const int Vocab = 12;
    private const int TfClasses = 3;

    /// <summary>
    /// Learnable sequence-classification task: each sequence is dominated (~70%) by its class marker token
    /// (ids 0..classes-1) with the rest filler tokens; the mean-pooled embedding leans toward the class marker.
    /// </summary>
    private static (Tensor<float> x, Tensor<float> y, int[] labels) MakeSeqData(int samples, int seed)
    {
        var rng = new Random(seed);
        var x = new Tensor<float>(new[] { samples, SeqLen });
        var y = new Tensor<float>(new[] { samples, TfClasses });
        var labels = new int[samples];
        for (int n = 0; n < samples; n++)
        {
            int c = rng.Next(TfClasses);
            labels[n] = c;
            for (int s = 0; s < SeqLen; s++)
            {
                x[n, s] = rng.NextDouble() < 0.7
                    ? c
                    : TfClasses + rng.Next(Vocab - TfClasses);
            }
            for (int k = 0; k < TfClasses; k++)
                y[n, k] = k == c ? 1f : 0f;
        }
        return (x, y, labels);
    }

    private static Transformer<float> BuildTransformer()
    {
        var architecture = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.SequenceClassification,
            numEncoderLayers: 2,
            numDecoderLayers: 0,
            numHeads: 4,
            modelDimension: 48,
            feedForwardDimension: 96,
            inputSize: SeqLen,
            outputSize: TfClasses,
            maxSequenceLength: SeqLen,
            vocabularySize: Vocab,
            randomSeed: 42);
        return new Transformer<float>(architecture, lossFunction: new CategoricalCrossEntropyLoss<float>());
    }

    [Fact(Timeout = 300000)]
    public async Task ConfigureCreditRule_DFA_TrainsTransformer_HeldOutAccuracyBeatsChance()
    {
        // Isolate credit-rule correctness from any GPU transformer-training issue: force the CPU engine.
        Environment.SetEnvironmentVariable("AIDOTNET_DISABLE_GPU", "1");
        AiDotNetEngine.ResetToCpu();

        var (trainX, trainY, _) = MakeSeqData(240, seed: 1);
        var (testX, _, testLabels) = MakeSeqData(120, seed: 999);

        var transformer = BuildTransformer();
        double beforeAcc = Accuracy<float>(transformer.Predict, testX, testLabels, TfClasses);

        var adam = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(
            null,
            new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                InitialLearningRate = 0.01,
                MaxIterations = 60,
                BatchSize = 16,
            });

        var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(transformer)
            .ConfigureOptimizer(adam)
            .ConfigureCreditRule(CreditRule.DirectFeedbackAlignment, seed: 42)
            .ConfigureLossFunction(new CategoricalCrossEntropyLoss<float>())
            .ConfigureDataLoader(new InMemoryDataLoader<float, Tensor<float>, Tensor<float>>(trainX, trainY))
            .BuildAsync();

        double afterAcc = Accuracy<float>(result.Predict, testX, testLabels, TfClasses);

        // Direct Feedback Alignment must move an actual Transformer's held-out top-1 accuracy well above chance.
        Assert.True(afterAcc >= 0.55,
            $"DFA-Transformer: held-out accuracy {afterAcc:F3} did not reach 0.55 (chance={1.0 / TfClasses:F3}, before={beforeAcc:F3}).");
        Assert.True(afterAcc > beforeAcc + 0.10,
            $"DFA-Transformer: accuracy did not improve enough (before={beforeAcc:F3}, after={afterAcc:F3}).");
    }

    // ===========================================================================================
    // API + scope
    // ===========================================================================================

    [Fact]
    public void ConfigureCreditRule_Backprop_LeavesDefaultPathUnchanged()
    {
        var mlp = BuildMlp();

        // The enum overload maps Backprop to a null rule → default reverse-mode path, unchanged.
        mlp.SetCreditRule(CreditRuleFactory<double>.Create(CreditRule.Backprop));
        Assert.Null(mlp.ActiveCreditRule);

        // The instance overload returns an exact-backprop rule that the engine bypasses (default path).
        var backprop = CreditRules.Backprop<double>();
        Assert.True(backprop.IsExactBackprop);
    }

    [Fact]
    public void CreditRules_Factory_ReturnsConfiguredInstances()
    {
        Assert.Equal("DirectFeedbackAlignment", CreditRules.DirectFeedbackAlignment<double>(seed: 42).Name);
        Assert.Equal("FeedbackAlignment", CreditRules.FeedbackAlignment<double>(seed: 7).Name);
        Assert.Equal("SignSymmetric", CreditRules.SignSymmetric<double>().Name);
        Assert.False(CreditRules.DirectFeedbackAlignment<double>().IsExactBackprop);
    }

    [Theory]
    [InlineData(CreditRule.DirectFeedbackAlignment)]
    [InlineData(CreditRule.FeedbackAlignment)]
    [InlineData(CreditRule.SignSymmetric)]
    public void CreditRuleGradients_PositivelyAlignWithBackprop_AfterTraining(CreditRule rule)
    {
        // Feedback-Alignment / Direct-Feedback-Alignment theory (Lillicrap et al. 2016; Nøkland 2016):
        // a credit rule's gradient is NOT guaranteed to align with back-prop at RANDOM initialization —
        // with fixed random feedback the expected alignment there is ~0. Alignment is an emergent
        // property that DEVELOPS as the forward weights adapt to the fixed feedback matrices during
        // training (the network learns to make its own forward path agree with the feedback path). So
        // we train each rule briefly, then assert its gradient has become positively aligned with the
        // true (back-prop) gradient — the property the rules actually guarantee, and the one that makes
        // them learn (verified end-to-end by the held-out-accuracy tests above).
        //
        // One [InlineData] per rule (not a loop) so a regression in ONE rule is reported on its own,
        // and the other rules are still exercised, instead of the run stopping at the first failure.
        var ops = MathHelper.GetNumericOperations<double>();
        var (trainX, trainY, _) = MakeBlobs(300, seed: 1);
        var (x, y, _) = MakeBlobs(64, seed: 7);
        var lr = ops.FromDouble(0.05);

        var mlp = BuildMlp();
        _ = mlp.Predict(trainX); // resolve shapes

        // Train with the credit rule so feedback alignment develops.
        mlp.SetCreditRule(CreditRuleFactory<double>.Create(rule, seed: 123));
        for (int step = 0; step < 60; step++)
        {
            var stepGrad = mlp.ComputeGradients(trainX, trainY);
            mlp.ApplyGradients(stepGrad, lr);
        }

        // At the trained weights, the credit-rule gradient must positively align with back-prop.
        var ruleGrad = mlp.ComputeGradients(x, y);
        mlp.SetCreditRule(null);
        var backpropGrad = mlp.ComputeGradients(x, y);

        double cos = Cosine(backpropGrad, ruleGrad);
        Assert.True(cos > 0.0,
            $"{rule} gradient should be positively aligned with back-prop after training (cosine={cos:F4}).");
    }

    [Fact]
    public void SignSymmetric_OnTransformer_ThrowsClearly_ButDFADoesNot()
    {
        Environment.SetEnvironmentVariable("AIDOTNET_DISABLE_GPU", "1");
        AiDotNetEngine.ResetToCpu();

        var (x, y, _) = MakeSeqData(4, seed: 3);
        var transformer = BuildTransformer();
        _ = transformer.Predict(x); // resolve shapes

        transformer.SetCreditRule(CreditRuleFactory<float>.Create(CreditRule.SignSymmetric));
        Assert.Throws<NotSupportedException>(() => transformer.ComputeGradients(x, y));

        // DFA generalizes to attention and must NOT throw.
        transformer.SetCreditRule(CreditRuleFactory<float>.Create(CreditRule.DirectFeedbackAlignment, seed: 1));
        var grad = transformer.ComputeGradients(x, y);
        Assert.True(grad.Length > 0);
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
