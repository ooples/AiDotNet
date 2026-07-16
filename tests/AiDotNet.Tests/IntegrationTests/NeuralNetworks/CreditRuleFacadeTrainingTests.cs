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
    private readonly Xunit.Abstractions.ITestOutputHelper _output;
    public CreditRuleFacadeTrainingTests(Xunit.Abstractions.ITestOutputHelper output) => _output = output;

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

    // A DEEP contiguous MLP (several hidden layers) on the same blob task — exercises the depth that the single
    // hidden layer of BuildMlp does not, which is where target-propagation rules (learned inverses) are meant to work.
    private static NeuralNetwork<double> BuildDeepBlobNet(int hiddenLayers, int width = 16)
    {
        var layers = new List<ILayer<double>> { new FullyConnectedLayer<double>(MlpDim, width, new ReLUActivation<double>()) };
        for (int i = 0; i < hiddenLayers - 1; i++)
            layers.Add(new FullyConnectedLayer<double>(width, width, new ReLUActivation<double>()));
        layers.Add(new FullyConnectedLayer<double>(width, MlpClasses, new SoftmaxActivation<double>()));
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            complexity: NetworkComplexity.Medium,
            inputSize: MlpDim,
            outputSize: MlpClasses,
            layers: layers);
        return new NeuralNetwork<double>(architecture);
    }

    [Theory]
    [InlineData(CreditRule.FeedbackAlignment, 0.60)]
    [InlineData(CreditRule.DirectFeedbackAlignment, 0.60)]
    [InlineData(CreditRule.SignSymmetric, 0.60)]
    [InlineData(CreditRule.KolenPollack, 0.60)]
    [InlineData(CreditRule.DirectKolenPollack, 0.60)]
    [InlineData(CreditRule.DRTP, 0.60)]
    [InlineData(CreditRule.DFANormalized, 0.60)]
    [InlineData(CreditRule.LocalErrorSignal, 0.60)]
    [InlineData(CreditRule.DifferenceTargetPropagation, 0.60)]
    [InlineData(CreditRule.DirectDifferenceTargetPropagation, 0.60)]
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

    /// <summary>
    /// The target-propagation rules (learned inverses) must train a genuinely <b>deep contiguous</b> network — three
    /// hidden layers — above chance, not just the single hidden layer of the learns-test above. This is the depth
    /// regime these rules are designed for (Difference Target Propagation chains inverses layer-to-layer; its direct
    /// variant routes a learned inverse from the output).
    /// </summary>
    [Theory]
    [InlineData(CreditRule.DifferenceTargetPropagation)]
    [InlineData(CreditRule.DirectDifferenceTargetPropagation)]
    public async Task TargetPropagation_TrainsDeepContiguousNet_BeatsChance(CreditRule rule)
    {
        var (trainX, trainY, _) = MakeBlobs(300, seed: 1);
        var (testX, _, testLabels) = MakeBlobs(120, seed: 999);
        var net = BuildDeepBlobNet(hiddenLayers: 3);
        double beforeAcc = Accuracy<double>(net.Predict, testX, testLabels, MlpClasses);

        var adam = new AdamOptimizer<double, Tensor<double>, Tensor<double>>(
            null,
            new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>
            {
                InitialLearningRate = 0.03,
                MaxIterations = 120,
                BatchSize = 32,
            });

        var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureModel(net)
            .ConfigureOptimizer(adam)
            .ConfigureCreditRule(rule, seed: 42)
            .ConfigureLossFunction(new CategoricalCrossEntropyLoss<double>())
            .ConfigureDataLoader(new InMemoryDataLoader<double, Tensor<double>, Tensor<double>>(trainX, trainY))
            .BuildAsync();

        double afterAcc = Accuracy<double>(result.Predict, testX, testLabels, MlpClasses);
        _output.WriteLine($"{rule} deep(3-hidden) blobs: before={beforeAcc:F3} after={afterAcc:F3} (chance={1.0 / MlpClasses:F3})");
        Assert.True(afterAcc >= 0.60,
            $"{rule}: deep-net held-out accuracy {afterAcc:F3} did not reach 0.60 (chance={1.0 / MlpClasses:F3}, before={beforeAcc:F3}).");
        Assert.True(afterAcc > beforeAcc + 0.10,
            $"{rule}: deep-net accuracy did not improve enough (before={beforeAcc:F3}, after={afterAcc:F3}).");
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
    // Per-rule learns table (every rule vs backprop vs chance) + depth-sensitive KP > DFA
    // ===========================================================================================

    /// <summary>
    /// Trains a small MLP through the facade with EVERY built-in credit rule on the same learnable blob task and
    /// reports a held-out top-1 accuracy table (rule → accuracy vs chance vs backprop). Every non-backprop rule
    /// must clearly beat chance — the regression guard the prior credit-rule feature lacked.
    /// </summary>
    [Fact(Timeout = 300000)]
    public async Task AllCreditRules_LearnMlp_HeldOutAccuracyTable()
    {
        var (trainX, trainY, _) = MakeBlobs(300, seed: 1);
        var (testX, _, testLabels) = MakeBlobs(120, seed: 999);
        double chance = 1.0 / MlpClasses;

        var rules = new (string name, CreditRule rule)[]
        {
            ("Backprop", CreditRule.Backprop),
            ("FeedbackAlignment", CreditRule.FeedbackAlignment),
            ("DirectFeedbackAlignment", CreditRule.DirectFeedbackAlignment),
            ("SignSymmetric", CreditRule.SignSymmetric),
            ("KolenPollack", CreditRule.KolenPollack),
            ("DirectKolenPollack", CreditRule.DirectKolenPollack),
            ("DRTP", CreditRule.DRTP),
            ("DFANormalized", CreditRule.DFANormalized),
            ("LocalErrorSignal", CreditRule.LocalErrorSignal),
            ("DifferenceTargetPropagation", CreditRule.DifferenceTargetPropagation),
            ("DirectDifferenceTargetPropagation", CreditRule.DirectDifferenceTargetPropagation),
        };

        _output.WriteLine($"rule                       heldout(top1)   chance={chance:F3}");
        foreach (var (name, rule) in rules)
        {
            var mlp = BuildMlp();
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

            double acc = Accuracy<double>(result.Predict, testX, testLabels, MlpClasses);
            _output.WriteLine($"{name,-26} {acc,10:F3}");
            Assert.True(acc >= 0.60,
                $"{name}: held-out accuracy {acc:F3} did not clearly beat chance {chance:F3} (>= 0.60 required).");
        }
    }

    // Concentric-rings task: nonlinear (the class is the radius band), backprop-solvable, and depth-sensitive —
    // fixed random feedback (DFA) loses credit quality with depth, while Kolen-Pollack's learned feedback holds.
    private const int RingDim = 2, RingClasses = 4;

    private static (Tensor<double> x, Tensor<double> y, int[] labels) MakeRings(int samples, int seed)
    {
        var rng = new Random(seed);
        var x = new Tensor<double>(new[] { samples, RingDim });
        var y = new Tensor<double>(new[] { samples, RingClasses });
        var labels = new int[samples];
        double band = 1.5 / RingClasses;
        for (int n = 0; n < samples; n++)
        {
            int lab = rng.Next(RingClasses);
            double rMin = lab * band + 0.02, rMax = (lab + 1) * band - 0.02;
            double r = rMin + rng.NextDouble() * (rMax - rMin);
            double th = rng.NextDouble() * 2 * Math.PI;
            x[n, 0] = r * Math.Cos(th);
            x[n, 1] = r * Math.Sin(th);
            labels[n] = lab;
            for (int k = 0; k < RingClasses; k++) y[n, k] = k == lab ? 1.0 : 0.0;
        }
        return (x, y, labels);
    }

    private static NeuralNetwork<double> BuildDeepRingsNet(int hiddenLayers, int width = 32)
    {
        var layers = new List<ILayer<double>> { new FullyConnectedLayer<double>(RingDim, width, new ReLUActivation<double>()) };
        for (int i = 0; i < hiddenLayers - 1; i++)
            layers.Add(new FullyConnectedLayer<double>(width, width, new ReLUActivation<double>()));
        layers.Add(new FullyConnectedLayer<double>(width, RingClasses, new SoftmaxActivation<double>()));
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            complexity: NetworkComplexity.Medium,
            inputSize: RingDim,
            outputSize: RingClasses,
            layers: layers);
        return new NeuralNetwork<double>(architecture);
    }

    private async Task<Func<Tensor<double>, Tensor<double>>> TrainRings(CreditRule rule, int hiddenLayers, int seed, Tensor<double> trX, Tensor<double> trY)
    {
        var net = BuildDeepRingsNet(hiddenLayers);
        var adam = new AdamOptimizer<double, Tensor<double>, Tensor<double>>(
            null,
            new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>
            {
                InitialLearningRate = 0.01,
                MaxIterations = 150,
                BatchSize = 32,
            });
        var builder = new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureModel(net)
            .ConfigureOptimizer(adam)
            .ConfigureLossFunction(new CategoricalCrossEntropyLoss<double>())
            .ConfigureDataLoader(new InMemoryDataLoader<double, Tensor<double>, Tensor<double>>(trX, trY));
        if (rule != CreditRule.Backprop) builder = builder.ConfigureCreditRule(rule, seed: seed);
        var result = await builder.BuildAsync();
        return result.Predict;
    }

    /// <summary>
    /// Depth-sensitive credit-assignment test: on a 4-hidden-layer net solving the nonlinear rings task,
    /// Kolen-Pollack's LEARNED feedback must (a) clearly beat vanilla Direct Feedback Alignment's fixed random
    /// feedback and (b) approach back-propagation. This is KP's defining advantage and only emerges with depth.
    /// Both rules are averaged over several feedback-init seeds so the comparison is robust to that randomness.
    /// </summary>
    [Fact(Timeout = 600000)]
    public async Task KolenPollack_BeatsVanillaDFA_OnDepthSensitiveRings()
    {
        const int hidden = 4;
        var (trX, trY, _) = MakeRings(1500, seed: 1);
        var (teX, _, teLab) = MakeRings(500, seed: 999);
        double chance = 1.0 / RingClasses;

        async Task<double> Run(CreditRule rule, int seed)
        {
            var predict = await TrainRings(rule, hidden, seed, trX, trY);
            return Accuracy<double>(predict, teX, teLab, RingClasses);
        }

        double backprop = await Run(CreditRule.Backprop, 0);

        var seeds = new[] { 1, 2, 3 };
        double dfaSum = 0, kpSum = 0;
        int perSeedKpWins = 0;
        _output.WriteLine($"rings {hidden} hidden layers, chance={chance:F3}, backprop={backprop:F3}");
        _output.WriteLine("  seed   DFA     KP");
        foreach (int s in seeds)
        {
            double dfaS = await Run(CreditRule.DirectFeedbackAlignment, s);
            double kpS = await Run(CreditRule.KolenPollack, s);
            dfaSum += dfaS; kpSum += kpS;
            if (kpS > dfaS) perSeedKpWins++;
            _output.WriteLine($"  {s,-4} {dfaS,7:F3} {kpS,7:F3}");
        }
        double dfa = dfaSum / seeds.Length;
        double kp = kpSum / seeds.Length;
        _output.WriteLine($"  mean DFA={dfa:F3}  KP={kp:F3}  (KP wins {perSeedKpWins}/{seeds.Length} seeds)");

        Assert.True(dfa > 0.60, $"DFA baseline should still learn (mean {dfa:F3}) for a fair comparison.");
        Assert.True(kp >= 0.90, $"Kolen-Pollack should approach backprop ({backprop:F3}); got mean {kp:F3}.");
        Assert.True(kp > dfa,
            $"Kolen-Pollack's learned feedback (mean {kp:F3}) must beat vanilla DFA's fixed feedback (mean {dfa:F3}) at depth {hidden}.");
        Assert.True(perSeedKpWins >= 2,
            $"Kolen-Pollack should beat DFA on a majority of seeds; won {perSeedKpWins}/{seeds.Length}.");
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

    [Fact]
    public void CreditRuleGradients_ReachEveryLayer_AndOutputMatchesBackprop_OnFixedNet()
    {
        var (x, y, _) = MakeBlobs(64, seed: 7);
        var mlp = BuildMlp();
        _ = mlp.Predict(x); // resolve shapes; weights are identical across all measurements below

        // Parameter layout: [hidden W][hidden b][output W][output b]. The first chunk is the first hidden layer's
        // weights (the layer furthest from the loss); the last two chunks are the output layer.
        var chunkLens = new List<int>();
        foreach (var c in mlp.GetParameterChunks()) chunkLens.Add(c.Length);
        int firstLen = chunkLens[0];
        int outputLen = chunkLens[chunkLens.Count - 1] + chunkLens[chunkLens.Count - 2];
        int total = 0; foreach (var l in chunkLens) total += l;
        int outputStart = total - outputLen;

        mlp.SetCreditRule(null);
        var backpropGrad = mlp.ComputeGradients(x, y);

        foreach (var rule in new[] { CreditRule.DirectFeedbackAlignment, CreditRule.FeedbackAlignment, CreditRule.SignSymmetric })
        {
            mlp.SetCreditRule(CreditRuleFactory<double>.Create(rule, seed: 123));
            var g = mlp.ComputeGradients(x, y);
            mlp.SetCreditRule(null);

            // Regression guard: credit must actually REACH the earliest hidden layer (a prior bug zeroed every
            // hidden layer's gradient — only the output layer trained, and only linearly-separable tasks "passed").
            double firstNorm = 0;
            for (int i = 0; i < firstLen; i++) firstNorm += g[i] * g[i];
            Assert.True(Math.Sqrt(firstNorm) > 1e-8,
                $"{rule}: first hidden layer received a zero gradient — credit did not reach it.");

            // The output layer is trained with the exact loss gradient, so its portion must match back-prop.
            double dot = 0, na = 0, nb = 0;
            for (int i = outputStart; i < total; i++) { dot += backpropGrad[i] * g[i]; na += backpropGrad[i] * backpropGrad[i]; nb += g[i] * g[i]; }
            double outCos = (na == 0 || nb == 0) ? 0 : dot / (Math.Sqrt(na) * Math.Sqrt(nb));
            Assert.True(outCos > 0.99,
                $"{rule}: output-layer gradient should equal the exact back-prop loss gradient (cosine={outCos:F4}).");
        }
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

    // ===========================================================================================
    // Public extensible base class — a user-authored rule outside the library trains through the facade
    // ===========================================================================================

    /// <summary>
    /// A minimal custom credit rule authored "outside" the library by subclassing the public
    /// <see cref="CreditRuleBase{T}"/> and using only its protected helpers — proving the base is genuinely
    /// extensible and that <c>ConfigureCreditRule(ICreditRule&lt;T&gt;)</c> accepts it.
    /// </summary>
    private sealed class CustomDirectRule<T> : CreditRuleBase<T>
    {
        public CustomDirectRule(int? seed = null) : base(seed) { }
        public override string Name => "CustomDirectRule";

        public override void ComputeTeachingSignals(ICreditAssignmentContext<T> context)
        {
            int outputFeatures = context.OutputError.Shape[1];
            var feedback = EnsureFeedback(context, (layers, i) =>
                layers[i].IsOutputLayer ? null : (outputFeatures, layers[i].FlatFeatureSize));
            var error = ErrorMatrix(context);
            foreach (var layer in context.Layers)
            {
                if (layer.IsOutputLayer) continue;
                var projected = ProjectThrough(error, feedback[layer.Index]!);
                layer.TeachingSignal = ToTeachingSignal(projected, layer.OutputShape);
            }
        }
    }

    [Fact(Timeout = 300000)]
    public async Task CustomCreditRuleBaseSubclass_TrainsMlp_ThroughFacade()
    {
        var (trainX, trainY, _) = MakeBlobs(300, seed: 1);
        var (testX, _, testLabels) = MakeBlobs(120, seed: 999);

        var mlp = BuildMlp();
        double before = Accuracy<double>(mlp.Predict, testX, testLabels, MlpClasses);

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
            .ConfigureCreditRule(new CustomDirectRule<double>(seed: 42)) // ICreditRule<T> overload
            .ConfigureLossFunction(new CategoricalCrossEntropyLoss<double>())
            .ConfigureDataLoader(new InMemoryDataLoader<double, Tensor<double>, Tensor<double>>(trainX, trainY))
            .BuildAsync();

        double after = Accuracy<double>(result.Predict, testX, testLabels, MlpClasses);
        Assert.True(after >= 0.60,
            $"Custom CreditRuleBase subclass should learn (held-out {after:F3} >= 0.60, chance {1.0 / MlpClasses:F3}, before {before:F3}).");
    }
}
