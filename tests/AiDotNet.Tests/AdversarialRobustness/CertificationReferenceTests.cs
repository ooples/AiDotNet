using AiDotNet.ActivationFunctions;
using AiDotNet.AdversarialRobustness.Attacks;
using AiDotNet.AdversarialRobustness.CertifiedRobustness;
using AiDotNet.AdversarialRobustness.Defenses;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.AdversarialRobustness;

/// <summary>
/// Reference tests for the certified defenses and AutoAttack against their papers: IBP (Gowal et al. 2018),
/// CROWN (Zhang et al. 2018), randomized smoothing CERTIFY/PREDICT (Cohen et al. 2019), and AutoAttack's
/// APGD schedule, DLR loss and Square schedule (Croce and Hein 2020; autoattack reference code).
/// </summary>
public class CertificationReferenceTests
{
    private const double Tolerance = 1e-9;

    /// <summary>A vector model computing its logits with a delegate.</summary>
    private sealed class DelegateModel : IFullModel<double, Vector<double>, Vector<double>>
    {
        private readonly Func<double[], double[]> _logits;
        private readonly int _inputs;
        private List<int> _activeFeatures;

        public DelegateModel(int inputs, Func<double[], double[]> logits)
        {
            _inputs = inputs;
            _logits = logits;
            _activeFeatures = Enumerable.Range(0, inputs).ToList();
        }

        public ILossFunction<double> DefaultLossFunction => new MeanSquaredErrorLoss<double>();
        public Vector<double> Predict(Vector<double> input) => new Vector<double>(_logits(input.ToArray()));
        public void Train(Vector<double> input, Vector<double> expectedOutput) { }
        public ModelMetadata<double> GetModelMetadata() => new();
        public byte[] Serialize() => Array.Empty<byte>();
        public void Deserialize(byte[] data) { }
        public void SaveModel(string filePath) { }
        public void LoadModel(string filePath) { }
        public void SaveState(Stream stream) { }
        public void LoadState(Stream stream) { }
        public IFullModel<double, Vector<double>, Vector<double>> Clone() => new DelegateModel(_inputs, _logits);
        public IFullModel<double, Vector<double>, Vector<double>> DeepCopy() => new DelegateModel(_inputs, _logits);
        public Vector<double> ComputeGradients(Vector<double> input, Vector<double> target, ILossFunction<double>? lossFunction = null) => new Vector<double>(_inputs);
        public void ApplyGradients(Vector<double> gradients, double learningRate) { }
        public Vector<double> GetParameters() => new Vector<double>(0);
        public void SetParameters(Vector<double> parameters) { }
        public IFullModel<double, Vector<double>, Vector<double>> WithParameters(Vector<double> parameters) => new DelegateModel(_inputs, _logits);
        public long ParameterCount => 0;
        public bool SupportsParameterInitialization => false;
        public IEnumerable<int> GetActiveFeatureIndices() => _activeFeatures;
        public void SetActiveFeatureIndices(IEnumerable<int> featureIndices) => _activeFeatures = featureIndices.ToList();
        public bool IsFeatureUsed(int featureIndex) => _activeFeatures.Contains(featureIndex);
        public Dictionary<string, double> GetFeatureImportance() => Enumerable.Range(0, _inputs).ToDictionary(i => $"Feature{i}", i => 1.0 / _inputs);
        public Vector<double> SanitizeParameters(Vector<double> parameters) => parameters;
        public void Dispose() { }
    }

    #region IBP and CROWN

    // Two inputs, a ReLU hidden layer of two neurons and two logits, stored in the library's [in, out] layout:
    // h = W1^T x + b1, z = W2^T relu(h) + b2. At x = (0.5, 0.5) with an L-infinity radius of 0.1:
    // h0 = x0 + 2 x1 in [1.2, 1.8] (active), h1 = -x0 + x1 - 0.1 in [-0.3, 0.1] (crossing),
    // z0 = a0 - a1, z1 = 0.5 a0 + 2 a1, so the clean logits are (1.5, 0.75) and the margin is 0.5 a0 - 3 a1.
    private static readonly double[,] W1 = { { 1.0, -1.0 }, { 2.0, 1.0 } };
    private static readonly double[] B1 = { 0.0, -0.1 };
    private static readonly double[,] W2 = { { 1.0, 0.5 }, { -1.0, 2.0 } };
    private static readonly double[] B2 = { 0.0, 0.0 };

    private static FeedForwardNeuralNetwork<double> ReferenceNetwork()
    {
        var hidden = new DenseLayer<double>(outputSize: 2, activationFunction: new ReLUActivation<double>(), biasMode: BiasMode.Always);
        var output = new DenseLayer<double>(outputSize: 2, biasMode: BiasMode.Always);
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 2,
            outputSize: 2,
            layers: new List<ILayer<double>> { hidden, output });
        var network = new FeedForwardNeuralNetwork<double>(architecture);
        network.Predict(new Tensor<double>(new[] { 1, 2 }));
        Assign(hidden, W1, B1);
        Assign(output, W2, B2);
        return network;
    }

    private static void Assign(ILayer<double> layer, double[,] weights, double[] biases)
    {
        var w = layer.GetWeights();
        var b = layer.GetBiases();
        Assert.NotNull(w);
        Assert.NotNull(b);
        if (w is null || b is null)
        {
            return;
        }

        Assert.Equal(weights.GetLength(0), w.Shape[0]);
        Assert.Equal(weights.GetLength(1), w.Shape[1]);
        for (int i = 0; i < weights.GetLength(0); i++)
        {
            for (int j = 0; j < weights.GetLength(1); j++)
            {
                w[i, j] = weights[i, j];
            }
        }

        Assert.Equal(biases.Length, b.Length);
        for (int j = 0; j < biases.Length; j++)
        {
            b[j] = biases[j];
        }
    }

    private static Tensor<double> ReferenceInput() => new Tensor<double>(new[] { 1, 2 }, new Vector<double>(new[] { 0.5, 0.5 }));

    private static CertifiedDefenseOptions<double> LInfinityOptions(double epsilon) => new()
    {
        NoiseSigma = epsilon,
        NormType = AdversarialNormType.LInfinity,
        NumSamples = 10,
        Seed = 1
    };

    [Fact(Timeout = 120000)]
    public async Task ReferenceNetwork_PredictsTheHandComputedLogits()
    {
        await Task.Yield();
        var output = ReferenceNetwork().Predict(ReferenceInput());
        Assert.Equal(1.5, output.GetFlat(0), 1e-12);
        Assert.Equal(0.75, output.GetFlat(1), 1e-12);
    }

    [Fact(Timeout = 120000)]
    public async Task Ibp_ReadsDenseWeightsAsInByOut_AndBoundsTheMarginThroughTheLastLayer()
    {
        await Task.Yield();

        // Gowal et al. (2018), Eq. 6: bounding c^T z with c = e_0 - e_1 merged into the last layer gives
        // 0.5 * 1.2 - 3 * 0.1 = 0.3 > 0. Separate logit intervals give z0 >= 1.2 - 0.1 = 1.1 and
        // z1 <= 0.9 + 0.2 = 1.1, which certify nothing.
        var ibp = new IntervalBoundPropagation<double, Tensor<double>, Tensor<double>>(LInfinityOptions(0.1));
        var result = ibp.CertifyPrediction(ReferenceInput(), ReferenceNetwork());

        Assert.Equal(0, result.PredictedClass);
        Assert.True(result.IsCertified);
        Assert.Equal(0.3, (double)result.CertificationDetails["MarginLowerBound"], 1e-12);
        Assert.True((bool)result.CertificationDetails["Sound"]);
        Assert.Equal(1.1, result.LowerBound, 1e-12);
        Assert.Equal(1.8, result.UpperBound, 1e-12);
        Assert.Equal(1.0 / (1.0 + Math.Exp(-0.3)), result.Confidence, 1e-12);
    }

    [Fact(Timeout = 120000)]
    public async Task Crown_UsesTheReluChordAndBeatsIbp()
    {
        await Task.Yield();

        // Zhang et al. (2018): the crossing neuron's upper line is u/(u-l) (h - l) = 0.25 h1 + 0.075; the margin
        // 0.5 a0 - 3 a1 >= 1.25 x0 + 0.25 x1 - 0.15, whose minimum over the ball is 0.6 - 0.1 * 1.5 = 0.45.
        var crown = new CROWNVerification<double, Tensor<double>, Tensor<double>>(LInfinityOptions(0.1));
        var result = crown.CertifyPrediction(ReferenceInput(), ReferenceNetwork());

        Assert.True(result.IsCertified);
        Assert.Equal(0.45, (double)result.CertificationDetails["MarginLowerBound"], 1e-12);

        // z0 = a0 - a1 >= 1.25 x0 + 1.75 x1 - 0.05 -> 1.45 - 0.1 * 3 = 1.15.
        Assert.Equal(1.15, result.LowerBound, 1e-12);
    }

    [Fact(Timeout = 120000)]
    public async Task Crown_MarginBoundIsSoundOverTheBall()
    {
        await Task.Yield();
        var crown = new CROWNVerification<double, Tensor<double>, Tensor<double>>(LInfinityOptions(0.1));
        var result = crown.CertifyPrediction(ReferenceInput(), ReferenceNetwork());
        double bound = (double)result.CertificationDetails["MarginLowerBound"];
        for (int i = 0; i <= 20; i++)
        {
            for (int j = 0; j <= 20; j++)
            {
                double x0 = 0.4 + 0.01 * i;
                double x1 = 0.4 + 0.01 * j;
                double a0 = Math.Max(0, x0 + 2 * x1);
                double a1 = Math.Max(0, -x0 + x1 - 0.1);
                Assert.True(0.5 * a0 - 3 * a1 >= bound - 1e-12);
            }
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Ibp_L2BallBoundsTheFirstLayerWithRowL2Norms()
    {
        await Task.Yield();

        // L2 radius 0.1: h0 = 1.5 +/- 0.1 * sqrt(5), h1 = -0.1 +/- 0.1 * sqrt(2), margin >= 0.5 l0 - 3 u1.
        var options = LInfinityOptions(0.1);
        options.NormType = AdversarialNormType.L2;
        var ibp = new IntervalBoundPropagation<double, Tensor<double>, Tensor<double>>(options);
        var result = ibp.CertifyPrediction(ReferenceInput(), ReferenceNetwork());
        double l0 = 1.5 - 0.1 * Math.Sqrt(5.0);
        double u1 = -0.1 + 0.1 * Math.Sqrt(2.0);
        Assert.Equal(0.5 * l0 - 3 * u1, (double)result.CertificationDetails["MarginLowerBound"], 1e-12);
    }

    [Fact(Timeout = 60000)]
    public async Task Crown_SigmoidRelaxationsEncloseTheFunction()
    {
        await Task.Yield();
        var lower = new Tensor<double>(new[] { 4 }, new Vector<double>(new[] { -3.0, -2.0, 0.5, -0.4 }));
        var upper = new Tensor<double>(new[] { 4 }, new Vector<double>(new[] { 2.0, -0.5, 3.0, 4.0 }));
        foreach (var activation in new[] { ActivationFunction.Sigmoid, ActivationFunction.Tanh })
        {
            var relaxation = BoundNetwork<double>.Relax(activation, lower, upper);
            for (int k = 0; k < 4; k++)
            {
                double l = lower.GetFlat(k), u = upper.GetFlat(k);
                for (int s = 0; s <= 200; s++)
                {
                    double z = l + (u - l) * s / 200.0;
                    double f = activation == ActivationFunction.Tanh ? Math.Tanh(z) : 1.0 / (1.0 + Math.Exp(-z));
                    double below = relaxation.LowerSlope.GetFlat(k) * z + relaxation.LowerIntercept.GetFlat(k);
                    double above = relaxation.UpperSlope.GetFlat(k) * z + relaxation.UpperIntercept.GetFlat(k);
                    Assert.True(below <= f + 1e-9, $"{activation} lower line above f at {z} on [{l}, {u}]");
                    Assert.True(above >= f - 1e-9, $"{activation} upper line below f at {z} on [{l}, {u}]");
                }
            }
        }
    }

    #endregion

    #region Randomized smoothing

    private static DelegateModel ConstantClass(int inputs, int classIndex, int classes) =>
        new(inputs, _ => Enumerable.Range(0, classes).Select(c => c == classIndex ? 1.0 : 0.0).ToArray());

    [Fact(Timeout = 60000)]
    public async Task Certify_UsesTheOneSidedClopperPearsonBound()
    {
        await Task.Yield();

        // Cohen et al. (2019), CERTIFY: with all n samples on c_A, LowerConfBound(n, n, 1 - alpha) = alpha^(1/n)
        // (one-sided), not the two-sided (alpha / 2)^(1/n), and R = sigma * Phi^-1(p_A_lower).
        var options = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.5,
            NumSamples = 100,
            SelectionSamples = 10,
            ConfidenceLevel = 0.999,
            Seed = 3
        };
        var smoothing = new RandomizedSmoothing<double, Vector<double>, Vector<double>>(options);
        var result = smoothing.CertifyPrediction(new Vector<double>(new[] { 0.2, 0.7 }), ConstantClass(2, 1, 3));

        double pLower = Math.Pow(0.001, 1.0 / 100);
        Assert.Equal(1, result.PredictedClass);
        Assert.True(result.IsCertified);
        Assert.Equal(pLower, result.LowerBound, 1e-6);
        Assert.Equal(0.5 * InverseNormal(pLower), result.CertifiedRadius, 1e-5);
    }

    [Fact(Timeout = 60000)]
    public async Task Certify_DoesNotClipTheNoisyInput()
    {
        await Task.Yield();

        // The Gaussian is added without clipping: at x = 1 a model that flips past 1 sees about half the samples
        // flip. Clipping to [0, 1] would make every sample class 0.
        var options = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.5,
            NumSamples = 200,
            SelectionSamples = 20,
            Seed = 11
        };
        var smoothing = new RandomizedSmoothing<double, Vector<double>, Vector<double>>(options);
        var model = new DelegateModel(1, x => x[0] > 1.0 ? new[] { 0.0, 1.0 } : new[] { 1.0, 0.0 });
        var result = smoothing.CertifyPrediction(new Vector<double>(new[] { 1.0 }), model);

        Assert.False(result.IsCertified);
        Assert.True((int)result.CertificationDetails["TopClassCount"] < 160);
    }

    [Fact(Timeout = 60000)]
    public async Task Predict_TwoSidedBinomialPValueMatchesTheExactTest()
    {
        await Task.Yield();

        // scipy.stats.binomtest(7, 10, 0.5).pvalue = 2 * P(X >= 7) = 2 * 176 / 1024.
        Assert.Equal(0.34375, RandomizedSmoothing<double, Vector<double>, Vector<double>>.TwoSidedBinomialPValue(7, 10), 1e-9);
        Assert.Equal(1.0, RandomizedSmoothing<double, Vector<double>, Vector<double>>.TwoSidedBinomialPValue(5, 10), 1e-9);
        Assert.Equal(2.0 / 1024, RandomizedSmoothing<double, Vector<double>, Vector<double>>.TwoSidedBinomialPValue(10, 10), 1e-9);
    }

    [Fact(Timeout = 60000)]
    public async Task EvaluateCertifiedAccuracy_AbstentionIsNeverCertifiedCorrect()
    {
        await Task.Yield();

        // A model split evenly between two classes makes CERTIFY abstain (p_A_lower < 0.5); the abstained top
        // class must not count as certified correct even at radius 0.
        var options = new CertifiedDefenseOptions<double> { NoiseSigma = 0.5, NumSamples = 100, SelectionSamples = 20, Seed = 5 };
        var smoothing = new RandomizedSmoothing<double, Vector<double>, Vector<double>>(options);
        var model = new DelegateModel(1, x => x[0] > 0.0 ? new[] { 0.0, 1.0 } : new[] { 1.0, 0.0 });
        var inputs = new[] { new Vector<double>(new[] { 0.0 }) };
        var top = smoothing.CertifyPrediction(inputs[0], model).PredictedClass;
        var labels = new[] { new Vector<double>(new[] { top == 0 ? 1.0 : 0.0, top == 1 ? 1.0 : 0.0 }) };

        var metrics = smoothing.EvaluateCertifiedAccuracy(inputs, labels, model, 0.0);
        Assert.Equal(0.0, metrics.CertifiedAccuracy, Tolerance);
        Assert.Equal(0.0, metrics.CertificationRate, Tolerance);
    }

    [Fact(Timeout = 60000)]
    public async Task AdaptiveSmoothing_CertifiesWithTheComposedSigma()
    {
        await Task.Yield();

        // The probe step (k queries at sigma_min) composes with the certification step into one Gaussian
        // mechanism of noise (k / sigma_min^2 + 1 / sigma^2)^(-1/2); the radius uses it, not sigma(x).
        var options = new CertifiedDefenseOptions<double> { NumSamples = 100, SelectionSamples = 10, ConfidenceLevel = 0.999, Seed = 7 };
        var adaptive = new AdaptiveRandomizedSmoothing<double, Vector<double>, Vector<double>>(options, 0.1, 1.0, 20);
        var result = adaptive.CertifyPrediction(new Vector<double>(new[] { 0.3 }), ConstantClass(1, 0, 2));

        double sigma = (double)result.CertificationDetails["AdaptiveSigma"];
        double effective = 1.0 / Math.Sqrt(20 / (0.1 * 0.1) + 1 / (sigma * sigma));
        Assert.Equal(effective, (double)result.CertificationDetails["EffectiveSigma"], 1e-12);
        Assert.Equal(effective * InverseNormal(Math.Pow(0.001, 1.0 / 100)), result.CertifiedRadius, 1e-5);
    }

    /// <summary>Acklam's inverse normal CDF refined by one Halley step (absolute error below 1e-12).</summary>
    private static double InverseNormal(double p)
    {
        double[] a = { -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00 };
        double[] b = { -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01 };
        double[] c = { -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00 };
        double[] d = { 7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00 };
        double x;
        if (p < 0.02425)
        {
            double q = Math.Sqrt(-2 * Math.Log(p));
            x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
        }
        else if (p <= 1 - 0.02425)
        {
            double q = p - 0.5, r = q * q;
            x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
        }
        else
        {
            double q = Math.Sqrt(-2 * Math.Log(1 - p));
            x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
        }

        double e = 0.5 * Erfc(-x / Math.Sqrt(2)) - p;
        double u = e * Math.Sqrt(2 * Math.PI) * Math.Exp(x * x / 2);
        return x - u / (1 + x * u / 2);
    }

    /// <summary>Complementary error function (Numerical Recipes erfcc, relative error below 1.2e-7).</summary>
    private static double Erfc(double x)
    {
        double z = Math.Abs(x);
        double t = 1 / (1 + 0.5 * z);
        double r = t * Math.Exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806
            + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
        return x >= 0 ? r : 2 - r;
    }

    #endregion

    #region AutoAttack

    [Fact(Timeout = 60000)]
    public async Task Apgd_CheckpointsFollowThePaperSchedule()
    {
        await Task.Yield();

        // Croce and Hein (2020), Sec. 3.2: p = 0, 0.22, 0.41, 0.57, 0.70, 0.80, 0.87, 0.93, 0.99 for N = 100.
        Assert.Equal(new[] { 22, 41, 57, 70, 80, 87, 93, 99 },
            AutoAttack<double, Vector<double>, Vector<double>>.ApgdCheckpoints(100));
    }

    [Fact(Timeout = 60000)]
    public async Task Square_FractionHalvesAtTheReferenceSchedule()
    {
        await Task.Yield();

        // autoattack square.p_selection on 10000-rescaled iterations.
        Assert.Equal(0.8, AutoAttack<double, Vector<double>, Vector<double>>.SquareFraction(0.8, 1, 5000), Tolerance);
        Assert.Equal(0.4, AutoAttack<double, Vector<double>, Vector<double>>.SquareFraction(0.8, 10, 5000), Tolerance);
        Assert.Equal(0.025, AutoAttack<double, Vector<double>, Vector<double>>.SquareFraction(0.8, 1000, 5000), Tolerance);
        Assert.Equal(0.8 / 512, AutoAttack<double, Vector<double>, Vector<double>>.SquareFraction(0.8, 4999, 5000), Tolerance);
        Assert.Equal((1, 3), AutoAttack<double, Vector<double>, Vector<double>>.SquareWindow(0.1, 1, 32, l2: false));
        Assert.Equal((5, 5), AutoAttack<double, Vector<double>, Vector<double>>.SquareWindow(0.1, 16, 16, l2: false));
        Assert.Equal((5, 5), AutoAttack<double, Vector<double>, Vector<double>>.SquareWindow(0.07, 16, 16, l2: true));
    }

    [Fact(Timeout = 60000)]
    public async Task Dlr_ValueAndGradientMatchTheDefinition()
    {
        await Task.Yield();

        // DLR(z, y) = -(z_y - max_{i != y} z_i) / (z_pi1 - z_pi3): z = (3, 1, 2, 0), y = 0 -> -(3 - 2) / (3 - 1).
        var attack = new AutoAttack<double, Vector<double>, Vector<double>>(new AdversarialAttackOptions<double>());
        var logits = new Vector<double>(new[] { 3.0, 1.0, 2.0, 0.0 });
        var (value, gradient) = attack.Loss(logits, AutoAttack<double, Vector<double>, Vector<double>>.LossKind.Dlr, 0, -1);
        Assert.Equal(-0.5, value, 1e-9);

        const double h = 1e-6;
        for (int i = 0; i < logits.Length; i++)
        {
            var plus = new Vector<double>(logits.ToArray());
            var minus = new Vector<double>(logits.ToArray());
            plus[i] += h;
            minus[i] -= h;
            double numeric = (attack.Loss(plus, AutoAttack<double, Vector<double>, Vector<double>>.LossKind.Dlr, 0, -1).Value
                - attack.Loss(minus, AutoAttack<double, Vector<double>, Vector<double>>.LossKind.Dlr, 0, -1).Value) / (2 * h);
            Assert.Equal(numeric, gradient[i], 1e-6);
        }

        // Targeted DLR: -(z_y - z_t) / (z_pi1 - (z_pi3 + z_pi4) / 2) with t = 2 -> -(3 - 2) / (3 - 0.5).
        var targeted = attack.Loss(logits, AutoAttack<double, Vector<double>, Vector<double>>.LossKind.TargetedDlr, 0, 2);
        Assert.Equal(-1.0 / 2.5, targeted.Value, 1e-9);
    }

    [Fact(Timeout = 120000)]
    public async Task AutoAttack_ReturnsMisclassifiedInputsUnchanged()
    {
        await Task.Yield();

        // Worst-case accounting: a point the model already misclassifies counts as broken and is not attacked.
        var attack = new AutoAttack<double, Vector<double>, Vector<double>>(new AdversarialAttackOptions<double> { Epsilon = 0.3, RandomSeed = 1 });
        var model = new DelegateModel(2, x => new[] { x[0], x[1] });
        var input = new Vector<double>(new[] { 0.2, 0.6 });
        var label = new Vector<double>(new[] { 1.0, 0.0 });
        var adversarial = attack.GenerateAdversarialExample(input, label, model);
        Assert.Equal(input.ToArray(), adversarial.ToArray());
    }

    [Fact(Timeout = 300000)]
    public async Task AutoAttack_FoolsALinearModelWithinTheLInfinityBall()
    {
        await Task.Yield();
        var attack = new AutoAttack<double, Vector<double>, Vector<double>>(new AdversarialAttackOptions<double>
        {
            Epsilon = 0.2,
            NormType = AdversarialNormType.LInfinity,
            Iterations = 20,
            RandomSeed = 2
        });
        var model = new DelegateModel(2, x => new[] { x[0] - x[1] + 0.1, x[1] - x[0] });
        var input = new Vector<double>(new[] { 0.5, 0.5 });
        var adversarial = attack.GenerateAdversarialExample(input, new Vector<double>(new[] { 1.0, 0.0 }), model);

        var logits = model.Predict(adversarial);
        Assert.True(logits[1] > logits[0]);
        for (int i = 0; i < 2; i++)
        {
            Assert.True(Math.Abs(adversarial[i] - input[i]) <= 0.2 + 1e-12);
            Assert.InRange(adversarial[i], 0.0, 1.0);
        }
    }

    #endregion
}
