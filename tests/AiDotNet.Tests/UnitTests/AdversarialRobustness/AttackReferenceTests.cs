using AiDotNet.AdversarialRobustness.Attacks;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.UnitTests.AdversarialRobustness;

/// <summary>
/// Reference tests for the attack algorithms against their papers and torchattacks:
/// PGD (Madry et al. 2018; torchattacks PGD / PGDL2), Carlini-Wagner L2 (Carlini and Wagner 2017;
/// torchattacks CW), and RobustBench-style robustness accounting.
/// </summary>
public class AttackReferenceTests
{
    /// <summary>
    /// Linear logit model: logits = W x + b. Its input gradients are known in closed form.
    /// </summary>
    private sealed class LinearLogitModel : IFullModel<double, Vector<double>, Vector<double>>
    {
        private readonly double[,] _weights;
        private readonly double[] _bias;
        private List<int> _activeFeatures;

        public LinearLogitModel(double[,] weights, double[] bias)
        {
            _weights = weights;
            _bias = bias;
            _activeFeatures = Enumerable.Range(0, weights.GetLength(1)).ToList();
        }

        public int Classes => _weights.GetLength(0);

        public int Inputs => _weights.GetLength(1);

        public double Weight(int c, int i) => _weights[c, i];

        public double[] Logits(double[] x)
        {
            var z = new double[Classes];
            for (int c = 0; c < Classes; c++)
            {
                z[c] = _bias[c];
                for (int i = 0; i < Inputs; i++)
                {
                    z[c] += _weights[c, i] * x[i];
                }
            }

            return z;
        }

        public ILossFunction<double> DefaultLossFunction => new MeanSquaredErrorLoss<double>();

        public Vector<double> Predict(Vector<double> input) => new Vector<double>(Logits(input.ToArray()));

        public void Train(Vector<double> input, Vector<double> expectedOutput) { }
        public ModelMetadata<double> GetModelMetadata() => new();
        public byte[] Serialize() => Array.Empty<byte>();
        public void Deserialize(byte[] data) { }
        public void SaveModel(string filePath) { }
        public void LoadModel(string filePath) { }
        public void SaveState(Stream stream) { }
        public void LoadState(Stream stream) { }
        public IFullModel<double, Vector<double>, Vector<double>> Clone() => new LinearLogitModel(_weights, _bias);
        public IFullModel<double, Vector<double>, Vector<double>> DeepCopy() => new LinearLogitModel(_weights, _bias);
        public Vector<double> ComputeGradients(Vector<double> input, Vector<double> target, ILossFunction<double>? lossFunction = null) => new Vector<double>(Inputs);
        public void ApplyGradients(Vector<double> gradients, double learningRate) { }
        public Vector<double> GetParameters() => new Vector<double>(Inputs * Classes);
        public void SetParameters(Vector<double> parameters) { }
        public IFullModel<double, Vector<double>, Vector<double>> WithParameters(Vector<double> parameters) => new LinearLogitModel(_weights, _bias);
        public long ParameterCount => Inputs * Classes;
        public bool SupportsParameterInitialization => ParameterCount > 0;
        public IEnumerable<int> GetActiveFeatureIndices() => _activeFeatures;
        public void SetActiveFeatureIndices(IEnumerable<int> featureIndices) => _activeFeatures = featureIndices.ToList();
        public bool IsFeatureUsed(int featureIndex) => _activeFeatures.Contains(featureIndex);
        public Dictionary<string, double> GetFeatureImportance() => Enumerable.Range(0, Inputs).ToDictionary(i => $"Feature{i}", i => 1.0 / Inputs);
        public Vector<double> SanitizeParameters(Vector<double> parameters) => parameters;
        public void Dispose() { }
    }

    /// <summary>
    /// An attack that returns a fixed transformation of the input, or throws.
    /// </summary>
    private sealed class ScriptedAttack : AdversarialAttackBase<double, Vector<double>, Vector<double>>
    {
        private readonly Func<Vector<double>, Vector<double>> _script;

        public ScriptedAttack(Func<Vector<double>, Vector<double>> script)
            : base(new AdversarialAttackOptions<double>())
        {
            _script = script;
        }

        public override Vector<double> GenerateAdversarialExample(
            Vector<double> input, Vector<double> trueLabel, IFullModel<double, Vector<double>, Vector<double>> targetModel)
            => _script(input);

        public override Vector<double> CalculatePerturbation(Vector<double> original, Vector<double> adversarial)
            => new Vector<double>(original.Length);
    }

    private static Vector<double> OneHot(int index, int count)
    {
        var label = new Vector<double>(count);
        label[index] = 1.0;
        return label;
    }

    private static int ArgMax(double[] values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] > values[best])
            {
                best = i;
            }
        }

        return best;
    }

    [Fact(Timeout = 60000)]
    public async Task PGD_L2_StepIsNormalizedGradientNotSign()
    {
        // torchattacks PGDL2: adv = x + alpha * g / ||g||_2. A sign step would move every coordinate by alpha.
        var model = new LinearLogitModel(
            new double[,] { { 1.0, -2.0, 0.5, 3.0 }, { -1.0, 0.25, 2.0, -0.5 } },
            new double[] { 0.1, -0.2 });
        var x = new[] { 0.5, 0.5, 0.5, 0.5 };
        var label = OneHot(0, 2);
        var options = new AdversarialAttackOptions<double>
        {
            Epsilon = 10.0,
            StepSize = 0.1,
            Iterations = 1,
            NormType = AdversarialNormType.L2,
            UseRandomStart = false
        };
        var attack = new PGDAttack<double, Vector<double>, Vector<double>>(options);

        var adversarial = attack.GenerateAdversarialExample(new Vector<double>(x), label, model);

        // MSE gradient of mean((Wx + b - y)^2) is (2 / C) W^T (Wx + b - y).
        var z = model.Logits(x);
        var gradient = new double[model.Inputs];
        for (int i = 0; i < model.Inputs; i++)
        {
            for (int c = 0; c < model.Classes; c++)
            {
                gradient[i] += 2.0 / model.Classes * model.Weight(c, i) * (z[c] - label[c]);
            }
        }

        double norm = Math.Sqrt(gradient.Sum(g => g * g));
        for (int i = 0; i < model.Inputs; i++)
        {
            Assert.Equal(x[i] + 0.1 * gradient[i] / norm, adversarial[i], 6);
        }
    }

    [Fact(Timeout = 60000)]
    public async Task PGD_LInfinityRandomStart_IsUniformInEpsilonBall()
    {
        // Madry et al. / torchattacks PGD: delta ~ U(-eps, eps). P(|delta| > 0.8 eps) = 0.2 for a uniform start,
        // but about 0.42 for a Gaussian with std eps clipped to the ball.
        const int n = 4000;
        const double epsilon = 0.25;
        var model = new LinearLogitModel(new double[2, n], new double[] { 0.0, 0.0 });
        var x = Enumerable.Repeat(0.5, n).ToArray();
        var options = new AdversarialAttackOptions<double>
        {
            Epsilon = epsilon,
            Iterations = 0,
            UseRandomStart = true,
            RandomSeed = 7
        };
        var attack = new PGDAttack<double, Vector<double>, Vector<double>>(options);

        var adversarial = attack.GenerateAdversarialExample(new Vector<double>(x), OneHot(0, 2), model);

        int outer = 0;
        for (int i = 0; i < n; i++)
        {
            double delta = adversarial[i] - 0.5;
            Assert.InRange(delta, -epsilon - 1e-12, epsilon + 1e-12);
            if (Math.Abs(delta) > 0.8 * epsilon)
            {
                outer++;
            }
        }

        Assert.InRange(outer / (double)n, 0.17, 0.23);
    }

    [Fact(Timeout = 60000)]
    public async Task PGD_RandomStart_DiffersAcrossCallsOnOneInstance()
    {
        // Each call draws a fresh start from the attack's seeded stream instead of reusing the same seed.
        var model = new LinearLogitModel(new double[2, 8], new double[] { 0.0, 0.0 });
        var options = new AdversarialAttackOptions<double>
        {
            Epsilon = 0.1,
            Iterations = 0,
            UseRandomStart = true,
            RandomSeed = 3
        };
        var attack = new PGDAttack<double, Vector<double>, Vector<double>>(options);
        var x = new Vector<double>(Enumerable.Repeat(0.5, 8).ToArray());

        var first = attack.GenerateAdversarialExample(x, OneHot(0, 2), model);
        var second = attack.GenerateAdversarialExample(x, OneHot(0, 2), model);

        Assert.Contains(Enumerable.Range(0, 8), i => Math.Abs(first[i] - second[i]) > 1e-9);
    }

    [Fact(Timeout = 60000)]
    public async Task CW_Untargeted_FindsMisclassifiedExampleWithSmallL2()
    {
        // Class 0 wins by 0.1 at x = (0.5, 0.5). The minimal L2 change that flips it moves the two pixels
        // by 0.025 each in opposite directions (||delta|| ~= 0.035). The paper's loss lowers the true logit.
        var model = new LinearLogitModel(
            new double[,] { { 1.0, -1.0 }, { -1.0, 1.0 } },
            new double[] { 0.1, 0.0 });
        var options = new AdversarialAttackOptions<double> { Iterations = 100, CWConstant = 1.0 };
        var attack = new CWAttack<double, Vector<double>, Vector<double>>(options);
        var x = new[] { 0.5, 0.5 };

        var adversarial = attack.GenerateAdversarialExample(new Vector<double>(x), OneHot(0, 2), model);

        var adv = adversarial.ToArray();
        Assert.Equal(1, ArgMax(model.Logits(adv)));
        double l2 = Math.Sqrt((adv[0] - x[0]) * (adv[0] - x[0]) + (adv[1] - x[1]) * (adv[1] - x[1]));
        Assert.InRange(l2, 0.035, 0.1);
    }

    [Fact(Timeout = 60000)]
    public async Task CW_Targeted_ReachesTargetClassAboveEveryOtherLogit()
    {
        // f = max(max_{i != t} Z_i - Z_t, -kappa): the target must beat every other class, not only
        // the classes other than the true label.
        var model = new LinearLogitModel(
            new double[,] { { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 }, { 0.0, 0.0, 1.0 } },
            new double[] { 0.0, 0.0, -0.05 });
        var options = new AdversarialAttackOptions<double>
        {
            Iterations = 100,
            IsTargeted = true,
            TargetClass = 2
        };
        var attack = new CWAttack<double, Vector<double>, Vector<double>>(options);
        var x = new[] { 0.5, 0.5, 0.5 };

        var adversarial = attack.GenerateAdversarialExample(new Vector<double>(x), OneHot(0, 3), model);

        Assert.Equal(2, ArgMax(model.Logits(adversarial.ToArray())));
    }

    [Fact(Timeout = 60000)]
    public async Task EvaluateRobustness_UsesRobustBenchAccounting()
    {
        // logits = x. Sample A (label 0) is correct and the swap attack fools it; sample B (label 1) is
        // already misclassified, so it is not robust even though the swapped input lands on its label.
        var model = new LinearLogitModel(new double[,] { { 1.0, 0.0 }, { 0.0, 1.0 } }, new double[] { 0.0, 0.0 });
        var result = new AiModelResult<double, Vector<double>, Vector<double>> { Model = model };
        var attack = new ScriptedAttack(v => new Vector<double>(new[] { v[1], v[0] }));
        var inputs = new[] { new Vector<double>(new[] { 1.0, 0.0 }), new Vector<double>(new[] { 1.0, 0.0 }) };
        var labels = new[] { OneHot(0, 2), OneHot(1, 2) };

        var stats = result.EvaluateRobustness(inputs, labels, attack);

        Assert.Equal(0.5, stats.CleanAccuracy, 12);
        Assert.Equal(0.0, stats.AdversarialAccuracy, 12);
        Assert.Equal(1.0, stats.AttackSuccessRate, 12);
        Assert.Equal(Math.Sqrt(2.0), stats.AveragePerturbationSize, 12);
        Assert.Equal(AdversarialNormType.LInfinity, stats.NormType);
    }

    [Fact(Timeout = 60000)]
    public async Task EvaluateRobustness_FailingAttackIsNotCountedAsDefended()
    {
        var model = new LinearLogitModel(new double[,] { { 1.0, 0.0 }, { 0.0, 1.0 } }, new double[] { 0.0, 0.0 });
        var result = new AiModelResult<double, Vector<double>, Vector<double>> { Model = model };
        var attack = new ScriptedAttack(_ => throw new InvalidOperationException("attack failed"));
        var inputs = new[] { new Vector<double>(new[] { 1.0, 0.0 }) };
        var labels = new[] { OneHot(0, 2) };

        Assert.Throws<InvalidOperationException>(() => result.EvaluateRobustness(inputs, labels, attack));
    }
}
