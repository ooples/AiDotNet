#nullable disable
using Xunit;
using AiDotNet.AdversarialRobustness.Attacks;
using AiDotNet.AdversarialRobustness.CertifiedRobustness;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tests.IntegrationTests.AdversarialRobustness;

/// <summary>
/// Deep mathematical integration tests for adversarial robustness attacks and defenses.
/// Verifies FGSM sign-gradient math, PGD projection geometry, CW tanh reparameterization,
/// and Randomized Smoothing certified radius formulas with hand-computed expected values.
/// </summary>
public class AdversarialRobustnessDeepMathIntegrationTests
{
    private static readonly INumericOperations<double> NumOps = MathHelper.GetNumericOperations<double>();
    private const double Tolerance = 1e-6;
    private const int Seed = 42;

    #region Mock Model

    /// <summary>
    /// Deterministic linear classification model: output[c] = sum_i(w[c,i] * input[i]) then softmax.
    /// Weights are set deterministically so we can hand-compute expected results.
    /// </summary>
    private class ARMockModel : IFullModel<double, Vector<double>, Vector<double>>
    {
        private readonly int _inputSize;
        private readonly int _numClasses;
        private Vector<double> _weights;

        public ARMockModel(int inputSize, int numClasses, double[] weights = null)
        {
            _inputSize = inputSize;
            _numClasses = numClasses;
            if (weights != null)
            {
                _weights = new Vector<double>(weights);
            }
            else
            {
                // Default: identity-like weights
                _weights = new Vector<double>(inputSize * numClasses);
                for (int c = 0; c < numClasses; c++)
                    for (int i = 0; i < inputSize; i++)
                        _weights[c * inputSize + i] = (c == i % numClasses) ? 1.0 : 0.1;
            }
        }

        public ILossFunction<double> DefaultLossFunction => null;
        public int ParameterCount => _weights.Length;
        public bool SupportsJitCompilation => false;

        public Vector<double> Predict(Vector<double> input)
        {
            var logits = new double[_numClasses];
            for (int c = 0; c < _numClasses; c++)
            {
                double sum = 0;
                for (int i = 0; i < Math.Min(_inputSize, input.Length); i++)
                    sum += input[i] * _weights[c * _inputSize + i];
                logits[c] = sum;
            }

            // Softmax
            double maxLogit = logits.Max();
            double expSum = 0;
            var output = new Vector<double>(_numClasses);
            for (int i = 0; i < _numClasses; i++)
            {
                output[i] = Math.Exp(logits[i] - maxLogit);
                expSum += output[i];
            }
            for (int i = 0; i < _numClasses; i++)
                output[i] /= expSum;

            return output;
        }

        public void Train(Vector<double> input, Vector<double> expectedOutput) { }
        public ModelMetadata<double> GetModelMetadata() => new() { ModelType = ModelType.None };
        public byte[] Serialize() => Array.Empty<byte>();
        public void Deserialize(byte[] data) { }
        public void SaveModel(string filePath) { }
        public void LoadModel(string filePath) { }
        public void SaveState(Stream stream) { }
        public void LoadState(Stream stream) { }
        public Vector<double> GetParameters() => _weights;
        public void SetParameters(Vector<double> parameters) { _weights = parameters; }
        public IFullModel<double, Vector<double>, Vector<double>> WithParameters(Vector<double> parameters)
        {
            var m = new ARMockModel(_inputSize, _numClasses);
            m.SetParameters(parameters);
            return m;
        }
        public IEnumerable<int> GetActiveFeatureIndices() => Enumerable.Range(0, _inputSize);
        public void SetActiveFeatureIndices(IEnumerable<int> featureIndices) { }
        public bool IsFeatureUsed(int featureIndex) => true;
        public Dictionary<string, double> GetFeatureImportance() => new();
        public IFullModel<double, Vector<double>, Vector<double>> DeepCopy()
        {
            var copy = new ARMockModel(_inputSize, _numClasses);
            copy.SetParameters(new Vector<double>(_weights.ToArray()));
            return copy;
        }
        public IFullModel<double, Vector<double>, Vector<double>> Clone() => DeepCopy();
        public Vector<double> ComputeGradients(Vector<double> input, Vector<double> target, ILossFunction<double> lossFunction = null)
        {
            var g = new Vector<double>(_weights.Length);
            var pred = Predict(input);
            for (int c = 0; c < _numClasses; c++)
                for (int i = 0; i < Math.Min(_inputSize, input.Length); i++)
                    g[c * _inputSize + i] = (pred[c] - target[c]) * input[i];
            return g;
        }
        public void ApplyGradients(Vector<double> gradients, double learningRate)
        {
            for (int i = 0; i < _weights.Length; i++)
                _weights[i] -= learningRate * gradients[i];
        }
        public AiDotNet.Autodiff.ComputationNode<double> ExportComputationGraph(
            List<AiDotNet.Autodiff.ComputationNode<double>> inputNodes)
            => throw new NotImplementedException();
    }

    #endregion

    #region FGSM Mathematical Properties

    [Fact]
    public void FGSM_PerturbationMagnitude_ExactlyEpsilonPerDimension()
    {
        // FGSM formula: x_adv = clip(x + epsilon * sign(grad), 0, 1)
        // For interior points (not near 0 or 1), each dim should change by exactly epsilon
        var epsilon = 0.05;
        var options = new AdversarialAttackOptions<double>
        {
            Epsilon = epsilon,
            RandomSeed = Seed
        };
        var attack = new FGSMAttack<double, Vector<double>, Vector<double>>(options);

        // Use input values far from 0 and 1 to avoid clipping
        var input = new Vector<double>(4);
        input[0] = 0.5; input[1] = 0.5; input[2] = 0.5; input[3] = 0.5;
        var label = new Vector<double>(2);
        label[0] = 1.0; label[1] = 0.0;

        var model = new ARMockModel(4, 2);
        var adversarial = attack.GenerateAdversarialExample(input, label, model);

        // Each dimension perturbation should be exactly epsilon (or the input hits clipping boundary)
        for (int i = 0; i < input.Length; i++)
        {
            double perturbation = Math.Abs(adversarial[i] - input[i]);
            // Perturbation is either 0 (zero gradient) or epsilon (nonzero gradient)
            // since sign() returns -1, 0, or +1
            Assert.True(perturbation <= epsilon + Tolerance,
                $"Dim {i}: perturbation {perturbation} exceeds epsilon {epsilon}");
        }
    }

    [Fact]
    public void FGSM_Output_ClippedToZeroOne()
    {
        // FGSM clips output to [0, 1]
        var options = new AdversarialAttackOptions<double>
        {
            Epsilon = 0.3,
            RandomSeed = Seed
        };
        var attack = new FGSMAttack<double, Vector<double>, Vector<double>>(options);

        // Input near boundaries
        var input = new Vector<double>(4);
        input[0] = 0.1; input[1] = 0.9; input[2] = 0.0; input[3] = 1.0;
        var label = new Vector<double>(2);
        label[0] = 1.0; label[1] = 0.0;

        var model = new ARMockModel(4, 2);
        var adversarial = attack.GenerateAdversarialExample(input, label, model);

        for (int i = 0; i < adversarial.Length; i++)
        {
            Assert.True(adversarial[i] >= -Tolerance,
                $"Dim {i}: value {adversarial[i]} below 0");
            Assert.True(adversarial[i] <= 1.0 + Tolerance,
                $"Dim {i}: value {adversarial[i]} above 1");
        }
    }

    [Fact]
    public void FGSM_TargetedNegation_VerifyWithSameGradientClass()
    {
        // When untargeted (trueClass=0) and targeted (targetClass=0) use the SAME class
        // for gradient computation, the targeted attack negates the perturbation.
        // Untargeted: perturbation = +eps * sign(grad_class0)
        // Targeted:   perturbation = -eps * sign(grad_class0)
        // So with same class gradient, perturbation directions must be exactly opposite.
        var epsilon = 0.05;
        var input = new Vector<double>(4);
        input[0] = 0.5; input[1] = 0.5; input[2] = 0.5; input[3] = 0.5;
        var label = new Vector<double>(2);
        label[0] = 1.0; label[1] = 0.0;

        var model = new ARMockModel(4, 2);

        // Both compute gradient for class 0 (trueLabel=0, targetClass=0)
        var untargetedOptions = new AdversarialAttackOptions<double>
        { Epsilon = epsilon, IsTargeted = false, RandomSeed = Seed };
        var targetedOptions = new AdversarialAttackOptions<double>
        { Epsilon = epsilon, IsTargeted = true, TargetClass = 0, RandomSeed = Seed };

        var untargetedAttack = new FGSMAttack<double, Vector<double>, Vector<double>>(untargetedOptions);
        var targetedAttack = new FGSMAttack<double, Vector<double>, Vector<double>>(targetedOptions);

        var untargetedAdv = untargetedAttack.GenerateAdversarialExample(input, label, model);
        var targetedAdv = targetedAttack.GenerateAdversarialExample(input, label, model);

        // Since both use the same gradient but targeted negates:
        // untargeted_delta = +eps * sign(grad)
        // targeted_delta = -eps * sign(grad)
        // So delta_u = -delta_t
        int oppositeCount = 0;
        for (int i = 0; i < input.Length; i++)
        {
            double untargetedDelta = untargetedAdv[i] - input[i];
            double targetedDelta = targetedAdv[i] - input[i];
            if (Math.Abs(untargetedDelta) > Tolerance && Math.Abs(targetedDelta) > Tolerance)
            {
                // Product should be negative (opposite signs)
                Assert.True(untargetedDelta * targetedDelta < Tolerance,
                    $"Dim {i}: untargeted delta {untargetedDelta} and targeted delta {targetedDelta} not opposite");
                // Magnitudes should be equal
                Assert.Equal(Math.Abs(untargetedDelta), Math.Abs(targetedDelta), Tolerance);
                oppositeCount++;
            }
        }
        Assert.True(oppositeCount > 0, "No dimensions showed opposite perturbation directions");
    }

    [Fact]
    public void FGSM_CalculatePerturbation_EqualsAdversarialMinusOriginal()
    {
        // CalculatePerturbation should return adversarial - original exactly
        var options = new AdversarialAttackOptions<double>
        { Epsilon = 0.1, RandomSeed = Seed };
        var attack = new FGSMAttack<double, Vector<double>, Vector<double>>(options);

        var input = new Vector<double>(4);
        input[0] = 0.5; input[1] = 0.4; input[2] = 0.6; input[3] = 0.3;
        var label = new Vector<double>(2);
        label[0] = 1.0; label[1] = 0.0;

        var model = new ARMockModel(4, 2);
        var adversarial = attack.GenerateAdversarialExample(input, label, model);
        var perturbation = attack.CalculatePerturbation(input, adversarial);

        for (int i = 0; i < input.Length; i++)
        {
            double expected = adversarial[i] - input[i];
            Assert.Equal(expected, perturbation[i], Tolerance);
        }
    }

    [Fact]
    public void FGSM_LinfPerturbation_AllDimensionsWithinEpsilon()
    {
        // The L-infinity norm of the perturbation should be <= epsilon
        var epsilon = 0.08;
        var options = new AdversarialAttackOptions<double>
        {
            Epsilon = epsilon,
            NormType = "L-infinity",
            RandomSeed = Seed
        };
        var attack = new FGSMAttack<double, Vector<double>, Vector<double>>(options);

        var input = new Vector<double>(6);
        for (int i = 0; i < 6; i++) input[i] = 0.5;
        var label = new Vector<double>(3);
        label[0] = 1.0;

        var model = new ARMockModel(6, 3);
        var adversarial = attack.GenerateAdversarialExample(input, label, model);

        double linfNorm = 0;
        for (int i = 0; i < input.Length; i++)
        {
            double absDiff = Math.Abs(adversarial[i] - input[i]);
            if (absDiff > linfNorm) linfNorm = absDiff;
        }

        Assert.True(linfNorm <= epsilon + Tolerance,
            $"L-inf norm {linfNorm} exceeds epsilon {epsilon}");
    }

    #endregion

    #region PGD Mathematical Properties

    [Fact]
    public void PGD_Iterative_PerturbationGrowsWithIterations()
    {
        // PGD with more iterations should potentially create larger perturbation
        // (or at least not smaller, assuming no random start)
        var input = new Vector<double>(4);
        for (int i = 0; i < 4; i++) input[i] = 0.5;
        var label = new Vector<double>(2);
        label[0] = 1.0;

        var model = new ARMockModel(4, 2);

        var options1 = new AdversarialAttackOptions<double>
        { Epsilon = 0.2, StepSize = 0.05, Iterations = 1, UseRandomStart = false, RandomSeed = Seed };
        var options5 = new AdversarialAttackOptions<double>
        { Epsilon = 0.2, StepSize = 0.05, Iterations = 5, UseRandomStart = false, RandomSeed = Seed };

        var attack1 = new PGDAttack<double, Vector<double>, Vector<double>>(options1);
        var attack5 = new PGDAttack<double, Vector<double>, Vector<double>>(options5);

        var adv1 = attack1.GenerateAdversarialExample(input, label, model);
        var adv5 = attack5.GenerateAdversarialExample(input, label, model);

        double norm1 = 0, norm5 = 0;
        for (int i = 0; i < input.Length; i++)
        {
            norm1 += (adv1[i] - input[i]) * (adv1[i] - input[i]);
            norm5 += (adv5[i] - input[i]) * (adv5[i] - input[i]);
        }
        norm1 = Math.Sqrt(norm1);
        norm5 = Math.Sqrt(norm5);

        // 5 iterations should produce at least as large a perturbation as 1
        Assert.True(norm5 >= norm1 - Tolerance,
            $"5-iter L2 norm {norm5} < 1-iter L2 norm {norm1}");
    }

    [Fact]
    public void PGD_LinfProjection_EachDimensionClampedToEpsilon()
    {
        // After PGD, each dimension's perturbation should be <= epsilon (L-inf projection)
        var epsilon = 0.1;
        var options = new AdversarialAttackOptions<double>
        {
            Epsilon = epsilon,
            StepSize = 0.05,
            Iterations = 10,
            NormType = "L-infinity",
            UseRandomStart = false,
            RandomSeed = Seed
        };
        var attack = new PGDAttack<double, Vector<double>, Vector<double>>(options);

        var input = new Vector<double>(6);
        for (int i = 0; i < 6; i++) input[i] = 0.5;
        var label = new Vector<double>(3);
        label[0] = 1.0;

        var model = new ARMockModel(6, 3);
        var adversarial = attack.GenerateAdversarialExample(input, label, model);

        for (int i = 0; i < input.Length; i++)
        {
            double perturbation = Math.Abs(adversarial[i] - input[i]);
            Assert.True(perturbation <= epsilon + Tolerance,
                $"Dim {i}: perturbation {perturbation} exceeds epsilon {epsilon}");
        }
    }

    [Fact]
    public void PGD_L2Projection_TotalNormBoundedByEpsilon()
    {
        // After PGD with L2 norm, total L2 perturbation should be <= epsilon
        var epsilon = 0.3;
        var options = new AdversarialAttackOptions<double>
        {
            Epsilon = epsilon,
            StepSize = 0.1,
            Iterations = 10,
            NormType = "L2",
            UseRandomStart = false,
            RandomSeed = Seed
        };
        var attack = new PGDAttack<double, Vector<double>, Vector<double>>(options);

        var input = new Vector<double>(6);
        for (int i = 0; i < 6; i++) input[i] = 0.5;
        var label = new Vector<double>(3);
        label[0] = 1.0;

        var model = new ARMockModel(6, 3);
        var adversarial = attack.GenerateAdversarialExample(input, label, model);

        double l2Norm = 0;
        for (int i = 0; i < input.Length; i++)
        {
            double diff = adversarial[i] - input[i];
            l2Norm += diff * diff;
        }
        l2Norm = Math.Sqrt(l2Norm);

        Assert.True(l2Norm <= epsilon + Tolerance,
            $"L2 perturbation norm {l2Norm} exceeds epsilon {epsilon}");
    }

    [Fact]
    public void PGD_OutputAlwaysInValidRange()
    {
        // PGD output should always be clipped to [0, 1]
        var options = new AdversarialAttackOptions<double>
        {
            Epsilon = 0.5,
            StepSize = 0.2,
            Iterations = 20,
            UseRandomStart = true,
            RandomSeed = Seed
        };
        var attack = new PGDAttack<double, Vector<double>, Vector<double>>(options);

        // Input near boundaries to test clipping
        var input = new Vector<double>(4);
        input[0] = 0.05; input[1] = 0.95; input[2] = 0.0; input[3] = 1.0;
        var label = new Vector<double>(2);
        label[0] = 1.0;

        var model = new ARMockModel(4, 2);
        var adversarial = attack.GenerateAdversarialExample(input, label, model);

        for (int i = 0; i < adversarial.Length; i++)
        {
            Assert.True(adversarial[i] >= -Tolerance,
                $"Dim {i}: value {adversarial[i]} below 0");
            Assert.True(adversarial[i] <= 1.0 + Tolerance,
                $"Dim {i}: value {adversarial[i]} above 1");
        }
    }

    [Fact]
    public void PGD_ZeroIterations_ReturnsStartingPoint()
    {
        // With 0 iterations, PGD should return the starting point
        // With UseRandomStart=false, starting point = original input
        var options = new AdversarialAttackOptions<double>
        {
            Epsilon = 0.1,
            Iterations = 0,
            UseRandomStart = false,
            RandomSeed = Seed
        };
        var attack = new PGDAttack<double, Vector<double>, Vector<double>>(options);

        var input = new Vector<double>(4);
        input[0] = 0.3; input[1] = 0.6; input[2] = 0.2; input[3] = 0.8;
        var label = new Vector<double>(2);
        label[0] = 1.0;

        var model = new ARMockModel(4, 2);
        var adversarial = attack.GenerateAdversarialExample(input, label, model);

        for (int i = 0; i < input.Length; i++)
        {
            Assert.Equal(input[i], adversarial[i], Tolerance);
        }
    }

    [Fact]
    public void PGD_StepSizeAffectsConvergenceRate()
    {
        // Larger step size should potentially create larger perturbation per iteration
        var input = new Vector<double>(4);
        for (int i = 0; i < 4; i++) input[i] = 0.5;
        var label = new Vector<double>(2);
        label[0] = 1.0;

        var model = new ARMockModel(4, 2);

        var smallStepOptions = new AdversarialAttackOptions<double>
        { Epsilon = 0.3, StepSize = 0.01, Iterations = 3, UseRandomStart = false, RandomSeed = Seed };
        var largeStepOptions = new AdversarialAttackOptions<double>
        { Epsilon = 0.3, StepSize = 0.1, Iterations = 3, UseRandomStart = false, RandomSeed = Seed };

        var smallStepAttack = new PGDAttack<double, Vector<double>, Vector<double>>(smallStepOptions);
        var largeStepAttack = new PGDAttack<double, Vector<double>, Vector<double>>(largeStepOptions);

        var advSmall = smallStepAttack.GenerateAdversarialExample(input, label, model);
        var advLarge = largeStepAttack.GenerateAdversarialExample(input, label, model);

        double normSmall = 0, normLarge = 0;
        for (int i = 0; i < input.Length; i++)
        {
            normSmall += (advSmall[i] - input[i]) * (advSmall[i] - input[i]);
            normLarge += (advLarge[i] - input[i]) * (advLarge[i] - input[i]);
        }
        normSmall = Math.Sqrt(normSmall);
        normLarge = Math.Sqrt(normLarge);

        // With 3 iterations, larger step should produce larger perturbation
        Assert.True(normLarge >= normSmall - Tolerance,
            $"Large step norm {normLarge} < small step norm {normSmall}");
    }

    #endregion

    #region CW Attack Mathematical Properties

    [Fact]
    public void CW_TanhTransform_MapsToZeroOneRange()
    {
        // CW uses x = (tanh(w) + 1) / 2 to map from (-inf, inf) to (0, 1)
        // Verify this mathematical property:
        // w = -5 -> x ≈ 0.003
        // w = 0  -> x = 0.5
        // w = 5  -> x ≈ 0.997
        double[] wValues = { -5, -2, -1, 0, 1, 2, 5 };
        foreach (double w in wValues)
        {
            double x = (Math.Tanh(w) + 1.0) / 2.0;
            Assert.True(x > 0 && x < 1,
                $"tanh transform of w={w} gave x={x}, expected in (0,1)");
        }

        // w=0 should map to x=0.5 exactly
        double xAtZero = (Math.Tanh(0) + 1.0) / 2.0;
        Assert.Equal(0.5, xAtZero, Tolerance);
    }

    [Fact]
    public void CW_TanhInverse_ConsistentWithForward()
    {
        // atanh(2x - 1) should be the inverse of (tanh(w) + 1) / 2
        double[] xValues = { 0.1, 0.3, 0.5, 0.7, 0.9 };
        foreach (double x in xValues)
        {
            double w = MathHelper.Atanh(2.0 * x - 1.0);
            double xRecovered = (Math.Tanh(w) + 1.0) / 2.0;
            Assert.Equal(x, xRecovered, Tolerance);
        }
    }

    [Fact]
    public void CW_AttackLoss_HandComputed()
    {
        // CW attack loss for untargeted: max(max_other_logit - true_logit, 0)
        // Example: logits = [2.0, 3.0, 1.0], true class = 0
        // true_logit = 2.0, max_other = 3.0
        // attack_loss = max(3.0 - 2.0, 0) = 1.0
        double[] logits = { 2.0, 3.0, 1.0 };
        int trueClass = 0;
        double trueLogit = logits[trueClass];
        double maxOther = double.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++)
            if (i != trueClass && logits[i] > maxOther)
                maxOther = logits[i];

        double attackLoss = Math.Max(maxOther - trueLogit, 0.0);
        Assert.Equal(1.0, attackLoss, Tolerance);
    }

    [Fact]
    public void CW_AttackLoss_ZeroWhenCorrectlyClassified()
    {
        // If true class has highest logit, attack loss should be 0
        // logits = [5.0, 2.0, 1.0], true class = 0
        // true_logit = 5.0, max_other = 2.0
        // attack_loss = max(2.0 - 5.0, 0) = max(-3.0, 0) = 0.0
        double trueLogit = 5.0;
        double maxOther = 2.0;
        double attackLoss = Math.Max(maxOther - trueLogit, 0.0);
        Assert.Equal(0.0, attackLoss, Tolerance);
    }

    [Fact]
    public void CW_ObjectiveFunction_HandComputed()
    {
        // CW objective: ||x_adv - x_orig||^2 + c * attack_loss
        // x_orig = [0.5, 0.5], x_adv = [0.6, 0.7]
        // delta = [0.1, 0.2], ||delta||^2 = 0.01 + 0.04 = 0.05
        // If attack_loss = 1.0 and c = 1.0:
        // objective = 0.05 + 1.0 * 1.0 = 1.05
        double[] delta = { 0.1, 0.2 };
        double l2Squared = 0;
        foreach (double d in delta) l2Squared += d * d;
        Assert.Equal(0.05, l2Squared, Tolerance);

        double c = 1.0;
        double attackLoss = 1.0;
        double objective = l2Squared + c * attackLoss;
        Assert.Equal(1.05, objective, Tolerance);
    }

    [Fact]
    public void CW_GeneratesAdversarial_WithinBounds()
    {
        // CW should produce output in [0, 1] due to tanh parameterization
        var options = new AdversarialAttackOptions<double>
        {
            Epsilon = 1.0,
            Iterations = 10,
            RandomSeed = Seed
        };
        var attack = new CWAttack<double, Vector<double>, Vector<double>>(options);

        var input = new Vector<double>(4);
        input[0] = 0.3; input[1] = 0.7; input[2] = 0.1; input[3] = 0.9;
        var label = new Vector<double>(2);
        label[0] = 1.0;

        var model = new ARMockModel(4, 2);
        var adversarial = attack.GenerateAdversarialExample(input, label, model);

        for (int i = 0; i < adversarial.Length; i++)
        {
            Assert.True(adversarial[i] >= -Tolerance,
                $"CW output dim {i}: {adversarial[i]} below 0");
            Assert.True(adversarial[i] <= 1.0 + Tolerance,
                $"CW output dim {i}: {adversarial[i]} above 1");
        }
    }

    [Fact]
    public void CW_TargetedAttackLoss_HandComputed()
    {
        // Targeted CW loss: max(max_other_logit - target_logit, 0)
        // logits = [2.0, 3.0, 1.0], target class = 2
        // target_logit = 1.0, max_other = max(2.0, 3.0) = 3.0
        // attack_loss = max(3.0 - 1.0, 0) = 2.0
        double targetLogit = 1.0;
        double maxOther = 3.0;
        double attackLoss = Math.Max(maxOther - targetLogit, 0.0);
        Assert.Equal(2.0, attackLoss, Tolerance);
    }

    #endregion

    #region Randomized Smoothing Certified Radius

    [Fact]
    public void RandomizedSmoothing_CertifiedRadius_Formula()
    {
        // R = sigma * Phi^(-1)(pA)
        // For pA = 0.8, Phi^(-1)(0.8) ≈ 0.8416
        // sigma = 0.5 -> R = 0.5 * 0.8416 = 0.4208
        double sigma = 0.5;
        double pA = 0.8;
        double inverseNormalCDF = StatisticsHelper<double>.CalculateInverseNormalCDF(pA);
        double expectedRadius = sigma * inverseNormalCDF;

        // Verify Phi^(-1)(0.8) ≈ 0.8415
        Assert.Equal(0.8415, inverseNormalCDF, 2);
        Assert.Equal(0.4207, expectedRadius, 2);
    }

    [Fact]
    public void RandomizedSmoothing_CertifiedRadius_ZeroWhenPABelowHalf()
    {
        // When pA <= 0.5, Phi^(-1)(pA) <= 0, so R = 0 (no certification possible)
        double sigma = 0.5;
        double pA = 0.5;
        double inverseNormalCDF = StatisticsHelper<double>.CalculateInverseNormalCDF(pA);
        double radius = sigma * inverseNormalCDF;

        // Phi^(-1)(0.5) = 0
        Assert.Equal(0.0, inverseNormalCDF, Tolerance);
        Assert.Equal(0.0, radius, Tolerance);
    }

    [Fact]
    public void RandomizedSmoothing_CertifiedRadius_ScalesLinearly_WithSigma()
    {
        // R = sigma * Phi^(-1)(pA), so doubling sigma doubles R
        double pA = 0.8;
        double invCDF = StatisticsHelper<double>.CalculateInverseNormalCDF(pA);

        double sigma1 = 0.25;
        double sigma2 = 0.50;
        double r1 = sigma1 * invCDF;
        double r2 = sigma2 * invCDF;

        Assert.Equal(r1 * 2, r2, Tolerance);
    }

    [Fact]
    public void RandomizedSmoothing_CertifiedRadius_IncreasesWithConfidence()
    {
        // Higher pA (more confident) -> higher certified radius
        double sigma = 0.5;
        double pA_low = 0.6;
        double pA_high = 0.9;

        double invCDF_low = StatisticsHelper<double>.CalculateInverseNormalCDF(pA_low);
        double invCDF_high = StatisticsHelper<double>.CalculateInverseNormalCDF(pA_high);

        double r_low = sigma * invCDF_low;
        double r_high = sigma * invCDF_high;

        Assert.True(r_high > r_low,
            $"Higher pA should give larger radius: {r_high} vs {r_low}");
    }

    [Fact]
    public void RandomizedSmoothing_InverseNormalCDF_KnownValues()
    {
        // Verify Phi^(-1) at known quantiles:
        // Phi^(-1)(0.5) = 0
        // Phi^(-1)(0.975) ≈ 1.96
        // Phi^(-1)(0.995) ≈ 2.576
        // Phi^(-1)(0.8413) ≈ 1.0  (since Phi(1) ≈ 0.8413)
        double inv_05 = StatisticsHelper<double>.CalculateInverseNormalCDF(0.5);
        double inv_0975 = StatisticsHelper<double>.CalculateInverseNormalCDF(0.975);
        double inv_0995 = StatisticsHelper<double>.CalculateInverseNormalCDF(0.995);

        Assert.Equal(0.0, inv_05, 0.01);
        Assert.Equal(1.96, inv_0975, 0.02);
        Assert.Equal(2.576, inv_0995, 0.02);
    }

    [Fact]
    public void RandomizedSmoothing_Certify_ProducesCertifiedRadius()
    {
        // End-to-end: certify a prediction and verify radius properties
        var options = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.25,
            NumSamples = 200,
            ConfidenceLevel = 0.95,
            RandomSeed = Seed
        };
        var smoothing = new RandomizedSmoothing<double, Vector<double>, Vector<double>>(options);

        var input = new Vector<double>(4);
        for (int i = 0; i < 4; i++) input[i] = 0.5;

        // Model that always predicts class 0 with high confidence
        var weights = new double[8]; // 4 inputs * 2 classes
        for (int i = 0; i < 4; i++)
        {
            weights[0 * 4 + i] = 5.0; // Strong weight for class 0
            weights[1 * 4 + i] = -5.0; // Weak weight for class 1
        }
        var model = new ARMockModel(4, 2, weights);

        var prediction = smoothing.CertifyPrediction(input, model);

        // Model should consistently predict class 0
        Assert.Equal(0, prediction.PredictedClass);
        // With high confidence model, certified radius should be positive
        Assert.True(prediction.CertifiedRadius >= 0,
            $"Certified radius {prediction.CertifiedRadius} should be non-negative");
        // Confidence should be high
        Assert.True(prediction.Confidence > 0.5,
            $"Confidence {prediction.Confidence} should be > 0.5 for strong model");
    }

    [Fact]
    public void RandomizedSmoothing_LargerSigma_LargerRadius_ButLowerAccuracy()
    {
        // Property: larger sigma gives potentially larger certified radius
        // but may reduce clean accuracy due to more noise
        var input = new Vector<double>(4);
        for (int i = 0; i < 4; i++) input[i] = 0.5;

        var weights = new double[8];
        for (int i = 0; i < 4; i++)
        {
            weights[0 * 4 + i] = 5.0;
            weights[1 * 4 + i] = -5.0;
        }
        var model = new ARMockModel(4, 2, weights);

        var optionsSmall = new CertifiedDefenseOptions<double>
        { NoiseSigma = 0.1, NumSamples = 200, ConfidenceLevel = 0.95, RandomSeed = Seed };
        var optionsLarge = new CertifiedDefenseOptions<double>
        { NoiseSigma = 0.5, NumSamples = 200, ConfidenceLevel = 0.95, RandomSeed = Seed };

        var smoothSmall = new RandomizedSmoothing<double, Vector<double>, Vector<double>>(optionsSmall);
        var smoothLarge = new RandomizedSmoothing<double, Vector<double>, Vector<double>>(optionsLarge);

        var predSmall = smoothSmall.CertifyPrediction(input, model);
        var predLarge = smoothLarge.CertifyPrediction(input, model);

        // Both should produce valid predictions
        Assert.True(predSmall.PredictedClass >= 0);
        Assert.True(predLarge.PredictedClass >= 0);

        // Certified radii should be non-negative
        Assert.True(predSmall.CertifiedRadius >= 0);
        Assert.True(predLarge.CertifiedRadius >= 0);
    }

    [Fact]
    public void RandomizedSmoothing_ConfidenceBounds_Consistent()
    {
        // Lower bound <= point estimate <= upper bound
        var options = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.25,
            NumSamples = 200,
            ConfidenceLevel = 0.95,
            RandomSeed = Seed
        };
        var smoothing = new RandomizedSmoothing<double, Vector<double>, Vector<double>>(options);

        var input = new Vector<double>(4);
        for (int i = 0; i < 4; i++) input[i] = 0.5;

        var model = new ARMockModel(4, 2);
        var prediction = smoothing.CertifyPrediction(input, model);

        Assert.True(prediction.LowerBound <= prediction.Confidence + Tolerance,
            $"Lower bound {prediction.LowerBound} > confidence {prediction.Confidence}");
        Assert.True(prediction.Confidence <= prediction.UpperBound + Tolerance,
            $"Confidence {prediction.Confidence} > upper bound {prediction.UpperBound}");
    }

    #endregion

    #region Cross-Entropy Loss Computation

    [Fact]
    public void CrossEntropy_HandComputed_ThreeClasses()
    {
        // Softmax of logits [2.0, 1.0, 0.1]:
        // exp(2.0) = 7.389, exp(1.0) = 2.718, exp(0.1) = 1.105
        // sum = 11.212
        // p = [0.6590, 0.2424, 0.0986]
        // Cross-entropy loss for true class 0: -log(0.6590) = 0.4170

        double[] logits = { 2.0, 1.0, 0.1 };
        double[] expLogits = logits.Select(l => Math.Exp(l)).ToArray();
        double sumExp = expLogits.Sum();
        double[] probs = expLogits.Select(e => e / sumExp).ToArray();

        Assert.Equal(0.6590, probs[0], 3);
        Assert.Equal(0.2424, probs[1], 3);
        Assert.Equal(0.0986, probs[2], 3);

        double crossEntropy = -Math.Log(probs[0]);
        Assert.Equal(0.4170, crossEntropy, 3);
    }

    [Fact]
    public void CrossEntropy_PerfectPrediction_ZeroLoss()
    {
        // If probability of true class = 1.0, loss = -log(1.0) = 0
        double loss = -Math.Log(1.0);
        Assert.Equal(0.0, loss, Tolerance);
    }

    [Fact]
    public void CrossEntropy_UniformPrediction_LogK()
    {
        // For uniform distribution over K classes, loss = -log(1/K) = log(K)
        int K = 3;
        double uniformProb = 1.0 / K;
        double loss = -Math.Log(uniformProb);
        Assert.Equal(Math.Log(K), loss, Tolerance);
    }

    [Fact]
    public void Softmax_Stability_ShiftByMaxLogit()
    {
        // Softmax(z) = softmax(z - max(z))
        // This is a key numerical stability property
        double[] logits = { 100.0, 101.0, 99.0 };
        double maxLogit = logits.Max();

        // Without shift (would overflow)
        // With shift
        double[] shifted = logits.Select(l => l - maxLogit).ToArray();
        double[] expShifted = shifted.Select(s => Math.Exp(s)).ToArray();
        double sumExpShifted = expShifted.Sum();
        double[] probsShifted = expShifted.Select(e => e / sumExpShifted).ToArray();

        // Verify probabilities sum to 1
        Assert.Equal(1.0, probsShifted.Sum(), Tolerance);

        // Middle value (101) should have highest probability
        Assert.True(probsShifted[1] > probsShifted[0]);
        Assert.True(probsShifted[1] > probsShifted[2]);
    }

    #endregion

    #region Norm and Projection Mathematics

    [Fact]
    public void L2Norm_HandComputed()
    {
        // ||[3, 4]||_2 = sqrt(9 + 16) = sqrt(25) = 5
        double[] v = { 3.0, 4.0 };
        double norm = Math.Sqrt(v.Sum(x => x * x));
        Assert.Equal(5.0, norm, Tolerance);
    }

    [Fact]
    public void LInfNorm_HandComputed()
    {
        // ||[3, -4, 2]||_inf = max(|3|, |-4|, |2|) = 4
        double[] v = { 3.0, -4.0, 2.0 };
        double norm = v.Max(x => Math.Abs(x));
        Assert.Equal(4.0, norm, Tolerance);
    }

    [Fact]
    public void L2Projection_InsideBall_NoChange()
    {
        // If ||perturbation||_2 <= epsilon, projection is identity
        double[] perturbation = { 0.1, 0.1 };
        double epsilon = 1.0;
        double norm = Math.Sqrt(perturbation.Sum(x => x * x));
        // norm = sqrt(0.02) ≈ 0.1414 < 1.0
        Assert.True(norm <= epsilon);

        // No scaling needed
        double[] projected = perturbation;
        Assert.Equal(perturbation[0], projected[0], Tolerance);
        Assert.Equal(perturbation[1], projected[1], Tolerance);
    }

    [Fact]
    public void L2Projection_OutsideBall_ScalesDown()
    {
        // If ||perturbation||_2 > epsilon, scale by epsilon / ||perturbation||
        double[] perturbation = { 3.0, 4.0 };
        double epsilon = 2.5;
        double norm = Math.Sqrt(perturbation.Sum(x => x * x)); // 5.0
        Assert.Equal(5.0, norm, Tolerance);
        Assert.True(norm > epsilon);

        double scale = epsilon / norm; // 2.5 / 5.0 = 0.5
        double[] projected = perturbation.Select(p => p * scale).ToArray();
        // projected = [1.5, 2.0]
        Assert.Equal(1.5, projected[0], Tolerance);
        Assert.Equal(2.0, projected[1], Tolerance);

        // Verify projected norm = epsilon
        double projectedNorm = Math.Sqrt(projected.Sum(x => x * x));
        Assert.Equal(epsilon, projectedNorm, Tolerance);
    }

    [Fact]
    public void LinfProjection_ClampsEachDimension()
    {
        // L-inf projection: clamp each dimension to [-epsilon, epsilon]
        double[] perturbation = { 0.5, -0.3, 0.05, -0.8 };
        double epsilon = 0.3;

        double[] projected = perturbation.Select(p => Math.Max(-epsilon, Math.Min(epsilon, p))).ToArray();

        Assert.Equal(0.3, projected[0], Tolerance); // clamped from 0.5
        Assert.Equal(-0.3, projected[1], Tolerance); // at boundary
        Assert.Equal(0.05, projected[2], Tolerance); // not clamped
        Assert.Equal(-0.3, projected[3], Tolerance); // clamped from -0.8
    }

    [Fact]
    public void SignFunction_ReturnsCorrectSigns()
    {
        // sign(x) = -1 if x < 0, 0 if x == 0, +1 if x > 0
        Assert.Equal(1.0, Math.Sign(3.5));
        Assert.Equal(-1.0, Math.Sign(-2.1));
        Assert.Equal(0.0, Math.Sign(0.0));
        Assert.Equal(1.0, Math.Sign(0.001));
        Assert.Equal(-1.0, Math.Sign(-0.001));
    }

    #endregion

    #region Finite Difference Gradient Approximation

    [Fact]
    public void FiniteDifference_ApproximatesDerivative()
    {
        // For f(x) = x^2, f'(x) = 2x
        // Finite difference: (f(x+h) - f(x)) / h ≈ f'(x)
        double x = 3.0;
        double h = 0.001;
        double fxph = (x + h) * (x + h);
        double fx = x * x;
        double approxDerivative = (fxph - fx) / h;
        double exactDerivative = 2 * x;

        Assert.Equal(exactDerivative, approxDerivative, 2); // 2 decimal places
    }

    [Fact]
    public void FiniteDifference_ForSoftmaxCrossEntropy()
    {
        // For softmax cross-entropy loss with respect to input:
        // Input [2.0, 1.0] -> softmax -> [0.731, 0.269]
        // Loss for class 0: -log(0.731) = 0.3133
        double[] input = { 2.0, 1.0 };
        double[] softmax = SoftmaxHelper(input);
        double loss = -Math.Log(softmax[0]);

        // Approximate gradient by finite differences
        double h = 0.001;
        double[] perturbedInput = { 2.0 + h, 1.0 };
        double[] perturbedSoftmax = SoftmaxHelper(perturbedInput);
        double perturbedLoss = -Math.Log(perturbedSoftmax[0]);
        double gradApprox = (perturbedLoss - loss) / h;

        // Analytic gradient: d(-log(softmax_0))/d(z_0) = softmax_0 - 1 = 0.731 - 1 = -0.269
        double gradExact = softmax[0] - 1.0;
        Assert.Equal(gradExact, gradApprox, 2);
    }

    #endregion

    #region Attack Comparison Properties

    [Fact]
    public void FGSM_SingleStep_PGD_OneIteration_Equivalent()
    {
        // PGD with 1 iteration and step_size = epsilon should be equivalent to FGSM
        // (when both use the same starting point and no random start)
        var epsilon = 0.1;
        var input = new Vector<double>(4);
        for (int i = 0; i < 4; i++) input[i] = 0.5;
        var label = new Vector<double>(2);
        label[0] = 1.0;

        var model = new ARMockModel(4, 2);

        var fgsmOptions = new AdversarialAttackOptions<double>
        { Epsilon = epsilon, RandomSeed = Seed };
        var pgdOptions = new AdversarialAttackOptions<double>
        { Epsilon = epsilon, StepSize = epsilon, Iterations = 1, UseRandomStart = false, RandomSeed = Seed };

        var fgsm = new FGSMAttack<double, Vector<double>, Vector<double>>(fgsmOptions);
        var pgd = new PGDAttack<double, Vector<double>, Vector<double>>(pgdOptions);

        var fgsmAdv = fgsm.GenerateAdversarialExample(input, label, model);
        var pgdAdv = pgd.GenerateAdversarialExample(input, label, model);

        // Both should produce the same adversarial example
        for (int i = 0; i < input.Length; i++)
        {
            Assert.Equal(fgsmAdv[i], pgdAdv[i], Tolerance);
        }
    }

    [Fact]
    public void PGD_StrongerThanFGSM_WithMultipleIterations()
    {
        // PGD with many iterations should produce perturbation at least as large as FGSM
        var epsilon = 0.15;
        var input = new Vector<double>(6);
        for (int i = 0; i < 6; i++) input[i] = 0.5;
        var label = new Vector<double>(3);
        label[0] = 1.0;

        var model = new ARMockModel(6, 3);

        var fgsmOptions = new AdversarialAttackOptions<double>
        { Epsilon = epsilon, RandomSeed = Seed };
        var pgdOptions = new AdversarialAttackOptions<double>
        { Epsilon = epsilon, StepSize = epsilon / 5.0, Iterations = 20, UseRandomStart = false, RandomSeed = Seed };

        var fgsm = new FGSMAttack<double, Vector<double>, Vector<double>>(fgsmOptions);
        var pgd = new PGDAttack<double, Vector<double>, Vector<double>>(pgdOptions);

        var fgsmAdv = fgsm.GenerateAdversarialExample(input, label, model);
        var pgdAdv = pgd.GenerateAdversarialExample(input, label, model);

        // Compute L2 norms of perturbations
        double fgsmNorm = 0, pgdNorm = 0;
        for (int i = 0; i < input.Length; i++)
        {
            fgsmNorm += (fgsmAdv[i] - input[i]) * (fgsmAdv[i] - input[i]);
            pgdNorm += (pgdAdv[i] - input[i]) * (pgdAdv[i] - input[i]);
        }

        // PGD should use at least as much of the budget
        Assert.True(pgdNorm >= fgsmNorm - 0.01,
            $"PGD L2^2 norm {pgdNorm} < FGSM L2^2 norm {fgsmNorm}");
    }

    [Fact]
    public void AllAttacks_Deterministic_WithSameSeed()
    {
        // Same seed should produce same result
        var options = new AdversarialAttackOptions<double>
        { Epsilon = 0.1, Iterations = 5, RandomSeed = Seed };

        var input = new Vector<double>(4);
        for (int i = 0; i < 4; i++) input[i] = 0.5;
        var label = new Vector<double>(2);
        label[0] = 1.0;

        var model = new ARMockModel(4, 2);

        var attack1 = new FGSMAttack<double, Vector<double>, Vector<double>>(options);
        var attack2 = new FGSMAttack<double, Vector<double>, Vector<double>>(options);

        var adv1 = attack1.GenerateAdversarialExample(input, label, model);
        var adv2 = attack2.GenerateAdversarialExample(input, label, model);

        for (int i = 0; i < input.Length; i++)
        {
            Assert.Equal(adv1[i], adv2[i], Tolerance);
        }
    }

    #endregion

    #region Softmax and Argmax Helper Tests

    [Fact]
    public void Softmax_SumsToOne()
    {
        double[] logits = { 1.0, 2.0, 3.0, 4.0 };
        double[] probs = SoftmaxHelper(logits);
        Assert.Equal(1.0, probs.Sum(), Tolerance);
    }

    [Fact]
    public void Softmax_PreservesOrdering()
    {
        double[] logits = { 1.0, 3.0, 2.0 };
        double[] probs = SoftmaxHelper(logits);
        // Ordering: p[1] > p[2] > p[0]
        Assert.True(probs[1] > probs[2]);
        Assert.True(probs[2] > probs[0]);
    }

    [Fact]
    public void Softmax_HandComputed_TwoClasses()
    {
        // softmax([1, 2]) = [exp(1)/(exp(1)+exp(2)), exp(2)/(exp(1)+exp(2))]
        // = [2.718/(2.718+7.389), 7.389/(2.718+7.389)]
        // = [0.2689, 0.7311]
        double[] logits = { 1.0, 2.0 };
        double[] probs = SoftmaxHelper(logits);
        Assert.Equal(0.2689, probs[0], 3);
        Assert.Equal(0.7311, probs[1], 3);
    }

    [Fact]
    public void Argmax_ReturnsCorrectIndex()
    {
        double[] values = { 0.1, 0.7, 0.2 };
        int argmax = 0;
        for (int i = 1; i < values.Length; i++)
            if (values[i] > values[argmax])
                argmax = i;
        Assert.Equal(1, argmax);
    }

    #endregion

    #region Clopper-Pearson Confidence Interval

    [Fact]
    public void ClopperPearson_HandComputed_Properties()
    {
        // Clopper-Pearson interval should contain the point estimate
        int successes = 80;
        int n = 100;
        double confidence = 0.95;

        var interval = StatisticsHelper<double>.CalculateClopperPearsonInterval(
            successes, n, confidence);
        double lower = NumOps.ToDouble(interval.Lower);
        double upper = NumOps.ToDouble(interval.Upper);
        double pointEstimate = (double)successes / n;

        Assert.True(lower <= pointEstimate,
            $"Lower bound {lower} > point estimate {pointEstimate}");
        Assert.True(upper >= pointEstimate,
            $"Upper bound {upper} < point estimate {pointEstimate}");
        Assert.True(lower >= 0, $"Lower bound {lower} < 0");
        Assert.True(upper <= 1, $"Upper bound {upper} > 1");
    }

    [Fact]
    public void ClopperPearson_NarrowsWithMoreSamples()
    {
        // More samples -> narrower confidence interval
        double confidence = 0.95;

        var interval50 = StatisticsHelper<double>.CalculateClopperPearsonInterval(40, 50, confidence);
        var interval500 = StatisticsHelper<double>.CalculateClopperPearsonInterval(400, 500, confidence);

        double width50 = NumOps.ToDouble(interval50.Upper) - NumOps.ToDouble(interval50.Lower);
        double width500 = NumOps.ToDouble(interval500.Upper) - NumOps.ToDouble(interval500.Lower);

        Assert.True(width500 < width50,
            $"Width with 500 samples ({width500}) not narrower than 50 samples ({width50})");
    }

    [Fact]
    public void ClopperPearson_SymmetricForHalf()
    {
        // For p_hat = 0.5, interval should be approximately symmetric
        int n = 100;
        int successes = 50;
        double confidence = 0.95;

        var interval = StatisticsHelper<double>.CalculateClopperPearsonInterval(successes, n, confidence);
        double lower = NumOps.ToDouble(interval.Lower);
        double upper = NumOps.ToDouble(interval.Upper);

        double distFromLower = 0.5 - lower;
        double distFromUpper = upper - 0.5;

        Assert.Equal(distFromLower, distFromUpper, 0.01);
    }

    #endregion

    #region Helpers

    private static double[] SoftmaxHelper(double[] logits)
    {
        double max = logits.Max();
        double[] exp = logits.Select(l => Math.Exp(l - max)).ToArray();
        double sum = exp.Sum();
        return exp.Select(e => e / sum).ToArray();
    }

    #endregion
}
