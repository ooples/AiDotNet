using AiDotNet.Models.Options;

namespace AiDotNet.AdversarialRobustness.Attacks;

/// <summary>
/// Implements the Carlini &amp; Wagner (C&amp;W) attack.
/// </summary>
/// <remarks>
/// <para>
/// C&amp;W is an optimization-based attack that formulates adversarial example generation as
/// an optimization problem, typically producing stronger attacks than gradient-based methods.
/// </para>
/// <para><b>For Beginners:</b> C&amp;W is one of the most sophisticated attacks. Instead of
/// following gradients, it treats creating adversarial examples as a carefully crafted
/// optimization problem. It's slower than FGSM or PGD but often finds adversarial examples
/// that are more subtle and harder to defend against.</para>
/// <para>
/// Original paper: "Towards Evaluating the Robustness of Neural Networks"
/// by Carlini &amp; Wagner (2017)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class CWAttack<T> : AdversarialAttackBase<T>
{
    /// <summary>
    /// Initializes a new instance of the C&amp;W attack.
    /// </summary>
    /// <param name="options">The configuration options for the attack.</param>
    public CWAttack(AdversarialAttackOptions<T> options) : base(options)
    {
    }

    /// <summary>
    /// Generates an adversarial example using the C&amp;W L2 attack.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The C&amp;W attack solves:
    /// minimize ||δ||_2 + c * f(x + δ)
    /// where f measures how well the attack succeeds.
    /// </para>
    /// <para><b>For Beginners:</b> This method tries to find the smallest possible change
    /// that will fool the model. It balances making the change small (hard to detect) with
    /// making the attack successful (fooling the model).</para>
    /// </remarks>
    /// <param name="input">The clean input to perturb.</param>
    /// <param name="trueLabel">The correct label for the input.</param>
    /// <param name="targetModel">The model to attack.</param>
    /// <returns>The adversarial example.</returns>
    public override T[] GenerateAdversarialExample(T[] input, int trueLabel, Func<T[], T[]> targetModel)
    {
        var c = 1.0; // Confidence parameter
        var learningRate = 0.01;
        var bestAdversarial = (T[])input.Clone();
        var bestPerturbation = double.PositiveInfinity;

        // Initialize perturbation variable (in tanh space for box constraints)
        var w = new double[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            // Initialize w such that tanh(w) ≈ x
            var x = NumOps.ToDouble(input[i]);
            var twoXMinusOne = 2.0 * x - 1.0;
            w[i] = Atanh(Clamp(twoXMinusOne, -0.9999, 0.9999));
        }

        // Optimization loop
        for (int iteration = 0; iteration < Options.Iterations; iteration++)
        {
            // Convert from tanh space to valid input range [0, 1]
            var adversarial = new T[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                var tanhW = Math.Tanh(w[i]);
                adversarial[i] = NumOps.FromDouble((tanhW + 1.0) / 2.0);
            }

            // Compute objective and gradient
            var output = targetModel(adversarial);
            var (objective, gradient) = ComputeObjectiveAndGradient(w, input, output, trueLabel, c, targetModel);

            // Update w using gradient descent
            for (int i = 0; i < w.Length; i++)
            {
                w[i] -= learningRate * gradient[i];
            }

            // Track best solution
            var perturbationNorm = NumOps.ToDouble(ComputeL2Norm(CalculatePerturbation(input, adversarial)));
            if (IsSuccessfulAttack(output, trueLabel) && perturbationNorm < bestPerturbation)
            {
                bestAdversarial = (T[])adversarial.Clone();
                bestPerturbation = perturbationNorm;
            }
        }

        return bestAdversarial;
    }

    /// <summary>
    /// Computes the objective function and its gradient.
    /// </summary>
    private (double objective, double[] gradient) ComputeObjectiveAndGradient(
        double[] w,
        T[] original,
        T[] output,
        int trueLabel,
        double c,
        Func<T[], T[]> targetModel)
    {
        var objective = ComputeObjective(w, original, output, trueLabel, c);

        // Approximate gradient in w-space using finite differences.
        var gradient = new double[w.Length];
        const double delta = 0.001;

        for (int i = 0; i < w.Length; i++)
        {
            var wPerturbed = (double[])w.Clone();
            wPerturbed[i] += delta;

            var advPerturbed = new T[wPerturbed.Length];
            for (int j = 0; j < wPerturbed.Length; j++)
            {
                var tanhW = Math.Tanh(wPerturbed[j]);
                advPerturbed[j] = NumOps.FromDouble((tanhW + 1.0) / 2.0);
            }

            var outputPerturbed = targetModel(advPerturbed);
            var perturbedObjective = ComputeObjective(wPerturbed, original, outputPerturbed, trueLabel, c);
            gradient[i] = (perturbedObjective - objective) / delta;
        }

        return (objective, gradient);
    }

    /// <summary>
    /// Computes the objective value for a given w.
    /// </summary>
    private double ComputeObjective(double[] w, T[] original, T[] output, int trueLabel, double c)
    {
        var adversarial = new T[w.Length];
        for (int i = 0; i < w.Length; i++)
        {
            var tanhW = Math.Tanh(w[i]);
            adversarial[i] = NumOps.FromDouble((tanhW + 1.0) / 2.0);
        }

        var perturbation = CalculatePerturbation(original, adversarial);
        var l2Distance = NumOps.ToDouble(ComputeL2Norm(perturbation));
        var l2DistanceSquared = l2Distance * l2Distance;

        var attackLoss = ComputeAttackLoss(output, trueLabel);
        return l2DistanceSquared + c * attackLoss;
    }

    /// <summary>
    /// Computes the attack loss for C&amp;W.
    /// </summary>
    private double ComputeAttackLoss(T[] output, int trueLabel)
    {
        var trueLogit = NumOps.ToDouble(output[trueLabel]);

        // Find the maximum logit that isn't the true class
        var maxOtherLogit = double.NegativeInfinity;
        for (int i = 0; i < output.Length; i++)
        {
            if (i == trueLabel)
                continue;

            maxOtherLogit = Math.Max(maxOtherLogit, NumOps.ToDouble(output[i]));
        }

        // Untargeted: maximize maxOther - true. Targeted: maximize true - target (i.e., push toward target).
        if (Options.IsTargeted)
        {
            var targetIndex = ClampInt(Options.TargetClass, 0, output.Length - 1);
            var targetLogit = NumOps.ToDouble(output[targetIndex]);
            return Math.Max(maxOtherLogit - targetLogit, 0.0);
        }

        return Math.Max(maxOtherLogit - trueLogit, 0.0);
    }

    private static double Clamp(double value, double min, double max)
    {
        if (value < min) return min;
        if (value > max) return max;
        return value;
    }

    private static int ClampInt(int value, int min, int max)
    {
        if (value < min) return min;
        if (value > max) return max;
        return value;
    }

    private static double Atanh(double x)
    {
        // net471: Math.Atanh not available.
        // atanh(x) = 0.5 * ln((1+x)/(1-x))
        return 0.5 * Math.Log((1.0 + x) / (1.0 - x));
    }

    /// <summary>
    /// Checks if the attack was successful.
    /// </summary>
    private bool IsSuccessfulAttack(T[] output, int trueLabel)
    {
        var predictedClass = 0;
        var maxValue = NumOps.ToDouble(output[0]);

        for (int i = 1; i < output.Length; i++)
        {
            var v = NumOps.ToDouble(output[i]);
            if (v > maxValue)
            {
                maxValue = v;
                predictedClass = i;
            }
        }

        return Options.IsTargeted
            ? predictedClass == Options.TargetClass
            : predictedClass != trueLabel;
    }
}
