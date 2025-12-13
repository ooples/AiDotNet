using System.Numerics;
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
    where T : struct, INumber<T>
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
        var c = T.CreateChecked(1.0); // Confidence parameter
        var learningRate = T.CreateChecked(0.01);
        var bestAdversarial = (T[])input.Clone();
        var bestPerturbation = T.CreateChecked(double.MaxValue);

        // Initialize perturbation variable (in tanh space for box constraints)
        var w = new T[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            // Initialize w such that tanh(w) ≈ x
            var twoXMinusOne = T.CreateChecked(2.0) * input[i] - T.One;
            w[i] = T.CreateChecked(Math.Atanh(Math.Clamp(double.CreateChecked(twoXMinusOne), -0.9999, 0.9999)));
        }

        // Optimization loop
        for (int iteration = 0; iteration < Options.Iterations; iteration++)
        {
            // Convert from tanh space to valid input range [0, 1]
            var adversarial = new T[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                var tanhW = T.CreateChecked(Math.Tanh(double.CreateChecked(w[i])));
                adversarial[i] = (tanhW + T.One) / T.CreateChecked(2.0);
            }

            // Compute objective and gradient
            var output = targetModel(adversarial);
            var (objective, gradient) = ComputeObjectiveAndGradient(adversarial, input, output, trueLabel, c);

            // Update w using gradient descent
            for (int i = 0; i < w.Length; i++)
            {
                w[i] -= learningRate * gradient[i];
            }

            // Track best solution
            var perturbationNorm = ComputeL2Norm(CalculatePerturbation(input, adversarial));
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
    private (T objective, T[] gradient) ComputeObjectiveAndGradient(T[] adversarial, T[] original, T[] output, int trueLabel, T c)
    {
        // Compute L2 distance term
        var perturbation = CalculatePerturbation(original, adversarial);
        var l2Distance = ComputeL2Norm(perturbation);
        var l2DistanceSquared = l2Distance * l2Distance;

        // Compute attack loss (measures attack success)
        var attackLoss = ComputeAttackLoss(output, trueLabel);

        // Total objective
        var objective = l2DistanceSquared + c * attackLoss;

        // Approximate gradient using finite differences
        var gradient = new T[adversarial.Length];
        var delta = T.CreateChecked(0.001);

        for (int i = 0; i < adversarial.Length; i++)
        {
            var perturbedAdv = (T[])adversarial.Clone();
            perturbedAdv[i] += delta;

            var perturbedOutput = targetModel(perturbedAdv);
            var perturbedPert = CalculatePerturbation(original, perturbedAdv);
            var perturbedL2 = ComputeL2Norm(perturbedPert);
            var perturbedL2Sq = perturbedL2 * perturbedL2;
            var perturbedLoss = ComputeAttackLoss(perturbedOutput, trueLabel);
            var perturbedObjective = perturbedL2Sq + c * perturbedLoss;

            gradient[i] = (perturbedObjective - objective) / delta;
        }

        return (objective, gradient);
    }

    /// <summary>
    /// Computes the attack loss for C&amp;W.
    /// </summary>
    private T ComputeAttackLoss(T[] output, int trueLabel)
    {
        // Find the maximum logit that isn't the true class
        T maxOtherLogit = T.CreateChecked(double.MinValue);
        for (int i = 0; i < output.Length; i++)
        {
            if (i != trueLabel && output[i] > maxOtherLogit)
            {
                maxOtherLogit = output[i];
            }
        }

        // f(x) = max(max(Z(x)_i for i != t) - Z(x)_t, -κ)
        // where κ is confidence parameter (we use 0 for simplicity)
        var loss = T.Max(maxOtherLogit - output[trueLabel], T.Zero);
        return loss;
    }

    /// <summary>
    /// Checks if the attack was successful.
    /// </summary>
    private bool IsSuccessfulAttack(T[] output, int trueLabel)
    {
        var predictedClass = 0;
        var maxValue = output[0];

        for (int i = 1; i < output.Length; i++)
        {
            if (output[i] > maxValue)
            {
                maxValue = output[i];
                predictedClass = i;
            }
        }

        if (Options.IsTargeted)
        {
            return predictedClass == Options.TargetClass;
        }
        else
        {
            return predictedClass != trueLabel;
        }
    }

    private Func<T[], T[]> targetModel = null!;
}
