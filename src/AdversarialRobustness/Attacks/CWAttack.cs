using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

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
    public override Vector<T> GenerateAdversarialExample(Vector<T> input, int trueLabel, IPredictiveModel<T, Vector<T>, Vector<T>> targetModel)
    {
        if (input == null)
        {
            throw new ArgumentNullException(nameof(input));
        }

        if (targetModel == null)
        {
            throw new ArgumentNullException(nameof(targetModel));
        }

        var c = 1.0; // Confidence parameter
        var learningRate = 0.01;
        var bestAdversarial = CloneInput(input);
        var bestPerturbation = double.PositiveInfinity;

        // Initialize perturbation variable (in tanh space for box constraints)
        var w = new double[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            // Initialize w such that tanh(w) ≈ x
            var x = NumOps.ToDouble(input[i]);
            var twoXMinusOne = 2.0 * x - 1.0;
            w[i] = MathHelper.Atanh(MathHelper.Clamp(twoXMinusOne, -0.9999, 0.9999));
        }

        // Optimization loop
        for (int iteration = 0; iteration < Options.Iterations; iteration++)
        {
            // Convert from tanh space to valid input range [0, 1]
            var adversarial = new Vector<T>(input.Length);
            for (int i = 0; i < input.Length; i++)
            {
                var tanhW = Math.Tanh(w[i]);
                adversarial[i] = NumOps.FromDouble((tanhW + 1.0) / 2.0);
            }

            // Compute objective and gradient
            var output = targetModel.Predict(adversarial);
            var (_, gradient) = ComputeObjectiveAndGradient(w, input, output, trueLabel, c, targetModel);

            // Update w using gradient descent
            for (int i = 0; i < w.Length; i++)
            {
                w[i] -= learningRate * gradient[i];
            }

            // Track best solution
            var perturbationNorm = NumOps.ToDouble(ComputeL2Norm(CalculatePerturbation(input, adversarial)));
            if (IsSuccessfulAttack(output, trueLabel) && perturbationNorm < bestPerturbation)
            {
                bestAdversarial = CloneInput(adversarial);
                bestPerturbation = perturbationNorm;
            }
        }

        return bestAdversarial;
    }

    private static Vector<T> CloneInput(Vector<T> input)
    {
        var clone = new Vector<T>(input.Length);
        for (int i = 0; i < input.Length; i++)
        {
            clone[i] = input[i];
        }
        return clone;
    }

    /// <summary>
    /// Computes the objective function and its gradient.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When the target model implements <see cref="IInputGradientComputable{T}"/>, this method uses
    /// analytic gradient computation via backpropagation, which is more accurate and efficient
    /// than finite-difference approximation.
    /// </para>
    /// <para>
    /// Falls back to finite-difference approximation for models that don't support analytic gradients.
    /// </para>
    /// </remarks>
    private (double objective, double[] gradient) ComputeObjectiveAndGradient(
        double[] w,
        Vector<T> original,
        Vector<T> output,
        int trueLabel,
        double c,
        IPredictiveModel<T, Vector<T>, Vector<T>> targetModel)
    {
        var objective = ComputeObjective(w, original, output, trueLabel, c);

        // Check if the model supports analytic gradients
        if (targetModel is IInputGradientComputable<T> gradientComputable)
        {
            return (objective, ComputeAnalyticGradient(w, original, output, trueLabel, c, gradientComputable));
        }

        // Fallback: approximate gradient in w-space using finite differences
        return (objective, ComputeFiniteDifferenceGradient(w, original, trueLabel, c, targetModel));
    }

    /// <summary>
    /// Computes the gradient analytically using the model's backpropagation capabilities.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The C&amp;W objective is: L = ||δ||² + c * f(x_adv)
    /// where δ = x_adv - x_orig and f is the attack loss.
    /// </para>
    /// <para>
    /// The gradient in w-space uses the chain rule:
    /// dL/dw = dL/dx_adv * dx_adv/dw
    /// where dx_adv/dw = (1 - tanh²(w))/2 due to the tanh parameterization.
    /// </para>
    /// </remarks>
    private double[] ComputeAnalyticGradient(
        double[] w,
        Vector<T> original,
        Vector<T> output,
        int trueLabel,
        double c,
        IInputGradientComputable<T> gradientComputable)
    {
        var n = w.Length;
        var gradient = new double[n];

        // Compute adversarial example and intermediate values
        var adversarial = new Vector<T>(n);
        var tanhW = new double[n];
        for (int i = 0; i < n; i++)
        {
            tanhW[i] = Math.Tanh(w[i]);
            adversarial[i] = NumOps.FromDouble((tanhW[i] + 1.0) / 2.0);
        }

        // Compute gradient of L2 perturbation term: d(||x_adv - x_orig||²)/dx_adv = 2 * (x_adv - x_orig)
        var perturbation = CalculatePerturbation(original, adversarial);
        var perturbGrad = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            perturbGrad[i] = NumOps.Multiply(NumOps.FromDouble(2.0), perturbation[i]);
        }

        // Compute gradient of attack loss term: df/dx_adv
        var outputGradient = ComputeAttackLossGradient(output, trueLabel);
        var inputGradient = gradientComputable.ComputeInputGradient(adversarial, outputGradient);

        // Combine gradients: dL/dx_adv = perturbGrad + c * inputGradient
        for (int i = 0; i < n; i++)
        {
            var dLdx = NumOps.ToDouble(perturbGrad[i]) + c * NumOps.ToDouble(inputGradient[i]);

            // Chain rule: dx_adv/dw = d[(tanh(w)+1)/2]/dw = (1 - tanh²(w))/2
            var dxdw = (1.0 - tanhW[i] * tanhW[i]) / 2.0;
            gradient[i] = dLdx * dxdw;
        }

        return gradient;
    }

    /// <summary>
    /// Computes the gradient of the attack loss with respect to the model output.
    /// </summary>
    /// <remarks>
    /// For untargeted attacks: maximize max_other - true_logit, so gradient is -1 at true class, +1 at max other.
    /// For targeted attacks: maximize target_logit - max_other, so gradient is +1 at target, -1 at max other.
    /// </remarks>
    private Vector<T> ComputeAttackLossGradient(Vector<T> output, int trueLabel)
    {
        var gradient = new Vector<T>(output.Length);
        var trueLogit = NumOps.ToDouble(output[trueLabel]);

        // Find the maximum logit that isn't the true class
        var maxOtherLogit = double.NegativeInfinity;
        var maxOtherIndex = -1;
        for (int i = 0; i < output.Length; i++)
        {
            if (i == trueLabel)
                continue;

            var logit = NumOps.ToDouble(output[i]);
            if (logit > maxOtherLogit)
            {
                maxOtherLogit = logit;
                maxOtherIndex = i;
            }
        }

        if (Options.IsTargeted)
        {
            var targetIndex = MathHelper.Clamp(Options.TargetClass, 0, output.Length - 1);
            var targetLogit = NumOps.ToDouble(output[targetIndex]);

            // Loss = max(0, max_other - target), derivative is non-zero only when loss > 0
            if (maxOtherLogit > targetLogit)
            {
                gradient[maxOtherIndex] = NumOps.FromDouble(1.0);
                gradient[targetIndex] = NumOps.FromDouble(-1.0);
            }
        }
        else
        {
            // Loss = max(0, max_other - true), derivative is non-zero only when loss > 0
            if (maxOtherLogit > trueLogit && maxOtherIndex >= 0)
            {
                gradient[maxOtherIndex] = NumOps.FromDouble(1.0);
                gradient[trueLabel] = NumOps.FromDouble(-1.0);
            }
        }

        return gradient;
    }

    /// <summary>
    /// Computes the gradient using finite-difference approximation as a fallback.
    /// </summary>
    private double[] ComputeFiniteDifferenceGradient(
        double[] w,
        Vector<T> original,
        int trueLabel,
        double c,
        IPredictiveModel<T, Vector<T>, Vector<T>> targetModel)
    {
        var gradient = new double[w.Length];
        const double delta = 0.001;

        // Compute base objective
        var baseAdv = new Vector<T>(w.Length);
        for (int i = 0; i < w.Length; i++)
        {
            var tanhW = Math.Tanh(w[i]);
            baseAdv[i] = NumOps.FromDouble((tanhW + 1.0) / 2.0);
        }
        var baseOutput = targetModel.Predict(baseAdv);
        var baseObjective = ComputeObjective(w, original, baseOutput, trueLabel, c);

        for (int i = 0; i < w.Length; i++)
        {
            var wPerturbed = (double[])w.Clone();
            wPerturbed[i] += delta;

            var advPerturbed = new Vector<T>(wPerturbed.Length);
            for (int j = 0; j < wPerturbed.Length; j++)
            {
                var tanhW = Math.Tanh(wPerturbed[j]);
                advPerturbed[j] = NumOps.FromDouble((tanhW + 1.0) / 2.0);
            }

            var outputPerturbed = targetModel.Predict(advPerturbed);
            var perturbedObjective = ComputeObjective(wPerturbed, original, outputPerturbed, trueLabel, c);
            gradient[i] = (perturbedObjective - baseObjective) / delta;
        }

        return gradient;
    }

    /// <summary>
    /// Computes the objective value for a given w.
    /// </summary>
    private double ComputeObjective(double[] w, Vector<T> original, Vector<T> output, int trueLabel, double c)
    {
        var adversarial = new Vector<T>(w.Length);
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
    private double ComputeAttackLoss(Vector<T> output, int trueLabel)
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
            var targetIndex = MathHelper.Clamp(Options.TargetClass, 0, output.Length - 1);
            var targetLogit = NumOps.ToDouble(output[targetIndex]);
            return Math.Max(maxOtherLogit - targetLogit, 0.0);
        }

        return Math.Max(maxOtherLogit - trueLogit, 0.0);
    }

    /// <summary>
    /// Checks if the attack was successful.
    /// </summary>
    private bool IsSuccessfulAttack(Vector<T> output, int trueLabel)
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
