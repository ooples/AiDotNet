using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AdversarialRobustness.Attacks;

/// <summary>
/// Implements the Fast Gradient Sign Method (FGSM) attack.
/// </summary>
/// <remarks>
/// <para>
/// FGSM is a simple yet effective white-box adversarial attack that uses the gradient
/// of the loss function to create adversarial examples in a single step.
/// </para>
/// <para><b>For Beginners:</b> FGSM is like finding the steepest hill and taking one big step
/// in that direction. It's fast but might not be as powerful as multi-step attacks like PGD.
/// Think of it as the "quick and dirty" attack - it's not the strongest, but it's very efficient.</para>
/// <para>
/// Original paper: "Explaining and Harnessing Adversarial Examples" by Goodfellow et al. (2014)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class FGSMAttack<T> : AdversarialAttackBase<T>
{
    /// <summary>
    /// Initializes a new instance of the FGSM attack.
    /// </summary>
    /// <param name="options">The configuration options for the attack.</param>
    public FGSMAttack(AdversarialAttackOptions<T> options) : base(options)
    {
    }

    /// <summary>
    /// Generates an adversarial example using the FGSM attack.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The FGSM attack computes:
    /// x_adv = x + epsilon * sign(∇_x Loss(x, y_true))
    /// </para>
    /// <para><b>For Beginners:</b> This method:
    /// 1. Calculates how the model's error changes when you modify the input
    /// 2. Takes the sign (direction) of this change
    /// 3. Moves the input in that direction by a small amount (epsilon)
    /// 4. The result fools the model while looking similar to the original</para>
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

        var epsilon = NumOps.FromDouble(Options.Epsilon);

        // Compute gradient approximation using finite differences
        var gradient = ComputeGradient(input, trueLabel, targetModel);

        // Apply FGSM perturbation
        var adversarial = new Vector<T>(input.Length);
        for (int i = 0; i < input.Length; i++)
        {
            var perturbation = NumOps.Multiply(epsilon, Sign(gradient[i]));

            // For targeted attacks, move towards the target class; for untargeted, move away from the true class
            adversarial[i] = NumOps.Add(input[i], Options.IsTargeted ? NumOps.Negate(perturbation) : perturbation);

            // Clip to valid range (typically [0, 1] for images)
            adversarial[i] = MathHelper.Clamp(adversarial[i], NumOps.Zero, NumOps.One);
        }

        return adversarial;
    }

    /// <summary>
    /// Computes the gradient of the loss with respect to the input.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When the target model implements <see cref="IInputGradientComputable{T}"/>, this method uses
    /// analytic gradient computation via backpropagation, which is more accurate and efficient.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how changing each input dimension
    /// affects the model's loss. With analytic gradients, we use the model's internal
    /// backpropagation; otherwise, we approximate by testing small changes.</para>
    /// </remarks>
    private Vector<T> ComputeGradient(Vector<T> input, int trueLabel, IPredictiveModel<T, Vector<T>, Vector<T>> targetModel)
    {
        // Determine which class to compute gradient for
        var targetClass = Options.IsTargeted ? Options.TargetClass : trueLabel;

        // Check if the model supports analytic gradients
        if (targetModel is IInputGradientComputable<T> gradientComputable)
        {
            return ComputeAnalyticGradient(input, targetClass, targetModel, gradientComputable);
        }

        // Fallback to finite differences
        return ComputeFiniteDifferenceGradient(input, targetClass, targetModel);
    }

    /// <summary>
    /// Computes the gradient analytically using the model's backpropagation capabilities.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For cross-entropy loss with softmax output, the gradient of the loss with respect to
    /// the logits is: ∂L/∂z = p - one_hot(target_class)
    /// where p is the softmax probabilities.
    /// </para>
    /// <para>
    /// This is then backpropagated through the model to get ∂L/∂x (the input gradient).
    /// </para>
    /// </remarks>
    private Vector<T> ComputeAnalyticGradient(
        Vector<T> input,
        int targetClass,
        IPredictiveModel<T, Vector<T>, Vector<T>> targetModel,
        IInputGradientComputable<T> gradientComputable)
    {
        // Get the model's output
        var output = targetModel.Predict(input);

        // Compute softmax probabilities
        var probabilities = Softmax(output);

        // Compute gradient of cross-entropy loss w.r.t. logits: ∂L/∂z = p - one_hot(target)
        // This is the standard gradient for cross-entropy loss with softmax
        var outputGradient = new Vector<T>(output.Length);
        for (int i = 0; i < output.Length; i++)
        {
            // Gradient: ∂L/∂z[target] = p[target] - 1, ∂L/∂z[i] = p[i] for i != target
            outputGradient[i] = i == targetClass
                ? NumOps.Subtract(probabilities[i], NumOps.One)
                : probabilities[i];
        }

        // Backpropagate to get input gradient
        return gradientComputable.ComputeInputGradient(input, outputGradient);
    }

    /// <summary>
    /// Computes the gradient using finite-difference approximation as a fallback.
    /// </summary>
    private Vector<T> ComputeFiniteDifferenceGradient(
        Vector<T> input,
        int targetClass,
        IPredictiveModel<T, Vector<T>, Vector<T>> targetModel)
    {
        var gradient = new Vector<T>(input.Length);
        var delta = NumOps.FromDouble(0.001); // Small perturbation for finite differences

        // Get the original prediction and loss
        var originalOutput = targetModel.Predict(input);
        var originalLoss = ComputeLoss(originalOutput, targetClass);

        // Compute gradient for each dimension
        for (int i = 0; i < input.Length; i++)
        {
            // Perturb the input slightly in dimension i
            var perturbedInput = new Vector<T>(input.Length);
            for (int j = 0; j < input.Length; j++)
            {
                perturbedInput[j] = input[j];
            }
            perturbedInput[i] = NumOps.Add(perturbedInput[i], delta);

            // Compute the loss with the perturbed input (use same targetClass for consistency)
            var perturbedOutput = targetModel.Predict(perturbedInput);
            var perturbedLoss = ComputeLoss(perturbedOutput, targetClass);

            // Approximate gradient using finite difference
            gradient[i] = NumOps.Divide(NumOps.Subtract(perturbedLoss, originalLoss), delta);
        }

        return gradient;
    }

    /// <summary>
    /// Computes the cross-entropy loss for classification.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Loss measures how wrong the model's prediction is.
    /// Higher loss means the model is more confused. We use this to guide our attack.</para>
    /// </remarks>
    private T ComputeLoss(Vector<T> output, int targetClass)
    {
        // Apply softmax to get probabilities
        var probabilities = Softmax(output);

        // Compute negative log-likelihood (cross-entropy loss)
        if (targetClass >= 0 && targetClass < probabilities.Length)
        {
            var prob = Math.Max(NumOps.ToDouble(probabilities[targetClass]), 1e-10); // Avoid log(0)
            return NumOps.FromDouble(-Math.Log(prob));
        }

        return NumOps.Zero;
    }

    /// <summary>
    /// Applies the softmax function to convert logits to probabilities.
    /// </summary>
    private Vector<T> Softmax(Vector<T> logits)
    {
        var probabilities = new Vector<T>(logits.Length);
        double maxLogit = NumOps.ToDouble(logits[0]);

        // Find max for numerical stability
        for (int i = 1; i < logits.Length; i++)
        {
            maxLogit = Math.Max(maxLogit, NumOps.ToDouble(logits[i]));
        }

        // Compute exp(logit - max)
        double sum = 0.0;
        for (int i = 0; i < logits.Length; i++)
        {
            var shifted = NumOps.ToDouble(logits[i]) - maxLogit;
            var expVal = Math.Exp(shifted);
            probabilities[i] = NumOps.FromDouble(expVal);
            sum += expVal;
        }

        // Edge case: if sum is zero or negative (shouldn't happen with valid inputs),
        // fall back to uniform distribution to avoid NaN/Infinity values
        if (sum <= 0.0)
        {
            var uniform = NumOps.FromDouble(1.0 / logits.Length);
            for (int i = 0; i < probabilities.Length; i++)
            {
                probabilities[i] = uniform;
            }
            return probabilities;
        }

        // Normalize to get valid probability distribution
        for (int i = 0; i < probabilities.Length; i++)
        {
            probabilities[i] = NumOps.FromDouble(NumOps.ToDouble(probabilities[i]) / sum);
        }

        return probabilities;
    }
}
