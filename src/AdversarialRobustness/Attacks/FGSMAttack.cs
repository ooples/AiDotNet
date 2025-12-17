using AiDotNet.Models.Options;

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
    public override T[] GenerateAdversarialExample(T[] input, int trueLabel, Func<T[], T[]> targetModel)
    {
        var epsilon = NumOps.FromDouble(Options.Epsilon);

        // Compute gradient approximation using finite differences
        var gradient = ComputeGradient(input, trueLabel, targetModel);

        // Apply FGSM perturbation
        var adversarial = new T[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            var perturbation = NumOps.Multiply(epsilon, Sign(gradient[i]));

            // For targeted attacks, move towards the target class; for untargeted, move away from the true class
            adversarial[i] = NumOps.Add(input[i], Options.IsTargeted ? NumOps.Negate(perturbation) : perturbation);

            // Clip to valid range (typically [0, 1] for images)
            adversarial[i] = Clip(adversarial[i], NumOps.Zero, NumOps.One);
        }

        return adversarial;
    }

    /// <summary>
    /// Computes an approximation of the gradient using finite differences.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Since we may not have access to the actual gradients,
    /// we approximate them by slightly changing each input dimension and seeing how the
    /// model's output changes. This is like testing the slope of a hill by taking tiny
    /// steps in each direction.</para>
    /// </remarks>
    private T[] ComputeGradient(T[] input, int trueLabel, Func<T[], T[]> targetModel)
    {
        var gradient = new T[input.Length];
        var delta = NumOps.FromDouble(0.001); // Small perturbation for finite differences

        // Get the original prediction
        var originalOutput = targetModel(input);
        var originalLoss = ComputeLoss(originalOutput, trueLabel);

        // Compute gradient for each dimension
        for (int i = 0; i < input.Length; i++)
        {
            // Perturb the input slightly in dimension i
            var perturbedInput = (T[])input.Clone();
            perturbedInput[i] = NumOps.Add(perturbedInput[i], delta);

            // Compute the loss with the perturbed input
            var perturbedOutput = targetModel(perturbedInput);
            var perturbedLoss = ComputeLoss(perturbedOutput, Options.IsTargeted ? Options.TargetClass : trueLabel);

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
    private T ComputeLoss(T[] output, int targetClass)
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
    private T[] Softmax(T[] logits)
    {
        var probabilities = new T[logits.Length];
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

        // Normalize
        if (sum <= 0.0)
            return probabilities;

        for (int i = 0; i < probabilities.Length; i++)
        {
            probabilities[i] = NumOps.FromDouble(NumOps.ToDouble(probabilities[i]) / sum);
        }

        return probabilities;
    }
}
