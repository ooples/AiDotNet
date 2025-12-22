using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AdversarialRobustness.Attacks;

/// <summary>
/// Implements the Projected Gradient Descent (PGD) attack.
/// </summary>
/// <remarks>
/// <para>
/// PGD is an iterative variant of FGSM that applies multiple small perturbation steps,
/// projecting back into the allowed perturbation region after each step.
/// </para>
/// <para><b>For Beginners:</b> PGD is like FGSM but repeated multiple times with smaller steps.
/// Instead of one big jump, it takes many small steps, checking after each step to make sure
/// it hasn't gone too far. This makes it much more powerful than FGSM but also slower.</para>
/// <para>
/// PGD is considered one of the strongest first-order adversarial attacks and is commonly
/// used for adversarial training and robustness evaluation.
/// </para>
/// <para>
/// Original paper: "Towards Deep Learning Models Resistant to Adversarial Attacks"
/// by Madry et al. (2017)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class PGDAttack<T> : AdversarialAttackBase<T>
{
    /// <summary>
    /// Initializes a new instance of the PGD attack.
    /// </summary>
    /// <param name="options">The configuration options for the attack.</param>
    public PGDAttack(AdversarialAttackOptions<T> options) : base(options)
    {
    }

    /// <summary>
    /// Generates an adversarial example using the PGD attack.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The PGD attack iteratively computes:
    /// x^(t+1) = Π_ε(x^(t) + α * sign(∇_x Loss(x^(t), y)))
    /// where Π_ε projects back into the epsilon-ball around the original input.
    /// </para>
    /// <para><b>For Beginners:</b> This method:
    /// 1. Starts from a random point near the original input (optional)
    /// 2. Takes a small step in the direction that increases the model's error
    /// 3. Makes sure the step didn't go too far from the original
    /// 4. Repeats this process multiple times
    /// 5. Returns the final adversarial example that's hard for the model</para>
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
        var stepSize = NumOps.FromDouble(Options.StepSize);

        // Initialize adversarial example
        var adversarial = Options.UseRandomStart
            ? RandomStartingPoint(input, epsilon)
            : CloneInput(input);

        // Perform iterative PGD steps
        for (int iteration = 0; iteration < Options.Iterations; iteration++)
        {
            // Compute gradient at current point
            var gradient = ComputeGradient(adversarial, trueLabel, targetModel);

            // Take a step in the gradient direction
            for (int i = 0; i < adversarial.Length; i++)
            {
                var perturbation = NumOps.Multiply(stepSize, Sign(gradient[i]));
                adversarial[i] = NumOps.Add(adversarial[i], Options.IsTargeted ? NumOps.Negate(perturbation) : perturbation);
            }

            // Project back into the epsilon-ball around the original input
            adversarial = ProjectToEpsilonBall(adversarial, input, epsilon);

            // Clip to valid range
            for (int i = 0; i < adversarial.Length; i++)
            {
                adversarial[i] = MathHelper.Clamp(adversarial[i], NumOps.Zero, NumOps.One);
            }
        }

        return adversarial;
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
    /// Generates a random starting point within the epsilon-ball.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For L-infinity norm, each dimension is independently sampled from [-epsilon, epsilon].
    /// For L2 norm, the perturbation is projected to the L2 ball to ensure the total
    /// perturbation magnitude doesn't exceed epsilon.
    /// </para>
    /// </remarks>
    private Vector<T> RandomStartingPoint(Vector<T> input, T epsilon)
    {
        var randomStart = new Vector<T>(input.Length);
        var perturbation = new Vector<T>(input.Length);

        // Generate random perturbation for each dimension
        for (int i = 0; i < input.Length; i++)
        {
            // Generate random perturbation in [-epsilon, epsilon]
            var randomValue = NumOps.FromDouble(Random.NextDouble() * 2.0 - 1.0);
            perturbation[i] = NumOps.Multiply(epsilon, randomValue);
        }

        // Project to appropriate norm ball
        perturbation = Options.NormType == "L2"
            ? ProjectL2(perturbation, epsilon)
            : ProjectLInfinity(perturbation, epsilon);

        // Apply perturbation and clip to valid range
        for (int i = 0; i < input.Length; i++)
        {
            randomStart[i] = NumOps.Add(input[i], perturbation[i]);
            randomStart[i] = MathHelper.Clamp(randomStart[i], NumOps.Zero, NumOps.One);
        }

        return randomStart;
    }

    /// <summary>
    /// Projects the adversarial example back into the epsilon-ball around the original input.
    /// </summary>
    private Vector<T> ProjectToEpsilonBall(Vector<T> adversarial, Vector<T> original, T epsilon)
    {
        var projected = new Vector<T>(adversarial.Length);
        var perturbation = new Vector<T>(adversarial.Length);

        // Compute current perturbation
        for (int i = 0; i < adversarial.Length; i++)
        {
            perturbation[i] = NumOps.Subtract(adversarial[i], original[i]);
        }

        // Project based on norm type
        perturbation = Options.NormType == "L2"
            ? ProjectL2(perturbation, epsilon)
            : ProjectLInfinity(perturbation, epsilon);

        // Apply projected perturbation
        for (int i = 0; i < adversarial.Length; i++)
        {
            projected[i] = NumOps.Add(original[i], perturbation[i]);
        }

        return projected;
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
        var delta = NumOps.FromDouble(0.001);

        var originalOutput = targetModel.Predict(input);
        var originalLoss = ComputeLoss(originalOutput, targetClass);

        for (int i = 0; i < input.Length; i++)
        {
            var perturbedInput = CloneInput(input);
            perturbedInput[i] = NumOps.Add(perturbedInput[i], delta);

            var perturbedOutput = targetModel.Predict(perturbedInput);
            var perturbedLoss = ComputeLoss(perturbedOutput, targetClass);

            gradient[i] = NumOps.Divide(NumOps.Subtract(perturbedLoss, originalLoss), delta);
        }

        return gradient;
    }

    /// <summary>
    /// Computes the cross-entropy loss.
    /// </summary>
    private T ComputeLoss(Vector<T> output, int targetClass)
    {
        var probabilities = Softmax(output);

        if (targetClass >= 0 && targetClass < probabilities.Length)
        {
            var prob = Math.Max(NumOps.ToDouble(probabilities[targetClass]), 1e-10);
            return NumOps.FromDouble(-Math.Log(prob));
        }

        return NumOps.Zero;
    }

    /// <summary>
    /// Applies the softmax function.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Uses numerical stability trick of subtracting the maximum logit before exponentiation.
    /// In the edge case where sum is zero or negative (which shouldn't occur with proper inputs),
    /// returns a uniform distribution as a fallback.
    /// </para>
    /// </remarks>
    private Vector<T> Softmax(Vector<T> logits)
    {
        var probabilities = new Vector<T>(logits.Length);
        double maxLogit = NumOps.ToDouble(logits[0]);

        for (int i = 1; i < logits.Length; i++)
        {
            maxLogit = Math.Max(maxLogit, NumOps.ToDouble(logits[i]));
        }

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

        for (int i = 0; i < probabilities.Length; i++)
        {
            probabilities[i] = NumOps.FromDouble(NumOps.ToDouble(probabilities[i]) / sum);
        }

        return probabilities;
    }
}
