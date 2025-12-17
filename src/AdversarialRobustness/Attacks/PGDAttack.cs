using AiDotNet.Models.Options;

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
    public override T[] GenerateAdversarialExample(T[] input, int trueLabel, Func<T[], T[]> targetModel)
    {
        var epsilon = NumOps.FromDouble(Options.Epsilon);
        var stepSize = NumOps.FromDouble(Options.StepSize);

        // Initialize adversarial example
        var adversarial = Options.UseRandomStart
            ? RandomStartingPoint(input, epsilon)
            : (T[])input.Clone();

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
                adversarial[i] = Clip(adversarial[i], NumOps.Zero, NumOps.One);
            }
        }

        return adversarial;
    }

    /// <summary>
    /// Generates a random starting point within the epsilon-ball.
    /// </summary>
    private T[] RandomStartingPoint(T[] input, T epsilon)
    {
        var randomStart = new T[input.Length];

        for (int i = 0; i < input.Length; i++)
        {
            // Generate random perturbation in [-epsilon, epsilon]
            var randomValue = NumOps.FromDouble(Random.NextDouble() * 2.0 - 1.0);
            var perturbation = NumOps.Multiply(epsilon, randomValue);

            randomStart[i] = NumOps.Add(input[i], perturbation);
            randomStart[i] = Clip(randomStart[i], NumOps.Zero, NumOps.One);
        }

        return randomStart;
    }

    /// <summary>
    /// Projects the adversarial example back into the epsilon-ball around the original input.
    /// </summary>
    private T[] ProjectToEpsilonBall(T[] adversarial, T[] original, T epsilon)
    {
        var projected = new T[adversarial.Length];
        var perturbation = new T[adversarial.Length];

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
    /// Computes an approximation of the gradient using finite differences.
    /// </summary>
    private T[] ComputeGradient(T[] input, int trueLabel, Func<T[], T[]> targetModel)
    {
        var gradient = new T[input.Length];
        var delta = NumOps.FromDouble(0.001);

        var originalOutput = targetModel(input);
        var originalLoss = ComputeLoss(originalOutput, Options.IsTargeted ? Options.TargetClass : trueLabel);

        for (int i = 0; i < input.Length; i++)
        {
            var perturbedInput = (T[])input.Clone();
            perturbedInput[i] = NumOps.Add(perturbedInput[i], delta);

            var perturbedOutput = targetModel(perturbedInput);
            var perturbedLoss = ComputeLoss(perturbedOutput, Options.IsTargeted ? Options.TargetClass : trueLabel);

            gradient[i] = NumOps.Divide(NumOps.Subtract(perturbedLoss, originalLoss), delta);
        }

        return gradient;
    }

    /// <summary>
    /// Computes the cross-entropy loss.
    /// </summary>
    private T ComputeLoss(T[] output, int targetClass)
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
    private T[] Softmax(T[] logits)
    {
        var probabilities = new T[logits.Length];
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

        if (sum <= 0.0)
            return probabilities;

        for (int i = 0; i < probabilities.Length; i++)
        {
            probabilities[i] = NumOps.FromDouble(NumOps.ToDouble(probabilities[i]) / sum);
        }

        return probabilities;
    }
}
