using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Safety;

/// <summary>
/// RACE: Robust Adversarial Concept Erasure for removing concepts resilient to red-teaming attacks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// RACE addresses the vulnerability of standard concept erasure methods to adversarial prompts
/// that can recover erased concepts. It uses an adversarial training loop: a red-team module
/// generates challenging prompts trying to recover the erased concept, while the erasure module
/// learns to be robust against these attacks. This min-max game produces erasure that resists
/// prompt-based attacks.
/// </para>
/// <para>
/// <b>For Beginners:</b> Standard concept erasure can sometimes be "tricked" by cleverly
/// worded prompts that recover the erased content. RACE prevents this by training with an
/// attacker and defender at the same time. The attacker tries to find prompts that bypass
/// the erasure, and the defender learns to block those too. It's like training a security
/// system by constantly testing it with new attack strategies.
/// </para>
/// <para>
/// Reference: Pham et al., "Robust Concept Erasure via Adversarial Training (RACE)", 2024
/// </para>
/// </remarks>
public class RACEEraser<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly double _adversarialWeight;
    private readonly double _erasureLearningRate;
    private readonly double _attackLearningRate;
    private readonly int _innerSteps;
    private readonly int _outerSteps;

    /// <summary>
    /// Initializes a new RACE eraser.
    /// </summary>
    /// <param name="adversarialWeight">Weight for adversarial loss component (default: 1.0).</param>
    /// <param name="erasureLearningRate">Learning rate for the erasure model updates (default: 1e-5).</param>
    /// <param name="attackLearningRate">Learning rate for the adversarial prompt updates (default: 1e-3).</param>
    /// <param name="innerSteps">Number of inner adversarial attack steps (default: 5).</param>
    /// <param name="outerSteps">Number of outer erasure update steps (default: 1000).</param>
    public RACEEraser(
        double adversarialWeight = 1.0,
        double erasureLearningRate = 1e-5,
        double attackLearningRate = 1e-3,
        int innerSteps = 5,
        int outerSteps = 1000)
    {
        _adversarialWeight = adversarialWeight;
        _erasureLearningRate = erasureLearningRate;
        _attackLearningRate = attackLearningRate;
        _innerSteps = innerSteps;
        _outerSteps = outerSteps;
    }

    /// <summary>
    /// Computes the adversarial erasure loss (defender's loss).
    /// </summary>
    /// <param name="erasedOutput">Model output with the concept supposedly erased.</param>
    /// <param name="targetOutput">Target output (what erased output should look like — neutral content).</param>
    /// <param name="adversarialOutput">Model output under adversarial prompt (attacker's best attempt).</param>
    /// <param name="adversarialTarget">Target for adversarial output (should also be neutral).</param>
    /// <returns>The combined erasure and adversarial robustness loss.</returns>
    public T ComputeDefenderLoss(
        Vector<T> erasedOutput,
        Vector<T> targetOutput,
        Vector<T> adversarialOutput,
        Vector<T> adversarialTarget)
    {
        // Standard erasure loss
        var erasureLoss = ComputeMSE(erasedOutput, targetOutput);

        // Adversarial robustness loss: erased model should also produce neutral for adversarial prompts
        var adversarialLoss = ComputeMSE(adversarialOutput, adversarialTarget);

        // Combined: erasure_loss + adv_weight * adversarial_loss
        var scaledAdv = NumOps.Multiply(NumOps.FromDouble(_adversarialWeight), adversarialLoss);
        return NumOps.Add(erasureLoss, scaledAdv);
    }

    /// <summary>
    /// Computes the attacker's loss (red-team objective, maximized during inner loop).
    /// </summary>
    /// <param name="adversarialOutput">Model output under adversarial prompt.</param>
    /// <param name="conceptTarget">Target output representing the erased concept (attacker wants similarity).</param>
    /// <returns>Negative MSE (attacker wants to minimize distance to concept).</returns>
    public T ComputeAttackerLoss(Vector<T> adversarialOutput, Vector<T> conceptTarget)
    {
        // Attacker maximizes similarity to erased concept = minimizes MSE to concept target
        // So attacker loss = -MSE (negate because defender minimizes)
        var mse = ComputeMSE(adversarialOutput, conceptTarget);
        return NumOps.Negate(mse);
    }

    /// <summary>
    /// Applies a single adversarial perturbation step to a prompt embedding.
    /// </summary>
    /// <param name="promptEmbedding">Current prompt embedding.</param>
    /// <param name="gradient">Gradient of attacker loss with respect to the prompt.</param>
    /// <returns>Perturbed prompt embedding.</returns>
    public Vector<T> AttackerStep(Vector<T> promptEmbedding, Vector<T> gradient)
    {
        // PGD-style attack: x = x + lr * sign(grad) for targeted attack
        var result = new Vector<T>(promptEmbedding.Length);
        var lr = NumOps.FromDouble(_attackLearningRate);

        for (int i = 0; i < result.Length; i++)
        {
            var grad = i < gradient.Length ? gradient[i] : NumOps.Zero;
            var sign = NumOps.FromDouble(NumOps.ToDouble(grad) >= 0 ? 1.0 : -1.0);
            var step = NumOps.Multiply(lr, sign);
            result[i] = NumOps.Add(promptEmbedding[i], step);
        }

        return result;
    }

    private T ComputeMSE(Vector<T> a, Vector<T> b)
    {
        var sum = NumOps.Zero;
        var len = Math.Min(a.Length, b.Length);
        for (int i = 0; i < len; i++)
        {
            var diff = NumOps.Subtract(a[i], b[i]);
            sum = NumOps.Add(sum, NumOps.Multiply(diff, diff));
        }
        return len > 0 ? NumOps.Divide(sum, NumOps.FromDouble(len)) : NumOps.Zero;
    }

    /// <summary>
    /// Gets the adversarial weight.
    /// </summary>
    public double AdversarialWeight => _adversarialWeight;

    /// <summary>
    /// Gets the erasure learning rate.
    /// </summary>
    public double ErasureLearningRate => _erasureLearningRate;

    /// <summary>
    /// Gets the attack learning rate.
    /// </summary>
    public double AttackLearningRate => _attackLearningRate;

    /// <summary>
    /// Gets the number of inner adversarial steps.
    /// </summary>
    public int InnerSteps => _innerSteps;

    /// <summary>
    /// Gets the number of outer erasure steps.
    /// </summary>
    public int OuterSteps => _outerSteps;
}
