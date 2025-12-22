using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AdversarialRobustness.Attacks;

/// <summary>
/// Implements the Carlini and Wagner (C and W) attack.
/// </summary>
/// <remarks>
/// <para>
/// C and W is an optimization-based attack that formulates adversarial example generation as
/// an optimization problem, typically producing stronger attacks than gradient-based methods.
/// </para>
/// <para><b>For Beginners:</b> C and W is one of the most sophisticated attacks. Instead of
/// following gradients, it treats creating adversarial examples as a carefully crafted
/// optimization problem. It's slower than FGSM or PGD but often finds adversarial examples
/// that are more subtle and harder to defend against.</para>
/// <para>
/// Original paper: "Towards Evaluating the Robustness of Neural Networks"
/// by Carlini and Wagner (2017)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type for the model.</typeparam>
/// <typeparam name="TOutput">The output data type for the model.</typeparam>
public class CWAttack<T, TInput, TOutput> : AdversarialAttackBase<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the C and W attack.
    /// </summary>
    /// <param name="options">The configuration options for the attack.</param>
    public CWAttack(AdversarialAttackOptions<T> options) : base(options)
    {
    }

    /// <summary>
    /// Generates an adversarial example using the C and W L2 attack.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The C and W attack solves:
    /// minimize ||delta||_2 + c * f(x + delta)
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
    public override TInput GenerateAdversarialExample(TInput input, TOutput trueLabel, IFullModel<T, TInput, TOutput> targetModel)
    {
        if (input == null)
        {
            throw new ArgumentNullException(nameof(input));
        }

        if (targetModel == null)
        {
            throw new ArgumentNullException(nameof(targetModel));
        }

        // Convert to vector representation
        var vectorInput = ConversionsHelper.ConvertToVector<T, TInput>(input);
        var vectorLabel = ConversionsHelper.ConvertToVector<T, TOutput>(trueLabel);

        var c = 1.0; // Confidence parameter
        var learningRate = 0.01;
        var bestAdversarial = CloneVector(vectorInput);
        var bestPerturbation = double.PositiveInfinity;

        // Extract class index from label vector
        var trueLabelIndex = GetClassIndex(vectorLabel);

        // Initialize perturbation variable (in tanh space for box constraints)
        var w = new double[vectorInput.Length];
        for (int i = 0; i < vectorInput.Length; i++)
        {
            // Initialize w such that tanh(w) approximately equals x
            var x = NumOps.ToDouble(vectorInput[i]);
            var twoXMinusOne = 2.0 * x - 1.0;
            w[i] = MathHelper.Atanh(MathHelper.Clamp(twoXMinusOne, -0.9999, 0.9999));
        }

        // Optimization loop
        for (int iteration = 0; iteration < Options.Iterations; iteration++)
        {
            // Convert from tanh space to valid input range [0, 1]
            var adversarial = TanhSpaceToInputSpace(w);

            // Compute objective and gradient
            var modelInput = ConversionsHelper.ConvertVectorToInput<T, TInput>(adversarial, input);
            var output = targetModel.Predict(modelInput);
            var outputVector = ConversionsHelper.ConvertToVector<T, TOutput>(output);
            var (_, gradient) = ComputeObjectiveAndGradient(w, vectorInput, outputVector, trueLabelIndex, c, input, targetModel);

            // Update w using gradient descent
            for (int i = 0; i < w.Length; i++)
            {
                w[i] -= learningRate * gradient[i];
            }

            // Track best solution
            var perturbation = Engine.Subtract<T>(adversarial, vectorInput);
            var perturbationNorm = NumOps.ToDouble(ComputeL2Norm(perturbation));
            if (IsSuccessfulAttack(outputVector, trueLabelIndex) && perturbationNorm < bestPerturbation)
            {
                bestAdversarial = CloneVector(adversarial);
                bestPerturbation = perturbationNorm;
            }
        }

        return ConversionsHelper.ConvertVectorToInput<T, TInput>(bestAdversarial, input);
    }

    /// <summary>
    /// Converts from tanh space (w) to valid input space [0, 1].
    /// </summary>
    private Vector<T> TanhSpaceToInputSpace(double[] w)
    {
        var result = new Vector<T>(w.Length);
        for (int i = 0; i < w.Length; i++)
        {
            var tanhW = Math.Tanh(w[i]);
            result[i] = NumOps.FromDouble((tanhW + 1.0) / 2.0);
        }
        return result;
    }

    /// <summary>
    /// Clones a vector using vectorized operations.
    /// </summary>
    private Vector<T> CloneVector(Vector<T> input)
    {
        var zeros = Engine.FillZero<T>(input.Length);
        return Engine.Add<T>(input, zeros);
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
        TInput referenceInput,
        IFullModel<T, TInput, TOutput> targetModel)
    {
        var objective = ComputeObjective(w, original, output, trueLabel, c);

        // Check if the model supports analytic gradients
        if (targetModel is IInputGradientComputable<T> gradientComputable)
        {
            return (objective, ComputeAnalyticGradient(w, original, output, trueLabel, c, gradientComputable));
        }

        // Fallback: approximate gradient in w-space using finite differences
        return (objective, ComputeFiniteDifferenceGradient(w, original, trueLabel, c, referenceInput, targetModel));
    }

    /// <summary>
    /// Computes the gradient analytically using the model's backpropagation capabilities.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The C and W objective is: L = ||delta||^2 + c * f(x_adv)
    /// where delta = x_adv - x_orig and f is the attack loss.
    /// </para>
    /// <para>
    /// The gradient in w-space uses the chain rule:
    /// dL/dw = dL/dx_adv * dx_adv/dw
    /// where dx_adv/dw = (1 - tanh^2(w))/2 due to the tanh parameterization.
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

        // Compute gradient of L2 perturbation term using vectorized operations:
        // d(||x_adv - x_orig||^2)/dx_adv = 2 * (x_adv - x_orig)
        var perturbation = Engine.Subtract<T>(adversarial, original);
        var two = NumOps.FromDouble(2.0);
        var perturbGrad = Engine.Multiply<T>(perturbation, two);

        // Compute gradient of attack loss term: df/dx_adv
        var outputGradient = ComputeAttackLossGradient(output, trueLabel);
        var inputGradient = gradientComputable.ComputeInputGradient(adversarial, outputGradient);

        // Combine gradients: dL/dx_adv = perturbGrad + c * inputGradient
        var scaledInputGrad = Engine.Multiply<T>(inputGradient, NumOps.FromDouble(c));
        var totalGrad = Engine.Add<T>(perturbGrad, scaledInputGrad);

        // Apply chain rule: dx_adv/dw = (1 - tanh^2(w))/2
        for (int i = 0; i < n; i++)
        {
            var dLdx = NumOps.ToDouble(totalGrad[i]);
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
        var gradient = Engine.FillZero<T>(output.Length);
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
        TInput referenceInput,
        IFullModel<T, TInput, TOutput> targetModel)
    {
        var gradient = new double[w.Length];
        const double delta = 0.001;

        // Compute base objective
        var baseAdv = TanhSpaceToInputSpace(w);
        var baseModelInput = ConversionsHelper.ConvertVectorToInput<T, TInput>(baseAdv, referenceInput);
        var baseOutput = targetModel.Predict(baseModelInput);
        var baseOutputVector = ConversionsHelper.ConvertToVector<T, TOutput>(baseOutput);
        var baseObjective = ComputeObjective(w, original, baseOutputVector, trueLabel, c);

        for (int i = 0; i < w.Length; i++)
        {
            var wPerturbed = (double[])w.Clone();
            wPerturbed[i] += delta;

            var advPerturbed = TanhSpaceToInputSpace(wPerturbed);

            var perturbedModelInput = ConversionsHelper.ConvertVectorToInput<T, TInput>(advPerturbed, referenceInput);
            var outputPerturbed = targetModel.Predict(perturbedModelInput);
            var outputPerturbedVector = ConversionsHelper.ConvertToVector<T, TOutput>(outputPerturbed);
            var perturbedObjective = ComputeObjective(wPerturbed, original, outputPerturbedVector, trueLabel, c);
            gradient[i] = (perturbedObjective - baseObjective) / delta;
        }

        return gradient;
    }

    /// <summary>
    /// Computes the objective value for a given w.
    /// </summary>
    private double ComputeObjective(double[] w, Vector<T> original, Vector<T> output, int trueLabel, double c)
    {
        var adversarial = TanhSpaceToInputSpace(w);

        // Use vectorized subtraction for perturbation
        var perturbation = Engine.Subtract<T>(adversarial, original);
        var l2Distance = NumOps.ToDouble(ComputeL2Norm(perturbation));
        var l2DistanceSquared = l2Distance * l2Distance;

        var attackLoss = ComputeAttackLoss(output, trueLabel);
        return l2DistanceSquared + c * attackLoss;
    }

    /// <summary>
    /// Computes the attack loss for C and W.
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

    /// <summary>
    /// Gets the class index from a label vector (argmax for one-hot or probability vectors).
    /// </summary>
    private int GetClassIndex(Vector<T> label)
    {
        if (label == null || label.Length == 0)
        {
            return 0;
        }

        int maxIndex = 0;
        T maxValue = label[0];
        for (int i = 1; i < label.Length; i++)
        {
            if (NumOps.GreaterThan(label[i], maxValue))
            {
                maxValue = label[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    /// <inheritdoc/>
    public override TInput CalculatePerturbation(TInput original, TInput adversarial)
    {
        if (original == null)
        {
            throw new ArgumentNullException(nameof(original));
        }

        if (adversarial == null)
        {
            throw new ArgumentNullException(nameof(adversarial));
        }

        var originalVector = ConversionsHelper.ConvertToVector<T, TInput>(original);
        var adversarialVector = ConversionsHelper.ConvertToVector<T, TInput>(adversarial);

        if (originalVector.Length != adversarialVector.Length)
        {
            throw new ArgumentException("Original and adversarial examples must have the same length.");
        }

        // Use vectorized subtraction
        var perturbation = Engine.Subtract<T>(adversarialVector, originalVector);

        return ConversionsHelper.ConvertVectorToInput<T, TInput>(perturbation, original);
    }
}
