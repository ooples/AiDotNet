using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
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
/// <typeparam name="TInput">The input data type for the model.</typeparam>
/// <typeparam name="TOutput">The output data type for the model.</typeparam>
public class FGSMAttack<T, TInput, TOutput> : AdversarialAttackBase<T, TInput, TOutput>
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
    /// x_adv = x + epsilon * sign(gradient_x Loss(x, y_true))
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

        // Convert to vector representation for gradient-based operations
        var vectorInput = ConversionsHelper.ConvertToVector<T, TInput>(input);
        var vectorLabel = ConversionsHelper.ConvertToVector<T, TOutput>(trueLabel);

        var epsilon = NumOps.FromDouble(Options.Epsilon);

        // Extract class index from label vector (argmax for one-hot or probability vectors)
        var trueLabelIndex = GetClassIndex(vectorLabel);

        // Compute gradient using vectorized operations
        var gradient = ComputeGradient(vectorInput, trueLabelIndex, input, targetModel);

        // Apply FGSM perturbation using vectorized operations:
        // perturbation = epsilon * sign(gradient)
        var signedGradient = SignVector(gradient);
        var perturbation = Engine.Multiply<T>(signedGradient, epsilon);

        // For targeted attacks, negate the perturbation (move towards target class)
        if (Options.IsTargeted)
        {
            perturbation = Engine.Negate<T>(perturbation);
        }

        // adversarial = input + perturbation
        var adversarial = Engine.Add<T>(vectorInput, perturbation);

        // Clip to valid range [0, 1] using vectorized clamp
        adversarial = Engine.Clamp<T>(adversarial, NumOps.Zero, NumOps.One);

        return ConversionsHelper.ConvertVectorToInput<T, TInput>(adversarial, input);
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
    private Vector<T> ComputeGradient(Vector<T> vectorInput, int trueLabel, TInput referenceInput, IFullModel<T, TInput, TOutput> targetModel)
    {
        // Determine which class to compute gradient for
        var targetClass = Options.IsTargeted ? Options.TargetClass : trueLabel;

        // Check if the model supports analytic gradients
        if (targetModel is IInputGradientComputable<T> gradientComputable)
        {
            return ComputeAnalyticGradient(vectorInput, targetClass, referenceInput, targetModel, gradientComputable);
        }

        // Fallback to finite differences
        return ComputeFiniteDifferenceGradient(vectorInput, targetClass, referenceInput, targetModel);
    }

    /// <summary>
    /// Computes the gradient analytically using the model's backpropagation capabilities.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For cross-entropy loss with softmax output, the gradient of the loss with respect to
    /// the logits is: dL/dz = p - one_hot(target_class)
    /// where p is the softmax probabilities.
    /// </para>
    /// <para>
    /// This is then backpropagated through the model to get dL/dx (the input gradient).
    /// </para>
    /// </remarks>
    private Vector<T> ComputeAnalyticGradient(
        Vector<T> vectorInput,
        int targetClass,
        TInput referenceInput,
        IFullModel<T, TInput, TOutput> targetModel,
        IInputGradientComputable<T> gradientComputable)
    {
        // Get the model's output
        var modelInput = ConversionsHelper.ConvertVectorToInput<T, TInput>(vectorInput, referenceInput);
        var output = targetModel.Predict(modelInput);
        var outputVector = ConversionsHelper.ConvertToVector<T, TOutput>(output);

        // Compute softmax probabilities using vectorized Engine operation
        var probabilities = Engine.Softmax<T>(outputVector);

        // Compute gradient of cross-entropy loss w.r.t. logits: dL/dz = p - one_hot(target)
        // Create one-hot vector for target class
        var oneHot = Engine.FillZero<T>(outputVector.Length);
        oneHot[targetClass] = NumOps.One;

        // outputGradient = probabilities - oneHot
        var outputGradient = Engine.Subtract<T>(probabilities, oneHot);

        // Backpropagate to get input gradient
        return gradientComputable.ComputeInputGradient(vectorInput, outputGradient);
    }

    /// <summary>
    /// Computes the gradient using finite-difference approximation as a fallback.
    /// </summary>
    private Vector<T> ComputeFiniteDifferenceGradient(
        Vector<T> vectorInput,
        int targetClass,
        TInput referenceInput,
        IFullModel<T, TInput, TOutput> targetModel)
    {
        var gradient = new Vector<T>(vectorInput.Length);
        var delta = NumOps.FromDouble(0.001); // Small perturbation for finite differences

        // Get the original prediction and loss
        var modelInput = ConversionsHelper.ConvertVectorToInput<T, TInput>(vectorInput, referenceInput);
        var originalOutput = targetModel.Predict(modelInput);
        var originalOutputVector = ConversionsHelper.ConvertToVector<T, TOutput>(originalOutput);
        var originalLoss = ComputeLoss(originalOutputVector, targetClass);

        // Create a delta vector for each dimension and compute gradients
        // Note: Finite differences inherently requires per-dimension evaluation
        // but we use vectorized operations within each evaluation
        for (int i = 0; i < vectorInput.Length; i++)
        {
            // Create perturbation vector with delta in dimension i
            var perturbationDelta = Engine.FillZero<T>(vectorInput.Length);
            perturbationDelta[i] = delta;

            // perturbedVector = vectorInput + perturbationDelta
            var perturbedVector = Engine.Add<T>(vectorInput, perturbationDelta);

            // Compute the loss with the perturbed input
            var perturbedModelInput = ConversionsHelper.ConvertVectorToInput<T, TInput>(perturbedVector, referenceInput);
            var perturbedOutput = targetModel.Predict(perturbedModelInput);
            var perturbedOutputVector = ConversionsHelper.ConvertToVector<T, TOutput>(perturbedOutput);
            var perturbedLoss = ComputeLoss(perturbedOutputVector, targetClass);

            // Approximate gradient using finite difference: (f(x+h) - f(x)) / h
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
        // Apply softmax to get probabilities using vectorized Engine operation
        var probabilities = Engine.Softmax<T>(output);

        // Compute negative log-likelihood (cross-entropy loss)
        if (targetClass >= 0 && targetClass < probabilities.Length)
        {
            // Use Engine.Log for the target probability, clamped to avoid log(0)
            var targetProb = probabilities[targetClass];
            var prob = Math.Max(NumOps.ToDouble(targetProb), 1e-10);
            return NumOps.FromDouble(-Math.Log(prob));
        }

        return NumOps.Zero;
    }

    /// <summary>
    /// Gets the class index from a label vector (argmax for one-hot or probability vectors).
    /// </summary>
    /// <param name="label">The label vector.</param>
    /// <returns>The index of the maximum value (class index).</returns>
    private int GetClassIndex(Vector<T> label)
    {
        if (label == null || label.Length == 0)
        {
            return 0;
        }

        // Find argmax - this is inherently a sequential operation
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

        // Use vectorized subtraction: perturbation = adversarial - original
        var perturbation = Engine.Subtract<T>(adversarialVector, originalVector);

        return ConversionsHelper.ConvertVectorToInput<T, TInput>(perturbation, original);
    }
}
