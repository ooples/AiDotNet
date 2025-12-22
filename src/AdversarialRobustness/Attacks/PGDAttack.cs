using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
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
/// <typeparam name="TInput">The input data type for the model.</typeparam>
/// <typeparam name="TOutput">The output data type for the model.</typeparam>
public class PGDAttack<T, TInput, TOutput> : AdversarialAttackBase<T, TInput, TOutput>
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
    /// x^(t+1) = Project_epsilon(x^(t) + alpha * sign(gradient_x Loss(x^(t), y)))
    /// where Project_epsilon projects back into the epsilon-ball around the original input.
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
        var stepSize = NumOps.FromDouble(Options.StepSize);

        // Extract class index from label vector
        var trueLabelIndex = GetClassIndex(vectorLabel);

        // Initialize adversarial example
        var adversarial = Options.UseRandomStart
            ? RandomStartingPoint(vectorInput, epsilon)
            : CloneVector(vectorInput);

        // Perform iterative PGD steps
        for (int iteration = 0; iteration < Options.Iterations; iteration++)
        {
            // Compute gradient at current point
            var gradient = ComputeGradient(adversarial, trueLabelIndex, input, targetModel);

            // Compute perturbation: stepSize * sign(gradient)
            var signedGradient = SignVector(gradient);
            var perturbation = Engine.Multiply<T>(signedGradient, stepSize);

            // For targeted attacks, negate the perturbation (move towards target class)
            if (Options.IsTargeted)
            {
                perturbation = Engine.Negate<T>(perturbation);
            }

            // adversarial = adversarial + perturbation
            adversarial = Engine.Add<T>(adversarial, perturbation);

            // Project back into the epsilon-ball around the original input
            adversarial = ProjectToEpsilonBall(adversarial, vectorInput, epsilon);

            // Clip to valid range [0, 1]
            adversarial = Engine.Clamp<T>(adversarial, NumOps.Zero, NumOps.One);
        }

        return ConversionsHelper.ConvertVectorToInput<T, TInput>(adversarial, input);
    }

    /// <summary>
    /// Clones a vector using vectorized operations.
    /// </summary>
    private Vector<T> CloneVector(Vector<T> input)
    {
        // Use Engine.Add with a zero vector to create a copy
        var zeros = Engine.FillZero<T>(input.Length);
        return Engine.Add<T>(input, zeros);
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
        // Generate random perturbation using Engine
        var perturbation = Engine.GenerateGaussianNoise<T>(
            input.Length,
            NumOps.Zero,
            epsilon,
            Options.RandomSeed);

        // Project to appropriate norm ball
        perturbation = Options.NormType == "L2"
            ? ProjectL2(perturbation, epsilon)
            : ProjectLInfinity(perturbation, epsilon);

        // randomStart = input + perturbation
        var randomStart = Engine.Add<T>(input, perturbation);

        // Clip to valid range [0, 1]
        return Engine.Clamp<T>(randomStart, NumOps.Zero, NumOps.One);
    }

    /// <summary>
    /// Projects the adversarial example back into the epsilon-ball around the original input.
    /// </summary>
    private Vector<T> ProjectToEpsilonBall(Vector<T> adversarial, Vector<T> original, T epsilon)
    {
        // Compute current perturbation: perturbation = adversarial - original
        var perturbation = Engine.Subtract<T>(adversarial, original);

        // Project based on norm type
        perturbation = Options.NormType == "L2"
            ? ProjectL2(perturbation, epsilon)
            : ProjectLInfinity(perturbation, epsilon);

        // Return projected point: original + projected_perturbation
        return Engine.Add<T>(original, perturbation);
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
        var delta = NumOps.FromDouble(0.001);

        var modelInput = ConversionsHelper.ConvertVectorToInput<T, TInput>(vectorInput, referenceInput);
        var originalOutput = targetModel.Predict(modelInput);
        var originalOutputVector = ConversionsHelper.ConvertToVector<T, TOutput>(originalOutput);
        var originalLoss = ComputeLoss(originalOutputVector, targetClass);

        for (int i = 0; i < vectorInput.Length; i++)
        {
            // Create perturbation vector with delta in dimension i
            var perturbationDelta = Engine.FillZero<T>(vectorInput.Length);
            perturbationDelta[i] = delta;

            // perturbedVector = vectorInput + perturbationDelta
            var perturbedVector = Engine.Add<T>(vectorInput, perturbationDelta);

            var perturbedModelInput = ConversionsHelper.ConvertVectorToInput<T, TInput>(perturbedVector, referenceInput);
            var perturbedOutput = targetModel.Predict(perturbedModelInput);
            var perturbedOutputVector = ConversionsHelper.ConvertToVector<T, TOutput>(perturbedOutput);
            var perturbedLoss = ComputeLoss(perturbedOutputVector, targetClass);

            gradient[i] = NumOps.Divide(NumOps.Subtract(perturbedLoss, originalLoss), delta);
        }

        return gradient;
    }

    /// <summary>
    /// Computes the cross-entropy loss.
    /// </summary>
    private T ComputeLoss(Vector<T> output, int targetClass)
    {
        var probabilities = Engine.Softmax<T>(output);

        if (targetClass >= 0 && targetClass < probabilities.Length)
        {
            var prob = Math.Max(NumOps.ToDouble(probabilities[targetClass]), 1e-10);
            return NumOps.FromDouble(-Math.Log(prob));
        }

        return NumOps.Zero;
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

        // Use vectorized subtraction: perturbation = adversarial - original
        var perturbation = Engine.Subtract<T>(adversarialVector, originalVector);

        return ConversionsHelper.ConvertVectorToInput<T, TInput>(perturbation, original);
    }
}
