using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.AnomalyDetection)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Towards Deep Learning Models Resistant to Adversarial Attacks", "https://arxiv.org/abs/1706.06083", Year = 2017, Authors = "Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, Adrian Vladu")]
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

        // Initialize adversarial example
        var adversarial = Options.UseRandomStart
            ? RandomStartingPoint(vectorInput, epsilon)
            : CloneVector(vectorInput);

        // Perform iterative PGD steps
        for (int iteration = 0; iteration < Options.Iterations; iteration++)
        {
            // Compute gradient of loss w.r.t. input
            Vector<T> gradient;
            if (targetModel is NeuralNetworks.NeuralNetworkBase<T> nnModel)
            {
                // Tape-based autodiff for neural network models
                gradient = ComputeTapeGradient(adversarial, vectorLabel, nnModel);
            }
            else
            {
                // Numerical gradient via central finite differences for black-box models
                gradient = ComputeNumericalGradient(adversarial, vectorLabel, input, targetModel);
            }

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
    /// Computes exact gradient of loss w.r.t. input using tape-based autodiff.
    /// Only works with NeuralNetworkBase models whose forward pass is tape-recorded.
    /// </summary>
    private Vector<T> ComputeTapeGradient(
        Vector<T> vectorInput,
        Vector<T> vectorLabel,
        NeuralNetworks.NeuralNetworkBase<T> nnModel)
    {
        var eng = AiDotNetEngine.Current;
        var inputTensor = Tensor<T>.FromVector(vectorInput);
        var targetTensor = Tensor<T>.FromVector(vectorLabel);
        using var tape = new GradientTape<T>();
        var outputTensor = nnModel.ForwardForTraining(inputTensor);
        var diff = eng.TensorSubtract(outputTensor, targetTensor);
        var squared = eng.TensorMultiply(diff, diff);
        var allAxes = Enumerable.Range(0, squared.Shape.Length).ToArray();
        var loss = eng.ReduceMean(squared, allAxes, keepDims: false);
        var grads = tape.ComputeGradients(loss, [inputTensor]);
        if (!grads.TryGetValue(inputTensor, out var g))
            throw new InvalidOperationException(
                "PGD: gradient computation failed — no gradient for input tensor.");
        return g.ToVector();
    }

    /// <summary>
    /// Computes the gradient of MSE loss w.r.t. input using central finite differences.
    /// This is the standard approach for adversarial attacks on black-box models:
    /// grad_i ≈ (loss(x + δ*e_i) - loss(x - δ*e_i)) / (2δ)
    /// </summary>
    private Vector<T> ComputeNumericalGradient(
        Vector<T> vectorInput,
        Vector<T> vectorLabel,
        TInput referenceInput,
        IFullModel<T, TInput, TOutput> targetModel)
    {
        const double delta = 1e-4;
        var gradient = new Vector<T>(vectorInput.Length);

        for (int i = 0; i < vectorInput.Length; i++)
        {
            // Forward: x + delta * e_i (clamped to [0,1] valid input domain)
            var plusInput = Engine.Add<T>(vectorInput, Engine.FillZero<T>(vectorInput.Length));
            plusInput[i] = NumOps.Add(plusInput[i], NumOps.FromDouble(delta));
            double plusVal = Math.Min(1.0, Math.Max(0.0, NumOps.ToDouble(plusInput[i])));
            plusInput[i] = NumOps.FromDouble(plusVal);
            var plusModelInput = ConversionsHelper.ConvertVectorToInput<T, TInput>(plusInput, referenceInput);
            var plusOutput = ConversionsHelper.ConvertToVector<T, TOutput>(targetModel.Predict(plusModelInput));
            var plusLoss = ComputeMseLoss(plusOutput, vectorLabel);

            // Backward: x - delta * e_i (clamped to [0,1])
            var minusInput = Engine.Add<T>(vectorInput, Engine.FillZero<T>(vectorInput.Length));
            minusInput[i] = NumOps.Subtract(minusInput[i], NumOps.FromDouble(delta));
            double minusVal = Math.Min(1.0, Math.Max(0.0, NumOps.ToDouble(minusInput[i])));
            minusInput[i] = NumOps.FromDouble(minusVal);
            var minusModelInput = ConversionsHelper.ConvertVectorToInput<T, TInput>(minusInput, referenceInput);
            var minusOutput = ConversionsHelper.ConvertToVector<T, TOutput>(targetModel.Predict(minusModelInput));
            var minusLoss = ComputeMseLoss(minusOutput, vectorLabel);

            // Central difference using actual clamped step width
            double actualDelta = plusVal - minusVal;
            gradient[i] = actualDelta > 1e-10
                ? NumOps.FromDouble((NumOps.ToDouble(plusLoss) - NumOps.ToDouble(minusLoss)) / actualDelta)
                : NumOps.Zero;
        }

        return gradient;
    }

    /// <summary>
    /// Computes the mean squared error loss between output and target vectors.
    /// </summary>
    private T ComputeMseLoss(Vector<T> output, Vector<T> target)
    {
        var diff = Engine.Subtract<T>(output, target);
        var squared = new Vector<T>(diff.Length);
        for (int i = 0; i < diff.Length; i++)
            squared[i] = NumOps.Multiply(diff[i], diff[i]);
        var sum = NumOps.Zero;
        for (int i = 0; i < squared.Length; i++)
        {
            sum = NumOps.Add(sum, squared[i]);
        }
        return NumOps.Divide(sum, NumOps.FromDouble(squared.Length));
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
