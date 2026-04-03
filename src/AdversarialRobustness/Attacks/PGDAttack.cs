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
[ModelPaper("Towards Deep Learning Models Resistant to Adversarial Attacks", "https://arxiv.org/abs/1706.06083", Year = 2017, Authors = "Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, Adrian Vladu")]
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
            // Compute exact gradient of loss w.r.t. input using tape-based autodiff
            Vector<T> gradient;
            {
                var eng = AiDotNetEngine.Current;
                var inputTensor = Tensor<T>.FromVector(adversarial);
                var targetTensor = Tensor<T>.FromVector(vectorLabel);
                using var tape = new GradientTape<T>();
                // Use ForwardForTraining so ops are tape-recorded for gradient computation
                Tensor<T> outputTensor;
                if (targetModel is NeuralNetworks.NeuralNetworkBase<T> nnModel)
                {
                    outputTensor = nnModel.ForwardForTraining(inputTensor);
                }
                else
                {
                    var modelIn = ConversionsHelper.ConvertVectorToInput<T, TInput>(adversarial, input);
                    var output = targetModel.Predict(modelIn);
                    outputTensor = Tensor<T>.FromVector(ConversionsHelper.ConvertToVector<T, TOutput>(output));
                }
                var diff = eng.TensorSubtract(outputTensor, targetTensor);
                var squared = eng.TensorMultiply(diff, diff);
                var allAxes = Enumerable.Range(0, squared.Shape.Length).ToArray();
                var loss = eng.ReduceMean(squared, allAxes, keepDims: false);
                var grads = tape.ComputeGradients(loss, [inputTensor]);
                if (!grads.TryGetValue(inputTensor, out var g))
                    throw new InvalidOperationException(
                        "PGD: gradient computation failed — no gradient for input tensor. " +
                        "The target model must be a NeuralNetworkBase<T> for tape-based adversarial attacks.");
                gradient = g.ToVector();
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
