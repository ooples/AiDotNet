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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.AnomalyDetection)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Explaining and Harnessing Adversarial Examples", "https://arxiv.org/abs/1412.6572", Year = 2014, Authors = "Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy")]
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

        if (trueLabel == null)
        {
            throw new ArgumentNullException(nameof(trueLabel));
        }

        if (targetModel == null)
        {
            throw new ArgumentNullException(nameof(targetModel));
        }

        // Convert to vector representation for gradient-based operations
        var vectorInput = ConversionsHelper.ConvertToVector<T, TInput>(input);
        var vectorLabel = ConversionsHelper.ConvertToVector<T, TOutput>(trueLabel);

        var epsilon = NumOps.FromDouble(Options.Epsilon);

        // Compute gradient of loss w.r.t. input
        Vector<T> gradient;
        if (targetModel is NeuralNetworks.NeuralNetworkBase<T> nnModel)
        {
            // Tape-based autodiff for neural network models
            gradient = ComputeTapeGradient(vectorInput, vectorLabel, nnModel);
        }
        else
        {
            // Numerical gradient via central finite differences for black-box models
            gradient = ComputeNumericalGradient(vectorInput, vectorLabel, input, targetModel);
        }

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
                "FGSM: gradient computation failed — no gradient for input tensor.");
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
