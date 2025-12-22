using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AdversarialRobustness.Attacks;

/// <summary>
/// Implements the AutoAttack framework - an ensemble of diverse attacks.
/// </summary>
/// <remarks>
/// <para>
/// AutoAttack combines multiple attack methods to provide a reliable evaluation of
/// adversarial robustness without manual parameter tuning.
/// </para>
/// <para><b>For Beginners:</b> AutoAttack is like having multiple expert attackers work together.
/// Instead of using just one attack method, it runs several different attacks and picks the
/// best result. This makes it very thorough and hard to defend against - if your model can
/// resist AutoAttack, it's genuinely robust!</para>
/// <para>
/// This implementation includes:
/// - PGD (Projected Gradient Descent)
/// - C and W (Carlini and Wagner)
/// - FGSM (Fast Gradient Sign Method)
/// </para>
/// <para>
/// Original paper: "Reliable evaluation of adversarial robustness with an ensemble of diverse
/// parameter-free attacks" by Croce and Hein (2020)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type for the model.</typeparam>
/// <typeparam name="TOutput">The output data type for the model.</typeparam>
public class AutoAttack<T, TInput, TOutput> : AdversarialAttackBase<T, TInput, TOutput>
{
    private readonly PGDAttack<T, TInput, TOutput> pgdAttack;
    private readonly CWAttack<T, TInput, TOutput> cwAttack;
    private readonly FGSMAttack<T, TInput, TOutput> fgsmAttack;

    /// <summary>
    /// Initializes a new instance of AutoAttack.
    /// </summary>
    /// <param name="options">The configuration options for the attack.</param>
    public AutoAttack(AdversarialAttackOptions<T> options) : base(options)
    {
        // Initialize component attacks with appropriate parameters
        var pgdOptions = new AdversarialAttackOptions<T>
        {
            Epsilon = options.Epsilon,
            StepSize = options.StepSize,
            Iterations = options.Iterations,
            NormType = options.NormType,
            IsTargeted = options.IsTargeted,
            TargetClass = options.TargetClass,
            UseRandomStart = true,
            RandomSeed = options.RandomSeed
        };

        pgdAttack = new PGDAttack<T, TInput, TOutput>(pgdOptions);
        cwAttack = new CWAttack<T, TInput, TOutput>(options);
        fgsmAttack = new FGSMAttack<T, TInput, TOutput>(options);
    }

    /// <summary>
    /// Generates an adversarial example using the AutoAttack ensemble.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method runs multiple different attacks on the same input
    /// and returns the best adversarial example found. It's like having a team of attackers
    /// each trying their own approach and keeping the most successful result.</para>
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

        // Convert label to vector for class index extraction
        var vectorLabel = ConversionsHelper.ConvertToVector<T, TOutput>(trueLabel);
        var trueLabelIndex = GetClassIndex(vectorLabel);
        var vectorInput = ConversionsHelper.ConvertToVector<T, TInput>(input);

        var candidates = new List<(TInput adversarial, double perturbationSize, bool successful)>();

        // Run PGD attack (strongest gradient-based attack)
        try
        {
            var pgdAdversarial = pgdAttack.GenerateAdversarialExample(input, trueLabel, targetModel);
            var pgdAdversarialVector = ConversionsHelper.ConvertToVector<T, TInput>(pgdAdversarial);
            var pgdPerturbation = Engine.Subtract<T>(pgdAdversarialVector, vectorInput);
            var pgdPerturbationNorm = NumOps.ToDouble(ComputeL2Norm(pgdPerturbation));
            var pgdOutput = targetModel.Predict(pgdAdversarial);
            var pgdSuccess = IsSuccessfulAttack(ConversionsHelper.ConvertToVector<T, TOutput>(pgdOutput), trueLabelIndex);
            candidates.Add((pgdAdversarial, pgdPerturbationNorm, pgdSuccess));
        }
        catch (ArgumentException)
        {
            // Continue if one attack fails
        }
        catch (InvalidOperationException)
        {
            // Continue if one attack fails
        }

        // Run C&W attack (optimization-based)
        try
        {
            var cwAdversarial = cwAttack.GenerateAdversarialExample(input, trueLabel, targetModel);
            var cwAdversarialVector = ConversionsHelper.ConvertToVector<T, TInput>(cwAdversarial);
            var cwPerturbation = Engine.Subtract<T>(cwAdversarialVector, vectorInput);
            var cwPerturbationNorm = NumOps.ToDouble(ComputeL2Norm(cwPerturbation));
            var cwOutput = targetModel.Predict(cwAdversarial);
            var cwSuccess = IsSuccessfulAttack(ConversionsHelper.ConvertToVector<T, TOutput>(cwOutput), trueLabelIndex);
            candidates.Add((cwAdversarial, cwPerturbationNorm, cwSuccess));
        }
        catch (ArgumentException)
        {
            // Continue if one attack fails
        }
        catch (InvalidOperationException)
        {
            // Continue if one attack fails
        }

        // Run FGSM attack (fast single-step)
        try
        {
            var fgsmAdversarial = fgsmAttack.GenerateAdversarialExample(input, trueLabel, targetModel);
            var fgsmAdversarialVector = ConversionsHelper.ConvertToVector<T, TInput>(fgsmAdversarial);
            var fgsmPerturbation = Engine.Subtract<T>(fgsmAdversarialVector, vectorInput);
            var fgsmPerturbationNorm = NumOps.ToDouble(ComputeL2Norm(fgsmPerturbation));
            var fgsmOutput = targetModel.Predict(fgsmAdversarial);
            var fgsmSuccess = IsSuccessfulAttack(ConversionsHelper.ConvertToVector<T, TOutput>(fgsmOutput), trueLabelIndex);
            candidates.Add((fgsmAdversarial, fgsmPerturbationNorm, fgsmSuccess));
        }
        catch (ArgumentException)
        {
            // Continue if one attack fails
        }
        catch (InvalidOperationException)
        {
            // Continue if one attack fails
        }

        // Select the best adversarial example
        // Prioritize: 1) Successful attacks, 2) Smallest perturbation
        TInput bestAdversarial = input;
        double bestPerturbation = double.PositiveInfinity;
        bool foundSuccessful = false;

        foreach (var (adversarial, perturbation, successful) in candidates)
        {
            // If we haven't found a successful attack yet, take any successful one
            if (!foundSuccessful && successful)
            {
                bestAdversarial = adversarial;
                bestPerturbation = perturbation;
                foundSuccessful = true;
            }
            // If we have found successful attacks, prefer the one with smallest perturbation
            else if (foundSuccessful && successful && perturbation < bestPerturbation)
            {
                bestAdversarial = adversarial;
                bestPerturbation = perturbation;
            }
            // If no successful attacks found, just take the smallest perturbation
            else if (!foundSuccessful && perturbation < bestPerturbation)
            {
                bestAdversarial = adversarial;
                bestPerturbation = perturbation;
            }
        }

        return bestAdversarial;
    }

    /// <summary>
    /// Checks if an attack was successful based on the model's output.
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
