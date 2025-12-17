using AiDotNet.Models.Options;

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
/// AutoAttack typically includes:
/// - APGD-CE (Auto PGD with Cross-Entropy loss)
/// - APGD-DLR (Auto PGD with DLR loss)
/// - FAB (Fast Adaptive Boundary)
/// - Square Attack
/// </para>
/// <para>
/// Original paper: "Reliable evaluation of adversarial robustness with an ensemble of diverse
/// parameter-free attacks" by Croce &amp; Hein (2020)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class AutoAttack<T> : AdversarialAttackBase<T>
{
    private readonly PGDAttack<T> pgdAttack;
    private readonly CWAttack<T> cwAttack;
    private readonly FGSMAttack<T> fgsmAttack;

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

        pgdAttack = new PGDAttack<T>(pgdOptions);
        cwAttack = new CWAttack<T>(options);
        fgsmAttack = new FGSMAttack<T>(options);
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
    public override T[] GenerateAdversarialExample(T[] input, int trueLabel, Func<T[], T[]> targetModel)
    {
        var candidates = new List<(T[] adversarial, double perturbationSize, bool successful)>();

        // Run PGD attack (strongest gradient-based attack)
        try
        {
            var pgdAdversarial = pgdAttack.GenerateAdversarialExample(input, trueLabel, targetModel);
            var pgdPerturbation = NumOps.ToDouble(ComputeL2Norm(CalculatePerturbation(input, pgdAdversarial)));
            var pgdSuccess = IsSuccessfulAttack(targetModel(pgdAdversarial), trueLabel);
            candidates.Add((pgdAdversarial, pgdPerturbation, pgdSuccess));
        }
        catch (Exception)
        {
            // Continue if one attack fails
        }

        // Run C&W attack (optimization-based)
        try
        {
            var cwAdversarial = cwAttack.GenerateAdversarialExample(input, trueLabel, targetModel);
            var cwPerturbation = NumOps.ToDouble(ComputeL2Norm(CalculatePerturbation(input, cwAdversarial)));
            var cwSuccess = IsSuccessfulAttack(targetModel(cwAdversarial), trueLabel);
            candidates.Add((cwAdversarial, cwPerturbation, cwSuccess));
        }
        catch (Exception)
        {
            // Continue if one attack fails
        }

        // Run FGSM attack (fast single-step)
        try
        {
            var fgsmAdversarial = fgsmAttack.GenerateAdversarialExample(input, trueLabel, targetModel);
            var fgsmPerturbation = NumOps.ToDouble(ComputeL2Norm(CalculatePerturbation(input, fgsmAdversarial)));
            var fgsmSuccess = IsSuccessfulAttack(targetModel(fgsmAdversarial), trueLabel);
            candidates.Add((fgsmAdversarial, fgsmPerturbation, fgsmSuccess));
        }
        catch (Exception)
        {
            // Continue if one attack fails
        }

        // Select the best adversarial example
        // Prioritize: 1) Successful attacks, 2) Smallest perturbation
        T[] bestAdversarial = input;
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
    private bool IsSuccessfulAttack(T[] output, int trueLabel)
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
}
