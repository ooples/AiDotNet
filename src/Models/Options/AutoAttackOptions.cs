using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration for the AutoAttack ensemble (Croce and Hein 2020).
/// </summary>
/// <remarks>
/// <para>
/// AutoAttack is parameter-free apart from the threat model: the step size, momentum and step-size schedule
/// of APGD are fixed by the paper. These options select the ensemble version and the per-attack budgets of
/// the reference implementation. <see cref="AdversarialAttackOptions{T}.Iterations"/> is the iteration
/// budget of APGD and FAB (100 in the paper).
/// </para>
/// <para><b>For Beginners:</b> You normally only set <see cref="AdversarialAttackOptions{T}.Epsilon"/> and
/// <see cref="AdversarialAttackOptions{T}.NormType"/>; the remaining values are the ones the paper used.</para>
/// <para><b>Reference:</b> Croce and Hein (2020), "Reliable evaluation of adversarial robustness with an ensemble
/// of diverse parameter-free attacks", ICML 2020 (https://arxiv.org/abs/2003.01690); reference code
/// fra31/auto-attack.</para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
[PaperDefault(nameof(Iterations), 100, Source = "Croce and Hein 2020, Sec. 4: APGD and FAB run N_iter = 100 iterations")]
public class AutoAttackOptions<T> : AdversarialAttackOptions<T>
{
    /// <summary>
    /// Initializes the options with the paper's defaults.
    /// </summary>
    public AutoAttackOptions()
    {
        Iterations = 100;
    }

    /// <summary>
    /// Initializes the options by copying every setting of another instance, including the inherited attack
    /// settings.
    /// </summary>
    /// <param name="other">The options to copy.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is null.</exception>
    public AutoAttackOptions(AutoAttackOptions<T> other)
    {
        if (other is null)
        {
            throw new ArgumentNullException(nameof(other));
        }

        Seed = other.Seed;
        Epsilon = other.Epsilon;
        StepSize = other.StepSize;
        Iterations = other.Iterations;
        NormType = other.NormType;
        IsTargeted = other.IsTargeted;
        TargetClass = other.TargetClass;
        UseRandomStart = other.UseRandomStart;
        RandomSeed = other.RandomSeed;
        CWConstant = other.CWConstant;
        CWConfidence = other.CWConfidence;
        CWLearningRate = other.CWLearningRate;
        Version = other.Version;
        TargetClasses = other.TargetClasses;
        SquareQueries = other.SquareQueries;
        SquareInitialFraction = other.SquareInitialFraction;
        FabAlphaMax = other.FabAlphaMax;
        FabEta = other.FabEta;
        FabBeta = other.FabBeta;
    }

    /// <summary>
    /// Gets or sets which attack ensemble runs.
    /// </summary>
    /// <value>Defaults to <see cref="AutoAttackVersion.Standard"/>.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Standard runs the four attacks of the paper's main evaluation. Plus adds APGD-DLR and
    /// untargeted FAB and uses five restarts, which is stronger but several times slower.</para>
    /// </remarks>
    [PaperDefault(AutoAttackVersion.Standard, Source = "Croce and Hein 2020, Sec. 4: APGD-CE, APGD-T, FAB-T and Square")]
    public AutoAttackVersion Version { get; set; } = AutoAttackVersion.Standard;

    /// <summary>
    /// Gets or sets how many of the highest-scoring wrong classes the targeted attacks try in turn.
    /// </summary>
    /// <value>Defaults to 9.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The targeted attacks try to push the input into each of the most likely wrong classes in
    /// turn. More classes make the evaluation stronger and slower; fewer make it faster but can miss an
    /// adversarial example.</para>
    /// </remarks>
    [PaperDefault(9, Source = "Croce and Hein 2020, Sec. 4: the targeted attacks use the top 9 target classes")]
    public int TargetClasses { get; set; } = 9;

    /// <summary>
    /// Gets or sets the query budget of the Square attack.
    /// </summary>
    /// <value>Defaults to 5000.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Square is a black-box random search: every query tries one random change and keeps it if
    /// it helps. More queries find more adversarial examples but cost one model evaluation each.</para>
    /// </remarks>
    [PaperDefault(5000, Source = "Croce and Hein 2020, Sec. 4: Square Attack with 5000 queries")]
    public int SquareQueries { get; set; } = 5000;

    /// <summary>
    /// Gets or sets the initial fraction of the input that one Square attack window covers.
    /// </summary>
    /// <value>Defaults to 0.8.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Square starts by changing large windows covering this fraction of the input and
    /// shrinks them as the search goes on. Larger values explore more coarsely at the start.</para>
    /// </remarks>
    [PaperDefault(0.8, Source = "Croce and Hein 2020, Sec. 4: Square Attack with p_init = 0.8")]
    public double SquareInitialFraction { get; set; } = 0.8;

    /// <summary>
    /// Gets or sets FAB's maximum step bias towards the original point.
    /// </summary>
    /// <value>Defaults to 0.1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> FAB mixes a step from the current point with a step from the original input; this caps how
    /// much weight the original input gets. Larger values keep the perturbation smaller but can slow the
    /// search down.</para>
    /// </remarks>
    [PaperDefault(0.1, Source = "Croce and Hein 2020, Sec. 4 (FAB, Croce and Hein 2020b): alpha_max = 0.1")]
    public double FabAlphaMax { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets FAB's overshoot factor of the projection step.
    /// </summary>
    /// <value>Defaults to 1.05.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> FAB steps slightly past the estimated decision boundary by this factor so the point really
    /// crosses it. Values above 1 overshoot more; 1 stops exactly on the linearized boundary.</para>
    /// </remarks>
    [PaperDefault(1.05, Source = "Croce and Hein 2020, Sec. 4 (FAB, Croce and Hein 2020b): eta = 1.05")]
    public double FabEta { get; set; } = 1.05;

    /// <summary>
    /// Gets or sets FAB's backward step factor after a successful step.
    /// </summary>
    /// <value>Defaults to 0.9.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> After FAB finds an adversarial point it moves back towards the original input by this factor
    /// to look for a smaller change. Smaller values step back further.</para>
    /// </remarks>
    [PaperDefault(0.9, Source = "Croce and Hein 2020, Sec. 4 (FAB, Croce and Hein 2020b): beta = 0.9")]
    public double FabBeta { get; set; } = 0.9;
}
