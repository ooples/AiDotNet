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
    /// Gets or sets which attack ensemble runs.
    /// </summary>
    /// <value>Defaults to <see cref="AutoAttackVersion.Standard"/>.</value>
    [PaperDefault(AutoAttackVersion.Standard, Source = "Croce and Hein 2020, Sec. 4: APGD-CE, APGD-T, FAB-T and Square")]
    public AutoAttackVersion Version { get; set; } = AutoAttackVersion.Standard;

    /// <summary>
    /// Gets or sets how many of the highest-scoring wrong classes the targeted attacks try in turn.
    /// </summary>
    /// <value>Defaults to 9.</value>
    [PaperDefault(9, Source = "Croce and Hein 2020, Sec. 4: the targeted attacks use the top 9 target classes")]
    public int TargetClasses { get; set; } = 9;

    /// <summary>
    /// Gets or sets the query budget of the Square attack.
    /// </summary>
    /// <value>Defaults to 5000.</value>
    [PaperDefault(5000, Source = "Croce and Hein 2020, Sec. 4: Square Attack with 5000 queries")]
    public int SquareQueries { get; set; } = 5000;

    /// <summary>
    /// Gets or sets the initial fraction of the input that one Square attack window covers.
    /// </summary>
    /// <value>Defaults to 0.8.</value>
    [PaperDefault(0.8, Source = "Croce and Hein 2020, Sec. 4: Square Attack with p_init = 0.8")]
    public double SquareInitialFraction { get; set; } = 0.8;

    /// <summary>
    /// Gets or sets FAB's maximum step bias towards the original point.
    /// </summary>
    /// <value>Defaults to 0.1.</value>
    [PaperDefault(0.1, Source = "Croce and Hein 2020, Sec. 4 (FAB, Croce and Hein 2020b): alpha_max = 0.1")]
    public double FabAlphaMax { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets FAB's overshoot factor of the projection step.
    /// </summary>
    /// <value>Defaults to 1.05.</value>
    [PaperDefault(1.05, Source = "Croce and Hein 2020, Sec. 4 (FAB, Croce and Hein 2020b): eta = 1.05")]
    public double FabEta { get; set; } = 1.05;

    /// <summary>
    /// Gets or sets FAB's backward step factor after a successful step.
    /// </summary>
    /// <value>Defaults to 0.9.</value>
    [PaperDefault(0.9, Source = "Croce and Hein 2020, Sec. 4 (FAB, Croce and Hein 2020b): beta = 0.9")]
    public double FabBeta { get; set; } = 0.9;
}
