namespace AiDotNet.AdversarialRobustness;

/// <summary>
/// One point on the robustness curve: the model's accuracy when attacked at a given perturbation budget.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public sealed class RobustnessCurvePoint<T>
{
    /// <summary>The perturbation budget (epsilon) at which the model was attacked.</summary>
    public double Epsilon { get; init; }

    /// <summary>The model's accuracy on the adversarial examples generated at this budget.</summary>
    public double RobustAccuracy { get; init; }
}

/// <summary>
/// The empirical adversarial-robustness audit produced by running a configured attack against the trained
/// model. Reports a robustness curve across perturbation budgets rather than a single number, plus a
/// robustness margin — more informative than the single robust-accuracy figure most tools report.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public sealed class AdversarialRobustnessReport<T>
{
    /// <summary>The configured attack's type name.</summary>
    public string AttackName { get; init; } = string.Empty;

    /// <summary>The model's accuracy on clean (un-attacked) inputs.</summary>
    public double CleanAccuracy { get; init; }

    /// <summary>The model's accuracy under attack at the configured perturbation budget.</summary>
    public double RobustAccuracy { get; init; }

    /// <summary>Clean accuracy minus robust accuracy at the configured budget — how much the attack costs.</summary>
    public double CleanVsRobustGap { get; init; }

    /// <summary>Fraction of initially-correct samples the attack flips to incorrect at the configured budget.</summary>
    public double AttackSuccessRate { get; init; }

    /// <summary>
    /// Mean magnitude of the perturbation the attack needed to fool the model (robustness margin) — larger
    /// means a more robust model.
    /// </summary>
    public double MeanPerturbationToFool { get; init; }

    /// <summary>
    /// Accuracy versus perturbation budget, in increasing-epsilon order. A single point when the sweep was
    /// unavailable (<see cref="SweepAvailable"/> is false).
    /// </summary>
    public IReadOnlyList<RobustnessCurvePoint<T>> RobustnessCurve { get; init; } = System.Array.Empty<RobustnessCurvePoint<T>>();

    /// <summary>Area under the robustness curve (normalized), a single summary of robustness across budgets.</summary>
    public double RobustAccuracyAuc { get; init; }

    /// <summary>Whether the epsilon sweep ran (the attack honored budget changes) or fell back to a single point.</summary>
    public bool SweepAvailable { get; init; }
}
