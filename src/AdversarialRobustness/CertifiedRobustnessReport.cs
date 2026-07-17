namespace AiDotNet.AdversarialRobustness;

/// <summary>
/// The certified-robustness audit produced by a configured certified defense, plus — when an empirical
/// attack is also configured — the certified-versus-empirical robustness sandwich.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Certified accuracy is a <b>guaranteed lower bound</b> on robustness (proven safe within a radius);
/// empirical robust accuracy from an attack is an <b>upper bound</b> (only "no attack found yet"). The
/// true robustness lies between them. Reporting both at the same perturbation budget — and the gap
/// between — is beyond tools that give one bound or the other.
/// </para>
/// </remarks>
public sealed class CertifiedRobustnessReport<T>
{
    /// <summary>The configured defense's type name.</summary>
    public string DefenseName { get; init; } = string.Empty;

    /// <summary>The perturbation radius the certification was evaluated at.</summary>
    public double RadiusUsed { get; init; }

    /// <summary>The certified accuracy metrics (certified accuracy, certification rate, mean/median radius).</summary>
    public required Models.CertifiedAccuracyMetrics<T> Metrics { get; init; }

    /// <summary>
    /// The empirical robust accuracy at the same radius (from the configured attack), or <c>null</c> when
    /// no attack was configured — the upper bound of the sandwich.
    /// </summary>
    public double? EmpiricalRobustAccuracy { get; init; }

    /// <summary>
    /// The proven-vs-unproven gap: empirical robust accuracy minus certified accuracy, when both are
    /// available. Large means much of the apparent robustness is not yet proven; near zero means the
    /// certificate is tight.
    /// </summary>
    public double? CertifiedEmpiricalGap { get; init; }

    /// <summary>Whether the certified-vs-empirical sandwich was computed (both bounds available).</summary>
    public bool SandwichAvailable { get; init; }
}
