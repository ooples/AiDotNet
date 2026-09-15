using AiDotNet.Evolution;

namespace AiDotNet.Configuration;

/// <summary>Immutable, prospectively declared screening, replication and confirmation policy.</summary>
/// <remarks>Support and thresholds are in evaluator quality units. Maximum cost includes correctness plus fitness.
/// Choose sample counts before seeing results. The defaults are a bounded example, not universal performance advice.</remarks>
public sealed class ProgramNoiseEvaluationOptions
{
    /// <summary>Creates a policy; construction of a session validates all ranges before any work.</summary>
    public ProgramNoiseEvaluationOptions(int screenSamples = 2, int searchSamples = 4, int confirmationSamples = 128,
        int auditCandidates = 8, int maximumChallenges = 16, double minimumQuality = 0, double maximumQuality = 1,
        double screenThreshold = .5, double usefulThreshold = .5, double minimumImprovement = .01,
        decimal maximumCostPerSample = 2, double confidence = .95,
        EvolutionOptimizationDirection direction = EvolutionOptimizationDirection.Maximize)
    {
        ScreenPlan = new EvolutionReplicationPlan(screenSamples, screenSamples, minimumQuality, maximumQuality,
            maximumCostPerSample, direction: direction);
        // Validate other replication counts even if screening rejects every candidate.
        _ = new EvolutionReplicationPlan(searchSamples, searchSamples, minimumQuality, maximumQuality, maximumCostPerSample, direction: direction);
        _ = new EvolutionReplicationPlan(confirmationSamples, confirmationSamples, minimumQuality, maximumQuality, maximumCostPerSample, direction: direction);
        if (double.IsNaN(screenThreshold) || screenThreshold < minimumQuality || screenThreshold > maximumQuality)
            throw new ArgumentOutOfRangeException(nameof(screenThreshold));
        ScreenThreshold = screenThreshold; SearchSamples = searchSamples; ConfirmationSamples = confirmationSamples;
        AuditCandidates = auditCandidates; MaximumChallenges = maximumChallenges; UsefulThreshold = usefulThreshold;
        MinimumImprovement = minimumImprovement; Confidence = confidence;
    }
    /// <summary>Gets the immutable cheap-screen sampling plan and common declared bounds/cost units.</summary>
    public EvolutionReplicationPlan ScreenPlan { get; }
    /// <summary>Gets the number of full-fidelity search samples for each challenger and incumbent.</summary>
    public int SearchSamples { get; }
    /// <summary>Gets the number of fresh hidden-confirmation and audit samples per program.</summary>
    public int ConfirmationSamples { get; }
    /// <summary>Gets the predeclared audit sample size.</summary>
    public int AuditCandidates { get; }
    /// <summary>Gets the finite comparison family size.</summary>
    public int MaximumChallenges { get; }
    /// <summary>Gets the cheap-screen mean threshold; this heuristic can falsely reject useful programs.</summary>
    public double ScreenThreshold { get; }
    /// <summary>Gets the full-fidelity true-mean usefulness threshold.</summary>
    public double UsefulThreshold { get; }
    /// <summary>Gets the minimum independently confirmed improvement required.</summary>
    public double MinimumImprovement { get; }
    /// <summary>Gets the nominal confidence subject to fresh independent measurement and frozen policy assumptions.</summary>
    public double Confidence { get; }
}
