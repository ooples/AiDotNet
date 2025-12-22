namespace AiDotNet.FederatedLearning.Privacy.Accounting;

/// <summary>
/// Privacy accountant using basic (naive) composition.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Basic composition simply adds up privacy spend across rounds:
/// - epsilon_total = sum(epsilon_round)
/// - delta_total = sum(delta_round)
/// This is simple but can be pessimistic compared to tighter accountants.
/// </remarks>
public sealed class BasicCompositionPrivacyAccountant : PrivacyAccountantBase
{
    private double _epsilonTotal;
    private double _deltaTotal;

    public override void AddRound(double epsilon, double delta, double samplingRate)
    {
        if (epsilon <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(epsilon), "Epsilon must be positive.");
        }

        if (delta <= 0.0 || delta >= 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(delta), "Delta must be in (0, 1).");
        }

        if (samplingRate <= 0.0 || samplingRate > 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(samplingRate), "Sampling rate must be in (0, 1].");
        }

        _epsilonTotal += epsilon;
        _deltaTotal += delta;
    }

    public override double GetTotalEpsilonConsumed() => _epsilonTotal;

    public override double GetTotalDeltaConsumed() => _deltaTotal;

    public override double GetEpsilonAtDelta(double targetDelta)
    {
        if (targetDelta <= 0.0 || targetDelta >= 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(targetDelta), "Target delta must be in (0, 1).");
        }

        return _epsilonTotal;
    }

    public override string GetAccountantName() => "Basic";
}

