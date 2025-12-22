using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Interfaces;

namespace AiDotNet.FederatedLearning.Privacy.Accounting;

/// <summary>
/// Base class for privacy accountants.
/// </summary>
public abstract class PrivacyAccountantBase : FederatedLearningComponentBase<double>, IPrivacyAccountant
{
    public abstract void AddRound(double epsilon, double delta, double samplingRate);

    public abstract double GetTotalEpsilonConsumed();

    public abstract double GetTotalDeltaConsumed();

    public abstract double GetEpsilonAtDelta(double targetDelta);

    public abstract string GetAccountantName();
}

