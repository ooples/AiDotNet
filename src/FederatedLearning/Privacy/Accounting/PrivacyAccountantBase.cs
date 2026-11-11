using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Interfaces;

namespace AiDotNet.FederatedLearning.Privacy.Accounting;

/// <summary>
/// Base class for privacy accountants.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> for provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public abstract class PrivacyAccountantBase : FederatedLearningComponentBase<double>, IPrivacyAccountant
{
    /// <inheritdoc />
    public abstract void AddRound(double epsilon, double delta, double samplingRate);

    /// <inheritdoc />
    public abstract double GetTotalEpsilonConsumed();

    /// <inheritdoc />
    public abstract double GetTotalDeltaConsumed();

    /// <inheritdoc />
    public abstract double GetEpsilonAtDelta(double targetDelta);

    /// <inheritdoc />
    public abstract string GetAccountantName();
}

