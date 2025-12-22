namespace AiDotNet.FederatedLearning.Privacy;

using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Interfaces;

/// <summary>
/// Base class for privacy mechanisms in federated learning.
/// </summary>
/// <typeparam name="TModel">Model/update representation.</typeparam>
/// <typeparam name="T">Numeric type.</typeparam>
public abstract class PrivacyMechanismBase<TModel, T> : FederatedLearningComponentBase<T>, IPrivacyMechanism<TModel>
{
    public abstract TModel ApplyPrivacy(TModel model, double epsilon, double delta);

    public abstract double GetPrivacyBudgetConsumed();

    public abstract string GetMechanismName();
}
