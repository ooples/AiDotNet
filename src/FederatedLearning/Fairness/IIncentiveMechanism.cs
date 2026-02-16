namespace AiDotNet.FederatedLearning.Fairness;

/// <summary>
/// Computes incentive rewards for federated learning participants based on their contributions.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Clients need motivation to participate in federated learning.
/// Without incentives, rational clients may free-ride (benefit from the global model without
/// contributing their own data). An incentive mechanism assigns rewards proportional to each
/// client's contribution, encouraging high-quality participation.</para>
///
/// <para><b>Example:</b> In a data marketplace, a hospital contributing rare disease data
/// (high marginal value) earns more than one contributing common cold data (low marginal value).</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for model parameters.</typeparam>
public interface IIncentiveMechanism<T>
{
    /// <summary>
    /// Computes incentive rewards for each client based on their contribution scores.
    /// </summary>
    /// <param name="contributionScores">Client contribution scores (from IClientContributionEvaluator).</param>
    /// <param name="totalBudget">Total reward budget to distribute. Default is 1.0 (normalized shares).</param>
    /// <returns>Dictionary mapping client IDs to their reward amount.</returns>
    Dictionary<int, double> ComputeRewards(
        Dictionary<int, double> contributionScores,
        double totalBudget = 1.0);

    /// <summary>
    /// Computes trust scores for clients based on their participation history.
    /// </summary>
    /// <param name="contributionHistory">Historical contribution scores per client. Key: clientId, Value: list of scores per round.</param>
    /// <returns>Trust scores in [0, 1] for each client. Higher = more trustworthy.</returns>
    Dictionary<int, double> ComputeTrustScores(
        Dictionary<int, List<double>> contributionHistory);

    /// <summary>
    /// Gets the name of this incentive mechanism.
    /// </summary>
    string MechanismName { get; }
}
