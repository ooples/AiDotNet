namespace AiDotNet.Models.Options;

/// <summary>
/// Specifies the type of fairness constraint to enforce during federated learning.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Fairness constraints ensure the global model treats all client groups
/// equitably. Without constraints, the model may perform well for majority clients but poorly
/// for underrepresented groups (e.g., a rural hospital in a federation dominated by urban ones).</para>
/// </remarks>
public enum FairnessConstraintType
{
    /// <summary>No fairness constraint applied.</summary>
    None,

    /// <summary>
    /// Demographic parity: model predictions should be independent of group membership.
    /// P(Y_hat=1 | Group=A) ≈ P(Y_hat=1 | Group=B).
    /// </summary>
    DemographicParity,

    /// <summary>
    /// Equalized odds: true positive and false positive rates should be equal across groups.
    /// P(Y_hat=1 | Y=y, Group=A) ≈ P(Y_hat=1 | Y=y, Group=B) for y in {0, 1}.
    /// </summary>
    EqualizedOdds,

    /// <summary>
    /// Equal opportunity: true positive rate should be equal across groups.
    /// P(Y_hat=1 | Y=1, Group=A) ≈ P(Y_hat=1 | Y=1, Group=B).
    /// </summary>
    EqualOpportunity,

    /// <summary>
    /// Minimax fairness: minimize the worst-case performance across all client groups.
    /// </summary>
    MinimaxFairness
}
