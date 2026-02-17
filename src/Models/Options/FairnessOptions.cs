namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for fairness constraints in federated learning.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options control how the federated learning system ensures fair
/// treatment of different client groups. For example, if your federation includes hospitals from both
/// wealthy and underserved areas, fairness constraints prevent the model from favoring wealthier
/// hospitals at the expense of underserved ones.</para>
/// </remarks>
public class FederatedFairnessOptions
{
    /// <summary>
    /// Gets or sets whether fairness constraints are enabled. Default is false.
    /// </summary>
    public bool Enabled { get; set; } = false;

    /// <summary>
    /// Gets or sets the type of fairness constraint to enforce. Default is None.
    /// </summary>
    public FairnessConstraintType ConstraintType { get; set; } = FairnessConstraintType.None;

    /// <summary>
    /// Gets or sets the maximum allowed fairness violation before corrective action.
    /// Range [0, 1]: 0 = strict equality, higher values allow more disparity.
    /// Default is 0.05 (5% tolerance).
    /// </summary>
    public double FairnessThreshold { get; set; } = 0.05;

    /// <summary>
    /// Gets or sets the weight of the fairness penalty in the aggregation objective.
    /// Higher values enforce fairness more strongly at the cost of overall accuracy.
    /// Default is 0.1.
    /// </summary>
    public double FairnessLambda { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the number of client groups for group-based fairness constraints.
    /// Clients are assigned to groups based on their data characteristics.
    /// Default is 2 (binary grouping).
    /// </summary>
    public int NumberOfGroups { get; set; } = 2;

    /// <summary>
    /// Gets or sets how often to evaluate fairness metrics (in rounds).
    /// Default is every round (1).
    /// </summary>
    public int EvaluationFrequency { get; set; } = 1;

    /// <summary>
    /// Gets or sets the boost factor for underperforming groups in minimax fairness.
    /// The worst-performing group gets this multiplier applied to its aggregation weight.
    /// Default is 1.5.
    /// </summary>
    public double MinimaxBoostFactor { get; set; } = 1.5;
}
