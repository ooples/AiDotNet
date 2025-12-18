namespace AiDotNet.KnowledgeDistillation.Teachers;

/// <summary>
/// Specifies how multiple teacher outputs are combined into a single supervision signal.
/// </summary>
/// <remarks>
/// <para>
/// When knowledge distillation uses more than one teacher model, their predictions must be aggregated
/// before being used to train the student. This enum selects the aggregation strategy.
/// </para>
/// <para><b>For Beginners:</b> If you have multiple “experts” (teachers), this chooses how to combine their answers.
/// You can average their confidence scores, or you can let them vote on the final answer.
/// </para>
/// </remarks>
public enum AggregationMode
{
    /// <summary>
    /// Averages the teachers' predictions.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Treat each teacher as giving a probability/confidence, then take the average.</para>
    /// </remarks>
    Average,

    /// <summary>
    /// Uses majority voting to select the final teacher prediction.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each teacher picks an answer (often by argmax), and the most common answer wins.</para>
    /// </remarks>
    Voting
}
