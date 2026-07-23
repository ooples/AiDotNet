namespace AiDotNet.Configuration;

/// <summary>
/// Configuration for evaluation during training.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Evaluation runs your agent without exploration
/// to measure true performance. This gives you an unbiased estimate of how well
/// the agent would perform when deployed.
/// </remarks>
public class RLEvaluationConfig
{
    /// <summary>
    /// Evaluate every N episodes.
    /// </summary>
    public int EvaluateEveryEpisodes { get; set; } = 100;

    /// <summary>
    /// Number of episodes to run during each evaluation.
    /// </summary>
    public int EvaluationEpisodes { get; set; } = 10;

    /// <summary>
    /// Whether to use deterministic actions during evaluation.
    /// </summary>
    public bool Deterministic { get; set; } = true;
}
