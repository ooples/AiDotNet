namespace AiDotNet.Models.Options;

/// <summary>
/// Metrics for evaluating fine-tuning quality.
/// </summary>
/// <remarks>
/// <para>
/// This class contains metrics relevant to various fine-tuning methods:
/// </para>
/// <list type="bullet">
/// <item><term>Loss metrics</term><description>Training and validation loss</description></item>
/// <item><term>Preference metrics</term><description>Win rate, preference accuracy</description></item>
/// <item><term>RL metrics</term><description>Reward scores, KL divergence</description></item>
/// <item><term>Safety metrics</term><description>Harmlessness, refusal rates</description></item>
/// </list>
/// <para><b>For Beginners:</b> These metrics tell you how well the fine-tuning worked.
/// Lower loss is generally better, higher win rates mean the model learned preferences well,
/// and safety metrics ensure the model behaves appropriately.</para>
/// </remarks>
/// <typeparam name="T">The numeric data type.</typeparam>
public class FineTuningMetrics<T>
{
    // ========== Core Training Metrics ==========

    /// <summary>
    /// Gets or sets the final training loss.
    /// </summary>
    public double TrainingLoss { get; set; }

    /// <summary>
    /// Gets or sets the validation loss.
    /// </summary>
    public double ValidationLoss { get; set; }

    /// <summary>
    /// Gets or sets the number of training steps completed.
    /// </summary>
    public int TrainingSteps { get; set; }

    /// <summary>
    /// Gets or sets the total training time in seconds.
    /// </summary>
    public double TrainingTimeSeconds { get; set; }

    /// <summary>
    /// Gets or sets the loss history over training.
    /// </summary>
    public List<double> LossHistory { get; set; } = new();

    // ========== Preference Metrics ==========

    /// <summary>
    /// Gets or sets the win rate against the reference model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Measures how often the fine-tuned model is preferred over the reference.
    /// Used by DPO, SimPO, and other preference methods.
    /// </para>
    /// <para>Values: 0.0 to 1.0 (higher is better)</para>
    /// </remarks>
    public double WinRate { get; set; }

    /// <summary>
    /// Gets or sets the preference accuracy on held-out data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// How accurately the model predicts which response is preferred.
    /// </para>
    /// </remarks>
    public double PreferenceAccuracy { get; set; }

    /// <summary>
    /// Gets or sets the chosen response log probability.
    /// </summary>
    public double ChosenLogProb { get; set; }

    /// <summary>
    /// Gets or sets the rejected response log probability.
    /// </summary>
    public double RejectedLogProb { get; set; }

    /// <summary>
    /// Gets or sets the margin between chosen and rejected log probabilities.
    /// </summary>
    public double LogProbMargin { get; set; }

    // ========== RL Metrics ==========

    /// <summary>
    /// Gets or sets the average reward achieved.
    /// </summary>
    public double AverageReward { get; set; }

    /// <summary>
    /// Gets or sets the reward standard deviation.
    /// </summary>
    public double RewardStd { get; set; }

    /// <summary>
    /// Gets or sets the KL divergence from the reference model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Measures how much the fine-tuned model has diverged from the reference.
    /// High KL divergence may indicate over-optimization.
    /// </para>
    /// </remarks>
    public double KLDivergence { get; set; }

    /// <summary>
    /// Gets or sets the entropy of the policy.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Higher entropy indicates more exploration/diversity in outputs.
    /// </para>
    /// </remarks>
    public double PolicyEntropy { get; set; }

    /// <summary>
    /// Gets or sets the value function loss (for actor-critic methods).
    /// </summary>
    public double ValueLoss { get; set; }

    /// <summary>
    /// Gets or sets the policy loss.
    /// </summary>
    public double PolicyLoss { get; set; }

    // ========== GRPO-Specific Metrics ==========

    /// <summary>
    /// Gets or sets the average group advantage.
    /// </summary>
    public double GroupAdvantage { get; set; }

    /// <summary>
    /// Gets or sets the within-group reward variance.
    /// </summary>
    public double GroupRewardVariance { get; set; }

    // ========== Safety/Alignment Metrics ==========

    /// <summary>
    /// Gets or sets the harmlessness score.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Measures how often the model avoids harmful outputs.
    /// </para>
    /// <para>Values: 0.0 to 1.0 (higher is better)</para>
    /// </remarks>
    public double HarmlessnessScore { get; set; }

    /// <summary>
    /// Gets or sets the helpfulness score.
    /// </summary>
    public double HelpfulnessScore { get; set; }

    /// <summary>
    /// Gets or sets the honesty score.
    /// </summary>
    public double HonestyScore { get; set; }

    /// <summary>
    /// Gets or sets the refusal rate for harmful prompts.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Measures how often the model appropriately refuses harmful requests.
    /// </para>
    /// </remarks>
    public double RefusalRate { get; set; }

    /// <summary>
    /// Gets or sets the false refusal rate.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Measures how often the model incorrectly refuses benign requests.
    /// Lower is better - too high indicates over-refusal.
    /// </para>
    /// </remarks>
    public double FalseRefusalRate { get; set; }

    // ========== Quality Metrics ==========

    /// <summary>
    /// Gets or sets the perplexity on validation data.
    /// </summary>
    public double Perplexity { get; set; }

    /// <summary>
    /// Gets or sets the BLEU score for generation quality.
    /// </summary>
    public double BleuScore { get; set; }

    /// <summary>
    /// Gets or sets the ROUGE-L score.
    /// </summary>
    public double RougeLScore { get; set; }

    /// <summary>
    /// Gets or sets the average output length.
    /// </summary>
    public double AverageOutputLength { get; set; }

    // ========== Knowledge Distillation Metrics ==========

    /// <summary>
    /// Gets or sets the distillation loss.
    /// </summary>
    public double DistillationLoss { get; set; }

    /// <summary>
    /// Gets or sets the agreement rate with teacher model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Measures how often the student agrees with the teacher's predictions.
    /// </para>
    /// </remarks>
    public double TeacherAgreementRate { get; set; }

    // ========== Computational Metrics ==========

    /// <summary>
    /// Gets or sets the peak memory usage in GB.
    /// </summary>
    public double PeakMemoryGB { get; set; }

    /// <summary>
    /// Gets or sets the throughput in samples per second.
    /// </summary>
    public double ThroughputSamplesPerSecond { get; set; }

    /// <summary>
    /// Gets or sets the number of trainable parameters.
    /// </summary>
    public long TrainableParameters { get; set; }

    // ========== Metadata ==========

    /// <summary>
    /// Gets or sets the fine-tuning method used.
    /// </summary>
    public string MethodName { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets when the training started.
    /// </summary>
    public DateTime TrainingStartTime { get; set; }

    /// <summary>
    /// Gets or sets when the training completed.
    /// </summary>
    public DateTime TrainingEndTime { get; set; }

    /// <summary>
    /// Gets or sets additional custom metrics.
    /// </summary>
    public Dictionary<string, double> CustomMetrics { get; set; } = new();

    /// <summary>
    /// Gets a summary of the metrics suitable for logging.
    /// </summary>
    /// <returns>A formatted string with key metrics.</returns>
    public string GetSummary()
    {
        var lines = new List<string>
        {
            $"Fine-Tuning Results ({MethodName})",
            $"=====================================",
            $"Training Loss: {TrainingLoss:F4}",
            $"Validation Loss: {ValidationLoss:F4}",
            $"Training Steps: {TrainingSteps}",
            $"Training Time: {TrainingTimeSeconds:F1}s"
        };

        if (WinRate > 0)
        {
            lines.Add($"Win Rate: {WinRate:P1}");
        }

        if (PreferenceAccuracy > 0)
        {
            lines.Add($"Preference Accuracy: {PreferenceAccuracy:P1}");
        }

        if (AverageReward != 0)
        {
            lines.Add($"Average Reward: {AverageReward:F3}");
            lines.Add($"KL Divergence: {KLDivergence:F4}");
        }

        if (HarmlessnessScore > 0 || HelpfulnessScore > 0)
        {
            lines.Add($"Harmlessness: {HarmlessnessScore:P1}");
            lines.Add($"Helpfulness: {HelpfulnessScore:P1}");
            lines.Add($"Honesty: {HonestyScore:P1}");
        }

        return string.Join(Environment.NewLine, lines);
    }
}
