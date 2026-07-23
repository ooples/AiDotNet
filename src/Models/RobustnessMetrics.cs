namespace AiDotNet.Models;

/// <summary>
/// Contains metrics for evaluating adversarial robustness of models.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class RobustnessMetrics<T>
{
    /// <summary>
    /// Gets or sets the accuracy on clean (non-adversarial) examples.
    /// </summary>
    public double CleanAccuracy { get; set; }

    /// <summary>
    /// Gets or sets the accuracy on adversarial examples.
    /// </summary>
    public double AdversarialAccuracy { get; set; }

    /// <summary>
    /// Gets or sets the average perturbation size needed to fool the model.
    /// </summary>
    public double AveragePerturbationSize { get; set; }

    /// <summary>
    /// Gets or sets the attack success rate (percentage of successful attacks).
    /// </summary>
    public double AttackSuccessRate { get; set; }

    /// <summary>
    /// Gets or sets the robustness score (combines clean and adversarial accuracy).
    /// </summary>
    public double RobustnessScore { get; set; }

    /// <summary>
    /// Gets or sets additional evaluation metrics.
    /// </summary>
    public Dictionary<string, double> AdditionalMetrics { get; set; } = new();
}
