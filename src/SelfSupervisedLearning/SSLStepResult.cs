namespace AiDotNet.SelfSupervisedLearning;

/// <summary>
/// Result of a single SSL training step.
/// </summary>
/// <typeparam name="T">The numeric type used for computations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This contains all the information from one training iteration,
/// including the loss value and any additional metrics specific to the SSL method.</para>
/// </remarks>
public class SSLStepResult<T>
{
    /// <summary>
    /// Gets or sets the primary loss value for this step.
    /// </summary>
    public T Loss { get; set; } = default!;

    /// <summary>
    /// Gets or sets additional metrics specific to the SSL method.
    /// </summary>
    /// <remarks>
    /// <para>Examples: accuracy of positive pair detection, embedding norm, collapse metrics.</para>
    /// </remarks>
    public Dictionary<string, T> Metrics { get; set; } = [];

    /// <summary>
    /// Gets or sets the number of positive pairs in this batch.
    /// </summary>
    public int NumPositivePairs { get; set; }

    /// <summary>
    /// Gets or sets the number of negative pairs in this batch.
    /// </summary>
    public int NumNegativePairs { get; set; }

    /// <summary>
    /// Gets or sets the current learning rate (if adaptive).
    /// </summary>
    public double? CurrentLearningRate { get; set; }

    /// <summary>
    /// Gets or sets the current temperature parameter (if applicable).
    /// </summary>
    public double? CurrentTemperature { get; set; }
}
