namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the NeuralStressTest model.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Stress testing simulates extreme market scenarios.
/// These options control how large the model is and how many scenarios it
/// tries to generate or evaluate.
/// </para>
/// </remarks>
public class NeuralStressTestOptions<T> : RiskModelOptions<T>
{
    /// <summary>
    /// Hidden dimension size.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This sets how many neurons are in the hidden layers.
    /// Larger values can model more complex stress patterns.
    /// </para>
    /// </remarks>
    public int HiddenDimension { get; set; } = 128;

    /// <summary>
    /// Number of scenarios to generate/evaluate.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is how many "what if" situations the model
    /// considers (e.g., market crash, rate spike, sector downturn).
    /// </para>
    /// </remarks>
    public int NumScenarios { get; set; } = 10;

    /// <summary>
    /// Dropout rate.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Dropout helps prevent overfitting by randomly
    /// turning off neurons during training.
    /// </para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.1;
}
