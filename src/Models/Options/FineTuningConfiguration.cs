using AiDotNet.Interfaces;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration for fine-tuning during model building.
/// </summary>
/// <remarks>
/// <para>
/// This configuration controls whether fine-tuning is enabled and which method/implementation is used.
/// When enabled and no custom implementation is provided, a default fine-tuning method is created
/// based on the configured options.
/// </para>
/// <para><b>For Beginners:</b> This is the fine-tuning "on/off switch" and settings bundle.
/// You can leave it alone to skip fine-tuning, or configure it to apply preference learning,
/// RLHF, or other alignment techniques after initial training.</para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type for the model.</typeparam>
/// <typeparam name="TOutput">The output data type for the model.</typeparam>
public class FineTuningConfiguration<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets whether fine-tuning is enabled.
    /// </summary>
    /// <remarks>
    /// When false, fine-tuning is skipped during model building.
    /// </remarks>
    public bool Enabled { get; set; } = false;

    /// <summary>
    /// Gets or sets the fine-tuning options.
    /// </summary>
    public FineTuningOptions<T> Options { get; set; } = new();

    /// <summary>
    /// Gets or sets an optional custom fine-tuning implementation.
    /// </summary>
    /// <remarks>
    /// When provided, this implementation is used instead of the default.
    /// </remarks>
    public IFineTuning<T, TInput, TOutput>? Implementation { get; set; }

    /// <summary>
    /// Gets or sets the training data for fine-tuning.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This data is used during the fine-tuning phase in BuildAsync.
    /// Different fine-tuning methods require different data:
    /// </para>
    /// <list type="bullet">
    /// <item><term>SFT</term><description>Requires Inputs and Outputs</description></item>
    /// <item><term>DPO/SimPO</term><description>Requires Inputs, ChosenOutputs, RejectedOutputs</description></item>
    /// <item><term>RLHF/GRPO</term><description>Requires Inputs and Rewards (or a reward function)</description></item>
    /// </list>
    /// </remarks>
    public FineTuningData<T, TInput, TOutput>? TrainingData { get; set; }

    /// <summary>
    /// Gets or sets optional validation data for fine-tuning.
    /// </summary>
    /// <remarks>
    /// If not provided and validation is needed, a portion of TrainingData will be split off.
    /// </remarks>
    public FineTuningData<T, TInput, TOutput>? ValidationData { get; set; }

    /// <summary>
    /// Gets or sets the ratio of data to use for validation if ValidationData is not provided.
    /// </summary>
    public double ValidationSplitRatio { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets whether to auto-split training data for validation.
    /// </summary>
    public bool AutoSplitForValidation { get; set; } = true;

    /// <summary>
    /// Creates a minimal configuration for SFT with default options.
    /// </summary>
    /// <param name="trainingData">The supervised training data.</param>
    /// <returns>A configured FineTuningConfiguration for SFT.</returns>
    public static FineTuningConfiguration<T, TInput, TOutput> ForSFT(
        FineTuningData<T, TInput, TOutput> trainingData)
    {
        return new FineTuningConfiguration<T, TInput, TOutput>
        {
            Enabled = true,
            Options = new FineTuningOptions<T> { MethodType = FineTuningMethodType.SFT },
            TrainingData = trainingData
        };
    }

    /// <summary>
    /// Creates a minimal configuration for DPO with default options.
    /// </summary>
    /// <param name="trainingData">The preference training data.</param>
    /// <returns>A configured FineTuningConfiguration for DPO.</returns>
    public static FineTuningConfiguration<T, TInput, TOutput> ForDPO(
        FineTuningData<T, TInput, TOutput> trainingData)
    {
        return new FineTuningConfiguration<T, TInput, TOutput>
        {
            Enabled = true,
            Options = new FineTuningOptions<T> { MethodType = FineTuningMethodType.DPO },
            TrainingData = trainingData
        };
    }

    /// <summary>
    /// Creates a minimal configuration for SimPO with default options.
    /// </summary>
    /// <param name="trainingData">The preference training data.</param>
    /// <returns>A configured FineTuningConfiguration for SimPO.</returns>
    public static FineTuningConfiguration<T, TInput, TOutput> ForSimPO(
        FineTuningData<T, TInput, TOutput> trainingData)
    {
        return new FineTuningConfiguration<T, TInput, TOutput>
        {
            Enabled = true,
            Options = new FineTuningOptions<T> { MethodType = FineTuningMethodType.SimPO },
            TrainingData = trainingData
        };
    }

    /// <summary>
    /// Creates a minimal configuration for GRPO with default options.
    /// </summary>
    /// <param name="trainingData">The RL training data with rewards.</param>
    /// <returns>A configured FineTuningConfiguration for GRPO.</returns>
    public static FineTuningConfiguration<T, TInput, TOutput> ForGRPO(
        FineTuningData<T, TInput, TOutput> trainingData)
    {
        return new FineTuningConfiguration<T, TInput, TOutput>
        {
            Enabled = true,
            Options = new FineTuningOptions<T> { MethodType = FineTuningMethodType.GRPO },
            TrainingData = trainingData
        };
    }

    /// <summary>
    /// Creates a minimal configuration for ORPO with default options.
    /// </summary>
    /// <param name="trainingData">The combined SFT + preference data.</param>
    /// <returns>A configured FineTuningConfiguration for ORPO.</returns>
    public static FineTuningConfiguration<T, TInput, TOutput> ForORPO(
        FineTuningData<T, TInput, TOutput> trainingData)
    {
        return new FineTuningConfiguration<T, TInput, TOutput>
        {
            Enabled = true,
            Options = new FineTuningOptions<T> { MethodType = FineTuningMethodType.ORPO },
            TrainingData = trainingData
        };
    }
}
