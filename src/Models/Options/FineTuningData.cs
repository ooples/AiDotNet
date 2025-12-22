namespace AiDotNet.Models.Options;

/// <summary>
/// Container for fine-tuning training and evaluation data.
/// </summary>
/// <remarks>
/// <para>
/// This class holds data for various fine-tuning methods. Different methods use
/// different subsets of the data:
/// </para>
/// <list type="bullet">
/// <item><term>SFT</term><description>Uses Inputs and Outputs</description></item>
/// <item><term>Preference methods</term><description>Uses Inputs, ChosenOutputs, RejectedOutputs</description></item>
/// <item><term>RL methods</term><description>Uses Inputs and Rewards</description></item>
/// <item><term>Ranking methods</term><description>Uses Inputs and RankedOutputs</description></item>
/// </list>
/// <para><b>For Beginners:</b> Think of this as a container that holds all the training
/// examples the model needs to learn from. Different training methods need different
/// kinds of examples.</para>
/// </remarks>
/// <typeparam name="T">The numeric data type.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
public class FineTuningData<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the input data samples.
    /// </summary>
    /// <remarks>
    /// Required for all fine-tuning methods.
    /// </remarks>
    public TInput[] Inputs { get; set; } = Array.Empty<TInput>();

    // ========== SFT Data ==========

    /// <summary>
    /// Gets or sets the target outputs for supervised fine-tuning.
    /// </summary>
    /// <remarks>
    /// Used by SFT methods. Each output corresponds to an input.
    /// </remarks>
    public TOutput[] Outputs { get; set; } = Array.Empty<TOutput>();

    // ========== Preference Data ==========

    /// <summary>
    /// Gets or sets the chosen (preferred) outputs for preference learning.
    /// </summary>
    /// <remarks>
    /// Used by DPO, IPO, KTO, SimPO, CPO, and other preference methods.
    /// </remarks>
    public TOutput[] ChosenOutputs { get; set; } = Array.Empty<TOutput>();

    /// <summary>
    /// Gets or sets the rejected outputs for preference learning.
    /// </summary>
    /// <remarks>
    /// Used by DPO, IPO, SimPO, CPO, and other pairwise preference methods.
    /// </remarks>
    public TOutput[] RejectedOutputs { get; set; } = Array.Empty<TOutput>();

    /// <summary>
    /// Gets or sets binary labels indicating if outputs are desirable.
    /// </summary>
    /// <remarks>
    /// Used by KTO which doesn't require pairwise data. True = desirable, False = undesirable.
    /// </remarks>
    public bool[] DesirabilityLabels { get; set; } = Array.Empty<bool>();

    // ========== Ranking Data ==========

    /// <summary>
    /// Gets or sets ranked outputs for ranking-based methods.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Used by RSO, RRHF, SLiC-HF, PRO.
    /// Each inner array contains outputs ranked from best to worst.
    /// </para>
    /// </remarks>
    public TOutput[][] RankedOutputs { get; set; } = Array.Empty<TOutput[]>();

    // ========== RL Data ==========

    /// <summary>
    /// Gets or sets reward values for RL-based methods.
    /// </summary>
    /// <remarks>
    /// Used by RLHF, PPO, GRPO, REINFORCE.
    /// </remarks>
    public double[] Rewards { get; set; } = Array.Empty<double>();

    /// <summary>
    /// Gets or sets advantages for PPO-style methods.
    /// </summary>
    public double[] Advantages { get; set; } = Array.Empty<double>();

    /// <summary>
    /// Gets or sets value estimates for critic-based methods.
    /// </summary>
    public double[] Values { get; set; } = Array.Empty<double>();

    // ========== Constitutional AI Data ==========

    /// <summary>
    /// Gets or sets critique-revision pairs for constitutional methods.
    /// </summary>
    /// <remarks>
    /// Each tuple contains (original response, critique, revised response).
    /// </remarks>
    public (TOutput Original, string Critique, TOutput Revised)[] CritiqueRevisions { get; set; }
        = Array.Empty<(TOutput, string, TOutput)>();

    // ========== Knowledge Distillation Data ==========

    /// <summary>
    /// Gets or sets teacher model logits/outputs for distillation.
    /// </summary>
    public TOutput[] TeacherOutputs { get; set; } = Array.Empty<TOutput>();

    /// <summary>
    /// Gets or sets teacher model confidence scores.
    /// </summary>
    public double[] TeacherConfidences { get; set; } = Array.Empty<double>();

    // ========== Metadata ==========

    /// <summary>
    /// Gets or sets optional sample weights for weighted training.
    /// </summary>
    public double[] SampleWeights { get; set; } = Array.Empty<double>();

    /// <summary>
    /// Gets or sets optional sample identifiers for tracking.
    /// </summary>
    public string[] SampleIds { get; set; } = Array.Empty<string>();

    /// <summary>
    /// Gets the number of samples in the dataset.
    /// </summary>
    public int Count => Inputs.Length;

    /// <summary>
    /// Gets whether this data is suitable for SFT.
    /// </summary>
    public bool HasSFTData => Inputs.Length > 0 && Outputs.Length == Inputs.Length;

    /// <summary>
    /// Gets whether this data is suitable for pairwise preference methods.
    /// </summary>
    public bool HasPairwisePreferenceData =>
        Inputs.Length > 0 &&
        ChosenOutputs.Length == Inputs.Length &&
        RejectedOutputs.Length == Inputs.Length;

    /// <summary>
    /// Gets whether this data is suitable for KTO (unpaired preferences).
    /// </summary>
    public bool HasUnpairedPreferenceData =>
        Inputs.Length > 0 &&
        (ChosenOutputs.Length > 0 || RejectedOutputs.Length > 0) &&
        DesirabilityLabels.Length == (ChosenOutputs.Length + RejectedOutputs.Length);

    /// <summary>
    /// Gets whether this data is suitable for ranking methods.
    /// </summary>
    public bool HasRankingData =>
        Inputs.Length > 0 &&
        RankedOutputs.Length == Inputs.Length;

    /// <summary>
    /// Gets whether this data is suitable for RL methods.
    /// </summary>
    public bool HasRLData =>
        Inputs.Length > 0 &&
        Rewards.Length == Inputs.Length;

    /// <summary>
    /// Gets whether this data is suitable for distillation.
    /// </summary>
    public bool HasDistillationData =>
        Inputs.Length > 0 &&
        TeacherOutputs.Length == Inputs.Length;

    /// <summary>
    /// Creates a subset of the data for the given indices.
    /// </summary>
    /// <param name="indices">The indices to include in the subset.</param>
    /// <returns>A new FineTuningData containing only the specified samples.</returns>
    public FineTuningData<T, TInput, TOutput> Subset(int[] indices)
    {
        return new FineTuningData<T, TInput, TOutput>
        {
            // Core data
            Inputs = indices.Select(i => Inputs[i]).ToArray(),
            Outputs = Outputs.Length > 0 ? indices.Select(i => Outputs[i]).ToArray() : Array.Empty<TOutput>(),

            // Preference data
            ChosenOutputs = ChosenOutputs.Length > 0 ? indices.Select(i => ChosenOutputs[i]).ToArray() : Array.Empty<TOutput>(),
            RejectedOutputs = RejectedOutputs.Length > 0 ? indices.Select(i => RejectedOutputs[i]).ToArray() : Array.Empty<TOutput>(),
            DesirabilityLabels = DesirabilityLabels.Length > 0 ? indices.Select(i => DesirabilityLabels[i]).ToArray() : Array.Empty<bool>(),

            // Ranking data
            RankedOutputs = RankedOutputs.Length > 0 ? indices.Select(i => RankedOutputs[i]).ToArray() : Array.Empty<TOutput[]>(),

            // RL data
            Rewards = Rewards.Length > 0 ? indices.Select(i => Rewards[i]).ToArray() : Array.Empty<double>(),
            Advantages = Advantages.Length > 0 ? indices.Select(i => Advantages[i]).ToArray() : Array.Empty<double>(),
            Values = Values.Length > 0 ? indices.Select(i => Values[i]).ToArray() : Array.Empty<double>(),

            // Constitutional AI data
            CritiqueRevisions = CritiqueRevisions.Length > 0 ? indices.Select(i => CritiqueRevisions[i]).ToArray() : Array.Empty<(TOutput, string, TOutput)>(),

            // Distillation data
            TeacherOutputs = TeacherOutputs.Length > 0 ? indices.Select(i => TeacherOutputs[i]).ToArray() : Array.Empty<TOutput>(),
            TeacherConfidences = TeacherConfidences.Length > 0 ? indices.Select(i => TeacherConfidences[i]).ToArray() : Array.Empty<double>(),

            // Metadata
            SampleWeights = SampleWeights.Length > 0 ? indices.Select(i => SampleWeights[i]).ToArray() : Array.Empty<double>(),
            SampleIds = SampleIds.Length > 0 ? indices.Select(i => SampleIds[i]).ToArray() : Array.Empty<string>()
        };
    }

    /// <summary>
    /// Splits the data into training and validation sets.
    /// </summary>
    /// <param name="validationRatio">The ratio of data to use for validation (0.0 to 1.0).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <returns>A tuple of (training data, validation data).</returns>
    public (FineTuningData<T, TInput, TOutput> Train, FineTuningData<T, TInput, TOutput> Validation) Split(
        double validationRatio = 0.1,
        int? seed = null)
    {
        var random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();

        var indices = Enumerable.Range(0, Count).ToArray();

        // Fisher-Yates shuffle
        for (int i = indices.Length - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }

        var validationCount = (int)(Count * validationRatio);
        var validationIndices = indices.Take(validationCount).ToArray();
        var trainIndices = indices.Skip(validationCount).ToArray();

        return (Subset(trainIndices), Subset(validationIndices));
    }
}
