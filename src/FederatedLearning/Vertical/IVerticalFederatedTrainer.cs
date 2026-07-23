using AiDotNet.Models.Options;
using AiDotNet.Tensors;

namespace AiDotNet.FederatedLearning.Vertical;

/// <summary>
/// Orchestrates the vertical federated learning training process.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This is the main coordinator for vertical FL training.
/// It handles the entire pipeline:</para>
/// <list type="number">
/// <item><description>Entity alignment via PSI (finding shared entities across parties)</description></item>
/// <item><description>Split model training (coordinating forward/backward passes across parties)</description></item>
/// <item><description>Secure gradient exchange (protecting intermediate values)</description></item>
/// <item><description>Missing feature handling (dealing with entities not in all parties)</description></item>
/// </list>
///
/// <para>Unlike horizontal FL trainers (which average model parameters), the VFL trainer
/// coordinates activations flowing through a split neural network. No party ever sees
/// another party's raw features.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface IVerticalFederatedTrainer<T>
{
    /// <summary>
    /// Registers a party to participate in VFL training.
    /// </summary>
    /// <param name="party">The party to register.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Before training can start, all parties must register with
    /// the trainer. At least one party must be a label holder.</para>
    /// </remarks>
    void RegisterParty(IVerticalParty<T> party);

    /// <summary>
    /// Performs entity alignment across all registered parties using PSI.
    /// Must be called after all parties are registered and before training begins.
    /// </summary>
    /// <param name="psiOptions">Options for the PSI protocol used for alignment.</param>
    /// <returns>A <see cref="VflAlignmentSummary"/> containing alignment statistics.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This step finds which entities (patients, customers) are shared
    /// across all parties. Only shared entities can be used for joint training.</para>
    /// </remarks>
    VflAlignmentSummary AlignEntities(PsiOptions? psiOptions = null);

    /// <summary>
    /// Runs a single training epoch over the aligned data.
    /// </summary>
    /// <returns>A <see cref="VflEpochResult{T}"/> containing epoch metrics.</returns>
    VflEpochResult<T> TrainEpoch();

    /// <summary>
    /// Runs the full training loop for the configured number of epochs.
    /// </summary>
    /// <returns>A <see cref="VflTrainingResult{T}"/> containing training history and final metrics.</returns>
    VflTrainingResult<T> Train();

    /// <summary>
    /// Makes predictions on aligned entity data using the split model.
    /// </summary>
    /// <param name="entityIndices">The entity indices to predict (from alignment).</param>
    /// <returns>Prediction tensor from the top model.</returns>
    Tensor<T> Predict(IReadOnlyList<int> entityIndices);

    /// <summary>
    /// Removes the influence of specified entities from the trained model (unlearning).
    /// </summary>
    /// <param name="entityIds">The entity IDs to unlearn.</param>
    /// <returns>The number of entities successfully unlearned.</returns>
    int UnlearnEntities(IReadOnlyList<string> entityIds);
}
