using AiDotNet.Models.Options;
using AiDotNet.Tensors;

namespace AiDotNet.FederatedLearning.Vertical;

/// <summary>
/// Represents a split neural network for vertical federated learning.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> A split neural network is divided into two parts:</para>
/// <list type="bullet">
/// <item><description><b>Bottom models:</b> One per party, processes that party's local features
/// to produce an embedding. Each party runs its bottom model locally.</description></item>
/// <item><description><b>Top model:</b> Runs at the coordinator, takes the combined embeddings
/// from all parties and produces the final prediction.</description></item>
/// </list>
///
/// <para>The "split point" is where the network is divided. Below the split, computation is
/// local and private. Above the split, computation uses combined information from all parties.</para>
///
/// <para>During training, forward passes flow upward (bottom -> top) and gradients flow
/// downward (top -> bottom). Only the embeddings and their gradients cross party boundaries,
/// never the raw features.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface ISplitModel<T>
{
    /// <summary>
    /// Gets the number of parties this split model is configured for.
    /// </summary>
    int NumberOfParties { get; }

    /// <summary>
    /// Gets the aggregation mode used to combine party embeddings.
    /// </summary>
    VflAggregationMode AggregationMode { get; }

    /// <summary>
    /// Computes the top model forward pass on the combined embeddings.
    /// </summary>
    /// <param name="combinedEmbeddings">The aggregated embeddings from all parties.</param>
    /// <returns>The top model output (predictions).</returns>
    Tensor<T> ForwardTopModel(Tensor<T> combinedEmbeddings);

    /// <summary>
    /// Computes the top model backward pass and returns gradients for each party's embedding.
    /// </summary>
    /// <param name="lossGradient">The gradient of the loss with respect to the top model output.</param>
    /// <param name="partyEmbeddings">The embeddings from each party (needed for gradient computation).</param>
    /// <returns>Gradients for each party's embedding, indexed by party order.</returns>
    IReadOnlyList<Tensor<T>> BackwardTopModel(Tensor<T> lossGradient, IReadOnlyList<Tensor<T>> partyEmbeddings);

    /// <summary>
    /// Aggregates embeddings from multiple parties according to the configured aggregation mode.
    /// </summary>
    /// <param name="partyEmbeddings">Embeddings from each party.</param>
    /// <returns>The aggregated embedding tensor.</returns>
    Tensor<T> AggregateEmbeddings(IReadOnlyList<Tensor<T>> partyEmbeddings);

    /// <summary>
    /// Updates the top model parameters using the computed gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    void UpdateTopModelParameters(double learningRate);

    /// <summary>
    /// Gets the current parameters of the top model for checkpointing.
    /// </summary>
    IReadOnlyList<Tensor<T>> GetTopModelParameters();

    /// <summary>
    /// Sets the top model parameters (for loading checkpoints or unlearning).
    /// </summary>
    void SetTopModelParameters(IReadOnlyList<Tensor<T>> parameters);
}
