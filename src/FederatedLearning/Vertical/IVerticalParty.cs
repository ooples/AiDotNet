using AiDotNet.Tensors;

namespace AiDotNet.FederatedLearning.Vertical;

/// <summary>
/// Represents a party in vertical federated learning that holds a subset of features.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> In vertical FL, each party holds different features (columns)
/// for the same entities (rows). For example, a bank has income and credit score columns,
/// while a hospital has diagnosis and prescription columns. Each party runs a local "bottom model"
/// on its features to produce an embedding (a compressed representation).</para>
///
/// <para>The party interface abstracts away the local computation: the VFL trainer asks each party
/// to compute forward passes on its local data and backward passes using gradients received
/// from the coordinator.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface IVerticalParty<T>
{
    /// <summary>
    /// Gets the unique identifier for this party.
    /// </summary>
    string PartyId { get; }

    /// <summary>
    /// Gets the number of features held by this party.
    /// </summary>
    int FeatureCount { get; }

    /// <summary>
    /// Gets the output dimension of this party's bottom model (embedding size).
    /// </summary>
    int EmbeddingDimension { get; }

    /// <summary>
    /// Gets whether this party holds the labels for supervised learning.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> In VFL, one party typically holds the prediction target
    /// (e.g., "did the patient recover?"). This party is the "label holder" and plays a
    /// special role in training: it computes the loss and initiates backpropagation.</para>
    /// </remarks>
    bool IsLabelHolder { get; }

    /// <summary>
    /// Gets the entity IDs that this party has data for.
    /// </summary>
    /// <returns>A read-only list of entity identifiers.</returns>
    IReadOnlyList<string> GetEntityIds();

    /// <summary>
    /// Computes the forward pass of this party's bottom model on the given entity indices.
    /// </summary>
    /// <param name="alignedIndices">The local row indices of entities to process (from PSI alignment).</param>
    /// <returns>The embedding tensor with shape [batchSize, embeddingDimension].</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the local computation step. The party takes its features
    /// for the specified rows and passes them through its local neural network to produce
    /// a compact representation (embedding) that is sent to the coordinator.</para>
    /// </remarks>
    Tensor<T> ComputeForward(IReadOnlyList<int> alignedIndices);

    /// <summary>
    /// Applies the backward pass using gradients received from the coordinator.
    /// Updates the local bottom model parameters.
    /// </summary>
    /// <param name="gradients">Gradient tensor with respect to this party's embedding output.</param>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    void ApplyBackward(Tensor<T> gradients, double learningRate);

    /// <summary>
    /// Gets the current parameters of this party's bottom model for checkpointing.
    /// </summary>
    /// <returns>A list of parameter tensors.</returns>
    IReadOnlyList<Tensor<T>> GetParameters();

    /// <summary>
    /// Sets the parameters of this party's bottom model (for loading checkpoints or unlearning).
    /// </summary>
    /// <param name="parameters">The parameter tensors to load.</param>
    void SetParameters(IReadOnlyList<Tensor<T>> parameters);
}
