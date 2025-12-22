namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the functionality for a client-side model in federated learning.
/// </summary>
/// <remarks>
/// This interface represents a model that exists on a client device or node in a federated
/// learning system. Each client maintains its own copy of the global model and trains it
/// on local data.
///
/// <b>For Beginners:</b> A client model is like a student's personal copy of study materials.
/// Each student (client) has their own copy, studies it with their own resources, and
/// contributes improvements back to the class.
///
/// Think of client models as distributed learners:
/// - Each client has a copy of the global model
/// - Clients train on their own private data
/// - Local training happens independently and in parallel
/// - Only model updates (not data) are sent to the server
///
/// For example, in smartphone keyboard prediction:
/// - Each phone has a copy of the global typing prediction model
/// - The phone learns from the user's typing patterns
/// - It sends model improvements (not actual typed text) to the server
/// - The server combines improvements from millions of phones
/// - Each phone gets the improved model back
///
/// This design ensures:
/// - Data privacy: Raw data never leaves the client
/// - Personalization: Can adapt to local data distribution
/// - Scalability: Training happens in parallel across all clients
/// </remarks>
/// <typeparam name="TData">The type of the local training data.</typeparam>
/// <typeparam name="TUpdate">The type of the model update to send to the server.</typeparam>
public interface IClientModel<TData, TUpdate>
{
    /// <summary>
    /// Trains the local model on the client's private data.
    /// </summary>
    /// <remarks>
    /// Local training is the core of federated learning where each client improves the model
    /// using their own data without sharing that data with anyone.
    ///
    /// <b>For Beginners:</b> This is like studying independently with your own materials.
    /// You use your personal notes and resources to learn, and later share what you learned,
    /// not the actual materials.
    ///
    /// The training process:
    /// 1. Receive the global model from the server
    /// 2. Train on local data for specified number of epochs
    /// 3. Compute the difference between updated and original model (the "update")
    /// 4. Prepare this update to send back to the server
    ///
    /// For example:
    /// - Client receives global model with accuracy 80%
    /// - Trains on local data for 5 epochs
    /// - Local model now has accuracy 85% on local data
    /// - Computes weight changes (delta) that improved the model
    /// - Sends these weight changes to server, not the local data
    /// </remarks>
    /// <param name="localData">The client's private training data.</param>
    /// <param name="epochs">Number of training iterations to perform on local data.</param>
    /// <param name="learningRate">Step size for gradient descent optimization.</param>
    void TrainLocal(TData localData, int epochs, double learningRate);

    /// <summary>
    /// Computes and retrieves the model update to send to the server.
    /// </summary>
    /// <remarks>
    /// The model update represents the improvements the client made through local training.
    /// This is typically the difference between the current model and the initial global model.
    ///
    /// <b>For Beginners:</b> This is like preparing a summary of what you learned from studying,
    /// rather than sharing your entire study materials. You share the insights, not the sources.
    ///
    /// The update typically contains:
    /// - Weight differences: New weights - original weights
    /// - Gradients: Direction and magnitude of improvement
    /// - Metadata: Number of local samples, local loss, etc.
    ///
    /// For example:
    /// - Original weight for feature "age": 0.5
    /// - After training, weight for "age": 0.6
    /// - Update to send: +0.1
    /// - This tells the server how to adjust that weight
    /// </remarks>
    /// <returns>The model update containing weight changes or gradients.</returns>
    TUpdate GetModelUpdate();

    /// <summary>
    /// Updates the local model with the new global model from the server.
    /// </summary>
    /// <remarks>
    /// After the server aggregates updates from all clients, it sends the improved global
    /// model back to clients for the next round of training.
    ///
    /// <b>For Beginners:</b> This is like receiving the updated textbook that incorporates
    /// everyone's contributions. You replace your old version with this improved version
    /// before the next study session.
    ///
    /// The update process:
    /// 1. Receive aggregated global model from server
    /// 2. Replace local model weights with global model weights
    /// 3. Optionally keep some personalized layers
    /// 4. Ready for next round of local training
    ///
    /// For example:
    /// - Round 1: Trained local model, sent update
    /// - Server aggregated all updates
    /// - Round 2: Receive improved global model
    /// - Use this as starting point for next round of training
    /// </remarks>
    /// <param name="globalModelUpdate">The aggregated global model from the server.</param>
    void UpdateFromGlobal(TUpdate globalModelUpdate);

    /// <summary>
    /// Gets the number of training samples available on this client.
    /// </summary>
    /// <remarks>
    /// Sample count is used to weight client contributions during aggregation.
    /// Clients with more data typically receive higher weights.
    ///
    /// <b>For Beginners:</b> This is like indicating how many practice problems you solved.
    /// If you solved 1000 problems and someone else solved 100, your insights about
    /// problem-solving patterns are likely more reliable.
    /// </remarks>
    /// <returns>The number of training samples on this client.</returns>
    int GetSampleCount();
}
