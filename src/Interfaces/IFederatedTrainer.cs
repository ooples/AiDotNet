namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the core functionality for federated learning trainers that coordinate distributed training across multiple clients.
/// </summary>
/// <remarks>
/// This interface represents the fundamental operations for federated learning systems where multiple clients
/// (devices, institutions, edge nodes) collaboratively train a shared model without sharing their raw data.
///
/// <b>For Beginners:</b> Federated learning is like group study where everyone learns from their own materials
/// but shares only their insights, not their actual study materials.
///
/// Think of federated learning as a privacy-preserving collaborative learning approach:
/// - Multiple clients (hospitals, phones, banks) have their own local data
/// - Each client trains a model on their local data independently
/// - Only model updates (not raw data) are shared with a central server
/// - The server aggregates these updates to improve the global model
/// - The improved global model is sent back to clients for the next round
///
/// For example, in healthcare:
/// - Multiple hospitals want to train a disease detection model
/// - Each hospital has patient data that cannot be shared due to privacy regulations
/// - Each hospital trains the model on their own data
/// - Only the learned patterns (model weights) are shared and combined
/// - This creates a better model while keeping patient data private
///
/// This interface provides methods for coordinating the federated training process.
/// </remarks>
/// <typeparam name="TModel">The type of the model being trained.</typeparam>
/// <typeparam name="TData">The type of the training data.</typeparam>
/// <typeparam name="TMetadata">The type of metadata returned by the training process.</typeparam>
public interface IFederatedTrainer<TModel, TData, TMetadata>
{
    /// <summary>
    /// Initializes the federated learning process with client configurations and the global model.
    /// </summary>
    /// <remarks>
    /// This method sets up the initial state for federated learning by:
    /// - Initializing the global model that will be shared across all clients
    /// - Registering client configurations (number of clients, data distribution, etc.)
    /// - Setting up communication channels for model updates
    ///
    /// <b>For Beginners:</b> Initialization is like setting up a study group before the first session.
    /// You need to know who's participating, what materials everyone has, and establish
    /// how you'll share information.
    /// </remarks>
    /// <param name="globalModel">The initial global model to be distributed to clients.</param>
    /// <param name="numberOfClients">The number of clients participating in federated learning.</param>
    void Initialize(TModel globalModel, int numberOfClients);

    /// <summary>
    /// Executes one round of federated learning where clients train locally and updates are aggregated.
    /// </summary>
    /// <remarks>
    /// A federated learning round consists of several steps:
    /// 1. The global model is sent to selected clients
    /// 2. Each client trains the model on their local data
    /// 3. Clients send their model updates back to the server
    /// 4. The server aggregates these updates using an aggregation strategy
    /// 5. The global model is updated with the aggregated result
    ///
    /// <b>For Beginners:</b> Think of this as one iteration in a collaborative learning cycle.
    /// Everyone gets the current version of the shared knowledge, studies independently,
    /// then contributes their improvements. These improvements are combined to create
    /// an even better version for the next round.
    ///
    /// For example:
    /// - Round 1: Clients receive initial model, train for 5 epochs, send updates
    /// - Server aggregates updates and improves global model
    /// - Round 2: Clients receive improved model, train again, send updates
    /// - This continues until the model reaches desired accuracy
    /// </remarks>
    /// <param name="clientData">Dictionary mapping client IDs to their local training data.</param>
    /// <param name="clientSelectionFraction">Fraction of clients to select for this round (0.0 to 1.0).</param>
    /// <param name="localEpochs">Number of training epochs each client should perform locally.</param>
    /// <returns>Metadata about the training round including accuracy, loss, and convergence metrics.</returns>
    TMetadata TrainRound(Dictionary<int, TData> clientData, double clientSelectionFraction = 1.0, int localEpochs = 1);

    /// <summary>
    /// Executes multiple rounds of federated learning until convergence or maximum rounds reached.
    /// </summary>
    /// <remarks>
    /// This method orchestrates the entire federated learning process by:
    /// - Running multiple training rounds
    /// - Monitoring convergence (when the model stops improving significantly)
    /// - Tracking performance metrics across rounds
    /// - Applying privacy mechanisms if configured
    ///
    /// <b>For Beginners:</b> This is the complete federated learning process from start to finish.
    /// It's like running an entire semester of study group sessions, where you continue meeting
    /// until everyone has mastered the material or you've run out of time.
    /// </remarks>
    /// <param name="clientData">Dictionary mapping client IDs to their local training data.</param>
    /// <param name="rounds">Maximum number of federated learning rounds to execute.</param>
    /// <param name="clientSelectionFraction">Fraction of clients to select per round (0.0 to 1.0).</param>
    /// <param name="localEpochs">Number of training epochs each client performs per round.</param>
    /// <returns>Aggregated metadata across all training rounds.</returns>
    TMetadata Train(Dictionary<int, TData> clientData, int rounds, double clientSelectionFraction = 1.0, int localEpochs = 1);

    /// <summary>
    /// Retrieves the current global model after federated training.
    /// </summary>
    /// <remarks>
    /// The global model represents the collective knowledge learned from all participating clients.
    ///
    /// <b>For Beginners:</b> This is the final product of the collaborative learning process -
    /// a model that benefits from all participants' data without ever accessing their raw data directly.
    /// </remarks>
    /// <returns>The trained global model.</returns>
    TModel GetGlobalModel();

    /// <summary>
    /// Sets the aggregation strategy used to combine client updates.
    /// </summary>
    /// <remarks>
    /// Different aggregation strategies handle various challenges in federated learning:
    /// - FedAvg: Simple weighted averaging of model updates
    /// - FedProx: Handles clients with different computational capabilities
    /// - FedBN: Special handling for batch normalization layers
    ///
    /// <b>For Beginners:</b> The aggregation strategy is the rule for combining everyone's
    /// contributions. Different rules work better for different situations, like how you might
    /// weight expert opinions more heavily in certain contexts.
    /// </remarks>
    /// <param name="strategy">The aggregation strategy to use.</param>
    void SetAggregationStrategy(IAggregationStrategy<TModel> strategy);
}
