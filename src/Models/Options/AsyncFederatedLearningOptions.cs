namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for asynchronous federated learning (FedAsync / FedBuff).
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> In synchronous federated learning, the server waits for a whole "round" of clients
/// before updating the global model. In asynchronous federated learning, the server can update as client
/// updates arrive (or in small buffers), which can reduce waiting on slow clients.
/// </remarks>
public class AsyncFederatedLearningOptions : ModelOptions
{
    /// <summary>
    /// Gets or sets the async mode.
    /// </summary>
    public FederatedAsyncMode Mode { get; set; } = FederatedAsyncMode.None;

    /// <summary>
    /// Gets or sets the base mixing rate used by FedAsync.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> FedAsync blends the current global model with an incoming client update.
    /// A value of 0.5 means "move halfway toward the client update" (before staleness weighting).
    /// </remarks>
    public double FedAsyncMixingRate { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the staleness weighting mode for asynchronous updates.
    /// </summary>
    public FederatedStalenessWeighting StalenessWeighting { get; set; } = FederatedStalenessWeighting.Inverse;

    /// <summary>
    /// Gets or sets the staleness decay rate used by "Exponential" and "Polynomial" weighting.
    /// </summary>
    public double StalenessDecayRate { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the maximum simulated client delay (in server steps) for in-memory async training.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This setting is used only by the in-memory simulator to model clients that
    /// respond late. A value of 0 makes the simulation effectively synchronous.
    /// </remarks>
    public int SimulatedMaxClientDelaySteps { get; set; } = 0;

    /// <summary>
    /// Gets or sets the buffer size for FedBuff (number of updates to accumulate before applying a server update).
    /// </summary>
    public int FedBuffBufferSize { get; set; } = 10;

    /// <summary>
    /// Gets or sets the maximum staleness allowed before rejecting an update.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> If this is greater than 0, updates older than this threshold are ignored.
    /// </remarks>
    public int RejectUpdatesWithStalenessGreaterThan { get; set; } = 0;

    // --- Semi-Async ---

    /// <summary>
    /// Gets or sets the number of async rounds between synchronization barriers (Semi-Async mode). Default: 5.
    /// </summary>
    public int SemiAsyncRoundsPerBarrier { get; set; } = 5;

    // --- AsyncFedED ---

    /// <summary>
    /// Gets or sets the maximum clients to select per round in AsyncFedED. Default: 10.
    /// </summary>
    public int AsyncFedEDSelectionBudget { get; set; } = 10;

    /// <summary>
    /// Gets or sets the exploration bonus for unvisited clients in AsyncFedED scheduling. Default: 0.1.
    /// </summary>
    public double AsyncFedEDExplorationBonus { get; set; } = 0.1;
}

