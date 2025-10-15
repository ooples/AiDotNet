namespace AiDotNet.FederatedLearning.Server
{
    /// <summary>
    /// Represents the operational status of a federated learning server.
    /// </summary>
    public enum ServerStatus
    {
        /// <summary>
        /// The server is initializing and not yet ready for operations.
        /// </summary>
        Initializing,

        /// <summary>
        /// The server is ready to accept clients and start training.
        /// </summary>
        Ready,

        /// <summary>
        /// The server is actively coordinating a training round.
        /// </summary>
        Training,

        /// <summary>
        /// The server is aggregating model updates from clients.
        /// </summary>
        Aggregating,

        /// <summary>
        /// The server has completed the federated learning process.
        /// </summary>
        Completed,

        /// <summary>
        /// The server encountered an error and is in an error state.
        /// </summary>
        Error,

        /// <summary>
        /// The server has been stopped manually.
        /// </summary>
        Stopped,

        /// <summary>
        /// The server is paused and can be resumed.
        /// </summary>
        Paused,

        /// <summary>
        /// The server is waiting for minimum clients to connect.
        /// </summary>
        WaitingForClients,

        /// <summary>
        /// The server is validating the global model.
        /// </summary>
        Validating
    }
}