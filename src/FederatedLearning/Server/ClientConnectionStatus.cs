namespace AiDotNet.FederatedLearning.Server
{
    /// <summary>
    /// Represents the connection status of a client in a federated learning system.
    /// </summary>
    public enum ClientConnectionStatus
    {
        /// <summary>
        /// The client is connected and ready to participate.
        /// </summary>
        Connected,

        /// <summary>
        /// The client has disconnected from the server.
        /// </summary>
        Disconnected,

        /// <summary>
        /// The client encountered an error.
        /// </summary>
        Error,

        /// <summary>
        /// The client is currently training on local data.
        /// </summary>
        Training,

        /// <summary>
        /// The client is updating its model parameters.
        /// </summary>
        Updating,

        /// <summary>
        /// The client is idle and waiting for instructions.
        /// </summary>
        Idle,

        /// <summary>
        /// The client is in the process of connecting.
        /// </summary>
        Connecting,

        /// <summary>
        /// The client connection has timed out.
        /// </summary>
        TimedOut
    }
}