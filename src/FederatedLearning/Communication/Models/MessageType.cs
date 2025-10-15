namespace AiDotNet.FederatedLearning.Communication.Models
{
    /// <summary>
    /// Types of messages used in federated learning communication
    /// </summary>
    public enum MessageType
    {
        /// <summary>
        /// Global model update from server to clients
        /// </summary>
        GlobalModelUpdate,

        /// <summary>
        /// Client model update to server
        /// </summary>
        ClientUpdate,

        /// <summary>
        /// Status update message
        /// </summary>
        StatusUpdate,

        /// <summary>
        /// Ping message for connection testing
        /// </summary>
        Ping,

        /// <summary>
        /// Pong response to ping
        /// </summary>
        Pong,

        /// <summary>
        /// Error message
        /// </summary>
        Error,

        /// <summary>
        /// Disconnect notification
        /// </summary>
        Disconnect
    }
}