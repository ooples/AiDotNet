namespace AiDotNet.FederatedLearning.Client.Enums
{
    /// <summary>
    /// Represents the current status of a federated learning client
    /// </summary>
    public enum ClientStatus
    {
        /// <summary>
        /// Client is ready to receive commands
        /// </summary>
        Ready,
        
        /// <summary>
        /// Client is currently training
        /// </summary>
        Training,
        
        /// <summary>
        /// Client is updating its local model
        /// </summary>
        UpdatingModel,
        
        /// <summary>
        /// Client is communicating with the server
        /// </summary>
        Communicating,
        
        /// <summary>
        /// Client encountered an error
        /// </summary>
        Error,
        
        /// <summary>
        /// Client is disconnected from the server
        /// </summary>
        Disconnected,

        /// <summary>
        /// Client is initializing
        /// </summary>
        Initializing,

        /// <summary>
        /// Client is validating its model
        /// </summary>
        Validating,

        /// <summary>
        /// Client is idle/waiting
        /// </summary>
        Idle,

        /// <summary>
        /// Client is synchronizing with server
        /// </summary>
        Synchronizing,

        /// <summary>
        /// Client is performing data preprocessing
        /// </summary>
        Preprocessing,

        /// <summary>
        /// Client is shutting down
        /// </summary>
        ShuttingDown
    }
}