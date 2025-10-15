namespace AiDotNet.FederatedLearning.Client
{
    /// <summary>
    /// Client status enumeration
    /// </summary>
    public enum ClientStatus
    {
        /// <summary>
        /// Client is ready to participate in training
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
        /// Client has encountered an error
        /// </summary>
        Error,
        
        /// <summary>
        /// Client is disconnected from the federated learning network
        /// </summary>
        Disconnected
    }
}