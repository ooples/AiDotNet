using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.LinearAlgebra;
using AiDotNet.FederatedLearning.Communication.Models;

namespace AiDotNet.FederatedLearning.Communication.Interfaces
{
    /// <summary>
    /// Interface for communication management in federated learning
    /// </summary>
    public interface ICommunicationManager : IDisposable
    {
        /// <summary>
        /// Communication settings
        /// </summary>
        CommunicationSettings Settings { get; set; }

        /// <summary>
        /// Send global model to client
        /// </summary>
        Task<bool> SendGlobalModelAsync(string clientId, Dictionary<string, Vector<double>> globalParameters);

        /// <summary>
        /// Receive client update
        /// </summary>
        Task<Dictionary<string, Vector<double>>> ReceiveClientUpdateAsync(string clientId, TimeSpan timeout);

        /// <summary>
        /// Send client update to server
        /// </summary>
        Task<bool> SendClientUpdateAsync(string clientId, Dictionary<string, Vector<double>> clientUpdate);

        /// <summary>
        /// Receive global model from server
        /// </summary>
        Task<Dictionary<string, Vector<double>>> ReceiveGlobalModelAsync(TimeSpan timeout);

        /// <summary>
        /// Send status update
        /// </summary>
        Task<bool> SendStatusUpdateAsync(string senderId, string receiverId, Dictionary<string, object> status);

        /// <summary>
        /// Broadcast message to multiple recipients
        /// </summary>
        Task<Dictionary<string, bool>> BroadcastMessageAsync(string senderId, List<string> receiverIds, Dictionary<string, Vector<double>> parameters);

        /// <summary>
        /// Check connection status
        /// </summary>
        Task<bool> CheckConnectionAsync(string peerId);

        /// <summary>
        /// Get communication statistics
        /// </summary>
        CommunicationStatistics GetStatistics();
    }
}