using System;
using System.Collections.Generic;

namespace AiDotNet.FederatedLearning.Server
{
    /// <summary>
    /// Client information for server tracking in federated learning scenarios.
    /// </summary>
    public class ClientInfo
    {
        /// <summary>
        /// Gets or sets the unique identifier for the client.
        /// </summary>
        public string ClientId { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the current connection status of the client.
        /// </summary>
        public ClientConnectionStatus Status { get; set; }

        /// <summary>
        /// Gets or sets the time when the client registered with the server.
        /// </summary>
        public DateTime RegistrationTime { get; set; }

        /// <summary>
        /// Gets or sets the time of the last communication from the client.
        /// </summary>
        public DateTime LastCommunication { get; set; }

        /// <summary>
        /// Gets or sets the size of the data available on the client.
        /// </summary>
        public int DataSize { get; set; }

        /// <summary>
        /// Gets or sets the version of the client software.
        /// </summary>
        public string ClientVersion { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets additional metadata associated with the client.
        /// </summary>
        public Dictionary<string, object> Metadata { get; set; }

        /// <summary>
        /// Initializes a new instance of the ClientInfo class.
        /// </summary>
        public ClientInfo()
        {
            Metadata = new Dictionary<string, object>();
            RegistrationTime = DateTime.UtcNow;
            LastCommunication = DateTime.UtcNow;
            Status = ClientConnectionStatus.Disconnected;
        }

        /// <summary>
        /// Initializes a new instance of the ClientInfo class with a client ID.
        /// </summary>
        /// <param name="clientId">The unique identifier for the client.</param>
        public ClientInfo(string clientId) : this()
        {
            ClientId = clientId ?? throw new ArgumentNullException(nameof(clientId));
        }

        /// <summary>
        /// Updates the last communication time to the current UTC time.
        /// </summary>
        public void UpdateLastCommunication()
        {
            LastCommunication = DateTime.UtcNow;
        }

        /// <summary>
        /// Checks if the client is considered active based on the last communication time.
        /// </summary>
        /// <param name="timeout">The timeout period after which a client is considered inactive.</param>
        /// <returns>True if the client is active; otherwise, false.</returns>
        public bool IsActive(TimeSpan timeout)
        {
            return DateTime.UtcNow - LastCommunication < timeout;
        }

        /// <summary>
        /// Creates a shallow copy of the current ClientInfo instance.
        /// </summary>
        /// <returns>A new ClientInfo instance with copied values.</returns>
        public ClientInfo Clone()
        {
            return new ClientInfo
            {
                ClientId = ClientId,
                Status = Status,
                RegistrationTime = RegistrationTime,
                LastCommunication = LastCommunication,
                DataSize = DataSize,
                ClientVersion = ClientVersion,
                Metadata = new Dictionary<string, object>(Metadata)
            };
        }

        /// <summary>
        /// Returns a string representation of the client information.
        /// </summary>
        /// <returns>A string containing the client ID and status.</returns>
        public override string ToString()
        {
            return $"Client {ClientId}: Status={Status}, DataSize={DataSize}, Version={ClientVersion}";
        }
    }
}