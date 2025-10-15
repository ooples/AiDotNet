using System;
using System.Collections.Generic;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.FederatedLearning.Communication.Models
{
    /// <summary>
    /// Represents a message in federated learning communication
    /// </summary>
    public class FederatedMessage
    {
        /// <summary>
        /// Type of the message
        /// </summary>
        public MessageType MessageType { get; set; }

        /// <summary>
        /// Identifier of the sender
        /// </summary>
        public string SenderId { get; set; } = string.Empty;

        /// <summary>
        /// Identifier of the receiver
        /// </summary>
        public string ReceiverId { get; set; } = string.Empty;

        /// <summary>
        /// Timestamp when the message was created
        /// </summary>
        public DateTime Timestamp { get; set; }

        /// <summary>
        /// Model parameters being transmitted
        /// </summary>
        public Dictionary<string, Vector<double>>? Parameters { get; set; }

        /// <summary>
        /// Additional metadata associated with the message
        /// </summary>
        public Dictionary<string, object>? Metadata { get; set; }

        /// <summary>
        /// Indicates whether the message is compressed
        /// </summary>
        public bool IsCompressed { get; set; }

        /// <summary>
        /// Indicates whether the message is encrypted
        /// </summary>
        public bool IsEncrypted { get; set; }

        /// <summary>
        /// Type of compression used
        /// </summary>
        public string? CompressionType { get; set; }

        /// <summary>
        /// Type of encryption used
        /// </summary>
        public string? EncryptionType { get; set; }

        /// <summary>
        /// Compressed data payload
        /// </summary>
        public byte[]? CompressedData { get; set; }

        /// <summary>
        /// Encrypted data payload
        /// </summary>
        public byte[]? EncryptedData { get; set; }
    }
}