using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.FederatedLearning.Communication.Models
{
    /// <summary>
    /// Statistics for federated learning communication
    /// </summary>
    public class CommunicationStatistics
    {
        /// <summary>
        /// Total number of messages sent
        /// </summary>
        public int MessagesSent { get; private set; }

        /// <summary>
        /// Total number of messages received
        /// </summary>
        public int MessagesReceived { get; private set; }

        /// <summary>
        /// Total number of errors encountered
        /// </summary>
        public int Errors { get; private set; }

        /// <summary>
        /// Total number of timeouts
        /// </summary>
        public int Timeouts { get; private set; }

        /// <summary>
        /// Total bytes sent
        /// </summary>
        public long TotalBytesSent { get; private set; }

        /// <summary>
        /// Total bytes received
        /// </summary>
        public long TotalBytesReceived { get; private set; }

        /// <summary>
        /// Average latency of communication
        /// </summary>
        public TimeSpan AverageLatency { get; private set; }

        /// <summary>
        /// Count of messages by type
        /// </summary>
        public Dictionary<MessageType, int> MessageCounts { get; private set; }
        
        private readonly List<TimeSpan> _latencies;

        /// <summary>
        /// Initializes a new instance of CommunicationStatistics
        /// </summary>
        public CommunicationStatistics()
        {
            MessageCounts = new Dictionary<MessageType, int>();
            _latencies = new List<TimeSpan>();
        }

        /// <summary>
        /// Record a sent message
        /// </summary>
        /// <param name="messageType">Type of message</param>
        /// <param name="size">Size of message in bytes</param>
        /// <param name="latency">Communication latency</param>
        /// <param name="success">Whether the send was successful</param>
        public void RecordMessageSent(MessageType messageType, int size, TimeSpan latency, bool success)
        {
            if (success)
            {
                MessagesSent++;
                TotalBytesSent += size;
                _latencies.Add(latency);
                UpdateAverageLatency();
                
                if (!MessageCounts.ContainsKey(messageType))
                    MessageCounts[messageType] = 0;
                MessageCounts[messageType]++;
            }
            else
            {
                Errors++;
            }
        }

        /// <summary>
        /// Record a received message
        /// </summary>
        /// <param name="messageType">Type of message</param>
        /// <param name="size">Size of message in bytes</param>
        /// <param name="latency">Communication latency</param>
        /// <param name="success">Whether the receive was successful</param>
        public void RecordMessageReceived(MessageType messageType, int size, TimeSpan latency, bool success)
        {
            if (success)
            {
                MessagesReceived++;
                TotalBytesReceived += size;
                _latencies.Add(latency);
                UpdateAverageLatency();
            }
            else
            {
                Errors++;
            }
        }

        /// <summary>
        /// Record an error
        /// </summary>
        /// <param name="messageType">Type of message that caused the error</param>
        /// <param name="error">Error description</param>
        public void RecordError(MessageType messageType, string error)
        {
            Errors++;
        }

        /// <summary>
        /// Record a timeout
        /// </summary>
        /// <param name="messageType">Type of message that timed out</param>
        public void RecordTimeout(MessageType messageType)
        {
            Timeouts++;
        }

        /// <summary>
        /// Update the average latency
        /// </summary>
        private void UpdateAverageLatency()
        {
            if (_latencies.Count > 0)
            {
                var totalTicks = _latencies.Sum(l => l.Ticks);
                AverageLatency = new TimeSpan(totalTicks / _latencies.Count);
            }
        }

        /// <summary>
        /// Reset all statistics
        /// </summary>
        public void Reset()
        {
            MessagesSent = 0;
            MessagesReceived = 0;
            Errors = 0;
            Timeouts = 0;
            TotalBytesSent = 0;
            TotalBytesReceived = 0;
            AverageLatency = TimeSpan.Zero;
            MessageCounts.Clear();
            _latencies.Clear();
        }

        /// <summary>
        /// Get a summary of the statistics
        /// </summary>
        /// <returns>String summary of statistics</returns>
        public override string ToString()
        {
            return $"Messages: Sent={MessagesSent}, Received={MessagesReceived}, " +
                   $"Errors={Errors}, Timeouts={Timeouts}, " +
                   $"Bytes: Sent={TotalBytesSent}, Received={TotalBytesReceived}, " +
                   $"Avg Latency={AverageLatency.TotalMilliseconds}ms";
        }
    }
}