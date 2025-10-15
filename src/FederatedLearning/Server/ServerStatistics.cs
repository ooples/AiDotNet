using System;
using System.Collections.Generic;

namespace AiDotNet.FederatedLearning.Server
{
    /// <summary>
    /// Represents statistics and metrics for a federated learning server.
    /// </summary>
    public class ServerStatistics
    {
        /// <summary>
        /// Gets or sets the server identifier.
        /// </summary>
        public string ServerId { get; set; }

        /// <summary>
        /// Gets or sets the current round number.
        /// </summary>
        public int CurrentRound { get; set; }

        /// <summary>
        /// Gets or sets the total number of connected clients.
        /// </summary>
        public int TotalConnectedClients { get; set; }

        /// <summary>
        /// Gets or sets the number of active clients.
        /// </summary>
        public int ActiveClients { get; set; }

        /// <summary>
        /// Gets or sets the total number of rounds configured.
        /// </summary>
        public int TotalRounds { get; set; }

        /// <summary>
        /// Gets or sets the current server status.
        /// </summary>
        public ServerStatus Status { get; set; }

        /// <summary>
        /// Gets or sets the average time per round.
        /// </summary>
        public TimeSpan AverageRoundTime { get; set; }

        /// <summary>
        /// Gets or sets the server uptime.
        /// </summary>
        public TimeSpan Uptime { get; set; }

        /// <summary>
        /// Gets or sets the timestamp when the server started.
        /// </summary>
        public DateTime StartTime { get; set; }

        /// <summary>
        /// Gets or sets the total data transferred in bytes.
        /// </summary>
        public long TotalDataTransferredBytes { get; set; }

        /// <summary>
        /// Gets or sets the current memory usage in megabytes.
        /// </summary>
        public double MemoryUsageMB { get; set; }

        /// <summary>
        /// Gets or sets the CPU usage percentage.
        /// </summary>
        public double CpuUsagePercent { get; set; }

        /// <summary>
        /// Gets or sets the number of failed rounds.
        /// </summary>
        public int FailedRounds { get; set; }

        /// <summary>
        /// Gets or sets the success rate of rounds.
        /// </summary>
        public double RoundSuccessRate { get; set; }

        /// <summary>
        /// Gets or sets additional custom metrics.
        /// </summary>
        public Dictionary<string, double> CustomMetrics { get; set; }

        /// <summary>
        /// Initializes a new instance of the ServerStatistics class.
        /// </summary>
        public ServerStatistics()
        {
            ServerId = string.Empty;
            CustomMetrics = new Dictionary<string, double>();
            StartTime = DateTime.UtcNow;
            Status = ServerStatus.Initializing;
        }

        /// <summary>
        /// Updates the uptime based on the current time.
        /// </summary>
        public void UpdateUptime()
        {
            Uptime = DateTime.UtcNow - StartTime;
        }

        /// <summary>
        /// Calculates the percentage of clients that are active.
        /// </summary>
        /// <returns>The percentage of active clients.</returns>
        public double GetActiveClientPercentage()
        {
            if (TotalConnectedClients == 0)
                return 0.0;

            return (double)ActiveClients / TotalConnectedClients * 100.0;
        }

        /// <summary>
        /// Calculates the progress percentage of the training.
        /// </summary>
        /// <returns>The training progress percentage.</returns>
        public double GetProgressPercentage()
        {
            if (TotalRounds == 0)
                return 0.0;

            return (double)CurrentRound / TotalRounds * 100.0;
        }

        /// <summary>
        /// Gets the average data transfer per client in megabytes.
        /// </summary>
        /// <returns>The average data transfer per client.</returns>
        public double GetAverageDataTransferPerClientMB()
        {
            if (TotalConnectedClients == 0)
                return 0.0;

            return TotalDataTransferredBytes / (1024.0 * 1024.0) / TotalConnectedClients;
        }

        /// <summary>
        /// Generates a formatted statistics report.
        /// </summary>
        /// <returns>A formatted string containing server statistics.</returns>
        public string GenerateReport()
        {
            UpdateUptime();
            
            return $"Server Statistics Report\n" +
                   $"========================\n" +
                   $"Server ID: {ServerId}\n" +
                   $"Status: {Status}\n" +
                   $"Uptime: {Uptime.TotalHours:F1} hours\n" +
                   $"Progress: {CurrentRound}/{TotalRounds} rounds ({GetProgressPercentage():F1}%)\n" +
                   $"Clients: {ActiveClients}/{TotalConnectedClients} active ({GetActiveClientPercentage():F1}%)\n" +
                   $"Average Round Time: {AverageRoundTime.TotalSeconds:F1}s\n" +
                   $"Round Success Rate: {RoundSuccessRate:P1}\n" +
                   $"Memory Usage: {MemoryUsageMB:F1} MB\n" +
                   $"CPU Usage: {CpuUsagePercent:F1}%\n" +
                   $"Data Transferred: {TotalDataTransferredBytes / (1024.0 * 1024.0):F1} MB total";
        }

        /// <summary>
        /// Creates a shallow copy of the current statistics.
        /// </summary>
        /// <returns>A new ServerStatistics instance with copied values.</returns>
        public ServerStatistics Clone()
        {
            return new ServerStatistics
            {
                ServerId = ServerId,
                CurrentRound = CurrentRound,
                TotalConnectedClients = TotalConnectedClients,
                ActiveClients = ActiveClients,
                TotalRounds = TotalRounds,
                Status = Status,
                AverageRoundTime = AverageRoundTime,
                Uptime = Uptime,
                StartTime = StartTime,
                TotalDataTransferredBytes = TotalDataTransferredBytes,
                MemoryUsageMB = MemoryUsageMB,
                CpuUsagePercent = CpuUsagePercent,
                FailedRounds = FailedRounds,
                RoundSuccessRate = RoundSuccessRate,
                CustomMetrics = new Dictionary<string, double>(CustomMetrics)
            };
        }
    }
}