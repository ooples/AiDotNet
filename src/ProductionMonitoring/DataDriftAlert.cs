using System;

namespace AiDotNet.ProductionMonitoring
{
    /// <summary>
    /// Alert for data drift detection
    /// </summary>
    public class DataDriftAlert
    {
        /// <summary>
        /// Gets or sets the alert ID
        /// </summary>
        public string AlertId { get; set; } = Guid.NewGuid().ToString();
        
        /// <summary>
        /// Gets or sets the timestamp of the alert
        /// </summary>
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
        
        /// <summary>
        /// Gets or sets the feature name
        /// </summary>
        public string FeatureName { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the drift score
        /// </summary>
        public double DriftScore { get; set; }
        
        /// <summary>
        /// Gets or sets the alert severity
        /// </summary>
        public string Severity { get; set; } = "Medium";
        
        /// <summary>
        /// Gets or sets the alert message
        /// </summary>
        public string Message { get; set; } = string.Empty;
    }
}