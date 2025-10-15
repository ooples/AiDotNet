namespace AiDotNet.ProductionMonitoring
{
    /// <summary>
    /// Configuration for monitoring alerts
    /// </summary>
    public class AlertConfiguration
    {
        /// <summary>
        /// Gets or sets the drift threshold
        /// </summary>
        public double DriftThreshold { get; set; } = 0.1;
        
        /// <summary>
        /// Gets or sets the performance degradation threshold
        /// </summary>
        public double PerformanceDegradationThreshold { get; set; } = 0.05;
        
        /// <summary>
        /// Gets or sets the prediction outlier threshold
        /// </summary>
        public double PredictionOutlierThreshold { get; set; } = 3.0;
        
        /// <summary>
        /// Gets or sets whether email alerts are enabled
        /// </summary>
        public bool EnableEmailAlerts { get; set; } = true;
        
        /// <summary>
        /// Gets or sets the alert email address
        /// </summary>
        public string AlertEmail { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the monitoring window size in hours
        /// </summary>
        public int MonitoringWindowHours { get; set; } = 24;
    }
}