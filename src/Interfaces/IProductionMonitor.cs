using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace AiDotNet.Interfaces
{
    /// <summary>
    /// Interface for production monitoring of deployed models
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations</typeparam>
    public interface IProductionMonitor<T> : IProductionMonitor
    {
        /// <summary>
        /// Monitors data drift between training and production data
        /// </summary>
        Task<DriftDetectionResult> DetectDataDriftAsync(Matrix<T> productionData, Matrix<T>? referenceData = null);

        /// <summary>
        /// Monitors concept drift in model predictions
        /// </summary>
        Task<DriftDetectionResult> DetectConceptDriftAsync(Vector<T> predictions, Vector<T> actuals);

        /// <summary>
        /// Tracks model performance metrics over time
        /// </summary>
        Task<PerformanceMetrics> GetPerformanceMetricsAsync(DateTime? startDate = null, DateTime? endDate = null);

        /// <summary>
        /// Evaluates overall model health
        /// </summary>
        Task<ModelHealthScore> GetModelHealthScoreAsync();

        /// <summary>
        /// Provides retraining recommendations based on monitoring data
        /// </summary>
        Task<RetrainingRecommendation> GetRetrainingRecommendationAsync();

        /// <summary>
        /// Registers an alert handler for monitoring events
        /// </summary>
        void RegisterAlertHandler(Action<MonitoringAlert> handler);

        /// <summary>
        /// Logs prediction for monitoring
        /// </summary>
        Task LogPredictionAsync(Vector<T> features, T prediction, DateTime? timestamp = null);
        
        /// <summary>
        /// Logs prediction with actual value for monitoring
        /// </summary>
        Task LogPredictionAsync(Vector<T> features, T prediction, T actual, DateTime? timestamp = null);

        /// <summary>
        /// Gets monitoring metrics for a specific time period
        /// </summary>
        Task<MonitoringMetricsCollection> GetMonitoringMetricsAsync(DateTime startDate, DateTime endDate);

        /// <summary>
        /// Configures monitoring thresholds
        /// </summary>
        void ConfigureThresholds(MonitoringThresholds thresholds);

        /// <summary>
        /// Checks for data drift in production data (synchronous version)
        /// </summary>
        /// <param name="productionData">Current production data to check</param>
        /// <param name="referenceData">Reference data to compare against (optional)</param>
        /// <returns>Drift detection result</returns>
        DriftDetectionResult CheckDataDrift(double[,] productionData, double[,]? referenceData = null);

        /// <summary>
        /// Determines if the model should be retrained based on monitoring data
        /// </summary>
        /// <returns>True if retraining is recommended, false otherwise</returns>
        bool ShouldRetrain();

        /// <summary>
        /// Configures drift detection settings
        /// </summary>
        /// <param name="method">Drift detection method to use</param>
        /// <param name="threshold">Drift threshold value</param>
        /// <param name="windowSize">Size of the monitoring window</param>
        void ConfigureDriftDetection(string method, double threshold, int windowSize = 1000);

        /// <summary>
        /// Configures automatic retraining settings
        /// </summary>
        /// <param name="enabled">Whether to enable automatic retraining</param>
        /// <param name="performanceThreshold">Performance threshold that triggers retraining</param>
        /// <param name="driftThreshold">Drift threshold that triggers retraining</param>
        void ConfigureRetraining(bool enabled, double performanceThreshold = 0.8, double driftThreshold = 0.3);
    }

    /// <summary>
    /// Result of drift detection
    /// </summary>
    public class DriftDetectionResult
    {
        public bool IsDriftDetected { get; set; }
        public double DriftScore { get; set; }
        public string DriftType { get; set; } = string.Empty;
        public Dictionary<string, double> FeatureDrifts { get; set; } = new();
        public DateTime DetectionTimestamp { get; set; }
        public string Details { get; set; } = string.Empty;
    }

    /// <summary>
    /// Model performance metrics
    /// </summary>
    public class PerformanceMetrics
    {
        public double Accuracy { get; set; }
        public double Precision { get; set; }
        public double Recall { get; set; }
        public double F1Score { get; set; }
        public double MAE { get; set; }
        public double RMSE { get; set; }
        public Dictionary<string, double> CustomMetrics { get; set; } = new();
        public DateTime Timestamp { get; set; }
        public int PredictionCount { get; set; }
    }

    /// <summary>
    /// Overall model health score
    /// </summary>
    public class ModelHealthScore
    {
        public double OverallScore { get; set; }
        public double DataQualityScore { get; set; }
        public double PerformanceScore { get; set; }
        public double StabilityScore { get; set; }
        public double DriftScore { get; set; }
        public string HealthStatus { get; set; } = string.Empty;
        public List<string> Issues { get; set; } = new();
        public DateTime EvaluationTimestamp { get; set; }
    }

    /// <summary>
    /// Retraining recommendation
    /// </summary>
    public class RetrainingRecommendation
    {
        public bool ShouldRetrain { get; set; }
        public string Urgency { get; set; } = string.Empty; // Low, Medium, High, Critical
        public List<string> Reasons { get; set; } = new();
        public DateTime RecommendationTimestamp { get; set; }
        public double ConfidenceScore { get; set; }
        public Dictionary<string, object> SuggestedActions { get; set; } = new();
    }

    /// <summary>
    /// Monitoring alert
    /// </summary>
    public class MonitoringAlert
    {
        public string AlertType { get; set; } = string.Empty;
        public string Severity { get; set; } = string.Empty; // Info, Warning, Error, Critical
        public string Message { get; set; } = string.Empty;
        public DateTime Timestamp { get; set; }
        public Dictionary<string, object> Context { get; set; } = new();
    }

    /// <summary>
    /// Collection of monitoring metrics
    /// </summary>
    public class MonitoringMetricsCollection
    {
        public List<PerformanceMetrics> PerformanceHistory { get; set; } = new();
        public List<DriftDetectionResult> DriftHistory { get; set; } = new();
        public Dictionary<string, List<double>> FeatureStatistics { get; set; } = new();
        public int TotalPredictions { get; set; }
        public DateTime StartDate { get; set; }
        public DateTime EndDate { get; set; }
    }

    /// <summary>
    /// Monitoring thresholds configuration
    /// </summary>
    public class MonitoringThresholds
    {
        public double DataDriftThreshold { get; set; } = 0.3;
        public double ConceptDriftThreshold { get; set; } = 0.2;
        public double PerformanceDropThreshold { get; set; } = 0.1;
        public double HealthScoreWarningThreshold { get; set; } = 0.7;
        public double HealthScoreCriticalThreshold { get; set; } = 0.5;
        public int MinimumSamplesForDriftDetection { get; set; } = 100;
    }
}
