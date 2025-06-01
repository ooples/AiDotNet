using AiDotNet.Enums;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace AiDotNet.Interfaces
{
    /// <summary>
    /// Interface for production monitoring of deployed models
    /// </summary>
    public interface IProductionMonitor
    {
        /// <summary>
        /// Monitors data drift between training and production data
        /// </summary>
        Task<DriftDetectionResult> DetectDataDriftAsync(double[,] productionData, double[,] referenceData = null);

        /// <summary>
        /// Monitors concept drift in model predictions
        /// </summary>
        Task<DriftDetectionResult> DetectConceptDriftAsync(double[] predictions, double[] actuals);

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
        Task LogPredictionAsync(double[] features, double prediction, double? actual = null, DateTime? timestamp = null);

        /// <summary>
        /// Gets monitoring metrics for a specific time period
        /// </summary>
        Task<MonitoringMetricsCollection> GetMonitoringMetricsAsync(DateTime startDate, DateTime endDate);

        /// <summary>
        /// Configures monitoring thresholds
        /// </summary>
        void ConfigureThresholds(MonitoringThresholds thresholds);
    }

    /// <summary>
    /// Result of drift detection
    /// </summary>
    public class DriftDetectionResult
    {
        public bool IsDriftDetected { get; set; }
        public double DriftScore { get; set; }
        public string DriftType { get; set; }
        public Dictionary<string, double> FeatureDrifts { get; set; }
        public DateTime DetectionTimestamp { get; set; }
        public string Details { get; set; }
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
        public Dictionary<string, double> CustomMetrics { get; set; }
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
        public string HealthStatus { get; set; }
        public List<string> Issues { get; set; }
        public DateTime EvaluationTimestamp { get; set; }
    }

    /// <summary>
    /// Retraining recommendation
    /// </summary>
    public class RetrainingRecommendation
    {
        public bool ShouldRetrain { get; set; }
        public string Urgency { get; set; } // Low, Medium, High, Critical
        public List<string> Reasons { get; set; }
        public DateTime RecommendationTimestamp { get; set; }
        public double ConfidenceScore { get; set; }
        public Dictionary<string, object> SuggestedActions { get; set; }
    }

    /// <summary>
    /// Monitoring alert
    /// </summary>
    public class MonitoringAlert
    {
        public string AlertType { get; set; }
        public string Severity { get; set; } // Info, Warning, Error, Critical
        public string Message { get; set; }
        public DateTime Timestamp { get; set; }
        public Dictionary<string, object> Context { get; set; }
    }

    /// <summary>
    /// Collection of monitoring metrics
    /// </summary>
    public class MonitoringMetricsCollection
    {
        public List<PerformanceMetrics> PerformanceHistory { get; set; }
        public List<DriftDetectionResult> DriftHistory { get; set; }
        public Dictionary<string, List<double>> FeatureStatistics { get; set; }
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