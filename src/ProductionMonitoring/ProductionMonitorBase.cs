using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AiDotNet.ProductionMonitoring
{
    /// <summary>
    /// Base class for production monitoring implementations
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations</typeparam>
    public abstract class ProductionMonitorBase<T> : IProductionMonitor<T>
    {
        protected readonly List<Action<MonitoringAlert>> _alertHandlers;
        protected readonly List<PredictionRecord> _predictionHistory;
        protected readonly List<PerformanceMetrics> _performanceHistory;
        protected readonly List<DriftDetectionResult> _driftHistory;
        protected MonitoringThresholds _thresholds;
        protected Matrix<T>? _referenceData;
        protected readonly object _lockObject = new object();
        protected static readonly INumericOperations<T> _ops = MathHelper.GetNumericOperations<T>();

        protected ProductionMonitorBase()
        {
            _alertHandlers = new List<Action<MonitoringAlert>>();
            _predictionHistory = new List<PredictionRecord>();
            _performanceHistory = new List<PerformanceMetrics>();
            _driftHistory = new List<DriftDetectionResult>();
            _thresholds = new MonitoringThresholds();
        }

        // Additional fields for IProductionMonitor interface
        protected DriftDetectionMethod _driftDetectionMethod = DriftDetectionMethod.KullbackLeiblerDivergence;
        protected double _driftDetectionThreshold = 0.3;
        protected bool _autoRetrainingEnabled = false;
        protected TimeSpan _retrainingCheckInterval = TimeSpan.FromDays(1);

        /// <summary>
        /// Sets reference data for drift detection
        /// </summary>
        public virtual void SetReferenceData(Matrix<T> referenceData)
        {
            _referenceData = referenceData;
        }

        /// <summary>
        /// Configures monitoring thresholds
        /// </summary>
        public virtual void ConfigureThresholds(MonitoringThresholds thresholds)
        {
            _thresholds = thresholds ?? throw new ArgumentNullException(nameof(thresholds));
        }

        /// <summary>
        /// Registers an alert handler
        /// </summary>
        public virtual void RegisterAlertHandler(Action<MonitoringAlert> handler)
        {
            if (handler != null)
            {
                lock (_lockObject)
                {
                    _alertHandlers.Add(handler);
                }
            }
        }

        /// <summary>
        /// Logs a prediction for monitoring
        /// </summary>
        public virtual async Task LogPredictionAsync(Vector<T> features, T prediction, DateTime? timestamp = null)
        {
            var record = new PredictionRecord
            {
                Features = features,
                Prediction = prediction,
                HasActual = false,
                Timestamp = timestamp ?? DateTime.UtcNow
            };

            lock (_lockObject)
            {
                _predictionHistory.Add(record);
            }

            await Task.CompletedTask;
        }
        
        /// <summary>
        /// Logs a prediction with actual value for monitoring
        /// </summary>
        public virtual async Task LogPredictionAsync(Vector<T> features, T prediction, T actual, DateTime? timestamp = null)
        {
            var record = new PredictionRecord
            {
                Features = features,
                Prediction = prediction,
                Actual = actual,
                HasActual = true,
                Timestamp = timestamp ?? DateTime.UtcNow
            };

            lock (_lockObject)
            {
                _predictionHistory.Add(record);
            }

            // Check if we should calculate performance metrics
            if (_predictionHistory.Count % 100 == 0)
            {
                await UpdatePerformanceMetricsAsync();
            }

            await Task.CompletedTask;
        }

        /// <summary>
        /// Detects data drift
        /// </summary>
        public abstract Task<DriftDetectionResult> DetectDataDriftAsync(Matrix<T> productionData, Matrix<T>? referenceData = null);

        /// <summary>
        /// Detects concept drift
        /// </summary>
        public abstract Task<DriftDetectionResult> DetectConceptDriftAsync(Vector<T> predictions, Vector<T> actuals);

        /// <summary>
        /// Gets performance metrics
        /// </summary>
        public virtual async Task<PerformanceMetrics> GetPerformanceMetricsAsync(DateTime? startDate = null, DateTime? endDate = null)
        {
            List<PredictionRecord> relevantRecords;
            
            lock (_lockObject)
            {
                relevantRecords = _predictionHistory
                    .Where(r => r.HasActual)
                    .Where(r => (!startDate.HasValue || r.Timestamp >= startDate.Value) &&
                               (!endDate.HasValue || r.Timestamp <= endDate.Value))
                    .ToList();
            }

            if (!relevantRecords.Any())
            {
                return new PerformanceMetrics
                {
                    Timestamp = DateTime.UtcNow,
                    PredictionCount = 0
                };
            }

            return await Task.FromResult(CalculatePerformanceMetrics(relevantRecords));
        }

        /// <summary>
        /// Gets model health score
        /// </summary>
        public abstract Task<ModelHealthScore> GetModelHealthScoreAsync();

        /// <summary>
        /// Gets retraining recommendation
        /// </summary>
        public abstract Task<RetrainingRecommendation> GetRetrainingRecommendationAsync();

        /// <summary>
        /// Gets monitoring metrics for a time period
        /// </summary>
        public virtual Task<MonitoringMetricsCollection> GetMonitoringMetricsAsync(DateTime startDate, DateTime endDate)
        {
            return Task.Run(() =>
            {
                lock (_lockObject)
                {
                    var performanceHistory = _performanceHistory
                        .Where(p => p.Timestamp >= startDate && p.Timestamp <= endDate)
                        .ToList();

                    var driftHistory = _driftHistory
                        .Where(d => d.DetectionTimestamp >= startDate && d.DetectionTimestamp <= endDate)
                        .ToList();

                    var relevantPredictions = _predictionHistory
                        .Where(p => p.Timestamp >= startDate && p.Timestamp <= endDate)
                        .ToList();

                    var featureStats = CalculateFeatureStatistics(relevantPredictions);

                    return new MonitoringMetricsCollection
                    {
                        PerformanceHistory = performanceHistory,
                        DriftHistory = driftHistory,
                        FeatureStatistics = featureStats,
                        TotalPredictions = relevantPredictions.Count,
                        StartDate = startDate,
                        EndDate = endDate
                    };
                }
            });
        }

        /// <summary>
        /// Sends an alert to registered handlers
        /// </summary>
        protected virtual void SendAlert(MonitoringAlert alert)
        {
            List<Action<MonitoringAlert>> handlers;
            lock (_lockObject)
            {
                handlers = _alertHandlers.ToList();
            }

            foreach (var handler in handlers)
            {
                try
                {
                    handler(alert);
                }
                catch (Exception ex)
                {
                    // Log error but don't let one handler failure affect others
                    Console.WriteLine($"Alert handler error: {ex.Message}");
                }
            }
        }

        /// <summary>
        /// Updates performance metrics based on recent predictions
        /// </summary>
        protected virtual async Task UpdatePerformanceMetricsAsync()
        {
            var metrics = await GetPerformanceMetricsAsync();
            
            lock (_lockObject)
            {
                _performanceHistory.Add(metrics);
            }

            // Check for performance degradation
            if (_performanceHistory.Count > 1)
            {
                var previousMetrics = _performanceHistory[_performanceHistory.Count - 2];
                var performanceDrop = previousMetrics.Accuracy - metrics.Accuracy;
                
                if (performanceDrop > _thresholds.PerformanceDropThreshold)
                {
                    SendAlert(new MonitoringAlert
                    {
                        AlertType = "PerformanceDegradation",
                        Severity = performanceDrop > _thresholds.PerformanceDropThreshold * 2 ? "Critical" : "Warning",
                        Message = $"Model performance dropped by {performanceDrop:P2}",
                        Timestamp = DateTime.UtcNow,
                        Context = new Dictionary<string, object>
                        {
                            ["CurrentAccuracy"] = metrics.Accuracy,
                            ["PreviousAccuracy"] = previousMetrics.Accuracy,
                            ["PerformanceDrop"] = performanceDrop
                        }
                    });
                }
            }
        }

        /// <summary>
        /// Calculates performance metrics from prediction records
        /// </summary>
        protected virtual PerformanceMetrics CalculatePerformanceMetrics(List<PredictionRecord> records)
        {
            var predictions = records.Select(r => r.Prediction).ToArray();
            var actuals = records.Select(r => r.Actual).ToArray();
            
            // For classification (assuming binary for simplicity)
            var threshold = 0.5;
            var predictedClasses = predictions.Select(p => p >= threshold ? 1.0 : 0.0).ToArray();
            var actualClasses = actuals.Select(a => a >= threshold ? 1.0 : 0.0).ToArray();
            
            var truePositives = predictedClasses.Zip(actualClasses, (p, a) => p == 1.0 && a == 1.0).Count(x => x);
            var falsePositives = predictedClasses.Zip(actualClasses, (p, a) => p == 1.0 && a == 0.0).Count(x => x);
            var falseNegatives = predictedClasses.Zip(actualClasses, (p, a) => p == 0.0 && a == 1.0).Count(x => x);
            var trueNegatives = predictedClasses.Zip(actualClasses, (p, a) => p == 0.0 && a == 0.0).Count(x => x);
            
            var accuracy = (double)(truePositives + trueNegatives) / records.Count;
            var precision = truePositives > 0 ? (double)truePositives / (truePositives + falsePositives) : 0;
            var recall = truePositives > 0 ? (double)truePositives / (truePositives + falseNegatives) : 0;
            var f1Score = precision + recall > 0 ? 2 * (precision * recall) / (precision + recall) : 0;
            
            // Regression metrics
            var mae = predictions.Zip(actuals, (p, a) => Math.Abs(p - a)).Average();
            var mse = predictions.Zip(actuals, (p, a) => Math.Pow(p - a, 2)).Average();
            var rmse = Math.Sqrt(mse);
            
            return new PerformanceMetrics
            {
                Accuracy = accuracy,
                Precision = precision,
                Recall = recall,
                F1Score = f1Score,
                MAE = mae,
                RMSE = rmse,
                Timestamp = DateTime.UtcNow,
                PredictionCount = records.Count,
                CustomMetrics = new Dictionary<string, double>()
            };
        }

        /// <summary>
        /// Calculates feature statistics
        /// </summary>
        protected virtual Dictionary<string, List<double>> CalculateFeatureStatistics(List<PredictionRecord> records)
        {
            var stats = new Dictionary<string, List<double>>();
            
            if (!records.Any()) return stats;
            
            var featureCount = records.First().Features.Length;
            
            for (int i = 0; i < featureCount; i++)
            {
                var featureValues = records.Select(r => r.Features[i]).ToList();
                var mean = featureValues.Average();
                var variance = featureValues.Select(v => Math.Pow(v - mean, 2)).Average();
                var stdDev = Math.Sqrt(variance);
                var min = featureValues.Min();
                var max = featureValues.Max();
                
                stats[$"feature_{i}"] = new List<double> { mean, stdDev, min, max };
            }
            
            return stats;
        }

        /// <summary>
        /// Internal class for storing prediction records
        /// </summary>
        protected class PredictionRecord
        {
            public Vector<T> Features { get; set; } = default!;
            public T Prediction { get; set; } = default!;
            public T Actual { get; set; } = default!;
            public bool HasActual { get; set; }
            public DateTime Timestamp { get; set; }
        }

        #region IProductionMonitor Implementation

        /// <summary>
        /// Configures drift detection settings
        /// </summary>
        public virtual void ConfigureDriftDetection(DriftDetectionMethod method, double threshold)
        {
            _driftDetectionMethod = method;
            _driftDetectionThreshold = threshold;
        }
        
        /// <summary>
        /// Configures automatic retraining settings
        /// </summary>
        public virtual void ConfigureRetraining(bool enabled, TimeSpan checkInterval)
        {
            _autoRetrainingEnabled = enabled;
            _retrainingCheckInterval = checkInterval;
        }

        #endregion
    }
}
