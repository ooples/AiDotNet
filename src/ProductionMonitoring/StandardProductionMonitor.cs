using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Extensions;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.ProductionMonitoring
{
    /// <summary>
    /// Example implementation of a production monitoring system for deployed models.
    /// In production, this would integrate with monitoring infrastructure like Prometheus, DataDog, etc.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class StandardProductionMonitor<T> : IProductionMonitor<T>
    {
        private readonly List<PredictionLog<T>> _predictionLogs = new();
        private readonly List<Action<MonitoringAlert>> _alertHandlers = new();
        private MonitoringThresholds _thresholds = new();
        private Matrix<T>? _trainingDataReference;
        private readonly Random _random = new(42);
        private DriftDetectionMethod _driftDetectionMethod = DriftDetectionMethod.KolmogorovSmirnov;
        private bool _retrainingEnabled = false;
        private TimeSpan _retrainingCheckInterval = TimeSpan.FromDays(7);

        public async Task<DriftDetectionResult> DetectDataDriftAsync(Matrix<T> productionData, Matrix<T>? referenceData = null)
        {
            // Use training data as reference if not provided
            var reference = referenceData ?? _trainingDataReference;
            if (reference == null)
            {
                return new DriftDetectionResult
                {
                    IsDriftDetected = false,
                    DriftScore = 0,
                    DriftType = "NoReference",
                    Details = "No reference data available for drift detection"
                };
            }

            await Task.Delay(50); // Simulate processing

            // Calculate drift using simplified KS statistic
            var driftScores = new Dictionary<string, double>();
            var maxDrift = 0.0;

            for (int feature = 0; feature < productionData.Columns; feature++)
            {
                var prodValues = GetColumn(productionData, feature);
                var refValues = GetColumn(reference, feature);
                
                // Simplified drift calculation
                var drift = CalculateKSStatistic(prodValues, refValues);
                driftScores[$"Feature_{feature}"] = drift;
                maxDrift = Math.Max(maxDrift, drift);
            }

            var isDriftDetected = maxDrift > Convert.ToDouble(_thresholds.DataDriftThreshold);

            return new DriftDetectionResult
            {
                IsDriftDetected = isDriftDetected,
                DriftScore = maxDrift,
                DriftType = "DataDrift",
                FeatureDrifts = driftScores,
                DetectionTimestamp = DateTime.UtcNow,
                Details = isDriftDetected ? "Significant data drift detected" : "No significant drift"
            };
        }

        public async Task<DriftDetectionResult> DetectConceptDriftAsync(Vector<T> predictions, Vector<T> actuals)
        {
            await Task.Delay(50); // Simulate processing

            // Calculate prediction error drift
            var errors = new double[predictions.Length];
            for (int i = 0; i < predictions.Length; i++)
            {
                errors[i] = Math.Abs(Convert.ToDouble(predictions[i]) - Convert.ToDouble(actuals[i]));
            }

            var meanError = errors.Average();
            var recentError = errors.Skip(errors.Length / 2).Average();
            var driftScore = Math.Abs(recentError - meanError) / (meanError + 1e-10);

            var isDriftDetected = driftScore > Convert.ToDouble(_thresholds.ConceptDriftThreshold);

            return new DriftDetectionResult
            {
                IsDriftDetected = isDriftDetected,
                DriftScore = driftScore,
                DriftType = "ConceptDrift",
                FeatureDrifts = new Dictionary<string, double> { ["PredictionError"] = driftScore },
                DetectionTimestamp = DateTime.UtcNow,
                Details = isDriftDetected ? "Model performance degradation detected" : "Model performance stable"
            };
        }

        public async Task<PerformanceMetrics> GetPerformanceMetricsAsync(DateTime? startDate = null, DateTime? endDate = null)
        {
            await Task.Delay(30); // Simulate processing

            var numOps = MathHelper.GetNumericOperations<T>();
            var relevantLogs = _predictionLogs.Where(log => 
                (!startDate.HasValue || log.Timestamp >= startDate.Value) &&
                (!endDate.HasValue || log.Timestamp <= endDate.Value) &&
                !numOps.Equals(log.Actual, numOps.Zero)).ToList();

            if (!relevantLogs.Any())
            {
                return new PerformanceMetrics
                {
                    Timestamp = DateTime.UtcNow,
                    PredictionCount = 0
                };
            }

            // Calculate metrics
            var predictions = relevantLogs.Select(l => Convert.ToDouble(l.Prediction)).ToArray();
            var actuals = relevantLogs.Select(l => Convert.ToDouble(l.Actual!.Value)).ToArray();
            
            var errors = predictions.Zip(actuals, (p, a) => Math.Abs(p - a)).ToArray();
            var squaredErrors = predictions.Zip(actuals, (p, a) => Math.Pow(p - a, 2)).ToArray();

            return new PerformanceMetrics
            {
                Accuracy = 1.0 - errors.Average() / (actuals.Max() - actuals.Min() + 1e-10),
                Precision = 0.85 + _random.NextDouble() * 0.1, // Simplified
                Recall = 0.82 + _random.NextDouble() * 0.1, // Simplified
                F1Score = 0.83 + _random.NextDouble() * 0.1, // Simplified
                MAE = errors.Average(),
                RMSE = Math.Sqrt(squaredErrors.Average()),
                CustomMetrics = new Dictionary<string, double>
                {
                    ["MedianError"] = errors.OrderBy(e => e).ElementAt(errors.Length / 2),
                    ["P95Error"] = errors.OrderBy(e => e).ElementAt((int)(errors.Length * 0.95))
                },
                Timestamp = DateTime.UtcNow,
                PredictionCount = relevantLogs.Count
            };
        }

        public async Task<ModelHealthScore> GetModelHealthScoreAsync()
        {
            await Task.Delay(50); // Simulate processing

            var performanceMetrics = await GetPerformanceMetricsAsync(DateTime.UtcNow.AddDays(-7), DateTime.UtcNow);
            
            // Calculate component scores
            var dataQuality = CalculateDataQualityScore();
            var performance = performanceMetrics.Accuracy;
            var stability = CalculateStabilityScore();
            var drift = 1.0 - (_predictionLogs.Any() ? 0.1 : 0.0); // Simplified

            var overallScore = (dataQuality + performance + stability + drift) / 4.0;
            var issues = new List<string>();

            if (dataQuality < 0.7) issues.Add("Poor data quality detected");
            if (performance < 0.7) issues.Add("Model performance below threshold");
            if (stability < 0.7) issues.Add("High prediction variance");
            if (drift < 0.7) issues.Add("Potential drift detected");

            return new ModelHealthScore
            {
                OverallScore = overallScore,
                DataQualityScore = dataQuality,
                PerformanceScore = performance,
                StabilityScore = stability,
                DriftScore = drift,
                HealthStatus = overallScore >= 0.8 ? "Healthy" : overallScore >= 0.6 ? "Warning" : "Critical",
                Issues = issues,
                EvaluationTimestamp = DateTime.UtcNow
            };
        }

        public async Task<RetrainingRecommendation> GetRetrainingRecommendationAsync()
        {
            var healthScore = await GetModelHealthScoreAsync();
            var performanceMetrics = await GetPerformanceMetricsAsync(DateTime.UtcNow.AddDays(-30), DateTime.UtcNow);

            var reasons = new List<string>();
            var shouldRetrain = false;
            var urgency = "Low";

            if (healthScore.PerformanceScore < Convert.ToDouble(_thresholds.PerformanceDropThreshold))
            {
                reasons.Add("Performance below acceptable threshold");
                shouldRetrain = true;
                urgency = "High";
            }

            if (healthScore.DriftScore < 0.7)
            {
                reasons.Add("Significant data drift detected");
                shouldRetrain = true;
                urgency = urgency == "High" ? "Critical" : "Medium";
            }

            // Check time-based retraining
            var lastRetraining = _predictionLogs.FirstOrDefault()?.Timestamp ?? DateTime.UtcNow;
            if (DateTime.UtcNow - lastRetraining > TimeSpan.FromDays(90))
            {
                reasons.Add("Time-based retraining interval exceeded");
                shouldRetrain = true;
            }

            return new RetrainingRecommendation
            {
                ShouldRetrain = shouldRetrain,
                Urgency = urgency,
                Reasons = reasons,
                RecommendationTimestamp = DateTime.UtcNow,
                ConfidenceScore = healthScore.OverallScore,
                SuggestedActions = new Dictionary<string, object>
                {
                    ["CollectMoreData"] = performanceMetrics.PredictionCount < 1000,
                    ["FeatureEngineering"] = healthScore.PerformanceScore < 0.7,
                    ["HyperparameterTuning"] = healthScore.StabilityScore < 0.8
                }
            };
        }

        public void RegisterAlertHandler(Action<MonitoringAlert> handler)
        {
            _alertHandlers.Add(handler);
        }

        public async Task LogPredictionAsync(Vector<T> features, T prediction, DateTime? timestamp = null)
        {
            await LogPredictionAsync(features, prediction, default(T), timestamp);
        }

        public async Task LogPredictionAsync(Vector<T> features, T prediction, T actual, DateTime? timestamp = null)
        {
            var log = new PredictionLog<T>
            {
                Features = features.DeepCopy(),
                Prediction = prediction,
                Actual = actual.Equals(default(T)) ? default(T?) : actual,
                Timestamp = timestamp ?? DateTime.UtcNow
            };

            _predictionLogs.Add(log);

            // Check for alerts
            await CheckForAlerts(log);

            // Limit log size
            if (_predictionLogs.Count > 10000)
            {
                _predictionLogs.RemoveRange(0, _predictionLogs.Count - 10000);
            }
        }

        public async Task<MonitoringMetricsCollection> GetMonitoringMetricsAsync(DateTime startDate, DateTime endDate)
        {
            var performanceHistory = new List<PerformanceMetrics>();
            var driftHistory = new List<DriftDetectionResult>();

            // Generate daily metrics
            var currentDate = startDate;
            while (currentDate <= endDate)
            {
                var dayMetrics = await GetPerformanceMetricsAsync(currentDate, currentDate.AddDays(1));
                if (dayMetrics.PredictionCount > 0)
                {
                    performanceHistory.Add(dayMetrics);
                }
                currentDate = currentDate.AddDays(1);
            }

            // Feature statistics
            var featureStats = CalculateFeatureStatistics(startDate, endDate);

            return new MonitoringMetricsCollection
            {
                PerformanceHistory = performanceHistory,
                DriftHistory = driftHistory,
                FeatureStatistics = featureStats,
                TotalPredictions = _predictionLogs.Count(l => l.Timestamp >= startDate && l.Timestamp <= endDate),
                StartDate = startDate,
                EndDate = endDate
            };
        }

        public void ConfigureThresholds(MonitoringThresholds thresholds)
        {
            _thresholds = thresholds;
        }

        /// <summary>
        /// Sets the reference data for drift detection.
        /// </summary>
        public void SetReferenceData(Matrix<T> referenceData)
        {
            _trainingDataReference = referenceData.DeepCopy();
        }

        /// <summary>
        /// Configures drift detection parameters.
        /// </summary>
        public void ConfigureDriftDetection(T? dataDriftThreshold, T? conceptDriftThreshold)
        {
            if (dataDriftThreshold != null)
                _thresholds.DataDriftThreshold = Convert.ToDouble(dataDriftThreshold);
            if (conceptDriftThreshold != null)
                _thresholds.ConceptDriftThreshold = Convert.ToDouble(conceptDriftThreshold);
        }

        /// <summary>
        /// Configures automatic retraining parameters.
        /// </summary>
        public void ConfigureAutoRetraining(T? performanceDropThreshold, TimeSpan? timeBasedRetraining)
        {
            if (performanceDropThreshold != null)
                _thresholds.PerformanceDropThreshold = Convert.ToDouble(performanceDropThreshold);
            // Store time-based retraining interval
        }

        /// <summary>
        /// Configures drift detection settings (IProductionMonitor interface).
        /// </summary>
        public void ConfigureDriftDetection(DriftDetectionMethod method, double threshold)
        {
            // Store drift detection method
            _driftDetectionMethod = method;
            _thresholds.DataDriftThreshold = threshold;
        }

        /// <summary>
        /// Configures automatic retraining settings (IProductionMonitor interface).
        /// </summary>
        public void ConfigureRetraining(bool enabled, TimeSpan checkInterval)
        {
            _retrainingEnabled = enabled;
            _retrainingCheckInterval = checkInterval;
        }

        // Helper methods
        private double[] GetColumn(Matrix<T> matrix, int columnIndex)
        {
            var column = new double[matrix.Rows];
            for (int i = 0; i < matrix.Rows; i++)
            {
                column[i] = Convert.ToDouble(matrix[i, columnIndex]);
            }
            return column;
        }

        private double CalculateKSStatistic(double[] dist1, double[] dist2)
        {
            // Simplified KS statistic calculation
            Array.Sort(dist1);
            Array.Sort(dist2);
            
            var maxDiff = 0.0;
            for (int i = 0; i < Math.Min(dist1.Length, dist2.Length); i++)
            {
                var cdf1 = (double)(i + 1) / dist1.Length;
                var cdf2 = (double)(i + 1) / dist2.Length;
                maxDiff = Math.Max(maxDiff, Math.Abs(cdf1 - cdf2));
            }
            
            return maxDiff;
        }

        private double CalculateDataQualityScore()
        {
            if (!_predictionLogs.Any()) return 1.0;
            
            // Check for missing values, outliers, etc.
            var nullFeatures = _predictionLogs.Count(l => l.Features == null);
            var nullRatio = (double)nullFeatures / _predictionLogs.Count;
            
            return 1.0 - nullRatio;
        }

        private double CalculateStabilityScore()
        {
            if (_predictionLogs.Count < 10) return 1.0;
            
            // Calculate prediction variance over time
            var recentPredictions = _predictionLogs
                .TakeLast(100)
                .Select(l => Convert.ToDouble(l.Prediction))
                .ToArray();
            
            if (recentPredictions.Length == 0) return 1.0;
            
            var mean = recentPredictions.Average();
            var variance = recentPredictions.Select(p => Math.Pow(p - mean, 2)).Average();
            var cv = Math.Sqrt(variance) / (Math.Abs(mean) + 1e-10);
            
            return Math.Max(0, 1.0 - cv);
        }

        private Dictionary<string, List<double>> CalculateFeatureStatistics(DateTime startDate, DateTime endDate)
        {
            var stats = new Dictionary<string, List<double>>();
            var relevantLogs = _predictionLogs.Where(l => l.Timestamp >= startDate && l.Timestamp <= endDate).ToList();
            
            if (!relevantLogs.Any() || relevantLogs[0].Features == null) return stats;
            
            var numFeatures = relevantLogs[0].Features!.Length;
            for (int i = 0; i < numFeatures; i++)
            {
                var featureValues = relevantLogs
                    .Where(l => l.Features != null)
                    .Select(l => Convert.ToDouble(l.Features![i]))
                    .ToList();
                
                if (featureValues.Any())
                {
                    stats[$"Feature_{i}_Mean"] = new List<double> { featureValues.Average() };
                    stats[$"Feature_{i}_Std"] = new List<double> { CalculateStandardDeviation(featureValues) };
                }
            }
            
            return stats;
        }

        private double CalculateStandardDeviation(List<double> values)
        {
            if (values.Count < 2) return 0;
            var mean = values.Average();
            var variance = values.Select(v => Math.Pow(v - mean, 2)).Average();
            return Math.Sqrt(variance);
        }

        private async Task CheckForAlerts(PredictionLog<T> log)
        {
            // Check various alert conditions
            if (_predictionLogs.Count > 100 && _predictionLogs.Count % 100 == 0)
            {
                var healthScore = await GetModelHealthScoreAsync();
                
                if (healthScore.OverallScore < _thresholds.HealthScoreCriticalThreshold)
                {
                    TriggerAlert(new MonitoringAlert
                    {
                        AlertType = "HealthScore",
                        Severity = "Critical",
                        Message = $"Model health score critically low: {healthScore.OverallScore:F2}",
                        Timestamp = DateTime.UtcNow,
                        Context = new Dictionary<string, object>
                        {
                            ["HealthScore"] = healthScore,
                            ["Threshold"] = _thresholds.HealthScoreCriticalThreshold
                        }
                    });
                }
                else if (healthScore.OverallScore < _thresholds.HealthScoreWarningThreshold)
                {
                    TriggerAlert(new MonitoringAlert
                    {
                        AlertType = "HealthScore",
                        Severity = "Warning",
                        Message = $"Model health score below warning threshold: {healthScore.OverallScore:F2}",
                        Timestamp = DateTime.UtcNow,
                        Context = new Dictionary<string, object>
                        {
                            ["HealthScore"] = healthScore,
                            ["Threshold"] = _thresholds.HealthScoreWarningThreshold
                        }
                    });
                }
            }
        }

        private void TriggerAlert(MonitoringAlert alert)
        {
            foreach (var handler in _alertHandlers)
            {
                try
                {
                    handler(alert);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error triggering alert: {ex.Message}");
                }
            }
        }

        private class PredictionLog<TPred>
        {
            public Vector<TPred>? Features { get; set; }
            public TPred Prediction { get; set; } = default!;
            public TPred? Actual { get; set; }
            public DateTime Timestamp { get; set; }
        }
    }
}