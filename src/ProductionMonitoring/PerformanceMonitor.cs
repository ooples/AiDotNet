using AiDotNet.Extensions;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AiDotNet.ProductionMonitoring
{
    /// <summary>
    /// Monitors and tracks model performance metrics over time
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations</typeparam>
    public class PerformanceMonitor<T> : ProductionMonitorBase<T>
    {
        private readonly TimeSpan _aggregationInterval = default!;
        private readonly Dictionary<string, List<double>> _customMetrics = default!;
        private readonly Dictionary<DateTime, AggregatedMetrics> _aggregatedMetrics = default!;
        private readonly int _maxHistoryDays;
        private DateTime _lastAggregationTime = default!;

        public PerformanceMonitor(TimeSpan? aggregationInterval = null, int maxHistoryDays = 90)
        {
            _aggregationInterval = aggregationInterval ?? TimeSpan.FromHours(1);
            _customMetrics = new Dictionary<string, List<double>>();
            _aggregatedMetrics = new Dictionary<DateTime, AggregatedMetrics>();
            _maxHistoryDays = maxHistoryDays;
            _lastAggregationTime = DateTime.UtcNow;
        }

        /// <summary>
        /// Adds a custom metric to track
        /// </summary>
        public void RegisterCustomMetric(string metricName)
        {
            lock (_lockObject)
            {
                if (!_customMetrics.ContainsKey(metricName))
                {
                    _customMetrics[metricName] = new List<double>();
                }
            }
        }

        /// <summary>
        /// Logs a custom metric value
        /// </summary>
        public async Task LogCustomMetricAsync(string metricName, double value, DateTime? timestamp = null)
        {
            lock (_lockObject)
            {
                if (!_customMetrics.ContainsKey(metricName))
                {
                    _customMetrics[metricName] = new List<double>();
                }
                _customMetrics[metricName].Add(value);
            }

            await CheckAggregationAsync();
        }

        /// <summary>
        /// Logs a prediction with additional metadata
        /// </summary>
        public override async Task LogPredictionAsync(Vector<T> features, T prediction, DateTime? timestamp = null)
        {
            await base.LogPredictionAsync(features, prediction, timestamp);
            await CheckAggregationAsync();
        }
        
        /// <summary>
        /// Logs a prediction with actual value and additional metadata
        /// </summary>
        public override async Task LogPredictionAsync(Vector<T> features, T prediction, T actual, DateTime? timestamp = null)
        {
            await base.LogPredictionAsync(features, prediction, actual, timestamp);
            await CheckAggregationAsync();
        }

        /// <summary>
        /// Gets performance metrics for a specific time period
        /// </summary>
        public override async Task<PerformanceMetrics> GetPerformanceMetricsAsync(DateTime? startDate = null, DateTime? endDate = null)
        {
            var baseMetrics = await base.GetPerformanceMetricsAsync(startDate, endDate);
            
            // Add custom metrics
            lock (_lockObject)
            {
                foreach (var kvp in _customMetrics)
                {
                    if (kvp.Value.Any())
                    {
                        baseMetrics.CustomMetrics[kvp.Key] = kvp.Value.Average();
                    }
                }
            }
            
            return baseMetrics;
        }

        /// <summary>
        /// Gets aggregated performance metrics over time
        /// </summary>
        public async Task<List<PerformanceMetrics>> GetAggregatedMetricsAsync(DateTime startDate, DateTime endDate, TimeSpan? interval = null)
        {
            interval = interval ?? _aggregationInterval;
            var metrics = new List<PerformanceMetrics>();
            
            lock (_lockObject)
            {
                var relevantMetrics = _aggregatedMetrics
                    .Where(kvp => kvp.Key >= startDate && kvp.Key <= endDate)
                    .OrderBy(kvp => kvp.Key)
                    .ToList();
                
                foreach (var kvp in relevantMetrics)
                {
                    metrics.Add(ConvertToPerformanceMetrics(kvp.Value, kvp.Key));
                }
            }
            
            return await Task.FromResult(metrics);
        }

        /// <summary>
        /// Gets performance trends and anomalies
        /// </summary>
        public async Task<PerformanceTrends> GetPerformanceTrendsAsync(int lookbackDays = 7)
        {
            var endDate = DateTime.UtcNow;
            var startDate = endDate.AddDays(-lookbackDays);
            
            var metrics = await GetAggregatedMetricsAsync(startDate, endDate);
            
            if (!metrics.Any())
            {
                return new PerformanceTrends
                {
                    TrendDirection = "Stable",
                    TrendStrength = 0,
                    Anomalies = new List<PerformanceAnomaly>()
                };
            }
            
            // Calculate trends
            var accuracyTrend = CalculateTrend(metrics.Select(m => m.Accuracy).ToList());
            var f1Trend = CalculateTrend(metrics.Select(m => m.F1Score).ToList());
            var maeTrend = CalculateTrend(metrics.Select(m => m.MAE).ToList());
            
            // Detect anomalies
            var anomalies = DetectAnomalies(metrics);
            
            // Overall trend
            var overallTrend = (accuracyTrend + f1Trend - maeTrend) / 3.0;
            var trendDirection = overallTrend > 0.1 ? "Improving" :
                                overallTrend < -0.1 ? "Degrading" : "Stable";
            
            return new PerformanceTrends
            {
                TrendDirection = trendDirection,
                TrendStrength = Math.Abs(overallTrend),
                AccuracyTrend = accuracyTrend,
                F1ScoreTrend = f1Trend,
                MAETrend = maeTrend,
                Anomalies = anomalies,
                AnalysisTimestamp = DateTime.UtcNow
            };
        }

        /// <summary>
        /// Detects data drift
        /// </summary>
        public override async Task<DriftDetectionResult> DetectDataDriftAsync(Matrix<T> productionData, Matrix<T>? referenceData = null)
        {
            // Performance monitor focuses on metrics, not data drift
            return Task.FromResult(new DriftDetectionResult
            {
                IsDriftDetected = false,
                DriftScore = 0,
                DriftType = "DataDrift",
                Details = "Use DataDriftDetector for data drift detection",
                DetectionTimestamp = DateTime.UtcNow
            });
        }

        /// <summary>
        /// Detects concept drift based on performance degradation
        /// </summary>
        public override async Task<DriftDetectionResult> DetectConceptDriftAsync(Vector<T> predictions, Vector<T> actuals)
        {
            // Simple performance-based concept drift detection
            var currentMetrics = CalculatePerformanceMetrics(
                predictions.Zip(actuals, (p, a) => new PredictionRecord 
                { 
                    Prediction = p, 
                    Actual = a,
                    Timestamp = DateTime.UtcNow,
                    Features = new double[0]
                }).ToList()
            );
            
            var historicalMetrics = await GetPerformanceMetricsAsync(
                DateTime.UtcNow.AddDays(-7), 
                DateTime.UtcNow.AddHours(-1)
            );
            
            var performanceDrop = historicalMetrics.Accuracy - currentMetrics.Accuracy;
            var isDrift = performanceDrop > _thresholds.PerformanceDropThreshold;
            
            return new DriftDetectionResult
            {
                IsDriftDetected = isDrift,
                DriftScore = performanceDrop,
                DriftType = "ConceptDrift",
                Details = $"Performance-based drift detection: Accuracy drop = {performanceDrop:F4}",
                DetectionTimestamp = DateTime.UtcNow
            };
        }

        /// <summary>
        /// Gets model health score based on performance metrics
        /// </summary>
        public override async Task<ModelHealthScore> GetModelHealthScoreAsync()
        {
            var trends = await GetPerformanceTrendsAsync(7);
            var recentMetrics = await GetPerformanceMetricsAsync(DateTime.UtcNow.AddDays(-1), DateTime.UtcNow);
            var weekMetrics = await GetPerformanceMetricsAsync(DateTime.UtcNow.AddDays(-7), DateTime.UtcNow);
            
            // Calculate scores
            var performanceScore = CalculatePerformanceHealthScore(recentMetrics);
            var stabilityScore = CalculateStabilityScore(trends);
            var dataQualityScore = await CalculateDataQualityScoreAsync();
            
            // Overall score
            var overallScore = (performanceScore * 0.5 + stabilityScore * 0.3 + dataQualityScore * 0.2);
            
            var healthStatus = overallScore >= 0.8 ? "Healthy" :
                              overallScore >= 0.6 ? "Warning" : "Critical";
            
            var issues = new List<string>();
            if (performanceScore < 0.7) issues.Add("Poor performance metrics");
            if (stabilityScore < 0.7) issues.Add("Unstable performance");
            if (dataQualityScore < 0.7) issues.Add("Data quality issues");
            if (trends.Anomalies.Any()) issues.Add($"{trends.Anomalies.Count} anomalies detected");
            
            return new ModelHealthScore
            {
                OverallScore = overallScore,
                DataQualityScore = dataQualityScore,
                PerformanceScore = performanceScore,
                StabilityScore = stabilityScore,
                DriftScore = 1.0, // Not applicable for performance monitor
                HealthStatus = healthStatus,
                Issues = issues,
                EvaluationTimestamp = DateTime.UtcNow
            };
        }

        /// <summary>
        /// Gets retraining recommendation based on performance
        /// </summary>
        public override async Task<RetrainingRecommendation> GetRetrainingRecommendationAsync()
        {
            var healthScore = await GetModelHealthScoreAsync();
            var trends = await GetPerformanceTrendsAsync(30);
            var recentMetrics = await GetPerformanceMetricsAsync(DateTime.UtcNow.AddDays(-7), DateTime.UtcNow);
            
            var reasons = new List<string>();
            var shouldRetrain = false;
            var urgency = "Low";
            
            // Check performance degradation
            if (trends.TrendDirection == "Degrading" && trends.TrendStrength > 0.2)
            {
                shouldRetrain = true;
                urgency = trends.TrendStrength > 0.5 ? "High" : "Medium";
                reasons.Add($"Performance degrading with strength {trends.TrendStrength:F2}");
            }
            
            // Check absolute performance
            if (recentMetrics.Accuracy < 0.7 || recentMetrics.F1Score < 0.65)
            {
                shouldRetrain = true;
                urgency = "High";
                reasons.Add($"Poor performance: Accuracy={recentMetrics.Accuracy:F2}, F1={recentMetrics.F1Score:F2}");
            }
            
            // Check anomalies
            if (trends.Anomalies.Count > 5)
            {
                shouldRetrain = true;
                if (urgency == "Low") urgency = "Medium";
                reasons.Add($"Frequent anomalies: {trends.Anomalies.Count} in last 30 days");
            }
            
            // Check health score
            if (healthScore.OverallScore < 0.6)
            {
                shouldRetrain = true;
                if (urgency != "High") urgency = "High";
                reasons.Add($"Poor model health: {healthScore.OverallScore:F2}");
            }
            
            var confidence = CalculateRetrainingConfidenceFromPerformance(healthScore, trends, recentMetrics);
            
            return new RetrainingRecommendation
            {
                ShouldRetrain = shouldRetrain,
                Urgency = urgency,
                Reasons = reasons,
                RecommendationTimestamp = DateTime.UtcNow,
                ConfidenceScore = confidence,
                SuggestedActions = new Dictionary<string, object>
                {
                    ["FeatureEngineering"] = recentMetrics.Accuracy < 0.75,
                    ["HyperparameterTuning"] = trends.TrendDirection == "Degrading",
                    ["IncreaseTrainingData"] = recentMetrics.PredictionCount < 1000,
                    ["EnsembleMethods"] = healthScore.StabilityScore < 0.7
                }
            };
        }

        // Private methods

        private async Task CheckAggregationAsync()
        {
            var now = DateTime.UtcNow;
            if (now - _lastAggregationTime >= _aggregationInterval)
            {
                await AggregateMetricsAsync();
                _lastAggregationTime = now;
                await CleanupOldDataAsync();
            }
        }

        private async Task AggregateMetricsAsync()
        {
            var aggregationTime = DateTime.UtcNow.Truncate(_aggregationInterval);
            
            lock (_lockObject)
            {
                var recentPredictions = _predictionHistory
                    .Where(p => p.Timestamp >= _lastAggregationTime && p.Timestamp < aggregationTime)
                    .ToList();
                
                if (!recentPredictions.Any()) return;
                
                var metrics = CalculatePerformanceMetrics(recentPredictions.Where(p => p.HasActual).ToList());
                
                var aggregated = new AggregatedMetrics
                {
                    Timestamp = aggregationTime,
                    Accuracy = metrics.Accuracy,
                    Precision = metrics.Precision,
                    Recall = metrics.Recall,
                    F1Score = metrics.F1Score,
                    MAE = metrics.MAE,
                    RMSE = metrics.RMSE,
                    PredictionCount = recentPredictions.Count,
                    ActualCount = recentPredictions.Count(p => p.HasActual),
                    CustomMetrics = new Dictionary<string, double>()
                };
                
                // Aggregate custom metrics
                foreach (var kvp in _customMetrics)
                {
                    if (kvp.Value.Any())
                    {
                        aggregated.CustomMetrics[kvp.Key] = kvp.Value.Average();
                        kvp.Value.Clear(); // Clear after aggregation
                    }
                }
                
                _aggregatedMetrics[aggregationTime] = aggregated;
            }
            
            await Task.CompletedTask;
        }

        private async Task CleanupOldDataAsync()
        {
            var cutoffDate = DateTime.UtcNow.AddDays(-_maxHistoryDays);
            
            lock (_lockObject)
            {
                // Clean prediction history
                _predictionHistory.RemoveAll(p => p.Timestamp < cutoffDate);
                
                // Clean aggregated metrics
                var oldKeys = _aggregatedMetrics.Keys.Where(k => k < cutoffDate).ToList();
                foreach (var key in oldKeys)
                {
                    _aggregatedMetrics.Remove(key);
                }
                
                // Clean performance history
                _performanceHistory.RemoveAll(p => p.Timestamp < cutoffDate);
                
                // Clean drift history
                _driftHistory.RemoveAll(d => d.DetectionTimestamp < cutoffDate);
            }
            
            await Task.CompletedTask;
        }

        private PerformanceMetrics ConvertToPerformanceMetrics(AggregatedMetrics aggregated, DateTime timestamp)
        {
            return new PerformanceMetrics
            {
                Accuracy = aggregated.Accuracy,
                Precision = aggregated.Precision,
                Recall = aggregated.Recall,
                F1Score = aggregated.F1Score,
                MAE = aggregated.MAE,
                RMSE = aggregated.RMSE,
                Timestamp = timestamp,
                PredictionCount = aggregated.PredictionCount,
                CustomMetrics = new Dictionary<string, double>(aggregated.CustomMetrics)
            };
        }

        private double CalculateTrend(List<double> values)
        {
            if (values.Count < 2) return 0;
            
            // Simple linear regression
            var n = values.Count;
            var xValues = Enumerable.Range(0, n).Select(i => (double)i).ToArray();
            var xMean = xValues.Average();
            var yMean = values.Average();
            
            var numerator = xValues.Zip(values, (x, y) => (x - xMean) * (y - yMean)).Sum();
            var denominator = xValues.Select(x => Math.Pow(x - xMean, 2)).Sum();
            
            return denominator == 0 ? 0 : numerator / denominator;
        }

        private List<PerformanceAnomaly> DetectAnomalies(List<PerformanceMetrics> metrics)
        {
            var anomalies = new List<PerformanceAnomaly>();
            
            if (metrics.Count < 10) return anomalies;
            
            // Use statistical methods to detect anomalies
            var accuracies = metrics.Select(m => m.Accuracy).ToList();
            var mean = accuracies.Average();
            var stdDev = Math.Sqrt(accuracies.Select(a => Math.Pow(a - mean, 2)).Average());
            
            for (int i = 0; i < metrics.Count; i++)
            {
                var zScore = Math.Abs((metrics[i].Accuracy - mean) / stdDev);
                
                if (zScore > 3) // 3 standard deviations
                {
                    anomalies.Add(new PerformanceAnomaly
                    {
                        Timestamp = metrics[i].Timestamp,
                        MetricName = "Accuracy",
                        Value = metrics[i].Accuracy,
                        ExpectedValue = mean,
                        AnomalyScore = zScore,
                        Type = metrics[i].Accuracy < mean ? "Drop" : "Spike"
                    });
                }
            }
            
            return anomalies;
        }

        private double CalculatePerformanceHealthScore(PerformanceMetrics metrics)
        {
            if (metrics.PredictionCount == 0) return 0.5;
            
            // Weighted combination of metrics
            var accuracyScore = metrics.Accuracy;
            var f1Score = metrics.F1Score;
            var errorScore = 1.0 - Math.Min(metrics.MAE / 2.0, 1.0); // Normalize MAE
            
            return (accuracyScore * 0.4 + f1Score * 0.4 + errorScore * 0.2);
        }

        private double CalculateStabilityScore(PerformanceTrends trends)
        {
            // Lower scores for degrading trends and anomalies
            var trendPenalty = trends.TrendDirection == "Degrading" ? trends.TrendStrength : 0;
            var anomalyPenalty = Math.Min(trends.Anomalies.Count * 0.1, 0.5);
            
            return Math.Max(0, 1.0 - trendPenalty - anomalyPenalty);
        }

        private Task<double> CalculateDataQualityScoreAsync()
        {
            lock (_lockObject)
            {
                if (!_predictionHistory.Any()) return Task.FromResult(1.0);

                var recent = _predictionHistory.TakeLast(1000).ToList();

                // Check for various data quality issues
                var missingFeatures = recent.Count(p => p.Features.Any(f => double.IsNaN(f) || double.IsInfinity(f)));
                var extremeValues = recent.Count(p => p.Features.Any(f => Math.Abs(f) > 1e6));
                var zeroVariance = CheckZeroVarianceFeatures(recent);

                var missingRatio = missingFeatures / (double)recent.Count();
                var extremeRatio = extremeValues / (double)recent.Count();

                return Task.FromResult(1.0 - (missingRatio * 0.4 + extremeRatio * 0.3 + zeroVariance * 0.3));
            }
        }

        private double CheckZeroVarianceFeatures(List<PredictionRecord> records)
        {
            if (!records.Any()) return 0;
            
            var featureCount = records.First().Features.Length;
            var zeroVarianceCount = 0;
            
            for (int i = 0; i < featureCount; i++)
            {
                var values = records.Select(r => r.Features[i]).ToArray();
                var variance = values.Distinct().Count() == 1 ? 0 : 1;
                if (variance == 0) zeroVarianceCount++;
            }
            
            return zeroVarianceCount / (double)featureCount;
        }

        private double CalculateRetrainingConfidenceFromPerformance(
            ModelHealthScore healthScore, 
            PerformanceTrends trends, 
            PerformanceMetrics metrics)
        {
            var healthPenalty = 1.0 - healthScore.OverallScore;
            var trendPenalty = trends.TrendDirection == "Degrading" ? trends.TrendStrength : 0;
            var performancePenalty = 1.0 - metrics.Accuracy;
            var anomalyPenalty = Math.Min(trends.Anomalies.Count * 0.05, 0.3);
            
            return Math.Min(1.0, healthPenalty * 0.3 + trendPenalty * 0.3 + performancePenalty * 0.3 + anomalyPenalty * 0.1);
        }

        // Helper classes

        private class AggregatedMetrics
        {
            public DateTime Timestamp { get; set; }
            public double Accuracy { get; set; }
            public double Precision { get; set; }
            public double Recall { get; set; }
            public double F1Score { get; set; }
            public double MAE { get; set; }
            public double RMSE { get; set; }
            public int PredictionCount { get; set; }
            public int ActualCount { get; set; }
            public Dictionary<string, double> CustomMetrics { get; set; } = new();
        }

        public class PerformanceTrends
        {
            public string TrendDirection { get; set; } = string.Empty;
            public double TrendStrength { get; set; }
            public double AccuracyTrend { get; set; }
            public double F1ScoreTrend { get; set; }
            public double MAETrend { get; set; }
            public List<PerformanceAnomaly> Anomalies { get; set; } = new();
            public DateTime AnalysisTimestamp { get; set; }
        }

        public class PerformanceAnomaly
        {
            public DateTime Timestamp { get; set; }
            public string MetricName { get; set; } = string.Empty;
            public double Value { get; set; }
            public double ExpectedValue { get; set; }
            public double AnomalyScore { get; set; }
            public string Type { get; set; } = string.Empty;
        }
    }

    // Extension method for DateTime truncation
    public static class DateTimeExtensions
    {
        public static DateTime Truncate(this DateTime dateTime, TimeSpan timeSpan)
        {
            if (timeSpan == TimeSpan.Zero) return dateTime;
            return dateTime.AddTicks(-(dateTime.Ticks % timeSpan.Ticks));
        }
    }
}
