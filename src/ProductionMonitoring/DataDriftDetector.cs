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
    /// Detects data drift between training and production data distributions
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations</typeparam>
    public class DataDriftDetector<T> : ProductionMonitorBase<T>
    {
        private readonly DriftDetectionMethod _method = default!;
        private readonly int _windowSize;
        private readonly Dictionary<int, FeatureStatistics> _baselineStats = default!;

        public enum DriftDetectionMethod
        {
            KolmogorovSmirnov,
            ChiSquared,
            JensenShannon,
            WassersteinDistance,
            PopulationStabilityIndex,
            KullbackLeibler
        }

        public DataDriftDetector(DriftDetectionMethod method = DriftDetectionMethod.KolmogorovSmirnov, int windowSize = 1000)
        {
            _method = method;
            _windowSize = windowSize;
            _baselineStats = new Dictionary<int, FeatureStatistics>();
        }

        /// <summary>
        /// Detects data drift between production and reference data
        /// </summary>
        public override async Task<DriftDetectionResult> DetectDataDriftAsync(Matrix<T> productionData, Matrix<T>? referenceData = null)
        {
            referenceData = referenceData ?? _referenceData;
            
            if (referenceData == null)
            {
                throw new InvalidOperationException("Reference data must be provided either in the method call or via SetReferenceData");
            }

            if (productionData.Rows < _thresholds.MinimumSamplesForDriftDetection)
            {
                return new DriftDetectionResult
                {
                    IsDriftDetected = false,
                    DriftScore = 0,
                    DriftType = "DataDrift",
                    Details = $"Insufficient samples for drift detection. Minimum required: {_thresholds.MinimumSamplesForDriftDetection}",
                    DetectionTimestamp = DateTime.UtcNow
                };
            }

            var featureDrifts = new Dictionary<string, double>();
            var featureCount = productionData.GetLength(1);
            double maxDrift = 0;

            // Calculate drift for each feature
            for (int featureIndex = 0; featureIndex < featureCount; featureIndex++)
            {
                var productionFeature = ExtractFeatureColumn(productionData, featureIndex);
                var referenceFeature = ExtractFeatureColumn(referenceData, featureIndex);

                double driftScore = await CalculateFeatureDriftAsync(productionFeature, referenceFeature);
                featureDrifts[$"feature_{featureIndex}"] = driftScore;
                maxDrift = Math.Max(maxDrift, driftScore);
            }

            var isDriftDetected = maxDrift > _thresholds.DataDriftThreshold;
            
            var result = new DriftDetectionResult
            {
                IsDriftDetected = isDriftDetected,
                DriftScore = maxDrift,
                DriftType = "DataDrift",
                FeatureDrifts = featureDrifts,
                DetectionTimestamp = DateTime.UtcNow,
                Details = $"Maximum drift score: {maxDrift:F4} (threshold: {_thresholds.DataDriftThreshold})"
            };

            lock (_lockObject)
            {
                _driftHistory.Add(result);
            }

            if (isDriftDetected)
            {
                var driftedFeatures = featureDrifts.Where(kv => kv.Value > _thresholds.DataDriftThreshold).ToList();
                
                SendAlert(new MonitoringAlert
                {
                    AlertType = "DataDrift",
                    Severity = maxDrift > _thresholds.DataDriftThreshold * 2 ? "Critical" : "Warning",
                    Message = $"Data drift detected in {driftedFeatures.Count} features",
                    Timestamp = DateTime.UtcNow,
                    Context = new Dictionary<string, object>
                    {
                        ["MaxDriftScore"] = maxDrift,
                        ["DriftedFeatures"] = driftedFeatures.Select(kv => kv.Key).ToList(),
                        ["Method"] = _method.ToString()
                    }
                });
            }

            return result;
        }

        /// <summary>
        /// Detects concept drift in predictions
        /// </summary>
        public override async Task<DriftDetectionResult> DetectConceptDriftAsync(Vector<T> predictions, Vector<T> actuals)
        {
            if (predictions.Length != actuals.Length)
            {
                throw new ArgumentException("Predictions and actuals must have the same length");
            }

            if (predictions.Length < _thresholds.MinimumSamplesForDriftDetection)
            {
                return new DriftDetectionResult
                {
                    IsDriftDetected = false,
                    DriftScore = 0,
                    DriftType = "ConceptDrift",
                    Details = $"Insufficient samples for drift detection. Minimum required: {_thresholds.MinimumSamplesForDriftDetection}",
                    DetectionTimestamp = DateTime.UtcNow
                };
            }

            // Calculate prediction errors
            var errors = predictions.Zip(actuals, (p, a) => Math.Abs(p - a)).ToArray();
            
            // Use Page-Hinkley test for concept drift detection
            var driftScore = await DetectConceptDriftPageHinkleyAsync(errors);
            
            var isDriftDetected = driftScore > _thresholds.ConceptDriftThreshold;
            
            var result = new DriftDetectionResult
            {
                IsDriftDetected = isDriftDetected,
                DriftScore = driftScore,
                DriftType = "ConceptDrift",
                DetectionTimestamp = DateTime.UtcNow,
                Details = $"Concept drift score: {driftScore:F4} (threshold: {_thresholds.ConceptDriftThreshold})"
            };

            if (isDriftDetected)
            {
                SendAlert(new MonitoringAlert
                {
                    AlertType = "ConceptDrift",
                    Severity = driftScore > _thresholds.ConceptDriftThreshold * 2 ? "Critical" : "Warning",
                    Message = "Concept drift detected in model predictions",
                    Timestamp = DateTime.UtcNow,
                    Context = new Dictionary<string, object>
                    {
                        ["DriftScore"] = driftScore,
                        ["MeanError"] = errors.Average(),
                        ["StdError"] = Math.Sqrt(errors.Select(e => Math.Pow(e - errors.Average(), 2)).Average())
                    }
                });
            }

            return result;
        }

        /// <summary>
        /// Gets model health score
        /// </summary>
        public override async Task<ModelHealthScore> GetModelHealthScoreAsync()
        {
            var recentPerformance = await GetPerformanceMetricsAsync(DateTime.UtcNow.AddDays(-7), DateTime.UtcNow);
            var recentDrifts = _driftHistory.Where(d => d.DetectionTimestamp > DateTime.UtcNow.AddDays(-7)).ToList();
            
            // Calculate component scores
            var dataQualityScore = CalculateDataQualityScore();
            var performanceScore = CalculatePerformanceScore(recentPerformance);
            var stabilityScore = CalculateStabilityScore();
            var driftScore = 1.0 - (recentDrifts.Any() ? recentDrifts.Max(d => d.DriftScore) : 0);
            
            // Calculate overall score (weighted average)
            var overallScore = (dataQualityScore * 0.2 + performanceScore * 0.4 + stabilityScore * 0.2 + driftScore * 0.2);
            
            var healthStatus = overallScore >= _thresholds.HealthScoreWarningThreshold ? "Healthy" :
                              overallScore >= _thresholds.HealthScoreCriticalThreshold ? "Warning" : "Critical";
            
            var issues = new List<string>();
            if (dataQualityScore < 0.7) issues.Add("Poor data quality");
            if (performanceScore < 0.7) issues.Add("Performance degradation");
            if (stabilityScore < 0.7) issues.Add("Model instability");
            if (driftScore < 0.7) issues.Add("Significant drift detected");
            
            return new ModelHealthScore
            {
                OverallScore = overallScore,
                DataQualityScore = dataQualityScore,
                PerformanceScore = performanceScore,
                StabilityScore = stabilityScore,
                DriftScore = driftScore,
                HealthStatus = healthStatus,
                Issues = issues,
                EvaluationTimestamp = DateTime.UtcNow
            };
        }

        /// <summary>
        /// Gets retraining recommendation
        /// </summary>
        public override async Task<RetrainingRecommendation> GetRetrainingRecommendationAsync()
        {
            var healthScore = await GetModelHealthScoreAsync();
            var recentDrifts = _driftHistory.Where(d => d.DetectionTimestamp > DateTime.UtcNow.AddDays(-30)).ToList();
            var performance = await GetPerformanceMetricsAsync(DateTime.UtcNow.AddDays(-30), DateTime.UtcNow);
            
            var reasons = new List<string>();
            var shouldRetrain = false;
            var urgency = "Low";
            
            // Check health score
            if (healthScore.OverallScore < _thresholds.HealthScoreCriticalThreshold)
            {
                shouldRetrain = true;
                urgency = "Critical";
                reasons.Add($"Model health score is critical: {healthScore.OverallScore:F2}");
            }
            else if (healthScore.OverallScore < _thresholds.HealthScoreWarningThreshold)
            {
                shouldRetrain = true;
                urgency = "High";
                reasons.Add($"Model health score is below warning threshold: {healthScore.OverallScore:F2}");
            }
            
            // Check drift frequency
            var significantDrifts = recentDrifts.Count(d => d.IsDriftDetected);
            if (significantDrifts > 5)
            {
                shouldRetrain = true;
                if (urgency == "Low") urgency = "Medium";
                reasons.Add($"Frequent drift detected: {significantDrifts} instances in the last 30 days");
            }
            
            // Check performance degradation
            if (performance.Accuracy < 0.8 || performance.F1Score < 0.75)
            {
                shouldRetrain = true;
                if (urgency == "Low") urgency = "High";
                reasons.Add($"Performance below acceptable threshold (Accuracy: {performance.Accuracy:F2}, F1: {performance.F1Score:F2})");
            }
            
            var confidenceScore = CalculateRetrainingConfidence(healthScore, recentDrifts, performance);
            
            return new RetrainingRecommendation
            {
                ShouldRetrain = shouldRetrain,
                Urgency = urgency,
                Reasons = reasons,
                RecommendationTimestamp = DateTime.UtcNow,
                ConfidenceScore = confidenceScore,
                SuggestedActions = new Dictionary<string, object>
                {
                    ["CollectMoreData"] = significantDrifts > 3,
                    ["FeatureEngineering"] = healthScore.DataQualityScore < 0.7,
                    ["HyperparameterTuning"] = performance.Accuracy < 0.85,
                    ["IncrementalLearning"] = urgency == "Low" && shouldRetrain
                }
            };
        }

        // Helper methods

        private double[] ExtractFeatureColumn(double[,] data, int featureIndex)
        {
            var column = new double[data.GetLength(0)];
            for (int i = 0; i < data.GetLength(0); i++)
            {
                column[i] = data[i, featureIndex];
            }
            return column;
        }

        private async Task<double> CalculateFeatureDriftAsync(double[] production, double[] reference)
        {
            return await Task.Run(() =>
            {
                switch (_method)
                {
                    case DriftDetectionMethod.KolmogorovSmirnov:
                        return CalculateKSStatistic(production, reference);
                    
                    case DriftDetectionMethod.ChiSquared:
                        return CalculateChiSquaredStatistic(production, reference);
                    
                    case DriftDetectionMethod.JensenShannon:
                        return CalculateJensenShannonDivergence(production, reference);
                    
                    case DriftDetectionMethod.WassersteinDistance:
                        return CalculateWassersteinDistance(production, reference);
                    
                    case DriftDetectionMethod.PopulationStabilityIndex:
                        return CalculatePSI(production, reference);
                    
                    case DriftDetectionMethod.KullbackLeibler:
                        return CalculateKLDivergence(production, reference);
                    
                    default:
                        return CalculateKSStatistic(production, reference);
                }
            });
        }

        private double CalculateKSStatistic(double[] sample1, double[] sample2)
        {
            var sorted1 = sample1.OrderBy(x => x).ToArray();
            var sorted2 = sample2.OrderBy(x => x).ToArray();
            
            double maxDiff = 0;
            int i = 0, j = 0;
            
            while (i < sorted1.Length && j < sorted2.Length)
            {
                double cdf1 = (double)(i + 1) / sorted1.Length;
                double cdf2 = (double)(j + 1) / sorted2.Length;
                
                maxDiff = Math.Max(maxDiff, Math.Abs(cdf1 - cdf2));
                
                if (sorted1[i] <= sorted2[j]) i++;
                else j++;
            }
            
            return maxDiff;
        }

        private double CalculateChiSquaredStatistic(double[] sample1, double[] sample2)
        {
            const int bins = 10;
            var min = Math.Min(sample1.Min(), sample2.Min());
            var max = Math.Max(sample1.Max(), sample2.Max());
            var binWidth = (max - min) / bins;
            
            var hist1 = CreateHistogram(sample1, bins, min, max);
            var hist2 = CreateHistogram(sample2, bins, min, max);
            
            double chiSquared = 0;
            for (int i = 0; i < bins; i++)
            {
                var expected = (hist1[i] + hist2[i]) / 2.0;
                if (expected > 0)
                {
                    chiSquared += Math.Pow(hist1[i] - expected, 2) / expected;
                    chiSquared += Math.Pow(hist2[i] - expected, 2) / expected;
                }
            }
            
            return chiSquared / (2 * bins);
        }

        private double CalculateJensenShannonDivergence(double[] sample1, double[] sample2)
        {
            const int bins = 20;
            var min = Math.Min(sample1.Min(), sample2.Min());
            var max = Math.Max(sample1.Max(), sample2.Max());
            
            var p = CreateProbabilityDistribution(sample1, bins, min, max);
            var q = CreateProbabilityDistribution(sample2, bins, min, max);
            var m = p.Zip(q, (a, b) => (a + b) / 2).ToArray();
            
            var klPM = CalculateKLDivergenceVectors(p, m);
            var klQM = CalculateKLDivergenceVectors(q, m);
            
            return Math.Sqrt((klPM + klQM) / 2);
        }

        private double CalculateWassersteinDistance(double[] sample1, double[] sample2)
        {
            var sorted1 = sample1.OrderBy(x => x).ToArray();
            var sorted2 = sample2.OrderBy(x => x).ToArray();
            
            // Compute empirical CDFs
            var allValues = sorted1.Concat(sorted2).Distinct().OrderBy(x => x).ToArray();
            double distance = 0;
            
            for (int i = 0; i < allValues.Length - 1; i++)
            {
                var cdf1 = sorted1.Count(x => x <= allValues[i]) / (double)sorted1.Length;
                var cdf2 = sorted2.Count(x => x <= allValues[i]) / (double)sorted2.Length;
                distance += Math.Abs(cdf1 - cdf2) * (allValues[i + 1] - allValues[i]);
            }
            
            return distance / (allValues.Last() - allValues.First());
        }

        private double CalculatePSI(double[] actual, double[] expected)
        {
            const int bins = 10;
            var min = Math.Min(actual.Min(), expected.Min());
            var max = Math.Max(actual.Max(), expected.Max());
            
            var actualDist = CreateProbabilityDistribution(actual, bins, min, max);
            var expectedDist = CreateProbabilityDistribution(expected, bins, min, max);
            
            double psi = 0;
            for (int i = 0; i < bins; i++)
            {
                if (actualDist[i] > 0 && expectedDist[i] > 0)
                {
                    psi += (actualDist[i] - expectedDist[i]) * Math.Log(actualDist[i] / expectedDist[i]);
                }
            }
            
            return psi;
        }

        private double CalculateKLDivergence(double[] sample1, double[] sample2)
        {
            const int bins = 20;
            var min = Math.Min(sample1.Min(), sample2.Min());
            var max = Math.Max(sample1.Max(), sample2.Max());
            
            var p = CreateProbabilityDistribution(sample1, bins, min, max);
            var q = CreateProbabilityDistribution(sample2, bins, min, max);
            
            return CalculateKLDivergenceVectors(p, q);
        }

        private double CalculateKLDivergenceVectors(double[] p, double[] q)
        {
            double kl = 0;
            for (int i = 0; i < p.Length; i++)
            {
                if (p[i] > 0 && q[i] > 0)
                {
                    kl += p[i] * Math.Log(p[i] / q[i]);
                }
            }
            return kl;
        }

        private double[] CreateHistogram(double[] data, int bins, double min, double max)
        {
            var histogram = new double[bins];
            var binWidth = (max - min) / bins;
            
            foreach (var value in data)
            {
                var binIndex = Math.Min((int)((value - min) / binWidth), bins - 1);
                histogram[binIndex]++;
            }
            
            return histogram;
        }

        private double[] CreateProbabilityDistribution(double[] data, int bins, double min, double max)
        {
            var histogram = CreateHistogram(data, bins, min, max);
            var sum = histogram.Sum();
            
            // Add small epsilon to avoid log(0)
            const double epsilon = 1e-10;
            return histogram.Select(h => (h + epsilon) / (sum + bins * epsilon)).ToArray();
        }

        private async Task<double> DetectConceptDriftPageHinkleyAsync(double[] errors)
        {
            return await Task.Run(() =>
            {
                const double delta = 0.005;
                const double lambda = 50;
                
                double sum = 0;
                double minSum = double.MaxValue;
                double ph = 0;
                
                for (int i = 0; i < errors.Length; i++)
                {
                    sum += errors[i] - errors.Take(i + 1).Average() - delta;
                    minSum = Math.Min(minSum, sum);
                    ph = Math.Max(ph, sum - minSum);
                    
                    if (ph > lambda)
                    {
                        return ph / lambda; // Normalized drift score
                    }
                }
                
                return ph / lambda;
            });
        }

        private double CalculateDataQualityScore()
        {
            lock (_lockObject)
            {
                if (!_predictionHistory.Any()) return 1.0;
                
                var recentPredictions = _predictionHistory.TakeLast(1000).ToList();

                // Check for missing values, outliers, etc.
                var missingValueRatio = recentPredictions.Count(p => p.Features.Any(f => double.IsNaN(f) || double.IsInfinity(f))) / (double)recentPredictions.Count();
                var outlierRatio = CalculateOutlierRatio(recentPredictions);
                
                return 1.0 - (missingValueRatio * 0.5 + outlierRatio * 0.5);
            }
        }

        private double CalculatePerformanceScore(PerformanceMetrics metrics)
        {
            if (metrics.PredictionCount == 0) return 1.0;
            
            // Combine multiple metrics into a single score
            var accuracyScore = metrics.Accuracy;
            var f1Score = metrics.F1Score;
            var errorScore = 1.0 - Math.Min(metrics.MAE, 1.0);
            
            return (accuracyScore + f1Score + errorScore) / 3.0;
        }

        private double CalculateStabilityScore()
        {
            lock (_lockObject)
            {
                if (_performanceHistory.Count < 5) return 1.0;
                
                var recentMetrics = _performanceHistory.TakeLast(10).ToList();
                var accuracies = recentMetrics.Select(m => m.Accuracy).ToArray();
                
                // Calculate coefficient of variation
                var mean = accuracies.Average();
                var stdDev = Math.Sqrt(accuracies.Select(a => Math.Pow(a - mean, 2)).Average());
                var cv = stdDev / mean;
                
                return 1.0 - Math.Min(cv * 5, 1.0); // Scale CV to [0, 1]
            }
        }

        private double CalculateOutlierRatio(List<PredictionRecord> records)
        {
            if (!records.Any()) return 0;
            
            var outlierCount = 0;
            var featureCount = records.First().Features.Length;
            
            for (int i = 0; i < featureCount; i++)
            {
                var values = records.Select(r => r.Features[i]).ToArray();
                var mean = values.Average();
                var stdDev = Math.Sqrt(values.Select(v => Math.Pow(v - mean, 2)).Average());
                
                outlierCount += values.Count(v => Math.Abs(v - mean) > 3 * stdDev);
            }
            
            return outlierCount / (double)(records.Count * featureCount);
        }

        private double CalculateRetrainingConfidence(ModelHealthScore healthScore, List<DriftDetectionResult> drifts, PerformanceMetrics performance)
        {
            var healthWeight = 0.4;
            var driftWeight = 0.3;
            var performanceWeight = 0.3;
            
            var healthConfidence = 1.0 - healthScore.OverallScore;
            var driftConfidence = drifts.Any() ? drifts.Average(d => d.DriftScore) : 0;
            var performanceConfidence = 1.0 - performance.Accuracy;
            
            return healthWeight * healthConfidence + driftWeight * driftConfidence + performanceWeight * performanceConfidence;
        }

        private class FeatureStatistics
        {
            public double Mean { get; set; }
            public double StdDev { get; set; }
            public double Min { get; set; }
            public double Max { get; set; }
        }
    }
}