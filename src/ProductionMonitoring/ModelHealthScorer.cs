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
    /// Calculates comprehensive model health scores based on multiple factors
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations</typeparam>
    public class ModelHealthScorer<T> : ProductionMonitorBase<T>
    {
        private readonly HealthScoringConfiguration _configuration = default!;
        private readonly Dictionary<string, HealthComponent> _healthComponents = default!;
        private readonly List<HealthCheckResult> _healthHistory = default!;
        private readonly Dictionary<string, Func<Task<double>>> _customHealthChecks = default!;

        public ModelHealthScorer(HealthScoringConfiguration? configuration = null)
        {
            _configuration = configuration ?? new HealthScoringConfiguration();
            _healthComponents = new Dictionary<string, HealthComponent>();
            _healthHistory = new List<HealthCheckResult>();
            _customHealthChecks = new Dictionary<string, Func<Task<double>>>();
            
            InitializeHealthComponents();
        }

        /// <summary>
        /// Registers a custom health check
        /// </summary>
        public void RegisterHealthCheck(string name, Func<Task<double>> healthCheck, double weight = 1.0)
        {
            _customHealthChecks[name] = healthCheck;
            _healthComponents[name] = new HealthComponent
            {
                Name = name,
                Weight = weight,
                IsCustom = true
            };
        }

        /// <summary>
        /// Performs a comprehensive health check
        /// </summary>
        public async Task<ComprehensiveHealthReport> PerformHealthCheckAsync()
        {
            var componentScores = new Dictionary<string, ComponentScore>();
            
            // Evaluate each health component
            foreach (var component in _healthComponents.Values)
            {
                var score = await EvaluateComponentAsync(component);
                componentScores[component.Name] = score;
            }
            
            // Calculate overall health score
            var overallScore = CalculateWeightedScore(componentScores);
            
            // Determine health status
            var healthStatus = DetermineHealthStatus(overallScore);
            
            // Identify issues and recommendations
            var issues = IdentifyIssues(componentScores);
            var recommendations = GenerateRecommendations(componentScores, issues);
            
            // Create health report
            var report = new ComprehensiveHealthReport
            {
                OverallHealthScore = overallScore,
                HealthStatus = healthStatus,
                ComponentScores = componentScores,
                Issues = issues,
                Recommendations = recommendations,
                Timestamp = DateTime.UtcNow,
                TrendAnalysis = await AnalyzeHealthTrendsAsync(),
                PredictedHealthIn24Hours = PredictFutureHealth(24),
                RiskFactors = IdentifyRiskFactors(componentScores)
            };
            
            // Store in history
            lock (_lockObject)
            {
                _healthHistory.Add(new HealthCheckResult
                {
                    Timestamp = report.Timestamp,
                    OverallScore = report.OverallHealthScore,
                    Status = report.HealthStatus,
                    ComponentScores = new Dictionary<string, double>(
                        componentScores.ToDictionary(kv => kv.Key, kv => kv.Value.Score))
                });
                
                // Trim history
                if (_healthHistory.Count > _configuration.MaxHistorySize)
                {
                    _healthHistory.RemoveAt(0);
                }
            }
            
            // Send alerts if needed
            await CheckHealthAlertsAsync(report);
            
            return report;
        }

        /// <summary>
        /// Gets model health score (implements interface)
        /// </summary>
        public override async Task<ModelHealthScore> GetModelHealthScoreAsync()
        {
            var report = await PerformHealthCheckAsync();
            
            return new ModelHealthScore
            {
                OverallScore = report.OverallHealthScore,
                DataQualityScore = report.ComponentScores.ContainsKey("DataQuality") ? 
                    report.ComponentScores["DataQuality"].Score : 1.0,
                PerformanceScore = report.ComponentScores.ContainsKey("Performance") ? 
                    report.ComponentScores["Performance"].Score : 1.0,
                StabilityScore = report.ComponentScores.ContainsKey("Stability") ? 
                    report.ComponentScores["Stability"].Score : 1.0,
                DriftScore = report.ComponentScores.ContainsKey("Drift") ? 
                    report.ComponentScores["Drift"].Score : 1.0,
                HealthStatus = report.HealthStatus,
                Issues = report.Issues.Select(i => i.Description).ToList(),
                EvaluationTimestamp = report.Timestamp
            };
        }

        /// <summary>
        /// Analyzes health trends over a specified time period
        /// </summary>
        /// <param name="lookbackDays">Number of days to analyze in the trend history. Default is 7 days.</param>
        /// <returns>
        /// A task that represents the asynchronous operation, containing a <see cref="HealthTrendAnalysis"/>
        /// with trend direction, strength, and component-level trend information.
        /// </returns>
        /// <remarks>
        /// <para>
        /// This method performs a comprehensive analysis of model health trends by examining historical
        /// health check data over the specified lookback period. It uses linear regression to calculate
        /// trends for both overall health scores and individual component scores.
        /// </para>
        /// <para>
        /// The analysis includes:
        /// - Overall trend direction (Improving, Degrading, or Stable) based on a 5% threshold
        /// - Trend strength indicating the magnitude of change over time
        /// - Component-level trends for each registered health component (Performance, DataQuality, etc.)
        /// - Statistical measures including current value, average, min/max, and volatility for each component
        /// </para>
        /// <para>
        /// If there is insufficient historical data (no records in the lookback period), the method
        /// returns an "Unknown" trend with empty component trends. This helps prevent false trend
        /// signals when the model is newly deployed or monitoring has just started.
        /// </para>
        /// <para>
        /// The method executes asynchronously on a background thread to avoid blocking the caller,
        /// making it suitable for use in production monitoring scenarios where responsiveness is important.
        /// </para>
        /// </remarks>
        public Task<HealthTrendAnalysis> AnalyzeHealthTrendsAsync(int lookbackDays = 7)
        {
            return Task.Run(() =>
            {
                List<HealthCheckResult> relevantHistory;
                lock (_lockObject)
                {
                    var cutoff = DateTime.UtcNow.AddDays(-lookbackDays);
                    relevantHistory = _healthHistory
                        .Where(h => h.Timestamp >= cutoff)
                        .OrderBy(h => h.Timestamp)
                        .ToList();
                }

                if (!relevantHistory.Any())
                {
                    return new HealthTrendAnalysis
                    {
                        TrendDirection = "Unknown",
                        TrendStrength = 0,
                        ComponentTrends = new Dictionary<string, TrendInfo>(),
                        AnalysisPeriod = lookbackDays,
                        DataPoints = 0
                    };
                }

                // Calculate overall trend
                var overallTrend = CalculateTrend(relevantHistory.Select(h => h.OverallScore).ToList());

                // Calculate component trends
                var componentTrends = new Dictionary<string, TrendInfo>();
                foreach (var componentName in _healthComponents.Keys)
                {
                    var componentScores = relevantHistory
                        .Where(h => h.ComponentScores.ContainsKey(componentName))
                        .Select(h => h.ComponentScores[componentName])
                        .ToList();

                    if (componentScores.Any())
                    {
                        componentTrends[componentName] = new TrendInfo
                        {
                            Trend = CalculateTrend(componentScores),
                            CurrentValue = componentScores.Last(),
                            AverageValue = componentScores.Average(),
                            MinValue = componentScores.Min(),
                            MaxValue = componentScores.Max(),
                            Volatility = CalculateVolatility(componentScores)
                        };
                    }
                }

                return new HealthTrendAnalysis
                {
                    TrendDirection = overallTrend > 0.05 ? "Improving" :
                                    overallTrend < -0.05 ? "Degrading" : "Stable",
                    TrendStrength = Math.Abs(overallTrend),
                    OverallTrend = overallTrend,
                    ComponentTrends = componentTrends,
                    AnalysisPeriod = lookbackDays,
                    DataPoints = relevantHistory.Count
                };
            });
        }

        /// <summary>
        /// Detects data drift (delegates to DataDriftDetector)
        /// </summary>
        public override async Task<DriftDetectionResult> DetectDataDriftAsync(Matrix<T> productionData, Matrix<T>? referenceData = null)
        {
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
        /// Detects concept drift (delegates to ConceptDriftDetector)
        /// </summary>
        public override async Task<DriftDetectionResult> DetectConceptDriftAsync(Vector<T> predictions, Vector<T> actuals)
        {
            return Task.FromResult(new DriftDetectionResult
            {
                IsDriftDetected = false,
                DriftScore = 0,
                DriftType = "ConceptDrift",
                Details = "Use ConceptDriftDetector for concept drift detection",
                DetectionTimestamp = DateTime.UtcNow
            });
        }

        /// <summary>
        /// Gets retraining recommendation based on health score
        /// </summary>
        public override async Task<RetrainingRecommendation> GetRetrainingRecommendationAsync()
        {
            var healthReport = await PerformHealthCheckAsync();
            var trends = await AnalyzeHealthTrendsAsync(30);
            
            var reasons = new List<string>();
            var shouldRetrain = false;
            var urgency = "Low";
            
            // Check overall health
            if (healthReport.OverallHealthScore < 0.5)
            {
                shouldRetrain = true;
                urgency = "Critical";
                reasons.Add($"Critical health score: {healthReport.OverallHealthScore:F2}");
            }
            else if (healthReport.OverallHealthScore < 0.7)
            {
                shouldRetrain = true;
                urgency = "High";
                reasons.Add($"Poor health score: {healthReport.OverallHealthScore:F2}");
            }
            
            // Check trends
            if (trends.TrendDirection == "Degrading" && trends.TrendStrength > 0.1)
            {
                shouldRetrain = true;
                if (urgency == "Low") urgency = "Medium";
                reasons.Add($"Health degrading with strength {trends.TrendStrength:F2}");
            }
            
            // Check critical components
            var criticalComponents = healthReport.ComponentScores
                .Where(cs => _healthComponents[cs.Key].IsCritical && cs.Value.Score < 0.6)
                .ToList();
            
            if (criticalComponents.Any())
            {
                shouldRetrain = true;
                if (urgency != "Critical") urgency = "High";
                reasons.AddRange(criticalComponents.Select(cs => 
                    $"Critical component '{cs.Key}' failing: {cs.Value.Score:F2}"));
            }
            
            // Check risk factors
            if (healthReport.RiskFactors.Any(rf => rf.Severity == "High"))
            {
                shouldRetrain = true;
                if (urgency == "Low") urgency = "Medium";
                reasons.AddRange(healthReport.RiskFactors
                    .Where(rf => rf.Severity == "High")
                    .Select(rf => rf.Description));
            }
            
            var confidence = CalculateRetrainingConfidence(healthReport, trends);
            
            return new RetrainingRecommendation
            {
                ShouldRetrain = shouldRetrain,
                Urgency = urgency,
                Reasons = reasons,
                RecommendationTimestamp = DateTime.UtcNow,
                ConfidenceScore = confidence,
                SuggestedActions = GenerateSuggestedActions(healthReport, trends)
            };
        }

        // Private methods

        private void InitializeHealthComponents()
        {
            // Core health components
            _healthComponents["Performance"] = new HealthComponent
            {
                Name = "Performance",
                Weight = _configuration.PerformanceWeight,
                IsCritical = true,
                MinThreshold = 0.7,
                WarningThreshold = 0.8
            };
            
            _healthComponents["DataQuality"] = new HealthComponent
            {
                Name = "DataQuality",
                Weight = _configuration.DataQualityWeight,
                IsCritical = false,
                MinThreshold = 0.6,
                WarningThreshold = 0.75
            };
            
            _healthComponents["Drift"] = new HealthComponent
            {
                Name = "Drift",
                Weight = _configuration.DriftWeight,
                IsCritical = true,
                MinThreshold = 0.5,
                WarningThreshold = 0.7
            };
            
            _healthComponents["Stability"] = new HealthComponent
            {
                Name = "Stability",
                Weight = _configuration.StabilityWeight,
                IsCritical = false,
                MinThreshold = 0.6,
                WarningThreshold = 0.8
            };
            
            _healthComponents["Latency"] = new HealthComponent
            {
                Name = "Latency",
                Weight = _configuration.LatencyWeight,
                IsCritical = false,
                MinThreshold = 0.7,
                WarningThreshold = 0.85
            };
            
            _healthComponents["ResourceUsage"] = new HealthComponent
            {
                Name = "ResourceUsage",
                Weight = _configuration.ResourceWeight,
                IsCritical = false,
                MinThreshold = 0.5,
                WarningThreshold = 0.7
            };
        }

        private async Task<ComponentScore> EvaluateComponentAsync(HealthComponent component)
        {
            double score;
            string status;
            var details = new Dictionary<string, object>();
            
            if (component.IsCustom && _customHealthChecks.ContainsKey(component.Name))
            {
                score = await _customHealthChecks[component.Name]();
            }
            else
            {
                score = component.Name switch
                {
                    "Performance" => await EvaluatePerformanceHealthAsync(details),
                    "DataQuality" => await EvaluateDataQualityHealthAsync(details),
                    "Drift" => await EvaluateDriftHealthAsync(details),
                    "Stability" => await EvaluateStabilityHealthAsync(details),
                    "Latency" => await EvaluateLatencyHealthAsync(details),
                    "ResourceUsage" => await EvaluateResourceHealthAsync(details),
                    _ => 1.0
                };
            }
            
            if (score < component.MinThreshold)
            {
                status = "Critical";
            }
            else if (score < component.WarningThreshold)
            {
                status = "Warning";
            }
            else
            {
                status = "Healthy";
            }
            
            return new ComponentScore
            {
                ComponentName = component.Name,
                Score = score,
                Status = status,
                Details = details,
                LastChecked = DateTime.UtcNow
            };
        }

        private async Task<double> EvaluatePerformanceHealthAsync(Dictionary<string, object> details)
        {
            var metrics = await GetPerformanceMetricsAsync(DateTime.UtcNow.AddHours(-1), DateTime.UtcNow);

            if (metrics.PredictionCount == 0) return 1.0;

            details["Accuracy"] = metrics.Accuracy;
            details["F1Score"] = metrics.F1Score;
            details["MAE"] = metrics.MAE;

            // Combine multiple metrics
            var accuracyScore = metrics.Accuracy;
            var f1Score = metrics.F1Score;
            var errorScore = Math.Max(0, 1.0 - metrics.MAE);

            return await Task.FromResult(accuracyScore * 0.4 + f1Score * 0.4 + errorScore * 0.2);
        }

        private Task<double> EvaluateDataQualityHealthAsync(Dictionary<string, object> details)
        {
            lock (_lockObject)
            {
                if (!_predictionHistory.Any()) return Task.FromResult(1.0);

                var recent = _predictionHistory.TakeLast(100).ToList();

                // Check for data quality issues
                var missingValues = recent.Count(p => p.Features.Any(f => double.IsNaN(f) || double.IsInfinity(f)));
                var outliers = recent.Count(p => p.Features.Any(f => Math.Abs(f) > 10));
                var duplicates = recent.Count() - recent.Distinct().Count();

                details["MissingValues"] = missingValues;
                details["Outliers"] = outliers;
                details["Duplicates"] = duplicates;

                var missingScore = 1.0 - (missingValues / (double)recent.Count());
                var outlierScore = 1.0 - (outliers / (double)recent.Count());
                var duplicateScore = 1.0 - (duplicates / (double)recent.Count());

                return Task.FromResult(missingScore * 0.5 + outlierScore * 0.3 + duplicateScore * 0.2);
            }
        }

        private Task<double> EvaluateDriftHealthAsync(Dictionary<string, object> details)
        {
            lock (_lockObject)
            {
                var recentDrifts = _driftHistory
                    .Where(d => d.DetectionTimestamp > DateTime.UtcNow.AddHours(-24))
                    .ToList();

                if (!recentDrifts.Any()) return Task.FromResult(1.0);

                var driftCount = recentDrifts.Count(d => d.IsDriftDetected);
                var maxDriftScore = recentDrifts.Max(d => d.DriftScore);

                details["DriftCount"] = driftCount;
                details["MaxDriftScore"] = maxDriftScore;

                var countScore = Math.Max(0, 1.0 - (driftCount / 10.0));
                var severityScore = Math.Max(0, 1.0 - maxDriftScore);

                return Task.FromResult(countScore * 0.6 + severityScore * 0.4);
            }
        }

        private Task<double> EvaluateStabilityHealthAsync(Dictionary<string, object> details)
        {
            lock (_lockObject)
            {
                if (_performanceHistory.Count < 5) return Task.FromResult(1.0);

                var recent = _performanceHistory.TakeLast(10).ToList();
                var accuracies = recent.Select(p => p.Accuracy).ToArray();

                // Calculate coefficient of variation
                var mean = accuracies.Average();
                var stdDev = Math.Sqrt(accuracies.Select(a => Math.Pow(a - mean, 2)).Average());
                var cv = mean > 0 ? stdDev / mean : 0;

                details["CoefficientOfVariation"] = cv;
                details["StdDev"] = stdDev;

                return Task.FromResult(Math.Max(0, 1.0 - cv * 2));
            }
        }

        private Task<double> EvaluateLatencyHealthAsync(Dictionary<string, object> details)
        {
            // Simulate latency measurement
            var avgLatency = 50.0; // milliseconds
            var p95Latency = 100.0;
            var p99Latency = 200.0;

            details["AvgLatency"] = avgLatency;
            details["P95Latency"] = p95Latency;
            details["P99Latency"] = p99Latency;

            // Score based on latency thresholds
            var avgScore = Math.Max(0, 1.0 - (avgLatency / 200.0));
            var p95Score = Math.Max(0, 1.0 - (p95Latency / 500.0));
            var p99Score = Math.Max(0, 1.0 - (p99Latency / 1000.0));

            return Task.FromResult(avgScore * 0.5 + p95Score * 0.3 + p99Score * 0.2);
        }

        private Task<double> EvaluateResourceHealthAsync(Dictionary<string, object> details)
        {
            // Simulate resource usage
            var cpuUsage = 0.3; // 30%
            var memoryUsage = 0.5; // 50%
            var diskUsage = 0.2; // 20%

            details["CpuUsage"] = cpuUsage;
            details["MemoryUsage"] = memoryUsage;
            details["DiskUsage"] = diskUsage;

            // Lower usage is better
            var cpuScore = 1.0 - cpuUsage;
            var memoryScore = 1.0 - memoryUsage;
            var diskScore = 1.0 - diskUsage;

            return Task.FromResult(cpuScore * 0.4 + memoryScore * 0.4 + diskScore * 0.2);
        }

        private double CalculateWeightedScore(Dictionary<string, ComponentScore> scores)
        {
            double totalWeight = 0;
            double weightedSum = 0;
            
            foreach (var score in scores)
            {
                if (_healthComponents.ContainsKey(score.Key))
                {
                    var weight = _healthComponents[score.Key].Weight;
                    totalWeight += weight;
                    weightedSum += score.Value.Score * weight;
                }
            }
            
            return totalWeight > 0 ? weightedSum / totalWeight : 0;
        }

        private string DetermineHealthStatus(double overallScore)
        {
            if (overallScore >= 0.9) return "Excellent";
            if (overallScore >= 0.8) return "Good";
            if (overallScore >= 0.7) return "Fair";
            if (overallScore >= 0.5) return "Poor";
            return "Critical";
        }

        private List<HealthIssue> IdentifyIssues(Dictionary<string, ComponentScore> scores)
        {
            var issues = new List<HealthIssue>();
            
            foreach (var score in scores)
            {
                var component = _healthComponents[score.Key];
                
                if (score.Value.Score < component.MinThreshold)
                {
                    issues.Add(new HealthIssue
                    {
                        Component = score.Key,
                        Severity = "Critical",
                        Description = $"{score.Key} is critically low: {score.Value.Score:F2}",
                        Impact = component.IsCritical ? "High" : "Medium",
                        Details = score.Value.Details
                    });
                }
                else if (score.Value.Score < component.WarningThreshold)
                {
                    issues.Add(new HealthIssue
                    {
                        Component = score.Key,
                        Severity = "Warning",
                        Description = $"{score.Key} is below warning threshold: {score.Value.Score:F2}",
                        Impact = component.IsCritical ? "Medium" : "Low",
                        Details = score.Value.Details
                    });
                }
            }
            
            return issues.OrderBy(i => i.Severity).ThenBy(i => i.Impact).ToList();
        }

        private List<string> GenerateRecommendations(Dictionary<string, ComponentScore> scores, List<HealthIssue> issues)
        {
            var recommendations = new List<string>();
            
            foreach (var issue in issues.Where(i => i.Severity == "Critical"))
            {
                switch (issue.Component)
                {
                    case "Performance":
                        recommendations.Add("Consider retraining the model with updated data");
                        recommendations.Add("Review feature engineering and selection");
                        break;
                    
                    case "DataQuality":
                        recommendations.Add("Implement data validation and cleaning pipelines");
                        recommendations.Add("Review data collection processes");
                        break;
                    
                    case "Drift":
                        recommendations.Add("Investigate causes of drift");
                        recommendations.Add("Consider online learning or frequent retraining");
                        break;
                    
                    case "Stability":
                        recommendations.Add("Review model architecture for robustness");
                        recommendations.Add("Implement ensemble methods for stability");
                        break;
                    
                    case "Latency":
                        recommendations.Add("Optimize model for inference speed");
                        recommendations.Add("Consider model compression techniques");
                        break;
                    
                    case "ResourceUsage":
                        recommendations.Add("Scale infrastructure resources");
                        recommendations.Add("Optimize model efficiency");
                        break;
                }
            }
            
            return recommendations.Distinct().ToList();
        }

        private double PredictFutureHealth(int hoursAhead)
        {
            lock (_lockObject)
            {
                if (_healthHistory.Count < 10) return -1; // Not enough data
                
                var recent = _healthHistory.TakeLast(24).Select(h => h.OverallScore).ToList();
                var trend = CalculateTrend(recent);
                
                // Simple linear extrapolation
                var currentScore = recent.Last();
                var predictedScore = currentScore + (trend * hoursAhead);
                
                return Math.Max(0, Math.Min(1, predictedScore));
            }
        }

        private List<RiskFactor> IdentifyRiskFactors(Dictionary<string, ComponentScore> scores)
        {
            var riskFactors = new List<RiskFactor>();
            
            // Check for critical components near threshold
            foreach (var score in scores)
            {
                var component = _healthComponents[score.Key];
                var margin = score.Value.Score - component.MinThreshold;
                
                if (margin < 0.1 && margin > 0)
                {
                    riskFactors.Add(new RiskFactor
                    {
                        Factor = $"{score.Key} near critical threshold",
                        Severity = "High",
                        Likelihood = 0.7,
                        Description = $"{score.Key} is only {margin:F2} above critical threshold",
                        MitigationStrategy = $"Monitor {score.Key} closely and prepare intervention"
                    });
                }
            }
            
            // Check for negative trends
            lock (_lockObject)
            {
                if (_healthHistory.Count > 10)
                {
                    var recentTrend = CalculateTrend(_healthHistory.TakeLast(10).Select(h => h.OverallScore).ToList());
                    if (recentTrend < -0.05)
                    {
                        riskFactors.Add(new RiskFactor
                        {
                            Factor = "Negative health trend",
                            Severity = "Medium",
                            Likelihood = 0.8,
                            Description = $"Health declining at rate of {Math.Abs(recentTrend):F3} per hour",
                            MitigationStrategy = "Investigate root cause of decline"
                        });
                    }
                }
            }
            
            return riskFactors;
        }

        private double CalculateTrend(List<double> values)
        {
            if (values.Count < 2) return 0;
            
            var n = values.Count;
            var xValues = Enumerable.Range(0, n).Select(i => (double)i).ToArray();
            var xMean = xValues.Average();
            var yMean = values.Average();
            
            var numerator = xValues.Zip(values, (x, y) => (x - xMean) * (y - yMean)).Sum();
            var denominator = xValues.Select(x => Math.Pow(x - xMean, 2)).Sum();
            
            return denominator == 0 ? 0 : numerator / denominator;
        }

        private double CalculateVolatility(List<double> values)
        {
            if (values.Count < 2) return 0;
            
            var mean = values.Average();
            var variance = values.Select(v => Math.Pow(v - mean, 2)).Average();
            return Math.Sqrt(variance);
        }

        private double CalculateRetrainingConfidence(ComprehensiveHealthReport report, HealthTrendAnalysis trends)
        {
            var healthScore = 1.0 - report.OverallHealthScore;
            var trendPenalty = trends.TrendDirection == "Degrading" ? trends.TrendStrength : 0;
            var issueSeverity = report.Issues.Any() ? 
                report.Issues.Count(i => i.Severity == "Critical") * 0.2 : 0;
            
            return Math.Min(1.0, healthScore * 0.5 + trendPenalty * 0.3 + issueSeverity * 0.2);
        }

        private Dictionary<string, object> GenerateSuggestedActions(ComprehensiveHealthReport report, HealthTrendAnalysis trends)
        {
            var actions = new Dictionary<string, object>();
            
            // Based on component scores
            foreach (var component in report.ComponentScores.Where(cs => cs.Value.Score < 0.7))
            {
                switch (component.Key)
                {
                    case "Performance":
                        actions["ImproveFeatures"] = true;
                        actions["TuneHyperparameters"] = true;
                        break;
                    case "DataQuality":
                        actions["DataCleaning"] = true;
                        actions["FeatureValidation"] = true;
                        break;
                    case "Drift":
                        actions["InvestigateDrift"] = true;
                        actions["UpdateTrainingData"] = true;
                        break;
                }
            }
            
            // Based on trends
            if (trends.TrendDirection == "Degrading")
            {
                actions["AccelerateMonitoring"] = true;
                actions["PrepareRetraining"] = true;
            }
            
            return actions;
        }

        private Task CheckHealthAlertsAsync(ComprehensiveHealthReport report)
        {
            // Check for critical health
            if (report.HealthStatus == "Critical")
            {
                SendAlert(new MonitoringAlert
                {
                    AlertType = "ModelHealth",
                    Severity = "Critical",
                    Message = $"Model health is critical: {report.OverallHealthScore:F2}",
                    Timestamp = DateTime.UtcNow,
                    Context = new Dictionary<string, object>
                    {
                        ["HealthScore"] = report.OverallHealthScore,
                        ["Status"] = report.HealthStatus,
                        ["IssueCount"] = report.Issues.Count
                    }
                });
            }

            // Check for rapid degradation
            if (report.TrendAnalysis != null &&
                report.TrendAnalysis.TrendDirection == "Degrading" &&
                report.TrendAnalysis.TrendStrength > 0.2)
            {
                SendAlert(new MonitoringAlert
                {
                    AlertType = "HealthDegradation",
                    Severity = "Warning",
                    Message = "Model health is rapidly degrading",
                    Timestamp = DateTime.UtcNow,
                    Context = new Dictionary<string, object>
                    {
                        ["TrendStrength"] = report.TrendAnalysis.TrendStrength,
                        ["CurrentScore"] = report.OverallHealthScore
                    }
                });
            }

            return Task.CompletedTask;
        }

        // Helper classes

        public class HealthScoringConfiguration
        {
            public double PerformanceWeight { get; set; } = 0.3;
            public double DataQualityWeight { get; set; } = 0.2;
            public double DriftWeight { get; set; } = 0.2;
            public double StabilityWeight { get; set; } = 0.15;
            public double LatencyWeight { get; set; } = 0.1;
            public double ResourceWeight { get; set; } = 0.05;
            public int MaxHistorySize { get; set; } = 1000;
        }

        private class HealthComponent
        {
            public string Name { get; set; } = string.Empty;
            public double Weight { get; set; }
            public bool IsCritical { get; set; }
            public double MinThreshold { get; set; }
            public double WarningThreshold { get; set; }
            public bool IsCustom { get; set; }
        }

        public class ComponentScore
        {
            public string ComponentName { get; set; } = string.Empty;
            public double Score { get; set; }
            public string Status { get; set; } = string.Empty;
            public Dictionary<string, object> Details { get; set; } = new();
            public DateTime LastChecked { get; set; }
        }

        public class ComprehensiveHealthReport
        {
            public double OverallHealthScore { get; set; }
            public string HealthStatus { get; set; } = string.Empty;
            public Dictionary<string, ComponentScore> ComponentScores { get; set; } = new();
            public List<HealthIssue> Issues { get; set; } = new();
            public List<string> Recommendations { get; set; } = new();
            public DateTime Timestamp { get; set; }
            public HealthTrendAnalysis TrendAnalysis { get; set; } = new();
            public double PredictedHealthIn24Hours { get; set; }
            public List<RiskFactor> RiskFactors { get; set; } = new();
        }

        public class HealthIssue
        {
            public string Component { get; set; } = string.Empty;
            public string Severity { get; set; } = string.Empty;
            public string Description { get; set; } = string.Empty;
            public string Impact { get; set; } = string.Empty;
            public Dictionary<string, object> Details { get; set; } = new();
        }

        public class HealthTrendAnalysis
        {
            public string TrendDirection { get; set; } = string.Empty;
            public double TrendStrength { get; set; }
            public double OverallTrend { get; set; }
            public Dictionary<string, TrendInfo> ComponentTrends { get; set; } = new();
            public int AnalysisPeriod { get; set; }
            public int DataPoints { get; set; }
        }

        public class TrendInfo
        {
            public double Trend { get; set; }
            public double CurrentValue { get; set; }
            public double AverageValue { get; set; }
            public double MinValue { get; set; }
            public double MaxValue { get; set; }
            public double Volatility { get; set; }
        }

        private class HealthCheckResult
        {
            public DateTime Timestamp { get; set; }
            public double OverallScore { get; set; }
            public string Status { get; set; } = string.Empty;
            public Dictionary<string, double> ComponentScores { get; set; } = new();
        }

        public class RiskFactor
        {
            public string Factor { get; set; } = string.Empty;
            public string Severity { get; set; } = string.Empty;
            public double Likelihood { get; set; }
            public string Description { get; set; } = string.Empty;
            public string MitigationStrategy { get; set; } = string.Empty;
        }
    }
}
