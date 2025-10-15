using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AiDotNet.ProductionMonitoring
{
    /// <summary>
    /// Comprehensive monitoring metrics collection and aggregation
    /// </summary>
    public class MonitoringMetrics
    {
        private readonly Dictionary<string, MetricTimeSeries> _metrics = default!;
        private readonly MetricsConfiguration _configuration = default!;
        private readonly object _lockObject = new object();

        public MonitoringMetrics(MetricsConfiguration? configuration = null)
        {
            _configuration = configuration ?? new MetricsConfiguration();
            _metrics = new Dictionary<string, MetricTimeSeries>();
            InitializeStandardMetrics();
        }

        /// <summary>
        /// Records a metric value
        /// </summary>
        public void RecordMetric(string metricName, double value, DateTime? timestamp = null, Dictionary<string, string>? tags = null)
        {
            timestamp = timestamp ?? DateTime.UtcNow;
            
            lock (_lockObject)
            {
                if (!_metrics.ContainsKey(metricName))
                {
                    _metrics[metricName] = new MetricTimeSeries
                    {
                        Name = metricName,
                        Unit = DetermineUnit(metricName),
                        Type = DetermineMetricType(metricName)
                    };
                }
                
                var metric = _metrics[metricName];
                metric.AddValue(value, timestamp.Value, tags);
                
                // Trim old data
                metric.TrimOldData(_configuration.RetentionPeriod);
            }
        }

        /// <summary>
        /// Records multiple metrics at once
        /// </summary>
        public void RecordMetrics(Dictionary<string, double> metrics, DateTime? timestamp = null)
        {
            foreach (var kvp in metrics)
            {
                RecordMetric(kvp.Key, kvp.Value, timestamp);
            }
        }

        /// <summary>
        /// Gets metric statistics for a time period
        /// </summary>
        public MetricStatistics GetMetricStatistics(string metricName, DateTime? startTime = null, DateTime? endTime = null)
        {
            lock (_lockObject)
            {
                if (!_metrics.ContainsKey(metricName))
                {
                    return new MetricStatistics { MetricName = metricName };
                }
                
                var metric = _metrics[metricName];
                return metric.CalculateStatistics(startTime, endTime);
            }
        }

        /// <summary>
        /// Gets aggregated metrics
        /// </summary>
        public AggregatedMetrics GetAggregatedMetrics(DateTime startTime, DateTime endTime, TimeSpan? aggregationInterval = null)
        {
            aggregationInterval = aggregationInterval ?? TimeSpan.FromMinutes(5);
            
            var aggregated = new AggregatedMetrics
            {
                StartTime = startTime,
                EndTime = endTime,
                AggregationInterval = aggregationInterval.Value,
                Metrics = new Dictionary<string, List<AggregatedDataPoint>>()
            };
            
            lock (_lockObject)
            {
                foreach (var metric in _metrics.Values)
                {
                    aggregated.Metrics[metric.Name] = metric.GetAggregatedData(startTime, endTime, aggregationInterval.Value);
                }
            }
            
            return aggregated;
        }

        /// <summary>
        /// Gets metric correlations
        /// </summary>
        public MetricCorrelations CalculateCorrelations(List<string>? metricNames = null, DateTime? startTime = null, DateTime? endTime = null)
        {
            metricNames = metricNames ?? _metrics.Keys.ToList();
            var correlations = new MetricCorrelations();
            
            lock (_lockObject)
            {
                for (int i = 0; i < metricNames.Count; i++)
                {
                    for (int j = i + 1; j < metricNames.Count; j++)
                    {
                        var metric1 = metricNames[i];
                        var metric2 = metricNames[j];
                        
                        if (_metrics.ContainsKey(metric1) && _metrics.ContainsKey(metric2))
                        {
                            var correlation = CalculateCorrelation(
                                _metrics[metric1], 
                                _metrics[metric2], 
                                startTime, 
                                endTime
                            );
                            
                            if (Math.Abs(correlation) > _configuration.CorrelationThreshold)
                            {
                                correlations.SignificantCorrelations.Add(new MetricCorrelation
                                {
                                    Metric1 = metric1,
                                    Metric2 = metric2,
                                    CorrelationCoefficient = correlation,
                                    Strength = GetCorrelationStrength(correlation)
                                });
                            }
                        }
                    }
                }
            }
            
            correlations.CalculatedAt = DateTime.UtcNow;
            return correlations;
        }

        /// <summary>
        /// Detects anomalies in metrics
        /// </summary>
        public async Task<MetricAnomalies> DetectAnomaliesAsync(string? metricName = null, DateTime? startTime = null, DateTime? endTime = null)
        {
            var anomalies = new MetricAnomalies
            {
                DetectionTimestamp = DateTime.UtcNow,
                Anomalies = new List<MetricAnomaly>()
            };
            
            var metricsToCheck = metricName != null ? new List<string> { metricName } : _metrics.Keys.ToList();
            
            foreach (var metric in metricsToCheck)
            {
                var metricAnomalies = await DetectMetricAnomaliesAsync(metric, startTime, endTime);
                anomalies.Anomalies.AddRange(metricAnomalies);
            }
            
            return anomalies;
        }

        /// <summary>
        /// Gets metric trends
        /// </summary>
        public MetricTrends AnalyzeTrends(List<string>? metricNames = null, int lookbackHours = 24)
        {
            metricNames = metricNames ?? _metrics.Keys.ToList();
            var trends = new MetricTrends
            {
                AnalysisTimestamp = DateTime.UtcNow,
                LookbackPeriod = TimeSpan.FromHours(lookbackHours),
                Trends = new Dictionary<string, TrendAnalysis>()
            };
            
            lock (_lockObject)
            {
                foreach (var metricName in metricNames)
                {
                    if (_metrics.ContainsKey(metricName))
                    {
                        trends.Trends[metricName] = AnalyzeMetricTrend(_metrics[metricName], lookbackHours);
                    }
                }
            }
            
            return trends;
        }

        /// <summary>
        /// Exports metrics data
        /// </summary>
        public MetricsExport ExportMetrics(DateTime startTime, DateTime endTime, List<string>? metricNames = null)
        {
            metricNames = metricNames ?? _metrics.Keys.ToList();
            
            var export = new MetricsExport
            {
                ExportTimestamp = DateTime.UtcNow,
                StartTime = startTime,
                EndTime = endTime,
                Metrics = new Dictionary<string, List<MetricDataPoint>>()
            };
            
            lock (_lockObject)
            {
                foreach (var metricName in metricNames)
                {
                    if (_metrics.ContainsKey(metricName))
                    {
                        export.Metrics[metricName] = _metrics[metricName].GetDataPoints(startTime, endTime);
                    }
                }
            }
            
            export.TotalDataPoints = export.Metrics.Values.Sum(m => m.Count);
            return export;
        }

        /// <summary>
        /// Gets metric summary dashboard
        /// </summary>
        public MetricsDashboard GetDashboard()
        {
            var dashboard = new MetricsDashboard
            {
                Timestamp = DateTime.UtcNow,
                MetricSummaries = new List<MetricSummary>(),
                HealthIndicators = new Dictionary<string, double>(),
                RecentAlerts = new List<string>()
            };
            
            lock (_lockObject)
            {
                foreach (var metric in _metrics.Values)
                {
                    var stats = metric.CalculateStatistics(DateTime.UtcNow.AddHours(-1), DateTime.UtcNow);
                    
                    dashboard.MetricSummaries.Add(new MetricSummary
                    {
                        MetricName = metric.Name,
                        CurrentValue = stats.LastValue,
                        Average = stats.Mean,
                        Min = stats.Min,
                        Max = stats.Max,
                        Trend = stats.Trend,
                        Unit = metric.Unit
                    });
                }
            }
            
            // Calculate health indicators
            dashboard.HealthIndicators["DataCompleteness"] = CalculateDataCompleteness();
            dashboard.HealthIndicators["MetricStability"] = CalculateMetricStability();
            dashboard.HealthIndicators["AnomalyRate"] = CalculateAnomalyRate();
            
            return dashboard;
        }

        // Private methods

        private void InitializeStandardMetrics()
        {
            // Initialize standard monitoring metrics
            var standardMetrics = new List<string>
            {
                "model.predictions.count",
                "model.predictions.latency",
                "model.accuracy",
                "model.precision",
                "model.recall",
                "model.f1_score",
                "model.mae",
                "model.rmse",
                "data.quality.score",
                "data.missing_values.ratio",
                "data.outliers.ratio",
                "drift.data.score",
                "drift.concept.score",
                "system.cpu.usage",
                "system.memory.usage",
                "system.disk.usage"
            };
            
            foreach (var metricName in standardMetrics)
            {
                _metrics[metricName] = new MetricTimeSeries
                {
                    Name = metricName,
                    Unit = DetermineUnit(metricName),
                    Type = DetermineMetricType(metricName)
                };
            }
        }

        private string DetermineUnit(string metricName)
        {
            if (metricName.Contains("latency")) return "ms";
            if (metricName.Contains("count")) return "count";
            if (metricName.Contains("ratio") || metricName.Contains("score")) return "ratio";
            if (metricName.Contains("usage")) return "percent";
            if (metricName.Contains("memory")) return "MB";
            if (metricName.Contains("disk")) return "GB";
            return "value";
        }

        private MetricType DetermineMetricType(string metricName)
        {
            if (metricName.Contains("count")) return MetricType.Counter;
            if (metricName.Contains("latency")) return MetricType.Histogram;
            return MetricType.Gauge;
        }

        private double CalculateCorrelation(MetricTimeSeries metric1, MetricTimeSeries metric2, DateTime? startTime, DateTime? endTime)
        {
            var data1 = metric1.GetDataPoints(startTime, endTime);
            var data2 = metric2.GetDataPoints(startTime, endTime);
            
            if (data1.Count < 2 || data2.Count < 2) return 0;
            
            // Align timestamps
            var aligned = AlignTimeSeries(data1, data2);
            if (aligned.Count < 2) return 0;
            
            var values1 = aligned.Select(p => p.Item1).ToArray();
            var values2 = aligned.Select(p => p.Item2).ToArray();
            
            return CalculatePearsonCorrelation(values1, values2);
        }

        private List<(double, double)> AlignTimeSeries(List<MetricDataPoint> series1, List<MetricDataPoint> series2)
        {
            var aligned = new List<(double, double)>();
            var tolerance = TimeSpan.FromSeconds(30);
            
            foreach (var point1 in series1)
            {
                var nearestPoint = series2
                    .Where(p => Math.Abs((p.Timestamp - point1.Timestamp).TotalSeconds) < tolerance.TotalSeconds)
                    .OrderBy(p => Math.Abs((p.Timestamp - point1.Timestamp).TotalSeconds))
                    .FirstOrDefault();
                
                if (nearestPoint != null)
                {
                    aligned.Add((point1.Value, nearestPoint.Value));
                }
            }
            
            return aligned;
        }

        private double CalculatePearsonCorrelation(double[] x, double[] y)
        {
            if (x.Length != y.Length || x.Length < 2) return 0;
            
            var n = x.Length;
            var sumX = x.Sum();
            var sumY = y.Sum();
            var sumXY = x.Zip(y, (a, b) => a * b).Sum();
            var sumX2 = x.Select(a => a * a).Sum();
            var sumY2 = y.Select(b => b * b).Sum();
            
            var numerator = n * sumXY - sumX * sumY;
            var denominator = Math.Sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
            
            return denominator == 0 ? 0 : numerator / denominator;
        }

        private string GetCorrelationStrength(double correlation)
        {
            var absCorr = Math.Abs(correlation);
            if (absCorr >= 0.9) return "Very Strong";
            if (absCorr >= 0.7) return "Strong";
            if (absCorr >= 0.5) return "Moderate";
            if (absCorr >= 0.3) return "Weak";
            return "Very Weak";
        }

        private async Task<List<MetricAnomaly>> DetectMetricAnomaliesAsync(string metricName, DateTime? startTime, DateTime? endTime)
        {
            var anomalies = new List<MetricAnomaly>();
            
            lock (_lockObject)
            {
                if (!_metrics.ContainsKey(metricName)) return anomalies;
                
                var metric = _metrics[metricName];
                var dataPoints = metric.GetDataPoints(startTime, endTime);
                
                if (dataPoints.Count < 10) return anomalies;
                
                // Use statistical method for anomaly detection
                var values = dataPoints.Select(p => p.Value).ToArray();
                var mean = values.Average();
                var stdDev = Math.Sqrt(values.Select(v => Math.Pow(v - mean, 2)).Average());
                
                for (int i = 0; i < dataPoints.Count; i++)
                {
                    var zScore = Math.Abs((dataPoints[i].Value - mean) / stdDev);
                    
                    if (zScore > _configuration.AnomalyZScoreThreshold)
                    {
                        anomalies.Add(new MetricAnomaly
                        {
                            MetricName = metricName,
                            Timestamp = dataPoints[i].Timestamp,
                            Value = dataPoints[i].Value,
                            ExpectedValue = mean,
                            AnomalyScore = zScore,
                            Type = dataPoints[i].Value > mean ? "Spike" : "Drop",
                            Severity = zScore > 4 ? "High" : "Medium"
                        });
                    }
                }
            }
            
            return await Task.FromResult(anomalies);
        }

        private TrendAnalysis AnalyzeMetricTrend(MetricTimeSeries metric, int lookbackHours)
        {
            var endTime = DateTime.UtcNow;
            var startTime = endTime.AddHours(-lookbackHours);
            var dataPoints = metric.GetDataPoints(startTime, endTime);
            
            if (dataPoints.Count < 2)
            {
                return new TrendAnalysis
                {
                    Direction = "Unknown",
                    Strength = 0,
                    Slope = 0
                };
            }
            
            // Calculate linear regression
            var n = dataPoints.Count;
            var xValues = Enumerable.Range(0, n).Select(i => (double)i).ToArray();
            var yValues = dataPoints.Select(p => p.Value).ToArray();
            
            var xMean = xValues.Average();
            var yMean = yValues.Average();
            
            var numerator = xValues.Zip(yValues, (x, y) => (x - xMean) * (y - yMean)).Sum();
            var denominator = xValues.Select(x => Math.Pow(x - xMean, 2)).Sum();
            
            var slope = denominator == 0 ? 0 : numerator / denominator;
            
            // Normalize slope by mean to get percentage change
            var normalizedSlope = yMean != 0 ? slope / yMean : 0;
            
            return new TrendAnalysis
            {
                Direction = normalizedSlope > 0.01 ? "Increasing" : 
                           normalizedSlope < -0.01 ? "Decreasing" : "Stable",
                Strength = Math.Abs(normalizedSlope),
                Slope = slope,
                StartValue = yValues.First(),
                EndValue = yValues.Last(),
                PercentageChange = yMean != 0 ? ((yValues.Last() - yValues.First()) / yValues.First()) * 100 : 0
            };
        }

        private double CalculateDataCompleteness()
        {
            lock (_lockObject)
            {
                if (!_metrics.Any()) return 0;
                
                var totalExpectedPoints = _metrics.Count * 24 * 60; // Assuming minute-level data
                var totalActualPoints = _metrics.Values.Sum(m => m.DataPointCount);
                
                return Math.Min(1.0, totalActualPoints / (double)totalExpectedPoints);
            }
        }

        private double CalculateMetricStability()
        {
            lock (_lockObject)
            {
                var stabilities = new List<double>();
                
                foreach (var metric in _metrics.Values)
                {
                    var stats = metric.CalculateStatistics(DateTime.UtcNow.AddHours(-1), DateTime.UtcNow);
                    if (stats.Count > 0 && stats.Mean > 0)
                    {
                        var cv = stats.StandardDeviation / stats.Mean;
                        stabilities.Add(Math.Max(0, 1 - cv));
                    }
                }
                
                return stabilities.Any() ? stabilities.Average() : 1.0;
            }
        }

        private double CalculateAnomalyRate()
        {
            // This would calculate the actual anomaly rate from detected anomalies
            return 0.02; // 2% anomaly rate as example
        }

        // Helper classes

        public class MetricsConfiguration
        {
            public TimeSpan RetentionPeriod { get; set; } = TimeSpan.FromDays(30);
            public double CorrelationThreshold { get; set; } = 0.5;
            public double AnomalyZScoreThreshold { get; set; } = 3.0;
        }

        private class MetricTimeSeries
        {
            private readonly List<MetricDataPoint> _dataPoints = new List<MetricDataPoint>();
            private readonly object _lock = new object();
            
            public string Name { get; set; } = string.Empty;
            public string Unit { get; set; } = string.Empty;
            public MetricType Type { get; set; }
            
            public int DataPointCount => _dataPoints.Count;
            
            public void AddValue(double value, DateTime timestamp, Dictionary<string, string>? tags = null)
            {
                lock (_lock)
                {
                    _dataPoints.Add(new MetricDataPoint
                    {
                        Timestamp = timestamp,
                        Value = value,
                        Tags = tags ?? new Dictionary<string, string>()
                    });
                    
                    // Keep sorted by timestamp
                    _dataPoints.Sort((a, b) => a.Timestamp.CompareTo(b.Timestamp));
                }
            }
            
            public void TrimOldData(TimeSpan retentionPeriod)
            {
                lock (_lock)
                {
                    var cutoff = DateTime.UtcNow - retentionPeriod;
                    _dataPoints.RemoveAll(p => p.Timestamp < cutoff);
                }
            }
            
            public List<MetricDataPoint> GetDataPoints(DateTime? startTime, DateTime? endTime)
            {
                lock (_lock)
                {
                    return _dataPoints
                        .Where(p => (!startTime.HasValue || p.Timestamp >= startTime.Value) &&
                                   (!endTime.HasValue || p.Timestamp <= endTime.Value))
                        .ToList();
                }
            }
            
            public MetricStatistics CalculateStatistics(DateTime? startTime, DateTime? endTime)
            {
                var dataPoints = GetDataPoints(startTime, endTime);
                
                if (!dataPoints.Any())
                {
                    return new MetricStatistics
                    {
                        MetricName = Name,
                        Count = 0
                    };
                }
                
                var values = dataPoints.Select(p => p.Value).ToArray();
                var mean = values.Average();
                var variance = values.Select(v => Math.Pow(v - mean, 2)).Average();
                
                return new MetricStatistics
                {
                    MetricName = Name,
                    Count = values.Length,
                    Sum = values.Sum(),
                    Mean = mean,
                    Min = values.Min(),
                    Max = values.Max(),
                    StandardDeviation = Math.Sqrt(variance),
                    Percentile50 = CalculatePercentile(values, 0.5),
                    Percentile95 = CalculatePercentile(values, 0.95),
                    Percentile99 = CalculatePercentile(values, 0.99),
                    LastValue = dataPoints.Last().Value,
                    FirstValue = dataPoints.First().Value,
                    Trend = CalculateTrend(dataPoints)
                };
            }
            
            public List<AggregatedDataPoint> GetAggregatedData(DateTime startTime, DateTime endTime, TimeSpan interval)
            {
                var dataPoints = GetDataPoints(startTime, endTime);
                var aggregated = new List<AggregatedDataPoint>();
                
                var currentBucket = startTime;
                while (currentBucket < endTime)
                {
                    var bucketEnd = currentBucket + interval;
                    var bucketPoints = dataPoints
                        .Where(p => p.Timestamp >= currentBucket && p.Timestamp < bucketEnd)
                        .Select(p => p.Value)
                        .ToList();
                    
                    if (bucketPoints.Any())
                    {
                        aggregated.Add(new AggregatedDataPoint
                        {
                            Timestamp = currentBucket,
                            Count = bucketPoints.Count,
                            Sum = bucketPoints.Sum(),
                            Average = bucketPoints.Average(),
                            Min = bucketPoints.Min(),
                            Max = bucketPoints.Max()
                        });
                    }
                    
                    currentBucket = bucketEnd;
                }
                
                return aggregated;
            }
            
            private double CalculatePercentile(double[] sortedValues, double percentile)
            {
                if (sortedValues.Length == 0) return 0;
                
                Array.Sort(sortedValues);
                var index = (int)Math.Ceiling(percentile * sortedValues.Length) - 1;
                return sortedValues[Math.Max(0, Math.Min(index, sortedValues.Length - 1))];
            }
            
            private string CalculateTrend(List<MetricDataPoint> dataPoints)
            {
                if (dataPoints.Count < 2) return "Unknown";
                
                var firstHalf = dataPoints.Take(dataPoints.Count / 2).Select(p => p.Value).Average();
                var secondHalf = dataPoints.Skip(dataPoints.Count / 2).Select(p => p.Value).Average();
                
                var change = (secondHalf - firstHalf) / firstHalf;
                
                if (change > 0.05) return "Increasing";
                if (change < -0.05) return "Decreasing";
                return "Stable";
            }
        }

        public enum MetricType
        {
            Counter,
            Gauge,
            Histogram
        }

        public class MetricDataPoint
        {
            public DateTime Timestamp { get; set; }
            public double Value { get; set; }
            public Dictionary<string, string> Tags { get; set; } = new Dictionary<string, string>();
        }

        public class MetricStatistics
        {
            public string MetricName { get; set; } = string.Empty;
            public int Count { get; set; }
            public double Sum { get; set; }
            public double Mean { get; set; }
            public double Min { get; set; }
            public double Max { get; set; }
            public double StandardDeviation { get; set; }
            public double Percentile50 { get; set; }
            public double Percentile95 { get; set; }
            public double Percentile99 { get; set; }
            public double LastValue { get; set; }
            public double FirstValue { get; set; }
            public string Trend { get; set; } = string.Empty;
        }

        public class AggregatedDataPoint
        {
            public DateTime Timestamp { get; set; }
            public int Count { get; set; }
            public double Sum { get; set; }
            public double Average { get; set; }
            public double Min { get; set; }
            public double Max { get; set; }
        }

        public class AggregatedMetrics
        {
            public DateTime StartTime { get; set; }
            public DateTime EndTime { get; set; }
            public TimeSpan AggregationInterval { get; set; }
            public Dictionary<string, List<AggregatedDataPoint>> Metrics { get; set; } = new Dictionary<string, List<AggregatedDataPoint>>();
        }

        public class MetricCorrelations
        {
            public List<MetricCorrelation> SignificantCorrelations { get; set; } = new List<MetricCorrelation>();
            public DateTime CalculatedAt { get; set; }
        }

        public class MetricCorrelation
        {
            public string Metric1 { get; set; } = string.Empty;
            public string Metric2 { get; set; } = string.Empty;
            public double CorrelationCoefficient { get; set; }
            public string Strength { get; set; } = string.Empty;
        }

        public class MetricAnomalies
        {
            public List<MetricAnomaly> Anomalies { get; set; } = new List<MetricAnomaly>();
            public DateTime DetectionTimestamp { get; set; }
        }

        public class MetricAnomaly
        {
            public string MetricName { get; set; } = string.Empty;
            public DateTime Timestamp { get; set; }
            public double Value { get; set; }
            public double ExpectedValue { get; set; }
            public double AnomalyScore { get; set; }
            public string Type { get; set; } = string.Empty;
            public string Severity { get; set; } = string.Empty;
        }

        public class MetricTrends
        {
            public Dictionary<string, TrendAnalysis> Trends { get; set; } = new Dictionary<string, TrendAnalysis>();
            public DateTime AnalysisTimestamp { get; set; }
            public TimeSpan LookbackPeriod { get; set; }
        }

        public class TrendAnalysis
        {
            public string Direction { get; set; } = string.Empty;
            public double Strength { get; set; }
            public double Slope { get; set; }
            public double StartValue { get; set; }
            public double EndValue { get; set; }
            public double PercentageChange { get; set; }
        }

        public class MetricsExport
        {
            public DateTime ExportTimestamp { get; set; }
            public DateTime StartTime { get; set; }
            public DateTime EndTime { get; set; }
            public Dictionary<string, List<MetricDataPoint>> Metrics { get; set; } = new Dictionary<string, List<MetricDataPoint>>();
            public int TotalDataPoints { get; set; }
        }

        public class MetricsDashboard
        {
            public DateTime Timestamp { get; set; }
            public List<MetricSummary> MetricSummaries { get; set; } = new List<MetricSummary>();
            public Dictionary<string, double> HealthIndicators { get; set; } = new Dictionary<string, double>();
            public List<string> RecentAlerts { get; set; } = new List<string>();
        }

        public class MetricSummary
        {
            public string MetricName { get; set; } = string.Empty;
            public double CurrentValue { get; set; }
            public double Average { get; set; }
            public double Min { get; set; }
            public double Max { get; set; }
            public string Trend { get; set; } = string.Empty;
            public string Unit { get; set; } = string.Empty;
        }
    }
}
