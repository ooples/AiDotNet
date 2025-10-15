using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.LinearAlgebra;
using AiDotNet.Helpers;
using AiDotNet.Models;

namespace AiDotNet.ProductionMonitoring
{
    /// <summary>
    /// Production monitoring system for ML models
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations</typeparam>
    public class DefaultProductionMonitor<T> : IDisposable
        where T : struct, IComparable<T>, IConvertible, IEquatable<T>
    {
        private readonly DataDriftDetector<T> driftDetector;
        private readonly PerformanceMonitor<T> performanceMonitor;
        private readonly AlertManager alertManager;
        private readonly ModelHealthScorer<T> healthScorer;
        private readonly RetrainingRecommender<T> retrainingRecommender;
        private readonly INumericOperations<T> ops;
        
        private readonly List<DataDriftAlert> recentAlerts;
        private readonly object lockObject = new object();
        
        public DefaultProductionMonitor()
        {
            driftDetector = new DataDriftDetector<T>();
            performanceMonitor = new PerformanceMonitor<T>();
            alertManager = new AlertManager();
            healthScorer = new ModelHealthScorer<T>();
            retrainingRecommender = new RetrainingRecommender<T>();
            ops = MathHelper.GetNumericOperations<T>();
            recentAlerts = new List<DataDriftAlert>();
        }
        
        public async Task MonitorInputDataAsync(Tensor<T> inputs)
        {
            await Task.Run(() =>
            {
                // Check for data drift
                var inputArray = new double[inputs.Length];
                for (int i = 0; i < inputs.Length; i++)
                {
                    inputArray[i] = Convert.ToDouble(inputs[i]);
                }
                
                // Convert input array to Matrix<T>
                var inputMatrix = new Matrix<T>(new T[] { inputs }, 1, inputs.Length);
                var driftResult = await driftDetector.DetectDataDriftAsync(inputMatrix);
                var driftScores = driftResult.FeatureDrifts ?? new Dictionary<string, double>();
                
                // Generate alerts for significant drift
                foreach (var score in driftScores.Where(s => s.Value > 0.1))
                {
                    var alert = new DataDriftAlert
                    {
                        FeatureName = $"Feature_{score.Key}",
                        DriftScore = score.Value,
                        Severity = score.Value > 0.3 ? "High" : "Medium",
                        Message = $"Data drift detected in Feature_{score.Key}: score={score.Value:F3}"
                    };
                    
                    lock (lockObject)
                    {
                        recentAlerts.Add(alert);
                    }
                    
                    await alertManager.SendAlertAsync(alert);
                }
            });
        }
        
        public async Task MonitorPredictionsAsync(Tensor<T> predictions)
        {
            await Task.Run(() =>
            {
                // Monitor prediction distribution
                var predArray = new double[predictions.Length];
                for (int i = 0; i < predictions.Length; i++)
                {
                    predArray[i] = Convert.ToDouble(predictions[i]);
                }
                
                var distribution = ComputePredictionDistribution(predArray);
                
                // Check for anomalies
                if (distribution.Skewness > 2.0 || distribution.Kurtosis > 7.0)
                {
                    var alert = new DataDriftAlert
                    {
                        FeatureName = "Predictions",
                        DriftScore = Math.Max(distribution.Skewness / 2.0, distribution.Kurtosis / 7.0),
                        Severity = "High",
                        Message = $"Abnormal prediction distribution detected: skewness={distribution.Skewness:F2}, kurtosis={distribution.Kurtosis:F2}"
                    };
                    
                    lock (lockObject)
                    {
                        recentAlerts.Add(alert);
                    }
                    
                    await alertManager.SendAlertAsync(alert);
                }
                
                // Update performance metrics
                performanceMonitor.RecordPrediction(predArray);
            });
        }
        
        public List<DataDriftAlert> GetRecentAlerts()
        {
            lock (lockObject)
            {
                // Return alerts from the last 24 hours
                var cutoff = DateTime.UtcNow.AddHours(-24);
                return recentAlerts.Where(a => a.Timestamp >= cutoff).ToList();
            }
        }
        
        public bool GetRetrainingRecommendation()
        {
            var metrics = new MonitoringMetrics
            {
                DriftScore = driftDetector.GetOverallDriftScore(),
                PerformanceDegradation = performanceMonitor.GetPerformanceDegradation(),
                TimeSinceLastRetrain = DateTime.UtcNow.AddDays(-30), // Example
                PredictionVolume = performanceMonitor.GetPredictionVolume()
            };
            
            return retrainingRecommender.ShouldRetrain(metrics);
        }
        
        public double GetHealthScore()
        {
            var metrics = new MonitoringMetrics
            {
                DriftScore = driftDetector.GetOverallDriftScore(),
                PerformanceDegradation = performanceMonitor.GetPerformanceDegradation(),
                AlertCount = GetRecentAlerts().Count,
                ResponseTime = performanceMonitor.GetAverageResponseTime()
            };
            
            return healthScorer.CalculateHealthScore(metrics);
        }
        
        private PredictionDistribution ComputePredictionDistribution(double[] predictions)
        {
            var dist = new PredictionDistribution();
            
            // Calculate statistics
            dist.Mean = predictions.Average();
            dist.StandardDeviation = Math.Sqrt(predictions.Select(p => Math.Pow(p - dist.Mean, 2)).Average());
            
            // Calculate skewness
            var n = predictions.Length;
            var m3 = predictions.Select(p => Math.Pow((p - dist.Mean) / dist.StandardDeviation, 3)).Sum() / n;
            dist.Skewness = m3;
            
            // Calculate kurtosis
            var m4 = predictions.Select(p => Math.Pow((p - dist.Mean) / dist.StandardDeviation, 4)).Sum() / n;
            dist.Kurtosis = m4 - 3; // Excess kurtosis
            
            // Create histogram
            var min = predictions.Min();
            var max = predictions.Max();
            var binCount = Math.Min(20, (int)Math.Sqrt(n));
            var binWidth = (max - min) / binCount;
            
            dist.BinEdges = new List<double>();
            dist.BinCounts = new List<int>(new int[binCount]);
            
            for (int i = 0; i <= binCount; i++)
            {
                dist.BinEdges.Add(min + i * binWidth);
            }
            
            foreach (var pred in predictions)
            {
                var binIdx = Math.Min((int)((pred - min) / binWidth), binCount - 1);
                dist.BinCounts[binIdx]++;
            }
            
            return dist;
        }
        
        public void Dispose()
        {
            // Check if components implement IDisposable before disposing
            if (driftDetector is IDisposable disposableDrift)
                disposableDrift.Dispose();
            if (performanceMonitor is IDisposable disposablePerf)
                disposablePerf.Dispose();
            if (alertManager is IDisposable disposableAlert)
                disposableAlert.Dispose();
            // healthScorer and retrainingRecommender don't implement IDisposable
        }
    }
}