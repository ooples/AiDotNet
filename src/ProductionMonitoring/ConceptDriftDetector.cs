using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AiDotNet.ProductionMonitoring
{
    /// <summary>
    /// Detects concept drift in model predictions over time
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations</typeparam>
    public class ConceptDriftDetector<T> : ProductionMonitorBase<T>
    {
        private readonly ConceptDriftMethod _method = default!;
        private readonly int _windowSize;
        private readonly Queue<T> _errorWindow;
        private readonly Dictionary<string, DriftDetectorState> _detectorStates;

        public enum ConceptDriftMethod
        {
            ADWIN,           // Adaptive Windowing
            DDM,             // Drift Detection Method
            EDDM,            // Early Drift Detection Method
            PageHinkley,     // Page-Hinkley Test
            KSWIN,           // Kolmogorov-Smirnov Windowing
            HDDM_A,          // Hoeffding Drift Detection Method A
            HDDM_W           // Hoeffding Drift Detection Method W
        }

        public ConceptDriftDetector(ConceptDriftMethod method = ConceptDriftMethod.ADWIN, int windowSize = 100)
        {
            _method = method;
            _windowSize = windowSize;
            _errorWindow = new Queue<T>();
            _detectorStates = new Dictionary<string, DriftDetectorState>();
            InitializeDetectorStates();
        }

        /// <summary>
        /// Detects concept drift in model predictions
        /// </summary>
        public override async Task<DriftDetectionResult> DetectConceptDriftAsync(Vector<T> predictions, Vector<T> actuals)
        {
            if (predictions.Length != actuals.Length)
            {
                throw new ArgumentException("Predictions and actuals must have the same length");
            }

            // Calculate errors
            var errors = predictions.Zip(actuals, (p, a) => Math.Abs(p - a)).ToArray();
            
            // Update error window
            foreach (var error in errors)
            {
                _errorWindow.Enqueue(error);
                if (_errorWindow.Count > _windowSize)
                {
                    _errorWindow.Dequeue();
                }
            }

            if (_errorWindow.Count < _thresholds.MinimumSamplesForDriftDetection)
            {
                return new DriftDetectionResult
                {
                    IsDriftDetected = false,
                    DriftScore = 0,
                    DriftType = "ConceptDrift",
                    Details = $"Insufficient samples for drift detection. Current: {_errorWindow.Count}, Required: {_thresholds.MinimumSamplesForDriftDetection}",
                    DetectionTimestamp = DateTime.UtcNow
                };
            }

            // Detect drift using selected method
            var driftResult = await DetectDriftAsync(errors);
            
            // Store result
            lock (_lockObject)
            {
                _driftHistory.Add(driftResult);
            }

            // Send alert if drift detected
            if (driftResult.IsDriftDetected)
            {
                SendAlert(new MonitoringAlert
                {
                    AlertType = "ConceptDrift",
                    Severity = driftResult.DriftScore > _thresholds.ConceptDriftThreshold * 2 ? "Critical" : "Warning",
                    Message = $"Concept drift detected using {_method} method",
                    Timestamp = DateTime.UtcNow,
                    Context = new Dictionary<string, object>
                    {
                        ["Method"] = _method.ToString(),
                        ["DriftScore"] = driftResult.DriftScore,
                        ["WindowSize"] = _windowSize,
                        ["ErrorRate"] = _errorWindow.Average()
                    }
                });
            }

            return driftResult;
        }

        /// <summary>
        /// Detects data drift (delegates to DataDriftDetector for specialized handling)
        /// </summary>
        public override async Task<DriftDetectionResult> DetectDataDriftAsync(Matrix<T> productionData, Matrix<T>? referenceData = null)
        {
            // For concept drift detector, we focus on prediction errors rather than feature distributions
            // This is a simplified implementation
            return Task.FromResult(new DriftDetectionResult
            {
                IsDriftDetected = false,
                DriftScore = 0,
                DriftType = "DataDrift",
                Details = "Use DataDriftDetector for comprehensive data drift detection",
                DetectionTimestamp = DateTime.UtcNow
            });
        }

        /// <summary>
        /// Gets model health score based on concept drift
        /// </summary>
        public override async Task<ModelHealthScore> GetModelHealthScoreAsync()
        {
            var recentDrifts = _driftHistory.Where(d => d.DetectionTimestamp > DateTime.UtcNow.AddDays(-7)).ToList();
            var performance = await GetPerformanceMetricsAsync(DateTime.UtcNow.AddDays(-7), DateTime.UtcNow);
            
            // Calculate drift-based health score
            var driftScore = 1.0;
            if (recentDrifts.Any())
            {
                var maxDrift = recentDrifts.Max(d => d.DriftScore);
                driftScore = Math.Max(0, 1.0 - maxDrift);
            }
            
            // Calculate stability based on error variance
            var stabilityScore = CalculateErrorStability();
            
            // Performance score
            var performanceScore = performance.PredictionCount > 0 ? 
                (performance.Accuracy + performance.F1Score) / 2.0 : 1.0;
            
            // Overall health
            var overallScore = (driftScore * 0.4 + stabilityScore * 0.3 + performanceScore * 0.3);
            
            var healthStatus = overallScore >= 0.8 ? "Healthy" :
                              overallScore >= 0.6 ? "Warning" : "Critical";
            
            var issues = new List<string>();
            if (driftScore < 0.7) issues.Add("Concept drift detected");
            if (stabilityScore < 0.7) issues.Add("High prediction variance");
            if (performanceScore < 0.7) issues.Add("Poor model performance");
            
            return new ModelHealthScore
            {
                OverallScore = overallScore,
                DataQualityScore = 1.0, // Not applicable for concept drift
                PerformanceScore = performanceScore,
                StabilityScore = stabilityScore,
                DriftScore = driftScore,
                HealthStatus = healthStatus,
                Issues = issues,
                EvaluationTimestamp = DateTime.UtcNow
            };
        }

        /// <summary>
        /// Gets retraining recommendation based on concept drift
        /// </summary>
        public override async Task<RetrainingRecommendation> GetRetrainingRecommendationAsync()
        {
            var recentDrifts = _driftHistory
                .Where(d => d.DetectionTimestamp > DateTime.UtcNow.AddDays(-30))
                .Where(d => d.IsDriftDetected)
                .ToList();
            
            var healthScore = await GetModelHealthScoreAsync();
            
            var reasons = new List<string>();
            var shouldRetrain = false;
            var urgency = "Low";
            
            // Check drift frequency
            if (recentDrifts.Count > 10)
            {
                shouldRetrain = true;
                urgency = "Critical";
                reasons.Add($"Frequent concept drift: {recentDrifts.Count} instances in 30 days");
            }
            else if (recentDrifts.Count > 5)
            {
                shouldRetrain = true;
                urgency = "High";
                reasons.Add($"Moderate concept drift: {recentDrifts.Count} instances in 30 days");
            }
            
            // Check drift severity
            if (recentDrifts.Any(d => d.DriftScore > _thresholds.ConceptDriftThreshold * 2))
            {
                shouldRetrain = true;
                if (urgency == "Low") urgency = "High";
                reasons.Add("Severe concept drift detected");
            }
            
            // Check health score
            if (healthScore.OverallScore < 0.6)
            {
                shouldRetrain = true;
                if (urgency != "Critical") urgency = "High";
                reasons.Add($"Poor model health: {healthScore.OverallScore:F2}");
            }
            
            // Calculate confidence
            var confidence = CalculateRetrainingConfidence(recentDrifts, healthScore);
            
            return new RetrainingRecommendation
            {
                ShouldRetrain = shouldRetrain,
                Urgency = urgency,
                Reasons = reasons,
                RecommendationTimestamp = DateTime.UtcNow,
                ConfidenceScore = confidence,
                SuggestedActions = new Dictionary<string, object>
                {
                    ["OnlineLearning"] = urgency == "High" || urgency == "Critical",
                    ["IncrementalUpdate"] = recentDrifts.Count > 3,
                    ["FullRetrain"] = urgency == "Critical",
                    ["AdjustLearningRate"] = recentDrifts.Any()
                }
            };
        }

        // Private methods

        private void InitializeDetectorStates()
        {
            _detectorStates["DDM"] = new DDMState();
            _detectorStates["EDDM"] = new EDDMState();
            _detectorStates["PageHinkley"] = new PageHinkleyState();
            _detectorStates["ADWIN"] = new ADWINState();
            _detectorStates["KSWIN"] = new KSWINState();
            _detectorStates["HDDM"] = new HDDMState();
        }

        private async Task<DriftDetectionResult> DetectDriftAsync(double[] errors)
        {
            return await Task.Run(() =>
            {
                switch (_method)
                {
                    case ConceptDriftMethod.ADWIN:
                        return DetectDriftADWIN(errors);
                    
                    case ConceptDriftMethod.DDM:
                        return DetectDriftDDM(errors);
                    
                    case ConceptDriftMethod.EDDM:
                        return DetectDriftEDDM(errors);
                    
                    case ConceptDriftMethod.PageHinkley:
                        return DetectDriftPageHinkley(errors);
                    
                    case ConceptDriftMethod.KSWIN:
                        return DetectDriftKSWIN(errors);
                    
                    case ConceptDriftMethod.HDDM_A:
                    case ConceptDriftMethod.HDDM_W:
                        return DetectDriftHDDM(errors, _method == ConceptDriftMethod.HDDM_A);
                    
                    default:
                        return DetectDriftADWIN(errors);
                }
            });
        }

        private DriftDetectionResult DetectDriftADWIN(double[] errors)
        {
            var state = (ADWINState)_detectorStates["ADWIN"];
            var isDrift = false;
            double maxDrift = 0;
            
            foreach (var error in errors)
            {
                state.Window.Add(error);
                
                // ADWIN algorithm
                if (state.Window.Count > 4)
                {
                    var mean = state.Window.Average();
                    var variance = state.Window.Select(e => Math.Pow(e - mean, 2)).Average();
                    
                    // Check for significant change in window
                    for (int cutPoint = state.Window.Count / 4; cutPoint < 3 * state.Window.Count / 4; cutPoint++)
                    {
                        var w1 = state.Window.Take(cutPoint).ToList();
                        var w2 = state.Window.Skip(cutPoint).ToList();
                        
                        var mean1 = w1.Average();
                        var mean2 = w2.Average();
                        
                        var epsilon = Math.Sqrt(2 * Math.Log(2 * state.Window.Count / state.Delta) / state.Window.Count);
                        var threshold = epsilon * Math.Sqrt(variance);
                        
                        var drift = Math.Abs(mean1 - mean2);
                        if (drift > threshold)
                        {
                            isDrift = true;
                            maxDrift = Math.Max(maxDrift, drift / threshold);
                            
                            // Remove old data
                            state.Window.RemoveRange(0, cutPoint);
                            break;
                        }
                    }
                }
                
                // Limit window size
                if (state.Window.Count > _windowSize * 2)
                {
                    state.Window.RemoveAt(0);
                }
            }
            
            return new DriftDetectionResult
            {
                IsDriftDetected = isDrift,
                DriftScore = maxDrift,
                DriftType = "ConceptDrift",
                Details = $"ADWIN detected drift with score {maxDrift:F4}",
                DetectionTimestamp = DateTime.UtcNow
            };
        }

        private DriftDetectionResult DetectDriftDDM(double[] errors)
        {
            var state = (DDMState)_detectorStates["DDM"];
            var isDrift = false;
            double maxDrift = 0;
            
            foreach (var error in errors)
            {
                state.NumSamples++;
                state.ErrorSum += error;
                state.ErrorSquaredSum += error * error;
                
                if (state.NumSamples < 30) continue;
                
                var errorRate = state.ErrorSum / state.NumSamples;
                var variance = (state.ErrorSquaredSum / state.NumSamples) - (errorRate * errorRate);
                var stdDev = Math.Sqrt(variance);
                
                if (state.MinErrorRate + state.MinStdDev == 0)
                {
                    state.MinErrorRate = errorRate;
                    state.MinStdDev = stdDev;
                }
                
                if (errorRate + stdDev <= state.MinErrorRate + state.MinStdDev)
                {
                    state.MinErrorRate = errorRate;
                    state.MinStdDev = stdDev;
                }
                
                var level = (errorRate + stdDev - state.MinErrorRate - state.MinStdDev) / (state.MinErrorRate + state.MinStdDev);
                
                if (level > state.DriftLevel)
                {
                    isDrift = true;
                    maxDrift = level;
                }
                else if (level > state.WarningLevel)
                {
                    maxDrift = Math.Max(maxDrift, level);
                }
            }
            
            return new DriftDetectionResult
            {
                IsDriftDetected = isDrift,
                DriftScore = maxDrift,
                DriftType = "ConceptDrift",
                Details = $"DDM detected {(isDrift ? "drift" : "no drift")} with level {maxDrift:F4}",
                DetectionTimestamp = DateTime.UtcNow
            };
        }

        private DriftDetectionResult DetectDriftEDDM(double[] errors)
        {
            var state = (EDDMState)_detectorStates["EDDM"];
            var isDrift = false;
            double maxDrift = 0;
            
            foreach (var error in errors)
            {
                state.NumSamples++;
                
                if (error > 0 && state.LastError > 0)
                {
                    var distance = state.NumSamples - state.LastErrorPosition;
                    state.DistanceSum += distance;
                    state.DistanceSquaredSum += distance * distance;
                    state.NumErrors++;
                    
                    if (state.NumErrors >= 30)
                    {
                        var meanDistance = state.DistanceSum / state.NumErrors;
                        var variance = (state.DistanceSquaredSum / state.NumErrors) - (meanDistance * meanDistance);
                        var stdDev = Math.Sqrt(variance);
                        
                        if (state.MaxMeanDistance + state.MaxStdDev == 0)
                        {
                            state.MaxMeanDistance = meanDistance;
                            state.MaxStdDev = stdDev;
                        }
                        
                        if (meanDistance + 2 * stdDev >= state.MaxMeanDistance + 2 * state.MaxStdDev)
                        {
                            state.MaxMeanDistance = meanDistance;
                            state.MaxStdDev = stdDev;
                        }
                        
                        var level = (meanDistance + 2 * stdDev) / (state.MaxMeanDistance + 2 * state.MaxStdDev);
                        
                        if (level < state.Alpha)
                        {
                            isDrift = true;
                            maxDrift = 1.0 - level;
                        }
                        else if (level < state.Beta)
                        {
                            maxDrift = Math.Max(maxDrift, 1.0 - level);
                        }
                    }
                }
                
                if (error > 0)
                {
                    state.LastError = error;
                    state.LastErrorPosition = state.NumSamples;
                }
            }
            
            return new DriftDetectionResult
            {
                IsDriftDetected = isDrift,
                DriftScore = maxDrift,
                DriftType = "ConceptDrift",
                Details = $"EDDM detected {(isDrift ? "drift" : "no drift")} with score {maxDrift:F4}",
                DetectionTimestamp = DateTime.UtcNow
            };
        }

        private DriftDetectionResult DetectDriftPageHinkley(double[] errors)
        {
            var state = (PageHinkleyState)_detectorStates["PageHinkley"];
            var isDrift = false;
            double maxDrift = 0;
            
            foreach (var error in errors)
            {
                state.Sum += error - state.Mean - state.Delta;
                state.Mean = (state.Mean * state.NumSamples + error) / (state.NumSamples + 1);
                state.NumSamples++;
                
                if (state.Sum < state.MinSum)
                {
                    state.MinSum = state.Sum;
                }
                
                var ph = state.Sum - state.MinSum;
                
                if (ph > state.Lambda)
                {
                    isDrift = true;
                    maxDrift = ph / state.Lambda;
                    
                    // Reset
                    state.Sum = 0;
                    state.MinSum = 0;
                }
                else
                {
                    maxDrift = Math.Max(maxDrift, ph / state.Lambda);
                }
            }
            
            return new DriftDetectionResult
            {
                IsDriftDetected = isDrift,
                DriftScore = maxDrift,
                DriftType = "ConceptDrift",
                Details = $"Page-Hinkley test detected {(isDrift ? "drift" : "no drift")} with PH statistic {maxDrift:F4}",
                DetectionTimestamp = DateTime.UtcNow
            };
        }

        private DriftDetectionResult DetectDriftKSWIN(double[] errors)
        {
            var state = (KSWINState)_detectorStates["KSWIN"];
            var isDrift = false;
            double maxDrift = 0;
            
            foreach (var error in errors)
            {
                state.Window.Add(error);
                
                if (state.Window.Count > state.WindowSize)
                {
                    state.Window.RemoveAt(0);
                }
                
                if (state.Window.Count == state.WindowSize)
                {
                    var midPoint = state.WindowSize / 2;
                    var window1 = state.Window.Take(midPoint).ToArray();
                    var window2 = state.Window.Skip(midPoint).ToArray();
                    
                    // Kolmogorov-Smirnov test
                    var ksStatistic = CalculateKSStatistic(window1, window2);
                    
                    if (ksStatistic > state.Alpha)
                    {
                        isDrift = true;
                        maxDrift = ksStatistic;
                        
                        // Keep only recent window
                        state.Window = state.Window.Skip(midPoint).ToList();
                    }
                    else
                    {
                        maxDrift = Math.Max(maxDrift, ksStatistic);
                    }
                }
            }
            
            return new DriftDetectionResult
            {
                IsDriftDetected = isDrift,
                DriftScore = maxDrift,
                DriftType = "ConceptDrift",
                Details = $"KSWIN detected {(isDrift ? "drift" : "no drift")} with KS statistic {maxDrift:F4}",
                DetectionTimestamp = DateTime.UtcNow
            };
        }

        private DriftDetectionResult DetectDriftHDDM(double[] errors, bool useMovingAverage)
        {
            var state = (HDDMState)_detectorStates["HDDM"];
            var isDrift = false;
            double maxDrift = 0;
            
            foreach (var error in errors)
            {
                state.NumSamples++;
                
                if (useMovingAverage)
                {
                    // HDDM-A: Moving average
                    state.MovingAverage.Add(error);
                    if (state.MovingAverage.Count > 10)
                    {
                        state.MovingAverage.RemoveAt(0);
                    }
                    
                    var currentMean = state.MovingAverage.Average();
                    
                    if (state.NumSamples > 10)
                    {
                        var diff = Math.Abs(currentMean - state.EstimatedMean);
                        var bound = state.ConfidenceLevel * Math.Sqrt(2 * Math.Log(2 / state.Delta) / state.NumSamples);
                        
                        if (diff > bound)
                        {
                            isDrift = true;
                            maxDrift = diff / bound;
                        }
                        else
                        {
                            maxDrift = Math.Max(maxDrift, diff / bound);
                        }
                    }
                    
                    state.EstimatedMean = currentMean;
                }
                else
                {
                    // HDDM-W: Weighted average
                    var weight = 1.0 / state.NumSamples;
                    state.EstimatedMean = (1 - weight) * state.EstimatedMean + weight * error;
                    
                    if (state.NumSamples > 10)
                    {
                        var diff = Math.Abs(error - state.EstimatedMean);
                        var bound = state.ConfidenceLevel * Math.Sqrt(2 * Math.Log(2 / state.Delta) / state.NumSamples);
                        
                        if (diff > bound)
                        {
                            isDrift = true;
                            maxDrift = diff / bound;
                        }
                        else
                        {
                            maxDrift = Math.Max(maxDrift, diff / bound);
                        }
                    }
                }
            }
            
            return new DriftDetectionResult
            {
                IsDriftDetected = isDrift,
                DriftScore = maxDrift,
                DriftType = "ConceptDrift",
                Details = $"HDDM-{(useMovingAverage ? "A" : "W")} detected {(isDrift ? "drift" : "no drift")} with score {maxDrift:F4}",
                DetectionTimestamp = DateTime.UtcNow
            };
        }

        private double CalculateKSStatistic(double[] sample1, double[] sample2)
        {
            var sorted1 = sample1.OrderBy(x => x).ToArray();
            var sorted2 = sample2.OrderBy(x => x).ToArray();
            
            double maxDiff = 0;
            var allValues = sorted1.Concat(sorted2).Distinct().OrderBy(x => x).ToArray();
            
            foreach (var value in allValues)
            {
                var cdf1 = sorted1.Count(x => x <= value) / (double)sorted1.Length;
                var cdf2 = sorted2.Count(x => x <= value) / (double)sorted2.Length;
                maxDiff = Math.Max(maxDiff, Math.Abs(cdf1 - cdf2));
            }
            
            return maxDiff;
        }

        private double CalculateErrorStability()
        {
            if (_errorWindow.Count < 10) return 1.0;
            
            var errors = _errorWindow.ToArray();
            var mean = errors.Average();
            var variance = errors.Select(e => Math.Pow(e - mean, 2)).Average();
            var cv = Math.Sqrt(variance) / mean;
            
            return Math.Max(0, 1.0 - cv);
        }

        private double CalculateRetrainingConfidence(List<DriftDetectionResult> drifts, ModelHealthScore healthScore)
        {
            if (!drifts.Any()) return 0.1;
            
            var driftFrequency = drifts.Count / 30.0; // Normalized by days
            var avgDriftScore = drifts.Average(d => d.DriftScore);
            var healthPenalty = 1.0 - healthScore.OverallScore;
            
            return Math.Min(1.0, driftFrequency * 0.3 + avgDriftScore * 0.4 + healthPenalty * 0.3);
        }

        // Detector state classes

        private abstract class DriftDetectorState { }

        private class DDMState : DriftDetectorState
        {
            public int NumSamples { get; set; }
            public double ErrorSum { get; set; }
            public double ErrorSquaredSum { get; set; }
            public double MinErrorRate { get; set; }
            public double MinStdDev { get; set; }
            public double WarningLevel { get; set; } = 2.0;
            public double DriftLevel { get; set; } = 3.0;
        }

        private class EDDMState : DriftDetectorState
        {
            public int NumSamples { get; set; }
            public int NumErrors { get; set; }
            public double DistanceSum { get; set; }
            public double DistanceSquaredSum { get; set; }
            public double MaxMeanDistance { get; set; }
            public double MaxStdDev { get; set; }
            public double LastError { get; set; }
            public int LastErrorPosition { get; set; }
            public double Alpha { get; set; } = 0.95;
            public double Beta { get; set; } = 0.9;
        }

        private class PageHinkleyState : DriftDetectorState
        {
            public double Sum { get; set; }
            public double MinSum { get; set; }
            public double Mean { get; set; }
            public int NumSamples { get; set; }
            public double Delta { get; set; } = 0.005;
            public double Lambda { get; set; } = 50;
        }

        private class ADWINState : DriftDetectorState
        {
            public List<double> Window { get; set; } = new List<double>();
            public double Delta { get; set; } = 0.002;
        }

        private class KSWINState : DriftDetectorState
        {
            public List<double> Window { get; set; } = new List<double>();
            public int WindowSize { get; set; } = 100;
            public double Alpha { get; set; } = 0.05;
        }

        private class HDDMState : DriftDetectorState
        {
            public int NumSamples { get; set; }
            public double EstimatedMean { get; set; }
            public List<double> MovingAverage { get; set; } = new List<double>();
            public double ConfidenceLevel { get; set; } = 3.0;
            public double Delta { get; set; } = 0.001;
        }
    }
}