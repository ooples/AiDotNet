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
    /// Provides intelligent retraining recommendations based on comprehensive monitoring data
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations</typeparam>
    public class RetrainingRecommender<T> : ProductionMonitorBase<T>
    {
        private readonly RecommenderConfiguration _configuration = default!;
        private readonly List<RetrainingEvent> _retrainingHistory = default!;
        private readonly Dictionary<string, RetrainingStrategy> _strategies = default!;
        private readonly Dictionary<string, double> _featureImportances = default!;
        
        public RetrainingRecommender(RecommenderConfiguration? configuration = null)
        {
            _configuration = configuration ?? new RecommenderConfiguration();
            _retrainingHistory = new List<RetrainingEvent>();
            _strategies = new Dictionary<string, RetrainingStrategy>();
            _featureImportances = new Dictionary<string, double>();
            
            InitializeStrategies();
        }

        /// <summary>
        /// Updates feature importances for better recommendations
        /// </summary>
        public void UpdateFeatureImportances(Dictionary<string, double> importances)
        {
            lock (_lockObject)
            {
                foreach (var kvp in importances)
                {
                    _featureImportances[kvp.Key] = kvp.Value;
                }
            }
        }

        /// <summary>
        /// Records a retraining event
        /// </summary>
        public void RecordRetrainingEvent(RetrainingEvent retrainingEvent)
        {
            lock (_lockObject)
            {
                _retrainingHistory.Add(retrainingEvent);
                
                // Keep only recent history
                var cutoff = DateTime.UtcNow.AddDays(-_configuration.HistoryRetentionDays);
                _retrainingHistory.RemoveAll(e => e.Timestamp < cutoff);
            }
        }

        /// <summary>
        /// Gets comprehensive retraining recommendation
        /// </summary>
        public override async Task<RetrainingRecommendation> GetRetrainingRecommendationAsync()
        {
            // Gather all monitoring data
            var monitoringData = await GatherMonitoringDataAsync();
            
            // Analyze each strategy
            var strategyResults = new Dictionary<string, StrategyEvaluation>();
            foreach (var strategy in _strategies.Values)
            {
                strategyResults[strategy.Name] = await EvaluateStrategyAsync(strategy, monitoringData);
            }
            
            // Select best strategy
            var bestStrategy = SelectBestStrategy(strategyResults);
            
            // Generate detailed recommendation
            var recommendation = await GenerateDetailedRecommendationAsync(bestStrategy, monitoringData);
            
            // Add historical context
            EnrichWithHistoricalContext(recommendation);
            
            return recommendation;
        }

        /// <summary>
        /// Gets retraining schedule recommendations
        /// </summary>
        public async Task<RetrainingSchedule> GetRetrainingScheduleAsync()
        {
            var monitoringData = await GatherMonitoringDataAsync();
            var schedule = new RetrainingSchedule();
            
            // Analyze patterns in historical retraining
            var patterns = AnalyzeRetrainingPatterns();
            
            // Predict optimal retraining times
            schedule.RecommendedSchedule = PredictOptimalSchedule(monitoringData, patterns);
            
            // Calculate expected benefits
            schedule.ExpectedBenefits = CalculateExpectedBenefits(schedule.RecommendedSchedule, monitoringData);
            
            // Estimate resource requirements
            schedule.ResourceRequirements = EstimateResourceRequirements(schedule.RecommendedSchedule);
            
            schedule.GeneratedAt = DateTime.UtcNow;
            schedule.ValidUntil = DateTime.UtcNow.AddDays(7);
            
            return schedule;
        }

        /// <summary>
        /// Evaluates retraining ROI (Return on Investment)
        /// </summary>
        public async Task<RetrainingROI> EvaluateRetrainingROIAsync()
        {
            var monitoringData = await GatherMonitoringDataAsync();
            var roi = new RetrainingROI();
            
            // Calculate current performance metrics
            roi.CurrentPerformance = monitoringData.CurrentPerformance;
            
            // Estimate performance after retraining
            roi.ExpectedPerformance = EstimatePostRetrainingPerformance(monitoringData);
            
            // Calculate costs
            roi.RetrainingCost = CalculateRetrainingCost(monitoringData);
            
            // Calculate benefits
            roi.ExpectedBenefit = CalculateExpectedBenefit(
                roi.CurrentPerformance, 
                roi.ExpectedPerformance,
                monitoringData
            );
            
            // Calculate ROI metrics
            roi.ROIPercentage = ((roi.ExpectedBenefit - roi.RetrainingCost) / roi.RetrainingCost) * 100;
            roi.PaybackPeriodDays = CalculatePaybackPeriod(roi.RetrainingCost, roi.ExpectedBenefit);
            roi.ConfidenceInterval = CalculateConfidenceInterval(roi.ROIPercentage, monitoringData);
            
            roi.CalculatedAt = DateTime.UtcNow;
            
            return roi;
        }

        /// <summary>
        /// Gets data requirements for effective retraining
        /// </summary>
        public async Task<DataRequirements> GetDataRequirementsAsync()
        {
            var monitoringData = await GatherMonitoringDataAsync();
            var requirements = new DataRequirements();
            
            // Analyze current data distribution
            var dataAnalysis = AnalyzeDataDistribution(monitoringData);
            
            // Identify data gaps
            requirements.DataGaps = IdentifyDataGaps(dataAnalysis, monitoringData);
            
            // Calculate required sample sizes
            requirements.RequiredSamples = CalculateRequiredSamples(dataAnalysis, monitoringData);
            
            // Identify underrepresented features
            requirements.UnderrepresentedFeatures = IdentifyUnderrepresentedFeatures(dataAnalysis);
            
            // Suggest data collection strategies
            requirements.CollectionStrategies = SuggestDataCollectionStrategies(requirements);
            
            requirements.AnalysisTimestamp = DateTime.UtcNow;
            
            return requirements;
        }

        // Interface implementations

        public override Task<DriftDetectionResult> DetectDataDriftAsync(Matrix<T> productionData, Matrix<T>? referenceData = null)
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

        public override Task<DriftDetectionResult> DetectConceptDriftAsync(Vector<T> predictions, Vector<T> actuals)
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

        public override async Task<ModelHealthScore> GetModelHealthScoreAsync()
        {
            var monitoringData = await GatherMonitoringDataAsync();
            
            return new ModelHealthScore
            {
                OverallScore = monitoringData.HealthScore,
                DataQualityScore = monitoringData.DataQualityScore,
                PerformanceScore = monitoringData.CurrentPerformance.Accuracy,
                StabilityScore = monitoringData.StabilityScore,
                DriftScore = 1.0 - monitoringData.DriftScore,
                HealthStatus = monitoringData.HealthScore > 0.7 ? "Healthy" : "Needs Attention",
                Issues = new List<string>(),
                EvaluationTimestamp = DateTime.UtcNow
            };
        }

        // Private methods

        private void InitializeStrategies()
        {
            // Full retraining strategy
            _strategies["FullRetrain"] = new RetrainingStrategy
            {
                Name = "FullRetrain",
                Description = "Complete model retraining from scratch",
                ApplicabilityScore = 1.0,
                EstimatedDuration = TimeSpan.FromHours(24),
                ResourceIntensity = 1.0,
                ExpectedImprovement = 0.15
            };
            
            // Incremental learning strategy
            _strategies["Incremental"] = new RetrainingStrategy
            {
                Name = "Incremental",
                Description = "Update model with new data incrementally",
                ApplicabilityScore = 0.8,
                EstimatedDuration = TimeSpan.FromHours(2),
                ResourceIntensity = 0.3,
                ExpectedImprovement = 0.08
            };
            
            // Fine-tuning strategy
            _strategies["FineTune"] = new RetrainingStrategy
            {
                Name = "FineTune",
                Description = "Fine-tune existing model parameters",
                ApplicabilityScore = 0.7,
                EstimatedDuration = TimeSpan.FromHours(4),
                ResourceIntensity = 0.4,
                ExpectedImprovement = 0.10
            };
            
            // Ensemble update strategy
            _strategies["EnsembleUpdate"] = new RetrainingStrategy
            {
                Name = "EnsembleUpdate",
                Description = "Update ensemble components selectively",
                ApplicabilityScore = 0.6,
                EstimatedDuration = TimeSpan.FromHours(8),
                ResourceIntensity = 0.6,
                ExpectedImprovement = 0.12
            };
            
            // Active learning strategy
            _strategies["ActiveLearning"] = new RetrainingStrategy
            {
                Name = "ActiveLearning",
                Description = "Retrain with actively selected samples",
                ApplicabilityScore = 0.5,
                EstimatedDuration = TimeSpan.FromHours(6),
                ResourceIntensity = 0.5,
                ExpectedImprovement = 0.11
            };
        }

        private async Task<MonitoringData> GatherMonitoringDataAsync()
        {
            var endDate = DateTime.UtcNow;
            var startDate = endDate.AddDays(-_configuration.LookbackDays);
            
            // Get performance metrics
            var performance = await GetPerformanceMetricsAsync(startDate, endDate);
            
            // Get drift information
            DriftInfo driftInfo;
            lock (_lockObject)
            {
                var recentDrifts = _driftHistory
                    .Where(d => d.DetectionTimestamp >= startDate)
                    .ToList();
                
                driftInfo = new DriftInfo
                {
                    DriftCount = recentDrifts.Count(d => d.IsDriftDetected),
                    MaxDriftScore = recentDrifts.Any() ? recentDrifts.Max(d => d.DriftScore) : 0,
                    DriftTypes = recentDrifts.Select(d => d.DriftType).Distinct().ToList()
                };
            }
            
            // Calculate other metrics
            var dataQuality = await CalculateDataQualityScoreAsync();
            var stability = CalculateStabilityScore();
            var healthScore = CalculateOverallHealthScore(performance, driftInfo, dataQuality, stability);
            
            return new MonitoringData
            {
                CurrentPerformance = performance,
                DriftInfo = driftInfo,
                DataQualityScore = dataQuality,
                StabilityScore = stability,
                HealthScore = healthScore,
                DriftScore = driftInfo.MaxDriftScore,
                Timestamp = DateTime.UtcNow
            };
        }

        private Task<StrategyEvaluation> EvaluateStrategyAsync(RetrainingStrategy strategy, MonitoringData data)
        {
            var evaluation = new StrategyEvaluation
            {
                Strategy = strategy,
                ApplicabilityScore = CalculateStrategyApplicability(strategy, data),
                ExpectedImprovement = EstimateStrategyImprovement(strategy, data),
                RiskScore = CalculateStrategyRisk(strategy, data),
                CostBenefitRatio = CalculateCostBenefitRatio(strategy, data)
            };

            // Adjust for historical performance
            if (_retrainingHistory.Any(h => h.Strategy == strategy.Name))
            {
                var historicalPerformance = _retrainingHistory
                    .Where(h => h.Strategy == strategy.Name)
                    .Average(h => h.PerformanceImprovement);

                evaluation.ExpectedImprovement = (evaluation.ExpectedImprovement + historicalPerformance) / 2;
            }

            evaluation.OverallScore = CalculateStrategyScore(evaluation);

            return Task.FromResult(evaluation);
        }

        private double CalculateStrategyApplicability(RetrainingStrategy strategy, MonitoringData data)
        {
            var baseScore = strategy.ApplicabilityScore;
            
            // Adjust based on drift type
            if (strategy.Name == "Incremental" && data.DriftInfo.DriftTypes.Contains("ConceptDrift"))
            {
                baseScore *= 0.7; // Less suitable for concept drift
            }
            
            // Adjust based on data quality
            if (strategy.Name == "ActiveLearning" && data.DataQualityScore < 0.6)
            {
                baseScore *= 1.2; // More suitable when data quality is poor
            }
            
            // Adjust based on stability
            if (strategy.Name == "FineTune" && data.StabilityScore > 0.8)
            {
                baseScore *= 1.1; // Good for stable models
            }
            
            return Math.Min(1.0, baseScore);
        }

        private double EstimateStrategyImprovement(RetrainingStrategy strategy, MonitoringData data)
        {
            var baseImprovement = strategy.ExpectedImprovement;
            
            // Adjust based on current performance
            var performanceGap = 1.0 - data.CurrentPerformance.Accuracy;
            baseImprovement *= performanceGap;
            
            // Adjust based on drift severity
            baseImprovement *= (1 + data.DriftScore);
            
            return Math.Min(performanceGap, baseImprovement);
        }

        private double CalculateStrategyRisk(RetrainingStrategy strategy, MonitoringData data)
        {
            var risk = 0.0;
            
            // Higher risk for more intensive strategies
            risk += strategy.ResourceIntensity * 0.3;
            
            // Risk increases with poor data quality
            risk += (1 - data.DataQualityScore) * 0.3;
            
            // Risk increases with instability
            risk += (1 - data.StabilityScore) * 0.2;
            
            // Risk based on time since last retraining
            var lastRetraining = _retrainingHistory.LastOrDefault();
            if (lastRetraining != null)
            {
                var daysSinceRetraining = (DateTime.UtcNow - lastRetraining.Timestamp).TotalDays;
                if (daysSinceRetraining < 7)
                {
                    risk += 0.2; // Too frequent retraining is risky
                }
            }
            
            return Math.Min(1.0, risk);
        }

        private double CalculateCostBenefitRatio(RetrainingStrategy strategy, MonitoringData data)
        {
            var cost = strategy.ResourceIntensity * strategy.EstimatedDuration.TotalHours;
            var benefit = EstimateStrategyImprovement(strategy, data) * 100; // Normalize to percentage
            
            return benefit / (cost + 1); // Avoid division by zero
        }

        private double CalculateStrategyScore(StrategyEvaluation evaluation)
        {
            return evaluation.ApplicabilityScore * 0.3 +
                   evaluation.ExpectedImprovement * 0.3 +
                   (1 - evaluation.RiskScore) * 0.2 +
                   Math.Min(1.0, evaluation.CostBenefitRatio / 10) * 0.2;
        }

        private StrategyEvaluation SelectBestStrategy(Dictionary<string, StrategyEvaluation> evaluations)
        {
            return evaluations.Values.OrderByDescending(e => e.OverallScore).First();
        }

        private Task<RetrainingRecommendation> GenerateDetailedRecommendationAsync(
            StrategyEvaluation bestStrategy,
            MonitoringData monitoringData)
        {
            var reasons = new List<string>();
            var urgency = DetermineUrgency(monitoringData, bestStrategy);

            // Add reasons based on monitoring data
            if (monitoringData.DriftInfo.DriftCount > 0)
            {
                reasons.Add($"Detected {monitoringData.DriftInfo.DriftCount} drift events");
            }

            if (monitoringData.CurrentPerformance.Accuracy < _configuration.MinAcceptableAccuracy)
            {
                reasons.Add($"Performance below threshold: {monitoringData.CurrentPerformance.Accuracy:F2}");
            }

            if (monitoringData.HealthScore < 0.6)
            {
                reasons.Add($"Poor model health: {monitoringData.HealthScore:F2}");
            }

            // Create suggested actions
            var suggestedActions = new Dictionary<string, object>
            {
                ["Strategy"] = bestStrategy.Strategy.Name,
                ["EstimatedDuration"] = bestStrategy.Strategy.EstimatedDuration.TotalHours,
                ["ExpectedImprovement"] = bestStrategy.ExpectedImprovement,
                ["DataPreparation"] = monitoringData.DataQualityScore < 0.8,
                ["FeatureEngineering"] = monitoringData.DriftInfo.DriftTypes.Contains("DataDrift"),
                ["HyperparameterTuning"] = bestStrategy.Strategy.Name == "FineTune"
            };

            return Task.FromResult(new RetrainingRecommendation
            {
                ShouldRetrain = urgency != "None",
                Urgency = urgency,
                Reasons = reasons,
                RecommendationTimestamp = DateTime.UtcNow,
                ConfidenceScore = bestStrategy.OverallScore,
                SuggestedActions = suggestedActions
            });
        }

        private string DetermineUrgency(MonitoringData data, StrategyEvaluation strategy)
        {
            if (data.HealthScore < 0.3 || data.CurrentPerformance.Accuracy < 0.5)
                return "Critical";
            
            if (data.HealthScore < 0.5 || data.DriftScore > 0.7)
                return "High";
            
            if (data.HealthScore < 0.7 || strategy.ExpectedImprovement > 0.1)
                return "Medium";
            
            if (strategy.ExpectedImprovement > 0.05)
                return "Low";
            
            return "None";
        }

        private void EnrichWithHistoricalContext(RetrainingRecommendation recommendation)
        {
            if (!_retrainingHistory.Any()) return;
            
            var lastRetraining = _retrainingHistory.Last();
            var daysSinceRetraining = (DateTime.UtcNow - lastRetraining.Timestamp).TotalDays;
            
            recommendation.SuggestedActions["DaysSinceLastRetraining"] = daysSinceRetraining;
            recommendation.SuggestedActions["LastRetrainingStrategy"] = lastRetraining.Strategy;
            recommendation.SuggestedActions["LastImprovement"] = lastRetraining.PerformanceImprovement;
            
            // Add pattern-based insights
            var avgTimeBetweenRetraining = CalculateAverageRetrainingInterval();
            if (daysSinceRetraining > avgTimeBetweenRetraining * 1.5)
            {
                recommendation.Reasons.Add($"Overdue for retraining (avg interval: {avgTimeBetweenRetraining:F0} days)");
            }
        }

        private RetrainingPatterns AnalyzeRetrainingPatterns()
        {
            var patterns = new RetrainingPatterns();
            
            if (_retrainingHistory.Count < 2) return patterns;
            
            // Analyze intervals
            var intervals = new List<double>();
            for (int i = 1; i < _retrainingHistory.Count; i++)
            {
                intervals.Add((_retrainingHistory[i].Timestamp - _retrainingHistory[i-1].Timestamp).TotalDays);
            }
            
            patterns.AverageInterval = intervals.Average();
            patterns.IntervalStdDev = Math.Sqrt(intervals.Select(i => Math.Pow(i - patterns.AverageInterval, 2)).Average());
            
            // Analyze performance improvements
            patterns.AverageImprovement = _retrainingHistory.Average(h => h.PerformanceImprovement);
            
            // Analyze strategy effectiveness
            patterns.StrategyEffectiveness = _retrainingHistory
                .GroupBy(h => h.Strategy)
                .ToDictionary(g => g.Key, g => g.Average(h => h.PerformanceImprovement));
            
            return patterns;
        }

        private List<ScheduledRetraining> PredictOptimalSchedule(MonitoringData data, RetrainingPatterns patterns)
        {
            var schedule = new List<ScheduledRetraining>();
            var currentDate = DateTime.UtcNow;
            
            // Base interval on historical patterns or configuration
            var baseInterval = patterns.AverageInterval > 0 ? patterns.AverageInterval : _configuration.DefaultRetrainingIntervalDays;
            
            // Adjust based on current health
            var intervalMultiplier = data.HealthScore > 0.8 ? 1.5 : 
                                   data.HealthScore > 0.6 ? 1.0 : 0.7;
            
            var adjustedInterval = baseInterval * intervalMultiplier;
            
            // Generate schedule for next 90 days
            for (int i = 0; i < 3; i++)
            {
                var scheduledDate = currentDate.AddDays(adjustedInterval * (i + 1));
                var urgency = i == 0 && data.HealthScore < 0.7 ? "High" : "Medium";
                
                schedule.Add(new ScheduledRetraining
                {
                    ScheduledDate = scheduledDate,
                    RecommendedStrategy = SelectStrategyForSchedule(data, i).Name,
                    ExpectedDuration = TimeSpan.FromHours(8),
                    Priority = urgency,
                    Prerequisites = GeneratePrerequisites(i)
                });
            }
            
            return schedule;
        }

        private RetrainingStrategy SelectStrategyForSchedule(MonitoringData data, int scheduleIndex)
        {
            if (scheduleIndex == 0 && data.DriftScore > 0.5)
            {
                return _strategies["FullRetrain"];
            }
            
            if (data.StabilityScore > 0.8)
            {
                return _strategies["FineTune"];
            }
            
            return _strategies["Incremental"];
        }

        private List<string> GeneratePrerequisites(int scheduleIndex)
        {
            var prerequisites = new List<string>
            {
                "Backup current model",
                "Prepare validation dataset",
                "Allocate compute resources"
            };
            
            if (scheduleIndex == 0)
            {
                prerequisites.Add("Perform data quality audit");
                prerequisites.Add("Update feature engineering pipeline");
            }
            
            return prerequisites;
        }

        private Dictionary<string, double> CalculateExpectedBenefits(List<ScheduledRetraining> schedule, MonitoringData data)
        {
            return new Dictionary<string, double>
            {
                ["ExpectedAccuracyImprovement"] = 0.05 * schedule.Count,
                ["ExpectedDriftReduction"] = 0.3 * schedule.Count,
                ["ExpectedStabilityImprovement"] = 0.1 * schedule.Count
            };
        }

        private ResourceRequirements EstimateResourceRequirements(List<ScheduledRetraining> schedule)
        {
            return new ResourceRequirements
            {
                TotalComputeHours = schedule.Sum(s => s.ExpectedDuration.TotalHours),
                EstimatedCost = schedule.Sum(s => s.ExpectedDuration.TotalHours * _configuration.HourlyCost),
                RequiredStorage = schedule.Count * _configuration.ModelSizeGB * 2, // Include backups
                RequiredMemory = _configuration.TrainingMemoryGB,
                RequiredGPUs = _configuration.RequiresGPU ? _configuration.GPUCount : 0
            };
        }

        private double EstimatePostRetrainingPerformance(MonitoringData data)
        {
            var baseImprovement = 0.05; // Conservative estimate
            
            // Adjust based on drift
            if (data.DriftScore > 0.5)
            {
                baseImprovement += 0.03;
            }
            
            // Adjust based on data quality
            if (data.DataQualityScore > 0.8)
            {
                baseImprovement += 0.02;
            }
            
            return Math.Min(0.99, data.CurrentPerformance.Accuracy + baseImprovement);
        }

        private double CalculateRetrainingCost(MonitoringData data)
        {
            var baseCost = _configuration.BaseRetrainingCost;
            
            // Adjust for data volume
            var dataVolumeFactor = 1.0; // Would be calculated from actual data volume
            
            // Adjust for complexity
            var complexityFactor = data.DriftScore > 0.5 ? 1.5 : 1.0;
            
            return baseCost * dataVolumeFactor * complexityFactor;
        }

        private double CalculateExpectedBenefit(PerformanceMetrics current, double expected, MonitoringData data)
        {
            var accuracyImprovement = expected - current.Accuracy;
            var benefitPerPoint = _configuration.BenefitPerAccuracyPoint;
            
            return accuracyImprovement * benefitPerPoint * _configuration.DailyPredictionVolume * 30; // 30 days
        }

        private double CalculatePaybackPeriod(double cost, double monthlyBenefit)
        {
            return cost / (monthlyBenefit / 30); // Convert to days
        }

        private (double lower, double upper) CalculateConfidenceInterval(double estimate, MonitoringData data)
        {
            var uncertainty = 0.2; // Base uncertainty
            
            // Adjust based on data quality
            uncertainty *= (2 - data.DataQualityScore);
            
            // Adjust based on historical variance
            if (_retrainingHistory.Count > 5)
            {
                var improvements = _retrainingHistory.Select(h => h.PerformanceImprovement).ToList();
                var variance = improvements.Select(i => Math.Pow(i - improvements.Average(), 2)).Average();
                uncertainty += Math.Sqrt(variance);
            }
            
            return (estimate * (1 - uncertainty), estimate * (1 + uncertainty));
        }

        private DataDistributionAnalysis AnalyzeDataDistribution(MonitoringData data)
        {
            // Simplified analysis
            return new DataDistributionAnalysis
            {
                TotalSamples = _predictionHistory.Count,
                FeatureDistributions = new Dictionary<string, Distribution>(),
                ClassBalance = new Dictionary<string, double>(),
                TemporalPatterns = new List<string>()
            };
        }

        private List<DataGap> IdentifyDataGaps(DataDistributionAnalysis analysis, MonitoringData data)
        {
            var gaps = new List<DataGap>();
            
            // Check for class imbalance
            if (analysis.ClassBalance.Any(cb => cb.Value < 0.1))
            {
                gaps.Add(new DataGap
                {
                    Type = "ClassImbalance",
                    Description = "Severe class imbalance detected",
                    Severity = "High",
                    AffectedFeatures = new List<string>()
                });
            }
            
            // Check for feature coverage
            lock (_lockObject)
            {
                if (_featureImportances.Any())
                {
                    var importantFeatures = _featureImportances
                        .Where(fi => fi.Value > 0.1)
                        .Select(fi => fi.Key)
                        .ToList();
                    
                    // Would check actual feature coverage here
                }
            }
            
            return gaps;
        }

        private Dictionary<string, int> CalculateRequiredSamples(DataDistributionAnalysis analysis, MonitoringData data)
        {
            var requirements = new Dictionary<string, int>();
            
            // Base requirement
            var baseRequirement = 1000;
            
            // Adjust based on model complexity
            var complexityFactor = 1.5; // Would be calculated from actual model
            
            requirements["MinimumTotal"] = (int)(baseRequirement * complexityFactor);
            requirements["PerClass"] = requirements["MinimumTotal"] / 2; // For binary classification
            
            return requirements;
        }

        private List<string> IdentifyUnderrepresentedFeatures(DataDistributionAnalysis analysis)
        {
            var underrepresented = new List<string>();
            
            lock (_lockObject)
            {
                foreach (var feature in _featureImportances.Where(fi => fi.Value > 0.05))
                {
                    // Would check actual representation
                    if (analysis.FeatureDistributions.ContainsKey(feature.Key))
                    {
                        // Check distribution quality
                    }
                }
            }
            
            return underrepresented;
        }

        private List<string> SuggestDataCollectionStrategies(DataRequirements requirements)
        {
            var strategies = new List<string>();
            
            if (requirements.DataGaps.Any(g => g.Type == "ClassImbalance"))
            {
                strategies.Add("Implement targeted sampling for minority classes");
                strategies.Add("Use synthetic data generation (SMOTE) for balancing");
            }
            
            if (requirements.UnderrepresentedFeatures.Any())
            {
                strategies.Add("Focus data collection on underrepresented feature combinations");
                strategies.Add("Implement active learning for efficient sampling");
            }
            
            strategies.Add("Set up continuous data quality monitoring");
            strategies.Add("Implement data validation pipelines");
            
            return strategies;
        }

        private double CalculateAverageRetrainingInterval()
        {
            if (_retrainingHistory.Count < 2) return _configuration.DefaultRetrainingIntervalDays;
            
            var intervals = new List<double>();
            for (int i = 1; i < _retrainingHistory.Count; i++)
            {
                intervals.Add((_retrainingHistory[i].Timestamp - _retrainingHistory[i-1].Timestamp).TotalDays);
            }
            
            return intervals.Average();
        }

        private Task<double> CalculateDataQualityScoreAsync()
        {
            lock (_lockObject)
            {
                if (!_predictionHistory.Any()) return Task.FromResult(1.0);

                var recent = _predictionHistory.TakeLast(1000).ToList();
                var missingRatio = recent.Count(p => p.Features.Any(f => double.IsNaN(f))) / (double)recent.Count();
                var duplicateRatio = (recent.Count() - recent.Distinct().Count()) / (double)recent.Count();

                return Task.FromResult(1.0 - (missingRatio * 0.5 + duplicateRatio * 0.5));
            }
        }

        private double CalculateStabilityScore()
        {
            lock (_lockObject)
            {
                if (_performanceHistory.Count < 5) return 1.0;
                
                var recent = _performanceHistory.TakeLast(10).Select(p => p.Accuracy).ToList();
                var mean = recent.Average();
                var variance = recent.Select(a => Math.Pow(a - mean, 2)).Average();
                var cv = Math.Sqrt(variance) / mean;
                
                return Math.Max(0, 1.0 - cv * 2);
            }
        }

        private double CalculateOverallHealthScore(PerformanceMetrics performance, DriftInfo drift, double dataQuality, double stability)
        {
            return performance.Accuracy * 0.4 +
                   (1 - drift.MaxDriftScore) * 0.2 +
                   dataQuality * 0.2 +
                   stability * 0.2;
        }

        // Helper classes

        public class RecommenderConfiguration
        {
            public int LookbackDays { get; set; } = 30;
            public int HistoryRetentionDays { get; set; } = 180;
            public double MinAcceptableAccuracy { get; set; } = 0.8;
            public double DefaultRetrainingIntervalDays { get; set; } = 30;
            public double HourlyCost { get; set; } = 10.0;
            public double BaseRetrainingCost { get; set; } = 1000.0;
            public double BenefitPerAccuracyPoint { get; set; } = 10000.0;
            public int DailyPredictionVolume { get; set; } = 10000;
            public double ModelSizeGB { get; set; } = 1.0;
            public double TrainingMemoryGB { get; set; } = 16.0;
            public bool RequiresGPU { get; set; } = true;
            public int GPUCount { get; set; } = 1;
        }

        public class RetrainingEvent
        {
            public DateTime Timestamp { get; set; }
            public string Strategy { get; set; } = string.Empty;
            public double PerformanceImprovement { get; set; }
            public TimeSpan Duration { get; set; }
            public double Cost { get; set; }
            public string Outcome { get; set; } = string.Empty;
            public Dictionary<string, object> Metadata { get; set; } = new();
        }

        private class RetrainingStrategy
        {
            public string Name { get; set; } = string.Empty;
            public string Description { get; set; } = string.Empty;
            public double ApplicabilityScore { get; set; }
            public TimeSpan EstimatedDuration { get; set; }
            public double ResourceIntensity { get; set; }
            public double ExpectedImprovement { get; set; }
        }

        private class MonitoringData
        {
            public PerformanceMetrics CurrentPerformance { get; set; } = new();
            public DriftInfo DriftInfo { get; set; } = new();
            public double DataQualityScore { get; set; }
            public double StabilityScore { get; set; }
            public double HealthScore { get; set; }
            public double DriftScore { get; set; }
            public DateTime Timestamp { get; set; }
        }

        private class DriftInfo
        {
            public int DriftCount { get; set; }
            public double MaxDriftScore { get; set; }
            public List<string> DriftTypes { get; set; } = new();
        }

        private class StrategyEvaluation
        {
            public RetrainingStrategy Strategy { get; set; } = new();
            public double ApplicabilityScore { get; set; }
            public double ExpectedImprovement { get; set; }
            public double RiskScore { get; set; }
            public double CostBenefitRatio { get; set; }
            public double OverallScore { get; set; }
        }

        public class RetrainingSchedule
        {
            public List<ScheduledRetraining> RecommendedSchedule { get; set; } = new();
            public Dictionary<string, double> ExpectedBenefits { get; set; } = new();
            public ResourceRequirements ResourceRequirements { get; set; } = new();
            public DateTime GeneratedAt { get; set; }
            public DateTime ValidUntil { get; set; }
        }

        public class ScheduledRetraining
        {
            public DateTime ScheduledDate { get; set; }
            public string RecommendedStrategy { get; set; } = string.Empty;
            public TimeSpan ExpectedDuration { get; set; }
            public string Priority { get; set; } = string.Empty;
            public List<string> Prerequisites { get; set; } = new();
        }

        public class ResourceRequirements
        {
            public double TotalComputeHours { get; set; }
            public double EstimatedCost { get; set; }
            public double RequiredStorage { get; set; }
            public double RequiredMemory { get; set; }
            public int RequiredGPUs { get; set; }
        }

        public class RetrainingROI
        {
            public PerformanceMetrics CurrentPerformance { get; set; } = new();
            public double ExpectedPerformance { get; set; }
            public double RetrainingCost { get; set; }
            public double ExpectedBenefit { get; set; }
            public double ROIPercentage { get; set; }
            public double PaybackPeriodDays { get; set; }
            public (double lower, double upper) ConfidenceInterval { get; set; }
            public DateTime CalculatedAt { get; set; }
        }

        public class DataRequirements
        {
            public List<DataGap> DataGaps { get; set; } = new();
            public Dictionary<string, int> RequiredSamples { get; set; } = new();
            public List<string> UnderrepresentedFeatures { get; set; } = new();
            public List<string> CollectionStrategies { get; set; } = new();
            public DateTime AnalysisTimestamp { get; set; }
        }

        public class DataGap
        {
            public string Type { get; set; } = string.Empty;
            public string Description { get; set; } = string.Empty;
            public string Severity { get; set; } = string.Empty;
            public List<string> AffectedFeatures { get; set; } = new();
        }

        private class RetrainingPatterns
        {
            public double AverageInterval { get; set; }
            public double IntervalStdDev { get; set; }
            public double AverageImprovement { get; set; }
            public Dictionary<string, double> StrategyEffectiveness { get; set; } = new();
        }

        private class DataDistributionAnalysis
        {
            public int TotalSamples { get; set; }
            public Dictionary<string, Distribution> FeatureDistributions { get; set; } = new();
            public Dictionary<string, double> ClassBalance { get; set; } = new();
            public List<string> TemporalPatterns { get; set; } = new();
        }

        private class Distribution
        {
            public double Mean { get; set; }
            public double StdDev { get; set; }
            public double Skewness { get; set; }
            public double Kurtosis { get; set; }
        }
    }
}
