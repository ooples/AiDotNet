using System;
using System.Collections.Generic;
using AiDotNet;
using AiDotNet.AutoML.Policies;
using AiDotNet.AutoML.Registry;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.AutoML
{
    public class AutoMLTaskFamilySupportTests
    {
        [Theory]
        [InlineData(AutoMLTaskFamily.TimeSeriesAnomalyDetection)]
        [InlineData(AutoMLTaskFamily.Ranking)]
        [InlineData(AutoMLTaskFamily.Recommendation)]
        public void AutoMLDefaultCandidateModelsPolicy_ReturnsCandidates_ForAdditionalTaskFamilies(AutoMLTaskFamily taskFamily)
        {
            foreach (AutoMLBudgetPreset preset in Enum.GetValues(typeof(AutoMLBudgetPreset)))
            {
                var candidates = AutoMLDefaultCandidateModelsPolicy.GetDefaultCandidates(taskFamily, featureCount: 8, preset);
                Assert.NotEmpty(candidates);

                foreach (var modelType in candidates)
                {
                    var model = AutoMLTabularModelFactory<double>.Create(modelType, new Dictionary<string, object>());
                    Assert.NotNull(model);

                    var space = AutoMLTabularSearchSpaceRegistry.GetDefaultSearchSpace(modelType);
                    Assert.NotNull(space);
                }
            }
        }

        [Theory]
        [InlineData(AutoMLTaskFamily.TimeSeriesAnomalyDetection, MetricType.AUCPR, true)]
        [InlineData(AutoMLTaskFamily.Ranking, MetricType.NormalizedDiscountedCumulativeGain, true)]
        [InlineData(AutoMLTaskFamily.Recommendation, MetricType.NormalizedDiscountedCumulativeGain, true)]
        public void AutoMLDefaultMetricPolicy_UsesIndustryDefaults_ForAdditionalTaskFamilies(
            AutoMLTaskFamily taskFamily,
            MetricType expectedMetric,
            bool expectedMaximize)
        {
            var (metric, maximize) = AutoMLDefaultMetricPolicy.GetDefault(taskFamily);
            Assert.Equal(expectedMetric, metric);
            Assert.Equal(expectedMaximize, maximize);
        }

        [Theory]
        [InlineData(AutoMLTaskFamily.TimeSeriesAnomalyDetection)]
        [InlineData(AutoMLTaskFamily.Ranking)]
        [InlineData(AutoMLTaskFamily.Recommendation)]
        public void AiModelBuilder_ConfigureAutoMLFacade_AllowsAdditionalTaskFamilies(AutoMLTaskFamily taskFamily)
        {
            var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>();
            var options = new AutoMLOptions<double, Matrix<double>, Vector<double>>
            {
                Budget = new AutoMLBudgetOptions
                {
                    Preset = AutoMLBudgetPreset.CI,
                    TrialLimitOverride = 2,
                    TimeLimitOverride = TimeSpan.FromSeconds(3)
                },
                TaskFamilyOverride = taskFamily,
                SearchStrategy = AutoMLSearchStrategy.RandomSearch
            };

            var exception = Record.Exception(() => builder.ConfigureAutoML(options));
            Assert.Null(exception);
        }
    }
}

