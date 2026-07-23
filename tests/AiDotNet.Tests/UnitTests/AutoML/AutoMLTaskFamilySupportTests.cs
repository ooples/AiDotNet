using System;
using System.Collections.Generic;
using AiDotNet;
using AiDotNet.AutoML.Policies;
using AiDotNet.AutoML.Registry;
using AiDotNet.Configuration;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Regression;
using AiDotNet.Tests.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Moq;
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

        [Fact(Timeout = 60000)]
        public async Task AiModelBuilder_ConfigureAutoMLSearchSpace_ExcludesDefaultsWithoutTaskFamilyOverride()
        {
            var x = new Matrix<double>(new double[,]
            {
                { 0.0, 1.0 },
                { 1.0, 2.0 },
                { 2.0, 3.0 },
                { 3.0, 4.0 },
                { 4.0, 5.0 },
                { 5.0, 6.0 },
                { 6.0, 7.0 },
                { 7.0, 8.0 },
                { 8.0, 9.0 },
                { 9.0, 10.0 }
            });
            var y = new Vector<double>(new[] { 1.0, 1.9, 3.1, 4.0, 5.2, 5.9, 7.1, 8.0, 9.1, 10.2 });
            List<Type>? capturedCandidates = null;
            var returnedModel = new MockFullModel(input => new Vector<double>(input.Rows), parameterCount: 2);
            var autoML = new Mock<IAutoMLModel<double, Matrix<double>, Vector<double>>>();
            autoML.SetupGet(a => a.TimeLimit).Returns(TimeSpan.FromSeconds(1));
            autoML.SetupGet(a => a.BestScore).Returns(0.0);
            autoML.Setup(a => a.GetTrialHistory()).Returns(new List<AiDotNet.AutoML.TrialResult>());
            autoML.Setup(a => a.SetCandidateModels(It.IsAny<List<Type>>()))
                .Callback<List<Type>>(candidates => capturedCandidates = candidates);
            autoML.Setup(a => a.SearchAsync(
                    It.IsAny<Matrix<double>>(),
                    It.IsAny<Vector<double>>(),
                    It.IsAny<Matrix<double>>(),
                    It.IsAny<Vector<double>>(),
                    It.IsAny<TimeSpan>(),
                    It.IsAny<CancellationToken>()))
                .ReturnsAsync(returnedModel);

            var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
                .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, y))
                .ConfigureModel(autoML.Object);
            var options = new AutoMLOptions<double, Matrix<double>, Vector<double>>
            {
                Budget = new AutoMLBudgetOptions { Preset = AutoMLBudgetPreset.CI },
                SearchStrategy = AutoMLSearchStrategy.RandomSearch,
                SearchSpace = new AutoMLSearchSpace
                {
                    ExcludedModels = new List<Type> { typeof(RandomForestRegression<>) }
                }
            };

            var optionsField = typeof(AiModelBuilder<double, Matrix<double>, Vector<double>>)
                .GetField("_autoMLOptions", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic);
            Assert.NotNull(optionsField);
            optionsField!.SetValue(builder, options);

            await builder.BuildAsync();

            Assert.NotNull(capturedCandidates);
            Assert.NotEmpty(capturedCandidates!);
            Assert.DoesNotContain(typeof(RandomForestRegression<>), capturedCandidates);
        }
    }
}

