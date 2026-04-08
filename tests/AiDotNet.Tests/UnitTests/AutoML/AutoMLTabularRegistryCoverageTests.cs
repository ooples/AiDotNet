using System;
using System.Collections.Generic;
using AiDotNet.AutoML.Policies;
using AiDotNet.AutoML.Registry;
using AiDotNet.Enums;
using Xunit;

namespace AiDotNetTests.UnitTests.AutoML
{
    public class AutoMLTabularRegistryCoverageTests
    {
        [Fact]
        public void AutoMLTabularModelFactory_CanCreate_DefaultRegressionCandidates_ForAllBudgets()
        {
            foreach (AutoMLBudgetPreset preset in Enum.GetValues(typeof(AutoMLBudgetPreset)))
            {
                var candidates = AutoMLDefaultCandidateModelsPolicy.GetDefaultCandidates(
                    AutoMLTaskFamily.Regression,
                    featureCount: 8,
                    preset);

                Assert.NotEmpty(candidates);

                foreach (var modelType in candidates)
                {
                    var model = AutoMLTabularModelFactory<double>.Create(modelType, new Dictionary<string, object>());
                    Assert.NotNull(model);
                }
            }
        }

        [Fact]
        public void AutoMLTabularModelFactory_CanCreate_DefaultBinaryCandidates_ForAllBudgets()
        {
            foreach (AutoMLBudgetPreset preset in Enum.GetValues(typeof(AutoMLBudgetPreset)))
            {
                var candidates = AutoMLDefaultCandidateModelsPolicy.GetDefaultCandidates(
                    AutoMLTaskFamily.BinaryClassification,
                    featureCount: 8,
                    preset);

                Assert.NotEmpty(candidates);

                foreach (var modelType in candidates)
                {
                    var model = AutoMLTabularModelFactory<double>.Create(modelType, new Dictionary<string, object>());
                    Assert.NotNull(model);
                }
            }
        }

        [Fact]
        public void AutoMLTabularModelFactory_CanCreate_DefaultMultiClassCandidates_ForAllBudgets()
        {
            foreach (AutoMLBudgetPreset preset in Enum.GetValues(typeof(AutoMLBudgetPreset)))
            {
                var candidates = AutoMLDefaultCandidateModelsPolicy.GetDefaultCandidates(
                    AutoMLTaskFamily.MultiClassClassification,
                    featureCount: 8,
                    preset);

                Assert.NotEmpty(candidates);

                foreach (var modelType in candidates)
                {
                    var model = AutoMLTabularModelFactory<double>.Create(modelType, new Dictionary<string, object>());
                    Assert.NotNull(model);
                }
            }
        }

        [Fact]
        public void AutoMLTabularSearchSpaceRegistry_ReturnsSpaces_ForDefaultCandidates()
        {
            foreach (AutoMLBudgetPreset preset in Enum.GetValues(typeof(AutoMLBudgetPreset)))
            {
                foreach (var taskFamily in new[]
                {
                    AutoMLTaskFamily.Regression,
                    AutoMLTaskFamily.BinaryClassification,
                    AutoMLTaskFamily.MultiClassClassification,
                    AutoMLTaskFamily.TimeSeriesForecasting
                })
                {
                    var candidates = AutoMLDefaultCandidateModelsPolicy.GetDefaultCandidates(taskFamily, featureCount: 8, preset);
                    foreach (var modelType in candidates)
                    {
                        var space = AutoMLTabularSearchSpaceRegistry.GetDefaultSearchSpace(modelType);
                        Assert.NotNull(space);
                    }
                }
            }
        }
    }
}

