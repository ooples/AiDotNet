using AiDotNet.Regression;

namespace AiDotNet.AutoML.Registry;

/// <summary>
/// Defines default hyperparameter search spaces for built-in tabular AutoML candidate models.
/// </summary>
/// <remarks>
/// <para>
/// These ranges are intentionally bounded and include industry-standard defaults.
/// </para>
/// <para>
/// <b>For Beginners:</b> A "search space" is the list of knobs AutoML can turn when trying different model settings
/// (for example, number of trees in a random forest).
/// </para>
/// </remarks>
internal static class AutoMLTabularSearchSpaceRegistry
{
    private static readonly Dictionary<Type, Func<Dictionary<string, ParameterRange>>> _spaces = new()
    {
        [typeof(PolynomialRegression<>)] = () => new Dictionary<string, ParameterRange>(StringComparer.Ordinal)
        {
            ["Degree"] = new ParameterRange
            {
                Type = ParameterType.Integer,
                MinValue = 1,
                MaxValue = 6,
                Step = 1,
                DefaultValue = 2
            }
        },

        [typeof(LogisticRegression<>)] = () => new Dictionary<string, ParameterRange>(StringComparer.Ordinal)
        {
            ["MaxIterations"] = new ParameterRange
            {
                Type = ParameterType.Integer,
                MinValue = 100,
                MaxValue = 5000,
                Step = 100,
                DefaultValue = 1000
            },
            ["LearningRate"] = new ParameterRange
            {
                Type = ParameterType.Float,
                MinValue = 1e-4,
                MaxValue = 0.1,
                UseLogScale = true,
                DefaultValue = 0.01
            },
            ["Tolerance"] = new ParameterRange
            {
                Type = ParameterType.Float,
                MinValue = 1e-6,
                MaxValue = 1e-2,
                UseLogScale = true,
                DefaultValue = 1e-4
            }
        },

        [typeof(MultinomialLogisticRegression<>)] = () => new Dictionary<string, ParameterRange>(StringComparer.Ordinal)
        {
            ["MaxIterations"] = new ParameterRange
            {
                Type = ParameterType.Integer,
                MinValue = 100,
                MaxValue = 5000,
                Step = 100,
                DefaultValue = 1000
            },
            ["LearningRate"] = new ParameterRange
            {
                Type = ParameterType.Float,
                MinValue = 1e-4,
                MaxValue = 0.1,
                UseLogScale = true,
                DefaultValue = 0.01
            },
            ["Tolerance"] = new ParameterRange
            {
                Type = ParameterType.Float,
                MinValue = 1e-6,
                MaxValue = 1e-2,
                UseLogScale = true,
                DefaultValue = 1e-4
            }
        },

        [typeof(RandomForestRegression<>)] = () => new Dictionary<string, ParameterRange>(StringComparer.Ordinal)
        {
            ["NumberOfTrees"] = new ParameterRange
            {
                Type = ParameterType.Integer,
                MinValue = 50,
                MaxValue = 500,
                Step = 10,
                DefaultValue = 100
            },
            ["MaxDepth"] = new ParameterRange
            {
                Type = ParameterType.Integer,
                MinValue = 2,
                MaxValue = 50,
                Step = 1,
                DefaultValue = 10
            },
            ["MinSamplesSplit"] = new ParameterRange
            {
                Type = ParameterType.Integer,
                MinValue = 2,
                MaxValue = 50,
                Step = 1,
                DefaultValue = 2
            },
            ["MaxFeatures"] = new ParameterRange
            {
                Type = ParameterType.Float,
                MinValue = 0.2,
                MaxValue = 1.0,
                Step = 0.05,
                DefaultValue = 1.0
            }
        },

        [typeof(DecisionTreeRegression<>)] = () => new Dictionary<string, ParameterRange>(StringComparer.Ordinal)
        {
            ["MaxDepth"] = new ParameterRange
            {
                Type = ParameterType.Integer,
                MinValue = 2,
                MaxValue = 50,
                Step = 1,
                DefaultValue = 10
            },
            ["MinSamplesSplit"] = new ParameterRange
            {
                Type = ParameterType.Integer,
                MinValue = 2,
                MaxValue = 50,
                Step = 1,
                DefaultValue = 2
            },
            ["MaxFeatures"] = new ParameterRange
            {
                Type = ParameterType.Float,
                MinValue = 0.2,
                MaxValue = 1.0,
                Step = 0.05,
                DefaultValue = 1.0
            },
            ["UseSoftTree"] = new ParameterRange
            {
                Type = ParameterType.Boolean,
                DefaultValue = false
            },
            ["SoftTreeTemperature"] = new ParameterRange
            {
                Type = ParameterType.Float,
                MinValue = 0.1,
                MaxValue = 10.0,
                UseLogScale = true,
                DefaultValue = 1.0
            }
        },

        [typeof(ExtremelyRandomizedTreesRegression<>)] = () => new Dictionary<string, ParameterRange>(StringComparer.Ordinal)
        {
            ["NumberOfTrees"] = new ParameterRange
            {
                Type = ParameterType.Integer,
                MinValue = 50,
                MaxValue = 500,
                Step = 10,
                DefaultValue = 100
            },
            ["MaxDepth"] = new ParameterRange
            {
                Type = ParameterType.Integer,
                MinValue = 2,
                MaxValue = 50,
                Step = 1,
                DefaultValue = 10
            },
            ["MinSamplesSplit"] = new ParameterRange
            {
                Type = ParameterType.Integer,
                MinValue = 2,
                MaxValue = 50,
                Step = 1,
                DefaultValue = 2
            },
            ["MaxFeatures"] = new ParameterRange
            {
                Type = ParameterType.Float,
                MinValue = 0.2,
                MaxValue = 1.0,
                Step = 0.05,
                DefaultValue = 1.0
            }
        },

        [typeof(AdaBoostR2Regression<>)] = () => new Dictionary<string, ParameterRange>(StringComparer.Ordinal)
        {
            ["NumberOfEstimators"] = new ParameterRange
            {
                Type = ParameterType.Integer,
                MinValue = 25,
                MaxValue = 300,
                Step = 5,
                DefaultValue = 50
            },
            ["MaxDepth"] = new ParameterRange
            {
                Type = ParameterType.Integer,
                MinValue = 1,
                MaxValue = 10,
                Step = 1,
                DefaultValue = 3
            },
            ["MinSamplesSplit"] = new ParameterRange
            {
                Type = ParameterType.Integer,
                MinValue = 2,
                MaxValue = 50,
                Step = 1,
                DefaultValue = 2
            }
        },

        [typeof(QuantileRegressionForests<>)] = () => new Dictionary<string, ParameterRange>(StringComparer.Ordinal)
        {
            ["NumberOfTrees"] = new ParameterRange
            {
                Type = ParameterType.Integer,
                MinValue = 50,
                MaxValue = 500,
                Step = 10,
                DefaultValue = 100
            },
            ["MaxDepth"] = new ParameterRange
            {
                Type = ParameterType.Integer,
                MinValue = 2,
                MaxValue = 50,
                Step = 1,
                DefaultValue = 10
            },
            ["MinSamplesSplit"] = new ParameterRange
            {
                Type = ParameterType.Integer,
                MinValue = 2,
                MaxValue = 50,
                Step = 1,
                DefaultValue = 2
            },
            ["MaxFeatures"] = new ParameterRange
            {
                Type = ParameterType.Float,
                MinValue = 0.2,
                MaxValue = 1.0,
                Step = 0.05,
                DefaultValue = 1.0
            }
        },

        [typeof(GradientBoostingRegression<>)] = () => new Dictionary<string, ParameterRange>(StringComparer.Ordinal)
        {
            ["NumberOfTrees"] = new ParameterRange
            {
                Type = ParameterType.Integer,
                MinValue = 50,
                MaxValue = 500,
                Step = 10,
                DefaultValue = 100
            },
            ["LearningRate"] = new ParameterRange
            {
                Type = ParameterType.Float,
                MinValue = 0.01,
                MaxValue = 0.3,
                UseLogScale = true,
                DefaultValue = 0.1
            },
            ["SubsampleRatio"] = new ParameterRange
            {
                Type = ParameterType.Float,
                MinValue = 0.5,
                MaxValue = 1.0,
                Step = 0.05,
                DefaultValue = 1.0
            },
            ["MaxDepth"] = new ParameterRange
            {
                Type = ParameterType.Integer,
                MinValue = 2,
                MaxValue = 20,
                Step = 1,
                DefaultValue = 10
            }
        },

        [typeof(KNearestNeighborsRegression<>)] = () => new Dictionary<string, ParameterRange>(StringComparer.Ordinal)
        {
            ["K"] = new ParameterRange
            {
                Type = ParameterType.Integer,
                MinValue = 1,
                MaxValue = 50,
                Step = 1,
                DefaultValue = 5
            }
        },

        [typeof(SupportVectorRegression<>)] = () => new Dictionary<string, ParameterRange>(StringComparer.Ordinal)
        {
            ["C"] = new ParameterRange
            {
                Type = ParameterType.Float,
                MinValue = 0.1,
                MaxValue = 100.0,
                UseLogScale = true,
                DefaultValue = 1.0
            },
            ["Epsilon"] = new ParameterRange
            {
                Type = ParameterType.Float,
                MinValue = 0.001,
                MaxValue = 1.0,
                UseLogScale = true,
                DefaultValue = 0.1
            }
        },

        [typeof(TimeSeriesRegression<>)] = () => new Dictionary<string, ParameterRange>(StringComparer.Ordinal)
        {
            ["P"] = new ParameterRange
            {
                Type = ParameterType.Integer,
                MinValue = 0,
                MaxValue = 5,
                Step = 1,
                DefaultValue = 1
            },
            ["D"] = new ParameterRange
            {
                Type = ParameterType.Integer,
                MinValue = 0,
                MaxValue = 2,
                Step = 1,
                DefaultValue = 1
            },
            ["Q"] = new ParameterRange
            {
                Type = ParameterType.Integer,
                MinValue = 0,
                MaxValue = 5,
                Step = 1,
                DefaultValue = 1
            },
            ["LagOrder"] = new ParameterRange
            {
                Type = ParameterType.Integer,
                MinValue = 1,
                MaxValue = 12,
                Step = 1,
                DefaultValue = 1
            },
            ["IncludeTrend"] = new ParameterRange
            {
                Type = ParameterType.Boolean,
                DefaultValue = true
            },
            ["SeasonalPeriod"] = new ParameterRange
            {
                Type = ParameterType.Integer,
                MinValue = 0,
                MaxValue = 24,
                Step = 1,
                DefaultValue = 0
            },
            ["AutocorrelationCorrection"] = new ParameterRange
            {
                Type = ParameterType.Boolean,
                DefaultValue = true
            }
        },
    };

    public static Dictionary<string, ParameterRange> GetDefaultSearchSpace(Type modelType)
    {
        var lookupType = modelType.IsGenericType ? modelType.GetGenericTypeDefinition() : modelType;

        if (_spaces.TryGetValue(lookupType, out var factory))
        {
            return factory();
        }

        return new Dictionary<string, ParameterRange>(StringComparer.Ordinal);
    }
}
