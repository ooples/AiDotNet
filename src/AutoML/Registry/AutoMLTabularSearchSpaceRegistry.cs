using AiDotNet.Enums;

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
    public static Dictionary<string, ParameterRange> GetDefaultSearchSpace(ModelType modelType)
    {
        return modelType switch
        {
            ModelType.PolynomialRegression => new Dictionary<string, ParameterRange>(StringComparer.Ordinal)
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

            ModelType.LogisticRegression => new Dictionary<string, ParameterRange>(StringComparer.Ordinal)
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

            ModelType.MultinomialLogisticRegression => new Dictionary<string, ParameterRange>(StringComparer.Ordinal)
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

            ModelType.RandomForest => new Dictionary<string, ParameterRange>(StringComparer.Ordinal)
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

            ModelType.DecisionTree => new Dictionary<string, ParameterRange>(StringComparer.Ordinal)
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

            ModelType.ExtremelyRandomizedTrees => new Dictionary<string, ParameterRange>(StringComparer.Ordinal)
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

            ModelType.AdaBoostR2 => new Dictionary<string, ParameterRange>(StringComparer.Ordinal)
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

            ModelType.QuantileRegressionForests => new Dictionary<string, ParameterRange>(StringComparer.Ordinal)
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

            ModelType.GradientBoosting => new Dictionary<string, ParameterRange>(StringComparer.Ordinal)
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

            ModelType.KNearestNeighbors => new Dictionary<string, ParameterRange>(StringComparer.Ordinal)
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

            ModelType.SupportVectorRegression => new Dictionary<string, ParameterRange>(StringComparer.Ordinal)
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

            ModelType.TimeSeriesRegression => new Dictionary<string, ParameterRange>(StringComparer.Ordinal)
            {
                // ARIMA defaults (P/D/Q) with conservative bounds suitable for v1.
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
                // Shared time-series regression knobs.
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

            _ => new Dictionary<string, ParameterRange>(StringComparer.Ordinal)
        };
    }
}
