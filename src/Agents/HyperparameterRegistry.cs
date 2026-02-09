using System.Globalization;
using AiDotNet.Enums;

namespace AiDotNet.Agents;

/// <summary>
/// Maps ModelType to lists of HyperparameterDefinition, bridging LLM parameter names to C# property names.
/// </summary>
/// <remarks>
/// <para>
/// The registry provides a centralized mapping between how an LLM refers to hyperparameters
/// (using common ML naming conventions like "n_estimators", "learning_rate") and the actual
/// C# property names on the model's options classes. It also provides validation ranges
/// for each parameter.
/// </para>
/// <para><b>For Beginners:</b> This is like a translation dictionary. When an AI agent says
/// "set n_estimators to 100", the registry knows that means "set NumberOfTrees to 100"
/// on a RandomForest model. It also knows that valid values are between 1 and 10000.
/// </para>
/// </remarks>
internal class HyperparameterRegistry
{
    private readonly Dictionary<ModelType, List<HyperparameterDefinition>> _registry = new();
    private readonly List<HyperparameterDefinition> _sharedDefinitions = new();

    /// <summary>
    /// Creates a new HyperparameterRegistry with default registrations for all known model types.
    /// </summary>
    public HyperparameterRegistry()
    {
        RegisterDefaults();
    }

    /// <summary>
    /// Looks up the C# property name for a given LLM parameter name and model type.
    /// </summary>
    /// <param name="modelType">The model type to look up.</param>
    /// <param name="llmParamName">The LLM-style parameter name (e.g., "n_estimators").</param>
    /// <returns>The C# property name, or null if no mapping exists.</returns>
    public string? GetPropertyName(ModelType modelType, string llmParamName)
    {
        var normalized = HyperparameterDefinition.NormalizeName(llmParamName);

        // Check model-specific definitions first
        if (_registry.TryGetValue(modelType, out var definitions))
        {
            var match = definitions.FirstOrDefault(def => def.MatchesAlias(normalized));
            if (match != null)
            {
                return match.PropertyName;
            }
        }

        // Fall back to shared definitions
        var sharedMatch = _sharedDefinitions.FirstOrDefault(def => def.MatchesAlias(normalized));
        return sharedMatch?.PropertyName;
    }

    /// <summary>
    /// Gets the full HyperparameterDefinition for a given LLM parameter name and model type.
    /// </summary>
    public HyperparameterDefinition? GetDefinition(ModelType modelType, string llmParamName)
    {
        var normalized = HyperparameterDefinition.NormalizeName(llmParamName);

        if (_registry.TryGetValue(modelType, out var definitions))
        {
            var match = definitions.FirstOrDefault(def => def.MatchesAlias(normalized));
            if (match != null)
            {
                return match;
            }
        }

        return _sharedDefinitions.FirstOrDefault(def => def.MatchesAlias(normalized));
    }

    /// <summary>
    /// Validates a hyperparameter value against its definition's constraints.
    /// </summary>
    public HyperparameterValidationResult Validate(ModelType modelType, string paramName, object value)
    {
        var definition = GetDefinition(modelType, paramName);
        if (definition == null)
        {
            return HyperparameterValidationResult.Valid();
        }

        if (!TryConvertToDouble(value, out var numericValue))
        {
            return HyperparameterValidationResult.Valid();
        }

        if (definition.MinValue.HasValue && numericValue < definition.MinValue.Value)
        {
            return HyperparameterValidationResult.WithWarning(
                $"Value {numericValue} for '{paramName}' is below the typical minimum of {definition.MinValue.Value}");
        }

        if (definition.MaxValue.HasValue && numericValue > definition.MaxValue.Value)
        {
            return HyperparameterValidationResult.WithWarning(
                $"Value {numericValue} for '{paramName}' is above the typical maximum of {definition.MaxValue.Value}");
        }

        return HyperparameterValidationResult.Valid();
    }

    /// <summary>
    /// Registers a custom hyperparameter definition for a specific model type.
    /// </summary>
    public void Register(ModelType modelType, HyperparameterDefinition definition)
    {
        definition.BuildNormalizedAliases();

        if (!_registry.TryGetValue(modelType, out var definitions))
        {
            definitions = new List<HyperparameterDefinition>();
            _registry[modelType] = definitions;
        }

        definitions.Add(definition);
    }

    private static bool TryConvertToDouble(object value, out double result)
    {
        try
        {
            result = Convert.ToDouble(value, CultureInfo.InvariantCulture);
            return true;
        }
        catch (InvalidCastException)
        {
            result = 0;
            return false;
        }
        catch (FormatException)
        {
            result = 0;
            return false;
        }
        catch (OverflowException)
        {
            result = 0;
            return false;
        }
    }

    private void RegisterDefaults()
    {
        // Tree-Based Models
        RegisterTreeModels();

        // Neural Networks
        RegisterNeuralNetworkModels();

        // Linear/Regularized Models
        RegisterLinearModels();

        // Neighbor/Kernel Models
        RegisterNeighborKernelModels();

        // Time Series Models
        RegisterTimeSeriesModels();

        // Shared/Cross-Model Parameters
        RegisterSharedParameters();
    }

    private void RegisterTreeModels()
    {
        var treeModels = new[] { ModelType.RandomForest, ModelType.GradientBoosting, ModelType.DecisionTree,
            ModelType.ConditionalInferenceTree, ModelType.ExtremelyRandomizedTrees, ModelType.HistGradientBoosting,
            ModelType.AdaBoostR2 };

        foreach (var modelType in treeModels)
        {
            Register(modelType, new HyperparameterDefinition
            {
                PropertyName = "MaxDepth",
                Aliases = new List<string> { "max_depth", "maxdepth", "tree_depth", "depth" },
                ValueType = typeof(int),
                MinValue = 1,
                MaxValue = 100
            });

            Register(modelType, new HyperparameterDefinition
            {
                PropertyName = "MinSamplesSplit",
                Aliases = new List<string> { "min_samples_split", "min_split", "minsamplessplit" },
                ValueType = typeof(int),
                MinValue = 2,
                MaxValue = 1000
            });
        }

        // Ensemble-specific: number of trees
        var ensembleModels = new[] { ModelType.RandomForest, ModelType.GradientBoosting,
            ModelType.ExtremelyRandomizedTrees, ModelType.HistGradientBoosting, ModelType.AdaBoostR2 };

        foreach (var modelType in ensembleModels)
        {
            Register(modelType, new HyperparameterDefinition
            {
                PropertyName = "NumberOfTrees",
                Aliases = new List<string> { "n_estimators", "num_trees", "ntrees", "number_of_trees", "n_trees" },
                ValueType = typeof(int),
                MinValue = 1,
                MaxValue = 10000
            });
        }

        // GradientBoosting-specific
        Register(ModelType.GradientBoosting, new HyperparameterDefinition
        {
            PropertyName = "LearningRate",
            Aliases = new List<string> { "learning_rate", "lr", "eta", "shrinkage" },
            ValueType = typeof(double),
            MinValue = 0.0001,
            MaxValue = 1.0
        });

        Register(ModelType.GradientBoosting, new HyperparameterDefinition
        {
            PropertyName = "SubsampleRatio",
            Aliases = new List<string> { "subsample", "subsample_ratio", "sample_rate", "bagging_fraction" },
            ValueType = typeof(double),
            MinValue = 0.1,
            MaxValue = 1.0
        });
    }

    private void RegisterNeuralNetworkModels()
    {
        var nnModels = new[] { ModelType.NeuralNetworkRegression, ModelType.MultilayerPerceptronRegression };

        foreach (var modelType in nnModels)
        {
            Register(modelType, new HyperparameterDefinition
            {
                PropertyName = "LearningRate",
                Aliases = new List<string> { "learning_rate", "lr", "eta", "step_size" },
                ValueType = typeof(double),
                MinValue = 1e-5,
                MaxValue = 1.0
            });

            Register(modelType, new HyperparameterDefinition
            {
                PropertyName = "Epochs",
                Aliases = new List<string> { "epochs", "num_epochs", "n_epochs", "iterations", "max_epochs" },
                ValueType = typeof(int),
                MinValue = 1,
                MaxValue = 100000
            });

            Register(modelType, new HyperparameterDefinition
            {
                PropertyName = "BatchSize",
                Aliases = new List<string> { "batch_size", "batchsize", "mini_batch_size" },
                ValueType = typeof(int),
                MinValue = 1,
                MaxValue = 10000
            });
        }
    }

    private void RegisterLinearModels()
    {
        Register(ModelType.PolynomialRegression, new HyperparameterDefinition
        {
            PropertyName = "Degree",
            Aliases = new List<string> { "degree", "polynomial_degree", "poly_degree" },
            ValueType = typeof(int),
            MinValue = 1,
            MaxValue = 20
        });

        var regularizedModels = new[] { ModelType.RidgeRegression, ModelType.LassoRegression, ModelType.ElasticNetRegression };

        foreach (var modelType in regularizedModels)
        {
            Register(modelType, new HyperparameterDefinition
            {
                PropertyName = "Alpha",
                Aliases = new List<string> { "alpha", "regularization", "lambda", "reg_strength", "penalty" },
                ValueType = typeof(double),
                MinValue = 0.0001,
                MaxValue = 1000
            });
        }
    }

    private void RegisterNeighborKernelModels()
    {
        Register(ModelType.KNearestNeighbors, new HyperparameterDefinition
        {
            PropertyName = "K",
            Aliases = new List<string> { "n_neighbors", "k", "num_neighbors", "k_neighbors" },
            ValueType = typeof(int),
            MinValue = 1,
            MaxValue = 1000
        });

        Register(ModelType.SupportVectorRegression, new HyperparameterDefinition
        {
            PropertyName = "C",
            Aliases = new List<string> { "C", "c", "cost", "regularization_param" },
            ValueType = typeof(double),
            MinValue = 0.001,
            MaxValue = 10000
        });

        Register(ModelType.SupportVectorRegression, new HyperparameterDefinition
        {
            PropertyName = "Epsilon",
            Aliases = new List<string> { "epsilon", "eps", "svr_epsilon" },
            ValueType = typeof(double),
            MinValue = 0.0,
            MaxValue = 10.0
        });
    }

    private void RegisterTimeSeriesModels()
    {
        Register(ModelType.TimeSeriesRegression, new HyperparameterDefinition
        {
            PropertyName = "LagOrder",
            Aliases = new List<string> { "lag_order", "lag", "lags", "p", "ar_order" },
            ValueType = typeof(int),
            MinValue = 1,
            MaxValue = 100
        });

        Register(ModelType.TimeSeriesRegression, new HyperparameterDefinition
        {
            PropertyName = "SeasonalPeriod",
            Aliases = new List<string> { "seasonal_period", "seasonality", "period", "s", "season_length" },
            ValueType = typeof(int),
            MinValue = 2,
            MaxValue = 365
        });
    }

    private void RegisterSharedParameters()
    {
        _sharedDefinitions.Add(CreateDefinition("Seed",
            new List<string> { "seed", "random_seed", "random_state", "rng_seed" },
            typeof(int), null, null));

        _sharedDefinitions.Add(CreateDefinition("UseIntercept",
            new List<string> { "use_intercept", "fit_intercept", "intercept" },
            typeof(bool), null, null));
    }

    private static HyperparameterDefinition CreateDefinition(
        string propertyName, List<string> aliases, Type valueType,
        double? minValue, double? maxValue)
    {
        var def = new HyperparameterDefinition
        {
            PropertyName = propertyName,
            Aliases = aliases,
            ValueType = valueType,
            MinValue = minValue,
            MaxValue = maxValue
        };
        def.BuildNormalizedAliases();
        return def;
    }
}
