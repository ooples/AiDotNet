using AiDotNet.AutoML;
using AiDotNet.Enums;

namespace AiDotNet.Finance.AutoML;

/// <summary>
/// Provides default AutoML search spaces for finance models.
/// </summary>
/// <remarks>
/// <para>
/// The search space defines which hyperparameters AutoML is allowed to explore.
/// Each model type has specific hyperparameters that can be tuned during AutoML
/// optimization, such as learning rate, hidden size, and dropout rate.
/// </para>
/// <para>
/// <b>For Beginners:</b> AutoML can either just pick a model or also tune settings.
/// This class tells AutoML which settings it is allowed to tune and what ranges
/// are reasonable for each setting. For example, a learning rate should typically
/// be between 0.0001 and 0.1, while the number of layers might be 1 to 8.
/// </para>
/// </remarks>
public sealed class FinancialSearchSpace
{
    private readonly FinancialDomain _domain;

    /// <summary>
    /// Initializes a new search space provider for the chosen finance domain.
    /// </summary>
    /// <param name="domain">The finance domain.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Different tasks (forecasting vs risk) use different models,
    /// so the search space is tied to the domain.
    /// </para>
    /// </remarks>
    public FinancialSearchSpace(FinancialDomain domain)
    {
        _domain = domain;
    }

    /// <summary>
    /// Gets the default search space for a specific model type.
    /// </summary>
    /// <param name="modelType">The model type to configure.</param>
    /// <returns>Dictionary of parameter ranges for AutoML sampling.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method returns a dictionary of hyperparameter names
    /// and their allowed ranges. AutoML will sample from these ranges to find the
    /// best configuration for your specific data.
    /// </para>
    /// </remarks>
    public Dictionary<string, ParameterRange> GetSearchSpace(ModelType modelType)
    {
        return modelType switch
        {
            ModelType.PatchTST => GetPatchTSTSearchSpace(),
            ModelType.ITransformer => GetITransformerSearchSpace(),
            ModelType.DeepAR => GetDeepARSearchSpace(),
            ModelType.NBEATS => GetNBEATSSearchSpace(),
            ModelType.TFT => GetTFTSearchSpace(),
            ModelType.NeuralVaR => GetNeuralVaRSearchSpace(),
            ModelType.TabNet => GetTabNetSearchSpace(),
            ModelType.TabTransformer => GetTabTransformerSearchSpace(),
            _ => GetDefaultSearchSpace()
        };
    }

    /// <summary>
    /// Gets the search space for PatchTST model.
    /// </summary>
    /// <returns>Dictionary of parameter ranges for PatchTST.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> PatchTST is a transformer-based model that divides time series
    /// into patches. Key hyperparameters include patch size, number of attention heads,
    /// and the hidden dimension size.
    /// </para>
    /// </remarks>
    private Dictionary<string, ParameterRange> GetPatchTSTSearchSpace()
    {
        var searchSpace = GetCommonForecastingSearchSpace();
        searchSpace["PatchLength"] = new ParameterRange
        {
            Type = ParameterType.Integer,
            MinValue = 4,
            MaxValue = 32,
            DefaultValue = 16
        };
        searchSpace["NumAttentionHeads"] = new ParameterRange
        {
            Type = ParameterType.Integer,
            MinValue = 1,
            MaxValue = 8,
            DefaultValue = 4
        };
        return searchSpace;
    }

    /// <summary>
    /// Gets the search space for iTransformer model.
    /// </summary>
    /// <returns>Dictionary of parameter ranges for iTransformer.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> iTransformer treats each variable as a token rather than
    /// each time step. Key hyperparameters include the number of transformer layers
    /// and attention heads.
    /// </para>
    /// </remarks>
    private Dictionary<string, ParameterRange> GetITransformerSearchSpace()
    {
        var searchSpace = GetCommonForecastingSearchSpace();
        searchSpace["NumAttentionHeads"] = new ParameterRange
        {
            Type = ParameterType.Integer,
            MinValue = 1,
            MaxValue = 8,
            DefaultValue = 4
        };
        searchSpace["FeedForwardDim"] = new ParameterRange
        {
            Type = ParameterType.Integer,
            MinValue = 64,
            MaxValue = 512,
            DefaultValue = 256
        };
        return searchSpace;
    }

    /// <summary>
    /// Gets the search space for DeepAR model.
    /// </summary>
    /// <returns>Dictionary of parameter ranges for DeepAR.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> DeepAR is a probabilistic forecasting model using LSTMs.
    /// Key hyperparameters include the LSTM hidden size, number of layers, and the
    /// number of samples for probabilistic predictions.
    /// </para>
    /// </remarks>
    private Dictionary<string, ParameterRange> GetDeepARSearchSpace()
    {
        var searchSpace = GetCommonForecastingSearchSpace();
        searchSpace["NumSamples"] = new ParameterRange
        {
            Type = ParameterType.Integer,
            MinValue = 50,
            MaxValue = 200,
            DefaultValue = 100
        };
        searchSpace["LikelihoodType"] = new ParameterRange
        {
            Type = ParameterType.Categorical,
            CategoricalValues = new List<object> { "Gaussian", "StudentT", "NegativeBinomial" },
            DefaultValue = "Gaussian"
        };
        return searchSpace;
    }

    /// <summary>
    /// Gets the search space for N-BEATS model.
    /// </summary>
    /// <returns>Dictionary of parameter ranges for N-BEATS.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> N-BEATS is a pure deep learning forecasting model with
    /// interpretable components. Key hyperparameters include the number of stacks,
    /// blocks per stack, and whether to use interpretable basis functions.
    /// </para>
    /// </remarks>
    private Dictionary<string, ParameterRange> GetNBEATSSearchSpace()
    {
        var searchSpace = GetCommonForecastingSearchSpace();
        searchSpace["NumStacks"] = new ParameterRange
        {
            Type = ParameterType.Integer,
            MinValue = 5,
            MaxValue = 50,
            DefaultValue = 30
        };
        searchSpace["NumBlocksPerStack"] = new ParameterRange
        {
            Type = ParameterType.Integer,
            MinValue = 1,
            MaxValue = 5,
            DefaultValue = 1
        };
        searchSpace["UseInterpretableBasis"] = new ParameterRange
        {
            Type = ParameterType.Boolean,
            DefaultValue = true
        };
        return searchSpace;
    }

    /// <summary>
    /// Gets the search space for Temporal Fusion Transformer (TFT) model.
    /// </summary>
    /// <returns>Dictionary of parameter ranges for TFT.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> TFT combines attention mechanisms with specialized gating
    /// for interpretable forecasting. Key hyperparameters include attention heads,
    /// hidden size, and whether to use variable selection.
    /// </para>
    /// </remarks>
    private Dictionary<string, ParameterRange> GetTFTSearchSpace()
    {
        var searchSpace = GetCommonForecastingSearchSpace();
        searchSpace["NumAttentionHeads"] = new ParameterRange
        {
            Type = ParameterType.Integer,
            MinValue = 1,
            MaxValue = 8,
            DefaultValue = 4
        };
        searchSpace["UseVariableSelection"] = new ParameterRange
        {
            Type = ParameterType.Boolean,
            DefaultValue = true
        };
        return searchSpace;
    }

    /// <summary>
    /// Gets the search space for NeuralVaR model.
    /// </summary>
    /// <returns>Dictionary of parameter ranges for NeuralVaR.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> NeuralVaR estimates Value at Risk using neural networks.
    /// Key hyperparameters include the confidence level for VaR estimation and the
    /// network architecture settings.
    /// </para>
    /// </remarks>
    private Dictionary<string, ParameterRange> GetNeuralVaRSearchSpace()
    {
        var searchSpace = GetCommonRiskSearchSpace();
        searchSpace["ConfidenceLevel"] = new ParameterRange
        {
            Type = ParameterType.Float,
            MinValue = 0.90,
            MaxValue = 0.99,
            DefaultValue = 0.95
        };
        return searchSpace;
    }

    /// <summary>
    /// Gets the search space for TabNet model.
    /// </summary>
    /// <returns>Dictionary of parameter ranges for TabNet.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> TabNet uses attentive feature selection for tabular data.
    /// Key hyperparameters include the number of decision steps and relaxation factor
    /// that controls feature reuse.
    /// </para>
    /// </remarks>
    private Dictionary<string, ParameterRange> GetTabNetSearchSpace()
    {
        var searchSpace = GetCommonTabularSearchSpace();
        searchSpace["NumDecisionSteps"] = new ParameterRange
        {
            Type = ParameterType.Integer,
            MinValue = 3,
            MaxValue = 10,
            DefaultValue = 5
        };
        searchSpace["RelaxationFactor"] = new ParameterRange
        {
            Type = ParameterType.Float,
            MinValue = 1.0,
            MaxValue = 2.0,
            DefaultValue = 1.5
        };
        return searchSpace;
    }

    /// <summary>
    /// Gets the search space for TabTransformer model.
    /// </summary>
    /// <returns>Dictionary of parameter ranges for TabTransformer.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> TabTransformer applies transformer attention to tabular data
    /// columns. Key hyperparameters include the number of attention heads and embedding
    /// dimensions for categorical features.
    /// </para>
    /// </remarks>
    private Dictionary<string, ParameterRange> GetTabTransformerSearchSpace()
    {
        var searchSpace = GetCommonTabularSearchSpace();
        searchSpace["NumAttentionHeads"] = new ParameterRange
        {
            Type = ParameterType.Integer,
            MinValue = 1,
            MaxValue = 8,
            DefaultValue = 4
        };
        searchSpace["EmbeddingDim"] = new ParameterRange
        {
            Type = ParameterType.Integer,
            MinValue = 8,
            MaxValue = 64,
            DefaultValue = 32
        };
        return searchSpace;
    }

    /// <summary>
    /// Gets common search space parameters for forecasting models.
    /// </summary>
    /// <returns>Dictionary of common parameter ranges for forecasting.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> These are hyperparameters that most forecasting models share,
    /// like learning rate, hidden layer size, and dropout for regularization.
    /// </para>
    /// </remarks>
    private Dictionary<string, ParameterRange> GetCommonForecastingSearchSpace()
    {
        return new Dictionary<string, ParameterRange>(StringComparer.Ordinal)
        {
            ["LearningRate"] = new ParameterRange
            {
                Type = ParameterType.Float,
                MinValue = 0.0001,
                MaxValue = 0.1,
                UseLogScale = true,
                DefaultValue = 0.001
            },
            ["HiddenSize"] = new ParameterRange
            {
                Type = ParameterType.Integer,
                MinValue = 16,
                MaxValue = 512,
                DefaultValue = 128
            },
            ["NumLayers"] = new ParameterRange
            {
                Type = ParameterType.Integer,
                MinValue = 1,
                MaxValue = 8,
                DefaultValue = 2
            },
            ["DropoutRate"] = new ParameterRange
            {
                Type = ParameterType.Float,
                MinValue = 0.0,
                MaxValue = 0.5,
                DefaultValue = 0.1
            },
            ["Epochs"] = new ParameterRange
            {
                Type = ParameterType.Integer,
                MinValue = 10,
                MaxValue = 500,
                DefaultValue = 100
            },
            ["BatchSize"] = new ParameterRange
            {
                Type = ParameterType.Integer,
                MinValue = 8,
                MaxValue = 128,
                DefaultValue = 32
            }
        };
    }

    /// <summary>
    /// Gets common search space parameters for risk models.
    /// </summary>
    /// <returns>Dictionary of common parameter ranges for risk models.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Risk models typically use smaller networks and require
    /// careful tuning of confidence levels for proper risk estimation.
    /// </para>
    /// </remarks>
    private Dictionary<string, ParameterRange> GetCommonRiskSearchSpace()
    {
        return new Dictionary<string, ParameterRange>(StringComparer.Ordinal)
        {
            ["LearningRate"] = new ParameterRange
            {
                Type = ParameterType.Float,
                MinValue = 0.0001,
                MaxValue = 0.01,
                UseLogScale = true,
                DefaultValue = 0.001
            },
            ["HiddenSize"] = new ParameterRange
            {
                Type = ParameterType.Integer,
                MinValue = 16,
                MaxValue = 256,
                DefaultValue = 64
            },
            ["NumLayers"] = new ParameterRange
            {
                Type = ParameterType.Integer,
                MinValue = 1,
                MaxValue = 6,
                DefaultValue = 2
            },
            ["DropoutRate"] = new ParameterRange
            {
                Type = ParameterType.Float,
                MinValue = 0.0,
                MaxValue = 0.5,
                DefaultValue = 0.1
            },
            ["Epochs"] = new ParameterRange
            {
                Type = ParameterType.Integer,
                MinValue = 10,
                MaxValue = 300,
                DefaultValue = 100
            },
            ["BatchSize"] = new ParameterRange
            {
                Type = ParameterType.Integer,
                MinValue = 16,
                MaxValue = 128,
                DefaultValue = 32
            }
        };
    }

    /// <summary>
    /// Gets common search space parameters for tabular models.
    /// </summary>
    /// <returns>Dictionary of common parameter ranges for tabular models.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Tabular models like TabNet and TabTransformer work on
    /// structured data with rows and columns. They typically need attention-related
    /// hyperparameters for feature selection.
    /// </para>
    /// </remarks>
    private Dictionary<string, ParameterRange> GetCommonTabularSearchSpace()
    {
        return new Dictionary<string, ParameterRange>(StringComparer.Ordinal)
        {
            ["LearningRate"] = new ParameterRange
            {
                Type = ParameterType.Float,
                MinValue = 0.0001,
                MaxValue = 0.05,
                UseLogScale = true,
                DefaultValue = 0.001
            },
            ["HiddenSize"] = new ParameterRange
            {
                Type = ParameterType.Integer,
                MinValue = 8,
                MaxValue = 256,
                DefaultValue = 64
            },
            ["NumLayers"] = new ParameterRange
            {
                Type = ParameterType.Integer,
                MinValue = 1,
                MaxValue = 6,
                DefaultValue = 2
            },
            ["DropoutRate"] = new ParameterRange
            {
                Type = ParameterType.Float,
                MinValue = 0.0,
                MaxValue = 0.4,
                DefaultValue = 0.1
            },
            ["Epochs"] = new ParameterRange
            {
                Type = ParameterType.Integer,
                MinValue = 10,
                MaxValue = 200,
                DefaultValue = 50
            },
            ["BatchSize"] = new ParameterRange
            {
                Type = ParameterType.Integer,
                MinValue = 32,
                MaxValue = 256,
                DefaultValue = 64
            }
        };
    }

    /// <summary>
    /// Gets a default search space for unsupported model types.
    /// </summary>
    /// <returns>Dictionary of minimal parameter ranges.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When a model type isn't specifically configured, we provide
    /// a minimal set of common hyperparameters that most neural networks use.
    /// </para>
    /// </remarks>
    private static Dictionary<string, ParameterRange> GetDefaultSearchSpace()
    {
        return new Dictionary<string, ParameterRange>(StringComparer.Ordinal)
        {
            ["LearningRate"] = new ParameterRange
            {
                Type = ParameterType.Float,
                MinValue = 0.0001,
                MaxValue = 0.1,
                UseLogScale = true,
                DefaultValue = 0.001
            },
            ["Epochs"] = new ParameterRange
            {
                Type = ParameterType.Integer,
                MinValue = 10,
                MaxValue = 200,
                DefaultValue = 50
            }
        };
    }
}
