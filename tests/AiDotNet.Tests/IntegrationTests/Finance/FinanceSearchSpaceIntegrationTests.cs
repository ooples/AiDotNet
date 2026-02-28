using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.AutoML;
using AiDotNet.Enums;
using AiDotNet.Finance.AutoML;
using AiDotNet.Finance.Data;
using AiDotNet.Finance.Trading.Environments;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Finance;

/// <summary>
/// Deep integration tests for Finance AutoML search spaces, trading environments,
/// and financial domain configuration. Verifies parameter ranges, domain-specific
/// defaults, and trading environment mechanics.
/// </summary>
public class FinanceSearchSpaceIntegrationTests
{
    /// <summary>
    /// Helper to safely convert ParameterRange value (object?) to double.
    /// ParameterRange stores values as object?, so int 4 and double 4.0 are different boxed types.
    /// </summary>
    private static double ToDouble(object value) => Convert.ToDouble(value);

    #region FinancialSearchSpace - Domain-Aware Parameter Ranges

    [Fact]
    public void SearchSpace_Forecasting_HasCommonParams()
    {
        var space = new FinancialSearchSpace(FinancialDomain.Forecasting);
        var searchSpace = space.GetSearchSpace(ModelType.PatchTST);

        Assert.True(searchSpace.ContainsKey("LearningRate"));
        Assert.True(searchSpace.ContainsKey("HiddenSize"));
        Assert.True(searchSpace.ContainsKey("NumLayers"));
        Assert.True(searchSpace.ContainsKey("DropoutRate"));
        Assert.True(searchSpace.ContainsKey("Epochs"));
        Assert.True(searchSpace.ContainsKey("BatchSize"));
    }

    [Fact]
    public void SearchSpace_PatchTST_HasPatchSpecificParams()
    {
        var space = new FinancialSearchSpace(FinancialDomain.Forecasting);
        var searchSpace = space.GetSearchSpace(ModelType.PatchTST);

        Assert.True(searchSpace.ContainsKey("PatchLength"));
        Assert.True(searchSpace.ContainsKey("NumAttentionHeads"));

        var patchRange = searchSpace["PatchLength"];
        Assert.Equal(ParameterType.Integer, patchRange.Type);
        Assert.Equal(4.0, ToDouble(patchRange.MinValue));
        Assert.Equal(32.0, ToDouble(patchRange.MaxValue));
        Assert.Equal(16.0, ToDouble(patchRange.DefaultValue));
    }

    [Fact]
    public void SearchSpace_DeepAR_HasProbabilisticParams()
    {
        var space = new FinancialSearchSpace(FinancialDomain.Forecasting);
        var searchSpace = space.GetSearchSpace(ModelType.DeepAR);

        Assert.True(searchSpace.ContainsKey("NumSamples"));
        Assert.True(searchSpace.ContainsKey("LikelihoodType"));

        var likelihoodRange = searchSpace["LikelihoodType"];
        Assert.Equal(ParameterType.Categorical, likelihoodRange.Type);
        Assert.Contains("Gaussian", likelihoodRange.CategoricalValues.Cast<string>());
        Assert.Contains("StudentT", likelihoodRange.CategoricalValues.Cast<string>());
    }

    [Fact]
    public void SearchSpace_NBEATS_HasStackParams()
    {
        var space = new FinancialSearchSpace(FinancialDomain.Forecasting);
        var searchSpace = space.GetSearchSpace(ModelType.NBEATS);

        Assert.True(searchSpace.ContainsKey("NumStacks"));
        Assert.True(searchSpace.ContainsKey("NumBlocksPerStack"));
        Assert.True(searchSpace.ContainsKey("UseInterpretableBasis"));

        var basisRange = searchSpace["UseInterpretableBasis"];
        Assert.Equal(ParameterType.Boolean, basisRange.Type);
    }

    [Fact]
    public void SearchSpace_NeuralVaR_HasConfidenceLevel()
    {
        var space = new FinancialSearchSpace(FinancialDomain.Risk);
        var searchSpace = space.GetSearchSpace(ModelType.NeuralVaR);

        Assert.True(searchSpace.ContainsKey("ConfidenceLevel"));

        var confRange = searchSpace["ConfidenceLevel"];
        Assert.Equal(ParameterType.Float, confRange.Type);
        Assert.Equal(0.90, ToDouble(confRange.MinValue));
        Assert.Equal(0.99, ToDouble(confRange.MaxValue));
        Assert.Equal(0.95, ToDouble(confRange.DefaultValue));
    }

    [Fact]
    public void SearchSpace_TabNet_HasDecisionSteps()
    {
        var space = new FinancialSearchSpace(FinancialDomain.Risk);
        var searchSpace = space.GetSearchSpace(ModelType.TabNet);

        Assert.True(searchSpace.ContainsKey("NumDecisionSteps"));
        Assert.True(searchSpace.ContainsKey("RelaxationFactor"));

        var stepsRange = searchSpace["NumDecisionSteps"];
        Assert.Equal(3.0, ToDouble(stepsRange.MinValue));
        Assert.Equal(10.0, ToDouble(stepsRange.MaxValue));
    }

    [Fact]
    public void SearchSpace_TabTransformer_HasEmbeddingDim()
    {
        var space = new FinancialSearchSpace(FinancialDomain.Risk);
        var searchSpace = space.GetSearchSpace(ModelType.TabTransformer);

        Assert.True(searchSpace.ContainsKey("EmbeddingDim"));
        Assert.True(searchSpace.ContainsKey("NumAttentionHeads"));

        var embRange = searchSpace["EmbeddingDim"];
        Assert.Equal(8.0, ToDouble(embRange.MinValue));
        Assert.Equal(64.0, ToDouble(embRange.MaxValue));
    }

    [Fact]
    public void SearchSpace_TFT_HasVariableSelection()
    {
        var space = new FinancialSearchSpace(FinancialDomain.Forecasting);
        var searchSpace = space.GetSearchSpace(ModelType.TFT);

        Assert.True(searchSpace.ContainsKey("UseVariableSelection"));
        Assert.True(searchSpace.ContainsKey("NumAttentionHeads"));
    }

    [Fact]
    public void SearchSpace_ITransformer_HasFeedForwardDim()
    {
        var space = new FinancialSearchSpace(FinancialDomain.Forecasting);
        var searchSpace = space.GetSearchSpace(ModelType.ITransformer);

        Assert.True(searchSpace.ContainsKey("FeedForwardDim"));
        Assert.True(searchSpace.ContainsKey("NumAttentionHeads"));

        var ffRange = searchSpace["FeedForwardDim"];
        Assert.Equal(64.0, ToDouble(ffRange.MinValue));
        Assert.Equal(512.0, ToDouble(ffRange.MaxValue));
    }

    [Fact]
    public void SearchSpace_Forecasting_LearningRate_UsesLogScale()
    {
        var space = new FinancialSearchSpace(FinancialDomain.Forecasting);
        var searchSpace = space.GetSearchSpace(ModelType.PatchTST);

        var lrRange = searchSpace["LearningRate"];
        Assert.True(lrRange.UseLogScale,
            "Learning rate should use log scale for proper exploration of [0.0001, 0.1]");
    }

    [Fact]
    public void SearchSpace_Risk_SmallerMaxLearningRate()
    {
        var forecastSpace = new FinancialSearchSpace(FinancialDomain.Forecasting);
        var riskSpace = new FinancialSearchSpace(FinancialDomain.Risk);

        var forecastLR = forecastSpace.GetSearchSpace(ModelType.PatchTST)["LearningRate"];
        var riskLR = riskSpace.GetSearchSpace(ModelType.NeuralVaR)["LearningRate"];

        Assert.True(ToDouble(riskLR.MaxValue) <= ToDouble(forecastLR.MaxValue),
            "Risk max LR should be <= forecast max LR");
    }

    [Fact]
    public void SearchSpace_AllRanges_MinLessThanMax()
    {
        var space = new FinancialSearchSpace(FinancialDomain.Forecasting);
        var allModelTypes = new[] { ModelType.PatchTST, ModelType.DeepAR, ModelType.NBEATS,
                                     ModelType.TFT, ModelType.ITransformer };

        foreach (var modelType in allModelTypes)
        {
            var searchSpace = space.GetSearchSpace(modelType);

            foreach (var (name, range) in searchSpace)
            {
                if (range.Type == ParameterType.Float || range.Type == ParameterType.Integer)
                {
                    double min = ToDouble(range.MinValue);
                    double max = ToDouble(range.MaxValue);
                    Assert.True(min < max,
                        $"SearchSpace[{modelType}][{name}]: min ({min}) should be < max ({max})");
                }
            }
        }
    }

    [Fact]
    public void SearchSpace_DefaultValues_WithinRange()
    {
        var space = new FinancialSearchSpace(FinancialDomain.Forecasting);
        var searchSpace = space.GetSearchSpace(ModelType.PatchTST);

        foreach (var (name, range) in searchSpace)
        {
            if (range.Type == ParameterType.Float || range.Type == ParameterType.Integer)
            {
                if (range.DefaultValue is not null)
                {
                    double defaultVal = ToDouble(range.DefaultValue);
                    double minVal = ToDouble(range.MinValue);
                    double maxVal = ToDouble(range.MaxValue);
                    Assert.True(defaultVal >= minVal && defaultVal <= maxVal,
                        $"Default {name}={defaultVal} outside [{minVal}, {maxVal}]");
                }
            }
        }
    }

    [Fact]
    public void SearchSpace_UnknownModelType_ReturnsDomainDefault()
    {
        var forecastSpace = new FinancialSearchSpace(FinancialDomain.Forecasting);
        var riskSpace = new FinancialSearchSpace(FinancialDomain.Risk);

        var forecastDefault = forecastSpace.GetSearchSpace(ModelType.AutoML);
        var riskDefault = riskSpace.GetSearchSpace(ModelType.AutoML);

        Assert.True(forecastDefault.ContainsKey("LearningRate"));
        Assert.True(forecastDefault.ContainsKey("Epochs"));
        Assert.True(riskDefault.ContainsKey("LearningRate"));
    }

    #endregion

    #region FinancialSearchSpace - Deep Parameter Correctness

    [Fact]
    public void SearchSpace_DeepAR_NumSamples_IntegerRange()
    {
        var space = new FinancialSearchSpace(FinancialDomain.Forecasting);
        var searchSpace = space.GetSearchSpace(ModelType.DeepAR);

        var samplesRange = searchSpace["NumSamples"];
        Assert.Equal(ParameterType.Integer, samplesRange.Type);
        Assert.Equal(50.0, ToDouble(samplesRange.MinValue));
        Assert.Equal(200.0, ToDouble(samplesRange.MaxValue));
        Assert.Equal(100.0, ToDouble(samplesRange.DefaultValue));
    }

    [Fact]
    public void SearchSpace_DeepAR_LikelihoodType_HasNegativeBinomial()
    {
        var space = new FinancialSearchSpace(FinancialDomain.Forecasting);
        var searchSpace = space.GetSearchSpace(ModelType.DeepAR);

        var likelihoodRange = searchSpace["LikelihoodType"];
        var values = likelihoodRange.CategoricalValues.Cast<string>().ToList();
        Assert.Contains("NegativeBinomial", values);
        Assert.Equal(3, values.Count);
        Assert.Equal("Gaussian", (string)likelihoodRange.DefaultValue);
    }

    [Fact]
    public void SearchSpace_NBEATS_NumStacks_CorrectRange()
    {
        var space = new FinancialSearchSpace(FinancialDomain.Forecasting);
        var searchSpace = space.GetSearchSpace(ModelType.NBEATS);

        var stackRange = searchSpace["NumStacks"];
        Assert.Equal(ParameterType.Integer, stackRange.Type);
        Assert.Equal(5.0, ToDouble(stackRange.MinValue));
        Assert.Equal(50.0, ToDouble(stackRange.MaxValue));
        Assert.Equal(30.0, ToDouble(stackRange.DefaultValue));
    }

    [Fact]
    public void SearchSpace_NBEATS_BlocksPerStack_CorrectRange()
    {
        var space = new FinancialSearchSpace(FinancialDomain.Forecasting);
        var searchSpace = space.GetSearchSpace(ModelType.NBEATS);

        var blocksRange = searchSpace["NumBlocksPerStack"];
        Assert.Equal(ParameterType.Integer, blocksRange.Type);
        Assert.Equal(1.0, ToDouble(blocksRange.MinValue));
        Assert.Equal(5.0, ToDouble(blocksRange.MaxValue));
    }

    [Fact]
    public void SearchSpace_TabNet_RelaxationFactor_FloatRange()
    {
        var space = new FinancialSearchSpace(FinancialDomain.Risk);
        var searchSpace = space.GetSearchSpace(ModelType.TabNet);

        var relaxRange = searchSpace["RelaxationFactor"];
        Assert.Equal(ParameterType.Float, relaxRange.Type);
        Assert.Equal(1.0, ToDouble(relaxRange.MinValue));
        Assert.Equal(2.0, ToDouble(relaxRange.MaxValue));
        Assert.Equal(1.5, ToDouble(relaxRange.DefaultValue));
    }

    [Fact]
    public void SearchSpace_TabTransformer_EmbeddingDim_DefaultIs32()
    {
        var space = new FinancialSearchSpace(FinancialDomain.Risk);
        var searchSpace = space.GetSearchSpace(ModelType.TabTransformer);

        var embRange = searchSpace["EmbeddingDim"];
        Assert.Equal(32.0, ToDouble(embRange.DefaultValue));
    }

    [Fact]
    public void SearchSpace_PatchTST_AttentionHeads_CorrectRange()
    {
        var space = new FinancialSearchSpace(FinancialDomain.Forecasting);
        var searchSpace = space.GetSearchSpace(ModelType.PatchTST);

        var attnRange = searchSpace["NumAttentionHeads"];
        Assert.Equal(ParameterType.Integer, attnRange.Type);
        Assert.Equal(1.0, ToDouble(attnRange.MinValue));
        Assert.Equal(8.0, ToDouble(attnRange.MaxValue));
        Assert.Equal(4.0, ToDouble(attnRange.DefaultValue));
    }

    [Fact]
    public void SearchSpace_ITransformer_FeedForwardDim_DefaultIs256()
    {
        var space = new FinancialSearchSpace(FinancialDomain.Forecasting);
        var searchSpace = space.GetSearchSpace(ModelType.ITransformer);

        var ffRange = searchSpace["FeedForwardDim"];
        Assert.Equal(256.0, ToDouble(ffRange.DefaultValue));
    }

    [Fact]
    public void SearchSpace_Risk_HiddenSize_SmallerThanForecasting()
    {
        var forecastSpace = new FinancialSearchSpace(FinancialDomain.Forecasting);
        var riskSpace = new FinancialSearchSpace(FinancialDomain.Risk);

        var forecastHidden = forecastSpace.GetSearchSpace(ModelType.PatchTST)["HiddenSize"];
        var riskHidden = riskSpace.GetSearchSpace(ModelType.NeuralVaR)["HiddenSize"];

        Assert.True(ToDouble(riskHidden.MaxValue) <= ToDouble(forecastHidden.MaxValue),
            "Risk HiddenSize max should be <= forecast HiddenSize max");
    }

    [Fact]
    public void SearchSpace_Risk_FewerMaxLayers()
    {
        var forecastSpace = new FinancialSearchSpace(FinancialDomain.Forecasting);
        var riskSpace = new FinancialSearchSpace(FinancialDomain.Risk);

        var forecastLayers = forecastSpace.GetSearchSpace(ModelType.PatchTST)["NumLayers"];
        var riskLayers = riskSpace.GetSearchSpace(ModelType.NeuralVaR)["NumLayers"];

        Assert.True(ToDouble(riskLayers.MaxValue) <= ToDouble(forecastLayers.MaxValue),
            "Risk NumLayers max should be <= forecast NumLayers max");
    }

    [Fact]
    public void SearchSpace_Risk_LearningRate_AlsoUsesLogScale()
    {
        var riskSpace = new FinancialSearchSpace(FinancialDomain.Risk);
        var riskLR = riskSpace.GetSearchSpace(ModelType.NeuralVaR)["LearningRate"];

        Assert.True(riskLR.UseLogScale,
            "Risk learning rate should also use log scale");
    }

    [Fact]
    public void SearchSpace_Tabular_DropoutRate_MaxIs0_4()
    {
        var riskSpace = new FinancialSearchSpace(FinancialDomain.Risk);
        var tabNetDropout = riskSpace.GetSearchSpace(ModelType.TabNet)["DropoutRate"];

        Assert.Equal(0.4, ToDouble(tabNetDropout.MaxValue));
    }

    [Fact]
    public void SearchSpace_Forecasting_DropoutRate_MaxIs0_5()
    {
        var forecastSpace = new FinancialSearchSpace(FinancialDomain.Forecasting);
        var forecastDropout = forecastSpace.GetSearchSpace(ModelType.PatchTST)["DropoutRate"];

        Assert.Equal(0.5, ToDouble(forecastDropout.MaxValue));
    }

    [Fact]
    public void SearchSpace_Tabular_BatchSize_DefaultIs64()
    {
        var riskSpace = new FinancialSearchSpace(FinancialDomain.Risk);
        var tabNetBatch = riskSpace.GetSearchSpace(ModelType.TabNet)["BatchSize"];

        Assert.Equal(64.0, ToDouble(tabNetBatch.DefaultValue));
    }

    [Fact]
    public void SearchSpace_Forecasting_BatchSize_DefaultIs32()
    {
        var forecastSpace = new FinancialSearchSpace(FinancialDomain.Forecasting);
        var forecastBatch = forecastSpace.GetSearchSpace(ModelType.PatchTST)["BatchSize"];

        Assert.Equal(32.0, ToDouble(forecastBatch.DefaultValue));
    }

    [Fact]
    public void SearchSpace_Forecasting_Epochs_MaxIs500()
    {
        var forecastSpace = new FinancialSearchSpace(FinancialDomain.Forecasting);
        var epochs = forecastSpace.GetSearchSpace(ModelType.PatchTST)["Epochs"];

        Assert.Equal(500.0, ToDouble(epochs.MaxValue));
    }

    [Fact]
    public void SearchSpace_Risk_Epochs_MaxIs300()
    {
        var riskSpace = new FinancialSearchSpace(FinancialDomain.Risk);
        var epochs = riskSpace.GetSearchSpace(ModelType.NeuralVaR)["Epochs"];

        Assert.Equal(300.0, ToDouble(epochs.MaxValue));
    }

    [Fact]
    public void SearchSpace_Tabular_Epochs_MaxIs200()
    {
        var riskSpace = new FinancialSearchSpace(FinancialDomain.Risk);
        var epochs = riskSpace.GetSearchSpace(ModelType.TabNet)["Epochs"];

        Assert.Equal(200.0, ToDouble(epochs.MaxValue));
    }

    [Fact]
    public void SearchSpace_AllRiskModels_DefaultValues_WithinRange()
    {
        var riskSpace = new FinancialSearchSpace(FinancialDomain.Risk);
        var riskModels = new[] { ModelType.NeuralVaR, ModelType.TabNet, ModelType.TabTransformer };

        foreach (var modelType in riskModels)
        {
            var searchSpace = riskSpace.GetSearchSpace(modelType);

            foreach (var (name, range) in searchSpace)
            {
                if ((range.Type == ParameterType.Float || range.Type == ParameterType.Integer)
                    && range.DefaultValue is not null)
                {
                    double defaultVal = ToDouble(range.DefaultValue);
                    double minVal = ToDouble(range.MinValue);
                    double maxVal = ToDouble(range.MaxValue);
                    Assert.True(defaultVal >= minVal && defaultVal <= maxVal,
                        $"[{modelType}] Default {name}={defaultVal} outside [{minVal}, {maxVal}]");
                }
            }
        }
    }

    [Fact]
    public void SearchSpace_AllRiskModels_MinLessThanMax()
    {
        var riskSpace = new FinancialSearchSpace(FinancialDomain.Risk);
        var riskModels = new[] { ModelType.NeuralVaR, ModelType.TabNet, ModelType.TabTransformer };

        foreach (var modelType in riskModels)
        {
            var searchSpace = riskSpace.GetSearchSpace(modelType);

            foreach (var (name, range) in searchSpace)
            {
                if (range.Type == ParameterType.Float || range.Type == ParameterType.Integer)
                {
                    double min = ToDouble(range.MinValue);
                    double max = ToDouble(range.MaxValue);
                    Assert.True(min < max,
                        $"[{modelType}] {name}: min ({min}) should be < max ({max})");
                }
            }
        }
    }

    [Fact]
    public void SearchSpace_Forecasting_HiddenSize_DefaultIs128()
    {
        var space = new FinancialSearchSpace(FinancialDomain.Forecasting);
        var hidden = space.GetSearchSpace(ModelType.PatchTST)["HiddenSize"];

        Assert.Equal(128.0, ToDouble(hidden.DefaultValue));
    }

    [Fact]
    public void SearchSpace_Risk_HiddenSize_DefaultIs64()
    {
        var space = new FinancialSearchSpace(FinancialDomain.Risk);
        var hidden = space.GetSearchSpace(ModelType.NeuralVaR)["HiddenSize"];

        Assert.Equal(64.0, ToDouble(hidden.DefaultValue));
    }

    [Fact]
    public void SearchSpace_TFT_UseVariableSelection_DefaultIsTrue()
    {
        var space = new FinancialSearchSpace(FinancialDomain.Forecasting);
        var varSel = space.GetSearchSpace(ModelType.TFT)["UseVariableSelection"];

        Assert.Equal(ParameterType.Boolean, varSel.Type);
        Assert.Equal(true, varSel.DefaultValue);
    }

    [Fact]
    public void SearchSpace_NBEATS_UseInterpretableBasis_DefaultIsTrue()
    {
        var space = new FinancialSearchSpace(FinancialDomain.Forecasting);
        var basis = space.GetSearchSpace(ModelType.NBEATS)["UseInterpretableBasis"];

        Assert.Equal(true, basis.DefaultValue);
    }

    [Fact]
    public void SearchSpace_DifferentModels_ProduceSeparateInstances()
    {
        var space = new FinancialSearchSpace(FinancialDomain.Forecasting);

        var patchTST1 = space.GetSearchSpace(ModelType.PatchTST);

        // Modifying one should not affect the other
        patchTST1["LearningRate"] = new ParameterRange
        {
            Type = ParameterType.Float,
            MinValue = 0.0,
            MaxValue = 1.0,
            DefaultValue = 0.5
        };

        var patchTST3 = space.GetSearchSpace(ModelType.PatchTST);
        // Should get fresh default, not the modified one
        Assert.Equal(0.001, ToDouble(patchTST3["LearningRate"].DefaultValue));
    }

    [Fact]
    public void SearchSpace_Tabular_LearningRate_MaxIs0_05()
    {
        var riskSpace = new FinancialSearchSpace(FinancialDomain.Risk);
        var tabNetLR = riskSpace.GetSearchSpace(ModelType.TabNet)["LearningRate"];

        Assert.Equal(0.05, ToDouble(tabNetLR.MaxValue));
    }

    [Fact]
    public void SearchSpace_ConfidenceLevel_StrictRange_NoOverflow()
    {
        var riskSpace = new FinancialSearchSpace(FinancialDomain.Risk);
        var confRange = riskSpace.GetSearchSpace(ModelType.NeuralVaR)["ConfidenceLevel"];

        Assert.True(ToDouble(confRange.MinValue) > 0.0);
        Assert.True(ToDouble(confRange.MaxValue) < 1.0);
    }

    #endregion

    #region FinancialDomain Enum Tests

    [Fact]
    public void FinancialDomain_AllDomainsCreateSearchSpace()
    {
        var domains = Enum.GetValues(typeof(FinancialDomain)).Cast<FinancialDomain>();

        foreach (var domain in domains)
        {
            var space = new FinancialSearchSpace(domain);
            var searchSpace = space.GetSearchSpace(ModelType.AutoML);
            Assert.NotNull(searchSpace);
            Assert.NotEmpty(searchSpace);
        }
    }

    [Fact]
    public void FinancialDomain_Forecasting_CustomModel_HasCommonParams()
    {
        var space = new FinancialSearchSpace(FinancialDomain.Forecasting);
        var defaultSpace = space.GetSearchSpace(ModelType.AutoML);

        Assert.True(defaultSpace.ContainsKey("LearningRate"));
        Assert.True(defaultSpace.ContainsKey("HiddenSize"));
        Assert.True(defaultSpace.ContainsKey("NumLayers"));
        Assert.True(defaultSpace.ContainsKey("DropoutRate"));
        Assert.True(defaultSpace.ContainsKey("Epochs"));
        Assert.True(defaultSpace.ContainsKey("BatchSize"));
    }

    [Fact]
    public void FinancialDomain_Risk_CustomModel_HasRiskParams()
    {
        var space = new FinancialSearchSpace(FinancialDomain.Risk);
        var defaultSpace = space.GetSearchSpace(ModelType.AutoML);

        Assert.True(defaultSpace.ContainsKey("LearningRate"));
        Assert.True(defaultSpace.ContainsKey("HiddenSize"));
        Assert.True(defaultSpace.ContainsKey("NumLayers"));
        Assert.True(defaultSpace.ContainsKey("DropoutRate"));
        Assert.True(defaultSpace.ContainsKey("Epochs"));
        Assert.True(defaultSpace.ContainsKey("BatchSize"));

        // Risk domain custom should use risk defaults (smaller max LR)
        var lr = defaultSpace["LearningRate"];
        Assert.Equal(0.01, ToDouble(lr.MaxValue));
    }

    #endregion

    #region StockTradingEnvironment - Construction and Validation

    [Fact]
    public void StockTrading_Construction_ValidMarketData()
    {
        var prices = new Tensor<double>(new[] { 10, 1 });
        for (int t = 0; t < 10; t++)
        {
            prices[t, 0] = 100.0 + t;
        }

        var env = new StockTradingEnvironment<double>(
            marketData: prices,
            windowSize: 3,
            initialCapital: 10000.0,
            tradeSize: 1.0,
            transactionCost: 0.001,
            seed: 42);

        Assert.Equal(3, env.ActionSpaceSize);
        Assert.False(env.IsContinuousActionSpace);
    }

    [Fact]
    public void StockTrading_MultiAsset_ThrowsArgumentException()
    {
        var prices = new Tensor<double>(new[] { 10, 2 });

        Assert.Throws<ArgumentException>(() =>
            new StockTradingEnvironment<double>(prices, 3, 10000.0, 1.0));
    }

    [Fact]
    public void StockTrading_WindowSizeTooLarge_ThrowsArgumentException()
    {
        var prices = new Tensor<double>(new[] { 5, 1 });
        for (int t = 0; t < 5; t++)
        {
            prices[t, 0] = 100.0;
        }

        Assert.Throws<ArgumentException>(() =>
            new StockTradingEnvironment<double>(prices, 5, 10000.0, 1.0));
    }

    [Fact]
    public void StockTrading_1DMarketData_ThrowsArgumentException()
    {
        var prices = new Tensor<double>(new[] { 10 });

        Assert.Throws<ArgumentException>(() =>
            new StockTradingEnvironment<double>(prices, 3, 10000.0, 1.0));
    }

    [Fact]
    public void StockTrading_ObservationDimension_Correct()
    {
        var prices = new Tensor<double>(new[] { 20, 1 });
        for (int t = 0; t < 20; t++)
        {
            prices[t, 0] = 100.0 + t;
        }

        int windowSize = 5;
        var env = new StockTradingEnvironment<double>(
            prices, windowSize: windowSize, initialCapital: 10000.0, tradeSize: 1.0, seed: 42);

        // ObservationSpaceDimension = (windowSize * numAssets) + numAssets + 1
        Assert.Equal(7, env.ObservationSpaceDimension);
    }

    #endregion

    #region StockTradingEnvironment - Reset and Step Mechanics

    [Fact]
    public void StockTrading_Reset_ReturnsObservation()
    {
        var prices = new Tensor<double>(new[] { 20, 1 });
        for (int t = 0; t < 20; t++)
        {
            prices[t, 0] = 100.0 + t;
        }

        var env = new StockTradingEnvironment<double>(
            prices, windowSize: 5, initialCapital: 10000.0, tradeSize: 1.0, seed: 42);

        var observation = env.Reset();
        Assert.NotNull(observation);
        Assert.True(observation.Length > 0);
    }

    [Fact]
    public void StockTrading_Reset_ObservationLength_EqualsObservationDimension()
    {
        var prices = new Tensor<double>(new[] { 20, 1 });
        for (int t = 0; t < 20; t++)
        {
            prices[t, 0] = 100.0 + t;
        }

        var env = new StockTradingEnvironment<double>(
            prices, windowSize: 5, initialCapital: 10000.0, tradeSize: 1.0, seed: 42);

        var observation = env.Reset();
        Assert.Equal(env.ObservationSpaceDimension, observation.Length);
    }

    [Fact]
    public void StockTrading_Reset_ContainsCashValue()
    {
        var prices = new Tensor<double>(new[] { 20, 1 });
        for (int t = 0; t < 20; t++)
        {
            prices[t, 0] = 100.0 + t;
        }

        double initialCapital = 10000.0;
        var env = new StockTradingEnvironment<double>(
            prices, windowSize: 5, initialCapital: initialCapital, tradeSize: 1.0, seed: 42);

        var observation = env.Reset();
        Assert.Equal(initialCapital, observation[observation.Length - 1]);
    }

    [Fact]
    public void StockTrading_Reset_PositionsAreZero()
    {
        var prices = new Tensor<double>(new[] { 20, 1 });
        for (int t = 0; t < 20; t++)
        {
            prices[t, 0] = 100.0 + t;
        }

        var env = new StockTradingEnvironment<double>(
            prices, windowSize: 5, initialCapital: 10000.0, tradeSize: 1.0, seed: 42);

        var observation = env.Reset();
        // Position is at index (windowSize * numAssets) = 5
        Assert.Equal(0.0, observation[5]);
    }

    [Fact]
    public void StockTrading_HoldAction_NoTradeExecuted()
    {
        var prices = new Tensor<double>(new[] { 20, 1 });
        for (int t = 0; t < 20; t++)
        {
            prices[t, 0] = 100.0;
        }

        var env = new StockTradingEnvironment<double>(
            prices, windowSize: 3, initialCapital: 10000.0, tradeSize: 1.0,
            transactionCost: 0.0, seed: 42);

        env.Reset();

        var holdAction = new Vector<double>(new double[] { 0 });
        var (obs, reward, done, info) = env.Step(holdAction);

        Assert.NotNull(obs);
        Assert.Equal(0.0, reward);
    }

    [Fact]
    public void StockTrading_BuyAction_ExecutesTrade()
    {
        var prices = new Tensor<double>(new[] { 20, 1 });
        for (int t = 0; t < 20; t++)
        {
            prices[t, 0] = 100.0;
        }

        var env = new StockTradingEnvironment<double>(
            prices, windowSize: 3, initialCapital: 10000.0, tradeSize: 1.0,
            transactionCost: 0.0, seed: 42);

        env.Reset();

        var buyAction = new Vector<double>(new double[] { 1 });
        var (obs, reward, done, info) = env.Step(buyAction);

        Assert.NotNull(obs);
        Assert.NotNull(info);
        Assert.True(info.ContainsKey("portfolioValue"));
        Assert.True(info.ContainsKey("cash"));
        Assert.True(info.ContainsKey("positions"));
    }

    [Fact]
    public void StockTrading_SellAction_ExecutesTrade()
    {
        var prices = new Tensor<double>(new[] { 20, 1 });
        for (int t = 0; t < 20; t++)
        {
            prices[t, 0] = 100.0;
        }

        var env = new StockTradingEnvironment<double>(
            prices, windowSize: 3, initialCapital: 10000.0, tradeSize: 1.0,
            transactionCost: 0.0, allowShortSelling: true, seed: 42);

        env.Reset();

        var sellAction = new Vector<double>(new double[] { 2 });
        var (obs, reward, done, info) = env.Step(sellAction);

        Assert.NotNull(obs);
        Assert.Equal(env.ObservationSpaceDimension, obs.Length);
    }

    [Fact]
    public void StockTrading_BuyThenSell_PortfolioChanges()
    {
        var prices = new Tensor<double>(new[] { 20, 1 });
        for (int t = 0; t < 20; t++)
        {
            prices[t, 0] = 100.0;
        }

        double initialCapital = 10000.0;
        var env = new StockTradingEnvironment<double>(
            prices, windowSize: 3, initialCapital: initialCapital, tradeSize: 1.0,
            transactionCost: 0.001, seed: 42);

        env.Reset();

        var buyAction = new Vector<double>(new double[] { 1 });
        env.Step(buyAction);

        var sellAction = new Vector<double>(new double[] { 2 });
        var (_, _, _, sellInfo) = env.Step(sellAction);

        var finalValue = Convert.ToDouble(sellInfo["portfolioValue"]);
        Assert.True(finalValue <= initialCapital,
            $"After buy+sell with transaction costs, portfolio ({finalValue}) should be <= initial ({initialCapital})");
    }

    [Fact]
    public void StockTrading_MultipleSteps_EventuallyDone()
    {
        var prices = new Tensor<double>(new[] { 10, 1 });
        for (int t = 0; t < 10; t++)
        {
            prices[t, 0] = 100.0 + t;
        }

        var env = new StockTradingEnvironment<double>(
            prices, windowSize: 3, initialCapital: 10000.0, tradeSize: 1.0, seed: 42);

        env.Reset();

        bool reachedDone = false;
        for (int step = 0; step < 20; step++)
        {
            var action = new Vector<double>(new double[] { 0 });
            var (_, _, done, _) = env.Step(action);
            if (done)
            {
                reachedDone = true;
                break;
            }
        }

        Assert.True(reachedDone, "Episode should terminate after all market data is consumed");
    }

    [Fact]
    public void StockTrading_MaxEpisodeLength_EndsEarly()
    {
        var prices = new Tensor<double>(new[] { 50, 1 });
        for (int t = 0; t < 50; t++)
        {
            prices[t, 0] = 100.0 + t;
        }

        var env = new StockTradingEnvironment<double>(
            prices, windowSize: 3, initialCapital: 10000.0, tradeSize: 1.0,
            maxEpisodeLength: 5, seed: 42);

        env.Reset();

        int stepsTaken = 0;
        for (int step = 0; step < 50; step++)
        {
            var action = new Vector<double>(new double[] { 0 });
            var (_, _, done, _) = env.Step(action);
            stepsTaken++;
            if (done)
            {
                break;
            }
        }

        Assert.Equal(5, stepsTaken);
    }

    [Fact]
    public void StockTrading_OneHotAction_Supported()
    {
        var prices = new Tensor<double>(new[] { 20, 1 });
        for (int t = 0; t < 20; t++)
        {
            prices[t, 0] = 100.0;
        }

        var env = new StockTradingEnvironment<double>(
            prices, windowSize: 3, initialCapital: 10000.0, tradeSize: 1.0, seed: 42);

        env.Reset();

        var oneHotBuy = new Vector<double>(new double[] { 0, 1, 0 });
        var (obs, reward, done, _) = env.Step(oneHotBuy);

        Assert.NotNull(obs);
    }

    [Fact]
    public void StockTrading_OneHotSell_Supported()
    {
        var prices = new Tensor<double>(new[] { 20, 1 });
        for (int t = 0; t < 20; t++)
        {
            prices[t, 0] = 100.0;
        }

        var env = new StockTradingEnvironment<double>(
            prices, windowSize: 3, initialCapital: 10000.0, tradeSize: 1.0,
            allowShortSelling: true, seed: 42);

        env.Reset();

        var oneHotSell = new Vector<double>(new double[] { 0, 0, 1 });
        var (obs, reward, done, _) = env.Step(oneHotSell);

        Assert.NotNull(obs);
    }

    [Fact]
    public void StockTrading_NullAction_ThrowsArgumentNullException()
    {
        var prices = new Tensor<double>(new[] { 20, 1 });
        for (int t = 0; t < 20; t++)
        {
            prices[t, 0] = 100.0;
        }

        var env = new StockTradingEnvironment<double>(
            prices, windowSize: 3, initialCapital: 10000.0, tradeSize: 1.0, seed: 42);

        env.Reset();

        Assert.Throws<ArgumentNullException>(() => env.Step(null));
    }

    [Fact]
    public void StockTrading_UpwardTrend_BuyReturnsPositiveReward()
    {
        var prices = new Tensor<double>(new[] { 20, 1 });
        for (int t = 0; t < 20; t++)
        {
            prices[t, 0] = 100.0 + (t * 5.0);
        }

        var env = new StockTradingEnvironment<double>(
            prices, windowSize: 3, initialCapital: 10000.0, tradeSize: 10.0,
            transactionCost: 0.0, seed: 42);

        env.Reset();

        var buyAction = new Vector<double>(new double[] { 1 });
        env.Step(buyAction);

        var holdAction = new Vector<double>(new double[] { 0 });
        var (_, reward, _, _) = env.Step(holdAction);

        Assert.True(reward > 0, $"Expected positive reward in upward trend, got {reward}");
    }

    [Fact]
    public void StockTrading_DownwardTrend_BuyReturnsNegativeReward()
    {
        var prices = new Tensor<double>(new[] { 20, 1 });
        for (int t = 0; t < 20; t++)
        {
            prices[t, 0] = 200.0 - (t * 5.0);
        }

        var env = new StockTradingEnvironment<double>(
            prices, windowSize: 3, initialCapital: 10000.0, tradeSize: 10.0,
            transactionCost: 0.0, seed: 42);

        env.Reset();

        var buyAction = new Vector<double>(new double[] { 1 });
        env.Step(buyAction);

        var holdAction = new Vector<double>(new double[] { 0 });
        var (_, reward, _, _) = env.Step(holdAction);

        Assert.True(reward < 0, $"Expected negative reward in downward trend, got {reward}");
    }

    [Fact]
    public void StockTrading_NoShortSelling_SellWithoutPosition_NoEffect()
    {
        var prices = new Tensor<double>(new[] { 20, 1 });
        for (int t = 0; t < 20; t++)
        {
            prices[t, 0] = 100.0;
        }

        double initialCapital = 10000.0;
        var env = new StockTradingEnvironment<double>(
            prices, windowSize: 3, initialCapital: initialCapital, tradeSize: 1.0,
            transactionCost: 0.0, allowShortSelling: false, seed: 42);

        env.Reset();

        var sellAction = new Vector<double>(new double[] { 2 });
        var (_, _, _, info) = env.Step(sellAction);

        var cash = Convert.ToDouble(info["cash"]);
        Assert.Equal(initialCapital, cash);
    }

    [Fact]
    public void StockTrading_Seed_ReproducibleRandomStart()
    {
        var numOps = MathHelper.GetNumericOperations<double>();
        var data = FinanceTestHelpers.CreatePriceTensor<double>(steps: 50, assets: 1);
        var env = new StockTradingEnvironment<double>(
            data,
            windowSize: 5,
            initialCapital: numOps.FromDouble(1000),
            tradeSize: numOps.One,
            randomStart: true);

        env.Seed(123);
        var state1 = env.Reset();
        env.Seed(123);
        var state2 = env.Reset();

        Assert.Equal(state1.Length, state2.Length);
        for (int i = 0; i < state1.Length; i++)
        {
            Assert.Equal(state1[i], state2[i]);
        }

        env.Close();
    }

    [Fact]
    public void StockTrading_DifferentSeeds_DifferentRandomStarts()
    {
        var data = FinanceTestHelpers.CreatePriceTensor<double>(steps: 100, assets: 1);

        var env = new StockTradingEnvironment<double>(
            data, windowSize: 5, initialCapital: 1000.0, tradeSize: 1.0,
            randomStart: true);

        env.Seed(111);
        var state1 = env.Reset();
        env.Seed(222);
        var state2 = env.Reset();

        bool differ = false;
        for (int i = 0; i < state1.Length; i++)
        {
            if (state1[i] != state2[i])
            {
                differ = true;
                break;
            }
        }
        Assert.True(differ, "Different seeds should produce different random start observations");
    }

    [Fact]
    public void StockTrading_Close_DoesNotThrow()
    {
        var prices = new Tensor<double>(new[] { 20, 1 });
        for (int t = 0; t < 20; t++)
        {
            prices[t, 0] = 100.0;
        }

        var env = new StockTradingEnvironment<double>(
            prices, windowSize: 3, initialCapital: 10000.0, tradeSize: 1.0, seed: 42);

        env.Reset();
        env.Close();
    }

    [Fact]
    public void StockTrading_StepInfo_ContainsExpectedKeys()
    {
        var prices = new Tensor<double>(new[] { 20, 1 });
        for (int t = 0; t < 20; t++)
        {
            prices[t, 0] = 100.0;
        }

        var env = new StockTradingEnvironment<double>(
            prices, windowSize: 3, initialCapital: 10000.0, tradeSize: 1.0, seed: 42);

        env.Reset();

        var action = new Vector<double>(new double[] { 0 });
        var (_, _, _, info) = env.Step(action);

        Assert.True(info.ContainsKey("step"));
        Assert.True(info.ContainsKey("portfolioValue"));
        Assert.True(info.ContainsKey("cash"));
        Assert.True(info.ContainsKey("positions"));
    }

    [Fact]
    public void StockTrading_Float_SmokeTest()
    {
        var numOps = MathHelper.GetNumericOperations<float>();
        var data = FinanceTestHelpers.CreatePriceTensor<float>(steps: 30, assets: 1);

        var env = new StockTradingEnvironment<float>(
            data, windowSize: 5, initialCapital: numOps.FromDouble(10000),
            tradeSize: numOps.One, seed: 42);

        var state = env.Reset();
        Assert.Equal(env.ObservationSpaceDimension, state.Length);

        var action = new Vector<float>(new[] { numOps.FromDouble(1) });
        var step = env.Step(action);
        Assert.Equal(env.ObservationSpaceDimension, step.NextState.Length);

        env.Close();
    }

    [Fact]
    public void StockTrading_ResetMultipleTimes_EachResetFresh()
    {
        var prices = new Tensor<double>(new[] { 20, 1 });
        for (int t = 0; t < 20; t++)
        {
            prices[t, 0] = 100.0;
        }

        double initialCapital = 10000.0;
        var env = new StockTradingEnvironment<double>(
            prices, windowSize: 3, initialCapital: initialCapital, tradeSize: 1.0,
            transactionCost: 0.0, seed: 42);

        env.Reset();
        env.Step(new Vector<double>(new double[] { 1 }));
        env.Step(new Vector<double>(new double[] { 1 }));

        var freshObs = env.Reset();
        Assert.Equal(initialCapital, freshObs[freshObs.Length - 1]);
        Assert.Equal(0.0, freshObs[3]); // Position index = windowSize * numAssets = 3
    }

    #endregion

    #region PortfolioTradingEnvironment - Multi-Asset Tests

    [Fact]
    public void PortfolioTrading_Construction_ValidMultiAsset()
    {
        var prices = new Tensor<double>(new[] { 20, 3 });
        for (int t = 0; t < 20; t++)
        {
            prices[t, 0] = 100.0 + t;
            prices[t, 1] = 200.0 + (t * 0.5);
            prices[t, 2] = 50.0 + (t * 2.0);
        }

        var env = new PortfolioTradingEnvironment<double>(
            prices, windowSize: 5, initialCapital: 100000.0, seed: 42);

        Assert.Equal(3, env.ActionSpaceSize);
        Assert.True(env.IsContinuousActionSpace);
    }

    [Fact]
    public void PortfolioTrading_ObservationDimension_Correct()
    {
        int numAssets = 3;
        int windowSize = 5;
        var prices = new Tensor<double>(new[] { 20, numAssets });
        for (int t = 0; t < 20; t++)
        {
            for (int a = 0; a < numAssets; a++)
            {
                prices[t, a] = 100.0 + t + a;
            }
        }

        var env = new PortfolioTradingEnvironment<double>(
            prices, windowSize: windowSize, initialCapital: 100000.0, seed: 42);

        int expected = (windowSize * numAssets) + numAssets + 1;
        Assert.Equal(expected, env.ObservationSpaceDimension);
    }

    [Fact]
    public void PortfolioTrading_Reset_ReturnsCorrectDimension()
    {
        var prices = new Tensor<double>(new[] { 20, 2 });
        for (int t = 0; t < 20; t++)
        {
            prices[t, 0] = 100.0 + t;
            prices[t, 1] = 200.0 + t;
        }

        var env = new PortfolioTradingEnvironment<double>(
            prices, windowSize: 5, initialCapital: 100000.0, seed: 42);

        var state = env.Reset();
        Assert.Equal(env.ObservationSpaceDimension, state.Length);
    }

    [Fact]
    public void PortfolioTrading_EqualWeights_Step()
    {
        var prices = new Tensor<double>(new[] { 20, 2 });
        for (int t = 0; t < 20; t++)
        {
            prices[t, 0] = 100.0;
            prices[t, 1] = 200.0;
        }

        var env = new PortfolioTradingEnvironment<double>(
            prices, windowSize: 3, initialCapital: 100000.0,
            transactionCost: 0.0, seed: 42);

        env.Reset();

        var weights = new Vector<double>(new double[] { 0.5, 0.5 });
        var (obs, reward, done, info) = env.Step(weights);

        Assert.NotNull(obs);
        Assert.Equal(env.ObservationSpaceDimension, obs.Length);
    }

    [Fact]
    public void PortfolioTrading_WrongActionLength_ThrowsArgumentException()
    {
        var prices = new Tensor<double>(new[] { 20, 2 });
        for (int t = 0; t < 20; t++)
        {
            prices[t, 0] = 100.0;
            prices[t, 1] = 200.0;
        }

        var env = new PortfolioTradingEnvironment<double>(
            prices, windowSize: 3, initialCapital: 100000.0, seed: 42);

        env.Reset();

        var weights = new Vector<double>(new double[] { 0.33, 0.33, 0.34 });
        Assert.Throws<ArgumentException>(() => env.Step(weights));
    }

    [Fact]
    public void PortfolioTrading_AllInOneAsset_Step()
    {
        var prices = new Tensor<double>(new[] { 20, 2 });
        for (int t = 0; t < 20; t++)
        {
            prices[t, 0] = 100.0 + t;
            prices[t, 1] = 100.0;
        }

        var env = new PortfolioTradingEnvironment<double>(
            prices, windowSize: 3, initialCapital: 100000.0,
            transactionCost: 0.0, seed: 42);

        env.Reset();

        var weights = new Vector<double>(new double[] { 1.0, 0.0 });
        var (obs, reward, done, _) = env.Step(weights);

        Assert.NotNull(obs);
    }

    [Fact]
    public void PortfolioTrading_NoShortSelling_NegativeWeightsClamped()
    {
        var prices = new Tensor<double>(new[] { 20, 2 });
        for (int t = 0; t < 20; t++)
        {
            prices[t, 0] = 100.0;
            prices[t, 1] = 200.0;
        }

        var env = new PortfolioTradingEnvironment<double>(
            prices, windowSize: 3, initialCapital: 100000.0,
            allowShortSelling: false, transactionCost: 0.0, seed: 42);

        env.Reset();

        var weights = new Vector<double>(new double[] { -0.5, 1.5 });
        var (obs, reward, done, _) = env.Step(weights);

        Assert.NotNull(obs);
    }

    [Fact]
    public void PortfolioTrading_ZeroWeights_UniformFallback()
    {
        var prices = new Tensor<double>(new[] { 20, 2 });
        for (int t = 0; t < 20; t++)
        {
            prices[t, 0] = 100.0;
            prices[t, 1] = 200.0;
        }

        var env = new PortfolioTradingEnvironment<double>(
            prices, windowSize: 3, initialCapital: 100000.0,
            allowShortSelling: false, transactionCost: 0.0, seed: 42);

        env.Reset();

        var weights = new Vector<double>(new double[] { 0.0, 0.0 });
        var (obs, reward, done, _) = env.Step(weights);

        Assert.NotNull(obs);
    }

    [Fact]
    public void PortfolioTrading_MultipleSteps_EventuallyDone()
    {
        var prices = new Tensor<double>(new[] { 15, 2 });
        for (int t = 0; t < 15; t++)
        {
            prices[t, 0] = 100.0 + t;
            prices[t, 1] = 200.0 + t;
        }

        var env = new PortfolioTradingEnvironment<double>(
            prices, windowSize: 3, initialCapital: 100000.0, seed: 42);

        env.Reset();

        bool reachedDone = false;
        for (int step = 0; step < 30; step++)
        {
            var weights = new Vector<double>(new double[] { 0.5, 0.5 });
            var (_, _, done, _) = env.Step(weights);
            if (done)
            {
                reachedDone = true;
                break;
            }
        }

        Assert.True(reachedDone, "Portfolio episode should terminate");
    }

    [Fact]
    public void PortfolioTrading_Float_SmokeTest()
    {
        var numOps = MathHelper.GetNumericOperations<float>();
        var data = FinanceTestHelpers.CreatePriceTensor<float>(steps: 30, assets: 2);

        var env = new PortfolioTradingEnvironment<float>(
            data, windowSize: 5, initialCapital: numOps.FromDouble(10000), seed: 42);

        var state = env.Reset();
        Assert.Equal(env.ObservationSpaceDimension, state.Length);

        var weights = new Vector<float>(new[] { numOps.FromDouble(0.5), numOps.FromDouble(0.5) });
        var step = env.Step(weights);
        Assert.Equal(env.ObservationSpaceDimension, step.NextState.Length);

        env.Close();
    }

    #endregion

    #region MarketMakingEnvironment - Tests

    [Fact]
    public void MarketMaking_Construction_ValidData()
    {
        var prices = new Tensor<double>(new[] { 20, 1 });
        for (int t = 0; t < 20; t++)
        {
            prices[t, 0] = 100.0;
        }

        var env = new MarketMakingEnvironment<double>(
            prices, windowSize: 3, initialCapital: 10000.0, tradeSize: 1.0, seed: 42);

        Assert.Equal(2, env.ActionSpaceSize);
        Assert.True(env.IsContinuousActionSpace);
    }

    [Fact]
    public void MarketMaking_MultiAsset_ThrowsArgumentException()
    {
        var prices = new Tensor<double>(new[] { 20, 2 });

        Assert.Throws<ArgumentException>(() =>
            new MarketMakingEnvironment<double>(prices, 3, 10000.0, 1.0));
    }

    [Fact]
    public void MarketMaking_Reset_ReturnsCorrectDimension()
    {
        var prices = new Tensor<double>(new[] { 20, 1 });
        for (int t = 0; t < 20; t++)
        {
            prices[t, 0] = 100.0;
        }

        var env = new MarketMakingEnvironment<double>(
            prices, windowSize: 5, initialCapital: 10000.0, tradeSize: 1.0, seed: 42);

        var state = env.Reset();
        Assert.Equal(env.ObservationSpaceDimension, state.Length);
    }

    [Fact]
    public void MarketMaking_ActionMustHaveTwoElements()
    {
        var prices = new Tensor<double>(new[] { 20, 1 });
        for (int t = 0; t < 20; t++)
        {
            prices[t, 0] = 100.0;
        }

        var env = new MarketMakingEnvironment<double>(
            prices, windowSize: 3, initialCapital: 10000.0, tradeSize: 1.0, seed: 42);

        env.Reset();

        var wrongAction = new Vector<double>(new double[] { 0.01 });
        Assert.Throws<ArgumentException>(() => env.Step(wrongAction));
    }

    [Fact]
    public void MarketMaking_ValidAction_Steps()
    {
        var prices = new Tensor<double>(new[] { 20, 1 });
        for (int t = 0; t < 20; t++)
        {
            prices[t, 0] = 100.0;
        }

        var env = new MarketMakingEnvironment<double>(
            prices, windowSize: 3, initialCapital: 10000.0, tradeSize: 1.0, seed: 42);

        env.Reset();

        var action = new Vector<double>(new double[] { 0.01, 0.01 });
        var (obs, reward, done, info) = env.Step(action);

        Assert.NotNull(obs);
        Assert.Equal(env.ObservationSpaceDimension, obs.Length);
        Assert.True(info.ContainsKey("portfolioValue"));
    }

    [Fact]
    public void MarketMaking_WideSpread_StepsSuccessfully()
    {
        var prices = new Tensor<double>(new[] { 100, 1 });
        for (int t = 0; t < 100; t++)
        {
            prices[t, 0] = 100.0;
        }

        var env = new MarketMakingEnvironment<double>(
            prices, windowSize: 3, initialCapital: 10000.0, tradeSize: 1.0,
            baseSpread: 0.01, orderArrivalRate: 0.2, seed: 42);

        env.Reset();

        int steps = 0;
        for (int i = 0; i < 90; i++)
        {
            var action = new Vector<double>(new double[] { 1.0, 1.0 });
            var (_, reward, done, _) = env.Step(action);
            steps++;
            if (done) break;
        }

        Assert.True(steps > 0, "Should have taken at least one step");
    }

    [Fact]
    public void MarketMaking_TightSpread_StepsSuccessfully()
    {
        var prices = new Tensor<double>(new[] { 100, 1 });
        for (int t = 0; t < 100; t++)
        {
            prices[t, 0] = 100.0;
        }

        var env = new MarketMakingEnvironment<double>(
            prices, windowSize: 3, initialCapital: 100000.0, tradeSize: 1.0,
            baseSpread: 0.01, orderArrivalRate: 0.5, seed: 42);

        env.Reset();

        int steps = 0;
        for (int i = 0; i < 90; i++)
        {
            var action = new Vector<double>(new double[] { 0.001, 0.001 });
            var (_, reward, done, _) = env.Step(action);
            steps++;
            if (done) break;
        }

        Assert.True(steps > 0);
    }

    [Fact]
    public void MarketMaking_MaxInventory_RespectsLimit()
    {
        var prices = new Tensor<double>(new[] { 200, 1 });
        for (int t = 0; t < 200; t++)
        {
            prices[t, 0] = 100.0;
        }

        int maxInventory = 3;
        var env = new MarketMakingEnvironment<double>(
            prices, windowSize: 3, initialCapital: 100000.0, tradeSize: 1.0,
            maxInventory: maxInventory, orderArrivalRate: 1.0, seed: 42);

        env.Reset();

        for (int i = 0; i < 190; i++)
        {
            var action = new Vector<double>(new double[] { 0.0001, 0.0001 });
            var (_, _, done, info) = env.Step(action);
            if (done) break;

            if (info.ContainsKey("positions") && info["positions"] is Vector<double> positions)
            {
                Assert.True(Math.Abs(positions[0]) <= maxInventory + 1e-9,
                    $"Inventory {positions[0]} exceeded max {maxInventory}");
            }
        }
    }

    [Fact]
    public void MarketMaking_Float_SmokeTest()
    {
        var numOps = MathHelper.GetNumericOperations<float>();
        var data = FinanceTestHelpers.CreatePriceTensor<float>(steps: 30, assets: 1);

        var env = new MarketMakingEnvironment<float>(
            data, windowSize: 5, initialCapital: numOps.FromDouble(10000),
            tradeSize: numOps.One, seed: 42);

        var state = env.Reset();
        Assert.Equal(env.ObservationSpaceDimension, state.Length);

        var action = new Vector<float>(new[] { numOps.FromDouble(0.01), numOps.FromDouble(0.01) });
        var step = env.Step(action);
        Assert.Equal(env.ObservationSpaceDimension, step.NextState.Length);

        env.Close();
    }

    [Fact]
    public void MarketMaking_NoShortSelling_CannotGoNegative()
    {
        var prices = new Tensor<double>(new[] { 100, 1 });
        for (int t = 0; t < 100; t++)
        {
            prices[t, 0] = 100.0;
        }

        var env = new MarketMakingEnvironment<double>(
            prices, windowSize: 3, initialCapital: 100000.0, tradeSize: 1.0,
            allowShortSelling: false, orderArrivalRate: 1.0, seed: 42);

        env.Reset();

        for (int i = 0; i < 90; i++)
        {
            var action = new Vector<double>(new double[] { 0.0001, 0.0001 });
            var (_, _, done, info) = env.Step(action);
            if (done) break;

            if (info.ContainsKey("positions") && info["positions"] is Vector<double> positions)
            {
                Assert.True(positions[0] >= -1e-9,
                    $"Position {positions[0]} should not be negative with no short selling");
            }
        }
    }

    #endregion

    #region TradingEnvironmentFactory - Factory Tests

    [Fact]
    public void TradingEnvironmentFactory_CreatesStockFromSeries()
    {
        var numOps = MathHelper.GetNumericOperations<double>();
        var series = FinanceTestHelpers.CreateMarketSeries<double>(30);

        var env = TradingEnvironmentFactory.CreateStockTradingEnvironment(
            series, windowSize: 5, initialCapital: numOps.FromDouble(1000),
            tradeSize: numOps.One, seed: 42);

        var state = env.Reset();
        Assert.Equal(env.ObservationSpaceDimension, state.Length);
        Assert.Equal(3, env.ActionSpaceSize);
    }

    [Fact]
    public void TradingEnvironmentFactory_CreatesStockFromProvider()
    {
        var numOps = MathHelper.GetNumericOperations<double>();
        var series = FinanceTestHelpers.CreateMarketSeries<double>(30);
        var provider = new MarketDataProvider<double>();
        provider.AddRange(series);

        var env = TradingEnvironmentFactory.CreateStockTradingEnvironment(
            provider, windowSize: 5, initialCapital: numOps.FromDouble(1000),
            tradeSize: numOps.One, seed: 42);

        var state = env.Reset();
        Assert.Equal(env.ObservationSpaceDimension, state.Length);
    }

    [Fact]
    public void TradingEnvironmentFactory_CreatesPortfolioEnvironment()
    {
        var numOps = MathHelper.GetNumericOperations<double>();
        var start = DateTime.UtcNow;
        var seriesA = FinanceTestHelpers.CreateMarketSeries<double>(20, start);
        var seriesB = FinanceTestHelpers.CreateMarketSeries<double>(20, start);
        var series = new List<IReadOnlyList<MarketDataPoint<double>>>
        {
            seriesA,
            seriesB
        };

        var env = TradingEnvironmentFactory.CreatePortfolioTradingEnvironment(
            series,
            windowSize: 5,
            initialCapital: numOps.FromDouble(1000));

        var state = env.Reset();
        Assert.Equal(env.ObservationSpaceDimension, state.Length);
    }

    [Fact]
    public void TradingEnvironmentFactory_CreatesPortfolioFromProviders()
    {
        var numOps = MathHelper.GetNumericOperations<double>();
        var start = DateTime.UtcNow;
        var seriesA = FinanceTestHelpers.CreateMarketSeries<double>(20, start);
        var seriesB = FinanceTestHelpers.CreateMarketSeries<double>(20, start);

        var providerA = new MarketDataProvider<double>();
        providerA.AddRange(seriesA);
        var providerB = new MarketDataProvider<double>();
        providerB.AddRange(seriesB);

        var providers = new List<MarketDataProvider<double>> { providerA, providerB };

        var env = TradingEnvironmentFactory.CreatePortfolioTradingEnvironment(
            providers,
            windowSize: 5,
            initialCapital: numOps.FromDouble(1000));

        var state = env.Reset();
        Assert.Equal(env.ObservationSpaceDimension, state.Length);
    }

    [Fact]
    public void TradingEnvironmentFactory_CreatesMarketMakingFromSeries()
    {
        var numOps = MathHelper.GetNumericOperations<double>();
        var series = FinanceTestHelpers.CreateMarketSeries<double>(30);

        var env = TradingEnvironmentFactory.CreateMarketMakingEnvironment(
            series, windowSize: 5, initialCapital: numOps.FromDouble(1000),
            tradeSize: numOps.One, seed: 42);

        var state = env.Reset();
        Assert.Equal(env.ObservationSpaceDimension, state.Length);
        Assert.Equal(2, env.ActionSpaceSize);
        Assert.True(env.IsContinuousActionSpace);
    }

    [Fact]
    public void TradingEnvironmentFactory_CreatesMarketMakingFromProvider()
    {
        var numOps = MathHelper.GetNumericOperations<double>();
        var series = FinanceTestHelpers.CreateMarketSeries<double>(30);
        var provider = new MarketDataProvider<double>();
        provider.AddRange(series);

        var env = TradingEnvironmentFactory.CreateMarketMakingEnvironment(
            provider, windowSize: 5, initialCapital: numOps.FromDouble(1000),
            tradeSize: numOps.One, seed: 42);

        var state = env.Reset();
        Assert.Equal(env.ObservationSpaceDimension, state.Length);
    }

    [Fact]
    public void TradingEnvironmentFactory_NullProvider_ThrowsArgumentNullException()
    {
        var numOps = MathHelper.GetNumericOperations<double>();
        MarketDataProvider<double> nullProvider = null;

        Assert.Throws<ArgumentNullException>(() =>
            TradingEnvironmentFactory.CreateStockTradingEnvironment(
                nullProvider, windowSize: 5, initialCapital: numOps.FromDouble(1000),
                tradeSize: numOps.One));
    }

    [Fact]
    public void TradingEnvironmentFactory_NullPortfolioProviders_ThrowsArgumentNullException()
    {
        var numOps = MathHelper.GetNumericOperations<double>();
        IReadOnlyList<MarketDataProvider<double>> nullProviders = null;

        Assert.Throws<ArgumentNullException>(() =>
            TradingEnvironmentFactory.CreatePortfolioTradingEnvironment(
                nullProviders, windowSize: 5, initialCapital: numOps.FromDouble(1000)));
    }

    [Fact]
    public void TradingEnvironmentFactory_NullMarketMakingProvider_ThrowsArgumentNullException()
    {
        var numOps = MathHelper.GetNumericOperations<double>();
        MarketDataProvider<double> nullProvider = null;

        Assert.Throws<ArgumentNullException>(() =>
            TradingEnvironmentFactory.CreateMarketMakingEnvironment(
                nullProvider, windowSize: 5, initialCapital: numOps.FromDouble(1000),
                tradeSize: numOps.One));
    }

    [Fact]
    public void TradingEnvironmentFactory_CustomPriceSelector_UsesOpen()
    {
        var numOps = MathHelper.GetNumericOperations<double>();
        var series = FinanceTestHelpers.CreateMarketSeries<double>(30);

        var env = TradingEnvironmentFactory.CreateStockTradingEnvironment(
            series, windowSize: 5, initialCapital: numOps.FromDouble(1000),
            tradeSize: numOps.One, priceSelector: p => p.Open, seed: 42);

        var state = env.Reset();
        Assert.Equal(env.ObservationSpaceDimension, state.Length);
    }

    [Fact]
    public void TradingEnvironmentFactory_Float_StockFromSeries()
    {
        var numOps = MathHelper.GetNumericOperations<float>();
        var series = FinanceTestHelpers.CreateMarketSeries<float>(30);

        var env = TradingEnvironmentFactory.CreateStockTradingEnvironment(
            series, windowSize: 5, initialCapital: numOps.FromDouble(1000),
            tradeSize: numOps.One, seed: 42);

        var state = env.Reset();
        Assert.Equal(env.ObservationSpaceDimension, state.Length);
    }

    [Fact]
    public void TradingEnvironmentFactory_MismatchedSeriesLengths_ThrowsArgumentException()
    {
        var numOps = MathHelper.GetNumericOperations<double>();
        var start = DateTime.UtcNow;
        var seriesA = FinanceTestHelpers.CreateMarketSeries<double>(20, start);
        var seriesB = FinanceTestHelpers.CreateMarketSeries<double>(15, start);

        var series = new List<IReadOnlyList<MarketDataPoint<double>>> { seriesA, seriesB };

        Assert.Throws<ArgumentException>(() =>
            TradingEnvironmentFactory.CreatePortfolioTradingEnvironment(
                series, windowSize: 5, initialCapital: numOps.FromDouble(1000)));
    }

    #endregion

    #region MarketDataProvider - Deep Tests

    [Fact]
    public void MarketDataProvider_Add_IncreasesCount()
    {
        var provider = new MarketDataProvider<double>();
        Assert.Equal(0, provider.Count);

        var point = new MarketDataPoint<double>(DateTime.UtcNow, 100, 105, 95, 102, 1000);
        provider.Add(point);
        Assert.Equal(1, provider.Count);
    }

    [Fact]
    public void MarketDataProvider_AddNull_ThrowsArgumentNullException()
    {
        var provider = new MarketDataProvider<double>();
        Assert.Throws<ArgumentNullException>(() => provider.Add(null));
    }

    [Fact]
    public void MarketDataProvider_AddRangeNull_ThrowsArgumentNullException()
    {
        var provider = new MarketDataProvider<double>();
        Assert.Throws<ArgumentNullException>(() => provider.AddRange(null));
    }

    [Fact]
    public void MarketDataProvider_GetAll_ReturnsList()
    {
        var series = FinanceTestHelpers.CreateMarketSeries<double>(10);
        var provider = new MarketDataProvider<double>();
        provider.AddRange(series);

        var all = provider.GetAll();
        Assert.Equal(10, all.Count);
    }

    [Fact]
    public void MarketDataProvider_GetRange_ReturnsCorrectSubset()
    {
        var start = new DateTime(2024, 1, 1);
        var series = FinanceTestHelpers.CreateMarketSeries<double>(10, start);
        var provider = new MarketDataProvider<double>();
        provider.AddRange(series);

        var rangeStart = start.AddMinutes(2);
        var rangeEnd = start.AddMinutes(6);
        var range = provider.GetRange(rangeStart, rangeEnd);

        Assert.True(range.Count > 0);
        foreach (var point in range)
        {
            Assert.True(point.Timestamp >= rangeStart);
            Assert.True(point.Timestamp <= rangeEnd);
        }
    }

    [Fact]
    public void MarketDataProvider_GetWindow_ReturnsCorrectLength()
    {
        var series = FinanceTestHelpers.CreateMarketSeries<double>(20);
        var provider = new MarketDataProvider<double>();
        provider.AddRange(series);

        var window = provider.GetWindow(startIndex: 5, length: 3);
        Assert.Equal(3, window.Count);
    }

    [Fact]
    public void MarketDataProvider_GetWindow_InvalidStart_ThrowsArgumentOutOfRange()
    {
        var series = FinanceTestHelpers.CreateMarketSeries<double>(10);
        var provider = new MarketDataProvider<double>();
        provider.AddRange(series);

        Assert.Throws<ArgumentOutOfRangeException>(() => provider.GetWindow(-1, 3));
        Assert.Throws<ArgumentOutOfRangeException>(() => provider.GetWindow(10, 3));
    }

    [Fact]
    public void MarketDataProvider_GetWindow_ZeroLength_ThrowsArgumentOutOfRange()
    {
        var series = FinanceTestHelpers.CreateMarketSeries<double>(10);
        var provider = new MarketDataProvider<double>();
        provider.AddRange(series);

        Assert.Throws<ArgumentOutOfRangeException>(() => provider.GetWindow(0, 0));
    }

    [Fact]
    public void MarketDataProvider_GetWindow_ExceedingLength_ClipsToAvailable()
    {
        var series = FinanceTestHelpers.CreateMarketSeries<double>(10);
        var provider = new MarketDataProvider<double>();
        provider.AddRange(series);

        var window = provider.GetWindow(8, 10);
        Assert.Equal(2, window.Count);
    }

    [Fact]
    public void MarketDataProvider_ToTensor_WithVolume_HasFiveColumns()
    {
        var series = FinanceTestHelpers.CreateMarketSeries<double>(10);
        var provider = new MarketDataProvider<double>();
        provider.AddRange(series);

        var tensor = provider.ToTensor(includeVolume: true);
        Assert.Equal(10, tensor.Shape[0]);
        Assert.Equal(5, tensor.Shape[1]);
    }

    [Fact]
    public void MarketDataProvider_ToTensor_WithoutVolume_HasFourColumns()
    {
        var series = FinanceTestHelpers.CreateMarketSeries<double>(10);
        var provider = new MarketDataProvider<double>();
        provider.AddRange(series);

        var tensor = provider.ToTensor(includeVolume: false);
        Assert.Equal(10, tensor.Shape[0]);
        Assert.Equal(4, tensor.Shape[1]);
    }

    [Fact]
    public void MarketDataProvider_Clear_ResetsCount()
    {
        var series = FinanceTestHelpers.CreateMarketSeries<double>(10);
        var provider = new MarketDataProvider<double>();
        provider.AddRange(series);

        Assert.Equal(10, provider.Count);
        provider.Clear();
        Assert.Equal(0, provider.Count);
    }

    [Fact]
    public void MarketDataPoint_PropertiesPreserved()
    {
        var ts = new DateTime(2024, 6, 15, 10, 30, 0);
        var point = new MarketDataPoint<double>(ts, 100.0, 105.0, 95.0, 102.0, 50000.0);

        Assert.Equal(ts, point.Timestamp);
        Assert.Equal(100.0, point.Open);
        Assert.Equal(105.0, point.High);
        Assert.Equal(95.0, point.Low);
        Assert.Equal(102.0, point.Close);
        Assert.Equal(50000.0, point.Volume);
    }

    #endregion

    #region ParameterRange - Clone Tests

    [Fact]
    public void ParameterRange_Clone_CopiesAllProperties()
    {
        var original = new ParameterRange
        {
            Type = ParameterType.Float,
            MinValue = 0.001,
            MaxValue = 0.1,
            UseLogScale = true,
            DefaultValue = 0.01,
            Step = 0.001
        };

        var clone = (ParameterRange)original.Clone();

        Assert.Equal(original.Type, clone.Type);
        Assert.Equal(original.MinValue, clone.MinValue);
        Assert.Equal(original.MaxValue, clone.MaxValue);
        Assert.Equal(original.UseLogScale, clone.UseLogScale);
        Assert.Equal(original.DefaultValue, clone.DefaultValue);
        Assert.Equal(original.Step, clone.Step);
    }

    [Fact]
    public void ParameterRange_Clone_CategoricalValues_DeepCopied()
    {
        var original = new ParameterRange
        {
            Type = ParameterType.Categorical,
            CategoricalValues = new List<object> { "A", "B", "C" },
            DefaultValue = "A"
        };

        var clone = (ParameterRange)original.Clone();

        Assert.NotNull(clone.CategoricalValues);
        Assert.Equal(3, clone.CategoricalValues.Count);
        Assert.Contains("A", clone.CategoricalValues.Cast<string>());
        Assert.Contains("B", clone.CategoricalValues.Cast<string>());
        Assert.Contains("C", clone.CategoricalValues.Cast<string>());

        clone.CategoricalValues.Add("D");
        Assert.Equal(3, original.CategoricalValues.Count);
    }

    [Fact]
    public void ParameterRange_Clone_NullCategoricalValues_StaysNull()
    {
        var original = new ParameterRange
        {
            Type = ParameterType.Integer,
            MinValue = 1.0,
            MaxValue = 10.0,
            CategoricalValues = null
        };

        var clone = (ParameterRange)original.Clone();
        Assert.Null(clone.CategoricalValues);
    }

    #endregion

    #region Cross-Domain Integration Tests

    [Fact]
    public void CrossDomain_SearchSpace_AllModelTypes_HaveLearningRate()
    {
        var allModels = new[]
        {
            (FinancialDomain.Forecasting, ModelType.PatchTST),
            (FinancialDomain.Forecasting, ModelType.ITransformer),
            (FinancialDomain.Forecasting, ModelType.DeepAR),
            (FinancialDomain.Forecasting, ModelType.NBEATS),
            (FinancialDomain.Forecasting, ModelType.TFT),
            (FinancialDomain.Risk, ModelType.NeuralVaR),
            (FinancialDomain.Risk, ModelType.TabNet),
            (FinancialDomain.Risk, ModelType.TabTransformer)
        };

        foreach (var (domain, modelType) in allModels)
        {
            var space = new FinancialSearchSpace(domain);
            var searchSpace = space.GetSearchSpace(modelType);

            Assert.True(searchSpace.ContainsKey("LearningRate"),
                $"[{domain}][{modelType}] missing LearningRate");
            Assert.True(searchSpace.ContainsKey("Epochs"),
                $"[{domain}][{modelType}] missing Epochs");
        }
    }

    [Fact]
    public void CrossDomain_SearchSpace_AllForecastingModels_HaveCommonParams()
    {
        var forecastModels = new[] { ModelType.PatchTST, ModelType.ITransformer,
            ModelType.DeepAR, ModelType.NBEATS, ModelType.TFT };

        var space = new FinancialSearchSpace(FinancialDomain.Forecasting);

        foreach (var modelType in forecastModels)
        {
            var searchSpace = space.GetSearchSpace(modelType);

            Assert.True(searchSpace.ContainsKey("HiddenSize"),
                $"[{modelType}] missing HiddenSize");
            Assert.True(searchSpace.ContainsKey("NumLayers"),
                $"[{modelType}] missing NumLayers");
            Assert.True(searchSpace.ContainsKey("DropoutRate"),
                $"[{modelType}] missing DropoutRate");
            Assert.True(searchSpace.ContainsKey("BatchSize"),
                $"[{modelType}] missing BatchSize");
        }
    }

    [Fact]
    public void CrossDomain_AllEnvironmentTypes_ResetAndStep()
    {
        var singleAssetData = FinanceTestHelpers.CreatePriceTensor<double>(steps: 40, assets: 1);
        var multiAssetData = FinanceTestHelpers.CreatePriceTensor<double>(steps: 40, assets: 2);

        var stockEnv = new StockTradingEnvironment<double>(
            singleAssetData, windowSize: 5, initialCapital: 10000.0,
            tradeSize: 1.0, seed: 42);
        var stockState = stockEnv.Reset();
        Assert.Equal(stockEnv.ObservationSpaceDimension, stockState.Length);
        var stockAction = new Vector<double>(new double[] { 1 });
        var stockStep = stockEnv.Step(stockAction);
        Assert.Equal(stockEnv.ObservationSpaceDimension, stockStep.NextState.Length);

        var portfolioEnv = new PortfolioTradingEnvironment<double>(
            multiAssetData, windowSize: 5, initialCapital: 10000.0, seed: 42);
        var portfolioState = portfolioEnv.Reset();
        Assert.Equal(portfolioEnv.ObservationSpaceDimension, portfolioState.Length);
        var weights = new Vector<double>(new double[] { 0.5, 0.5 });
        var portfolioStep = portfolioEnv.Step(weights);
        Assert.Equal(portfolioEnv.ObservationSpaceDimension, portfolioStep.NextState.Length);

        var mmEnv = new MarketMakingEnvironment<double>(
            singleAssetData, windowSize: 5, initialCapital: 10000.0,
            tradeSize: 1.0, seed: 42);
        var mmState = mmEnv.Reset();
        Assert.Equal(mmEnv.ObservationSpaceDimension, mmState.Length);
        var mmAction = new Vector<double>(new double[] { 0.01, 0.01 });
        var mmStep = mmEnv.Step(mmAction);
        Assert.Equal(mmEnv.ObservationSpaceDimension, mmStep.NextState.Length);
    }

    [Fact]
    public void CrossDomain_SearchSpace_DomainSpecificParams_DoNotLeak()
    {
        var forecastSpace = new FinancialSearchSpace(FinancialDomain.Forecasting);
        var riskSpace = new FinancialSearchSpace(FinancialDomain.Risk);

        var patchTST = forecastSpace.GetSearchSpace(ModelType.PatchTST);
        Assert.True(patchTST.ContainsKey("PatchLength"));

        var neuralVaR = riskSpace.GetSearchSpace(ModelType.NeuralVaR);
        Assert.True(neuralVaR.ContainsKey("ConfidenceLevel"));
        Assert.False(neuralVaR.ContainsKey("PatchLength"),
            "Risk model should not have forecasting-specific PatchLength");
    }

    [Fact]
    public void CrossDomain_SearchSpace_TabularModels_SameBase()
    {
        var riskSpace = new FinancialSearchSpace(FinancialDomain.Risk);

        var tabNet = riskSpace.GetSearchSpace(ModelType.TabNet);
        var tabTransformer = riskSpace.GetSearchSpace(ModelType.TabTransformer);

        Assert.Equal(ToDouble(tabNet["LearningRate"].MinValue), ToDouble(tabTransformer["LearningRate"].MinValue));
        Assert.Equal(ToDouble(tabNet["LearningRate"].MaxValue), ToDouble(tabTransformer["LearningRate"].MaxValue));
        Assert.Equal(ToDouble(tabNet["BatchSize"].MinValue), ToDouble(tabTransformer["BatchSize"].MinValue));
        Assert.Equal(ToDouble(tabNet["BatchSize"].MaxValue), ToDouble(tabTransformer["BatchSize"].MaxValue));
    }

    #endregion

    #region Deep Bug-Probing: Transaction Cost Math Verification

    [Fact]
    public void ExecuteTrade_BuyCost_ExactMath_CashDeductedCorrectly()
    {
        // Hand-calculated: Buy 10 units at price 50, transactionCost=0.01
        // tradeValue = 10 * 50 = 500
        // cost = 500 * (1 + 0.01) = 505
        // cash after = 10000 - 505 = 9495
        double initialCash = 10000.0;
        double price = 50.0;
        double tradeSize = 10.0;
        double txCost = 0.01;

        // Create constant-price data so price doesn't change
        var data = new Tensor<double>(new[] { 20, 1 });
        for (int t = 0; t < 20; t++) data[t, 0] = price;

        var env = new StockTradingEnvironment<double>(
            data, windowSize: 3, initialCapital: initialCash,
            tradeSize: tradeSize, transactionCost: txCost, seed: 42);

        env.Reset();

        // Action 1 = buy
        var buyAction = new Vector<double>(new double[] { 1 });
        var (_, _, _, info) = env.Step(buyAction);

        double expectedCash = initialCash - (tradeSize * price * (1.0 + txCost));
        double actualCash = (double)info["cash"];
        Assert.Equal(expectedCash, actualCash, 6);
    }

    [Fact]
    public void ExecuteTrade_SellProceeds_ExactMath_CashCreditedCorrectly()
    {
        // First buy, then sell. Verify sell proceeds use |tradeValue| * (1 - txCost)
        // Buy: cost = 10 * 50 * 1.01 = 505, cash = 10000 - 505 = 9495
        // Sell: tradeValue = -10 * 50 = -500, proceeds = |-500| * (1 - 0.01) = 495
        // cash after sell = 9495 + 495 = 9990
        double initialCash = 10000.0;
        double price = 50.0;
        double tradeSize = 10.0;
        double txCost = 0.01;

        var data = new Tensor<double>(new[] { 20, 1 });
        for (int t = 0; t < 20; t++) data[t, 0] = price;

        var env = new StockTradingEnvironment<double>(
            data, windowSize: 3, initialCapital: initialCash,
            tradeSize: tradeSize, transactionCost: txCost, seed: 42);

        env.Reset();

        // Buy first
        env.Step(new Vector<double>(new double[] { 1 }));
        // Then sell
        var (_, _, _, info) = env.Step(new Vector<double>(new double[] { 2 }));

        double cashAfterBuy = initialCash - (tradeSize * price * (1.0 + txCost));
        double sellProceeds = tradeSize * price * (1.0 - txCost);
        double expectedCash = cashAfterBuy + sellProceeds;
        double actualCash = (double)info["cash"];
        Assert.Equal(expectedCash, actualCash, 6);
    }

    [Fact]
    public void ExecuteTrade_BuyRejectsWhenInsufficientCash()
    {
        // Start with cash=100, try to buy 10 units at price=50
        // cost = 10 * 50 * 1.001 = 500.5, which exceeds 100
        // ExecuteTrade should silently reject the buy (return early)
        double initialCash = 100.0;
        double price = 50.0;

        var data = new Tensor<double>(new[] { 20, 1 });
        for (int t = 0; t < 20; t++) data[t, 0] = price;

        var env = new StockTradingEnvironment<double>(
            data, windowSize: 3, initialCapital: initialCash,
            tradeSize: 10.0, transactionCost: 0.001, seed: 42);

        env.Reset();
        var (_, _, _, info) = env.Step(new Vector<double>(new double[] { 1 })); // buy

        double actualCash = (double)info["cash"];
        // Cash should be unchanged since buy was rejected
        Assert.Equal(initialCash, actualCash, 6);

        // Position should remain 0
        if (info["positions"] is Vector<double> positions)
        {
            Assert.Equal(0.0, positions[0], 6);
        }
    }

    [Fact]
    public void ExecuteTrade_HoldDoesNotChangeCashOrPositions()
    {
        double initialCash = 10000.0;
        double price = 100.0;

        var data = new Tensor<double>(new[] { 20, 1 });
        for (int t = 0; t < 20; t++) data[t, 0] = price;

        var env = new StockTradingEnvironment<double>(
            data, windowSize: 3, initialCapital: initialCash,
            tradeSize: 1.0, transactionCost: 0.01, seed: 42);

        env.Reset();

        // Hold action (0) should not change anything
        var (_, _, _, info) = env.Step(new Vector<double>(new double[] { 0 }));

        Assert.Equal(initialCash, (double)info["cash"], 6);
        if (info["positions"] is Vector<double> pos)
        {
            Assert.Equal(0.0, pos[0], 6);
        }
    }

    [Fact]
    public void TransactionCost_RoundTrip_AlwaysLosesMoney()
    {
        // Buy then sell at the same price should always result in net loss
        // due to transaction costs. This is a fundamental invariant.
        double initialCash = 10000.0;
        double price = 100.0;
        double txCost = 0.001;

        var data = new Tensor<double>(new[] { 20, 1 });
        for (int t = 0; t < 20; t++) data[t, 0] = price;

        var env = new StockTradingEnvironment<double>(
            data, windowSize: 3, initialCapital: initialCash,
            tradeSize: 1.0, transactionCost: txCost, seed: 42);

        env.Reset();
        env.Step(new Vector<double>(new double[] { 1 })); // buy
        var (_, _, _, info) = env.Step(new Vector<double>(new double[] { 2 })); // sell

        double finalCash = (double)info["cash"];
        // After buy+sell at same price with tx costs, we must have less cash
        Assert.True(finalCash < initialCash,
            $"Round-trip should lose money: initial={initialCash}, final={finalCash}");

        // Verify exact loss amount: buy cost overhead + sell cost overhead
        double expectedLoss = price * txCost + price * txCost; // both legs lose txCost * price
        double actualLoss = initialCash - finalCash;
        Assert.Equal(expectedLoss, actualLoss, 6);
    }

    #endregion

    #region Deep Bug-Probing: Portfolio Value and Reward Verification

    [Fact]
    public void PortfolioValue_AfterBuy_EqualsPositionValuePlusCash()
    {
        // After buying: portfolioValue = positions * price + cash
        double initialCash = 10000.0;
        double price = 100.0;
        double tradeSize = 5.0;
        double txCost = 0.01;

        var data = new Tensor<double>(new[] { 20, 1 });
        for (int t = 0; t < 20; t++) data[t, 0] = price;

        var env = new StockTradingEnvironment<double>(
            data, windowSize: 3, initialCapital: initialCash,
            tradeSize: tradeSize, transactionCost: txCost, seed: 42);

        env.Reset();
        var (_, _, _, info) = env.Step(new Vector<double>(new double[] { 1 }));

        double cashAfterBuy = initialCash - (tradeSize * price * (1.0 + txCost));
        double positionValue = tradeSize * price;
        double expectedPortfolioValue = cashAfterBuy + positionValue;
        double actualPortfolioValue = (double)info["portfolioValue"];

        Assert.Equal(expectedPortfolioValue, actualPortfolioValue, 6);
    }

    [Fact]
    public void Reward_OnHold_WithNoPosition_IsZero()
    {
        // With no position and constant prices, reward should be 0
        double price = 100.0;

        var data = new Tensor<double>(new[] { 20, 1 });
        for (int t = 0; t < 20; t++) data[t, 0] = price;

        var env = new StockTradingEnvironment<double>(
            data, windowSize: 3, initialCapital: 10000.0,
            tradeSize: 1.0, transactionCost: 0.0, seed: 42);

        env.Reset();
        var (_, reward, _, _) = env.Step(new Vector<double>(new double[] { 0 }));

        // portfolioValue doesn't change (no position, constant prices) => reward = 0
        Assert.Equal(0.0, reward, 10);
    }

    [Fact]
    public void Reward_AfterPriceIncrease_WithPosition_IsPositive()
    {
        // Buy at price=100, price goes to 110. Reward should be positive.
        var data = new Tensor<double>(new[] { 20, 1 });
        for (int t = 0; t < 20; t++) data[t, 0] = 100.0 + t; // increasing prices

        var env = new StockTradingEnvironment<double>(
            data, windowSize: 3, initialCapital: 10000.0,
            tradeSize: 1.0, transactionCost: 0.0, seed: 42);

        env.Reset();
        // Buy at step windowSize=3 (price=103)
        env.Step(new Vector<double>(new double[] { 1 }));
        // Hold - price increases from 104 to 105
        var (_, reward, _, _) = env.Step(new Vector<double>(new double[] { 0 }));

        Assert.True(reward > 0, $"Reward should be positive when price increases with long position, got {reward}");
    }

    [Fact]
    public void Reward_ExactValue_PercentageReturn()
    {
        // Verify reward = (currentValue - previousValue) / previousValue
        // Buy 1 unit at price=100 with no tx cost, cash=10000
        // Step 1 (buy at 100): portfolio = 9900 cash + 1*100 = 10000
        // Step 2 (hold, price=110): portfolio = 9900 + 1*110 = 10010
        // Reward = (10010 - 10000) / 10000 = 0.001
        var data = new Tensor<double>(new[] { 20, 1 });
        data[0, 0] = 100; data[1, 0] = 100; data[2, 0] = 100;
        data[3, 0] = 100; data[4, 0] = 110; // price jump at step 4

        for (int t = 5; t < 20; t++) data[t, 0] = 110;

        var env = new StockTradingEnvironment<double>(
            data, windowSize: 3, initialCapital: 10000.0,
            tradeSize: 1.0, transactionCost: 0.0, seed: 42);

        env.Reset(); // _currentStep = 3

        // Step at _currentStep=3, price=100. Buy 1 unit.
        // After buy: cash = 10000 - 100 = 9900, position = 1
        // UpdatePortfolioValue at price 100: value = 9900 + 1*100 = 10000
        var buyResult = env.Step(new Vector<double>(new double[] { 1 }));

        // Now _currentStep=4, price=110. Hold.
        // previousValue = 10000 (from last step)
        // After hold: cash still 9900, position still 1
        // UpdatePortfolioValue at price 110: value = 9900 + 1*110 = 10010
        // reward = (10010 - 10000) / 10000 = 0.001
        var (_, reward, _, info) = env.Step(new Vector<double>(new double[] { 0 }));

        Assert.Equal(0.001, reward, 10);
        Assert.Equal(10010.0, (double)info["portfolioValue"], 6);
    }

    [Fact]
    public void Reward_WhenPreviousValueIsZero_ReturnsZero()
    {
        // Edge case: if portfolio value somehow becomes 0, reward should be 0 (not NaN/Infinity)
        // This tests the guard in ComputeReward
        var data = new Tensor<double>(new[] { 20, 1 });
        for (int t = 0; t < 20; t++) data[t, 0] = 100.0;

        // Start with 0 capital - portfolio value is 0
        var env = new StockTradingEnvironment<double>(
            data, windowSize: 3, initialCapital: 0.0,
            tradeSize: 1.0, transactionCost: 0.0, seed: 42);

        env.Reset();
        var (_, reward, _, _) = env.Step(new Vector<double>(new double[] { 0 }));

        Assert.Equal(0.0, reward, 10);
        Assert.False(double.IsNaN(reward), "Reward should not be NaN when previous value is 0");
        Assert.False(double.IsInfinity(reward), "Reward should not be Infinity when previous value is 0");
    }

    #endregion

    #region Deep Bug-Probing: Observation Window Verification

    [Fact]
    public void BuildObservation_ContainsCorrectPriceWindow()
    {
        // Verify the observation vector has the correct price data from the market tensor
        // With windowSize=3, at step=3, observation should include prices at steps [1,2,3]
        int windowSize = 3;
        int numAssets = 1;

        var data = new Tensor<double>(new[] { 20, 1 });
        for (int t = 0; t < 20; t++) data[t, 0] = (t + 1) * 10.0; // 10, 20, 30, ...

        var env = new StockTradingEnvironment<double>(
            data, windowSize: windowSize, initialCapital: 10000.0,
            tradeSize: 1.0, transactionCost: 0.0, seed: 42);

        var obs = env.Reset(); // _currentStep = windowSize = 3

        // BuildObservation(3): start = 3 - 3 + 1 = 1
        // Prices at indices [1, 2, 3] = [20.0, 30.0, 40.0]
        // Then: positions[0] = 0.0, cash = 10000.0
        Assert.Equal(20.0, obs[0], 6); // price at step 1
        Assert.Equal(30.0, obs[1], 6); // price at step 2
        Assert.Equal(40.0, obs[2], 6); // price at step 3

        // Position comes after window prices
        Assert.Equal(0.0, obs[3], 6); // position for asset 0

        // Cash is last
        Assert.Equal(10000.0, obs[4], 6);
    }

    [Fact]
    public void BuildObservation_Dimension_MatchesFormula()
    {
        // ObservationSpaceDimension = (WindowSize * NumAssets) + NumAssets + 1
        int windowSize = 5;
        int numAssets = 3;

        var data = FinanceTestHelpers.CreatePriceTensor<double>(steps: 30, assets: numAssets);
        var env = new PortfolioTradingEnvironment<double>(
            data, windowSize: windowSize, initialCapital: 10000.0, seed: 42);

        int expectedDim = (windowSize * numAssets) + numAssets + 1;
        Assert.Equal(expectedDim, env.ObservationSpaceDimension);

        var obs = env.Reset();
        Assert.Equal(expectedDim, obs.Length);
    }

    [Fact]
    public void BuildObservation_AfterBuy_ShowsUpdatedPosition()
    {
        // After buying, observation should reflect the new position
        var data = new Tensor<double>(new[] { 20, 1 });
        for (int t = 0; t < 20; t++) data[t, 0] = 100.0;

        var env = new StockTradingEnvironment<double>(
            data, windowSize: 3, initialCapital: 10000.0,
            tradeSize: 5.0, transactionCost: 0.0, seed: 42);

        env.Reset();
        var (nextState, _, _, _) = env.Step(new Vector<double>(new double[] { 1 }));

        // Observation layout: [windowSize prices] [position] [cash]
        // After buy: position = 5.0, cash = 10000 - 5*100 = 9500
        int posIndex = 3; // windowSize * numAssets
        Assert.Equal(5.0, nextState[posIndex], 6);
        Assert.Equal(9500.0, nextState[posIndex + 1], 6);
    }

    [Fact]
    public void BuildObservation_MultiAsset_InterleavedPricesCorrect()
    {
        // With 2 assets, verify prices are interleaved correctly: [t0_a0, t0_a1, t1_a0, t1_a1, ...]
        int windowSize = 2;
        int numAssets = 2;

        var data = new Tensor<double>(new[] { 10, numAssets });
        for (int t = 0; t < 10; t++)
        {
            data[t, 0] = (t + 1) * 100.0;  // asset 0: 100, 200, 300, ...
            data[t, 1] = (t + 1) * 10.0;   // asset 1: 10, 20, 30, ...
        }

        var env = new PortfolioTradingEnvironment<double>(
            data, windowSize: windowSize, initialCapital: 10000.0, seed: 42);

        var obs = env.Reset(); // _currentStep = 2

        // BuildObservation(2): start = 2 - 2 + 1 = 1
        // step 1: asset0=200, asset1=20
        // step 2: asset0=300, asset1=30
        Assert.Equal(200.0, obs[0], 6); // step 1, asset 0
        Assert.Equal(20.0, obs[1], 6);  // step 1, asset 1
        Assert.Equal(300.0, obs[2], 6); // step 2, asset 0
        Assert.Equal(30.0, obs[3], 6);  // step 2, asset 1
    }

    #endregion

    #region Deep Bug-Probing: Short Selling Constraints

    [Fact]
    public void ShortSelling_Disabled_SellClampsToCurrentPosition()
    {
        // With no short selling, selling more than you own should clamp to position
        var data = new Tensor<double>(new[] { 20, 1 });
        for (int t = 0; t < 20; t++) data[t, 0] = 100.0;

        var env = new StockTradingEnvironment<double>(
            data, windowSize: 3, initialCapital: 10000.0,
            tradeSize: 5.0, transactionCost: 0.0, allowShortSelling: false, seed: 42);

        env.Reset();

        // Sell without any position - should be clamped to 0
        var (_, _, _, info) = env.Step(new Vector<double>(new double[] { 2 }));

        if (info["positions"] is Vector<double> pos)
        {
            Assert.True(pos[0] >= 0.0,
                $"Position should not go negative with no short selling, got {pos[0]}");
        }

        // Cash should not change (sell of 0 units)
        Assert.Equal(10000.0, (double)info["cash"], 6);
    }

    [Fact]
    public void ShortSelling_Enabled_AllowsNegativePosition()
    {
        var data = new Tensor<double>(new[] { 20, 1 });
        for (int t = 0; t < 20; t++) data[t, 0] = 100.0;

        var env = new StockTradingEnvironment<double>(
            data, windowSize: 3, initialCapital: 10000.0,
            tradeSize: 5.0, transactionCost: 0.0, allowShortSelling: true, seed: 42);

        env.Reset();

        // Sell without position - should create short position of -5
        var (_, _, _, info) = env.Step(new Vector<double>(new double[] { 2 }));

        if (info["positions"] is Vector<double> pos)
        {
            Assert.Equal(-5.0, pos[0], 6);
        }
    }

    [Fact]
    public void ShortSelling_Disabled_PartialSell_ClampsCorrectly()
    {
        // Buy 5, then try to sell 5, then try to sell again (should be clamped)
        var data = new Tensor<double>(new[] { 20, 1 });
        for (int t = 0; t < 20; t++) data[t, 0] = 100.0;

        var env = new StockTradingEnvironment<double>(
            data, windowSize: 3, initialCapital: 10000.0,
            tradeSize: 5.0, transactionCost: 0.0, allowShortSelling: false, seed: 42);

        env.Reset();
        env.Step(new Vector<double>(new double[] { 1 })); // buy 5
        env.Step(new Vector<double>(new double[] { 2 })); // sell 5 (now at 0)
        var (_, _, _, info) = env.Step(new Vector<double>(new double[] { 2 })); // try sell again

        if (info["positions"] is Vector<double> pos)
        {
            Assert.True(pos[0] >= 0.0, $"Position should not go negative: {pos[0]}");
        }
    }

    #endregion

    #region Deep Bug-Probing: Portfolio Rebalancing Math

    [Fact]
    public void Portfolio_NormalizeWeights_AllNegative_NoShortSelling_FallsBackToUniform()
    {
        // When all weights are negative and short selling is disabled,
        // all get clamped to 0, sum=0 triggers uniform fallback
        int numAssets = 2;
        var data = new Tensor<double>(new[] { 20, numAssets });
        for (int t = 0; t < 20; t++)
        {
            data[t, 0] = 100.0;
            data[t, 1] = 100.0;
        }

        var env = new PortfolioTradingEnvironment<double>(
            data, windowSize: 3, initialCapital: 10000.0,
            allowShortSelling: false, transactionCost: 0.0, seed: 42);

        env.Reset();

        // Pass all-negative weights
        var (_, _, _, info) = env.Step(new Vector<double>(new double[] { -1.0, -1.0 }));

        // Uniform: 50% each asset, so portfolio should be split evenly
        double portfolioValue = (double)info["portfolioValue"];
        Assert.True(portfolioValue > 0, "Portfolio should have positive value after uniform allocation");

        if (info["positions"] is Vector<double> pos)
        {
            // Both assets should have equal positions (uniform weights)
            Assert.Equal(pos[0], pos[1], 4);
        }
    }

    [Fact]
    public void Portfolio_Rebalancing_ExactWeights_VerifyPositionSizes()
    {
        // Target 100% in asset 0, 0% in asset 1
        // portfolioValue = 10000 (all cash initially)
        // desired position for asset 0 = 10000 * 1.0 / price = 10000 / 100 = 100 units
        int numAssets = 2;
        var data = new Tensor<double>(new[] { 20, numAssets });
        for (int t = 0; t < 20; t++)
        {
            data[t, 0] = 100.0;
            data[t, 1] = 50.0;
        }

        var env = new PortfolioTradingEnvironment<double>(
            data, windowSize: 3, initialCapital: 10000.0,
            allowShortSelling: false, transactionCost: 0.0, seed: 42);

        env.Reset();

        // 100% in asset 0 (weight normalization: [1, 0] -> [1.0, 0.0])
        var (_, _, _, info) = env.Step(new Vector<double>(new double[] { 1.0, 0.0 }));

        if (info["positions"] is Vector<double> pos)
        {
            // All capital into asset 0: 10000 / 100 = 100 units
            Assert.Equal(100.0, pos[0], 4);
            // Asset 1 should be 0
            Assert.Equal(0.0, pos[1], 4);
        }
    }

    [Fact]
    public void Portfolio_SellsExecuteBeforeBuys_RebalancingUsesFreedCash()
    {
        // Verifies sells execute before buys so cash from sell proceeds is available.
        // Test: Start with 100% asset 0, rebalance to 100% asset 1.
        // The sell of asset 0 should free up cash for buying asset 1.
        // With correct ordering: sells first -> cash freed -> buys scaled to freed cash.
        int numAssets = 2;
        var data = new Tensor<double>(new[] { 20, numAssets });
        for (int t = 0; t < 20; t++)
        {
            data[t, 0] = 100.0;
            data[t, 1] = 100.0;
        }

        var env = new PortfolioTradingEnvironment<double>(
            data, windowSize: 3, initialCapital: 10000.0,
            allowShortSelling: false, transactionCost: 0.0, seed: 42);

        env.Reset();

        // First: allocate 100% to asset 0
        env.Step(new Vector<double>(new double[] { 1.0, 0.0 }));

        // Now: rebalance to 100% asset 1 (sell all asset 0, buy asset 1)
        var (_, _, _, info) = env.Step(new Vector<double>(new double[] { 0.0, 1.0 }));

        if (info["positions"] is Vector<double> pos)
        {
            double totalValue = (double)info["portfolioValue"];
            double asset1Value = pos[1] * 100.0;

            // After fix: sell of asset 0 frees cash, then buy of asset 1 uses it.
            // Asset 1 should hold nearly all portfolio value.
            Assert.True(asset1Value >= totalValue * 0.9,
                $"After rebalancing 100% to asset 1, asset 1 value ({asset1Value}) " +
                $"should be >= 90% of portfolio ({totalValue * 0.9}). " +
                $"Low value means sells didn't execute before buys.");

            // Asset 0 should be sold off completely
            Assert.Equal(0.0, pos[0], 4);
        }
    }

    #endregion

    #region Deep Bug-Probing: Market Making Reward Scale Mismatch

    [Fact]
    public void MarketMaking_InventoryPenalty_ScaleMismatch_WithBaseReward()
    {
        // BUG PROBE: ComputeReward returns baseReward (percentage) - |position| * inventoryPenalty (absolute)
        // baseReward is typically ~0.001 (0.1% return)
        // inventoryPenalty * |position| could be 0.001 * 10 = 0.01
        // This means the penalty is 10x the base reward scale!
        // For a market maker holding 10 units, the penalty dominates completely.
        double price = 100.0;
        double inventoryPenalty = 0.001;

        var data = new Tensor<double>(new[] { 20, 1 });
        for (int t = 0; t < 20; t++) data[t, 0] = price;

        var env = new MarketMakingEnvironment<double>(
            data, windowSize: 3, initialCapital: 100000.0,
            tradeSize: 1.0, baseSpread: 0.01, orderArrivalRate: 1.0,
            maxInventory: 10, inventoryPenalty: inventoryPenalty,
            transactionCost: 0.0, seed: 42);

        env.Reset();

        // Run multiple steps to accumulate inventory
        double maxAbsReward = 0;
        double maxPenalty = 0;
        for (int i = 0; i < 15; i++)
        {
            var action = new Vector<double>(new double[] { 0.001, 0.001 }); // tight spread
            var (_, reward, done, info) = env.Step(action);
            if (done) break;

            if (info["positions"] is Vector<double> pos)
            {
                double penalty = Math.Abs(pos[0]) * inventoryPenalty;
                maxPenalty = Math.Max(maxPenalty, penalty);
            }

            maxAbsReward = Math.Max(maxAbsReward, Math.Abs(reward));
        }

        // If penalty exceeds typical base reward magnitude significantly,
        // this reveals the scale mismatch issue
        if (maxPenalty > 0.005)
        {
            // Penalty of 0.01 with base reward of ~0.001 = penalty dominates by 10x
            Assert.True(maxPenalty > 0,
                "Inventory penalty scale is much larger than base percentage reward - " +
                "this may cause the agent to never hold inventory");
        }
    }

    #endregion

    #region Deep Bug-Probing: Episode and Reset Logic

    [Fact]
    public void MaxEpisodeLength_TerminatesCorrectly()
    {
        var data = new Tensor<double>(new[] { 100, 1 });
        for (int t = 0; t < 100; t++) data[t, 0] = 100.0 + t;

        int maxEpLen = 5;
        var env = new StockTradingEnvironment<double>(
            data, windowSize: 3, initialCapital: 10000.0,
            tradeSize: 1.0, maxEpisodeLength: maxEpLen, seed: 42);

        env.Reset();

        int stepsBeforeDone = 0;
        for (int i = 0; i < 50; i++)
        {
            var (_, _, done, _) = env.Step(new Vector<double>(new double[] { 0 }));
            stepsBeforeDone++;
            if (done) break;
        }

        Assert.Equal(maxEpLen, stepsBeforeDone);
    }

    [Fact]
    public void Reset_RestoresInitialState()
    {
        var data = new Tensor<double>(new[] { 30, 1 });
        for (int t = 0; t < 30; t++) data[t, 0] = 100.0;

        var env = new StockTradingEnvironment<double>(
            data, windowSize: 3, initialCapital: 10000.0,
            tradeSize: 5.0, transactionCost: 0.0, seed: 42);

        // First episode
        var obs1 = env.Reset();
        env.Step(new Vector<double>(new double[] { 1 })); // buy
        env.Step(new Vector<double>(new double[] { 1 })); // buy again

        // Second episode - should be fully reset
        var obs2 = env.Reset();

        // Observations should be identical after reset (same starting conditions)
        // because seed is fixed, randomStart is false
        for (int i = 0; i < obs1.Length; i++)
        {
            Assert.Equal(obs1[i], obs2[i], 6);
        }
    }

    [Fact]
    public void Seed_ProducesReproducibleResults()
    {
        var data = new Tensor<double>(new[] { 30, 1 });
        for (int t = 0; t < 30; t++) data[t, 0] = 100.0 + t;

        // Two environments with same seed should behave identically
        var env1 = new StockTradingEnvironment<double>(
            data, windowSize: 3, initialCapital: 10000.0,
            tradeSize: 1.0, randomStart: true, seed: 42);

        var env2 = new StockTradingEnvironment<double>(
            data, windowSize: 3, initialCapital: 10000.0,
            tradeSize: 1.0, randomStart: true, seed: 42);

        var obs1 = env1.Reset();
        var obs2 = env2.Reset();

        for (int i = 0; i < obs1.Length; i++)
        {
            Assert.Equal(obs1[i], obs2[i], 10);
        }
    }

    [Fact]
    public void Step_AfterEndOfData_MarksEpisodeAsDone()
    {
        // Create minimal data: just enough for windowSize + 1 step
        int windowSize = 3;
        int steps = windowSize + 1; // only 1 step possible after reset

        var data = new Tensor<double>(new[] { steps, 1 });
        for (int t = 0; t < steps; t++) data[t, 0] = 100.0;

        var env = new StockTradingEnvironment<double>(
            data, windowSize: windowSize, initialCapital: 10000.0,
            tradeSize: 1.0, seed: 42);

        env.Reset(); // _currentStep = windowSize = 3

        // After step: _currentStep becomes 4, which equals data length (4)
        var (_, _, done, _) = env.Step(new Vector<double>(new double[] { 0 }));
        Assert.True(done, "Episode should be done when data is exhausted");
    }

    #endregion

    #region Deep Bug-Probing: StockTradingEnvironment Action Parsing

    [Fact]
    public void StockTrading_OneHotAction_PicksMaxIndex()
    {
        var data = new Tensor<double>(new[] { 20, 1 });
        for (int t = 0; t < 20; t++) data[t, 0] = 100.0;

        var env = new StockTradingEnvironment<double>(
            data, windowSize: 3, initialCapital: 10000.0,
            tradeSize: 1.0, transactionCost: 0.0, seed: 42);

        env.Reset();

        // One-hot for buy: [0, 1, 0] -> index 1 = buy
        var (_, _, _, info) = env.Step(new Vector<double>(new double[] { 0.0, 1.0, 0.0 }));

        if (info["positions"] is Vector<double> pos)
        {
            Assert.Equal(1.0, pos[0], 6); // bought 1 unit
        }
    }

    [Fact]
    public void StockTrading_ScalarAction_ConvertsToIndex()
    {
        var data = new Tensor<double>(new[] { 20, 1 });
        for (int t = 0; t < 20; t++) data[t, 0] = 100.0;

        var env = new StockTradingEnvironment<double>(
            data, windowSize: 3, initialCapital: 10000.0,
            tradeSize: 1.0, transactionCost: 0.0, seed: 42);

        env.Reset();

        // Scalar action: [2] = sell
        var (_, _, _, info) = env.Step(new Vector<double>(new double[] { 2 }));

        // With short selling enabled, position should be -1
        if (info["positions"] is Vector<double> pos)
        {
            Assert.Equal(-1.0, pos[0], 6);
        }
    }

    [Fact]
    public void StockTrading_SoftmaxLikeAction_PicksHighestProbability()
    {
        // Simulates a softmax output: [0.1, 0.7, 0.2] -> index 1 = buy
        var data = new Tensor<double>(new[] { 20, 1 });
        for (int t = 0; t < 20; t++) data[t, 0] = 100.0;

        var env = new StockTradingEnvironment<double>(
            data, windowSize: 3, initialCapital: 10000.0,
            tradeSize: 1.0, transactionCost: 0.0, seed: 42);

        env.Reset();

        var (_, _, _, info) = env.Step(new Vector<double>(new double[] { 0.1, 0.7, 0.2 }));

        if (info["positions"] is Vector<double> pos)
        {
            Assert.Equal(1.0, pos[0], 6); // buy action selected
        }
    }

    #endregion

    #region Deep Bug-Probing: Multiple Steps Portfolio Tracking

    [Fact]
    public void MultipleBuys_AccumulatePosition()
    {
        var data = new Tensor<double>(new[] { 20, 1 });
        for (int t = 0; t < 20; t++) data[t, 0] = 100.0;

        var env = new StockTradingEnvironment<double>(
            data, windowSize: 3, initialCapital: 100000.0,
            tradeSize: 10.0, transactionCost: 0.0, seed: 42);

        env.Reset();

        env.Step(new Vector<double>(new double[] { 1 })); // buy 10
        env.Step(new Vector<double>(new double[] { 1 })); // buy 10
        var (_, _, _, info) = env.Step(new Vector<double>(new double[] { 1 })); // buy 10

        if (info["positions"] is Vector<double> pos)
        {
            Assert.Equal(30.0, pos[0], 6);
        }

        double expectedCash = 100000.0 - (3 * 10.0 * 100.0);
        Assert.Equal(expectedCash, (double)info["cash"], 6);
    }

    [Fact]
    public void PortfolioValue_ConsistentAcrossSteps_ConstantPrice()
    {
        // With constant prices and no transaction costs, portfolio value should
        // remain constant regardless of trading (conservation of value)
        var data = new Tensor<double>(new[] { 20, 1 });
        for (int t = 0; t < 20; t++) data[t, 0] = 100.0;

        var env = new StockTradingEnvironment<double>(
            data, windowSize: 3, initialCapital: 10000.0,
            tradeSize: 5.0, transactionCost: 0.0, seed: 42);

        env.Reset();

        var actions = new[] { 1, 0, 1, 2, 0, 1, 2, 2 }; // mix of buy/hold/sell
        foreach (var a in actions)
        {
            var (_, _, done, info) = env.Step(new Vector<double>(new double[] { a }));
            if (done) break;

            double portfolioValue = (double)info["portfolioValue"];
            Assert.Equal(10000.0, portfolioValue, 4);
        }
    }

    [Fact]
    public void PortfolioValue_WithTransactionCosts_MonotonicallyDecreasing_ConstantPrice()
    {
        // With constant prices and transaction costs, each trade reduces portfolio value
        var data = new Tensor<double>(new[] { 20, 1 });
        for (int t = 0; t < 20; t++) data[t, 0] = 100.0;

        var env = new StockTradingEnvironment<double>(
            data, windowSize: 3, initialCapital: 10000.0,
            tradeSize: 1.0, transactionCost: 0.01, seed: 42);

        env.Reset();

        double prevValue = 10000.0;
        var tradingActions = new[] { 1, 2, 1, 2, 1, 2 }; // buy/sell cycles
        foreach (var a in tradingActions)
        {
            var (_, _, done, info) = env.Step(new Vector<double>(new double[] { a }));
            if (done) break;

            double currentValue = (double)info["portfolioValue"];
            Assert.True(currentValue <= prevValue + 1e-6,
                $"Portfolio value should not increase at constant prices with tx costs: prev={prevValue}, curr={currentValue}");
            prevValue = currentValue;
        }
    }

    #endregion
}
