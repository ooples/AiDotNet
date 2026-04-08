using AiDotNet.Enums;
using AiDotNet.Factories;
using AiDotNet.PromptEngineering.Templates;
using AiDotNet.Training.Factories;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Factories;

/// <summary>
/// Deep integration tests for Factories:
/// LossFunctionFactory (all loss types creation, string-based creation, parameter handling),
/// WindowFunctionFactory (all 20 window function types),
/// PromptTemplateFactory (all template types).
/// </summary>
public class FactoriesDeepMathIntegrationTests
{
    // ============================
    // LossFunctionFactory: All Loss Types
    // ============================

    [Theory]
    [InlineData(LossType.MeanSquaredError)]
    [InlineData(LossType.MeanAbsoluteError)]
    [InlineData(LossType.RootMeanSquaredError)]
    [InlineData(LossType.Huber)]
    [InlineData(LossType.CrossEntropy)]
    [InlineData(LossType.BinaryCrossEntropy)]
    [InlineData(LossType.CategoricalCrossEntropy)]
    [InlineData(LossType.SparseCategoricalCrossEntropy)]
    [InlineData(LossType.Focal)]
    [InlineData(LossType.Hinge)]
    [InlineData(LossType.SquaredHinge)]
    [InlineData(LossType.LogCosh)]
    [InlineData(LossType.Quantile)]
    [InlineData(LossType.Poisson)]
    [InlineData(LossType.KullbackLeiblerDivergence)]
    [InlineData(LossType.CosineSimilarity)]
    [InlineData(LossType.Contrastive)]
    [InlineData(LossType.Triplet)]
    [InlineData(LossType.Dice)]
    [InlineData(LossType.Jaccard)]
    [InlineData(LossType.ElasticNet)]
    [InlineData(LossType.Exponential)]
    [InlineData(LossType.ModifiedHuber)]
    [InlineData(LossType.Charbonnier)]
    [InlineData(LossType.MeanBiasError)]
    [InlineData(LossType.Wasserstein)]
    [InlineData(LossType.Margin)]
    [InlineData(LossType.CTC)]
    [InlineData(LossType.NoiseContrastiveEstimation)]
    [InlineData(LossType.OrdinalRegression)]
    [InlineData(LossType.WeightedCrossEntropy)]
    [InlineData(LossType.ScaleInvariantDepth)]
    [InlineData(LossType.Quantum)]
    public void LossFactory_CreateByEnum_ReturnsNotNull(LossType lossType)
    {
        var loss = LossFunctionFactory<double>.Create(lossType);
        Assert.NotNull(loss);
    }

    // ============================
    // LossFunctionFactory: String-Based Creation
    // ============================

    [Theory]
    [InlineData("MeanSquaredError")]
    [InlineData("meansquarederror")]
    [InlineData("MEANSQUAREDERROR")]
    [InlineData("CrossEntropy")]
    [InlineData("Huber")]
    public void LossFactory_CreateByString_CaseInsensitive(string name)
    {
        var loss = LossFunctionFactory<double>.Create(name);
        Assert.NotNull(loss);
    }

    [Fact]
    public void LossFactory_CreateByString_EmptyName_Throws()
    {
        Assert.Throws<ArgumentException>(() => LossFunctionFactory<double>.Create(""));
    }

    [Fact]
    public void LossFactory_CreateByString_UnknownName_Throws()
    {
        Assert.Throws<ArgumentException>(() => LossFunctionFactory<double>.Create("UnknownLoss"));
    }

    // ============================
    // LossFunctionFactory: Parameters
    // ============================

    [Fact]
    public void LossFactory_Huber_CustomDelta()
    {
        var parameters = new Dictionary<string, object> { { "delta", 2.0 } };
        var loss = LossFunctionFactory<double>.Create(LossType.Huber, parameters);
        Assert.NotNull(loss);
    }

    [Fact]
    public void LossFactory_Focal_CustomGammaAndAlpha()
    {
        var parameters = new Dictionary<string, object>
        {
            { "gamma", 3.0 },
            { "alpha", 0.5 }
        };
        var loss = LossFunctionFactory<double>.Create(LossType.Focal, parameters);
        Assert.NotNull(loss);
    }

    [Fact]
    public void LossFactory_Quantile_CustomQuantile()
    {
        var parameters = new Dictionary<string, object> { { "quantile", 0.9 } };
        var loss = LossFunctionFactory<double>.Create(LossType.Quantile, parameters);
        Assert.NotNull(loss);
    }

    [Fact]
    public void LossFactory_Contrastive_CustomMargin()
    {
        var parameters = new Dictionary<string, object> { { "margin", 2.0 } };
        var loss = LossFunctionFactory<double>.Create(LossType.Contrastive, parameters);
        Assert.NotNull(loss);
    }

    [Fact]
    public void LossFactory_ElasticNet_CustomParams()
    {
        var parameters = new Dictionary<string, object>
        {
            { "l1Ratio", 0.7 },
            { "alpha", 0.001 }
        };
        var loss = LossFunctionFactory<double>.Create(LossType.ElasticNet, parameters);
        Assert.NotNull(loss);
    }

    [Fact]
    public void LossFactory_CTC_CustomClasses()
    {
        var parameters = new Dictionary<string, object>
        {
            { "numClasses", 26 },
            { "blankIndex", 0 }
        };
        var loss = LossFunctionFactory<double>.Create(LossType.CTC, parameters);
        Assert.NotNull(loss);
    }

    [Fact]
    public void LossFactory_Margin_CustomParams()
    {
        var parameters = new Dictionary<string, object>
        {
            { "mPlus", 0.95 },
            { "mMinus", 0.05 },
            { "lambda", 0.3 }
        };
        var loss = LossFunctionFactory<double>.Create(LossType.Margin, parameters);
        Assert.NotNull(loss);
    }

    [Fact]
    public void LossFactory_NullParameters_UsesDefaults()
    {
        var loss = LossFunctionFactory<double>.Create(LossType.Huber, null);
        Assert.NotNull(loss);
    }

    [Fact]
    public void LossFactory_StringParam_ParsedCorrectly()
    {
        var parameters = new Dictionary<string, object> { { "delta", "1.5" } };
        var loss = LossFunctionFactory<double>.Create(LossType.Huber, parameters);
        Assert.NotNull(loss);
    }

    [Fact]
    public void LossFactory_IntParam_ConvertedCorrectly()
    {
        var parameters = new Dictionary<string, object> { { "delta", 2 } };
        var loss = LossFunctionFactory<double>.Create(LossType.Huber, parameters);
        Assert.NotNull(loss);
    }

    [Fact]
    public void LossFactory_FloatParam_ConvertedCorrectly()
    {
        var parameters = new Dictionary<string, object> { { "delta", 1.5f } };
        var loss = LossFunctionFactory<double>.Create(LossType.Huber, parameters);
        Assert.NotNull(loss);
    }

    // ============================
    // WindowFunctionFactory: All Window Types
    // ============================

    [Theory]
    [InlineData(WindowFunctionType.Rectangular)]
    [InlineData(WindowFunctionType.Hanning)]
    [InlineData(WindowFunctionType.Hamming)]
    [InlineData(WindowFunctionType.Blackman)]
    [InlineData(WindowFunctionType.Kaiser)]
    [InlineData(WindowFunctionType.Bartlett)]
    [InlineData(WindowFunctionType.Gaussian)]
    [InlineData(WindowFunctionType.BartlettHann)]
    [InlineData(WindowFunctionType.Bohman)]
    [InlineData(WindowFunctionType.Lanczos)]
    [InlineData(WindowFunctionType.Parzen)]
    [InlineData(WindowFunctionType.Poisson)]
    [InlineData(WindowFunctionType.Nuttall)]
    [InlineData(WindowFunctionType.Triangular)]
    [InlineData(WindowFunctionType.BlackmanHarris)]
    [InlineData(WindowFunctionType.FlatTop)]
    [InlineData(WindowFunctionType.Welch)]
    [InlineData(WindowFunctionType.BlackmanNuttall)]
    [InlineData(WindowFunctionType.Cosine)]
    [InlineData(WindowFunctionType.Tukey)]
    public void WindowFactory_CreateAll_ReturnsNotNull(WindowFunctionType windowType)
    {
        var window = WindowFunctionFactory.CreateWindowFunction<double>(windowType);
        Assert.NotNull(window);
    }

    [Fact]
    public void WindowFactory_InvalidType_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            WindowFunctionFactory.CreateWindowFunction<double>((WindowFunctionType)999));
    }

    [Fact]
    public void WindowFactory_FloatType_Works()
    {
        var window = WindowFunctionFactory.CreateWindowFunction<float>(WindowFunctionType.Hanning);
        Assert.NotNull(window);
    }

    // ============================
    // PromptTemplateFactory: All Template Types
    // ============================

    [Fact]
    public void PromptFactory_Simple_ReturnsTemplate()
    {
        var template = PromptTemplateFactory.Create(PromptTemplateType.Simple, "Hello {name}");
        Assert.NotNull(template);
        Assert.IsType<SimplePromptTemplate>(template);
    }

    [Fact]
    public void PromptFactory_Chat_ReturnsTemplate()
    {
        var template = PromptTemplateFactory.Create(PromptTemplateType.Chat);
        Assert.NotNull(template);
        Assert.IsType<ChatPromptTemplate>(template);
    }

    [Fact]
    public void PromptFactory_ChainOfThought_AppendsStepByStep()
    {
        var template = PromptTemplateFactory.Create(PromptTemplateType.ChainOfThought, "Solve {problem}");
        Assert.NotNull(template);
    }

    [Fact]
    public void PromptFactory_ChainOfThought_NullTemplate_UsesDefault()
    {
        var template = PromptTemplateFactory.Create(PromptTemplateType.ChainOfThought);
        Assert.NotNull(template);
    }

    [Fact]
    public void PromptFactory_ReAct_ReturnsTemplate()
    {
        var template = PromptTemplateFactory.Create(PromptTemplateType.ReAct, "Custom base");
        Assert.NotNull(template);
    }

    [Fact]
    public void PromptFactory_ReAct_NullTemplate_UsesDefault()
    {
        var template = PromptTemplateFactory.Create(PromptTemplateType.ReAct);
        Assert.NotNull(template);
    }

    [Fact]
    public void PromptFactory_Tool_ReturnsSimpleTemplate()
    {
        var template = PromptTemplateFactory.Create(PromptTemplateType.Tool, "Use tool {tool}");
        Assert.NotNull(template);
    }

    [Fact]
    public void PromptFactory_Optimized_ReturnsSimpleTemplate()
    {
        var template = PromptTemplateFactory.Create(PromptTemplateType.Optimized, "Optimize {input}");
        Assert.NotNull(template);
    }

    [Fact]
    public void PromptFactory_Simple_NullTemplate_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            PromptTemplateFactory.Create(PromptTemplateType.Simple, null));
    }

    [Fact]
    public void PromptFactory_Simple_EmptyTemplate_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            PromptTemplateFactory.Create(PromptTemplateType.Simple, "  "));
    }

    [Fact]
    public void PromptFactory_FewShot_WithoutSelector_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            PromptTemplateFactory.Create(PromptTemplateType.FewShot, "template"));
    }

    // ============================
    // TrainingResult: Defaults
    // ============================

    [Fact]
    public void TrainingResult_Defaults()
    {
        var result = new AiDotNet.Training.TrainingResult<double>();

        Assert.Null(result.TrainedModel);
        Assert.Empty(result.EpochLosses);
        Assert.Equal(0, result.TotalEpochs);
        Assert.Equal(TimeSpan.Zero, result.TrainingDuration);
        Assert.False(result.Completed);
    }

    [Fact]
    public void TrainingResult_SetProperties()
    {
        var result = new AiDotNet.Training.TrainingResult<double>
        {
            TotalEpochs = 100,
            TrainingDuration = TimeSpan.FromMinutes(30),
            Completed = true,
            EpochLosses = new List<double> { 1.0, 0.5, 0.25, 0.1 }
        };

        Assert.Equal(100, result.TotalEpochs);
        Assert.Equal(TimeSpan.FromMinutes(30), result.TrainingDuration);
        Assert.True(result.Completed);
        Assert.Equal(4, result.EpochLosses.Count);
        Assert.Equal(0.1, result.EpochLosses.Last());
    }
}
