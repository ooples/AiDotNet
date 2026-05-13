using AiDotNet.Diffusion.Conditioning;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Enums;
using AiDotNet.Tokenization;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.UnitTests.Diffusion.Models;

/// <summary>
/// Contract tests for Phase 1 conditioning infrastructure: text conditioners and noise predictors.
/// Tests pass an explicit (small-vocab) tokenizer — the conditioner ctors now require one,
/// matching the PyTorch convention where model construction and tokenizer loading are
/// separate concerns.
/// </summary>
public class NewConditionerContractTests : DiffusionUnitTestBase
{
    #region Text Conditioner Constructor Tests

    [Fact(Timeout = 120000)]
    public async Task SigLIPTextConditioner_Construct_CreatesValidConditioner()
    {
        var conditioner = new SigLIPTextConditioner<double>(ClipTokenizerFactory.CreateSimple());

        Assert.NotNull(conditioner);
        Assert.True(conditioner.ProducesPooledOutput);
        Assert.True(conditioner.EmbeddingDimension > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task SigLIP2TextConditioner_Construct_CreatesValidConditioner()
    {
        var conditioner = new SigLIP2TextConditioner<double>(ClipTokenizerFactory.CreateSimple());

        Assert.NotNull(conditioner);
        Assert.True(conditioner.ProducesPooledOutput);
        Assert.True(conditioner.EmbeddingDimension > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task DistilledT5TextConditioner_Construct_CreatesValidConditioner()
    {
        var tokenizer = LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.FlanT5);
        var conditioner = new DistilledT5TextConditioner<double>(tokenizer);

        Assert.NotNull(conditioner);
        Assert.True(conditioner.EmbeddingDimension > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task GemmaTextConditioner_Construct_CreatesValidConditioner()
    {
        var tokenizer = LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.LLaMA);
        var conditioner = new GemmaTextConditioner<double>(tokenizer);

        Assert.NotNull(conditioner);
        Assert.True(conditioner.EmbeddingDimension > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task Qwen2TextConditioner_Construct_CreatesValidConditioner()
    {
        var tokenizer = LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.Qwen);
        var conditioner = new Qwen2TextConditioner<double>(tokenizer);

        Assert.NotNull(conditioner);
        Assert.True(conditioner.EmbeddingDimension > 0);
    }

    #endregion

    #region Noise Predictor Constructor Tests

    [Fact(Timeout = 120000)]
    public async Task MMDiTXNoisePredictor_DefaultConstructor_CreatesValidPredictor()
    {
        var predictor = new MMDiTXNoisePredictor<double>();

        Assert.NotNull(predictor);
        Assert.True(predictor.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task FluxDoubleStreamPredictor_DefaultConstructor_CreatesValidPredictor()
    {
        var predictor = new FluxDoubleStreamPredictor<double>();

        Assert.NotNull(predictor);
        Assert.True(predictor.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task SiTPredictor_DefaultConstructor_CreatesValidPredictor()
    {
        var predictor = new SiTPredictor<double>();

        Assert.NotNull(predictor);
        Assert.True(predictor.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task EMMDiTPredictor_DefaultConstructor_CreatesValidPredictor()
    {
        var predictor = new EMMDiTPredictor<double>();

        Assert.NotNull(predictor);
        Assert.True(predictor.ParameterCount > 0);
    }

    #endregion

    #region Parameterizable Contract Tests

    [Fact(Timeout = 120000)]
    public async Task MMDiTXNoisePredictor_GetSetParameters_RoundTrips()
    {
        var predictor = new MMDiTXNoisePredictor<double>();

        var parameters = predictor.GetParameters();
        Assert.True(parameters.Length > 0);

        predictor.SetParameters(parameters);
        var retrieved = predictor.GetParameters();
        Assert.Equal(parameters.Length, retrieved.Length);
    }

    [Fact(Timeout = 120000)]
    public async Task FluxDoubleStreamPredictor_GetSetParameters_RoundTrips()
    {
        var predictor = new FluxDoubleStreamPredictor<double>();

        var parameters = predictor.GetParameters();
        Assert.True(parameters.Length > 0);

        predictor.SetParameters(parameters);
        var retrieved = predictor.GetParameters();
        Assert.Equal(parameters.Length, retrieved.Length);
    }

    [Fact(Timeout = 120000)]
    public async Task SiTPredictor_GetSetParameters_RoundTrips()
    {
        var predictor = new SiTPredictor<double>();

        var parameters = predictor.GetParameters();
        Assert.True(parameters.Length > 0);

        predictor.SetParameters(parameters);
        var retrieved = predictor.GetParameters();
        Assert.Equal(parameters.Length, retrieved.Length);
    }

    #endregion
}
