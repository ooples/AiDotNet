using AiDotNet.Diffusion.Conditioning;
using AiDotNet.Diffusion.NoisePredictors;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion.Models;

/// <summary>
/// Contract tests for Phase 1 conditioning infrastructure: text conditioners and noise predictors.
/// </summary>
public class NewConditionerContractTests
{
    #region Text Conditioner Constructor Tests

    [Fact]
    public void SigLIPTextConditioner_DefaultConstructor_CreatesValidConditioner()
    {
        var conditioner = new SigLIPTextConditioner<double>();

        Assert.NotNull(conditioner);
        Assert.True(conditioner.ProducesPooledOutput);
        Assert.True(conditioner.EmbeddingDimension > 0);
    }

    [Fact]
    public void SigLIP2TextConditioner_DefaultConstructor_CreatesValidConditioner()
    {
        var conditioner = new SigLIP2TextConditioner<double>();

        Assert.NotNull(conditioner);
        Assert.True(conditioner.ProducesPooledOutput);
        Assert.True(conditioner.EmbeddingDimension > 0);
    }

    [Fact]
    public void DistilledT5TextConditioner_DefaultConstructor_CreatesValidConditioner()
    {
        var conditioner = new DistilledT5TextConditioner<double>();

        Assert.NotNull(conditioner);
        Assert.True(conditioner.EmbeddingDimension > 0);
    }

    [Fact]
    public void GemmaTextConditioner_DefaultConstructor_CreatesValidConditioner()
    {
        var conditioner = new GemmaTextConditioner<double>();

        Assert.NotNull(conditioner);
        Assert.True(conditioner.EmbeddingDimension > 0);
    }

    [Fact]
    public void Qwen2TextConditioner_DefaultConstructor_CreatesValidConditioner()
    {
        var conditioner = new Qwen2TextConditioner<double>();

        Assert.NotNull(conditioner);
        Assert.True(conditioner.EmbeddingDimension > 0);
    }

    #endregion

    #region Noise Predictor Constructor Tests

    [Fact]
    public void MMDiTXNoisePredictor_DefaultConstructor_CreatesValidPredictor()
    {
        var predictor = new MMDiTXNoisePredictor<double>();

        Assert.NotNull(predictor);
        Assert.True(predictor.ParameterCount > 0);
    }

    [Fact]
    public void FluxDoubleStreamPredictor_DefaultConstructor_CreatesValidPredictor()
    {
        var predictor = new FluxDoubleStreamPredictor<double>();

        Assert.NotNull(predictor);
        Assert.True(predictor.ParameterCount > 0);
    }

    [Fact]
    public void SiTPredictor_DefaultConstructor_CreatesValidPredictor()
    {
        var predictor = new SiTPredictor<double>();

        Assert.NotNull(predictor);
        Assert.True(predictor.ParameterCount > 0);
    }

    [Fact]
    public void EMMDiTPredictor_DefaultConstructor_CreatesValidPredictor()
    {
        var predictor = new EMMDiTPredictor<double>();

        Assert.NotNull(predictor);
        Assert.True(predictor.ParameterCount > 0);
    }

    #endregion

    #region Parameterizable Contract Tests

    [Fact]
    public void MMDiTXNoisePredictor_GetSetParameters_RoundTrips()
    {
        var predictor = new MMDiTXNoisePredictor<double>();

        var parameters = predictor.GetParameters();
        Assert.True(parameters.Length > 0);

        predictor.SetParameters(parameters);
        var retrieved = predictor.GetParameters();
        Assert.Equal(parameters.Length, retrieved.Length);
    }

    [Fact]
    public void FluxDoubleStreamPredictor_GetSetParameters_RoundTrips()
    {
        var predictor = new FluxDoubleStreamPredictor<double>();

        var parameters = predictor.GetParameters();
        Assert.True(parameters.Length > 0);

        predictor.SetParameters(parameters);
        var retrieved = predictor.GetParameters();
        Assert.Equal(parameters.Length, retrieved.Length);
    }

    [Fact]
    public void SiTPredictor_GetSetParameters_RoundTrips()
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
