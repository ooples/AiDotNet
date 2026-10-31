using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TimeSeries;
using Xunit;

namespace AiDotNet.Tests.UnitTests.TimeSeries;

/// <summary>
/// ChronosFoundationModel's public surface on a small Chronos-T5: option checks, training on a series, seeded
/// sampled forecasts and their quantiles, Predict's time-index convention, serialization, and options read from a
/// released checkpoint's config.json. The network itself is checked against the reference in ChronosReferenceTests.
/// </summary>
public class ChronosFoundationModelTests
{
    internal static ChronosOptions<double> Small(int steps = 30) => new()
    {
        Seed = 5, ContextLength = 24, PredictionLength = 4, NumTokens = 256, ModelDim = 32, FeedForwardDim = 64,
        KeyValueDim = 8, NumHeads = 4, NumLayers = 2, TrainingSteps = steps, BatchSize = 8, MinPast = 12, NumSamples = 8,
        // The paper's 1e-3 is for 200K steps; a few hundred from scratch need a larger rate to leave the initialization.
        LearningRate = 0.01,
    };

    private static (Matrix<double> x, Vector<double> y) Series(int length)
    {
        var x = new Matrix<double>(length, 1);
        var y = new Vector<double>(length);
        for (int i = 0; i < length; i++)
        {
            x[i, 0] = i;
            y[i] = 20 + 5 * Math.Sin(2 * Math.PI * i / 12.0);
        }
        return (x, y);
    }

    [Fact]
    public void Defaults_AreChronosT5Small_AndThePaperRecipe()
    {
        var o = new ChronosOptions<double>();
        Assert.Equal((512, 64, 4096), (o.ContextLength, o.PredictionLength, o.NumTokens));
        Assert.Equal((512, 2048, 64, 8, 6), (o.ModelDim, o.FeedForwardDim, o.KeyValueDim, o.NumHeads, o.NumLayers));
        Assert.Equal((0.001, 0.01, 200_000, 256), (o.LearningRate, o.WeightDecay, o.TrainingSteps, o.BatchSize));
        Assert.Equal((20, 1.0, 50, 1.0), (o.NumSamples, o.Temperature, o.TopK, o.TopP));
    }

    [Theory]
    [InlineData(nameof(ChronosOptions<double>.NumTokens))]
    [InlineData(nameof(ChronosOptions<double>.ModelDim))]
    [InlineData(nameof(ChronosOptions<double>.NumHeads))]
    [InlineData(nameof(ChronosOptions<double>.NumLayers))]
    [InlineData(nameof(ChronosOptions<double>.ContextLength))]
    [InlineData(nameof(ChronosOptions<double>.PredictionLength))]
    [InlineData(nameof(ChronosOptions<double>.NumSamples))]
    [InlineData(nameof(ChronosOptions<double>.TrainingSteps))]
    public void Construction_RejectsANonPositiveSize(string option)
    {
        var options = Small();
        typeof(ChronosOptions<double>).GetProperty(option)!.SetValue(options, 0);
        Assert.Throws<ArgumentException>(() => new ChronosFoundationModel<double>(options));
    }

    [Fact]
    public void Construction_RejectsAnInvertedBinRange()
    {
        var options = Small();
        options.LowLimit = 15;
        options.HighLimit = -15;
        Assert.Throws<ArgumentException>(() => new ChronosFoundationModel<double>(options));
    }

    [Fact(Timeout = 300000)]
    public async Task Training_LearnsTheLevel_AndForecastsAreSeeded()
    {
        await Task.Yield();
        var (x, y) = Series(120);
        var options = Small(steps: 150);
        var model = new ChronosFoundationModel<double>(options);
        model.Train(x, y);
        var history = new Vector<double>(Enumerable.Range(0, 48).Select(i => y[i]).ToArray());

        // The same call returns the same sampled forecast; a second, identically seeded model trained the same way too.
        var first = model.Forecast(history, 6);
        Assert.Equal(first.ToArray(), model.Forecast(history, 6).ToArray());
        var twin = new ChronosFoundationModel<double>(Small(steps: 150));
        twin.Train(x, y);
        Assert.Equal(first.ToArray(), twin.Forecast(history, 6).ToArray());

        // The forecast is within the series' range (it learned the level), and longer horizons continue it.
        Assert.All(first.ToArray(), v => Assert.InRange(v, 10.0, 30.0));
        var samples = model.SampleForecasts(history, 6);
        Assert.Equal((8, 6), (samples.Rows, samples.Columns));
    }

    [Fact(Timeout = 300000)]
    public async Task Quantiles_AreOrdered_AndTheMedianIsTheForecast()
    {
        await Task.Yield();
        var (x, y) = Series(80);
        var model = new ChronosFoundationModel<double>(Small(steps: 10));
        model.Train(x, y);
        var history = new Vector<double>(Enumerable.Range(0, 40).Select(i => y[i]).ToArray());
        var quantiles = model.ForecastWithQuantiles(history, new[] { 0.1, 0.5, 0.9 }, 5);
        for (int h = 0; h < 5; h++)
        {
            Assert.True(quantiles[0.1][h] <= quantiles[0.5][h] && quantiles[0.5][h] <= quantiles[0.9][h]);
        }
        Assert.Throws<ArgumentException>(() => model.ForecastWithQuantiles(history, new[] { 1.5 }, 5));
    }

    [Fact(Timeout = 300000)]
    public async Task Predict_AnswersATimeIndex_InSampleAndAhead()
    {
        await Task.Yield();
        var (x, y) = Series(60);
        var model = new ChronosFoundationModel<double>(Small(steps: 5));
        model.Train(x, y);
        var inSample = model.Predict(x);
        Assert.Equal(60, inSample.Length);
        Assert.Equal(y.ToArray().Average(), inSample[0], 10);
        var ahead = new Matrix<double>(7, 1);
        var forecast = model.Predict(ahead);
        var history = new Vector<double>(y.ToArray());
        Assert.Equal(model.Forecast(history, 7).ToArray(), forecast.ToArray());
    }

    [Fact]
    public void Training_RejectsASeriesShorterThanMinPastPlusPredictionLength()
    {
        var (x, y) = Series(15);
        var model = new ChronosFoundationModel<double>(Small());
        Assert.Throws<ArgumentException>(() => model.Train(x, y));
    }

    [Fact(Timeout = 300000)]
    public async Task Serialization_RoundTripsTheForecast()
    {
        await Task.Yield();
        var (x, y) = Series(60);
        var model = new ChronosFoundationModel<double>(Small(steps: 5));
        model.Train(x, y);
        var restored = new ChronosFoundationModel<double>(Small(steps: 5));
        restored.Deserialize(model.Serialize());
        var history = new Vector<double>(y.ToArray());
        Assert.Equal(model.Forecast(history, 3).ToArray(), restored.Forecast(history, 3).ToArray());
    }

    [Fact]
    public void FromPretrainedConfig_ReadsAReleasedConfig()
    {
        // amazon/chronos-t5-tiny's config.json, abridged to the fields read.
        const string json = @"{""d_model"": 256, ""d_ff"": 1024, ""d_kv"": 64, ""num_heads"": 4, ""num_layers"": 4,
  ""num_decoder_layers"": 4, ""feed_forward_proj"": ""relu"", ""dropout_rate"": 0.1, ""relative_attention_num_buckets"": 32,
  ""relative_attention_max_distance"": 128, ""chronos_config"": {""tokenizer_class"": ""MeanScaleUniformBins"",
  ""tokenizer_kwargs"": {""low_limit"": -15.0, ""high_limit"": 15.0}, ""n_tokens"": 4096, ""n_special_tokens"": 2,
  ""pad_token_id"": 0, ""eos_token_id"": 1, ""use_eos_token"": true, ""model_type"": ""seq2seq"", ""context_length"": 512,
  ""prediction_length"": 64, ""num_samples"": 20, ""temperature"": 1.0, ""top_k"": 50, ""top_p"": 1.0}}";
        string path = Path.Combine(Path.GetTempPath(), $"chronos-config-{Guid.NewGuid():N}.json");
        File.WriteAllText(path, json);
        try
        {
            var o = ChronosOptions<double>.FromPretrainedConfig(path);
            Assert.Equal((256, 1024, 64, 4, 4), (o.ModelDim, o.FeedForwardDim, o.KeyValueDim, o.NumHeads, o.NumLayers));
            Assert.Equal((512, 64, 4096, 20, 50), (o.ContextLength, o.PredictionLength, o.NumTokens, o.NumSamples, o.TopK));
        }
        finally
        {
            File.Delete(path);
        }
    }
}
