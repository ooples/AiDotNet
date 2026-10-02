using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TimeSeries;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.TimeSeries;

/// <summary>
/// The deep time-series models honour <c>Options.Seed</c>. They used hard-coded seeds and ignored it, so runs with
/// different seeds were the same run and seed-to-seed variance could not be measured.
/// </summary>
public class TimeSeriesSeedTests
{
    public static TheoryData<string> Models => new()
    {
        "DeepAR", "NBEATS", "NHiTS", "TFT", "Informer", "Autoformer", "TiDE",
    };

    private static Vector<double> InitialParameters(string model, int? seed) => model switch
    {
        "DeepAR" => new DeepARModel<double>(new DeepAROptions<double>
            { LookbackWindow = 8, ForecastHorizon = 2, HiddenSize = 8, NumLayers = 1, Seed = seed }).GetParameters(),
        "NBEATS" => new NBEATSModel<double>(new NBEATSModelOptions<double>
            { NumStacks = 2, NumBlocksPerStack = 1, LookbackWindow = 8, ForecastHorizon = 2, HiddenLayerSize = 8, NumHiddenLayers = 1, Seed = seed }).GetParameters(),
        "NHiTS" => new NHiTSModel<double>(new NHiTSOptions<double>
            { LookbackWindow = 8, ForecastHorizon = 2, HiddenLayerSize = 8, Seed = seed }).GetParameters(),
        "TFT" => new TemporalFusionTransformer<double>(new TemporalFusionTransformerOptions<double>
            { LookbackWindow = 8, ForecastHorizon = 2, HiddenSize = 8, NumAttentionHeads = 2, Seed = seed }).GetParameters(),
        "Informer" => new InformerModel<double>(new InformerOptions<double>
            { LookbackWindow = 8, ForecastHorizon = 2, EmbeddingDim = 8, NumAttentionHeads = 2, Seed = seed }).GetParameters(),
        "Autoformer" => new AutoformerModel<double>(new AutoformerOptions<double>
            { LookbackWindow = 8, ForecastHorizon = 2, EmbeddingDim = 8, NumAttentionHeads = 2, Seed = seed }).GetParameters(),
        "TiDE" => new TiDEModel<double>(new TiDEOptions<double>
            { LookbackWindow = 8, ForecastHorizon = 2, HiddenSize = 8, Seed = seed }).GetParameters(),
        _ => throw new ArgumentOutOfRangeException(nameof(model)),
    };

    private static bool Same(Vector<double> a, Vector<double> b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
        {
            if (a[i] != b[i]) return false;
        }

        return true;
    }

    [Theory]
    [MemberData(nameof(Models))]
    public void SameSeed_GivesIdenticalInitialWeights(string model)
    {
        var a = InitialParameters(model, 7);
        Assert.True(a.Length > 0, $"{model} exposed no parameters");
        Assert.True(Same(a, InitialParameters(model, 7)), $"{model} is not reproducible under one seed");
    }

    [Theory]
    [MemberData(nameof(Models))]
    public void DifferentSeeds_GiveDifferentInitialWeights(string model)
    {
        Assert.False(Same(InitialParameters(model, 1), InitialParameters(model, 2)), $"{model} ignores Options.Seed");
    }

    [Theory]
    [MemberData(nameof(Models))]
    public void NoSeed_StaysReproducible(string model)
    {
        Assert.True(Same(InitialParameters(model, null), InitialParameters(model, null)));
    }

    /// <summary>DLinear and NLinear start from the 1/L averaging weights by design; the seed drives their training
    /// shuffle, so it shows up in the TRAINED weights.</summary>
    [Theory]
    [InlineData("DLinear")]
    [InlineData("NLinear")]
    public void LinearModels_SeedChangesTheTrainedWeights(string model)
    {
        Vector<double> Trained(int seed)
        {
            const int n = 80;
            var x = new Matrix<double>(n, 1);
            var y = new Vector<double>(n);
            for (int i = 0; i < n; i++)
            {
                x[i, 0] = i;
                y[i] = Math.Sin(i * 0.3) + 0.01 * i;
            }

            var m = model == "DLinear"
                ? (TimeSeriesModelBase<double>)new DLinearModel<double>(new DLinearOptions<double>
                    { LookbackWindow = 8, ForecastHorizon = 2, Epochs = 2, BatchSize = 4, Seed = seed })
                : new NLinearModel<double>(new NLinearOptions<double>
                    { LookbackWindow = 8, ForecastHorizon = 2, Epochs = 2, BatchSize = 4, Seed = seed });
            m.Train(x, y);
            return m.GetParameters();
        }

        Assert.True(Same(Trained(3), Trained(3)), $"{model} training is not reproducible under one seed");
        Assert.False(Same(Trained(3), Trained(4)), $"{model} training ignores Options.Seed");
    }

    [Fact]
    public void NBeatsBlocks_DoNotShareInitialWeights()
    {
        // Two stacks of one generic block each: the two blocks have the same shape. With one shared seed their
        // initial weights were identical.
        var p = InitialParameters("NBEATS", seed: null);
        Assert.Equal(0, p.Length % 2);
        var half = p.Length / 2;
        var differs = false;
        for (int i = 0; i < half && !differs; i++)
        {
            differs = p[i] != p[half + i];
        }

        Assert.True(differs, "every N-BEATS block started from the same weights");
    }
}
