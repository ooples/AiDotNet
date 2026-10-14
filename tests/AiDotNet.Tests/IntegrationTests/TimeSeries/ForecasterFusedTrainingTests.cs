using AiDotNet.Models.Options;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TimeSeries;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.TimeSeries;

/// <summary>
/// Every tape-trained forecaster steps through the shared fused path (#1804). With compilation on, every step must
/// actually run fused, not fall back to the eager tape without saying so, and it must take the step the eager tape
/// takes.
/// </summary>
[Collection("FusedOptimizerGlobalState")]
public class ForecasterFusedTrainingTests
{
    // The fused plan takes Adam's beta1, beta2 and epsilon as float, so double training drifts from the eager step
    // by a few 1e-8 of the parameter scale per step. A model that trained a different step differs by far more.
    private const double RelativeTolerance = 1e-5;

    public static TheoryData<string> Models => new()
    {
        "DeepAR", "NBEATS", "NHiTS", "TFT", "Informer",
    };

    private static TimeSeriesModelBase<double> Create(string model, int epochs, int batch) => model switch
    {
        "DeepAR" => new DeepARModel<double>(new DeepAROptions<double>
            { LookbackWindow = 8, ForecastHorizon = 2, HiddenSize = 8, NumLayers = 1, Epochs = epochs, BatchSize = batch, Seed = 5 }),
        "NBEATS" => new NBEATSModel<double>(new NBEATSModelOptions<double>
            { NumStacks = 2, NumBlocksPerStack = 1, LookbackWindow = 8, ForecastHorizon = 2, HiddenLayerSize = 8, NumHiddenLayers = 1, Epochs = epochs, BatchSize = batch, Seed = 5 }),
        "NHiTS" => new NHiTSModel<double>(new NHiTSOptions<double>
            { LookbackWindow = 8, ForecastHorizon = 2, HiddenLayerSize = 8, Epochs = epochs, BatchSize = batch, Seed = 5 }),
        "TFT" => new TemporalFusionTransformer<double>(new TemporalFusionTransformerOptions<double>
            { LookbackWindow = 8, ForecastHorizon = 2, HiddenSize = 8, NumAttentionHeads = 2, Epochs = epochs, BatchSize = batch, Seed = 5 }),
        "Informer" => new InformerModel<double>(new InformerOptions<double>
            { LookbackWindow = 8, ForecastHorizon = 2, EmbeddingDim = 8, NumAttentionHeads = 2, Epochs = epochs, BatchSize = batch, Seed = 5 }),
        _ => throw new ArgumentOutOfRangeException(nameof(model)),
    };

    private sealed record RunResult(Vector<double> Parameters, int FusedSteps, int EagerSteps, Vector<double> Initial);

    private static RunResult Train(string model, bool compile, int epochs = 2, int batch = 8, double nudge = 0.0)
    {
        const int n = 64;
        var x = new Matrix<double>(n, 1);
        var y = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            x[i, 0] = i;
            y[i] = Math.Sin(i * 0.3) + 0.01 * i;
        }

        var original = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = compile });
            CompiledTapeTrainingStep<double>.Invalidate();
            var m = Create(model, epochs, batch);
            if (nudge != 0.0) { var p = m.GetParameters(); p[0] += nudge; m.SetParameters(p); }
            var initial = m.GetParameters();
            m.Train(x, y);
            return new RunResult(m.GetParameters(), m.FusedTrainingStepCount, m.EagerTrainingStepCount, initial);
        }
        finally
        {
            TensorCodecOptions.SetCurrent(original);
            CompiledTapeTrainingStep<double>.Invalidate();
        }
    }

    private static double MaxAbsDiff(Vector<double> a, Vector<double> b)
    {
        double worst = 0.0;
        for (int i = 0; i < a.Length; i++) worst = Math.Max(worst, Math.Abs(a[i] - b[i]));
        return worst;
    }

    [Theory(Timeout = 300000)]
    [MemberData(nameof(Models))]
    public async Task CompiledTraining_RunsEveryStepFused_AndTakesTheEagerStep(string model)
    {
        await Task.Yield();
        var eager = Train(model, compile: false);
        var fused = Train(model, compile: true);

        Assert.Equal(0, eager.FusedSteps);
        Assert.True(eager.EagerSteps > 0, $"{model} ran no training step.");
        Assert.Equal(0, fused.EagerSteps);
        Assert.Equal(eager.EagerSteps, fused.FusedSteps);

        Assert.Equal(eager.Parameters.Length, fused.Parameters.Length);
        double scale = 0.0;
        for (int i = 0; i < eager.Parameters.Length; i++) scale = Math.Max(scale, Math.Abs(eager.Parameters[i]));
        double worst = MaxAbsDiff(eager.Parameters, fused.Parameters);
        Assert.True(worst <= RelativeTolerance * Math.Max(scale, 1.0),
            $"{model}: fused training diverged from eager by {worst:G6} (parameter scale {scale:G6}).");
    }
}