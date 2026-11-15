using AiDotNet.Enums;
using AiDotNet.Finance.Forecasting.Neural;
using AiDotNet.Finance.Forecasting.Transformers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;

/// <summary>The time-series forecasting rows (informer, autoformer, deepar, nbeats).</summary>
/// <remarks>
/// <para>Each row wraps the shipped tensor-interface forecaster (<c>AiDotNet.Finance.Forecasting.*</c>, a
/// <see cref="NeuralNetworkBase{T}"/>): its own <c>Train(input, target)</c> step, its own <c>Predict</c>. The
/// <c>AiDotNet.TimeSeries.*Model</c> classes share these networks (InformerNetwork, AutoformerNetwork, DeepARNetwork)
/// but train on one whole series per <c>Train</c> call, re-windowing, re-normalizing and building a fresh optimizer each
/// call, so they have no per-batch step to time against a PyTorch step.</para>
/// <para>Shape: the ETT protocol the transformer papers report: a 96-step lookback, a 24-step horizon, 7 channels
/// (N-BEATS is univariate, one channel). Every row uses its paper's default sizes. The data is a noisy daily cycle
/// <c>1 + 0.5 sin(2 pi (t + phase) / 24) + 0.1 N(0, 1)</c>, a random phase per series, positive as DeepAR's
/// <c>1 + mean(z)</c> scale requires; benchmark.py's <c>forecast_batch</c> draws the same distribution.</para>
/// <para>Optimizer: each paper's Adam (AdamW with no weight decay) with the global gradient norm clipped to 1, as the
/// harness clips every PyTorch row. benchmark.py's twins are layer-for-layer: each report records both parameter counts.</para>
/// </remarks>
internal enum TimeSeriesFamily
{
    /// <summary>Informer (Zhou et al. 2021) at the paper's ETTh1 setting: d 512, 8 heads, 2 encoder layers with
    /// distilling, 1 decoder layer, d_ff 2048, ProbSparse factor 5, 48 start tokens, dropout 0.05; MSE.</summary>
    Informer,

    /// <summary>Autoformer (Wu et al. 2021) at the paper's setting: d 512, 8 heads, 2 encoder and 1 decoder layer, d_ff
    /// 2048, moving average 25, auto-correlation factor 1, 48 start tokens, dropout 0.05; MSE.</summary>
    Autoformer,

    /// <summary>DeepAR (Salinas et al. 2020): 3 LSTM layers of 40, dropout 0.1, a Gaussian head, a 20-wide embedding
    /// of each of the 7 series; the training loss is the Gaussian negative log-likelihood of every step of the
    /// 96 + 24 window, and a forecast is the median of 200 sample paths.</summary>
    DeepAR,

    /// <summary>N-BEATS-G (Oreshkin et al. 2020): 30 generic blocks of 4 ReLU layers of 512 with backcast and forecast
    /// heads, plus the final forecast projection AiDotNet's NBEATSFinance adds; univariate, MSE.</summary>
    NBeats,
}

internal sealed class AiDotNetTimeSeriesModel : IBenchmarkModel
{
    internal const int Lookback = 96;
    internal const int Horizon = 24;
    internal const int Channels = 7;
    internal const int LabelLength = 48;

    private readonly NeuralNetworkBase<float> _model;
    private readonly Random _random;
    private readonly int _channels;
    private Tensor<float> _input = Tensor<float>.Empty();
    private Tensor<float> _label = Tensor<float>.Empty();
    private bool _trained;

    public AiDotNetTimeSeriesModel(int seed, TimeSeriesFamily family)
    {
        _random = new Random(seed);
        _channels = family == TimeSeriesFamily.NBeats ? 1 : Channels;
        // Paper learning rates: Informer and Autoformer 1e-4, DeepAR and N-BEATS 1e-3.
        double learningRate = family is TimeSeriesFamily.Informer or TimeSeriesFamily.Autoformer ? 1e-4 : 1e-3;
        var optimizer = new AdamWOptimizer<float, Tensor<float>, Tensor<float>>(null,
            new AdamWOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                InitialLearningRate = learningRate, Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-8, WeightDecay = 0.0,
                EnableGradientClipping = true, MaxGradientNorm = 1.0,
            });
        // A [96, channels] grid: the forecasters read the feature count from the architecture's width.
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.TwoDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputHeight: Lookback, inputWidth: _channels, inputDepth: 1, outputSize: Horizon * _channels);

        _model = family switch
        {
            TimeSeriesFamily.Informer => new Informer<float>(architecture, new InformerOptions<float>
            {
                LookbackWindow = Lookback, LabelLength = LabelLength, ForecastHorizon = Horizon, EmbeddingDim = 512,
                NumAttentionHeads = 8, NumEncoderLayers = 2, NumDecoderLayers = 1, FeedForwardDim = 2048,
                SamplingFactor = 5, Attention = AiDotNet.TimeSeries.InformerAttention.ProbSparse, Distil = true,
                Mix = true, UseGelu = true, DropoutRate = 0.05, Seed = seed,
            }, optimizer),
            TimeSeriesFamily.Autoformer => new Autoformer<float>(architecture, new AutoformerOptions<float>
            {
                LookbackWindow = Lookback, LabelLength = LabelLength, ForecastHorizon = Horizon, EmbeddingDim = 512,
                NumAttentionHeads = 8, NumEncoderLayers = 2, NumDecoderLayers = 1, FeedForwardDim = 2048,
                MovingAverageKernel = 25, AutoCorrelationFactor = 1, UseGelu = true, DropoutRate = 0.05, Seed = seed,
            }, optimizer),
            TimeSeriesFamily.DeepAR => new DeepAR<float>(architecture, new DeepAROptions<float>
            {
                LookbackWindow = Lookback, ForecastHorizon = Horizon, HiddenSize = 40, NumLayers = 3, DropoutRate = 0.1,
                LikelihoodType = "Gaussian", NumSamples = 200, EmbeddingDimension = 20,
                ScaleHandling = DeepARScaleHandling.PaperMean, Seed = seed,
            }, optimizer),
            TimeSeriesFamily.NBeats => new NBEATSFinance<float>(architecture, new NBEATSModelOptions<float>
            {
                LookbackWindow = Lookback, ForecastHorizon = Horizon, NumStacks = 30, NumBlocksPerStack = 1,
                HiddenLayerSize = 512, NumHiddenLayers = 4,
                // NBEATSFinance always builds generic blocks; say so rather than leave the default (true) unread.
                UseInterpretableBasis = false, Seed = seed,
            }, optimizer),
            _ => throw new ArgumentOutOfRangeException(nameof(family)),
        };

        LoadSyntheticBatch(1);
        Forward();
        ParameterCount = _model.TrainableParameterCount;
    }

    public long ParameterCount { get; }

    public (Tensor<float> Input, Tensor<float> Label) CurrentBatch
    {
        get => (_input, _label);
        set => (_input, _label) = value;
    }

    /// <summary>A [B, 96, C] history and its [B, 24, C] continuation (N-BEATS: [B, 96, 1] and [B, 24]).</summary>
    public void LoadSyntheticBatch(int batchSize)
    {
        int steps = Lookback + Horizon;
        var history = new float[batchSize * Lookback * _channels];
        var future = new float[batchSize * Horizon * _channels];
        for (int b = 0; b < batchSize; b++)
            for (int c = 0; c < _channels; c++)
            {
                double phase = _random.NextDouble() * 24.0;
                for (int t = 0; t < steps; t++)
                {
                    double u1 = 1.0 - _random.NextDouble(), u2 = _random.NextDouble();
                    double noise = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                    float value = (float)(1.0 + 0.5 * Math.Sin(2.0 * Math.PI * (t + phase) / 24.0) + 0.1 * noise);
                    if (t < Lookback) history[(b * Lookback + t) * _channels + c] = value;
                    else future[(b * Horizon + t - Lookback) * _channels + c] = value;
                }
            }
        // NBEATSFinance's input contract is [B, L, 1]; its forecast is [B, H].
        _input = new Tensor<float>(history, [batchSize, Lookback, _channels]);
        _label = _channels == 1
            ? new Tensor<float>(future, [batchSize, Horizon])
            : new Tensor<float>(future, [batchSize, Horizon, _channels]);
    }

    public void Forward() => _ = _model.Predict(_input);

    public void Backward()
    {
        // The forecaster's Train is the whole step: forward, its loss (MSE, or DeepAR's likelihood), backward, update.
        _model.Train(_input, _label);
        _trained = true;
    }

    public void Step()
    {
        // The optimizer step ran inside Backward's Train call.
    }

    public double? LastLoss => _trained ? _model.GetLastLoss() : null;
}
