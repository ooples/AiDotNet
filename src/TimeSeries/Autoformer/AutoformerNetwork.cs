using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TimeSeries;

/// <summary>The sizes of an Autoformer network (the reference's <c>Model</c> configuration).</summary>
internal sealed record AutoformerConfiguration(
    int EncoderInputs, int DecoderInputs, int Outputs, int ModelDim, int Heads, int EncoderLayers, int DecoderLayers,
    int FeedForwardDim, int MovingAverage, int Factor, double Dropout, bool Gelu, int TimeFeatures, int LabelLength,
    int PredictionLength);

/// <summary>
/// The Autoformer network (Wu et al., NeurIPS 2021), as the official code (thuml/Autoformer, <c>models/Autoformer.py</c>,
/// <c>layers/</c>) builds it: series decomposition (a replicate-padded moving average) splits the input into the
/// seasonal and trend parts that open the decoder; the encoder and decoder replace self-attention with the
/// Auto-Correlation mechanism and decompose after every block, accumulating the decoder's trend through a circular
/// convolution; the forecast is the trend plus the projected seasonal part.
/// </summary>
/// <remarks>
/// <para>Shared by <see cref="AutoformerModel{T}"/> and <see cref="AiDotNet.Finance.Forecasting.Transformers.Autoformer{T}"/>.
/// Its layers are added to the owner's list, so they train and save with the owner.</para>
/// <para>Auto-Correlation (§3.2) estimates the autocorrelation of every lag τ, R(τ) = Σ_t q_{t+τ}·k_t (circular), as
/// the reference does with an FFT, and only its mean over heads and channels selects and weights the delays. That
/// mean is the wrapped-diagonal sum of Q·Kᵀ divided by the width, computed here exactly. The k = ⌊c·ln L⌋ strongest
/// lags aggregate the values rolled by each lag, weighted by the softmax of their correlations: in training the lags
/// shared by the batch (the reference's <c>time_delay_agg_training</c>), at inference each series' own
/// (<c>time_delay_agg_inference</c>).</para>
/// </remarks>
internal sealed class AutoformerNetwork<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public AutoformerNetwork(IEngine engine, List<LayerBase<T>> layers, AutoformerConfiguration configuration)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        if (c.ModelDim % c.Heads != 0)
            throw new ArgumentException($"The model width {c.ModelDim} must be divisible by the {c.Heads} heads.");
        if (c.MovingAverage <= 0 || c.MovingAverage % 2 == 0)
            throw new ArgumentException($"The moving-average window ({c.MovingAverage}) must be a positive odd number.");
        if (c.LabelLength < 0 || c.PredictionLength <= 0 || c.Factor <= 0)
            throw new ArgumentException("The start-token length must be non-negative; the prediction length and factor positive.");
        TLayer Own<TLayer>(TLayer layer) where TLayer : LayerBase<T>
        {
            layers.Add(layer);
            return layer;
        }
        DenseLayer<T> Linear(int outputs) => Own(new DenseLayer<T>(outputs, (IActivationFunction<T>)new IdentityActivation<T>()));
        BiasFreeLinearLayer<T> Plain(int inputs, int outputs) => Own(new BiasFreeLinearLayer<T>(inputs, outputs));
        LayerNormalizationLayer<T> Norm() => Own(new LayerNormalizationLayer<T>(c.ModelDim, 1e-5));
        Correlation CorrelationLayer() => new(Linear(c.ModelDim), Linear(c.ModelDim), Linear(c.ModelDim), Linear(c.ModelDim));

        EncoderEmbedding = new Embedding(Plain(3 * c.EncoderInputs, c.ModelDim), c.TimeFeatures > 0 ? Plain(c.TimeFeatures, c.ModelDim) : null);
        DecoderEmbedding = new Embedding(Plain(3 * c.DecoderInputs, c.ModelDim), c.TimeFeatures > 0 ? Plain(c.TimeFeatures, c.ModelDim) : null);
        for (int i = 0; i < c.EncoderLayers; i++)
            EncoderLayers.Add(new EncoderLayer(CorrelationLayer(), Plain(c.ModelDim, c.FeedForwardDim), Plain(c.FeedForwardDim, c.ModelDim)));
        EncoderNorm = Norm();
        for (int i = 0; i < c.DecoderLayers; i++)
            DecoderLayers.Add(new DecoderLayer(CorrelationLayer(), CorrelationLayer(), Plain(c.ModelDim, c.FeedForwardDim),
                Plain(c.FeedForwardDim, c.ModelDim), Plain(3 * c.ModelDim, c.Outputs)));
        DecoderNorm = Norm();
        Projection = Linear(c.Outputs);
    }

    /// <summary>A <c>DataEmbedding_wo_pos</c>: the bias-free circular-conv token embedding (a linear map of
    /// [x_{t−1}, x_t, x_{t+1}]) and the optional bias-free time-feature embedding; no position embedding.</summary>
    internal sealed record Embedding(BiasFreeLinearLayer<T> Token, BiasFreeLinearLayer<T>? Time);

    /// <summary>An <c>AutoCorrelationLayer</c>'s query, key, value and output projections.</summary>
    internal sealed record Correlation(DenseLayer<T> Query, DenseLayer<T> Key, DenseLayer<T> Value, DenseLayer<T> Output);

    /// <summary>An <c>EncoderLayer</c>: Auto-Correlation and the bias-free kernel-1 convolution feed-forward.</summary>
    internal sealed record EncoderLayer(Correlation SelfCorrelation, BiasFreeLinearLayer<T> Conv1, BiasFreeLinearLayer<T> Conv2);

    /// <summary>A <c>DecoderLayer</c>: self and cross Auto-Correlation, the feed-forward, and the bias-free circular
    /// convolution (k = 3) that projects the accumulated trend.</summary>
    internal sealed record DecoderLayer(Correlation SelfCorrelation, Correlation CrossCorrelation, BiasFreeLinearLayer<T> Conv1,
        BiasFreeLinearLayer<T> Conv2, BiasFreeLinearLayer<T> TrendProjection);

    public AutoformerConfiguration Configuration { get; }
    public Embedding EncoderEmbedding { get; }
    public Embedding DecoderEmbedding { get; }
    public List<EncoderLayer> EncoderLayers { get; } = new();
    public LayerNormalizationLayer<T> EncoderNorm { get; }
    public List<DecoderLayer> DecoderLayers { get; } = new();
    public LayerNormalizationLayer<T> DecoderNorm { get; }
    public DenseLayer<T> Projection { get; }

    // ------------------------------------------------------------------ the forward

    /// <summary>
    /// The forecast <c>[B, PredictionLength, Outputs]</c> of the encoder input <c>[B, L, EncoderInputs]</c>, with the
    /// time features <c>[B, L, F]</c> and <c>[B, LabelLength + PredictionLength, F]</c> when the network has them. The
    /// decoder input is built from the decomposition of the encoder input, as in the reference.
    /// </summary>
    public Tensor<T> Forward(Tensor<T> encoderInput, Tensor<T>? encoderMarks, Tensor<T>? decoderMarks, bool training, Random random)
    {
        var c = Configuration;
        int batch = encoderInput.Shape[0], length = encoderInput.Shape[1], channels = encoderInput.Shape[2];
        if (c.LabelLength > length)
            throw new ArgumentException($"The start tokens ({c.LabelLength}) are longer than the input ({length}).");

        // Decoder initialization: the last LabelLength steps of the seasonal part then zeros, and of the trend part
        // then the input's mean.
        var (seasonal, trend) = Decompose(encoderInput);
        var mean = _engine.TensorTile(_engine.ReduceMean(encoderInput, new[] { 1 }, keepDims: true), new[] { 1, c.PredictionLength, 1 });
        Tensor<T> Last(Tensor<T> x) => _engine.TensorSlice(x, new[] { 0, length - c.LabelLength, 0 }, new[] { batch, c.LabelLength, channels });
        var seasonalInit = _engine.TensorConcatenate(new[] { Last(seasonal), new Tensor<T>(new[] { batch, c.PredictionLength, channels }) }, 1);
        var trendInit = _engine.TensorConcatenate(new[] { Last(trend), mean }, 1);

        var x = Embed(EncoderEmbedding, encoderInput, encoderMarks, training, random);
        foreach (var layer in EncoderLayers) x = Encode(layer, x, training, random);
        var memory = SeasonalNorm(EncoderNorm, x);

        var y = Embed(DecoderEmbedding, seasonalInit, decoderMarks, training, random);
        var trendSum = trendInit;
        foreach (var layer in DecoderLayers)
        {
            var (next, residualTrend) = Decode(layer, y, memory, training, random);
            y = next;
            trendSum = _engine.TensorAdd(trendSum, residualTrend);
        }
        var output = _engine.TensorAdd(trendSum, Dense(Projection, SeasonalNorm(DecoderNorm, y)));
        int steps = output.Shape[1];
        return _engine.TensorSlice(output, new[] { 0, steps - c.PredictionLength, 0 }, new[] { batch, c.PredictionLength, c.Outputs });
    }

    // ------------------------------------------------------------------ building blocks

    private Tensor<T> Dense(LayerBase<T> layer, Tensor<T> x)
    {
        int batch = x.Shape[0], length = x.Shape[1];
        var y = layer.Forward(_engine.Reshape(x, new[] { batch * length, x.Shape[2] }));
        return _engine.Reshape(y, new[] { batch, length, y.Shape[1] });
    }

    private Tensor<T> Drop(Tensor<T> x, bool training, Random random)
    {
        double rate = Configuration.Dropout;
        if (!training || rate <= 0) return x;
        var mask = new Tensor<T>(x._shape);
        var keep = NumOps.FromDouble(1.0 / (1.0 - rate));
        for (int i = 0; i < mask.Length; i++) mask[i] = random.NextDouble() < rate ? NumOps.Zero : keep;
        return _engine.TensorMultiply(x, mask);
    }

    private Tensor<T> Activation(Tensor<T> x) => Configuration.Gelu ? ActivationHelper.ExactGelu(x, _engine) : _engine.ReLU(x);

    private Tensor<T> CircularNeighbours(Tensor<T> x)
    {
        int batch = x.Shape[0], length = x.Shape[1], channels = x.Shape[2];
        if (length == 1) return _engine.TensorConcatenate(new[] { x, x, x }, 2);
        Tensor<T> Steps(int from, int count) => _engine.TensorSlice(x, new[] { 0, from, 0 }, new[] { batch, count, channels });
        var previous = _engine.TensorConcatenate(new[] { Steps(length - 1, 1), Steps(0, length - 1) }, 1);
        var next = _engine.TensorConcatenate(new[] { Steps(1, length - 1), Steps(0, 1) }, 1);
        return _engine.TensorConcatenate(new[] { previous, x, next }, 2);
    }

    private Tensor<T> Embed(Embedding e, Tensor<T> x, Tensor<T>? marks, bool training, Random random)
    {
        var sum = Dense(e.Token, CircularNeighbours(x));
        if (e.Time is not null)
        {
            if (marks is null || marks.Shape[1] != x.Shape[1] || marks.Shape[2] != Configuration.TimeFeatures)
                throw new ArgumentException($"The network embeds {Configuration.TimeFeatures} time features per step; supply them for all {x.Shape[1]} steps.");
            sum = _engine.TensorAdd(sum, Dense(e.Time, marks));
        }
        return Drop(sum, training, random);
    }

    // series_decomp: the trend is the moving average with the ends replicated (kernel−1)/2 times; the seasonal part is
    // the rest. The average is one constant [L, L] matrix (replicated ends folded into the first and last columns).
    private (Tensor<T> Seasonal, Tensor<T> Trend) Decompose(Tensor<T> x)
    {
        int batch = x.Shape[0], length = x.Shape[1], k = Configuration.MovingAverage, half = (k - 1) / 2;
        var average = new Tensor<T>(new[] { 1, length, length });
        var weight = NumOps.FromDouble(1.0 / k);
        for (int t = 0; t < length; t++)
            for (int j = t - half; j <= t + half; j++)
            {
                int source = Math.Min(Math.Max(j, 0), length - 1);
                average[0, t, source] = NumOps.Add(average[0, t, source], weight);
            }
        var trend = _engine.BatchMatMul(_engine.TensorTile(average, new[] { batch, 1, 1 }), x);
        return (_engine.TensorSubtract(x, trend), trend);
    }

    // my_Layernorm: LayerNorm, then the mean over time subtracted.
    private Tensor<T> SeasonalNorm(LayerNormalizationLayer<T> norm, Tensor<T> x)
    {
        var normalized = norm.Forward(x);
        var bias = _engine.TensorTile(_engine.ReduceMean(normalized, new[] { 1 }, keepDims: true), new[] { 1, x.Shape[1], 1 });
        return _engine.TensorSubtract(normalized, bias);
    }

    // AutoCorrelationLayer over queries [B, L, D] and keys/values [B, S, D].
    private Tensor<T> Correlate(Correlation a, Tensor<T> queries, Tensor<T> keys, bool training)
    {
        int batch = queries.Shape[0], length = queries.Shape[1], s = keys.Shape[1], d = Configuration.ModelDim;
        var q = Dense(a.Query, queries);
        var k = Dense(a.Key, keys);
        var v = Dense(a.Value, keys);
        // Keys and values padded with zeros, or cut to their first L steps, to the queries' length.
        if (length > s)
        {
            var zeros = new Tensor<T>(new[] { batch, length - s, d });
            k = _engine.TensorConcatenate(new[] { k, zeros }, 1);
            v = _engine.TensorConcatenate(new[] { v, zeros }, 1);
        }
        else if (length < s)
        {
            k = _engine.TensorSlice(k, new[] { 0, 0, 0 }, new[] { batch, length, d });
            v = _engine.TensorSlice(v, new[] { 0, 0, 0 }, new[] { batch, length, d });
        }

        // The mean autocorrelation of every lag: R̄(τ) = Σ_t Σ_c q[(t+τ) mod L, c]·k[t, c] / D.
        var products = _engine.BatchMatMul(q, _engine.TensorPermute(k, new[] { 0, 2, 1 }));                    // [B, L, L]
        var diagonals = new int[length * length];
        for (int tau = 0; tau < length; tau++)
            for (int t = 0; t < length; t++) diagonals[tau * length + t] = ((t + tau) % length) * length + t;
        var picked = _engine.TensorGather(_engine.Reshape(products, new[] { batch, length * length }),
            new Tensor<int>(new[] { length * length }, new Vector<int>(diagonals)), axis: 1);
        var mean = _engine.TensorMultiplyScalar(
            _engine.ReduceSum(_engine.Reshape(picked, new[] { batch, length, length }), new[] { 2 }, keepDims: false),
            NumOps.FromDouble(1.0 / d));                                                                       // [B, L]

        // k = ⌊c·ln L⌋, which is zero for very short inputs: then nothing is aggregated, as in the reference.
        int top = Math.Min((int)(Configuration.Factor * Math.Log(length)), length);
        var host = mean.GetCpuData();
        Tensor<T> aggregated;
        if (top <= 0)
            aggregated = _engine.TensorMultiplyScalar(v, NumOps.Zero);
        else if (training)
        {
            // The lags strongest on average over the batch, shared by every series.
            var lags = Enumerable.Range(0, length)
                .OrderByDescending(tau => Enumerable.Range(0, batch).Sum(b => NumOps.ToDouble(host[b * length + tau])))
                .Take(top).ToArray();
            var weights = _engine.Softmax(_engine.TensorGather(mean, new Tensor<int>(new[] { top }, new Vector<int>(lags)), axis: 1), axis: 1);
            aggregated = new Tensor<T>(new[] { batch, length, d });
            for (int i = 0; i < top; i++)
            {
                var rolled = _engine.TensorRoll(v, new[] { -lags[i] }, new[] { 1 });
                var w = _engine.Reshape(_engine.TensorSlice(weights, new[] { 0, i }, new[] { batch, 1 }), new[] { batch, 1, 1 });
                aggregated = _engine.TensorAdd(aggregated, _engine.TensorMultiply(rolled, _engine.TensorTile(w, new[] { 1, length, d })));
            }
        }
        else
        {
            // Each series' own strongest lags.
            var rows = new Tensor<T>[batch];
            for (int b = 0; b < batch; b++)
            {
                int row = b;
                var lags = Enumerable.Range(0, length).OrderByDescending(tau => NumOps.ToDouble(host[row * length + tau])).Take(top).ToArray();
                var meanRow = _engine.TensorSlice(mean, new[] { b, 0 }, new[] { 1, length });
                var weights = _engine.Softmax(_engine.TensorGather(meanRow, new Tensor<int>(new[] { top }, new Vector<int>(lags)), axis: 1), axis: 1);
                var series = _engine.TensorSlice(v, new[] { b, 0, 0 }, new[] { 1, length, d });
                var sum = new Tensor<T>(new[] { 1, length, d });
                for (int i = 0; i < top; i++)
                {
                    var rolled = _engine.TensorRoll(series, new[] { -lags[i] }, new[] { 1 });
                    var w = _engine.Reshape(_engine.TensorSlice(weights, new[] { 0, i }, new[] { 1, 1 }), new[] { 1, 1, 1 });
                    sum = _engine.TensorAdd(sum, _engine.TensorMultiply(rolled, _engine.TensorTile(w, new[] { 1, length, d })));
                }
                rows[b] = sum;
            }
            aggregated = batch == 1 ? rows[0] : _engine.TensorConcatenate(rows, 0);
        }
        return Dense(a.Output, aggregated);
    }

    private Tensor<T> FeedForward(BiasFreeLinearLayer<T> conv1, BiasFreeLinearLayer<T> conv2, Tensor<T> x, bool training, Random random)
        => Drop(Dense(conv2, Drop(Activation(Dense(conv1, x)), training, random)), training, random);

    private Tensor<T> Encode(EncoderLayer l, Tensor<T> x, bool training, Random random)
    {
        x = _engine.TensorAdd(x, Drop(Correlate(l.SelfCorrelation, x, x, training), training, random));
        x = Decompose(x).Seasonal;
        return Decompose(_engine.TensorAdd(x, FeedForward(l.Conv1, l.Conv2, x, training, random))).Seasonal;
    }

    private (Tensor<T> Seasonal, Tensor<T> Trend) Decode(DecoderLayer l, Tensor<T> x, Tensor<T> memory, bool training, Random random)
    {
        x = _engine.TensorAdd(x, Drop(Correlate(l.SelfCorrelation, x, x, training), training, random));
        var (s1, t1) = Decompose(x);
        x = _engine.TensorAdd(s1, Drop(Correlate(l.CrossCorrelation, s1, memory, training), training, random));
        var (s2, t2) = Decompose(x);
        var (s3, t3) = Decompose(_engine.TensorAdd(s2, FeedForward(l.Conv1, l.Conv2, s2, training, random)));
        var residual = _engine.TensorAdd(_engine.TensorAdd(t1, t2), t3);
        return (s3, Dense(l.TrendProjection, CircularNeighbours(residual)));
    }

    // ------------------------------------------------------------------ initialization and loading

    /// <summary>The reference's initialization: PyTorch's default for <c>nn.Linear</c>/<c>nn.Conv1d</c>
    /// (U(±1/√fan_in)), Kaiming-normal (fan_in, leaky_relu) token convolutions, unit norms.</summary>
    public void InitializeLikeReference(Random random)
    {
        var c = Configuration;
        double[] UniformArray(int count, int fanIn)
        {
            double bound = 1.0 / Math.Sqrt(fanIn);
            var a = new double[count];
            for (int i = 0; i < count; i++) a[i] = (2 * random.NextDouble() - 1) * bound;
            return a;
        }
        void Uniform(DenseLayer<T> layer, int inputs, int outputs)
            => TorchParameters.Linear(_engine, layer, inputs, outputs, UniformArray(outputs * inputs, inputs), UniformArray(outputs, inputs));
        void Plain(BiasFreeLinearLayer<T> layer, int inputs, int outputs)
            => TorchParameters.Linear(layer, inputs, outputs, UniformArray(outputs * inputs, inputs));
        void TokenConv(BiasFreeLinearLayer<T> layer, int channels)
        {
            int fanIn = 3 * channels;
            double std = Math.Sqrt(2.0) / Math.Sqrt(fanIn);
            var w = new double[c.ModelDim * fanIn];
            for (int i = 0; i < w.Length; i++)
            {
                double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
                w[i] = std * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            }
            TorchParameters.Linear(layer, fanIn, c.ModelDim, w);
        }
        void CorrelationInit(Correlation a)
        {
            Uniform(a.Query, c.ModelDim, c.ModelDim);
            Uniform(a.Key, c.ModelDim, c.ModelDim);
            Uniform(a.Value, c.ModelDim, c.ModelDim);
            Uniform(a.Output, c.ModelDim, c.ModelDim);
        }
        TokenConv(EncoderEmbedding.Token, c.EncoderInputs);
        if (EncoderEmbedding.Time is not null) Plain(EncoderEmbedding.Time, c.TimeFeatures, c.ModelDim);
        TokenConv(DecoderEmbedding.Token, c.DecoderInputs);
        if (DecoderEmbedding.Time is not null) Plain(DecoderEmbedding.Time, c.TimeFeatures, c.ModelDim);
        foreach (var l in EncoderLayers)
        {
            CorrelationInit(l.SelfCorrelation);
            Plain(l.Conv1, c.ModelDim, c.FeedForwardDim);
            Plain(l.Conv2, c.FeedForwardDim, c.ModelDim);
        }
        foreach (var l in DecoderLayers)
        {
            CorrelationInit(l.SelfCorrelation);
            CorrelationInit(l.CrossCorrelation);
            Plain(l.Conv1, c.ModelDim, c.FeedForwardDim);
            Plain(l.Conv2, c.FeedForwardDim, c.ModelDim);
            Plain(l.TrendProjection, 3 * c.ModelDim, c.Outputs);
        }
        Uniform(Projection, c.ModelDim, c.Outputs);
    }

    /// <summary>Loads parameters named as the reference's <c>Model</c> state dict.</summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read)
    {
        var c = Configuration;
        int d = c.ModelDim;
        void Lin(DenseLayer<T> layer, string name, int inputs, int outputs)
            => TorchParameters.Linear(_engine, layer, inputs, outputs, read($"{name}.weight", new[] { outputs, inputs }), read($"{name}.bias", new[] { outputs }));
        // Conv1d weight [out, in, 3] → the linear map of [x_{t−1}, x_t, x_{t+1}]: column k·in + c.
        void Conv(BiasFreeLinearLayer<T> layer, string name, int inputs, int outputs)
        {
            var w = read($"{name}.weight", new[] { outputs, inputs, 3 });
            var unfolded = new double[outputs * 3 * inputs];
            for (int o = 0; o < outputs; o++)
                for (int ch = 0; ch < inputs; ch++)
                    for (int k = 0; k < 3; k++) unfolded[o * 3 * inputs + k * inputs + ch] = w[(o * inputs + ch) * 3 + k];
            TorchParameters.Linear(layer, 3 * inputs, outputs, unfolded);
        }
        void Pointwise(BiasFreeLinearLayer<T> layer, string name, int inputs, int outputs)
            => TorchParameters.Linear(layer, inputs, outputs, read($"{name}.weight", new[] { outputs, inputs, 1 }));
        void Norm(LayerNormalizationLayer<T> layer, string name)
            => TorchParameters.LayerNorm(_engine, layer, read($"{name}.weight", new[] { d }), read($"{name}.bias", new[] { d }));
        void Corr(Correlation a, string name)
        {
            Lin(a.Query, $"{name}.query_projection", d, d);
            Lin(a.Key, $"{name}.key_projection", d, d);
            Lin(a.Value, $"{name}.value_projection", d, d);
            Lin(a.Output, $"{name}.out_projection", d, d);
        }
        void Embedding(Embedding e, string name, int channels)
        {
            Conv(e.Token, $"{name}.value_embedding.tokenConv", channels, d);
            if (e.Time is not null)
                TorchParameters.Linear(e.Time, c.TimeFeatures, d, read($"{name}.temporal_embedding.embed.weight", new[] { d, c.TimeFeatures }));
        }
        Embedding(EncoderEmbedding, "enc_embedding", c.EncoderInputs);
        Embedding(DecoderEmbedding, "dec_embedding", c.DecoderInputs);
        for (int i = 0; i < EncoderLayers.Count; i++)
        {
            var l = EncoderLayers[i];
            string p = $"encoder.attn_layers.{i}";
            Corr(l.SelfCorrelation, $"{p}.attention");
            Pointwise(l.Conv1, $"{p}.conv1", d, c.FeedForwardDim);
            Pointwise(l.Conv2, $"{p}.conv2", c.FeedForwardDim, d);
        }
        Norm(EncoderNorm, "encoder.norm.layernorm");
        for (int i = 0; i < DecoderLayers.Count; i++)
        {
            var l = DecoderLayers[i];
            string p = $"decoder.layers.{i}";
            Corr(l.SelfCorrelation, $"{p}.self_attention");
            Corr(l.CrossCorrelation, $"{p}.cross_attention");
            Pointwise(l.Conv1, $"{p}.conv1", d, c.FeedForwardDim);
            Pointwise(l.Conv2, $"{p}.conv2", c.FeedForwardDim, d);
            Conv(l.TrendProjection, $"{p}.projection", d, c.Outputs);
        }
        Norm(DecoderNorm, "decoder.norm.layernorm");
        Lin(Projection, "decoder.projection", d, c.Outputs);
    }
}
