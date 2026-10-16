using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TimeSeries;

/// <summary>The likelihood of a DeepAR network's observations (Salinas et al. 2020, §3.1).</summary>
public enum DeepARLikelihood
{
    /// <summary>Gaussian for real-valued data: μ = wᵀh + b, σ = softplus(wᵀh + b), both multiplied by ν.</summary>
    Gaussian,

    /// <summary>Negative binomial for count data: μ = ν·softplus(·), α = softplus(·)/√ν, Var = μ + μ²α.</summary>
    NegativeBinomial,

    /// <summary>Student-t, the GluonTS reference's default: df = 2 + softplus(·), loc and softplus scale times ν.</summary>
    StudentT,
}

/// <summary>The sizes of a DeepAR network.</summary>
/// <remarks>With <paramref name="Items"/> &gt; 0 each series has a categorical item id whose learned embedding of
/// <paramref name="EmbeddingDim"/> values is part of every input (§3.4, "a single categorical item feature, for which an
/// embedding is learned").</remarks>
internal sealed record DeepARConfiguration(int Covariates, int HiddenSize, int Layers, double Dropout,
    DeepARLikelihood Likelihood, DeepARScaleHandling Scaling, int Items = 0, int EmbeddingDim = 0);

/// <summary>
/// The DeepAR network (Salinas et al., "DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks",
/// IJF 2020): a multi-layer LSTM over [z_{t−1}/ν, x_t] whose top state parametrizes the likelihood of z_t, the
/// likelihood's scale-dependent parameters multiplied back by the series scale ν.
/// </summary>
/// <remarks>
/// <para>Shared by <see cref="DeepARModel{T}"/> and <see cref="AiDotNet.Finance.Forecasting.Neural.DeepAR{T}"/>. Its
/// layers are added to the owner's list, so they train and save with the owner.</para>
/// <para>Training maximizes the log-likelihood over the whole window, conditioning and prediction range alike (§3.2,
/// "we also include the likelihood terms for t = 0, …, t₀ − 1"). Forecasting is ancestral sampling (§2): the
/// conditioning range sets the state, then each step's sample is fed back as the next input. The LSTM is the standard
/// cell (PyTorch gate order i, f, g, o) with the forget-gate bias initialized to 1 (Appendix: "standard LSTM cells with
/// a forget bias set to 1.0"); dropout between layers follows the GluonTS reference.</para>
/// </remarks>
internal sealed class DeepARNetwork<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public DeepARNetwork(IEngine engine, List<LayerBase<T>> layers, DeepARConfiguration configuration)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        if (c.Covariates < 0 || c.HiddenSize <= 0 || c.Layers <= 0)
            throw new ArgumentException("DeepAR needs a non-negative covariate count and positive hidden size and depth.");
        if (c.Items < 0 || (c.Items > 0 && c.EmbeddingDim <= 0))
            throw new ArgumentException("An item embedding needs a positive embedding size.");
        TLayer Own<TLayer>(TLayer layer) where TLayer : LayerBase<T>
        {
            layers.Add(layer);
            return layer;
        }
        DenseLayer<T> Linear(int outputs) => Own(new DenseLayer<T>(outputs, (IActivationFunction<T>)new IdentityActivation<T>()));
        if (c.Items > 0) ItemEmbedding = Own(new BiasFreeLinearLayer<T>(c.Items, c.EmbeddingDim));
        for (int l = 0; l < c.Layers; l++)
            Cells.Add(new Cell(Linear(4 * c.HiddenSize), Linear(4 * c.HiddenSize)));
        Head = Linear(ParameterCount(c.Likelihood));
    }

    /// <summary>An LSTM layer: the input-to-hidden and hidden-to-hidden maps, each with its bias (PyTorch's
    /// <c>weight_ih</c>/<c>bias_ih</c> and <c>weight_hh</c>/<c>bias_hh</c>).</summary>
    internal sealed record Cell(DenseLayer<T> InputToHidden, DenseLayer<T> HiddenToHidden);

    public DeepARConfiguration Configuration { get; }
    public List<Cell> Cells { get; } = new();

    /// <summary>The projection of the top state to the likelihood's raw parameters.</summary>
    public DenseLayer<T> Head { get; }

    /// <summary>The item embedding (a linear map of the one-hot item id), when the network has items.</summary>
    public BiasFreeLinearLayer<T>? ItemEmbedding { get; }

    private int InputSize => 1 + Configuration.Covariates + (Configuration.Items > 0 ? Configuration.EmbeddingDim : 0);

    // The embeddings [B, E] of the items of every row, or null without items.
    private Tensor<T>? Embed(int[]? items, int rows)
    {
        if (ItemEmbedding is null) return null;
        if (items is null || items.Length != rows)
            throw new ArgumentException($"This network embeds an item per series: supply {rows} item ids.");
        var oneHot = new Tensor<T>(new[] { rows, Configuration.Items });
        for (int r = 0; r < rows; r++)
        {
            if (items[r] < 0 || items[r] >= Configuration.Items)
                throw new ArgumentOutOfRangeException(nameof(items), $"Item id {items[r]} is outside 0..{Configuration.Items - 1}.");
            oneHot[r, items[r]] = NumOps.One;
        }
        return ItemEmbedding.Forward(oneHot);
    }

    private static int ParameterCount(DeepARLikelihood likelihood) => likelihood == DeepARLikelihood.StudentT ? 3 : 2;

    // ------------------------------------------------------------------ scale

    /// <summary>The scale ν of each series from its conditioning range <c>[B, t₀]</c> (§3.3).</summary>
    public double[] Scale(double[,] conditioning)
    {
        int batch = conditioning.GetLength(0), length = conditioning.GetLength(1);
        var scale = new double[batch];
        for (int b = 0; b < batch; b++)
        {
            double sum = 0;
            for (int t = 0; t < length; t++)
                sum += Configuration.Scaling == DeepARScaleHandling.MeanAbsolute ? Math.Abs(conditioning[b, t]) : conditioning[b, t];
            double mean = length > 0 ? sum / length : 0;
            scale[b] = Configuration.Scaling switch
            {
                DeepARScaleHandling.PaperMean => 1 + mean,
                DeepARScaleHandling.MeanAbsolute => Math.Max(mean, 1e-10),
                _ => 1.0,
            };
            if (scale[b] <= 0)
                throw new ArgumentException(
                    $"The paper's scale 1 + mean(z) is {scale[b]} for a series with mean {mean}: it assumes non-negative data. Use DeepARScaleHandling.MeanAbsolute for signed series.");
        }
        return scale;
    }

    // ------------------------------------------------------------------ the LSTM

    private Tensor<T> Drop(Tensor<T> x, bool training, Random random)
    {
        double rate = Configuration.Dropout;
        if (!training || rate <= 0) return x;
        var mask = new Tensor<T>(x._shape);
        var keep = NumOps.FromDouble(1.0 / (1.0 - rate));
        for (int i = 0; i < mask.Length; i++) mask[i] = random.NextDouble() < rate ? NumOps.Zero : keep;
        return _engine.TensorMultiply(x, mask);
    }

    /// <summary>The recurrent state: each layer's hidden and cell state <c>[B, H]</c>.</summary>
    internal sealed class State
    {
        public State(Tensor<T>[] hidden, Tensor<T>[] cell)
        {
            Hidden = hidden;
            Cell = cell;
        }

        public Tensor<T>[] Hidden { get; }
        public Tensor<T>[] Cell { get; }
    }

    public State Initial(int batch)
    {
        int layers = Configuration.Layers, h = Configuration.HiddenSize;
        var hidden = new Tensor<T>[layers];
        var cell = new Tensor<T>[layers];
        for (int l = 0; l < layers; l++)
        {
            hidden[l] = new Tensor<T>(new[] { batch, h });
            cell[l] = new Tensor<T>(new[] { batch, h });
        }
        return new State(hidden, cell);
    }

    // One time step of the stack on inputs [B, 1 + C]; returns the top hidden state [B, H].
    private Tensor<T> Step(Tensor<T> input, State state, bool training, Random random)
    {
        int batch = input.Shape[0], h = Configuration.HiddenSize;
        var x = input;
        for (int l = 0; l < Cells.Count; l++)
        {
            var gates = _engine.TensorAdd(Cells[l].InputToHidden.Forward(x), Cells[l].HiddenToHidden.Forward(state.Hidden[l]));
            Tensor<T> Gate(int k) => _engine.TensorSlice(gates, new[] { 0, k * h }, new[] { batch, h });
            var i = _engine.Sigmoid(Gate(0));
            var f = _engine.Sigmoid(Gate(1));
            var g = _engine.Tanh(Gate(2));
            var o = _engine.Sigmoid(Gate(3));
            var cell = _engine.TensorAdd(_engine.TensorMultiply(f, state.Cell[l]), _engine.TensorMultiply(i, g));
            var hidden = _engine.TensorMultiply(o, _engine.Tanh(cell));
            state.Cell[l] = cell;
            state.Hidden[l] = hidden;
            x = l < Cells.Count - 1 ? Drop(hidden, training, random) : hidden;
        }
        return x;
    }

    // [z_{t−1}/ν, x_t, item embedding] for every series at one step.
    private Tensor<T> Input(double[] previous, double[] scale, double[,,]? covariates, int t, Tensor<T>? embedding)
    {
        int batch = previous.Length, c = Configuration.Covariates;
        var input = new Tensor<T>(new[] { batch, 1 + c });
        for (int b = 0; b < batch; b++)
        {
            input[b, 0] = NumOps.FromDouble(previous[b] / scale[b]);
            for (int k = 0; k < c; k++) input[b, 1 + k] = NumOps.FromDouble(covariates![b, t, k]);
        }
        return embedding is null ? input : _engine.TensorConcatenate(new[] { input, embedding }, 1);
    }

    // ------------------------------------------------------------------ likelihood

    private Tensor<T> Column(Tensor<T> x, int k) => _engine.TensorSlice(x, new[] { 0, k }, new[] { x.Shape[0], 1 });

    private Tensor<T> ScaleColumn(double[] scale, Func<double, double> map)
    {
        var t = new Tensor<T>(new[] { scale.Length, 1 });
        for (int b = 0; b < scale.Length; b++) t[b, 0] = NumOps.FromDouble(map(scale[b]));
        return t;
    }

    /// <summary>The likelihood parameters of one step, scale-adjusted (each <c>[B, 1]</c>).</summary>
    public (Tensor<T> A, Tensor<T> B, Tensor<T>? C) Parameters(Tensor<T> top, double[] scale)
    {
        var raw = Head.Forward(top);
        var nu = ScaleColumn(scale, s => s);
        switch (Configuration.Likelihood)
        {
            case DeepARLikelihood.Gaussian:
                return (_engine.TensorMultiply(Column(raw, 0), nu), _engine.TensorMultiply(_engine.Softplus(Column(raw, 1)), nu), null);
            case DeepARLikelihood.NegativeBinomial:
                return (_engine.TensorMultiply(_engine.Softplus(Column(raw, 0)), nu),
                    _engine.TensorMultiply(_engine.Softplus(Column(raw, 1)), ScaleColumn(scale, s => 1.0 / Math.Sqrt(s))), null);
            default:
                var df = _engine.TensorAddScalar(_engine.Softplus(Column(raw, 0)), NumOps.FromDouble(2.0));
                return (df, _engine.TensorMultiply(Column(raw, 1), nu), _engine.TensorMultiply(_engine.Softplus(Column(raw, 2)), nu));
        }
    }

    /// <summary>The negative log-likelihood <c>[B, 1]</c> of observations z <c>[B, 1]</c>.</summary>
    public Tensor<T> NegativeLogLikelihood((Tensor<T> A, Tensor<T> B, Tensor<T>? C) p, Tensor<T> z)
    {
        var e = _engine;
        T Const(double v) => NumOps.FromDouble(v);
        switch (Configuration.Likelihood)
        {
            case DeepARLikelihood.Gaussian:
            {
                var standardized = e.TensorDivide(e.TensorSubtract(z, p.A), p.B);
                return e.TensorAddScalar(e.TensorAdd(e.TensorLog(p.B), e.TensorMultiplyScalar(e.TensorMultiply(standardized, standardized), Const(0.5))),
                    Const(0.5 * Math.Log(2 * Math.PI)));
            }
            case DeepARLikelihood.NegativeBinomial:
            {
                // log p = lgamma(z + 1/α) − lgamma(z + 1) − lgamma(1/α) − (1/α)·log(1 + αμ) + z·(log αμ − log(1 + αμ))
                var r = e.TensorReciprocal(p.B);
                var alphaMu = e.TensorMultiply(p.B, p.A);
                var logOnePlus = e.TensorLog(e.TensorAddScalar(alphaMu, Const(1.0)));
                var logPmf = e.TensorAdd(
                    e.TensorSubtract(e.TensorSubtract(e.TensorLgamma(e.TensorAdd(z, r)), e.TensorLgamma(e.TensorAddScalar(z, Const(1.0)))), e.TensorLgamma(r)),
                    e.TensorSubtract(e.TensorMultiply(z, e.TensorSubtract(e.TensorLog(alphaMu), logOnePlus)), e.TensorMultiply(r, logOnePlus)));
                return e.TensorMultiplyScalar(logPmf, Const(-1.0));
            }
            default:
            {
                // log p = lgamma((ν+1)/2) − lgamma(ν/2) − ½ log(νπ) − log s − (ν+1)/2 · log(1 + ((z − loc)/s)²/ν)
                var df = p.A;
                var standardized = e.TensorDivide(e.TensorSubtract(z, p.B), p.C!);
                var halfDfPlus = e.TensorMultiplyScalar(e.TensorAddScalar(df, Const(1.0)), Const(0.5));
                var logPdf = e.TensorSubtract(
                    e.TensorSubtract(
                        e.TensorSubtract(e.TensorLgamma(halfDfPlus), e.TensorLgamma(e.TensorMultiplyScalar(df, Const(0.5)))),
                        e.TensorMultiplyScalar(e.TensorLog(e.TensorMultiplyScalar(df, Const(Math.PI))), Const(0.5))),
                    e.TensorAdd(e.TensorLog(p.C!),
                        e.TensorMultiply(halfDfPlus, e.TensorLog(e.TensorAddScalar(e.TensorDivide(e.TensorMultiply(standardized, standardized), df), Const(1.0))))));
                return e.TensorMultiplyScalar(logPdf, Const(-1.0));
            }
        }
    }

    /// <summary>
    /// The mean negative log-likelihood of windows of observations <c>[B, T + 1]</c> — z₀ (the value before the window,
    /// zero at a series start) then z₁ … z_T — over every step, the first <paramref name="conditioning"/> steps
    /// setting ν; <paramref name="covariates"/> <c>[B, T, C]</c> are aligned to z₁ … z_T, and <paramref name="items"/>
    /// gives each row's item id when the network has items.
    /// </summary>
    public Tensor<T> Loss(double[,] window, double[,,]? covariates, int conditioning, bool training, Random random, int[]? items = null)
    {
        int batch = window.GetLength(0), steps = window.GetLength(1) - 1;
        if (steps <= 0) throw new ArgumentException("A training window needs at least one observation after z₀.");
        if (conditioning <= 0 || conditioning > steps) throw new ArgumentException("The conditioning range must be within the window.");
        var conditioningValues = new double[batch, conditioning];
        for (int b = 0; b < batch; b++)
            for (int t = 0; t < conditioning; t++) conditioningValues[b, t] = window[b, 1 + t];
        var scale = Scale(conditioningValues);
        var embedding = Embed(items, batch);
        var state = Initial(batch);
        Tensor<T>? total = null;
        var previous = new double[batch];
        var observed = new Tensor<T>(new[] { batch, 1 });
        for (int t = 0; t < steps; t++)
        {
            for (int b = 0; b < batch; b++) previous[b] = window[b, t];
            var top = Step(Input(previous, scale, covariates, t, embedding), state, training, random);
            var z = new Tensor<T>(new[] { batch, 1 });
            for (int b = 0; b < batch; b++) z[b, 0] = NumOps.FromDouble(window[b, t + 1]);
            var nll = NegativeLogLikelihood(Parameters(top, scale), z);
            total = total is null ? nll : _engine.TensorAdd(total, nll);
        }
        return _engine.TensorMultiplyScalar(_engine.ReduceSum(total!, new[] { 0, 1 }, keepDims: false), NumOps.FromDouble(1.0 / ((double)batch * steps)));
    }

    // ------------------------------------------------------------------ forecasting

    // The location (mean for the Gaussian and negative binomial, loc for the Student-t) of step parameters [B, 1].
    private Tensor<T> Location((Tensor<T> A, Tensor<T> B, Tensor<T>? C) p)
        => Configuration.Likelihood == DeepARLikelihood.StudentT ? p.B : p.A;

    /// <summary>
    /// A deterministic, differentiable forecast <c>[B, H]</c>: the history <c>[B, t₀ + 1]</c> sets ν and the state, then
    /// each step's predicted location is fed back as the next input. Forecasts proper are sampled
    /// (<see cref="Sample"/>); this path serves owners that need a tape-connected forward of the forecast's shape.
    /// </summary>
    public Tensor<T> MeanPath(double[,] history, double[,,]? historyCovariates, double[,,]? futureCovariates, int horizon,
        bool training, Random random, int[]? items = null)
    {
        int batch = history.GetLength(0), length = history.GetLength(1) - 1;
        if (length <= 0) throw new ArgumentException("Forecasting needs at least one observation.");
        var conditioningValues = new double[batch, length];
        for (int b = 0; b < batch; b++)
            for (int t = 0; t < length; t++) conditioningValues[b, t] = history[b, 1 + t];
        var scale = Scale(conditioningValues);
        var embedding = Embed(items, batch);
        var state = Initial(batch);
        var previous = new double[batch];
        for (int t = 0; t < length; t++)
        {
            for (int b = 0; b < batch; b++) previous[b] = history[b, t];
            Step(Input(previous, scale, historyCovariates, t, embedding), state, training, random);
        }
        for (int b = 0; b < batch; b++) previous[b] = history[b, length];
        var steps = new Tensor<T>[horizon];
        for (int h = 0; h < horizon; h++)
        {
            var top = Step(Input(previous, scale, futureCovariates, h, embedding), state, training, random);
            steps[h] = Location(Parameters(top, scale));
            for (int b = 0; b < batch; b++) previous[b] = NumOps.ToDouble(steps[h][b, 0]);
        }
        return horizon == 1 ? steps[0] : _engine.TensorConcatenate(steps, 1);
    }

    /// <summary>
    /// Ancestral sample paths <c>[B][S][H]</c>: the history <c>[B, t₀ + 1]</c> (z₀ … z_{t₀}) sets ν and the state, then
    /// each of <paramref name="horizon"/> steps draws ẑ from the likelihood and feeds it back.
    /// <paramref name="historyCovariates"/> <c>[B, t₀, C]</c> and <paramref name="futureCovariates"/> <c>[B, H, C]</c>
    /// are required when the network has covariates, and <paramref name="items"/> when it has items.
    /// </summary>
    public double[][][] Sample(double[,] history, double[,,]? historyCovariates, double[,,]? futureCovariates, int horizon, int samples,
        Random random, int[]? items = null)
    {
        int batch = history.GetLength(0), length = history.GetLength(1) - 1;
        if (length <= 0) throw new ArgumentException("Forecasting needs at least one observation.");
        var conditioningValues = new double[batch, length];
        for (int b = 0; b < batch; b++)
            for (int t = 0; t < length; t++) conditioningValues[b, t] = history[b, 1 + t];
        var scale = Scale(conditioningValues);
        using var _ = new NoGradScope<T>();

        // The conditioning range, once per series; its state is copied to every sample path.
        var embedding = Embed(items, batch);
        var state = Initial(batch);
        var previous = new double[batch];
        Tensor<T>? top = null;
        for (int t = 0; t < length; t++)
        {
            for (int b = 0; b < batch; b++) previous[b] = history[b, t];
            top = Step(Input(previous, scale, historyCovariates, t, embedding), state, training: false, random);
        }
        int rows = batch * samples;
        Tensor<T> Repeat(Tensor<T> x)
        {
            var r = new Tensor<T>(new[] { rows, x.Shape[1] });
            for (int b = 0; b < batch; b++)
                for (int s = 0; s < samples; s++)
                    for (int j = 0; j < x.Shape[1]; j++) r[b * samples + s, j] = x[b, j];
            return r;
        }
        var paths = new State(state.Hidden.Select(Repeat).ToArray(), state.Cell.Select(Repeat).ToArray());
        var rowEmbedding = embedding is null ? null : Repeat(embedding);
        var rowScale = new double[rows];
        var last = new double[rows];
        for (int b = 0; b < batch; b++)
            for (int s = 0; s < samples; s++)
            {
                rowScale[b * samples + s] = scale[b];
                last[b * samples + s] = history[b, length];
            }
        double[,,]? rowCovariates = null;
        if (Configuration.Covariates > 0)
        {
            if (futureCovariates is null) throw new ArgumentException("This network has covariates: supply their future values.");
            rowCovariates = new double[rows, horizon, Configuration.Covariates];
            for (int b = 0; b < batch; b++)
                for (int s = 0; s < samples; s++)
                    for (int h = 0; h < horizon; h++)
                        for (int k = 0; k < Configuration.Covariates; k++) rowCovariates[b * samples + s, h, k] = futureCovariates[b, h, k];
        }

        var result = new double[batch][][];
        for (int b = 0; b < batch; b++)
        {
            result[b] = new double[samples][];
            for (int s = 0; s < samples; s++) result[b][s] = new double[horizon];
        }
        for (int h = 0; h < horizon; h++)
        {
            var output = Step(Input(last, rowScale, rowCovariates, h, rowEmbedding), paths, training: false, random);
            var p = Parameters(output, rowScale);
            for (int r = 0; r < rows; r++)
            {
                double draw = Draw(NumOps.ToDouble(p.A[r, 0]), NumOps.ToDouble(p.B[r, 0]), p.C is null ? 0 : NumOps.ToDouble(p.C[r, 0]), random);
                result[r / samples][r % samples][h] = draw;
                last[r] = draw;
            }
        }
        return result;
    }

    private double Draw(double a, double b, double c, Random random)
    {
        switch (Configuration.Likelihood)
        {
            case DeepARLikelihood.Gaussian:
                return a + b * StandardNormal(random);
            case DeepARLikelihood.NegativeBinomial:
            {
                // Gamma–Poisson mixture: λ ~ Gamma(1/α, αμ), z ~ Poisson(λ).
                double lambda = GammaSample(1.0 / b, b * a, random);
                return Poisson(lambda, random);
            }
            default:
            {
                // loc + scale · N / sqrt(χ²_ν / ν)
                double chi = 2.0 * GammaSample(a / 2.0, 1.0, random);
                return b + c * StandardNormal(random) / Math.Sqrt(chi / a);
            }
        }
    }

    private static double StandardNormal(Random random)
    {
        double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    // Marsaglia–Tsang (2000), with the shape < 1 boost.
    private static double GammaSample(double shape, double scale, Random random)
    {
        if (shape < 1)
            return GammaSample(shape + 1, scale, random) * Math.Pow(random.NextDouble(), 1.0 / shape);
        double d = shape - 1.0 / 3, c = 1.0 / Math.Sqrt(9 * d);
        while (true)
        {
            double x, v;
            do
            {
                x = StandardNormal(random);
                v = 1 + c * x;
            } while (v <= 0);
            v = v * v * v;
            double u = random.NextDouble();
            if (u < 1 - 0.0331 * x * x * x * x || Math.Log(u) < 0.5 * x * x + d * (1 - v + Math.Log(v)))
                return d * v * scale;
        }
    }

    // Knuth's product method for small means, the PTRS transformed rejection (Hörmann 1993) for large ones.
    private static double Poisson(double lambda, Random random)
    {
        if (lambda <= 0) return 0;
        if (lambda < 30)
        {
            double limit = Math.Exp(-lambda), product = random.NextDouble();
            int k = 0;
            while (product > limit)
            {
                k++;
                product *= random.NextDouble();
            }
            return k;
        }
        double slam = Math.Sqrt(lambda), loglam = Math.Log(lambda);
        double b = 0.931 + 2.53 * slam, a = -0.059 + 0.02483 * b, invAlpha = 1.1239 + 1.1328 / (b - 3.4), vr = 0.9277 - 3.6224 / (b - 2);
        while (true)
        {
            double u = random.NextDouble() - 0.5, v = random.NextDouble(), us = 0.5 - Math.Abs(u);
            double k = Math.Floor((2 * a / us + b) * u + lambda + 0.43);
            if (us >= 0.07 && v <= vr) return k;
            if (k < 0 || (us < 0.013 && v > us)) continue;
            if (Math.Log(v) + Math.Log(invAlpha) - Math.Log(a / (us * us) + b) <= -lambda + k * loglam - LogFactorial(k)) return k;
        }
    }

    // ln k!: exact below 20, Stirling's series (error below 1e-14 there) above.
    private static double LogFactorial(double k)
    {
        if (k < 20)
        {
            double sum = 0;
            for (int i = 2; i <= (int)k; i++) sum += Math.Log(i);
            return sum;
        }
        double n = k + 1;
        return (n - 0.5) * Math.Log(n) - n + 0.5 * Math.Log(2 * Math.PI) + 1.0 / (12 * n) - 1.0 / (360 * n * n * n) + 1.0 / (1260 * Math.Pow(n, 5));
    }

    // ------------------------------------------------------------------ initialization and loading

    /// <summary>
    /// PyTorch's LSTM initialization (every weight and bias U(±1/√H)) with the forget-gate bias raised to 1 (the
    /// paper's appendix), and <c>nn.Linear</c>'s for the head.
    /// </summary>
    public void InitializeLikeReference(Random random)
    {
        int h = Configuration.HiddenSize;
        double bound = 1.0 / Math.Sqrt(h);
        double[] Uniform(int count, double limit)
        {
            var a = new double[count];
            for (int i = 0; i < count; i++) a[i] = (2 * random.NextDouble() - 1) * limit;
            return a;
        }
        for (int l = 0; l < Cells.Count; l++)
        {
            int inputs = l == 0 ? InputSize : h;
            var biasIh = Uniform(4 * h, bound);
            var biasHh = Uniform(4 * h, bound);
            for (int j = h; j < 2 * h; j++)
            {
                biasIh[j] = 1.0;
                biasHh[j] = 0.0;
            }
            TorchParameters.Linear(_engine, Cells[l].InputToHidden, inputs, 4 * h, Uniform(4 * h * inputs, bound), biasIh);
            TorchParameters.Linear(_engine, Cells[l].HiddenToHidden, h, 4 * h, Uniform(4 * h * h, bound), biasHh);
        }
        int outputs = ParameterCount(Configuration.Likelihood);
        TorchParameters.Linear(_engine, Head, h, outputs, Uniform(outputs * h, bound), Uniform(outputs, bound));
        if (ItemEmbedding is not null)
        {
            // nn.Embedding's N(0, 1).
            var table = new double[Configuration.EmbeddingDim * Configuration.Items];
            for (int i = 0; i < table.Length; i++)
            {
                double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
                table[i] = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            }
            TorchParameters.Linear(ItemEmbedding, Configuration.Items, Configuration.EmbeddingDim, table);
        }
    }

    /// <summary>Loads parameters named as a PyTorch <c>nn.LSTM</c> (<c>lstm.weight_ih_l{k}</c>, …) and the head
    /// (<c>head.weight</c>, <c>head.bias</c>).</summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read)
    {
        int h = Configuration.HiddenSize;
        for (int l = 0; l < Cells.Count; l++)
        {
            int inputs = l == 0 ? InputSize : h;
            TorchParameters.Linear(_engine, Cells[l].InputToHidden, inputs, 4 * h,
                read($"lstm.weight_ih_l{l}", new[] { 4 * h, inputs }), read($"lstm.bias_ih_l{l}", new[] { 4 * h }));
            TorchParameters.Linear(_engine, Cells[l].HiddenToHidden, h, 4 * h,
                read($"lstm.weight_hh_l{l}", new[] { 4 * h, h }), read($"lstm.bias_hh_l{l}", new[] { 4 * h }));
        }
        int outputs = ParameterCount(Configuration.Likelihood);
        TorchParameters.Linear(_engine, Head, h, outputs, read("head.weight", new[] { outputs, h }), read("head.bias", new[] { outputs }));
    }
}
