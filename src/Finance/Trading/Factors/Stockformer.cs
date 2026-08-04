using AiDotNet.Attributes;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Optimizers;
using AiDotNet.Interfaces;
using AiDotNet.Enums;
using AiDotNet.Finance.Base;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;

namespace AiDotNet.Finance.Trading.Factors;

/// <summary>
/// Stockformer: price-volume factor stock selection using a wavelet band split and a dual-frequency
/// spatiotemporal encoder, trained on returns and direction simultaneously.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Bohan Ma, Yushan Xue, Yuan Lu and Jing Chen, "Stockformer: A price-volume factor stock selection
/// model based on wavelet transform and multi-task self-attention networks" (arXiv:2401.06139;
/// Expert Systems with Applications 273:126803, 2025). Reference implementation:
/// github.com/Eric991005/Multitask-Stockformer.
/// </para>
/// <para>
/// <b>REPLACES FactorTransformer.</b> That class cited arXiv:2206.06516 as "FactorFormer:
/// Factor-guided Transformer for Stock Return Prediction". That identifier is "Gauss-Bonnet black
/// holes in (2+1) dimensions: Perturbative aspects and entropy features", a general-relativity paper
/// in Physical Review D — the citation was fabricated, and the implementation was a plain transformer
/// with none of the four contributions below.
/// </para>
/// <para><b>The four contributions, and where each lives:</b></para>
/// <list type="number">
/// <item><description><b>Wavelet band split</b> — <see cref="StockformerBands{T}"/>. A single-level
/// sym2 DWT separating trend from fluctuation. A PREPROCESSING stage: the reference performs it in the
/// training script and feeds the network two already-split inputs.</description></item>
/// <item><description><b>Dual-frequency spatiotemporal encoder</b> —
/// <see cref="StockformerDualEncoder{T}"/>. Low band through temporal self-attention, high band
/// through a causal TCN, each with its own spatial attention over the stock graph.</description></item>
/// <item><description><b>Graph embedding</b> — a struc2vec-derived adjacency supplied via
/// <see cref="Adjacency"/>, precomputed rather than learned.</description></item>
/// <item><description><b>Multi-task heads</b> — return regression and direction classification,
/// combined by <see cref="StockformerMultiTaskLoss{T}"/> as an unweighted 1:1 sum of masked MAE and
/// cross-entropy.</description></item>
/// </list>
/// <para><b>For Beginners:</b> This ranks stocks. It splits each stock's history into a slow trend and
/// fast wiggles, studies each with machinery suited to it, lets stocks influence one another through a
/// similarity graph, and then predicts both how much a stock will move and which way — learning both
/// at once, which the paper shows works better than either alone.</para>
/// </remarks>
/// <example>
/// <code>
/// var model = new Stockformer&lt;double&gt;(new StockformerOptions&lt;double&gt;
/// {
///     NumAssets = 50,
///     NumFeatures = 16,
///     HiddenDimension = 32,
///     SequenceLength = 20,
/// });
///
/// // Rows are stocks, columns are timesteps of the return series.
/// var returns = new Matrix&lt;double&gt;(50, 20);
/// var prediction = model.PredictBands(returns);
/// Console.WriteLine(prediction.Returns.Length);     // one predicted return per stock
/// Console.WriteLine(prediction.DirectionLogits.Length); // stocks x direction classes
/// </code>
/// </example>
[ModelDomain(ModelDomain.Finance)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Regression)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Stockformer: A price-volume factor stock selection model based on wavelet transform and multi-task self-attention networks",
    "https://arxiv.org/abs/2401.06139",
    Year = 2025,
    Authors = "Bohan Ma, Yushan Xue, Yuan Lu, Jing Chen")]
public class Stockformer<T> : CrossSectionalGraphModelBase<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    private readonly StockformerOptions<T> _options;
    private readonly StockformerBands<T> _bands;

    // Every trainable weight lives in a LAYER, not a bare Matrix<T>. That is what makes the model a
    // real member of the family: parameters register through Layers, the tape records each Forward, so
    // gradients flow and Training_ShouldReduceLoss / GradientFlow_ShouldBeNonZeroAndFinite /
    // Clone_AfterTraining_ShouldPreserveLearnedWeights are satisfiable rather than merely asserted.
    // The paper's routing is preserved by how these are COMBINED, not by hand-rolled arithmetic.
    private DenseLayer<T>? _lift;          // scalar band value -> model width
    private DenseLayer<T>? _lowTemporal;   // low band: attention-style value projection
    private DenseLayer<T>? _highTemporal;  // high band: causal-conv-style projection
    private DenseLayer<T>? _spatialLow;    // per-band graph mixing (separate instances, per reference)
    private DenseLayer<T>? _spatialHigh;
    private DenseLayer<T>? _fusion;        // cross-band fusion
    private DenseLayer<T>? _returnHead;
    private DenseLayer<T>? _directionHead;
    private LayerNormalizationLayer<T>? _fusionNorm;
    private StockformerDualEncoder<T>? _encoder;

    /// <summary>Gets the Stockformer configuration in use.</summary>
    /// <remarks>
    /// Named Configuration, not Options: NeuralNetworkBase already exposes an Options member, and
    /// shadowing it would give callers different objects depending on the static type they hold.
    /// </remarks>
    public StockformerOptions<T> Configuration => _options;

    /// <summary>Gets the band splitter.</summary>
    public StockformerBands<T> Bands => _bands;

    /// <summary>
    /// The optimizer the base trains with: Adam at the reference configuration.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <c>learning_rate = 0.001</c>, <c>betas = (0.9, 0.999)</c>, <c>weight_decay = 1e-4</c>, all from
    /// the reference's <c>config/Multitask_Stock.conf</c> and train script.
    /// </para>
    /// <para>
    /// Overriding this matters: the base's hook returns null by default, so it fell back to a generic
    /// optimizer and <see cref="StockformerOptions{T}.LearningRate"/> was read by NOTHING. The paper's
    /// learning rate was sitting in the options as decoration while training used something else.
    /// </para>
    /// </remarks>
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? TrainingOptimizer =>
        _trainingOptimizer ??= new AdamOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate,
                Beta1 = 0.9,
                Beta2 = 0.999,
                // Transformers are trained with a learning-rate warm-up essentially without
                // exception — it is in the original "Attention Is All You Need" schedule and every
                // descendant since, because the first updates land while the attention softmax is
                // still near-uniform and the resulting steps are larger than the loss surface
                // supports. This model took the full rate from step one.
                //
                // The instability was hidden rather than absent: BuildArchitecture applied no seed,
                // so the initialisation differed on every construction and Training_ShouldReduceLoss
                // sampled a fresh starting point each run. Seeding construction made the model
                // reproducible and turned that into a deterministic failure (loss rising
                // 0.0167 -> 0.1763). The warm-up removes the cause instead of re-hiding it behind an
                // unseeded RNG.
                LearningRateScheduler = new LinearWarmupScheduler(
                    baseLearningRate: _options.LearningRate,
                    warmupSteps: WarmupSteps,
                    totalSteps: 0,
                    // One step's worth rather than 0, so the first update is not a no-op.
                    warmupInitLr: _options.LearningRate / WarmupSteps,
                    decayMode: LinearWarmupScheduler.DecayMode.Constant),
                SchedulerStepMode = SchedulerStepMode.StepPerBatch,
            });

    /// <summary>Linear warm-up length, in optimizer steps, before the full learning rate applies.</summary>
    private const int WarmupSteps = 5;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _trainingOptimizer;

    /// <summary>
    /// Creates a Stockformer.
    /// </summary>
    /// <param name="options">Configuration; defaults follow the reference config.</param>
    public Stockformer(StockformerOptions<T>? options = null)
        : base(BuildArchitecture(options ?? new StockformerOptions<T>()),
               (options ?? new StockformerOptions<T>()).SequenceLength,
               (options ?? new StockformerOptions<T>()).PredictionHorizon,
               (options ?? new StockformerOptions<T>()).NumFeatures)
    {
        _options = options ?? new StockformerOptions<T>();

        if (_options.HiddenDimension <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), _options.HiddenDimension,
                "HiddenDimension must be positive.");
        if (_options.NumDirectionClasses <= 1)
            throw new ArgumentOutOfRangeException(nameof(options), _options.NumDirectionClasses,
                "NumDirectionClasses must be at least 2 for the classification task to be meaningful.");

        _bands = new StockformerBands<T>(_options.WaveletOrder, _options.WaveletLevels);

        // NeuralNetworkBase drives layer construction through InitializeLayers.
        InitializeLayers();
    }

    /// <summary>
    /// The model's four outputs for one cross-section.
    /// </summary>
    /// <param name="Returns">Predicted return per stock, from the fused representation.</param>
    /// <param name="LowReturns">Predicted return per stock, from the low-frequency representation.</param>
    /// <param name="DirectionLogits">Direction logits, <c>[stocks * classes]</c>, fused.</param>
    /// <param name="LowDirectionLogits">Direction logits, low-frequency.</param>
    /// <remarks>
    /// Four, not two. The reference applies BOTH heads to BOTH representations and supervises all
    /// four, which is why the loss has four terms.
    /// </remarks>
    public readonly record struct Prediction(
        Vector<T> Returns,
        Vector<T> LowReturns,
        Vector<T> DirectionLogits,
        Vector<T> LowDirectionLogits);

    /// <summary>
    /// Runs the full pipeline: band split, dual encoding, fusion, then both heads on both
    /// representations.
    /// </summary>
    /// <param name="perStockReturns">Rows are stocks, columns are timesteps.</param>
    public Prediction PredictBands(Matrix<T> perStockReturns)
    {
        var (fusedLast, lowLast) = ForwardCore(perStockReturns, out int assets);

        return new Prediction(
            ToVector(_returnHead!.Forward(fusedLast)),
            ToVector(_returnHead!.Forward(lowLast)),
            ToVector(_directionHead!.Forward(fusedLast)),
            ToVector(_directionHead!.Forward(lowLast)));
    }

    private static Vector<T> ToVector(Tensor<T> t)
    {
        var v = new Vector<T>(t.Length);
        for (int i = 0; i < t.Length; i++) v[i] = t[i];
        return v;
    }

    /// <summary>
    /// The paper's training objective for one cross-section.
    /// </summary>
    /// <param name="perStockReturns">Input window, rows are stocks.</param>
    /// <param name="returnTarget">Realized forward return per stock.</param>
    /// <param name="directionTarget">Realized direction class per stock.</param>
    /// <returns>Regression term, classification term, and their unweighted sum.</returns>
    public (double Regression, double Classification, double Total) ComputeLoss(
        Matrix<T> perStockReturns, Vector<T> returnTarget, Vector<T> directionTarget)
    {
        var prediction = PredictBands(perStockReturns);
        return StockformerMultiTaskLoss<T>.Compute(
            prediction.Returns, prediction.LowReturns,
            returnTarget, returnTarget,
            prediction.DirectionLogits, prediction.LowDirectionLogits,
            directionTarget,
            _options.NumDirectionClasses,
            _options.MissingValueSentinel);
    }

    /// <summary>Gets metadata describing this model.</summary>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "Stockformer",
            Version = "1.0",
            Description = "Price-volume factor stock selection via wavelet band split and dual-frequency spatiotemporal attention",
            FeatureCount = _options.NumFeatures,
        };
        metadata.SetProperty("architecture", "dual-frequency-spatiotemporal");
        metadata.SetProperty("wavelet", $"sym{_options.WaveletOrder}");
        metadata.SetProperty("wavelet_levels", _options.WaveletLevels);
        metadata.SetProperty("hidden_dimension", _options.HiddenDimension);
        metadata.SetProperty("heads", _options.NumHeads);
        metadata.SetProperty("direction_classes", _options.NumDirectionClasses);
        metadata.SetProperty("multi_task", true);

        // AdditionalInfo as well as Properties: the family's metadata contract asserts this collection
        // is non-empty, and populating only Properties left it empty. They are separate dictionaries on
        // ModelMetadata, and a reader cannot be expected to know which one a given model chose.
        metadata.AdditionalInfo["Architecture"] = "dual-frequency-spatiotemporal";
        metadata.AdditionalInfo["Wavelet"] = $"sym{_options.WaveletOrder}";
        metadata.AdditionalInfo["WaveletLevels"] = _options.WaveletLevels;
        metadata.AdditionalInfo["HiddenDimension"] = _options.HiddenDimension;
        metadata.AdditionalInfo["NumHeads"] = _options.NumHeads;
        metadata.AdditionalInfo["DirectionClasses"] = _options.NumDirectionClasses;
        metadata.AdditionalInfo["Tasks"] = string.Join(", ", TaskNames);
        metadata.AdditionalInfo["Paper"] = "arXiv:2401.06139";
        return metadata;
    }

    // ------------------------------------------------------------------ base-class contract

    /// <inheritdoc/>
    /// <remarks>
    /// Two: return regression and direction classification. The paper trains them jointly, and that
    /// joint training is its contribution, so both are first-class rather than one being primary and
    /// the other incidental.
    /// </remarks>
    public override int TaskCount => 2;

    /// <inheritdoc/>
    public override IReadOnlyList<string> TaskNames { get; } = new[] { "return", "direction" };

    /// <inheritdoc/>
    /// <remarks>
    /// Returns the FUSED head for each task. The low-frequency variants exist to be supervised during
    /// training (see <see cref="ComputeLoss"/>), not as separate predictions — surfacing four tensors
    /// here would imply four independent answers when there are two.
    /// </remarks>
    public override IReadOnlyList<Tensor<T>> PredictAllTasks(Tensor<T> input)
    {
        var (fusedLast, _) = ForwardCore(ToMatrix(input), out _);
        return new[] { _returnHead!.Forward(fusedLast), _directionHead!.Forward(fusedLast) };
    }

    /// <inheritdoc/>
    /// <remarks>
    /// The primary head: predicted return per stock. Callers wanting the direction head use
    /// <see cref="PredictAllTasks"/>.
    /// </remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        // No Vector round-trip: that would sever the tape and leave training with no gradient path
        // from the loss back through the encoder.
        var (fusedLast, _) = ForwardCore(ToMatrix(input), out _);
        return _returnHead!.Forward(fusedLast);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// The financial base routes its native forward pass through here, so leaving it unimplemented
    /// throws NotSupportedException from every path that trains, forecasts or reads metadata — which
    /// is what made eleven generated ModelFamily tests fail for a single missing override.
    /// </para>
    /// <para>
    /// Quantiles are not supported: Stockformer emits a point return and a direction distribution, not
    /// a predictive interval over returns. Silently ignoring a caller's quantile request would hand
    /// back point estimates dressed as an interval, so a request for them is refused.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ForecastNative(Tensor<T> input, double[]? quantiles)
    {
        if (quantiles is { Length: > 0 })
        {
            throw new NotSupportedException(
                "Stockformer does not produce return quantiles. It predicts a point return plus a " +
                "direction distribution; returning point estimates for quantile requests would " +
                "misrepresent them as an interval. Use PredictAllTasks for the direction logits.");
        }

        return PredictCore(input);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// No layer list. This model's parameters live in the encoder and heads it constructs directly
    /// rather than in a <c>Layers</c> collection, because the dual-band routing is not expressible as
    /// a linear stack — the two bands take different operators and rejoin at the fusion step.
    /// </remarks>
    protected override void InitializeLayers()
    {
        // Idempotent: NeuralNetworkBase may also drive this, and appending twice would double every
        // parameter and silently break the flat parameter contract.
        if (Layers.Count > 0) return;

        int width = _options.HiddenDimension;

        _lift = new DenseLayer<T>(width);
        _lowTemporal = new DenseLayer<T>(width);
        // ReLU on the high band, matching the reference's relu after its temporal convolution.
        // Cast required: ReLUActivation implements both the scalar and vector activation interfaces,
        // so the DenseLayer overloads are ambiguous without it.
        _highTemporal = new DenseLayer<T>(width, (IActivationFunction<T>)new ReLUActivation<T>());
        _spatialLow = new DenseLayer<T>(width);
        _spatialHigh = new DenseLayer<T>(width);
        _fusion = new DenseLayer<T>(width);
        // Present in the reference's adaptiveFusion, and required for training stability.
        _fusionNorm = new LayerNormalizationLayer<T>(width);
        _returnHead = new DenseLayer<T>(1);
        _directionHead = new DenseLayer<T>(_options.NumDirectionClasses);

        Layers.AddRange(new ILayer<T>[]
        {
            _lift, _lowTemporal, _highTemporal, _spatialLow, _spatialHigh, _fusion, _fusionNorm,
            _returnHead, _directionHead,
        });
    }

    /// <summary>
    /// The encoder, built lazily over this model's layers.
    /// </summary>
    /// <remarks>
    /// Constructed on demand rather than in the constructor because the layers must exist first, and
    /// InitializeLayers is what creates them.
    /// </remarks>
    private StockformerDualEncoder<T> Encoder
    {
        get
        {
            if (Layers.Count == 0) InitializeLayers();
            return _encoder ??= new StockformerDualEncoder<T>(
                _options.HiddenDimension, kernelWidth: 3,
                _lowTemporal!, _highTemporal!, _spatialLow!, _spatialHigh!, _fusion!, _fusionNorm!);
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Delegates to the layers in order, which is what keeps this consistent with
    /// <c>GetParameters</c> and lets clone, save and load round-trip. Nothing lives outside a layer.
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));

        int offset = 0;
        foreach (var layer in Layers)
        {
            int count = checked((int)layer.ParameterCount);
            if (count <= 0) continue;
            layer.SetParameters(parameters.Slice(offset, count));
            offset += count;
        }
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        => new Stockformer<T>(_options) { Adjacency = Adjacency };

    /// <summary>
    /// Reinterprets a <c>[stocks, time]</c> (or <c>[1, stocks, time]</c>) tensor as a matrix.
    /// </summary>
    /// <remarks>
    /// The natural input here is a cross-section matrix, but the base's contract is tensor-shaped, so
    /// the conversion is explicit and validated rather than assumed.
    /// </remarks>
    private static Matrix<T> ToMatrix(Tensor<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));

        int rank = input.Shape.Length;
        int stocks, time;
        if (rank == 2) { stocks = input.Shape[0]; time = input.Shape[1]; }
        else if (rank == 3 && input.Shape[0] == 1) { stocks = input.Shape[1]; time = input.Shape[2]; }
        else
        {
            var dims = new int[rank];
            for (int d = 0; d < rank; d++) dims[d] = input.Shape[d];
            throw new ArgumentException(
                $"Expected [stocks, time] or [1, stocks, time]; got [{string.Join(", ", dims)}].", nameof(input));
        }

        var m = new Matrix<T>(stocks, time);
        for (int s = 0; s < stocks; s++)
            for (int t = 0; t < time; t++) m[s, t] = input[(s * time) + t];
        return m;
    }

    /// <summary>Builds the architecture descriptor the financial base requires.</summary>
    private static NeuralNetworkArchitecture<T> BuildArchitecture(StockformerOptions<T> options)
        => new(inputFeatures: Math.Max(1, options.NumFeatures),
               outputSize: Math.Max(1, options.NumAssets))
        {
            // Carry the seed onto the architecture, which is what NeuralNetworkBase reads to give
            // every layer a deterministic RandomSeed. StockformerOptions documents that the
            // reference config's seed of 1 applies when the inherited Seed is unset, but nothing
            // was applying it: the architecture was built without a seed, so each layer initialised
            // from an unseeded RNG and two models built from identical options disagreed from the
            // first prediction (SeededConstructionIsReproducible: -0.382183177397 vs a value
            // differing past the 10th decimal). Mirrors FactorVAE, the sibling model in this folder,
            // which already resolves Seed ?? default the same way.
            RandomSeed = options.Seed ?? 1,
        };

    /// <summary>Lifts each scalar band value to the model width through the lift layer.</summary>
    private Tensor<T> Lift(Matrix<T> band, int stocks, int time)
    {
        var flat = new Tensor<T>(new[] { stocks * time, 1 });
        for (int s = 0; s < stocks; s++)
            for (int t = 0; t < time; t++) flat[(s * time) + t] = band[s, t];

        var lifted = _lift!.Forward(flat);
        return Engine.Reshape(lifted, new[] { stocks, time, _options.HiddenDimension });
    }

    /// <summary>
    /// Selects the final timestep of every asset by MATMUL against a one-hot selector.
    /// </summary>
    /// <remarks>
    /// Not by indexing. Copying elements out of the representation into a fresh tensor severs the
    /// gradient tape, so the heads would train while the encoder behind them received nothing — the
    /// model would report non-zero gradient at the heads and zero everywhere else, and
    /// Training_ShouldReduceLoss would stall for a reason invisible from the outside.
    /// </remarks>
    private Tensor<T> LastStep(Tensor<T> representation, int assets, int time)
    {
        int width = _options.HiddenDimension;
        var selector = new Tensor<T>(new[] { 1, time });
        selector[time - 1] = Ops.One;

        var timeFirst = Engine.Reshape(
            Engine.TensorPermute(representation, new[] { 1, 0, 2 }), new[] { time, assets * width });
        return Engine.Reshape(Engine.TensorMatMul(selector, timeFirst), new[] { assets, width });
    }

    /// <summary>
    /// The tensor-native forward pass. Everything from input to head output is an Engine op or a
    /// layer Forward, so a tape wrapped around this call sees the whole graph.
    /// </summary>
    /// <returns>The fused representation's last step, the low band's, and the band time length.</returns>
    private (Tensor<T> FusedLast, Tensor<T> LowLast) ForwardCore(Matrix<T> perStockReturns, out int assets)
    {
        if (perStockReturns is null) throw new ArgumentNullException(nameof(perStockReturns));

        assets = perStockReturns.Rows;
        if (assets == 0)
            throw new ArgumentException("At least one asset is required.", nameof(perStockReturns));

        var (lowMatrix, highMatrix) = _bands.SplitAll(perStockReturns);
        int time = lowMatrix.Columns;
        if (time == 0)
            throw new ArgumentException(
                $"An input window of {perStockReturns.Columns} timesteps decomposes to a zero-length " +
                $"band at {_options.WaveletLevels} level(s). Lengthen the window.", nameof(perStockReturns));

        var low = Lift(lowMatrix, assets, time);
        var high = Lift(highMatrix, assets, time);
        var (fused, lowEncoded) = Encoder.Encode(low, high, ResolveGraph(assets));

        return (LastStep(fused, assets, time), LastStep(lowEncoded, assets, time));
    }

}
