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
/// <see cref="StockformerDualEncoder{T}"/>. Low band through temporal self-attention (Eq. 7), high
/// band through a causal TCN, each then through self-attention over assets (Eq. 10), fused by two
/// summed attention terms (Eq. 11).</description></item>
/// <item><description><b>Graph embedding</b> — a struc2vec-derived per-asset embedding supplied via
/// <see cref="CrossSectionalGraphModelBase{T}.AssetEmbedding"/> and ADDED to the features per Eq. 10,
/// precomputed rather than learned. Not an adjacency matrix: the paper sums it in as a prior and lets
/// attention learn the relationships.</description></item>
/// <item><description><b>Multi-task heads</b> — return regression and direction classification,
/// combined by <see cref="StockformerMultiTaskLoss{T}"/> as <c>L_reg + lambda*L_cla</c> (Eq. 12),
/// masked MAE plus cross-entropy, with lambda defaulting to the reference's 1.0.</description></item>
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
    private DenseLayer<T>? _lift;           // scalar band value -> model width
    private DenseLayer<T>? _lowUpsample;    // Eq. 3-4: learnable inverse DWT, low band
    private DenseLayer<T>? _highUpsample;   // Eq. 3-4: learnable inverse DWT, high band
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
        if (perStockReturns is null) throw new ArgumentNullException(nameof(perStockReturns));

        // Single-feature convenience: one series per asset. The paper's real input carries D factors,
        // so prefer the Tensor overload for anything beyond a smoke test.
        var asTensor = new Tensor<T>(new[] { perStockReturns.Rows, perStockReturns.Columns });
        for (int a = 0; a < perStockReturns.Rows; a++)
            for (int t = 0; t < perStockReturns.Columns; t++)
                asTensor[(a * perStockReturns.Columns) + t] = perStockReturns[a, t];
        return PredictBands(asTensor);
    }

    /// <summary>
    /// Runs the full pipeline on a <c>[assets, time, features]</c> factor tensor.
    /// </summary>
    /// <remarks>
    /// This is the paper's input shape: D price-volume factors per asset per timestep (D = 360 in the
    /// reference), not a single return series.
    /// </remarks>
    public Prediction PredictBands(Tensor<T> features)
    {
        var (fusedLast, lowLast) = ForwardCore(features, out int assets);

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
            _options.MissingValueSentinel,
            _options.TaskLossWeight);
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
        var (fusedLast, _) = ForwardCore(input, out _);
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
        var (fusedLast, _) = ForwardCore(input, out _);
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
        // Eq. 3-4: the inverse DWT is LEARNABLE (X_l = W^g g^T Xbar_l + b^g) and restores each band to
        // the FULL window length. Treating the whole transform as fixed preprocessing, and feeding the
        // encoder half-length bands, was wrong.
        _lowUpsample = new DenseLayer<T>(_options.SequenceLength);
        _highUpsample = new DenseLayer<T>(_options.SequenceLength);
        // Present in the reference's adaptiveFusion; the paper's method section does not mention it.
        //
        // FIXED width. In the real forward pass this layer only ever sees width-sized activations, and
        // the lazy constructor locks its gamma to whatever shape it happens to be handed first — which,
        // if that first call comes from a generic walk over Layers, is the wrong width permanently.
        _fusionNorm = new LayerNormalizationLayer<T>(width);
        _returnHead = new DenseLayer<T>(1);
        _directionHead = new DenseLayer<T>(_options.NumDirectionClasses);

        // ORDER MATTERS, for a reason unrelated to the forward pass. The dual-band routing means these
        // layers are NOT applied in collection order, but the family's layer-activation probe walks
        // Layers sequentially, feeding each one the previous one's output. So the sequence must at least
        // be shape-COMPATIBLE end to end, or the probe throws mid-walk:
        //   _lift (-> width), then everything width->width, then the heads, and the width->sequence
        //   upsample filters LAST.
        // Putting the upsamplers early emitted a sequence-width tensor into a width-sized LayerNorm and
        // produced "Gamma shape (128) does not match the last 1 dimensions of input shape (1, 20, 20)".
        Layers.Add(_lift);
        Layers.Add(_fusionNorm);

        // L stacked encoder layers (paper) / layers = 2 (reference config). NumLayers was previously
        // read by nothing — the encoder ran a single pass.
        var stack = new List<StockformerDualEncoder<T>.Layer>();
        for (int l = 0; l < Math.Max(1, _options.NumLayers); l++)
        {
            // ReLU after the high band's causal convolution, per the reference. Cast required:
            // ReLUActivation implements both activation interfaces, so the overloads are ambiguous.
            var highProjection = new DenseLayer<T>(width, (IActivationFunction<T>)new ReLUActivation<T>());
            Layers.Add(highProjection);

            stack.Add(new StockformerDualEncoder<T>.Layer(
                LowTemporal: MakeAttention(width, causal: true),    // Eq. 7, over time
                LowSpatial: MakeAttention(width, causal: false),    // Eq. 10, over assets
                HighSpatial: MakeAttention(width, causal: false),
                HighTemporalProjection: highProjection));
        }

        _encoder = new StockformerDualEncoder<T>(
            width, kernelWidth: 3, stack,
            fusionSelf: MakeAttention(width, causal: true),         // Eq. 11 term 1
            fusionCross: MakeAttention(width, causal: true),        // Eq. 11 term 2
            fusionNorm: _fusionNorm);

        // Heads, then the width->sequence upsamplers last, so the sequential probe stays shape-valid.
        Layers.Add(_returnHead);
        Layers.Add(_directionHead);
        Layers.Add(_lowUpsample);
        Layers.Add(_highUpsample);
    }

    /// <summary>
    /// Builds one attention block and registers its three projections as model layers.
    /// </summary>
    /// <remarks>
    /// Registering here is what makes W^Q, W^K and W^V real parameters. Projections held as bare
    /// matrices would be invisible to both the optimizer and the gradient tape.
    /// </remarks>
    private StockformerAttention<T> MakeAttention(int width, bool causal)
    {
        var q = new DenseLayer<T>(width);
        var k = new DenseLayer<T>(width);
        var v = new DenseLayer<T>(width);
        Layers.Add(q);
        Layers.Add(k);
        Layers.Add(v);
        return new StockformerAttention<T>(width, q, k, v, causal);
    }

    /// <summary>The encoder, built by <see cref="InitializeLayers"/> over this model's layers.</summary>
    private StockformerDualEncoder<T> Encoder
    {
        get
        {
            if (_encoder is null) InitializeLayers();
            return _encoder!;
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
        => new Stockformer<T>(_options) { AssetGraph = AssetGraph, AssetEmbedding = AssetEmbedding };

    /// <summary>Builds the architecture descriptor the financial base requires.</summary>
    /// <remarks>
    /// <para>
    /// Declares a SEQUENCE input — <c>inputHeight = sequenceLength</c>, <c>inputWidth = numFeatures</c>.
    /// TwoDimensional specifically, because <c>GetInputShape()</c> maps it to
    /// <c>[InputHeight, InputWidth]</c>, and the family's test base derives the fed shape from exactly
    /// that declaration (<c>GetArchitecture().GetInputShape()</c>) with a batch axis prepended. So this
    /// produces <c>[batch, sequence, features]</c> — the shape the model actually wants. Choosing
    /// ThreeDimensional instead inserts a depth axis and yields <c>[batch, depth, sequence, features]</c>.
    /// </para>
    /// <para>
    /// The previous form passed only <c>inputFeatures</c>, so the architecture described a flat
    /// 360-wide vector with no time axis and callers fed <c>[1, 1, 360]</c>: ONE timestep. A wavelet
    /// window cannot be built from one step, and an earlier revision only appeared to work because it
    /// read that shape as 360 TIMESTEPS of a single feature — decomposing across the feature axis
    /// instead of time, which is not what the paper does.
    /// </para>
    /// </remarks>
    private static NeuralNetworkArchitecture<T> BuildArchitecture(StockformerOptions<T> options)
        // BOTH halves of this are load-bearing and were found independently.
        //
        // TwoDimensional with inputHeight/inputWidth: the family derives the fed input shape from
        // GetArchitecture().GetInputShape(), and TwoDimensional maps to [InputHeight, InputWidth], so
        // this produces [batch, sequence, features]. The previous inputFeatures-only form described a
        // flat vector with no time axis and callers were handed a SINGLE timestep, which no wavelet
        // window can be built from.
        //
        // RandomSeed: NeuralNetworkBase reads it to give every layer a deterministic seed. Without it
        // each layer initialised from an unseeded RNG and two models built from identical options
        // disagreed from the first prediction, so StockformerOptions' documented "reference seed of 1"
        // applied to nothing. Mirrors FactorVAE, the sibling in this folder.
        => new(inputType: InputType.TwoDimensional,
               taskType: NeuralNetworkTaskType.Regression,
               inputHeight: Math.Max(2, options.SequenceLength),
               inputWidth: Math.Max(1, options.NumFeatures),
               outputSize: Math.Max(1, options.NumAssets))
        {
            RandomSeed = options.Seed ?? 1,
        };


    /// <summary>Projects the D input factors to the model width, through the lift layer.</summary>
    /// <exception cref="InvalidOperationException">
    /// The lift layer has not been built yet. It is created in the layer-construction pass, so a call
    /// that reaches here first is a wiring bug rather than a user error.
    /// </exception>
    private Tensor<T> Lift(Tensor<T> band, int assets, int time, int featureCount)
    {
        var lift = _lift ?? throw new InvalidOperationException(
            $"{nameof(Stockformer<T>)}: the lift layer is not built. Construct the layers before a forward pass.");

        var flat = Engine.Reshape(band, new[] { assets * time, featureCount });
        return Engine.Reshape(
            lift.Forward(flat), new[] { assets, time, _options.HiddenDimension });
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
    private (Tensor<T> FusedLast, Tensor<T> LowLast) ForwardCore(Tensor<T> features, out int assets)
    {
        if (features is null) throw new ArgumentNullException(nameof(features));

        (assets, int time, int featureCount) = ReadShape(features);

        // Eq. 2-4: the DWT is applied to the FEATURE tensor, per (asset, feature) series over time.
        // An earlier revision decomposed a single scalar return per asset-timestep and lifted it with
        // DenseLayer(1 -> width). That is a RANK-ONE projection: all width dimensions become scalar
        // multiples of one number, so the model had almost no capacity and could not fit its training
        // data (observed as training MSE ABOVE test MSE). The paper's input is D = 360 price-volume
        // factors per asset per timestep, not one return.
        var (low, high) = SplitFeatureTensor(features, assets, time, featureCount);

        // Project D features to the model width. With D > 1 this is a real linear map rather than a
        // rank-one bottleneck.
        var lowLifted = Lift(low, assets, time, featureCount);
        var highLifted = Lift(high, assets, time, featureCount);

        // Eq. 10 needs BOTH embeddings: rho^spa (structural, per asset) and rho^tem (per timestep).
        var (fused, lowEncoded) = Encoder.Encode(
            lowLifted, highLifted,
            ResolveEmbedding(assets, _options.HiddenDimension),
            TemporalEmbedding(time));

        return (LastStep(fused, assets, time), LastStep(lowEncoded, assets, time));
    }

    /// <summary>
    /// Reads <c>[assets, time, features]</c>, accepting <c>[assets, time]</c> as a single-feature case.
    /// </summary>
    private (int Assets, int Time, int Features) ReadShape(Tensor<T> input)
    {
        int rank = input.Shape.Length;

        // [batch, depth, sequence, features] — what a ThreeDimensional architecture declaration causes
        // the family to feed. The batch axis stands in for the asset cross-section: a single-asset batch
        // is a legitimate degenerate case, and the graph/embedding resolve to their neutral values.
        if (rank == 4)
        {
            if (input.Shape[1] != 1)
            {
                throw new ArgumentException(
                    $"Input depth is {input.Shape[1]}; this model expects depth 1 because its input is a " +
                    "[sequence, features] window per asset, with no third spatial axis.", nameof(input));
            }
            return (input.Shape[0], input.Shape[2], input.Shape[3]);
        }

        if (rank == 3) return (input.Shape[0], input.Shape[1], input.Shape[2]);
        if (rank == 2) return (input.Shape[0], input.Shape[1], 1);

        var dims = new int[rank];
        for (int d = 0; d < rank; d++) dims[d] = input.Shape[d];
        throw new ArgumentException(
            $"Expected [assets, time, features], [batch, 1, time, features] or [assets, time]; got " +
            $"[{string.Join(", ", dims)}].", nameof(input));
    }

    /// <summary>
    /// Applies the DWT independently to every (asset, feature) series along time.
    /// </summary>
    /// <remarks>
    /// Eq. 3-4 filter the feature tensor, so each of the D factor series is decomposed separately
    /// rather than the transform being applied to one aggregate return.
    /// </remarks>
    private (Tensor<T> Low, Tensor<T> High) SplitFeatureTensor(
        Tensor<T> features, int assets, int time, int featureCount)
    {
        int bandLength = _bands.BandLength(time);
        if (bandLength == 0)
        {
            var dims = new int[features.Shape.Length];
            for (int d = 0; d < dims.Length; d++) dims[d] = features.Shape[d];
            throw new ArgumentException(
                $"An input window of {time} timesteps decomposes to a zero-length band at " +
                $"{_options.WaveletLevels} level(s). Input shape was [{string.Join(", ", dims)}], read as " +
                $"assets={assets}, time={time}, features={featureCount}. Lengthen the window, or the " +
                "axis order is not what this model expects.", nameof(features));
        }

        var series = new Vector<T>(time);
        var lowBands = new Matrix<T>(assets * featureCount, bandLength);
        var highBands = new Matrix<T>(assets * featureCount, bandLength);

        for (int a = 0; a < assets; a++)
        {
            for (int f = 0; f < featureCount; f++)
            {
                for (int t = 0; t < time; t++)
                    series[t] = features[((a * time) + t) * featureCount + f];

                var (lo, hi) = _bands.Split(series);
                int row = (a * featureCount) + f;
                for (int t = 0; t < bandLength; t++)
                {
                    lowBands[row, t] = t < lo.Length ? lo[t] : Ops.Zero;
                    highBands[row, t] = t < hi.Length ? hi[t] : Ops.Zero;
                }
            }
        }

        // Eq. 3-4: learnable inverse filters restore each band to the FULL window length.
        return (Restore(lowBands, _lowUpsample!, assets, time, featureCount),
                Restore(highBands, _highUpsample!, assets, time, featureCount));
    }

    /// <summary>
    /// Eq. 3-4: learnable upsampling of every band series back to <paramref name="time"/> steps.
    /// </summary>
    private Tensor<T> Restore(Matrix<T> bands, ILayer<T> filter, int assets, int time, int featureCount)
    {
        int rows = bands.Rows;
        var input = new Tensor<T>(new[] { rows, bands.Columns });
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < bands.Columns; c++) input[(r * bands.Columns) + c] = bands[r, c];

        var restored = filter.Forward(input);   // [rows, time]

        // Back to [assets, time, features].
        var result = new Tensor<T>(new[] { assets, time, featureCount });
        for (int a = 0; a < assets; a++)
        {
            for (int f = 0; f < featureCount; f++)
            {
                int row = (a * featureCount) + f;
                for (int t = 0; t < time; t++)
                    result[((a * time) + t) * featureCount + f] = restored[(row * time) + t];
            }
        }
        return result;
    }

    /// <summary>rho^tem: sinusoidal position encoding over the full window.</summary>
    private Matrix<T> TemporalEmbedding(int time)
    {
        int width = _options.HiddenDimension;
        var te = new Matrix<T>(time, width);
        for (int t = 0; t < time; t++)
        {
            for (int f = 0; f < width; f++)
            {
                double angle = t / Math.Pow(10000.0, 2.0 * (f / 2) / width);
                te[t, f] = Ops.FromDouble(f % 2 == 0 ? Math.Sin(angle) : Math.Cos(angle));
            }
        }
        return te;
    }
}
