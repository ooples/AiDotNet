using AiDotNet.Attributes;
using AiDotNet.Enums;
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
public class Stockformer<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    private readonly StockformerOptions<T> _options;
    private readonly StockformerBands<T> _bands;
    private readonly StockformerDualEncoder<T> _encoder;

    // Input projection: raw per-stock band value -> model width.
    private readonly Matrix<T> _inputProjection;

    // The two output heads, applied to BOTH representations (four outputs total).
    private readonly Matrix<T> _returnHead;
    private readonly Matrix<T> _directionHead;

    /// <summary>
    /// Gets or sets the stock similarity graph, <c>[stocks, stocks]</c>.
    /// </summary>
    /// <remarks>
    /// The reference derives this from struc2vec embeddings over a correlation graph and supplies it
    /// as data (<c>adjgat</c>); it is not learned. When unset, an identity graph is used so each stock
    /// sees only itself — which disables the paper's cross-sectional mechanism, so set it for any real
    /// use.
    /// </remarks>
    public Matrix<T>? Adjacency { get; set; }

    /// <summary>Gets the configuration in use.</summary>
    public StockformerOptions<T> Options => _options;

    /// <summary>Gets the band splitter.</summary>
    public StockformerBands<T> Bands => _bands;

    /// <summary>
    /// Creates a Stockformer.
    /// </summary>
    /// <param name="options">Configuration; defaults follow the reference config.</param>
    public Stockformer(StockformerOptions<T>? options = null)
    {
        _options = options ?? new StockformerOptions<T>();

        if (_options.HiddenDimension <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), _options.HiddenDimension,
                "HiddenDimension must be positive.");
        if (_options.NumDirectionClasses <= 1)
            throw new ArgumentOutOfRangeException(nameof(options), _options.NumDirectionClasses,
                "NumDirectionClasses must be at least 2 for the classification task to be meaningful.");

        // The reference config's seed is 1; ModelOptions.Seed wins when the caller sets it.
        var random = new Random(_options.Seed ?? 1);

        _bands = new StockformerBands<T>(_options.WaveletOrder, _options.WaveletLevels);
        _encoder = new StockformerDualEncoder<T>(
            _options.HiddenDimension, _options.SpatialSamples, kernelWidth: 3, random);

        double scale = 1.0 / Math.Sqrt(Math.Max(1, _options.HiddenDimension));
        _inputProjection = Random(1, _options.HiddenDimension, scale, random);
        _returnHead = Random(_options.HiddenDimension, 1, scale, random);
        _directionHead = Random(_options.HiddenDimension, _options.NumDirectionClasses, scale, random);
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

        int stocks = perStockReturns.Rows;
        if (stocks == 0)
            throw new ArgumentException("At least one stock is required.", nameof(perStockReturns));

        // 1. Split into low/high bands. Note the band is HALF the input length per level.
        var (lowMatrix, highMatrix) = _bands.SplitAll(perStockReturns);
        int time = lowMatrix.Columns;
        if (time == 0)
            throw new ArgumentException(
                $"An input window of {perStockReturns.Columns} timesteps decomposes to a zero-length " +
                $"band at {_options.WaveletLevels} level(s). Lengthen the window.", nameof(perStockReturns));

        // 2. Lift each scalar band value to the model width.
        var low = Lift(lowMatrix, stocks, time);
        var high = Lift(highMatrix, stocks, time);

        // 3. Encode and fuse.
        var adjacency = Adjacency ?? Identity(stocks);
        var temporalEmbedding = TemporalEmbedding(time);
        var (fused, lowEncoded) = _encoder.Encode(low, high, temporalEmbedding, adjacency);

        // 4. Both heads on both representations.
        return new Prediction(
            Head(fused, stocks, time, _returnHead, 1),
            Head(lowEncoded, stocks, time, _returnHead, 1),
            Head(fused, stocks, time, _directionHead, _options.NumDirectionClasses),
            Head(lowEncoded, stocks, time, _directionHead, _options.NumDirectionClasses));
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
    public ModelMetadata<T> GetModelMetadata()
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
        return metadata;
    }

    private Tensor<T> Lift(Matrix<T> band, int stocks, int time)
    {
        int width = _options.HiddenDimension;
        var lifted = new Tensor<T>(new[] { stocks, time, width });
        for (int s = 0; s < stocks; s++)
        {
            for (int t = 0; t < time; t++)
            {
                double v = Ops.ToDouble(band[s, t]);
                int baseIndex = ((s * time) + t) * width;
                for (int f = 0; f < width; f++)
                    lifted[baseIndex + f] = Ops.FromDouble(v * Ops.ToDouble(_inputProjection[0, f]));
            }
        }
        return lifted;
    }

    /// <summary>
    /// Applies a head to the LAST timestep of each stock's representation.
    /// </summary>
    /// <remarks>
    /// The final step is the one carrying the whole encoded window, and it is the position a forecast
    /// is made from.
    /// </remarks>
    private Vector<T> Head(Tensor<T> representation, int stocks, int time, Matrix<T> head, int outputs)
    {
        int width = _options.HiddenDimension;
        var result = new Vector<T>(stocks * outputs);
        int last = time - 1;

        for (int s = 0; s < stocks; s++)
        {
            int baseIndex = ((s * time) + last) * width;
            for (int o = 0; o < outputs; o++)
            {
                double acc = 0.0;
                for (int f = 0; f < width; f++)
                    acc += Ops.ToDouble(representation[baseIndex + f]) * Ops.ToDouble(head[f, o]);
                result[(s * outputs) + o] = Ops.FromDouble(acc);
            }
        }
        return result;
    }

    /// <summary>Sinusoidal position encoding over the BAND length, not the input window.</summary>
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

    private static Matrix<T> Identity(int n)
    {
        var m = new Matrix<T>(n, n);
        for (int i = 0; i < n; i++) m[i, i] = Ops.One;
        return m;
    }

    private static Matrix<T> Random(int rows, int columns, double scale, Random random)
    {
        var m = new Matrix<T>(rows, columns);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < columns; c++)
                m[r, c] = Ops.FromDouble((random.NextDouble() * 2.0 - 1.0) * scale);
        return m;
    }
}
