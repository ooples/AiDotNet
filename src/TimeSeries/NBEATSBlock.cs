using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Enums;

namespace AiDotNet.TimeSeries;

/// <summary>
/// Represents a single block in the N-BEATS architecture.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Each N-BEATS block consists of:
/// 1. A stack of fully connected layers (the "theta" network)
/// 2. A basis expansion layer for generating backcast (reconstruction of input)
/// 3. A basis expansion layer for generating forecast (prediction of future)
/// </para>
/// <para>
/// The block architecture implements a doubly residual stacking principle:
/// - Backcast residual: Input minus backcast is passed to the next block
/// - Forecast addition: Forecasts from all blocks are summed for the final prediction
/// </para>
/// <para><b>For Beginners:</b> A block is the basic building unit of N-BEATS. Think of it like
/// a specialized predictor that:
/// 1. Looks at the input time series
/// 2. Tries to reconstruct what it saw (backcast)
/// 3. Predicts the future (forecast)
/// 4. Passes the "leftover" patterns it couldn't explain to the next block
///
/// Multiple blocks work together, with each one focusing on different aspects of the data.
/// </para>
/// </remarks>
// Rank 1 only, and that is this block's own declaration rather than a simplification: the base
// constructor is handed CreateInputShape(lookbackWindow) => new[] { lookbackWindow } and
// CreateOutputShape(...) => new[] { lookbackWindow + forecastHorizon }, both one-dimensional.
// ForwardTraced backs that up - its first statement is Engine.Reshape(input, [_lookbackWindow, 1]),
// which only succeeds for an input holding exactly _lookbackWindow elements.
//
// The axis is Time on both sides because both ARE time: the input is a lookback window of history, and
// the output is the concatenation [backcast(lookbackWindow) | forecast(forecastHorizon)] documented on
// ForwardTraced. Naming it Features would suggest the two ends are unrelated quantities when the whole
// doubly-residual scheme depends on their being the same series.
//
// ForwardTape also accepts rank-2 [B, L], but it is a separate public entry point used by NBEATSModel,
// NOT the LayerBase forward path this contract describes - so no rank-2 layout is declared here.
[TensorLayout(TensorAxis.Time, Direction = TensorLayoutDirection.Input,
    Note = "A lookback window of exactly lookbackWindow steps.")]
[TensorLayout(TensorAxis.Time, Direction = TensorLayoutDirection.Output,
    Note = "Backcast and forecast concatenated: [lookbackWindow | forecastHorizon].")]
[AutoParameters]
internal partial class NBEATSBlock<T> : NeuralNetworks.Layers.LayerBase<T>, IShapeContract
{
    private readonly int _lookbackWindow;
    private readonly int _forecastHorizon;
    private readonly int _hiddenLayerSize;
    private readonly int _numHiddenLayers;
    private readonly int _thetaSizeBackcast;
    private readonly int _thetaSizeForecast;
    private readonly bool _useInterpretableBasis;
    private readonly int _polynomialDegree;

    /// <summary>
    /// Initializes a new instance with default settings.
    /// </summary>
    public NBEATSBlock()
        : this(64, 16, 128, 4, 64, 16, false)
    {
    }

    /// <summary>
    /// Weights for the fully connected layers (theta network), stored as Tensor&lt;T&gt;
    /// for tape-based automatic differentiation.
    /// </summary>
    private List<Tensor<T>> _fcWeights;

    /// <summary>
    /// Biases for the fully connected layers (theta network), stored as Tensor&lt;T&gt;
    /// for tape-based automatic differentiation.
    /// </summary>
    private List<Tensor<T>> _fcBiases;

    /// <summary>
    /// Precomputed basis matrix for backcast expansion: [lookbackWindow, thetaSizeBackcast].
    /// </summary>
    private Tensor<T> _basisBackcast;

    /// <summary>
    /// Precomputed basis matrix for forecast expansion: [forecastHorizon, thetaSizeForecast].
    /// </summary>
    private Tensor<T> _basisForecast;

    /// <summary>
    /// Initializes a new instance of the NBEATSBlock class.
    /// </summary>
    /// <param name="lookbackWindow">The number of historical time steps used as input.</param>
    /// <param name="forecastHorizon">The number of future time steps to predict.</param>
    /// <param name="hiddenLayerSize">The size of hidden layers in the fully connected network.</param>
    /// <param name="numHiddenLayers">The number of hidden layers.</param>
    /// <param name="thetaSizeBackcast">The size of the theta vector for backcast basis expansion.</param>
    /// <param name="thetaSizeForecast">The size of the theta vector for forecast basis expansion.</param>
    /// <param name="useInterpretableBasis">Whether to use interpretable basis functions.</param>
    /// <param name="polynomialDegree">The polynomial degree for trend basis (if interpretable).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a new block with specific parameters:
    /// - lookbackWindow: How far back in time the block looks
    /// - forecastHorizon: How far forward in time the block predicts
    /// - hiddenLayerSize: How many neurons in each hidden layer (bigger = more capacity)
    /// - numHiddenLayers: How many hidden layers (deeper = more complex patterns)
    /// - useInterpretableBasis: Whether to use human-understandable basis functions
    /// </para>
    /// </remarks>
    /// <summary>
    /// Validates <paramref name="lookbackWindow"/> and returns the corresponding
    /// LayerBase input shape. Runs BEFORE the base ctor so invalid values surface
    /// as <see cref="ArgumentException"/> with the argument name instead of a
    /// downstream shape error.
    /// </summary>
    private static int[] CreateInputShape(int lookbackWindow)
    {
        if (lookbackWindow <= 0)
        {
            throw new ArgumentException("Lookback window must be positive.", nameof(lookbackWindow));
        }
        return new[] { lookbackWindow };
    }

    /// <summary>
    /// Validates <paramref name="forecastHorizon"/> (and re-checks lookback for
    /// consistency) and returns the corresponding LayerBase output shape.
    /// </summary>
    private static int[] CreateOutputShape(int lookbackWindow, int forecastHorizon)
    {
        if (lookbackWindow <= 0)
        {
            throw new ArgumentException("Lookback window must be positive.", nameof(lookbackWindow));
        }
        if (forecastHorizon <= 0)
        {
            throw new ArgumentException("Forecast horizon must be positive.", nameof(forecastHorizon));
        }
        return new[] { lookbackWindow + forecastHorizon };
    }

    public NBEATSBlock(
        int lookbackWindow,
        int forecastHorizon,
        int hiddenLayerSize,
        int numHiddenLayers,
        int thetaSizeBackcast,
        int thetaSizeForecast,
        bool useInterpretableBasis,
        int polynomialDegree = 3)
        : base(
            CreateInputShape(lookbackWindow),
            CreateOutputShape(lookbackWindow, forecastHorizon))
    {
        // Primary-argument validation happens inside the static shape factories
        // above so `lookbackWindow` / `forecastHorizon` are rejected BEFORE
        // LayerBase<T> consumes them — users see the nameof(...)-tagged
        // ArgumentException instead of a downstream shape error from the base.
        // (The two blocks that previously validated those here are now in
        // CreateInputShape / CreateOutputShape below.)
        if (hiddenLayerSize <= 0)
        {
            throw new ArgumentException("Hidden layer size must be positive.", nameof(hiddenLayerSize));
        }
        if (numHiddenLayers <= 0)
        {
            throw new ArgumentException("Number of hidden layers must be positive.", nameof(numHiddenLayers));
        }
        if (thetaSizeBackcast <= 0)
        {
            throw new ArgumentException("Backcast theta size must be positive.", nameof(thetaSizeBackcast));
        }
        if (thetaSizeForecast <= 0)
        {
            throw new ArgumentException("Forecast theta size must be positive.", nameof(thetaSizeForecast));
        }
        if (useInterpretableBasis && polynomialDegree < 0)
        {
            throw new ArgumentException("Polynomial degree must be non-negative for interpretable basis.", nameof(polynomialDegree));
        }
        // Interpretable-basis builders cap usable theta at polynomialDegree + 1
        // (ComputeBasisTensor populates only that many rows; ApplyBasisExpansion
        // slices to the same count). Silently accepting oversized theta sizes
        // would allocate trainable weights that are mathematically disconnected
        // from the output — dead parameters that waste memory and mask bugs
        // during gradient checks.
        if (useInterpretableBasis && thetaSizeBackcast > polynomialDegree + 1)
        {
            throw new ArgumentException(
                $"Backcast theta size ({thetaSizeBackcast}) cannot exceed polynomialDegree + 1 ({polynomialDegree + 1}) for interpretable basis.",
                nameof(thetaSizeBackcast));
        }
        if (useInterpretableBasis && thetaSizeForecast > polynomialDegree + 1)
        {
            throw new ArgumentException(
                $"Forecast theta size ({thetaSizeForecast}) cannot exceed polynomialDegree + 1 ({polynomialDegree + 1}) for interpretable basis.",
                nameof(thetaSizeForecast));
        }

        _lookbackWindow = lookbackWindow;
        _forecastHorizon = forecastHorizon;
        _hiddenLayerSize = hiddenLayerSize;
        _numHiddenLayers = numHiddenLayers;
        _thetaSizeBackcast = thetaSizeBackcast;
        _thetaSizeForecast = thetaSizeForecast;
        _useInterpretableBasis = useInterpretableBasis;
        _polynomialDegree = polynomialDegree;

        _fcWeights = new List<Tensor<T>>();
        _fcBiases = new List<Tensor<T>>();

        if (_useInterpretableBasis)
        {
            // Interpretable blocks: fixed polynomial basis (not trainable)
            // Per Oreshkin et al. 2020 Section 3.3
            _basisBackcast = ComputeBasisTensor(_thetaSizeBackcast, _lookbackWindow);
            _basisForecast = ComputeBasisTensor(_thetaSizeForecast, _forecastHorizon);
        }
        else
        {
            // Generic blocks: V_b and V_f are fully learnable linear functions.
            // Per Oreshkin et al. 2020 Section 3.2:
            // "In the generic architecture, we do not restrict g^b and g^f to a
            //  particular functional form, and instead make them fully learnable"
            // Initialize near identity for stable initial behavior.
            var data_b = new T[_lookbackWindow * _thetaSizeBackcast];
            var data_f = new T[_forecastHorizon * _thetaSizeForecast];
            for (int i = 0; i < _lookbackWindow; i++)
                for (int j = 0; j < _thetaSizeBackcast; j++)
                    data_b[i * _thetaSizeBackcast + j] = (i == j) ? NumOps.One : NumOps.Zero;
            for (int i = 0; i < _forecastHorizon; i++)
                for (int j = 0; j < _thetaSizeForecast; j++)
                    data_f[i * _thetaSizeForecast + j] = (i == j) ? NumOps.One : NumOps.Zero;
            _basisBackcast = new Tensor<T>(data_b, new[] { _lookbackWindow, _thetaSizeBackcast });
            _basisForecast = new Tensor<T>(data_f, new[] { _forecastHorizon, _thetaSizeForecast });
        }

        InitializeWeights();
    }

    /// <summary>
    /// Initializes the weights and biases for the fully connected layers.
    /// Uses He initialization for ReLU networks and registers all parameters as trainable
    /// for tape-based autodiff.
    /// </summary>
    private void InitializeWeights()
    {
        var random = RandomHelper.CreateSeededRandom(42);

        // First layer: lookbackWindow -> hiddenLayerSize
        int inputSize = _lookbackWindow;
        double stddev = Math.Sqrt(2.0 / inputSize);
        var weight = CreateWeightTensor(_hiddenLayerSize, inputSize, stddev, random);
        _fcWeights.Add(weight);
        RegisterTrainableParameter(weight, PersistentTensorRole.Weights);

        var bias = CreateBiasTensor(_hiddenLayerSize, 0.01);
        _fcBiases.Add(bias);
        RegisterTrainableParameter(bias, PersistentTensorRole.Biases);

        // Hidden layers: hiddenLayerSize -> hiddenLayerSize
        for (int layer = 1; layer < _numHiddenLayers; layer++)
        {
            stddev = Math.Sqrt(2.0 / _hiddenLayerSize);
            weight = CreateWeightTensor(_hiddenLayerSize, _hiddenLayerSize, stddev, random);
            _fcWeights.Add(weight);
            RegisterTrainableParameter(weight, PersistentTensorRole.Weights);

            bias = CreateBiasTensor(_hiddenLayerSize, 0.01);
            _fcBiases.Add(bias);
            RegisterTrainableParameter(bias, PersistentTensorRole.Biases);
        }

        // Output layer for backcast theta: hiddenLayerSize -> thetaSizeBackcast
        stddev = Math.Sqrt(2.0 / (_hiddenLayerSize + _thetaSizeBackcast));
        weight = CreateWeightTensor(_thetaSizeBackcast, _hiddenLayerSize, stddev, random);
        _fcWeights.Add(weight);
        RegisterTrainableParameter(weight, PersistentTensorRole.Weights);

        bias = CreateBiasTensor(_thetaSizeBackcast, 0.0);
        _fcBiases.Add(bias);
        RegisterTrainableParameter(bias, PersistentTensorRole.Biases);

        // Output layer for forecast theta: hiddenLayerSize -> thetaSizeForecast
        stddev = Math.Sqrt(2.0 / (_hiddenLayerSize + _thetaSizeForecast));
        weight = CreateWeightTensor(_thetaSizeForecast, _hiddenLayerSize, stddev, random);
        _fcWeights.Add(weight);
        RegisterTrainableParameter(weight, PersistentTensorRole.Weights);

        bias = CreateBiasTensor(_thetaSizeForecast, 0.0);
        _fcBiases.Add(bias);
        RegisterTrainableParameter(bias, PersistentTensorRole.Biases);

        // For generic blocks: register V_b and V_f as trainable
        // Per Oreshkin et al. 2020 Section 3.2
        if (!_useInterpretableBasis)
        {
            RegisterTrainableParameter(_basisBackcast, PersistentTensorRole.Weights);
            RegisterTrainableParameter(_basisForecast, PersistentTensorRole.Weights);
        }
    }

    /// <summary>
    /// Creates a weight tensor with He initialization.
    /// </summary>
    private Tensor<T> CreateWeightTensor(int rows, int cols, double stddev, Random random)
    {
        var data = new T[rows * cols];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = NumOps.FromDouble(random.NextDouble() * stddev * 2 - stddev);
        }
        return new Tensor<T>(new[] { rows, cols }, new Vector<T>(data));
    }

    /// <summary>
    /// Creates a bias tensor initialized to a constant value.
    /// </summary>
    private Tensor<T> CreateBiasTensor(int size, double initValue)
    {
        var data = new T[size];
        for (int i = 0; i < size; i++)
        {
            data[i] = NumOps.FromDouble(initValue);
        }
        // Store column-shaped [size, 1] up-front so ForwardTape can feed the bias
        // straight into TensorAdd ([hidden, B] + [hidden, 1]) without an
        // Engine.Reshape on every forward pass (re-profile #4: the per-forward
        // reshape node was ~1% of driver wall). Same contiguous data as [size],
        // so gradient flow / optimizer moments are bit-identical.
        return new Tensor<T>(new[] { size, 1 }, new Vector<T>(data));
    }

    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// Hand-written rather than generated, because input and output share a rank AND an axis role while
    /// differing in SIZE - exactly the case a generated <c>Same(Time)</c> would get wrong. The block
    /// emits a longer series than it consumes: <c>CreateOutputShape</c> returns
    /// <c>new[] { lookbackWindow + forecastHorizon }</c>, and <c>ForwardTraced</c> produces it by
    /// concatenating a <c>[lookbackWindow]</c> backcast with a <c>[forecastHorizon]</c> forecast along
    /// axis 0.
    /// </para>
    /// <para>
    /// <c>Fixed</c> rather than a window or a scale, because the output length is set by construction and
    /// not by the input. Both basis matrices are precomputed at their configured extents
    /// (<c>_basisBackcast</c> is <c>[lookbackWindow, thetaSizeBackcast]</c>, <c>_basisForecast</c> is
    /// <c>[forecastHorizon, thetaSizeForecast]</c>), so the two halves come out at those sizes whatever
    /// is fed in - and the reshape at the top of <c>ForwardTraced</c> already refuses any input that is
    /// not exactly <c>_lookbackWindow</c> long.
    /// </para>
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        // Rank 1 is the only form the LayerBase path accepts; see the layout note on the class.
        if (inputRank != 1 || _lookbackWindow <= 0 || _forecastHorizon <= 0) return null;

        return new[]
        {
            new OutputAxisContract(
                TensorAxis.Time, AxisRelation.Fixed(_lookbackWindow + _forecastHorizon)),
        };
    }

    /// <summary>
    /// LayerBase Forward -- uses tape-tracked Engine operations for automatic differentiation.
    /// Output tensor layout: [backcast(lookbackWindow) | forecast(forecastHorizon)].
    /// </summary>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        // Use Engine.Reshape for tape-tracked reshaping
        var x = Engine.Reshape(input, [_lookbackWindow, 1]);

        // Pass through hidden layers with ReLU
        for (int layer = 0; layer < _numHiddenLayers; layer++)
        {
            // Linear: y = W @ x + b
            var linear = Engine.TensorMatMul(_fcWeights[layer], x);
            // Add bias: reshape bias to column [hidden, 1]
            var biasCol = Engine.Reshape(_fcBiases[layer], [_hiddenLayerSize, 1]);
            linear = Engine.TensorAdd(linear, biasCol);
            // ReLU activation
            x = Engine.ReLU(linear);
        }

        // Compute theta for backcast: [thetaSizeBackcast, 1]
        int backcastLayerIdx = _numHiddenLayers;
        var thetaBackcast = Engine.TensorMatMul(_fcWeights[backcastLayerIdx], x);
        var bcBiasCol = Engine.Reshape(_fcBiases[backcastLayerIdx], [_thetaSizeBackcast, 1]);
        thetaBackcast = Engine.TensorAdd(thetaBackcast, bcBiasCol);

        // Compute theta for forecast: [thetaSizeForecast, 1]
        int forecastLayerIdx = _numHiddenLayers + 1;
        var thetaForecast = Engine.TensorMatMul(_fcWeights[forecastLayerIdx], x);
        var fcBiasCol = Engine.Reshape(_fcBiases[forecastLayerIdx], [_thetaSizeForecast, 1]);
        thetaForecast = Engine.TensorAdd(thetaForecast, fcBiasCol);

        // Basis expansion: backcast = B_backcast @ theta_backcast
        var backcast = Engine.TensorMatMul(_basisBackcast, thetaBackcast); // [lookbackWindow, 1]
        // Basis expansion: forecast = B_forecast @ theta_forecast
        var forecast = Engine.TensorMatMul(_basisForecast, thetaForecast); // [forecastHorizon, 1]

        // Concatenate backcast and forecast into output: flatten to 1D
        var backcastFlat = Engine.Reshape(backcast, [_lookbackWindow]);
        var forecastFlat = Engine.Reshape(forecast, [_forecastHorizon]);

        // Engine.TensorConcatenate along axis 0 is a 1:1 replacement for the scalar
        // copy loop: it produces the same [lookbackWindow + forecastHorizon] 1D tensor
        // by copying backcastFlat elements followed by forecastFlat elements.
        var output = Engine.TensorConcatenate([backcastFlat, forecastFlat], axis: 0);

        return output;
    }

    /// <summary>
    /// Tape-tracked forward pass that returns separate backcast and forecast tensors.
    /// Used by the NBEATSModel during training for residual block-by-block processing.
    /// Accepts rank-1 <c>[L]</c> (single sample) or rank-2 <c>[B, L]</c> (batch)
    /// input; output ranks match the input rank so the caller can thread the
    /// residual through the stack without reshaping between blocks.
    /// </summary>
    public (Tensor<T> backcast, Tensor<T> forecast) ForwardTape(Tensor<T> input)
    {
        int inputRank = input.Rank;
        int batchSize = inputRank == 2 ? input.Shape[0] : 1;

        // Canonicalize to column-major form [L, B] so weight @ x maps
        // straight to [hidden, B] without transposes per layer — this is
        // the vectorization Oreshkin et al. 2019 §3.3 describes (batched
        // Adam updates). For single-sample inputs we collapse to [L, 1].
        Tensor<T> x;
        if (inputRank == 2)
        {
            // [B, L] -> [L, B] via permute so matmul (weight [hidden, L] @ x)
            // yields [hidden, B], one column per sample.
            x = Engine.TensorPermute(input, new[] { 1, 0 });
        }
        else
        {
            x = Engine.Reshape(input, [_lookbackWindow, 1]);
        }

        // Hidden layers: y = ReLU(W x + b). Bias broadcasts across B columns.
        // Engine.TensorAdd handles the [hidden, 1] -> [hidden, B]
        // broadcast natively; TensorAdd requires shapes to match exactly
        // and would otherwise throw at the batched ([hidden, B>1]) call
        // sites.
        for (int layer = 0; layer < _numHiddenLayers; layer++)
        {
            var linear = Engine.TensorMatMul(_fcWeights[layer], x);  // [hidden, B]
            // Biases are stored column-shaped [hidden, 1] (see CreateBiasTensor), so
            // they feed TensorAdd directly — no per-forward Engine.Reshape.
            linear = Engine.TensorAdd(linear, _fcBiases[layer]);
            x = Engine.ReLU(linear);
        }

        // theta_backcast = W_bc x + b_bc         shape [theta_bc, B]
        int backcastLayerIdx = _numHiddenLayers;
        var thetaBackcast = Engine.TensorMatMul(_fcWeights[backcastLayerIdx], x);
        thetaBackcast = Engine.TensorAdd(thetaBackcast, _fcBiases[backcastLayerIdx]);

        // theta_forecast = W_fc x + b_fc        shape [theta_fc, B]
        int forecastLayerIdx = _numHiddenLayers + 1;
        var thetaForecast = Engine.TensorMatMul(_fcWeights[forecastLayerIdx], x);
        thetaForecast = Engine.TensorAdd(thetaForecast, _fcBiases[forecastLayerIdx]);

        // Basis expansion (paper §3.3): backcast = V_b @ theta_bc,
        // forecast = V_f @ theta_fc. Output shapes [L, B] and [H, B].
        var backcastRaw = Engine.TensorMatMul(_basisBackcast, thetaBackcast);    // [L, B]
        var forecastRaw = Engine.TensorMatMul(_basisForecast, thetaForecast);    // [H, B]

        if (inputRank == 2)
        {
            // Restore [B, L] and [B, H] so caller sees the same rank as input.
            var backcast = Engine.TensorPermute(backcastRaw, new[] { 1, 0 });
            var forecast = Engine.TensorPermute(forecastRaw, new[] { 1, 0 });
            return (backcast, forecast);
        }
        else
        {
            // Single-sample path (test / inference-via-training hooks):
            // drop the trailing singleton dim.
            var backcast = Engine.Reshape(backcastRaw, [_lookbackWindow]);
            var forecast = Engine.Reshape(forecastRaw, [_forecastHorizon]);
            return (backcast, forecast);
        }
    }

    public override bool SupportsTraining => true;

    public override void ResetState() { /* stateless layer -- no recurrent state to reset */ }

    /// <summary>
    /// Throws <see cref="InvalidOperationException"/>: this block is trained
    /// through the tape-based optimizer path and has no eager scalar-step update.
    /// </summary>
    /// <remarks>
    /// N-BEATS blocks register their parameters via <c>RegisterTrainableParameter</c>
    /// and are updated by the compiled training plan that <see cref="CompiledTapeTrainingStep{T}"/>
    /// drives. Calling <c>UpdateParameters(learningRate)</c> directly bypasses
    /// that path and would silently lose updates, so fail fast to catch the
    /// misuse at the training boundary rather than later as a silent accuracy
    /// regression.
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        throw new InvalidOperationException(
            $"{nameof(NBEATSBlock<T>)} uses tape-based optimization. " +
            "Update parameters through the optimizer / training step, " +
            "not directly via UpdateParameters(learningRate).");
    }

    /// <summary>
    /// Non-tape forward pass for inference (used by PredictSingle).
    /// Uses plain matrix/vector operations without tape overhead.
    /// </summary>
    public (Vector<T> backcast, Vector<T> forecast) ForwardInternal(Vector<T> input)
    {
        if (input.Length != _lookbackWindow)
        {
            throw new ArgumentException(
                $"Input length ({input.Length}) must match lookback window ({_lookbackWindow}).",
                nameof(input));
        }

        // Pass through fully connected layers with ReLU activation
        Vector<T> x = input.Clone();

        for (int layer = 0; layer < _numHiddenLayers; layer++)
        {
            // Linear transformation: y = Wx + b using tensor operations
            var xCol = new Tensor<T>(new[] { x.Length, 1 }, x);
            var wxResult = Engine.TensorMatMul(_fcWeights[layer], xCol);
            Vector<T> linear = new Vector<T>(_hiddenLayerSize);
            var biasVec = _fcBiases[layer].ToVector();
            for (int i = 0; i < _hiddenLayerSize; i++)
            {
                linear[i] = NumOps.Add(biasVec[i], wxResult[i, 0]);
            }

            // ReLU activation
            x = new Vector<T>(linear.Length);
            for (int i = 0; i < linear.Length; i++)
            {
                x[i] = NumOps.GreaterThan(linear[i], NumOps.Zero) ? linear[i] : NumOps.Zero;
            }
        }

        // Compute theta for backcast
        int backcastLayerIdx = _numHiddenLayers;
        var xColTheta = new Tensor<T>(new[] { x.Length, 1 }, x);
        var bcWx = Engine.TensorMatMul(_fcWeights[backcastLayerIdx], xColTheta);
        var bcBiasVec = _fcBiases[backcastLayerIdx].ToVector();
        Vector<T> thetaBackcast = new Vector<T>(_thetaSizeBackcast);
        for (int i = 0; i < _thetaSizeBackcast; i++)
        {
            thetaBackcast[i] = NumOps.Add(bcBiasVec[i], bcWx[i, 0]);
        }

        // Compute theta for forecast
        int forecastLayerIdx = _numHiddenLayers + 1;
        var fcWx = Engine.TensorMatMul(_fcWeights[forecastLayerIdx], xColTheta);
        var fcBiasVec = _fcBiases[forecastLayerIdx].ToVector();
        Vector<T> thetaForecast = new Vector<T>(_thetaSizeForecast);
        for (int i = 0; i < _thetaSizeForecast; i++)
        {
            thetaForecast[i] = NumOps.Add(fcBiasVec[i], fcWx[i, 0]);
        }

        // Apply basis expansion. Pass the matching basis tensor so generic
        // blocks multiply by their learned V_b / V_f matrices (keeping this
        // path consistent with Forward() / ForwardTape() and with the
        // parameter export/import of _basisBackcast / _basisForecast).
        Vector<T> backcast = ApplyBasisExpansion(thetaBackcast, _basisBackcast, _lookbackWindow);
        Vector<T> forecast = ApplyBasisExpansion(thetaForecast, _basisForecast, _forecastHorizon);

        return (backcast, forecast);
    }

    /// <summary>
    /// Computes the basis matrix as a Tensor for tape-tracked operations.
    /// Shape: [outputLength, thetaSize].
    /// </summary>
    private Tensor<T> ComputeBasisTensor(int thetaSize, int outputLength)
    {
        var data = new T[outputLength * thetaSize];

        if (_useInterpretableBasis)
        {
            for (int t = 0; t < outputLength; t++)
            {
                double tNormalized = (double)t / outputLength;
                for (int p = 0; p < Math.Min(thetaSize, _polynomialDegree + 1); p++)
                {
                    data[t * thetaSize + p] = NumOps.FromDouble(Math.Pow(tNormalized, p));
                }
            }
        }
        else
        {
            // Generic basis per Oreshkin et al. (2020): when thetaSize == outputLength,
            // theta IS the output directly (identity basis). When they differ, use a
            // simple identity-like mapping (1 on the diagonal, 0 elsewhere).
            for (int t = 0; t < outputLength; t++)
            {
                for (int k = 0; k < thetaSize; k++)
                {
                    data[t * thetaSize + k] = (t == k)
                        ? NumOps.One
                        : NumOps.Zero;
                }
            }
        }

        return new Tensor<T>(new[] { outputLength, thetaSize }, new Vector<T>(data));
    }

    /// <summary>
    /// Computes the basis matrix as a Matrix (for legacy operations).
    /// Shape: [outputLength, thetaSize].
    /// </summary>
    private Matrix<T> ComputeBasisMatrix(int thetaSize, int outputLength)
    {
        var basis = new Matrix<T>(outputLength, thetaSize);

        if (_useInterpretableBasis)
        {
            for (int t = 0; t < outputLength; t++)
            {
                double tNormalized = (double)t / outputLength;
                for (int p = 0; p < Math.Min(thetaSize, _polynomialDegree + 1); p++)
                {
                    basis[t, p] = NumOps.FromDouble(Math.Pow(tNormalized, p));
                }
            }
        }
        else
        {
            // Generic basis: identity matrix (theta IS the output)
            for (int t = 0; t < outputLength; t++)
            {
                for (int k = 0; k < thetaSize; k++)
                {
                    basis[t, k] = (t == k) ? NumOps.One : NumOps.Zero;
                }
            }
        }

        return basis;
    }

    /// <summary>
    /// Expands the theta coefficients into an output time series of the requested length.
    /// </summary>
    /// <param name="theta">The theta coefficient vector produced by the fc head.</param>
    /// <param name="basis">
    /// The basis matrix for the generic branch — shape [outputLength, theta.Length].
    /// Ignored when <see cref="_useInterpretableBasis"/> is <c>true</c> (the closed-form
    /// polynomial basis is computed on the fly from <see cref="_polynomialDegree"/>).
    /// </param>
    /// <param name="outputLength">Length of the expanded output vector.</param>
    private Vector<T> ApplyBasisExpansion(Vector<T> theta, Tensor<T> basis, int outputLength)
    {
        Vector<T> output = new Vector<T>(outputLength);

        if (_useInterpretableBasis)
        {
            for (int t = 0; t < outputLength; t++)
            {
                T value = NumOps.Zero;
                T tNormalized = NumOps.FromDouble((double)t / outputLength);

                for (int p = 0; p < Math.Min(theta.Length, _polynomialDegree + 1); p++)
                {
                    T power = NumOps.One;
                    for (int k = 0; k < p; k++)
                    {
                        power = NumOps.Multiply(power, tNormalized);
                    }
                    value = NumOps.Add(value, NumOps.Multiply(theta[p], power));
                }

                output[t] = value;
            }
        }
        else
        {
            // Generic basis: output = basis · theta. Must use the learned V_b/V_f
            // matrices per Oreshkin et al. 2020 Section 3.2 — they round-trip through
            // GetParameters/SetParameters as trainable weights, and the tape-based
            // Forward path multiplies by the same tensors. Returning theta directly
            // here (as the pre-fix code did) made PredictSingle diverge from both
            // training and model-load state.
            for (int t = 0; t < outputLength; t++)
            {
                T value = NumOps.Zero;
                for (int k = 0; k < theta.Length; k++)
                {
                    value = NumOps.Add(value, NumOps.Multiply(basis[t, k], theta[k]));
                }
                output[t] = value;
            }
        }

        return output;
    }

    /// <summary>
    /// Re-registers all weight and bias tensors as trainable parameters.
    /// Called after SetParameters replaces tensor instances.
    /// </summary>
    private void ReRegisterParameters()
    {
        // Clear and re-register (RegisterTrainableParameter handles dedup)
        foreach (var w in _fcWeights)
            RegisterTrainableParameter(w, PersistentTensorRole.Weights);
        foreach (var b in _fcBiases)
            RegisterTrainableParameter(b, PersistentTensorRole.Biases);

        // Generic blocks also learn the basis matrices — re-register them after
        // SetParameters replaces the tensor instances. Interpretable blocks use
        // fixed polynomial bases that are not trainable, so skip.
        if (!_useInterpretableBasis)
        {
            RegisterTrainableParameter(_basisBackcast, PersistentTensorRole.Weights);
            RegisterTrainableParameter(_basisForecast, PersistentTensorRole.Weights);
        }
    }

}
