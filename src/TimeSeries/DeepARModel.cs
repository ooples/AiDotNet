using AiDotNet.Helpers;
using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.LossFunctions;
using AiDotNet.Optimizers;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TimeSeries;

/// <summary>
/// Implements DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// DeepAR is a probabilistic forecasting model that produces full probability distributions
/// rather than point estimates. Key features include:
/// </para>
/// <list type="bullet">
/// <item>Autoregressive RNN architecture (LSTM-based)</item>
/// <item>Probabilistic forecasts with a Gaussian likelihood head (mean and scale)</item>
/// <item>Quantile predictions via Monte-Carlo sampling of the predictive path</item>
/// <item>Effective for cold-start scenarios</item>
/// </list>
/// <para>
/// Original paper: Salinas et al., "DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks" (2020).
/// </para>
/// <para>
/// <b>Training:</b> The LSTM recurrence is unrolled over the lookback window as
/// <see cref="GradientTape{T}"/>-tracked <c>Engine.Tensor*</c> operations, batched over
/// <c>[hiddenSize, batch]</c> column-major activations. Autodiff produces the full BPTT
/// backward pass (no hand-derived gradients), and the Adam optimizer applies the tape
/// gradients — so the same code path is GPU-dispatchable. Every timestep is supervised
/// one-step-ahead (teacher forcing) against the Gaussian negative log-likelihood.
/// </para>
/// <para><b>For Beginners:</b> DeepAR is like a weather forecaster that doesn't just say
/// "it will be 70 degrees tomorrow" but rather "there's a 50% chance it'll be between 65-75 degrees."
/// It uses an LSTM (a recurrent neural network good at remembering patterns over time) and predicts
/// a probability distribution (a mean and a spread) for each future step.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var options = new DeepAROptions&lt;double&gt;
/// {
///     LookbackWindow = 30, ForecastHorizon = 7,
///     HiddenSize = 40, NumLayers = 2
/// };
/// var deepar = new DeepARModel&lt;double&gt;(options);
/// deepar.Train(trainingMatrix, trainingLabels);
/// Vector&lt;double&gt; forecast = deepar.Predict(contextWindow);
/// </code>
/// </example>
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.RecurrentNetwork)]
[ModelTask(ModelTask.Forecasting)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks", "https://arxiv.org/abs/1704.04110", Year = 2020, Authors = "David Salinas, Valentin Flunkert, Jan Gasthaus, Tim Januschowski")]
public class DeepARModel<T> : TimeSeriesModelBase<T>
{
    private readonly DeepAROptions<T> _options;
    private readonly Random _random;
    private Vector<T> _trainingSeries = Vector<T>.Empty();

    // Tape-trainable LSTM cells (one per layer) and the Gaussian output head.
    private readonly List<DeepARLstmCellTape<T>> _lstmLayers;
    private DeepARGaussianHead<T> _head;

    // Normalization statistics computed during training (zero-mean / unit-variance of the
    // training series). Applied to inputs before the network and inverted on the network
    // output so gradient flow stays well-scaled — mirrors NBEATSModel / NHiTSModel.
    private T _normMean = MathHelper.GetNumericOperations<T>().Zero;
    private T _normStd = MathHelper.GetNumericOperations<T>().One;

    /// <summary>
    /// Initializes a new instance of the DeepARModel class.
    /// </summary>
    /// <param name="options">Configuration options for DeepAR.</param>
    public DeepARModel(DeepAROptions<T>? options = null)
        : base(options ?? new DeepAROptions<T>())
    {
        _options = options ?? new DeepAROptions<T>();
        Options = _options;
        _random = RandomHelper.CreateSeededRandom(42);
        _lstmLayers = new List<DeepARLstmCellTape<T>>();
        _head = new DeepARGaussianHead<T>(_options.HiddenSize, seed: 12345);

        ValidateDeepAROptions();
        InitializeModel();
    }

    /// <summary>
    /// Validates DeepAR-specific options.
    /// </summary>
    private void ValidateDeepAROptions()
    {
        if (_options.LookbackWindow <= 0)
            throw new ArgumentException("Lookback window must be positive.");

        if (_options.ForecastHorizon <= 0)
            throw new ArgumentException("Forecast horizon must be positive.");

        if (_options.HiddenSize <= 0)
            throw new ArgumentException("Hidden size must be positive.");

        if (_options.NumLayers <= 0)
            throw new ArgumentException("Number of layers must be positive.");

        if (_options.NumSamples <= 0)
            throw new ArgumentException("Number of samples must be positive.");
    }

    /// <summary>
    /// Initializes the LSTM stack and Gaussian output head with tape-trainable parameters.
    /// </summary>
    private void InitializeModel()
    {
        _lstmLayers.Clear();

        // The recurrence consumes one series value per timestep (inputSize = 1). Covariates
        // are not fed into the recurrence in this univariate formulation; CovariateSize is
        // retained on the options for API compatibility but does not change the cell shape.
        for (int i = 0; i < _options.NumLayers; i++)
        {
            int layerInputSize = (i == 0) ? 1 : _options.HiddenSize;
            _lstmLayers.Add(new DeepARLstmCellTape<T>(layerInputSize, _options.HiddenSize, 42 + i * 1000));
        }

        _head = new DeepARGaussianHead<T>(_options.HiddenSize, seed: 12345);
    }

    private IReadOnlyList<Interfaces.ILayer<T>> AllLayers()
    {
        var layers = new List<Interfaces.ILayer<T>>(_lstmLayers.Count + 1);
        foreach (var lstm in _lstmLayers)
            layers.Add(lstm);
        layers.Add(_head);
        return layers;
    }

    /// <summary>
    /// Trains DeepAR with tape-based automatic differentiation (BPTT through the unrolled
    /// LSTM) and the Adam optimizer, on the Gaussian negative log-likelihood.
    /// </summary>
    /// <remarks>
    /// The label vector <paramref name="y"/> is interpreted as the univariate series. Each
    /// sample unrolls the LSTM over an L-step lookback window and is supervised one-step-ahead
    /// at every timestep (teacher forcing): the input at step t is y_{s+t} and the target is
    /// y_{s+t+1}. Autodiff yields gradients for every LSTM gate weight/bias and the Gaussian
    /// head; the previous implementation used hand-derived per-sample scalar gradients, which
    /// this rewrite removes entirely. The mean is emitted as a residual on the current
    /// observation (μ_t = x_t + Δ(h_t)); see <see cref="ForwardTape"/>.
    /// </remarks>
    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        // Store training series BEFORE the training loop for cancellation safety.
        _trainingSeries = new Vector<T>(y.Length);
        for (int i = 0; i < y.Length; i++)
            _trainingSeries[i] = y[i];
        ModelParameters = new Vector<T>(1);
        ModelParameters[0] = NumOps.FromDouble(y.Length);

        // Normalize the series to zero mean / unit variance for stable gradient flow.
        T yMean = NumOps.Zero;
        for (int i = 0; i < y.Length; i++)
            yMean = NumOps.Add(yMean, y[i]);
        yMean = NumOps.Divide(yMean, NumOps.FromDouble(y.Length));

        T yVar = NumOps.Zero;
        for (int i = 0; i < y.Length; i++)
        {
            T diff = NumOps.Subtract(y[i], yMean);
            yVar = NumOps.Add(yVar, NumOps.Multiply(diff, diff));
        }
        yVar = NumOps.Divide(yVar, NumOps.FromDouble(y.Length));
        T yStd = NumOps.Sqrt(yVar);
        if (NumOps.LessThanOrEquals(yStd, NumOps.FromDouble(1e-10)))
            yStd = NumOps.One;

        _normMean = yMean;
        _normStd = yStd;

        var yNorm = new Vector<T>(y.Length);
        for (int i = 0; i < y.Length; i++)
            yNorm[i] = NumOps.Divide(NumOps.Subtract(y[i], yMean), yStd);

        // Adam optimizer (Salinas et al. 2020 use Adam).
        var adamOptions = new AdamOptimizerOptions<T, Matrix<T>, Vector<T>>
        {
            InitialLearningRate = _options.LearningRate
        };
        var optimizer = new AdamOptimizer<T, Matrix<T>, Vector<T>>(null, adamOptions);

        // Collect every registered weight/bias tensor from the LSTM stack and the head.
        var trainableParams = Training.TapeTrainingStep<T>.CollectParameters(AllLayers(), -1);

        int lookback = _options.LookbackWindow;
        int numSamples = y.Length;

        bool timeBounded = _options.MaxTrainingTimeSeconds > 0;
        int maxEpochs = timeBounded ? int.MaxValue : _options.Epochs;

        // Best-checkpoint restore: snapshot parameters at the end of every epoch whose mean
        // training loss improves on the best seen, and restore the best snapshot after
        // training so late-epoch Adam divergence cannot degrade the returned model.
        double bestLoss = double.PositiveInfinity;
        List<Vector<T>>? bestSnapshot = null;
        _epochLosses.Clear();

        for (int epoch = 0; epoch < maxEpochs; epoch++)
        {
            if (timeBounded && TrainingCancellationToken.IsCancellationRequested)
                break;
            TrainingCancellationToken.ThrowIfCancellationRequested();

            var indices = Enumerable.Range(0, numSamples).OrderBy(_ => _random.Next()).ToList();

            double epochLossSum = 0.0;
            int epochSampleCount = 0;

            for (int batchStart = 0; batchStart < numSamples; batchStart += _options.BatchSize)
            {
                if (timeBounded && TrainingCancellationToken.IsCancellationRequested)
                    break;
                TrainingCancellationToken.ThrowIfCancellationRequested();

                int batchEnd = Math.Min(batchStart + _options.BatchSize, numSamples);
                int batchCount = batchEnd - batchStart;

                // Keep only anchors with a complete lookback AND a one-step-ahead target for
                // every timestep in the window: window inputs y[idx-L .. idx-1], targets
                // y[idx-L+1 .. idx]. That requires idx ∈ [L, N-1].
                var validIndices = new List<int>(batchCount);
                for (int bi = 0; bi < batchCount; bi++)
                {
                    int idx = indices[batchStart + bi];
                    if (idx < lookback || idx >= yNorm.Length)
                        continue;
                    validIndices.Add(idx);
                }

                if (validIndices.Count == 0)
                    continue;

                int b = validIndices.Count;

                // Per-timestep input tensors [1, B] and the [B, L] target window.
                var inputSteps = new Tensor<T>[lookback];
                for (int t = 0; t < lookback; t++)
                {
                    var xt = new Tensor<T>(new[] { 1, b });
                    for (int bi = 0; bi < b; bi++)
                    {
                        int idx = validIndices[bi];
                        xt[0, bi] = yNorm[idx - lookback + t];
                    }
                    inputSteps[t] = xt;
                }

                var targetData = new T[b * lookback];
                for (int bi = 0; bi < b; bi++)
                {
                    int idx = validIndices[bi];
                    for (int t = 0; t < lookback; t++)
                        targetData[bi * lookback + t] = yNorm[idx - lookback + t + 1];
                }
                var batchTarget = new Tensor<T>(new[] { b, lookback }, new Vector<T>(targetData));

                using var tape = new GradientTape<T>();

                var (meanBL, scaleBL) = ForwardTape(inputSteps, b);
                var batchLoss = ComputeTrainingLoss(meanBL, scaleBL, batchTarget);

                var allGrads = tape.ComputeGradients(batchLoss, sources: null);
                var grads = new Dictionary<Tensor<T>, Tensor<T>>(
                    Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
                foreach (var param in trainableParams)
                {
                    if (allGrads.TryGetValue(param, out var grad))
                        grads[param] = grad;
                }

                if (batchLoss.Length > 0)
                {
                    double bl = Convert.ToDouble(batchLoss[0]);
                    epochLossSum += bl * b;
                    epochSampleCount += b;
                }

                var context = new TapeStepContext<T>(
                    trainableParams, grads,
                    batchLoss.Length > 0 ? batchLoss[0] : NumOps.Zero);

                optimizer.Step(context);
            }

            if (epochSampleCount > 0)
            {
                double epochLoss = epochLossSum / epochSampleCount;
                _epochLosses.Add(epochLoss);
                if (!double.IsNaN(epochLoss) && !double.IsInfinity(epochLoss) && epochLoss < bestLoss)
                {
                    bestLoss = epochLoss;
                    bestSnapshot = SnapshotParameters();
                }
            }
        }

        if (bestSnapshot is not null)
            RestoreParameters(bestSnapshot);
    }

    /// <summary>
    /// Tape-tracked batched forward: unrolls the LSTM over the timesteps in
    /// <paramref name="inputSteps"/> (each <c>[1, B]</c>) and returns the per-step Gaussian
    /// parameters as <c>([B, L] mean, [B, L] scale)</c>. Uses <c>Engine.Tensor*</c> ops so
    /// <see cref="GradientTape{T}"/> differentiates the loss w.r.t. every registered weight.
    /// </summary>
    private (Tensor<T> mean, Tensor<T> scale) ForwardTape(Tensor<T>[] inputSteps, int batch)
    {
        int layers = _lstmLayers.Count;
        int h = _options.HiddenSize;

        // Zero initial hidden/cell state per layer, [H, B] column-major.
        var hState = new Tensor<T>[layers];
        var cState = new Tensor<T>[layers];
        for (int l = 0; l < layers; l++)
        {
            hState[l] = new Tensor<T>(new[] { h, batch });
            cState[l] = new Tensor<T>(new[] { h, batch });
        }

        var meanSteps = new Tensor<T>[inputSteps.Length];
        var scaleRawSteps = new Tensor<T>[inputSteps.Length];

        for (int t = 0; t < inputSteps.Length; t++)
        {
            Tensor<T> layerInput = inputSteps[t]; // [1, B]
            for (int l = 0; l < layers; l++)
            {
                var (hNew, cNew) = _lstmLayers[l].Step(layerInput, hState[l], cState[l]);
                hState[l] = hNew;
                cState[l] = cNew;
                layerInput = hNew; // feed to next layer
            }

            // Top-layer hidden -> Gaussian parameters for the next value. The head predicts a
            // RESIDUAL (delta) that is added to the current observation, so the mean is
            // μ_t = x_t + Δ(h_t). This residual/skip parameterization keeps the model at the
            // strong persistence prior (μ ≈ last value) at initialization and lets training
            // learn only the one-step change — the plain "predict the level from hidden" head
            // otherwise collapses to a smoothed mean on series with a fast seasonal component.
            var (deltaT, scaleRawT) = _head.Project(hState[layers - 1]); // each [1, B]
            meanSteps[t] = Engine.TensorAdd(inputSteps[t], deltaT);
            scaleRawSteps[t] = scaleRawT;
        }

        // Concatenate the per-step [1, B] slices along axis 0 -> [L, B], then permute to [B, L].
        var meanLB = Engine.TensorConcatenate(meanSteps, axis: 0);
        var scaleRawLB = Engine.TensorConcatenate(scaleRawSteps, axis: 0);
        var meanBL = Engine.TensorPermute(meanLB, new[] { 1, 0 });
        var scaleRawBL = Engine.TensorPermute(scaleRawLB, new[] { 1, 0 });

        // Softplus for a strictly-positive scale.
        var scaleBL = Engine.Softplus(scaleRawBL);
        return (meanBL, scaleBL);
    }

    /// <summary>
    /// Training loss for the Gaussian head, built entirely from tape-tracked engine ops:
    /// <c>MSE(μ, y) + mean( log σ + (y − μ_detached)² / (2σ²) )</c>.
    /// </summary>
    /// <remarks>
    /// A pure Gaussian negative log-likelihood couples μ and σ through <c>∂NLL/∂μ = (μ−y)/σ²</c>:
    /// when the mean is hard to fit early in training the optimizer inflates σ, which vanishes
    /// that gradient and collapses μ to the series mean (verified empirically — μ froze at the
    /// global mean and never tracked). Splitting the objective fixes this: the mean is pulled to
    /// the target by an undivided MSE gradient (no σ escape hatch), while σ is still trained to
    /// the residual scale via the NLL term with the mean <see cref="IEngine.StopGradient"/>-ed
    /// so the σ branch cannot perturb μ. The Gaussian likelihood head therefore stays fully
    /// trained and usable for predictive intervals. This mirrors the MSE-for-stability choice in
    /// NBEATSModel and is the "train the mean on MSE" simplification permitted for this model.
    /// </remarks>
    private Tensor<T> ComputeTrainingLoss(Tensor<T> mean, Tensor<T> scale, Tensor<T> target)
    {
        var allAxes = Enumerable.Range(0, mean.Shape.Length).ToArray();

        // Mean branch: plain MSE — gradient (μ − y) is independent of σ, so μ always tracks.
        var diffMean = Engine.TensorSubtract(mean, target);
        var meanMse = Engine.ReduceMean(Engine.TensorMultiply(diffMean, diffMean), allAxes, keepDims: false);

        // Scale branch: Gaussian NLL over the detached mean so σ learns the residual spread
        // without feeding gradient back into μ.
        var eps = NumOps.FromDouble(1e-6);
        var muDetached = Engine.StopGradient(mean);
        var diff = Engine.TensorSubtract(target, muDetached);
        var diff2 = Engine.TensorMultiply(diff, diff);
        var variance = Engine.TensorMultiply(scale, scale);
        var twoVar = Engine.TensorAddScalar(Engine.TensorMultiplyScalar(variance, NumOps.FromDouble(2.0)), eps);
        var quad = Engine.TensorDivide(diff2, twoVar);
        var logSigma = Engine.TensorLog(Engine.TensorAddScalar(scale, eps));
        var scaleTerm = Engine.ReduceMean(Engine.TensorAdd(quad, logSigma), allAxes, keepDims: false);

        return Engine.TensorAdd(meanMse, scaleTerm);
    }

    private List<Vector<T>> SnapshotParameters()
    {
        var snap = new List<Vector<T>>(_lstmLayers.Count + 1);
        foreach (var lstm in _lstmLayers)
            snap.Add(lstm.GetParameters());
        snap.Add(_head.GetParameters());
        return snap;
    }

    private void RestoreParameters(List<Vector<T>> snapshot)
    {
        for (int i = 0; i < _lstmLayers.Count; i++)
            _lstmLayers[i].SetParameters(snapshot[i]);
        _head.SetParameters(snapshot[_lstmLayers.Count]);
    }

    public override Vector<T> Predict(Matrix<T> input)
    {
        if (TryPredictFromTimeIndexCalibration(input, _trainingSeries, out var calibratedPredictions))
        {
            return calibratedPredictions;
        }

        int n = input.Rows;
        var predictions = new Vector<T>(n);

        // Each input row is an independent lookback window — forecast it from its own content.
        for (int i = 0; i < n; i++)
        {
            predictions[i] = PredictSingle(input.GetRow(i));
        }

        return predictions;
    }

    public override T PredictSingle(Vector<T> input)
    {
        var (mean, _) = PredictDistribution(input);
        return mean;
    }

    /// <summary>
    /// Predicts the Gaussian parameters (mean and scale, in the ORIGINAL data scale) for the
    /// value that follows the provided lookback window. Runs the same <c>Engine.Tensor*</c>
    /// LSTM recurrence used in training (B = 1, sequential unroll) so there is no
    /// scalar/tape divergence between inference and training.
    /// </summary>
    private (T mean, T scale) PredictDistribution(Vector<T> input)
    {
        // If the window is shorter than the lookback, pad from the training tail.
        if (input.Length < _options.LookbackWindow && _trainingSeries.Length >= _options.LookbackWindow)
        {
            var lb = new Vector<T>(_options.LookbackWindow);
            int start = _trainingSeries.Length - _options.LookbackWindow;
            for (int j = 0; j < _options.LookbackWindow; j++)
                lb[j] = _trainingSeries[start + j];
            input = lb;
        }

        int layers = _lstmLayers.Count;
        int h = _options.HiddenSize;

        var hState = new Tensor<T>[layers];
        var cState = new Tensor<T>[layers];
        for (int l = 0; l < layers; l++)
        {
            hState[l] = new Tensor<T>(new[] { h, 1 });
            cState[l] = new Tensor<T>(new[] { h, 1 });
        }

        // Unroll over the normalized window (these Engine ops run eagerly outside a tape).
        T lastNorm = NumOps.Zero;
        for (int t = 0; t < input.Length; t++)
        {
            T normValue = NumOps.Divide(NumOps.Subtract(input[t], _normMean), _normStd);
            lastNorm = normValue;
            var xt = new Tensor<T>(new[] { 1, 1 });
            xt[0, 0] = normValue;

            Tensor<T> layerInput = xt;
            for (int l = 0; l < layers; l++)
            {
                var (hNew, cNew) = _lstmLayers[l].Step(layerInput, hState[l], cState[l]);
                hState[l] = hNew;
                cState[l] = cNew;
                layerInput = hNew;
            }
        }

        var (deltaT, scaleRawT) = _head.Project(hState[layers - 1]); // each [1, 1]
        // μ = last observed value + learned residual (matches the training parameterization).
        T meanNorm = NumOps.Add(lastNorm, deltaT[0, 0]);
        T scaleNorm = Softplus(scaleRawT[0, 0]);

        // Denormalize into the original data scale.
        T mean = NumOps.Add(NumOps.Multiply(meanNorm, _normStd), _normMean);
        T scale = NumOps.Multiply(scaleNorm, _normStd);

        T minScale = NumOps.FromDouble(1e-6);
        if (NumOps.LessThan(scale, minScale))
            scale = minScale;

        return (mean, scale);
    }

    private T Softplus(T v)
    {
        T threshold = NumOps.FromDouble(20.0);
        if (NumOps.GreaterThan(v, threshold))
            return v;
        if (NumOps.LessThan(v, NumOps.FromDouble(-20.0)))
            return NumOps.Exp(v);
        return NumOps.Log(NumOps.Add(NumOps.One, NumOps.Exp(v)));
    }

    /// <summary>
    /// Generates probabilistic forecasts with quantile predictions by Monte-Carlo sampling
    /// the autoregressive predictive path.
    /// </summary>
    public Dictionary<double, Vector<T>> ForecastWithQuantiles(Vector<T> history, double[] quantiles)
    {
        var result = new Dictionary<double, Vector<T>>();
        var samples = new List<Vector<T>>();

        for (int s = 0; s < _options.NumSamples; s++)
        {
            var forecast = new Vector<T>(_options.ForecastHorizon);
            Vector<T> context = history.Clone();

            for (int hStep = 0; hStep < _options.ForecastHorizon; hStep++)
            {
                var (mean, scale) = PredictDistribution(context);

                T sample = NumOps.Add(mean, NumOps.Multiply(scale, NumOps.FromDouble(_random.NextGaussian())));
                forecast[hStep] = sample;

                var newContext = new Vector<T>(context.Length);
                for (int i = 0; i < context.Length - 1; i++)
                    newContext[i] = context[i + 1];
                newContext[context.Length - 1] = sample;
                context = newContext;
            }

            samples.Add(forecast);
        }

        foreach (var q in quantiles)
        {
            var quantileForecast = new Vector<T>(_options.ForecastHorizon);

            for (int hStep = 0; hStep < _options.ForecastHorizon; hStep++)
            {
                var values = new List<double>();
                foreach (var sample in samples)
                    values.Add(Convert.ToDouble(sample[hStep]));
                values.Sort();

                int idx = (int)(q * values.Count);
                idx = Math.Max(0, Math.Min(idx, values.Count - 1));
                quantileForecast[hStep] = NumOps.FromDouble(values[idx]);
            }

            result[q] = quantileForecast;
        }

        return result;
    }

    protected override void SerializeCore(BinaryWriter writer)
    {
        writer.Write(_options.HiddenSize);
        writer.Write(_options.NumLayers);

        writer.Write(_lstmLayers.Count);
        foreach (var lstm in _lstmLayers)
            lstm.Serialize(writer);

        _head.Serialize(writer);

        writer.Write(Convert.ToDouble(_normMean));
        writer.Write(Convert.ToDouble(_normStd));

        writer.Write(_trainingSeries.Length);
        for (int i = 0; i < _trainingSeries.Length; i++)
            writer.Write(NumOps.ToDouble(_trainingSeries[i]));
    }

    protected override void DeserializeCore(BinaryReader reader)
    {
        _options.HiddenSize = reader.ReadInt32();
        _options.NumLayers = reader.ReadInt32();

        InitializeModel();

        int numLayers = reader.ReadInt32();
        for (int i = 0; i < numLayers && i < _lstmLayers.Count; i++)
            _lstmLayers[i].Deserialize(reader);

        _head.Deserialize(reader);

        _normMean = NumOps.FromDouble(reader.ReadDouble());
        _normStd = NumOps.FromDouble(reader.ReadDouble());

        try
        {
            int tsLen = reader.ReadInt32();
            _trainingSeries = new Vector<T>(tsLen);
            for (int i = 0; i < tsLen; i++)
                _trainingSeries[i] = NumOps.FromDouble(reader.ReadDouble());
        }
        catch (EndOfStreamException)
        {
            _trainingSeries = Vector<T>.Empty();
        }
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "DeepAR",
            Description = "Probabilistic forecasting with autoregressive recurrent networks (tape-trained BPTT, Gaussian likelihood)",
            Complexity = ParameterCount,
            FeatureCount = _options.LookbackWindow,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "HiddenSize", _options.HiddenSize },
                { "NumLayers", _options.NumLayers },
                { "LikelihoodType", _options.LikelihoodType },
                { "ForecastHorizon", _options.ForecastHorizon },
                { "ProductionReady", true }
            }
        };
    }

    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        return new DeepARModel<T>(new DeepAROptions<T>(_options));
    }

    public override IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        var clone = new DeepARModel<T>(new DeepAROptions<T>(_options));
        // Trained layers are read-only after training — safe to share by reference.
        clone._lstmLayers.Clear();
        clone._lstmLayers.AddRange(_lstmLayers);
        clone._head = _head;
        if (_trainingSeries.Length > 0)
            clone._trainingSeries = new Vector<T>(_trainingSeries);
        if (ModelParameters is not null && ModelParameters.Length > 0)
            clone.ModelParameters = new Vector<T>(ModelParameters);
        clone._normMean = _normMean;
        clone._normStd = _normStd;
        return clone;
    }

    public override IFullModel<T, Matrix<T>, Vector<T>> DeepCopy() => Clone();

    public override long ParameterCount
    {
        get
        {
            int count = 0;
            foreach (var lstm in _lstmLayers)
                count += (int)lstm.ParameterCount;
            count += (int)_head.ParameterCount;
            return count;
        }
    }

    /// <summary>
    /// Mean training loss recorded at the end of each epoch of the most recent
    /// <see cref="TrainCore"/> call (the objective being minimized by Adam). Exposed so callers
    /// can confirm the tape-driven optimizer is actually reducing the loss.
    /// </summary>
    internal IReadOnlyList<double> TrainingLossHistory => _epochLosses.AsReadOnly();
    private readonly List<double> _epochLosses = new();
}

/// <summary>
/// A single LSTM cell whose one-timestep transition is expressed entirely with
/// <c>Engine.Tensor*</c> operations, so a <see cref="GradientTape{T}"/> can differentiate
/// through it (BPTT) and the Adam optimizer updates the registered weights from tape
/// gradients. Activations are column-major <c>[hiddenSize, batch]</c>.
/// </summary>
internal class DeepARLstmCellTape<T> : NeuralNetworks.Layers.LayerBase<T>
{
    private readonly int _inputSize;
    private readonly int _hiddenSize;

    // Input-to-gate and hidden-to-gate weights, stacked over the 4 gates in the order
    // (input, forget, cell, output): _wx is [4H, inputSize], _wh is [4H, H], _bias is [4H].
    private readonly Tensor<T> _wx;
    private readonly Tensor<T> _wh;
    private readonly Tensor<T> _bias;

    public int InputSize => _inputSize;
    public int HiddenSize => _hiddenSize;

    public override long ParameterCount => _wx.Length + _wh.Length + _bias.Length;
    public override bool SupportsTraining => true;
    public override void ResetState() { }
    public override void UpdateParameters(T learningRate) { /* tape-based optimizer updates registered params */ }

    /// <summary>
    /// Single-step forward from a zero initial state (satisfies the <c>ILayer</c> contract).
    /// The model drives the recurrence via <see cref="Step"/>; this convenience overload runs
    /// one timestep on <paramref name="input"/> <c>[inputSize, B]</c> and returns the hidden
    /// state <c>[H, B]</c>.
    /// </summary>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        int batch = input.Shape.Length > 1 ? input.Shape[1] : 1;
        var h0 = new Tensor<T>(new[] { _hiddenSize, batch });
        var c0 = new Tensor<T>(new[] { _hiddenSize, batch });
        var (h, _) = Step(input, h0, c0);
        return h;
    }

    public DeepARLstmCellTape(int inputSize, int hiddenSize, int seed = 42)
        : base(new[] { inputSize }, new[] { hiddenSize })
    {
        _inputSize = inputSize;
        _hiddenSize = hiddenSize;

        var random = RandomHelper.CreateSeededRandom(seed);
        double stddevX = Math.Sqrt(2.0 / (inputSize + hiddenSize));
        double stddevH = Math.Sqrt(2.0 / (hiddenSize + hiddenSize));

        _wx = CreateRandomTensor(new[] { 4 * hiddenSize, inputSize }, stddevX, random);
        _wh = CreateRandomTensor(new[] { 4 * hiddenSize, hiddenSize }, stddevH, random);
        _bias = new Tensor<T>(new[] { 4 * hiddenSize });

        // Forget-gate bias initialized to 1 (a standard LSTM trick that helps gradients flow
        // early in training). Gate order is (i, f, g, o), so forget occupies [H, 2H).
        for (int i = _hiddenSize; i < 2 * _hiddenSize; i++)
            _bias[i] = NumOps.One;

        RegisterTrainableParameter(_wx, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_wh, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_bias, PersistentTensorRole.Biases);
    }

    private Tensor<T> CreateRandomTensor(int[] shape, double stddev, Random random)
    {
        var tensor = new Tensor<T>(shape);
        int total = tensor.Length;
        for (int i = 0; i < total; i++)
            tensor[i] = NumOps.FromDouble((random.NextDouble() * 2 - 1) * stddev);
        return tensor;
    }

    /// <summary>
    /// Computes one LSTM step. Inputs are column-major: <paramref name="xt"/> is
    /// <c>[inputSize, B]</c>, <paramref name="hPrev"/> and <paramref name="cPrev"/> are
    /// <c>[H, B]</c>. Returns the new hidden and cell states, both <c>[H, B]</c>. The same
    /// method serves training (under a tape) and inference (eager) — identical ops, so there
    /// is no scalar/tape divergence.
    /// </summary>
    public (Tensor<T> h, Tensor<T> c) Step(Tensor<T> xt, Tensor<T> hPrev, Tensor<T> cPrev)
    {
        // Pre-activations for all 4 gates: [4H, B].
        var fromX = Engine.TensorMatMul(_wx, xt);      // [4H, B]
        var fromH = Engine.TensorMatMul(_wh, hPrev);   // [4H, B]
        var preact = Engine.TensorAdd(fromX, fromH);
        var biasCol = Engine.Reshape(_bias, new[] { 4 * _hiddenSize, 1 });
        preact = Engine.TensorBroadcastAdd(preact, biasCol);

        // Slice the stacked pre-activations into the four gates along axis 0.
        var iPre = Engine.TensorNarrow(preact, 0, 0, _hiddenSize);
        var fPre = Engine.TensorNarrow(preact, 0, _hiddenSize, _hiddenSize);
        var gPre = Engine.TensorNarrow(preact, 0, 2 * _hiddenSize, _hiddenSize);
        var oPre = Engine.TensorNarrow(preact, 0, 3 * _hiddenSize, _hiddenSize);

        var iGate = Engine.Sigmoid(iPre);
        var fGate = Engine.Sigmoid(fPre);
        var gGate = Engine.Tanh(gPre);
        var oGate = Engine.Sigmoid(oPre);

        // c = f ⊙ c_prev + i ⊙ g ; h = o ⊙ tanh(c).
        var cNew = Engine.TensorAdd(
            Engine.TensorMultiply(fGate, cPrev),
            Engine.TensorMultiply(iGate, gGate));
        var hNew = Engine.TensorMultiply(oGate, Engine.Tanh(cNew));

        return (hNew, cNew);
    }

    public override Vector<T> GetParameters()
    {
        var p = new T[_wx.Length + _wh.Length + _bias.Length];
        int idx = 0;
        for (int i = 0; i < _wx.Length; i++) p[idx++] = _wx[i];
        for (int i = 0; i < _wh.Length; i++) p[idx++] = _wh[i];
        for (int i = 0; i < _bias.Length; i++) p[idx++] = _bias[i];
        return new Vector<T>(p);
    }

    public override void SetParameters(Vector<T> parameters)
    {
        int idx = 0;
        for (int i = 0; i < _wx.Length; i++) _wx[i] = parameters[idx++];
        for (int i = 0; i < _wh.Length; i++) _wh[i] = parameters[idx++];
        for (int i = 0; i < _bias.Length; i++) _bias[i] = parameters[idx++];
    }

    public override void Serialize(BinaryWriter writer)
    {
        writer.Write(_inputSize);
        writer.Write(_hiddenSize);
        WriteTensor(writer, _wx);
        WriteTensor(writer, _wh);
        WriteTensor(writer, _bias);
    }

    public override void Deserialize(BinaryReader reader)
    {
        reader.ReadInt32(); // inputSize
        reader.ReadInt32(); // hiddenSize
        ReadTensorInto(reader, _wx);
        ReadTensorInto(reader, _wh);
        ReadTensorInto(reader, _bias);
    }

    private static void WriteTensor(BinaryWriter writer, Tensor<T> tensor)
    {
        writer.Write(tensor.Shape.Length);
        foreach (var dim in tensor._shape)
            writer.Write(dim);
        for (int i = 0; i < tensor.Length; i++)
            writer.Write(Convert.ToDouble(tensor[i]));
    }

    private void ReadTensorInto(BinaryReader reader, Tensor<T> tensor)
    {
        int rank = reader.ReadInt32();
        var shape = new int[rank];
        for (int d = 0; d < rank; d++)
            shape[d] = reader.ReadInt32();
        int total = shape.Aggregate(1, (a, bb) => a * bb);
        for (int i = 0; i < total; i++)
        {
            double v = reader.ReadDouble();
            if (i < tensor.Length)
                tensor[i] = NumOps.FromDouble(v);
        }
    }
}

/// <summary>
/// Gaussian output head for DeepAR: projects the top LSTM hidden state to a mean and a raw
/// scale (softplus is applied by the caller). Expressed with <c>Engine.Tensor*</c> ops so the
/// tape differentiates the likelihood w.r.t. the projection weights. Activations are
/// column-major <c>[*, batch]</c>.
/// </summary>
internal class DeepARGaussianHead<T> : NeuralNetworks.Layers.LayerBase<T>
{
    private readonly int _hiddenSize;

    private readonly Tensor<T> _meanW;   // [1, H]
    private readonly Tensor<T> _meanB;   // [1]
    private readonly Tensor<T> _scaleW;  // [1, H]
    private readonly Tensor<T> _scaleB;  // [1]

    public override long ParameterCount => _meanW.Length + _meanB.Length + _scaleW.Length + _scaleB.Length;
    public override bool SupportsTraining => true;
    public override void ResetState() { }
    public override void UpdateParameters(T learningRate) { /* tape-based optimizer updates registered params */ }

    /// <summary>
    /// Projects the hidden state <paramref name="input"/> <c>[H, B]</c> to the Gaussian mean
    /// <c>[1, B]</c> (satisfies the <c>ILayer</c> contract; the model uses <see cref="Project"/>).
    /// </summary>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        var (mean, _) = Project(input);
        return mean;
    }

    public DeepARGaussianHead(int hiddenSize, int seed = 12345)
        : base(new[] { hiddenSize }, new[] { 1 })
    {
        _hiddenSize = hiddenSize;

        var random = RandomHelper.CreateSeededRandom(seed);
        double stddev = Math.Sqrt(2.0 / hiddenSize);

        _meanW = CreateRandomTensor(new[] { 1, hiddenSize }, stddev, random);
        _meanB = new Tensor<T>(new[] { 1 });
        _scaleW = CreateRandomTensor(new[] { 1, hiddenSize }, stddev, random);
        _scaleB = new Tensor<T>(new[] { 1 });

        RegisterTrainableParameter(_meanW, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_meanB, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_scaleW, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_scaleB, PersistentTensorRole.Biases);
    }

    private Tensor<T> CreateRandomTensor(int[] shape, double stddev, Random random)
    {
        var tensor = new Tensor<T>(shape);
        int total = tensor.Length;
        for (int i = 0; i < total; i++)
            tensor[i] = NumOps.FromDouble((random.NextDouble() * 2 - 1) * stddev);
        return tensor;
    }

    /// <summary>
    /// Projects the top hidden state <paramref name="h"/> (<c>[H, B]</c>) to the Gaussian mean
    /// and raw (pre-softplus) scale, each <c>[1, B]</c>. Same ops for tape and eager use.
    /// </summary>
    public (Tensor<T> mean, Tensor<T> scaleRaw) Project(Tensor<T> h)
    {
        var mean = Engine.TensorMatMul(_meanW, h);   // [1, B]
        var meanBCol = Engine.Reshape(_meanB, new[] { 1, 1 });
        mean = Engine.TensorBroadcastAdd(mean, meanBCol);

        var scaleRaw = Engine.TensorMatMul(_scaleW, h); // [1, B]
        var scaleBCol = Engine.Reshape(_scaleB, new[] { 1, 1 });
        scaleRaw = Engine.TensorBroadcastAdd(scaleRaw, scaleBCol);

        return (mean, scaleRaw);
    }

    public override Vector<T> GetParameters()
    {
        var p = new T[_meanW.Length + _meanB.Length + _scaleW.Length + _scaleB.Length];
        int idx = 0;
        for (int i = 0; i < _meanW.Length; i++) p[idx++] = _meanW[i];
        for (int i = 0; i < _meanB.Length; i++) p[idx++] = _meanB[i];
        for (int i = 0; i < _scaleW.Length; i++) p[idx++] = _scaleW[i];
        for (int i = 0; i < _scaleB.Length; i++) p[idx++] = _scaleB[i];
        return new Vector<T>(p);
    }

    public override void SetParameters(Vector<T> parameters)
    {
        int idx = 0;
        for (int i = 0; i < _meanW.Length; i++) _meanW[i] = parameters[idx++];
        for (int i = 0; i < _meanB.Length; i++) _meanB[i] = parameters[idx++];
        for (int i = 0; i < _scaleW.Length; i++) _scaleW[i] = parameters[idx++];
        for (int i = 0; i < _scaleB.Length; i++) _scaleB[i] = parameters[idx++];
    }

    public override void Serialize(BinaryWriter writer)
    {
        writer.Write(_hiddenSize);
        WriteTensor(writer, _meanW);
        WriteTensor(writer, _meanB);
        WriteTensor(writer, _scaleW);
        WriteTensor(writer, _scaleB);
    }

    public override void Deserialize(BinaryReader reader)
    {
        reader.ReadInt32(); // hiddenSize
        ReadTensorInto(reader, _meanW);
        ReadTensorInto(reader, _meanB);
        ReadTensorInto(reader, _scaleW);
        ReadTensorInto(reader, _scaleB);
    }

    private static void WriteTensor(BinaryWriter writer, Tensor<T> tensor)
    {
        writer.Write(tensor.Shape.Length);
        foreach (var dim in tensor._shape)
            writer.Write(dim);
        for (int i = 0; i < tensor.Length; i++)
            writer.Write(Convert.ToDouble(tensor[i]));
    }

    private void ReadTensorInto(BinaryReader reader, Tensor<T> tensor)
    {
        int rank = reader.ReadInt32();
        var shape = new int[rank];
        for (int d = 0; d < rank; d++)
            shape[d] = reader.ReadInt32();
        int total = shape.Aggregate(1, (a, bb) => a * bb);
        for (int i = 0; i < total; i++)
        {
            double v = reader.ReadDouble();
            if (i < tensor.Length)
                tensor[i] = NumOps.FromDouble(v);
        }
    }
}
