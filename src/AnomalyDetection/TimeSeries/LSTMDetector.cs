using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.TimeSeries;

/// <summary>
/// Implements LSTM-based anomaly detection using prediction error.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> LSTM (Long Short-Term Memory) learns patterns in sequential data
/// and predicts the next value. Anomalies are detected when the actual value differs
/// significantly from the predicted value.
/// </para>
/// <para>
/// The algorithm follows the prediction-based formulation of Malhotra et al. (2015) and
/// Hundman et al. (2018, NASA telemanom):
/// 1. Train an LSTM to FORECAST the next value of the series.
/// 2. For each point, compute the forecast error.
/// 3. High forecast errors indicate anomalies.
/// </para>
/// <para>
/// Because the core of the detector IS a forecasting LSTM, the model also exposes a normal
/// forecasting surface: <see cref="Train(Matrix{T}, Vector{T})"/> learns a target series and
/// <see cref="Predict"/> returns one-step-ahead forecasts. The recurrent core runs on the
/// shared tensor Engine (tape-based BPTT) rather than a hand-rolled scalar loop.
/// </para>
/// <para>
/// Reference: Malhotra et al. (2015), "Long Short Term Memory Networks for Anomaly Detection
/// in Time Series"; Hundman et al. (2018), "Detecting Spacecraft Anomalies Using LSTMs and
/// Nonparametric Dynamic Thresholding"; Hochreiter &amp; Schmidhuber (1997), "Long Short-Term Memory".
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.RecurrentNetwork)]
[ModelCategory(ModelCategory.TimeSeriesModel)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Long Short-Term Memory", "https://doi.org/10.1162/neco.1997.9.8.1735", Year = 1997, Authors = "Sepp Hochreiter, Jürgen Schmidhuber")]
public partial class LSTMDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _hiddenDim;
    private readonly int _seqLength;
    private readonly int _epochs;
    private readonly double _learningRate;

    // Engine-backed forecasting core: LSTM (returns the full hidden sequence) -> Dense
    // projection back to the feature dimension. Trained via the standard tape-based
    // NeuralNetworkBase.Train path (CpuEngine.LstmSequenceForward + autograd), not a
    // hand-rolled scalar BPTT loop.
    private NeuralNetwork<T>? _forecaster;
    private int _inputDim;

    // The normalized training series and the normalization parameters, kept so Predict can
    // produce in-sample one-step-ahead forecasts and score new data against the learned model.
    [Buffer]
    private Matrix<T>? _normalizedSeries;
    [Buffer]
    private Vector<T>? _dataMeans;
    [Buffer]
    private Vector<T>? _dataStds;

    /// <summary>Gets the hidden dimension of the LSTM.</summary>
    public int HiddenDim => _hiddenDim;

    /// <summary>Gets the sequence/window length.</summary>
    public int SeqLength => _seqLength;

    /// <summary>
    /// Creates a new LSTM anomaly detector.
    /// </summary>
    /// <param name="hiddenDim">Dimension of the LSTM hidden state. Default is 64.</param>
    /// <param name="seqLength">Length of the input window. Default is 10.</param>
    /// <param name="epochs">Number of training epochs. Default is 50.</param>
    /// <param name="learningRate">Learning rate. Default is 0.001.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public LSTMDetector(int hiddenDim = 64, int seqLength = 10, int epochs = 50,
        double learningRate = 0.001, double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (hiddenDim < 1)
            throw new ArgumentOutOfRangeException(nameof(hiddenDim),
                "Hidden dimensions must be at least 1. Recommended is 64.");
        if (seqLength < 1)
            throw new ArgumentOutOfRangeException(nameof(seqLength),
                "Sequence length must be at least 1. Recommended is 10.");
        if (epochs < 1)
            throw new ArgumentOutOfRangeException(nameof(epochs),
                "Epochs must be at least 1. Recommended is 50.");
        if (learningRate <= 0)
            throw new ArgumentOutOfRangeException(nameof(learningRate),
                "Learning rate must be positive. Recommended is 0.001.");

        _hiddenDim = hiddenDim;
        _seqLength = seqLength;
        _epochs = epochs;
        _learningRate = learningRate;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        int n = X.Rows;
        _inputDim = X.Columns;

        if (n < _seqLength + 1)
            throw new ArgumentException(
                $"Not enough samples for sequence length {_seqLength}. Need at least {_seqLength + 1} samples.",
                nameof(X));
        if (_inputDim < 1)
            throw new ArgumentException("Input must have at least 1 feature.", nameof(X));

        var (normalizedData, means, stds) = NormalizeData(X);
        _dataMeans = means;
        _dataStds = stds;
        _normalizedSeries = normalizedData;

        TrainForecaster(normalizedData);

        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    /// <summary>
    /// Returns one-step-ahead forecasts of the learned series in the ORIGINAL (denormalized)
    /// scale — the model's underlying prediction surface (Malhotra et al. 2015; Hundman et al.
    /// 2018). <see cref="Predict"/> follows the <c>IAnomalyDetector</c> contract and returns
    /// anomaly LABELS (−1/+1); use this method when you want the raw forecast the anomaly score
    /// is derived from.
    /// </summary>
    /// <param name="steps">Number of leading series positions to forecast.</param>
    public Vector<T> Forecast(int steps)
    {
        EnsureFitted();
        if (steps < 1) throw new ArgumentOutOfRangeException(nameof(steps), "steps must be at least 1.");
        return ForecastInSample(steps);
    }

    /// <inheritdoc/>
    public override Vector<T> ScoreAnomalies(Matrix<T> X)
    {
        EnsureFitted();
        return ScoreAnomaliesInternal(X);
    }

    // ---- Engine-backed forecasting core -------------------------------------------------

    private void TrainForecaster(Matrix<T> normalizedSeries)
    {
        int n = normalizedSeries.Rows;

        // LSTM (full-sequence) -> Dense projection back to the feature width. The Dense head
        // is linear (Identity): forecasting is a regression onto the continuous next value.
        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: _inputDim,
            outputSize: _inputDim,
            layers: new List<ILayer<T>>
            {
                new LSTMLayer<T>(_hiddenDim),
                new DenseLayer<T>(_inputDim, new IdentityActivation<T>() as IActivationFunction<T>)
            });

        _forecaster = new NeuralNetwork<T>(architecture);

        // Sliding-window teacher forcing, batched. Each window is an independent length-
        // _seqLength sequence (the LSTM resets its state per batch element), with the same
        // window shifted one step ahead as the target, so the last position of each window
        // learns to forecast the value that follows it. Stacking every window into one
        // [numWindows, seqLength, feat] batch gives the full per-window gradient signal in a
        // single tape pass per epoch — many more effective updates than a single
        // full-sequence step, without thousands of Train() calls.
        int numWindows = n - _seqLength;
        var series = AnomalyTensorOps.ToTensor(normalizedSeries);
        var inputTensor = Windows(series, numWindows, 0);
        var targetTensor = Windows(series, numWindows, 1);

        for (int epoch = 0; epoch < _epochs; epoch++)
            _forecaster.Train(inputTensor, targetTensor);
    }

    /// <summary>
    /// Runs every length-_seqLength window of the stored series through the forecaster and
    /// returns the last-position forecast of each window: <c>windowForecast[w]</c> is the
    /// (normalized) prediction of series position <c>w + _seqLength</c>.
    /// </summary>
    private Tensor<T>? PredictWindows()
    {
        var series = _normalizedSeries;
        if (_forecaster is null || series is null) return null;
        int n = series.Rows;
        int numWindows = n - _seqLength;
        if (numWindows < 1) return null;

        var batch = Windows(AnomalyTensorOps.ToTensor(series), numWindows, 0);
        return _forecaster.Predict(batch);
    }

    /// <summary>
    /// One-step-ahead in-sample forecasts (denormalized), feature 0, padded/continued to
    /// <paramref name="steps"/> positions. Position 0 has no history so it returns the actual
    /// first value; position t (t&gt;=1) is the model's forecast of the learned series at t.
    /// </summary>
    // The windows series[w + offset .. w + offset + seqLength) for w < count, as [count, seqLength, features].
    private Tensor<T> Windows(Tensor<T> series, int count, int offset)
    {
        var engine = Engine;
        int length = _seqLength;
        var rows = new int[count * length];
        for (int w = 0; w < count; w++)
        {
            for (int t = 0; t < length; t++)
            {
                rows[w * length + t] = w + t + offset;
            }
        }

        int features = series.Shape[1];
        var gathered = engine.TensorGather(series, Tensor<int>.FromVector(Vector<int>.Wrap(rows), new[] { rows.Length }), 0);
        return engine.Reshape(gathered, new[] { count, length, features });
    }

    // The forecast at the last position of every window ([numWindows, features]): window w forecasts
    // position w + seqLength.
    private Tensor<T> LastStep(Tensor<T> windowOut)
    {
        var engine = Engine;
        int count = windowOut.Shape[0];
        int features = windowOut.Shape[2];
        return engine.Reshape(
            engine.TensorSlice(windowOut, new[] { 0, _seqLength - 1, 0 }, new[] { count, 1, features }),
            new[] { count, features });
    }

    private Vector<T> ForecastInSample(int steps)
    {
        var series = _normalizedSeries;
        var means = _dataMeans;
        var stds = _dataStds;
        if (_forecaster is null || series is null || means is null || stds is null)
            return new Vector<T>(steps);

        var engine = Engine;
        int n = series.Rows;
        int numWindows = n - _seqLength;
        var windowForecasts = PredictWindows();

        // Candidates: the actual series values (feature 0) followed by the window forecasts (feature 0)
        var seriesValues = engine.TensorSliceAxis(AnomalyTensorOps.ToTensor(series), 1, 0);
        var candidates = windowForecasts is null
            ? seriesValues
            : engine.TensorConcatenate(new[] { seriesValues, engine.TensorSliceAxis(LastStep(windowForecasts), 1, 0) }, 0);

        var pick = new int[steps];
        for (int p = 0; p < steps; p++)
        {
            if (p < _seqLength || windowForecasts is null)
            {
                // No full window of history yet — the actual value is the best available
                // estimate (a 0-error warm-up, the standard one-step-ahead convention).
                pick[p] = Math.Min(p, n - 1);
            }
            else if (p - _seqLength < numWindows)
            {
                // The forecast of window p - _seqLength is for series position p.
                pick[p] = n + p - _seqLength;
            }
            else
            {
                // Beyond the learned horizon: hold the final window's forecast.
                pick[p] = n + numWindows - 1;
            }
        }

        // Denormalize feature 0 back to the original scale.
        var normValues = engine.TensorGather(candidates, Tensor<int>.FromVector(Vector<int>.Wrap(pick), new[] { steps }), 0);
        return engine.TensorAddScalar(engine.TensorMultiplyScalar(normValues, stds[0]), means[0]).ToVector();
    }

    private Vector<T> ScoreAnomaliesInternal(Matrix<T> X)
    {
        ValidateInput(X);

        if (X.Columns != _inputDim)
            throw new ArgumentException(
                $"Input has {X.Columns} features but model was trained with {_inputDim} features.",
                nameof(X));

        var means = _dataMeans;
        var stds = _dataStds;
        if (_forecaster is null || means is null || stds is null)
            throw new InvalidOperationException("Model not properly fitted.");

        var engine = Engine;
        int n = X.Rows;

        // Normalize the incoming data with the learned statistics.
        var normalized = AnomalyTensorOps.DivideRow(
            AnomalyTensorOps.SubtractRow(AnomalyTensorOps.ToTensor(X), means.AsTensor()), stds.AsTensor());

        // Insufficient history (and no forecaster windows): score by distance from the (normalized) mean.
        var squaredNorm = engine.ReduceSum(engine.TensorSquare(normalized), new[] { 1 }, false);

        // Forecast each position i (>= _seqLength) from its preceding window and batch the
        // whole set of windows into one Predict() call. windowOut[w, last] predicts position
        // w + _seqLength; the score is the squared forecast error ||normalized[i] - forecast(i)||^2.
        int numWindows = n - _seqLength;
        if (numWindows < 1)
        {
            return squaredNorm.ToVector();
        }

        var windowOut = _forecaster.Predict(Windows(normalized, numWindows, 0));
        var actual = engine.TensorSlice(normalized, new[] { _seqLength, 0 }, new[] { numWindows, _inputDim });
        var errors = engine.ReduceSum(engine.TensorSquare(engine.TensorSubtract(actual, LastStep(windowOut))), new[] { 1 }, false);
        var warmUp = engine.TensorSlice(squaredNorm, new[] { 0 }, new[] { _seqLength });
        return engine.TensorConcatenate(new[] { warmUp, errors }, 0).ToVector();
    }

    private (Matrix<T> normalized, Vector<T> means, Vector<T> stds) NormalizeData(Matrix<T> data)
    {
        // Per-feature mean and population std (std below 1e-10 replaced by 1), then (x - mean) / std
        var engine = Engine;
        var values = AnomalyTensorOps.ToTensor(data);
        var (std, mean) = engine.TensorStdMean(values, new[] { 0 }, 0, false);
        var tiny = engine.TensorLessThan(std, NumOps.FromDouble(1e-10));
        var stds = engine.TensorWhere(tiny, Tensor<T>.CreateOnes(new[] { data.Columns }), std);
        var normalized = AnomalyTensorOps.DivideRow(AnomalyTensorOps.SubtractRow(values, mean), stds);
        return (normalized.ToMatrix(), mean.ToVector(), stds.ToVector());
    }
}
