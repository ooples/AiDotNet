using AiDotNet.Helpers;
using AiDotNet.Interfaces;
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
/// The algorithm works by:
/// 1. Train LSTM to predict next value in sequence
/// 2. For each point, compute prediction error
/// 3. High prediction errors indicate anomalies
/// </para>
/// <para>
/// <b>When to use:</b>
/// - Time series data with temporal dependencies
/// - When patterns have long-term dependencies
/// - Sequential anomaly detection
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Hidden dimensions: 64
/// - Sequence length: 10
/// - Epochs: 50
/// - Learning rate: 0.001
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Hochreiter, S. and Schmidhuber, J. (1997).
/// "Long Short-Term Memory." Neural Computation.
/// </para>
/// </remarks>
public class LSTMDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _hiddenDim;
    private readonly int _seqLength;
    private readonly int _epochs;
    private readonly double _learningRate;

    // LSTM weights (simplified single layer)
    // Gates: forget (f), input (i), cell (c), output (o)
    private Matrix<T>? _Wf; // Forget gate
    private Matrix<T>? _Wi; // Input gate
    private Matrix<T>? _Wc; // Cell gate
    private Matrix<T>? _Wo; // Output gate
    private Vector<T>? _bf;
    private Vector<T>? _bi;
    private Vector<T>? _bc;
    private Vector<T>? _bo;

    // Output layer
    private Matrix<T>? _Wy;
    private Vector<T>? _by;

    private int _inputDim;

    // Normalization parameters
    private Vector<T>? _dataMeans;
    private Vector<T>? _dataStds;

    /// <summary>
    /// Gets the hidden dimensions.
    /// </summary>
    public int HiddenDim => _hiddenDim;

    /// <summary>
    /// Gets the sequence length.
    /// </summary>
    public int SeqLength => _seqLength;

    /// <summary>
    /// Creates a new LSTM anomaly detector.
    /// </summary>
    /// <param name="hiddenDim">Dimensions of LSTM hidden state. Default is 64.</param>
    /// <param name="seqLength">Length of input sequences. Default is 10.</param>
    /// <param name="epochs">Number of training epochs. Default is 50.</param>
    /// <param name="learningRate">Learning rate. Default is 0.001.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public LSTMDetector(int hiddenDim = 64, int seqLength = 10, int epochs = 50,
        double learningRate = 0.001, double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (hiddenDim < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(hiddenDim),
                "Hidden dimensions must be at least 1. Recommended is 64.");
        }

        if (seqLength < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(seqLength),
                "Sequence length must be at least 1. Recommended is 10.");
        }

        if (epochs < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(epochs),
                "Epochs must be at least 1. Recommended is 50.");
        }

        if (learningRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(learningRate),
                "Learning rate must be positive. Recommended is 0.001.");
        }

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
        {
            throw new ArgumentException(
                $"Not enough samples for sequence length {_seqLength}. Need at least {_seqLength + 1} samples.",
                nameof(X));
        }

        if (_inputDim < 1)
        {
            throw new ArgumentException(
                "Input must have at least 1 feature.",
                nameof(X));
        }

        // Normalize data
        var (normalizedData, means, stds) = NormalizeData(X);
        _dataMeans = means;
        _dataStds = stds;

        // Initialize weights
        InitializeWeights();

        // Create sequences
        var (sequences, targets) = CreateSequences(normalizedData);

        // Train
        Train(sequences, targets);

        // Calculate scores for training data
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    private (Matrix<T> normalized, Vector<T> means, Vector<T> stds) NormalizeData(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;

        var means = new Vector<T>(d);
        var stds = new Vector<T>(d);

        for (int j = 0; j < d; j++)
        {
            T sum = NumOps.Zero;
            for (int i = 0; i < n; i++)
            {
                sum = NumOps.Add(sum, data[i, j]);
            }
            means[j] = NumOps.Divide(sum, NumOps.FromDouble(n));

            T variance = NumOps.Zero;
            for (int i = 0; i < n; i++)
            {
                T diff = NumOps.Subtract(data[i, j], means[j]);
                variance = NumOps.Add(variance, NumOps.Multiply(diff, diff));
            }
            double stdVal = Math.Sqrt(NumOps.ToDouble(variance) / n);
            if (stdVal < 1e-10) stdVal = 1;
            stds[j] = NumOps.FromDouble(stdVal);
        }

        var normalized = new Matrix<T>(n, d);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < d; j++)
            {
                T diff = NumOps.Subtract(data[i, j], means[j]);
                normalized[i, j] = NumOps.Divide(diff, stds[j]);
            }
        }

        return (normalized, means, stds);
    }

    private void InitializeWeights()
    {
        int inputSize = _inputDim + _hiddenDim; // Input + previous hidden state
        double scale = Math.Sqrt(2.0 / inputSize);
        double scaleOut = Math.Sqrt(2.0 / _hiddenDim);

        _Wf = InitializeMatrix(inputSize, _hiddenDim, scale);
        _Wi = InitializeMatrix(inputSize, _hiddenDim, scale);
        _Wc = InitializeMatrix(inputSize, _hiddenDim, scale);
        _Wo = InitializeMatrix(inputSize, _hiddenDim, scale);

        _bf = new Vector<T>(_hiddenDim);
        _bi = new Vector<T>(_hiddenDim);
        _bc = new Vector<T>(_hiddenDim);
        _bo = new Vector<T>(_hiddenDim);

        // Initialize forget gate bias to 1 (helps with gradient flow)
        for (int i = 0; i < _hiddenDim; i++)
        {
            _bf[i] = NumOps.One;
            _bi[i] = NumOps.Zero;
            _bc[i] = NumOps.Zero;
            _bo[i] = NumOps.Zero;
        }

        _Wy = InitializeMatrix(_hiddenDim, _inputDim, scaleOut);
        _by = new Vector<T>(_inputDim);
        for (int i = 0; i < _inputDim; i++)
        {
            _by[i] = NumOps.Zero;
        }
    }

    private Matrix<T> InitializeMatrix(int rows, int cols, double scale)
    {
        var matrix = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                double u1 = 1.0 - _random.NextDouble();
                double u2 = 1.0 - _random.NextDouble();
                double val = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2) * scale;
                matrix[i, j] = NumOps.FromDouble(val);
            }
        }
        return matrix;
    }

    private (Vector<T>[][] sequences, Vector<T>[] targets) CreateSequences(Matrix<T> data)
    {
        int n = data.Rows - _seqLength;
        var sequences = new Vector<T>[n][];
        var targets = new Vector<T>[n];

        for (int i = 0; i < n; i++)
        {
            sequences[i] = new Vector<T>[_seqLength];
            for (int t = 0; t < _seqLength; t++)
            {
                sequences[i][t] = data.GetRow(i + t);
            }
            targets[i] = data.GetRow(i + _seqLength);
        }

        return (sequences, targets);
    }

    private void Train(Vector<T>[][] sequences, Vector<T>[] targets)
    {
        // Capture nullable fields for proper null checking
        var Wf = _Wf;
        var Wi = _Wi;
        var Wc = _Wc;
        var Wo = _Wo;
        var bf = _bf;
        var bi = _bi;
        var bc = _bc;
        var bo = _bo;
        var Wy = _Wy;
        var by = _by;

        if (Wf == null || Wi == null || Wc == null || Wo == null ||
            bf == null || bi == null || bc == null || bo == null ||
            Wy == null || by == null)
        {
            throw new InvalidOperationException("Weights not initialized.");
        }

        int n = sequences.Length;
        int batchSize = Math.Min(32, n);
        int inputSize = _inputDim + _hiddenDim;

        for (int epoch = 0; epoch < _epochs; epoch++)
        {
            var indices = Enumerable.Range(0, n).OrderBy(_ => _random.NextDouble()).ToArray();

            for (int batch = 0; batch < n; batch += batchSize)
            {
                int actualBatchSize = Math.Min(batchSize, n - batch);

                // Initialize gradient accumulators (using double for intermediate computation during backprop)
                var dWf = new double[inputSize, _hiddenDim];
                var dWi = new double[inputSize, _hiddenDim];
                var dWc = new double[inputSize, _hiddenDim];
                var dWo = new double[inputSize, _hiddenDim];
                var dbf = new double[_hiddenDim];
                var dbi = new double[_hiddenDim];
                var dbc = new double[_hiddenDim];
                var dbo = new double[_hiddenDim];
                var dWy = new double[_hiddenDim, _inputDim];
                var dby = new double[_inputDim];

                for (int b = 0; b < actualBatchSize; b++)
                {
                    int idx = indices[batch + b];
                    var seq = sequences[idx];
                    var target = targets[idx];

                    // Forward pass with caching all intermediate values
                    var (prediction, hStates, cStates, fGates, iGates, cCandidates, oGates, concats) =
                        ForwardWithCache(seq);

                    // Compute output layer gradient
                    var dOutput = new Vector<T>(_inputDim);
                    for (int j = 0; j < _inputDim; j++)
                    {
                        T diff = NumOps.Subtract(prediction[j], target[j]);
                        dOutput[j] = NumOps.Multiply(NumOps.FromDouble(2.0), diff);
                    }

                    // Backprop through output layer
                    var hFinal = hStates[seq.Length];
                    for (int j = 0; j < _inputDim; j++)
                    {
                        double dOutJ = NumOps.ToDouble(dOutput[j]);
                        dby[j] += dOutJ;
                        for (int i = 0; i < _hiddenDim; i++)
                        {
                            dWy[i, j] += NumOps.ToDouble(hFinal[i]) * dOutJ;
                        }
                    }

                    // Gradient w.r.t. final hidden state
                    var dh = new Vector<T>(_hiddenDim);
                    for (int i = 0; i < _hiddenDim; i++)
                    {
                        T sum = NumOps.Zero;
                        for (int j = 0; j < _inputDim; j++)
                        {
                            sum = NumOps.Add(sum, NumOps.Multiply(Wy[i, j], dOutput[j]));
                        }
                        dh[i] = sum;
                    }

                    // BPTT through time steps
                    var dc = new Vector<T>(_hiddenDim);
                    for (int i = 0; i < _hiddenDim; i++)
                    {
                        dc[i] = NumOps.Zero;
                    }

                    for (int t = seq.Length - 1; t >= 0; t--)
                    {
                        var f = fGates[t];
                        var ig = iGates[t];
                        var cCand = cCandidates[t];
                        var o = oGates[t];
                        var cPrev = t > 0 ? cStates[t] : CreateZeroVector(_hiddenDim);
                        var cCurr = cStates[t + 1];
                        var concat = concats[t];

                        // Gradient of h = o * tanh(c)
                        var tanhC = new Vector<T>(_hiddenDim);
                        for (int i = 0; i < _hiddenDim; i++)
                        {
                            double cVal = NumOps.ToDouble(cCurr[i]);
                            tanhC[i] = NumOps.FromDouble(Math.Tanh(cVal));
                        }

                        var do_gate = new Vector<T>(_hiddenDim);
                        for (int i = 0; i < _hiddenDim; i++)
                        {
                            do_gate[i] = NumOps.Multiply(dh[i], tanhC[i]);
                            double tanhCVal = NumOps.ToDouble(tanhC[i]);
                            double tanhDerivative = 1.0 - tanhCVal * tanhCVal;
                            T dcFromH = NumOps.Multiply(dh[i], NumOps.Multiply(o[i], NumOps.FromDouble(tanhDerivative)));
                            dc[i] = NumOps.Add(dc[i], dcFromH);
                        }

                        var df = new Vector<T>(_hiddenDim);
                        var di = new Vector<T>(_hiddenDim);
                        var dcCand = new Vector<T>(_hiddenDim);

                        for (int i = 0; i < _hiddenDim; i++)
                        {
                            df[i] = NumOps.Multiply(dc[i], cPrev[i]);
                            di[i] = NumOps.Multiply(dc[i], cCand[i]);
                            dcCand[i] = NumOps.Multiply(dc[i], ig[i]);
                        }

                        // Apply gate derivatives
                        var dfPre = new Vector<T>(_hiddenDim);
                        var diPre = new Vector<T>(_hiddenDim);
                        var doPre = new Vector<T>(_hiddenDim);
                        var dcCandPre = new Vector<T>(_hiddenDim);

                        for (int i = 0; i < _hiddenDim; i++)
                        {
                            // Sigmoid derivative: s * (1 - s)
                            double fVal = NumOps.ToDouble(f[i]);
                            double igVal = NumOps.ToDouble(ig[i]);
                            double oVal = NumOps.ToDouble(o[i]);
                            double cCandVal = NumOps.ToDouble(cCand[i]);

                            dfPre[i] = NumOps.Multiply(df[i], NumOps.FromDouble(fVal * (1.0 - fVal)));
                            diPre[i] = NumOps.Multiply(di[i], NumOps.FromDouble(igVal * (1.0 - igVal)));
                            doPre[i] = NumOps.Multiply(do_gate[i], NumOps.FromDouble(oVal * (1.0 - oVal)));
                            // Tanh derivative: 1 - tanh^2
                            dcCandPre[i] = NumOps.Multiply(dcCand[i], NumOps.FromDouble(1.0 - cCandVal * cCandVal));
                        }

                        // Accumulate gradients for gate weights
                        for (int i = 0; i < inputSize; i++)
                        {
                            double concatI = NumOps.ToDouble(concat[i]);
                            for (int j = 0; j < _hiddenDim; j++)
                            {
                                dWf[i, j] += concatI * NumOps.ToDouble(dfPre[j]);
                                dWi[i, j] += concatI * NumOps.ToDouble(diPre[j]);
                                dWc[i, j] += concatI * NumOps.ToDouble(dcCandPre[j]);
                                dWo[i, j] += concatI * NumOps.ToDouble(doPre[j]);
                            }
                        }

                        for (int j = 0; j < _hiddenDim; j++)
                        {
                            dbf[j] += NumOps.ToDouble(dfPre[j]);
                            dbi[j] += NumOps.ToDouble(diPre[j]);
                            dbc[j] += NumOps.ToDouble(dcCandPre[j]);
                            dbo[j] += NumOps.ToDouble(doPre[j]);
                        }

                        // Gradient w.r.t. concat
                        var dConcat = new Vector<T>(inputSize);
                        for (int i = 0; i < inputSize; i++)
                        {
                            T sum = NumOps.Zero;
                            for (int j = 0; j < _hiddenDim; j++)
                            {
                                sum = NumOps.Add(sum, NumOps.Multiply(Wf[i, j], dfPre[j]));
                                sum = NumOps.Add(sum, NumOps.Multiply(Wi[i, j], diPre[j]));
                                sum = NumOps.Add(sum, NumOps.Multiply(Wc[i, j], dcCandPre[j]));
                                sum = NumOps.Add(sum, NumOps.Multiply(Wo[i, j], doPre[j]));
                            }
                            dConcat[i] = sum;
                        }

                        // Extract dh_prev from dConcat
                        var dhPrev = new Vector<T>(_hiddenDim);
                        for (int i = 0; i < _hiddenDim; i++)
                        {
                            dhPrev[i] = dConcat[_inputDim + i];
                        }

                        // Update dc for next iteration
                        var dcPrev = new Vector<T>(_hiddenDim);
                        for (int i = 0; i < _hiddenDim; i++)
                        {
                            dcPrev[i] = NumOps.Multiply(dc[i], f[i]);
                        }

                        dh = dhPrev;
                        dc = dcPrev;
                    }
                }

                // Apply gradients using NumOps
                double lr = _learningRate / actualBatchSize;
                double clipValue = 5.0;

                for (int i = 0; i < inputSize; i++)
                {
                    for (int j = 0; j < _hiddenDim; j++)
                    {
                        double clippedGrad;
                        clippedGrad = Math.Max(-clipValue, Math.Min(clipValue, dWf[i, j]));
                        Wf[i, j] = NumOps.Subtract(Wf[i, j], NumOps.FromDouble(lr * clippedGrad));
                        clippedGrad = Math.Max(-clipValue, Math.Min(clipValue, dWi[i, j]));
                        Wi[i, j] = NumOps.Subtract(Wi[i, j], NumOps.FromDouble(lr * clippedGrad));
                        clippedGrad = Math.Max(-clipValue, Math.Min(clipValue, dWc[i, j]));
                        Wc[i, j] = NumOps.Subtract(Wc[i, j], NumOps.FromDouble(lr * clippedGrad));
                        clippedGrad = Math.Max(-clipValue, Math.Min(clipValue, dWo[i, j]));
                        Wo[i, j] = NumOps.Subtract(Wo[i, j], NumOps.FromDouble(lr * clippedGrad));
                    }
                }

                for (int j = 0; j < _hiddenDim; j++)
                {
                    bf[j] = NumOps.Subtract(bf[j], NumOps.FromDouble(lr * Math.Max(-clipValue, Math.Min(clipValue, dbf[j]))));
                    bi[j] = NumOps.Subtract(bi[j], NumOps.FromDouble(lr * Math.Max(-clipValue, Math.Min(clipValue, dbi[j]))));
                    bc[j] = NumOps.Subtract(bc[j], NumOps.FromDouble(lr * Math.Max(-clipValue, Math.Min(clipValue, dbc[j]))));
                    bo[j] = NumOps.Subtract(bo[j], NumOps.FromDouble(lr * Math.Max(-clipValue, Math.Min(clipValue, dbo[j]))));
                }

                for (int i = 0; i < _hiddenDim; i++)
                {
                    for (int j = 0; j < _inputDim; j++)
                    {
                        double clippedGrad = Math.Max(-clipValue, Math.Min(clipValue, dWy[i, j]));
                        Wy[i, j] = NumOps.Subtract(Wy[i, j], NumOps.FromDouble(lr * clippedGrad));
                    }
                }

                for (int j = 0; j < _inputDim; j++)
                {
                    by[j] = NumOps.Subtract(by[j], NumOps.FromDouble(lr * Math.Max(-clipValue, Math.Min(clipValue, dby[j]))));
                }
            }
        }
    }

    private Vector<T> CreateZeroVector(int size)
    {
        var v = new Vector<T>(size);
        for (int i = 0; i < size; i++)
        {
            v[i] = NumOps.Zero;
        }
        return v;
    }

    private (Vector<T> output, Vector<T>[] hStates, Vector<T>[] cStates,
             Vector<T>[] fGates, Vector<T>[] iGates, Vector<T>[] cCandidates,
             Vector<T>[] oGates, Vector<T>[] concats) ForwardWithCache(Vector<T>[] sequence)
    {
        // Capture nullable fields for proper null checking
        var Wf = _Wf;
        var Wi = _Wi;
        var Wc = _Wc;
        var Wo = _Wo;
        var bf = _bf;
        var bi = _bi;
        var bc = _bc;
        var bo = _bo;
        var Wy = _Wy;
        var by = _by;

        if (Wf == null || Wi == null || Wc == null || Wo == null ||
            bf == null || bi == null || bc == null || bo == null ||
            Wy == null || by == null)
        {
            throw new InvalidOperationException("Weights not initialized.");
        }

        int seqLen = sequence.Length;
        int inputSize = _inputDim + _hiddenDim;

        // Initialize states
        var hStates = new Vector<T>[seqLen + 1];
        var cStates = new Vector<T>[seqLen + 1];
        hStates[0] = CreateZeroVector(_hiddenDim);
        cStates[0] = CreateZeroVector(_hiddenDim);

        // Cache for BPTT
        var fGates = new Vector<T>[seqLen];
        var iGates = new Vector<T>[seqLen];
        var cCandidates = new Vector<T>[seqLen];
        var oGates = new Vector<T>[seqLen];
        var concats = new Vector<T>[seqLen];

        for (int t = 0; t < seqLen; t++)
        {
            var x = sequence[t];
            var hPrev = hStates[t];
            var cPrev = cStates[t];

            // Concatenate input and previous hidden state
            var concat = new Vector<T>(inputSize);
            for (int i = 0; i < _inputDim; i++)
            {
                concat[i] = x[i];
            }
            for (int i = 0; i < _hiddenDim; i++)
            {
                concat[_inputDim + i] = hPrev[i];
            }
            concats[t] = concat;

            // Forget gate
            var f = new Vector<T>(_hiddenDim);
            for (int j = 0; j < _hiddenDim; j++)
            {
                T sum = bf[j];
                for (int i = 0; i < inputSize; i++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(concat[i], Wf[i, j]));
                }
                double sigInput = NumOps.ToDouble(sum);
                f[j] = NumOps.FromDouble(Sigmoid(sigInput));
            }
            fGates[t] = f;

            // Input gate
            var ig = new Vector<T>(_hiddenDim);
            for (int j = 0; j < _hiddenDim; j++)
            {
                T sum = bi[j];
                for (int i = 0; i < inputSize; i++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(concat[i], Wi[i, j]));
                }
                double sigInput = NumOps.ToDouble(sum);
                ig[j] = NumOps.FromDouble(Sigmoid(sigInput));
            }
            iGates[t] = ig;

            // Cell candidate
            var cCand = new Vector<T>(_hiddenDim);
            for (int j = 0; j < _hiddenDim; j++)
            {
                T sum = bc[j];
                for (int i = 0; i < inputSize; i++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(concat[i], Wc[i, j]));
                }
                double tanhInput = NumOps.ToDouble(sum);
                cCand[j] = NumOps.FromDouble(Math.Tanh(tanhInput));
            }
            cCandidates[t] = cCand;

            // Output gate
            var o = new Vector<T>(_hiddenDim);
            for (int j = 0; j < _hiddenDim; j++)
            {
                T sum = bo[j];
                for (int i = 0; i < inputSize; i++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(concat[i], Wo[i, j]));
                }
                double sigInput = NumOps.ToDouble(sum);
                o[j] = NumOps.FromDouble(Sigmoid(sigInput));
            }
            oGates[t] = o;

            // New cell and hidden state
            var cNew = new Vector<T>(_hiddenDim);
            var hNew = new Vector<T>(_hiddenDim);
            for (int j = 0; j < _hiddenDim; j++)
            {
                // c = f * cPrev + i * cCand
                cNew[j] = NumOps.Add(
                    NumOps.Multiply(f[j], cPrev[j]),
                    NumOps.Multiply(ig[j], cCand[j]));
                // h = o * tanh(c)
                double tanhC = Math.Tanh(NumOps.ToDouble(cNew[j]));
                hNew[j] = NumOps.Multiply(o[j], NumOps.FromDouble(tanhC));
            }

            hStates[t + 1] = hNew;
            cStates[t + 1] = cNew;
        }

        // Output layer
        var hFinal = hStates[seqLen];
        var output = new Vector<T>(_inputDim);
        for (int j = 0; j < _inputDim; j++)
        {
            T sum = by[j];
            for (int i = 0; i < _hiddenDim; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(hFinal[i], Wy[i, j]));
            }
            output[j] = sum;
        }

        return (output, hStates, cStates, fGates, iGates, cCandidates, oGates, concats);
    }

    private (Vector<T> output, Vector<T> h, Vector<T> c) Forward(Vector<T>[] sequence)
    {
        // Capture nullable fields for proper null checking
        var Wy = _Wy;
        var by = _by;

        if (Wy == null || by == null)
        {
            throw new InvalidOperationException("Weights not initialized.");
        }

        var h = CreateZeroVector(_hiddenDim);
        var c = CreateZeroVector(_hiddenDim);

        foreach (var x in sequence)
        {
            (h, c) = LSTMCell(x, h, c);
        }

        // Output layer
        var output = new Vector<T>(_inputDim);
        for (int j = 0; j < _inputDim; j++)
        {
            T sum = by[j];
            for (int i = 0; i < _hiddenDim; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(h[i], Wy[i, j]));
            }
            output[j] = sum;
        }

        return (output, h, c);
    }

    private (Vector<T> h, Vector<T> c) LSTMCell(Vector<T> x, Vector<T> hPrev, Vector<T> cPrev)
    {
        // Capture nullable fields for proper null checking
        var Wf = _Wf;
        var Wi = _Wi;
        var Wc = _Wc;
        var Wo = _Wo;
        var bf = _bf;
        var bi = _bi;
        var bc = _bc;
        var bo = _bo;

        if (Wf == null || Wi == null || Wc == null || Wo == null ||
            bf == null || bi == null || bc == null || bo == null)
        {
            throw new InvalidOperationException("Weights not initialized.");
        }

        int inputSize = _inputDim + _hiddenDim;

        // Concatenate input and previous hidden state
        var concat = new Vector<T>(inputSize);
        for (int i = 0; i < _inputDim; i++)
        {
            concat[i] = x[i];
        }
        for (int i = 0; i < _hiddenDim; i++)
        {
            concat[_inputDim + i] = hPrev[i];
        }

        // Forget gate
        var f = new Vector<T>(_hiddenDim);
        for (int j = 0; j < _hiddenDim; j++)
        {
            T sum = bf[j];
            for (int i = 0; i < inputSize; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(concat[i], Wf[i, j]));
            }
            double sigInput = NumOps.ToDouble(sum);
            f[j] = NumOps.FromDouble(Sigmoid(sigInput));
        }

        // Input gate
        var ig = new Vector<T>(_hiddenDim);
        for (int j = 0; j < _hiddenDim; j++)
        {
            T sum = bi[j];
            for (int i = 0; i < inputSize; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(concat[i], Wi[i, j]));
            }
            double sigInput = NumOps.ToDouble(sum);
            ig[j] = NumOps.FromDouble(Sigmoid(sigInput));
        }

        // Cell candidate
        var cCandidate = new Vector<T>(_hiddenDim);
        for (int j = 0; j < _hiddenDim; j++)
        {
            T sum = bc[j];
            for (int i = 0; i < inputSize; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(concat[i], Wc[i, j]));
            }
            double tanhInput = NumOps.ToDouble(sum);
            cCandidate[j] = NumOps.FromDouble(Math.Tanh(tanhInput));
        }

        // Output gate
        var o = new Vector<T>(_hiddenDim);
        for (int j = 0; j < _hiddenDim; j++)
        {
            T sum = bo[j];
            for (int i = 0; i < inputSize; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(concat[i], Wo[i, j]));
            }
            double sigInput = NumOps.ToDouble(sum);
            o[j] = NumOps.FromDouble(Sigmoid(sigInput));
        }

        // New cell state and hidden state
        var cNew = new Vector<T>(_hiddenDim);
        var hNew = new Vector<T>(_hiddenDim);
        for (int j = 0; j < _hiddenDim; j++)
        {
            // c = f * cPrev + i * cCand
            cNew[j] = NumOps.Add(
                NumOps.Multiply(f[j], cPrev[j]),
                NumOps.Multiply(ig[j], cCandidate[j]));
            // h = o * tanh(c)
            double tanhC = Math.Tanh(NumOps.ToDouble(cNew[j]));
            hNew[j] = NumOps.Multiply(o[j], NumOps.FromDouble(tanhC));
        }

        return (hNew, cNew);
    }

    private static double Sigmoid(double x)
    {
        return 1.0 / (1.0 + Math.Exp(-Math.Max(-500, Math.Min(500, x))));
    }

    /// <inheritdoc/>
    public override Vector<T> ScoreAnomalies(Matrix<T> X)
    {
        EnsureFitted();
        return ScoreAnomaliesInternal(X);
    }

    private Vector<T> ScoreAnomaliesInternal(Matrix<T> X)
    {
        ValidateInput(X);

        if (X.Columns != _inputDim)
        {
            throw new ArgumentException(
                $"Input has {X.Columns} features but model was trained with {_inputDim} features.",
                nameof(X));
        }

        var dataMeans = _dataMeans;
        var dataStds = _dataStds;
        if (dataMeans == null || dataStds == null)
        {
            throw new InvalidOperationException("Model not properly fitted. Normalization parameters missing.");
        }

        int n = X.Rows;
        var scores = new Vector<T>(n);

        // Normalize data into Matrix<T>
        var normalizedData = new Matrix<T>(n, _inputDim);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < _inputDim; j++)
            {
                T diff = NumOps.Subtract(X[i, j], dataMeans[j]);
                normalizedData[i, j] = NumOps.Divide(diff, dataStds[j]);
            }
        }

        // Score each point based on prediction error
        for (int i = 0; i < n; i++)
        {
            T score;

            if (i < _seqLength)
            {
                // Not enough history - use simple distance from mean
                score = NumOps.Zero;
                for (int j = 0; j < _inputDim; j++)
                {
                    T val = normalizedData[i, j];
                    score = NumOps.Add(score, NumOps.Multiply(val, val));
                }
            }
            else
            {
                // Build sequence from previous points
                var seq = new Vector<T>[_seqLength];
                for (int t = 0; t < _seqLength; t++)
                {
                    seq[t] = normalizedData.GetRow(i - _seqLength + t);
                }

                // Predict and compute error
                var (prediction, _, _) = Forward(seq);
                score = NumOps.Zero;
                for (int j = 0; j < _inputDim; j++)
                {
                    T diff = NumOps.Subtract(normalizedData[i, j], prediction[j]);
                    score = NumOps.Add(score, NumOps.Multiply(diff, diff));
                }
            }

            scores[i] = score;
        }

        return scores;
    }
}
