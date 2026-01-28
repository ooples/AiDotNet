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
    private double[,]? _Wf; // Forget gate
    private double[,]? _Wi; // Input gate
    private double[,]? _Wc; // Cell gate
    private double[,]? _Wo; // Output gate
    private double[]? _bf;
    private double[]? _bi;
    private double[]? _bc;
    private double[]? _bo;

    // Output layer
    private double[,]? _Wy;
    private double[]? _by;

    private int _inputDim;

    // Normalization parameters
    private double[]? _dataMeans;
    private double[]? _dataStds;

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

        // Convert to double array
        var data = new double[n][];
        for (int i = 0; i < n; i++)
        {
            data[i] = new double[_inputDim];
            for (int j = 0; j < _inputDim; j++)
            {
                data[i][j] = NumOps.ToDouble(X[i, j]);
            }
        }

        // Normalize data
        var (normalizedData, means, stds) = NormalizeData(data);
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

    private (double[][] normalized, double[] means, double[] stds) NormalizeData(double[][] data)
    {
        int n = data.Length;
        int d = data[0].Length;

        var means = new double[d];
        var stds = new double[d];

        for (int j = 0; j < d; j++)
        {
            for (int i = 0; i < n; i++)
            {
                means[j] += data[i][j];
            }
            means[j] /= n;

            for (int i = 0; i < n; i++)
            {
                stds[j] += Math.Pow(data[i][j] - means[j], 2);
            }
            stds[j] = Math.Sqrt(stds[j] / n);
            if (stds[j] < 1e-10) stds[j] = 1;
        }

        var normalized = new double[n][];
        for (int i = 0; i < n; i++)
        {
            normalized[i] = new double[d];
            for (int j = 0; j < d; j++)
            {
                normalized[i][j] = (data[i][j] - means[j]) / stds[j];
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

        _bf = new double[_hiddenDim];
        _bi = new double[_hiddenDim];
        _bc = new double[_hiddenDim];
        _bo = new double[_hiddenDim];

        // Initialize forget gate bias to 1 (helps with gradient flow)
        for (int i = 0; i < _hiddenDim; i++)
        {
            _bf[i] = 1.0;
        }

        _Wy = InitializeMatrix(_hiddenDim, _inputDim, scaleOut);
        _by = new double[_inputDim];
    }

    private double[,] InitializeMatrix(int rows, int cols, double scale)
    {
        var matrix = new double[rows, cols];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                double u1 = 1.0 - _random.NextDouble();
                double u2 = 1.0 - _random.NextDouble();
                matrix[i, j] = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2) * scale;
            }
        }
        return matrix;
    }

    private (double[][][] sequences, double[][] targets) CreateSequences(double[][] data)
    {
        int n = data.Length - _seqLength;
        var sequences = new double[n][][];
        var targets = new double[n][];

        for (int i = 0; i < n; i++)
        {
            sequences[i] = new double[_seqLength][];
            for (int t = 0; t < _seqLength; t++)
            {
                sequences[i][t] = data[i + t];
            }
            targets[i] = data[i + _seqLength];
        }

        return (sequences, targets);
    }

    private void Train(double[][][] sequences, double[][] targets)
    {
        int n = sequences.Length;
        int batchSize = Math.Min(32, n);
        int inputSize = _inputDim + _hiddenDim;

        for (int epoch = 0; epoch < _epochs; epoch++)
        {
            var indices = Enumerable.Range(0, n).OrderBy(_ => _random.NextDouble()).ToArray();

            for (int batch = 0; batch < n; batch += batchSize)
            {
                int actualBatchSize = Math.Min(batchSize, n - batch);

                // Initialize gradient accumulators
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
                    var dOutput = new double[_inputDim];
                    for (int j = 0; j < _inputDim; j++)
                    {
                        dOutput[j] = 2 * (prediction[j] - target[j]);
                    }

                    // Backprop through output layer
                    var hFinal = hStates[seq.Length];
                    for (int j = 0; j < _inputDim; j++)
                    {
                        dby[j] += dOutput[j];
                        for (int i = 0; i < _hiddenDim; i++)
                        {
                            dWy[i, j] += hFinal[i] * dOutput[j];
                        }
                    }

                    // Gradient w.r.t. final hidden state
                    var dh = new double[_hiddenDim];
                    for (int i = 0; i < _hiddenDim; i++)
                    {
                        for (int j = 0; j < _inputDim; j++)
                        {
                            dh[i] += _Wy![i, j] * dOutput[j];
                        }
                    }

                    // BPTT through time steps
                    var dc = new double[_hiddenDim];
                    for (int t = seq.Length - 1; t >= 0; t--)
                    {
                        var f = fGates[t];
                        var ig = iGates[t];
                        var cCand = cCandidates[t];
                        var o = oGates[t];
                        var cPrev = t > 0 ? cStates[t] : new double[_hiddenDim];
                        var cCurr = cStates[t + 1];
                        var concat = concats[t];

                        // Gradient of h = o * tanh(c)
                        // dL/do = dL/dh * tanh(c)
                        // dL/dc += dL/dh * o * (1 - tanh(c)^2)
                        var tanhC = new double[_hiddenDim];
                        for (int i = 0; i < _hiddenDim; i++)
                        {
                            tanhC[i] = Math.Tanh(cCurr[i]);
                        }

                        var do_gate = new double[_hiddenDim];
                        for (int i = 0; i < _hiddenDim; i++)
                        {
                            do_gate[i] = dh[i] * tanhC[i];
                            dc[i] += dh[i] * o[i] * (1 - tanhC[i] * tanhC[i]);
                        }

                        // Gradient of c = f * c_prev + i * c_candidate
                        // dL/df = dL/dc * c_prev
                        // dL/di = dL/dc * c_candidate
                        // dL/dc_candidate = dL/dc * i
                        // dL/dc_prev = dL/dc * f (carried to next iteration)
                        var df = new double[_hiddenDim];
                        var di = new double[_hiddenDim];
                        var dcCand = new double[_hiddenDim];

                        for (int i = 0; i < _hiddenDim; i++)
                        {
                            df[i] = dc[i] * cPrev[i];
                            di[i] = dc[i] * cCand[i];
                            dcCand[i] = dc[i] * ig[i];
                        }

                        // Apply gate derivatives (sigmoid: s' = s*(1-s), tanh: t' = 1-t^2)
                        var dfPre = new double[_hiddenDim];
                        var diPre = new double[_hiddenDim];
                        var doPre = new double[_hiddenDim];
                        var dcCandPre = new double[_hiddenDim];

                        for (int i = 0; i < _hiddenDim; i++)
                        {
                            dfPre[i] = df[i] * f[i] * (1 - f[i]);
                            diPre[i] = di[i] * ig[i] * (1 - ig[i]);
                            doPre[i] = do_gate[i] * o[i] * (1 - o[i]);
                            dcCandPre[i] = dcCand[i] * (1 - cCand[i] * cCand[i]);
                        }

                        // Accumulate gradients for gate weights
                        for (int i = 0; i < inputSize; i++)
                        {
                            for (int j = 0; j < _hiddenDim; j++)
                            {
                                dWf[i, j] += concat[i] * dfPre[j];
                                dWi[i, j] += concat[i] * diPre[j];
                                dWc[i, j] += concat[i] * dcCandPre[j];
                                dWo[i, j] += concat[i] * doPre[j];
                            }
                        }

                        for (int j = 0; j < _hiddenDim; j++)
                        {
                            dbf[j] += dfPre[j];
                            dbi[j] += diPre[j];
                            dbc[j] += dcCandPre[j];
                            dbo[j] += doPre[j];
                        }

                        // Gradient w.r.t. concat = [x, h_prev]
                        // We need dL/dh_prev for the next iteration
                        var dConcat = new double[inputSize];
                        for (int i = 0; i < inputSize; i++)
                        {
                            for (int j = 0; j < _hiddenDim; j++)
                            {
                                dConcat[i] += _Wf![i, j] * dfPre[j];
                                dConcat[i] += _Wi![i, j] * diPre[j];
                                dConcat[i] += _Wc![i, j] * dcCandPre[j];
                                dConcat[i] += _Wo![i, j] * doPre[j];
                            }
                        }

                        // Extract dh_prev from dConcat (last _hiddenDim elements)
                        var dhPrev = new double[_hiddenDim];
                        for (int i = 0; i < _hiddenDim; i++)
                        {
                            dhPrev[i] = dConcat[_inputDim + i];
                        }

                        // Update dc for next iteration (dc_prev = dc * f)
                        var dcPrev = new double[_hiddenDim];
                        for (int i = 0; i < _hiddenDim; i++)
                        {
                            dcPrev[i] = dc[i] * f[i];
                        }

                        dh = dhPrev;
                        dc = dcPrev;
                    }
                }

                // Apply gradients
                double lr = _learningRate / actualBatchSize;
                double clipValue = 5.0; // Gradient clipping

                for (int i = 0; i < inputSize; i++)
                {
                    for (int j = 0; j < _hiddenDim; j++)
                    {
                        _Wf![i, j] -= lr * Math.Max(-clipValue, Math.Min(clipValue, dWf[i, j]));
                        _Wi![i, j] -= lr * Math.Max(-clipValue, Math.Min(clipValue, dWi[i, j]));
                        _Wc![i, j] -= lr * Math.Max(-clipValue, Math.Min(clipValue, dWc[i, j]));
                        _Wo![i, j] -= lr * Math.Max(-clipValue, Math.Min(clipValue, dWo[i, j]));
                    }
                }

                for (int j = 0; j < _hiddenDim; j++)
                {
                    _bf![j] -= lr * Math.Max(-clipValue, Math.Min(clipValue, dbf[j]));
                    _bi![j] -= lr * Math.Max(-clipValue, Math.Min(clipValue, dbi[j]));
                    _bc![j] -= lr * Math.Max(-clipValue, Math.Min(clipValue, dbc[j]));
                    _bo![j] -= lr * Math.Max(-clipValue, Math.Min(clipValue, dbo[j]));
                }

                for (int i = 0; i < _hiddenDim; i++)
                {
                    for (int j = 0; j < _inputDim; j++)
                    {
                        _Wy![i, j] -= lr * Math.Max(-clipValue, Math.Min(clipValue, dWy[i, j]));
                    }
                }

                for (int j = 0; j < _inputDim; j++)
                {
                    _by![j] -= lr * Math.Max(-clipValue, Math.Min(clipValue, dby[j]));
                }
            }
        }
    }

    private (double[] output, double[][] hStates, double[][] cStates,
             double[][] fGates, double[][] iGates, double[][] cCandidates,
             double[][] oGates, double[][] concats) ForwardWithCache(double[][] sequence)
    {
        int seqLen = sequence.Length;
        int inputSize = _inputDim + _hiddenDim;

        // Initialize states
        var hStates = new double[seqLen + 1][];
        var cStates = new double[seqLen + 1][];
        hStates[0] = new double[_hiddenDim];
        cStates[0] = new double[_hiddenDim];

        // Cache for BPTT
        var fGates = new double[seqLen][];
        var iGates = new double[seqLen][];
        var cCandidates = new double[seqLen][];
        var oGates = new double[seqLen][];
        var concats = new double[seqLen][];

        for (int t = 0; t < seqLen; t++)
        {
            var x = sequence[t];
            var hPrev = hStates[t];
            var cPrev = cStates[t];

            // Concatenate input and previous hidden state
            var concat = new double[inputSize];
            Array.Copy(x, 0, concat, 0, _inputDim);
            Array.Copy(hPrev, 0, concat, _inputDim, _hiddenDim);
            concats[t] = concat;

            // Forget gate
            var f = new double[_hiddenDim];
            for (int j = 0; j < _hiddenDim; j++)
            {
                f[j] = _bf![j];
                for (int i = 0; i < inputSize; i++)
                {
                    f[j] += concat[i] * _Wf![i, j];
                }
                f[j] = Sigmoid(f[j]);
            }
            fGates[t] = f;

            // Input gate
            var ig = new double[_hiddenDim];
            for (int j = 0; j < _hiddenDim; j++)
            {
                ig[j] = _bi![j];
                for (int i = 0; i < inputSize; i++)
                {
                    ig[j] += concat[i] * _Wi![i, j];
                }
                ig[j] = Sigmoid(ig[j]);
            }
            iGates[t] = ig;

            // Cell candidate
            var cCand = new double[_hiddenDim];
            for (int j = 0; j < _hiddenDim; j++)
            {
                cCand[j] = _bc![j];
                for (int i = 0; i < inputSize; i++)
                {
                    cCand[j] += concat[i] * _Wc![i, j];
                }
                cCand[j] = Math.Tanh(cCand[j]);
            }
            cCandidates[t] = cCand;

            // Output gate
            var o = new double[_hiddenDim];
            for (int j = 0; j < _hiddenDim; j++)
            {
                o[j] = _bo![j];
                for (int i = 0; i < inputSize; i++)
                {
                    o[j] += concat[i] * _Wo![i, j];
                }
                o[j] = Sigmoid(o[j]);
            }
            oGates[t] = o;

            // New cell and hidden state
            var cNew = new double[_hiddenDim];
            var hNew = new double[_hiddenDim];
            for (int j = 0; j < _hiddenDim; j++)
            {
                cNew[j] = f[j] * cPrev[j] + ig[j] * cCand[j];
                hNew[j] = o[j] * Math.Tanh(cNew[j]);
            }

            hStates[t + 1] = hNew;
            cStates[t + 1] = cNew;
        }

        // Output layer
        var hFinal = hStates[seqLen];
        var output = new double[_inputDim];
        for (int j = 0; j < _inputDim; j++)
        {
            output[j] = _by![j];
            for (int i = 0; i < _hiddenDim; i++)
            {
                output[j] += hFinal[i] * _Wy![i, j];
            }
        }

        return (output, hStates, cStates, fGates, iGates, cCandidates, oGates, concats);
    }

    private (double[] output, double[] h, double[] c) Forward(double[][] sequence)
    {
        var h = new double[_hiddenDim];
        var c = new double[_hiddenDim];

        foreach (var x in sequence)
        {
            (h, c) = LSTMCell(x, h, c);
        }

        // Output layer
        var output = new double[_inputDim];
        for (int j = 0; j < _inputDim; j++)
        {
            output[j] = _by![j];
            for (int i = 0; i < _hiddenDim; i++)
            {
                output[j] += h[i] * _Wy![i, j];
            }
        }

        return (output, h, c);
    }

    private (double[] h, double[] c) LSTMCell(double[] x, double[] hPrev, double[] cPrev)
    {
        int inputSize = _inputDim + _hiddenDim;
        var concat = new double[inputSize];
        Array.Copy(x, 0, concat, 0, _inputDim);
        Array.Copy(hPrev, 0, concat, _inputDim, _hiddenDim);

        // Forget gate
        var f = new double[_hiddenDim];
        for (int j = 0; j < _hiddenDim; j++)
        {
            f[j] = _bf![j];
            for (int i = 0; i < inputSize; i++)
            {
                f[j] += concat[i] * _Wf![i, j];
            }
            f[j] = Sigmoid(f[j]);
        }

        // Input gate
        var ig = new double[_hiddenDim];
        for (int j = 0; j < _hiddenDim; j++)
        {
            ig[j] = _bi![j];
            for (int i = 0; i < inputSize; i++)
            {
                ig[j] += concat[i] * _Wi![i, j];
            }
            ig[j] = Sigmoid(ig[j]);
        }

        // Cell candidate
        var cCandidate = new double[_hiddenDim];
        for (int j = 0; j < _hiddenDim; j++)
        {
            cCandidate[j] = _bc![j];
            for (int i = 0; i < inputSize; i++)
            {
                cCandidate[j] += concat[i] * _Wc![i, j];
            }
            cCandidate[j] = Math.Tanh(cCandidate[j]);
        }

        // Output gate
        var o = new double[_hiddenDim];
        for (int j = 0; j < _hiddenDim; j++)
        {
            o[j] = _bo![j];
            for (int i = 0; i < inputSize; i++)
            {
                o[j] += concat[i] * _Wo![i, j];
            }
            o[j] = Sigmoid(o[j]);
        }

        // New cell state and hidden state
        var cNew = new double[_hiddenDim];
        var hNew = new double[_hiddenDim];
        for (int j = 0; j < _hiddenDim; j++)
        {
            cNew[j] = f[j] * cPrev[j] + ig[j] * cCandidate[j];
            hNew[j] = o[j] * Math.Tanh(cNew[j]);
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

        // Convert to normalized double array
        var data = new double[n][];
        for (int i = 0; i < n; i++)
        {
            data[i] = new double[_inputDim];
            for (int j = 0; j < _inputDim; j++)
            {
                data[i][j] = (NumOps.ToDouble(X[i, j]) - dataMeans[j]) / dataStds[j];
            }
        }

        // Score each point based on prediction error
        for (int i = 0; i < n; i++)
        {
            double score;

            if (i < _seqLength)
            {
                // Not enough history - use simple distance from mean
                score = 0;
                for (int j = 0; j < _inputDim; j++)
                {
                    score += data[i][j] * data[i][j];
                }
            }
            else
            {
                // Build sequence from previous points
                var seq = new double[_seqLength][];
                for (int t = 0; t < _seqLength; t++)
                {
                    seq[t] = data[i - _seqLength + t];
                }

                // Predict and compute error
                var (prediction, _, _) = Forward(seq);
                score = 0;
                for (int j = 0; j < _inputDim; j++)
                {
                    score += Math.Pow(data[i][j] - prediction[j], 2);
                }
            }

            scores[i] = NumOps.FromDouble(score);
        }

        return scores;
    }
}
