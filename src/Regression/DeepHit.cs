using AiDotNet.Helpers;
using AiDotNet.Models.Options;

namespace AiDotNet.Regression;

/// <summary>
/// DeepHit: A deep learning approach to survival analysis with competing risks.
/// </summary>
/// <remarks>
/// <para>
/// DeepHit directly learns the distribution of survival times without making the proportional
/// hazards assumption. It outputs the probability mass function (PMF) of event times across
/// discrete time bins and can handle multiple competing risks.
/// </para>
/// <para>
/// <b>For Beginners:</b> Unlike DeepSurv (which assumes factors affect risk proportionally at all times),
/// DeepHit learns the actual probability of an event at each specific time point. This is useful when:
///
/// - Risk factors affect survival differently at different times
/// - You want to predict exact probabilities at specific time horizons
/// - You have competing risks (multiple ways an event can happen)
///
/// Example: "What's the probability a patient experiences disease recurrence (risk 1) vs side effects (risk 2)
/// within 1 year, 2 years, or 5 years?"
///
/// Key concepts:
/// - Time bins: The time axis is divided into discrete bins (e.g., months 0-12, 12-24, 24-36...)
/// - PMF: Probability Mass Function - probability of event at each time bin
/// - CIF: Cumulative Incidence Function - probability of event by time t
/// - Survival: Probability of no event by time t
/// </para>
/// <para>
/// Reference: Lee, C. et al. (2018). "DeepHit: A Deep Learning Approach to Survival Analysis
/// with Competing Risks". AAAI Conference on Artificial Intelligence.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class DeepHit<T> : AsyncDecisionTreeRegressionBase<T>
{
    /// <summary>
    /// Shared network weights.
    /// </summary>
    private List<T[,]> _sharedWeights;

    /// <summary>
    /// Shared network biases.
    /// </summary>
    private List<T[]> _sharedBiases;

    /// <summary>
    /// Cause-specific network weights (one list per cause).
    /// </summary>
    private List<List<T[,]>> _causeWeights;

    /// <summary>
    /// Cause-specific network biases (one list per cause).
    /// </summary>
    private List<List<T[]>> _causeBiases;

    /// <summary>
    /// Output layer weights (for each cause, maps to time bins).
    /// </summary>
    private List<T[,]> _outputWeights;

    /// <summary>
    /// Output layer biases (for each cause).
    /// </summary>
    private List<T[]> _outputBiases;

    /// <summary>
    /// Number of features.
    /// </summary>
    private int _numFeatures;

    /// <summary>
    /// Time bin edges (discretization of time axis).
    /// </summary>
    private T[]? _timeBinEdges;

    /// <summary>
    /// Configuration options.
    /// </summary>
    private readonly DeepHitOptions _options;

    /// <summary>
    /// Random number generator.
    /// </summary>
    private readonly Random _random;

    /// <inheritdoc/>
    public override int NumberOfTrees => 1;

    /// <summary>
    /// Initializes a new instance of DeepHit.
    /// </summary>
    /// <param name="options">Configuration options.</param>
    /// <param name="regularization">Optional regularization.</param>
    public DeepHit(DeepHitOptions? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(null, regularization)
    {
        _options = options ?? new DeepHitOptions();
        _sharedWeights = [];
        _sharedBiases = [];
        _causeWeights = [];
        _causeBiases = [];
        _outputWeights = [];
        _outputBiases = [];
        _numFeatures = 0;
        _random = _options.Seed.HasValue ? RandomHelper.CreateSeededRandom(_options.Seed.Value) : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Trains the DeepHit model.
    /// </summary>
    /// <param name="x">Feature matrix.</param>
    /// <param name="times">Observed times (time to event or censoring).</param>
    /// <param name="events">Event indicators (0 = censored, 1..K = event type for K competing risks).</param>
    public async Task TrainAsync(Matrix<T> x, Vector<T> times, Vector<T> events)
    {
        _numFeatures = x.Columns;
        int n = x.Rows;

        // Create time bins
        InitializeTimeBins(times);

        // Initialize network architecture
        InitializeNetwork();

        // Convert times to bin indices
        var timeBinIndices = ConvertTimesToBins(times);

        double bestLoss = double.MaxValue;
        int patienceCounter = 0;
        var bestWeights = SaveWeights();

        for (int epoch = 0; epoch < _options.Epochs; epoch++)
        {
            // Mini-batch training
            var shuffledIndices = ShuffleArray(Enumerable.Range(0, n).ToArray());
            double epochLoss = 0;
            int numBatches = 0;

            for (int b = 0; b < n; b += _options.BatchSize)
            {
                int batchEnd = Math.Min(b + _options.BatchSize, n);
                var batchIndices = shuffledIndices.Skip(b).Take(batchEnd - b).ToArray();

                // Forward pass
                var (pmfs, sharedOutputs, causeOutputs) = ForwardPass(x, batchIndices);

                // Compute loss and gradients
                var (loss, gradients) = ComputeLossAndGradients(
                    pmfs, timeBinIndices, events, batchIndices);

                epochLoss += loss;
                numBatches++;

                // Backward pass and update
                BackwardPass(x, batchIndices, sharedOutputs, causeOutputs, gradients);
            }

            epochLoss /= numBatches;

            // Early stopping
            if (_options.EarlyStoppingPatience.HasValue)
            {
                if (epochLoss < bestLoss - 1e-6)
                {
                    bestLoss = epochLoss;
                    patienceCounter = 0;
                    bestWeights = SaveWeights();
                }
                else
                {
                    patienceCounter++;
                    if (patienceCounter >= _options.EarlyStoppingPatience.Value)
                    {
                        RestoreWeights(bestWeights);
                        break;
                    }
                }
            }
        }

        await CalculateFeatureImportancesAsync(x.Columns);
    }

    /// <inheritdoc/>
    public override async Task TrainAsync(Matrix<T> x, Vector<T> y)
    {
        // For standard interface, assume all events are type 1 (single risk)
        var events = new Vector<T>(y.Length);
        for (int i = 0; i < y.Length; i++)
        {
            events[i] = NumOps.One;
        }

        await TrainAsync(x, y, events);
    }

    /// <inheritdoc/>
    public override async Task<Vector<T>> PredictAsync(Matrix<T> input)
    {
        // Return expected event time (weighted average of time bin centers)
        return await Task.Run(() => PredictExpectedTime(input));
    }

    /// <summary>
    /// Predicts the probability mass function (PMF) of event time for each sample.
    /// </summary>
    /// <param name="input">Input feature matrix.</param>
    /// <returns>Array of PMFs, where pmfs[sample][risk][timeBin] is the probability.</returns>
    public T[,,] PredictPMF(Matrix<T> input)
    {
        var indices = Enumerable.Range(0, input.Rows).ToArray();
        var (pmfs, _, _) = ForwardPass(input, indices);

        var result = new T[input.Rows, _options.NumRisks, _options.NumTimeBins];
        for (int i = 0; i < input.Rows; i++)
        {
            for (int k = 0; k < _options.NumRisks; k++)
            {
                for (int t = 0; t < _options.NumTimeBins; t++)
                {
                    result[i, k, t] = pmfs[i][k][t];
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Predicts survival probability S(t) = P(T > t) at specified times.
    /// </summary>
    /// <param name="input">Input feature matrix.</param>
    /// <param name="times">Times at which to evaluate survival probability.</param>
    /// <returns>Matrix where [i,j] is P(T > times[j] | X_i).</returns>
    public Matrix<T> PredictSurvival(Matrix<T> input, T[] times)
    {
        var pmf = PredictPMF(input);
        var survivalProbs = new Matrix<T>(input.Rows, times.Length);

        for (int i = 0; i < input.Rows; i++)
        {
            for (int j = 0; j < times.Length; j++)
            {
                double t = NumOps.ToDouble(times[j]);
                int binIndex = GetTimeBinIndex(t);

                // S(t) = 1 - sum of PMF up to time bin
                double cumProb = 0;
                for (int k = 0; k < _options.NumRisks; k++)
                {
                    for (int b = 0; b <= binIndex && b < _options.NumTimeBins; b++)
                    {
                        cumProb += NumOps.ToDouble(pmf[i, k, b]);
                    }
                }

                survivalProbs[i, j] = NumOps.FromDouble(Math.Max(0, 1 - cumProb));
            }
        }

        return survivalProbs;
    }

    /// <summary>
    /// Predicts cumulative incidence function (CIF) for a specific risk.
    /// </summary>
    /// <param name="input">Input feature matrix.</param>
    /// <param name="times">Times at which to evaluate CIF.</param>
    /// <param name="riskIndex">Index of the risk (0 to NumRisks-1).</param>
    /// <returns>Matrix where [i,j] is P(T ≤ times[j], event type = risk | X_i).</returns>
    public Matrix<T> PredictCIF(Matrix<T> input, T[] times, int riskIndex = 0)
    {
        if (riskIndex < 0 || riskIndex >= _options.NumRisks)
        {
            throw new ArgumentOutOfRangeException(nameof(riskIndex),
                $"Risk index must be between 0 and {_options.NumRisks - 1}");
        }

        var pmf = PredictPMF(input);
        var cif = new Matrix<T>(input.Rows, times.Length);

        for (int i = 0; i < input.Rows; i++)
        {
            for (int j = 0; j < times.Length; j++)
            {
                double t = NumOps.ToDouble(times[j]);
                int binIndex = GetTimeBinIndex(t);

                // CIF(t, k) = sum of PMF_k up to time bin
                double cumProb = 0;
                for (int b = 0; b <= binIndex && b < _options.NumTimeBins; b++)
                {
                    cumProb += NumOps.ToDouble(pmf[i, riskIndex, b]);
                }

                cif[i, j] = NumOps.FromDouble(cumProb);
            }
        }

        return cif;
    }

    /// <summary>
    /// Predicts expected time to event.
    /// </summary>
    /// <param name="input">Input feature matrix.</param>
    /// <returns>Vector of expected event times.</returns>
    public Vector<T> PredictExpectedTime(Matrix<T> input)
    {
        var pmf = PredictPMF(input);
        var expectedTimes = new Vector<T>(input.Rows);

        for (int i = 0; i < input.Rows; i++)
        {
            double expected = 0;
            double totalProb = 0;

            for (int k = 0; k < _options.NumRisks; k++)
            {
                for (int t = 0; t < _options.NumTimeBins; t++)
                {
                    double prob = NumOps.ToDouble(pmf[i, k, t]);
                    double time = GetTimeBinCenter(t);
                    expected += prob * time;
                    totalProb += prob;
                }
            }

            // Normalize by total probability (may be < 1 for censored observations)
            expectedTimes[i] = totalProb > 0
                ? NumOps.FromDouble(expected / totalProb)
                : NumOps.FromDouble(GetTimeBinCenter(_options.NumTimeBins - 1));
        }

        return expectedTimes;
    }

    /// <summary>
    /// Predicts median survival time for each sample.
    /// </summary>
    /// <param name="input">Input feature matrix.</param>
    /// <returns>Vector of median survival times.</returns>
    public Vector<T> PredictMedianSurvivalTime(Matrix<T> input)
    {
        var pmf = PredictPMF(input);
        var medianTimes = new Vector<T>(input.Rows);

        for (int i = 0; i < input.Rows; i++)
        {
            double cumProb = 0;

            // Find time bin where cumulative probability crosses 0.5
            for (int t = 0; t < _options.NumTimeBins; t++)
            {
                for (int k = 0; k < _options.NumRisks; k++)
                {
                    cumProb += NumOps.ToDouble(pmf[i, k, t]);
                }

                if (cumProb >= 0.5)
                {
                    medianTimes[i] = NumOps.FromDouble(GetTimeBinCenter(t));
                    break;
                }
            }

            if (cumProb < 0.5)
            {
                medianTimes[i] = NumOps.FromDouble(GetTimeBinCenter(_options.NumTimeBins - 1));
            }
        }

        return medianTimes;
    }

    /// <summary>
    /// Computes the concordance index (C-index) for model evaluation.
    /// </summary>
    /// <param name="x">Feature matrix.</param>
    /// <param name="times">Observed times.</param>
    /// <param name="events">Event indicators.</param>
    /// <returns>C-index between 0 and 1 (0.5 = random, 1 = perfect).</returns>
    public double ComputeCIndex(Matrix<T> x, Vector<T> times, Vector<T> events)
    {
        var expectedTimes = PredictExpectedTime(x);
        int concordant = 0;
        int discordant = 0;

        for (int i = 0; i < x.Rows; i++)
        {
            if (NumOps.ToDouble(events[i]) == 0) continue;  // Skip censored

            for (int j = 0; j < x.Rows; j++)
            {
                if (i == j) continue;

                double ti = NumOps.ToDouble(times[i]);
                double tj = NumOps.ToDouble(times[j]);

                // i had event before j
                if (ti < tj)
                {
                    // DeepHit predicts expected time - lower expected time = higher risk
                    double pi = NumOps.ToDouble(expectedTimes[i]);
                    double pj = NumOps.ToDouble(expectedTimes[j]);

                    if (pi < pj) concordant++;  // Correct: i predicted to fail earlier
                    else if (pi > pj) discordant++;
                }
            }
        }

        int total = concordant + discordant;
        return total > 0 ? (double)concordant / total : 0.5;
    }

    /// <summary>
    /// Computes the time-dependent AUC at specific time horizons.
    /// </summary>
    /// <param name="x">Feature matrix.</param>
    /// <param name="times">Observed times.</param>
    /// <param name="events">Event indicators.</param>
    /// <param name="horizon">Time horizon for evaluation.</param>
    /// <returns>AUC at the specified horizon.</returns>
    public double ComputeTimeDependentAUC(Matrix<T> x, Vector<T> times, Vector<T> events, T horizon)
    {
        var horizonArray = new[] { horizon };
        var survivalProbs = PredictSurvival(x, horizonArray);

        double horizonValue = NumOps.ToDouble(horizon);
        var cases = new List<int>();
        var controls = new List<int>();

        for (int i = 0; i < x.Rows; i++)
        {
            double ti = NumOps.ToDouble(times[i]);
            double ei = NumOps.ToDouble(events[i]);

            if (ti <= horizonValue && ei > 0)
            {
                cases.Add(i);  // Event occurred before horizon
            }
            else if (ti > horizonValue)
            {
                controls.Add(i);  // Survived past horizon
            }
        }

        if (cases.Count == 0 || controls.Count == 0)
        {
            return 0.5;
        }

        // Compute AUC
        int concordant = 0;
        int total = 0;

        foreach (int caseIdx in cases)
        {
            double caseRisk = 1 - NumOps.ToDouble(survivalProbs[caseIdx, 0]);  // Risk = 1 - S(t)

            foreach (int controlIdx in controls)
            {
                double controlRisk = 1 - NumOps.ToDouble(survivalProbs[controlIdx, 0]);
                total++;

                if (caseRisk > controlRisk)
                {
                    concordant++;
                }
                else if (Math.Abs(caseRisk - controlRisk) < 1e-10)
                {
                    // Tie - count as 0.5
                    concordant++;
                    total++;
                }
            }
        }

        return total > 0 ? (double)concordant / total : 0.5;
    }

    /// <summary>
    /// Initializes time bin edges based on observed times.
    /// </summary>
    private void InitializeTimeBins(Vector<T> times)
    {
        double minTime = times.Min(t => NumOps.ToDouble(t));
        double maxTime = times.Max(t => NumOps.ToDouble(t));

        // Add small buffer
        double range = maxTime - minTime;
        maxTime += range * 0.01;

        _timeBinEdges = new T[_options.NumTimeBins + 1];
        double binWidth = (maxTime - minTime) / _options.NumTimeBins;

        for (int i = 0; i <= _options.NumTimeBins; i++)
        {
            _timeBinEdges[i] = NumOps.FromDouble(minTime + i * binWidth);
        }
    }

    /// <summary>
    /// Converts times to bin indices.
    /// </summary>
    private int[] ConvertTimesToBins(Vector<T> times)
    {
        var binIndices = new int[times.Length];

        for (int i = 0; i < times.Length; i++)
        {
            binIndices[i] = GetTimeBinIndex(NumOps.ToDouble(times[i]));
        }

        return binIndices;
    }

    /// <summary>
    /// Gets the bin index for a given time.
    /// </summary>
    private int GetTimeBinIndex(double time)
    {
        if (_timeBinEdges == null)
        {
            return 0;
        }

        for (int i = 1; i < _timeBinEdges.Length; i++)
        {
            if (time < NumOps.ToDouble(_timeBinEdges[i]))
            {
                return i - 1;
            }
        }

        return _options.NumTimeBins - 1;
    }

    /// <summary>
    /// Gets the center time of a bin.
    /// </summary>
    private double GetTimeBinCenter(int binIndex)
    {
        if (_timeBinEdges == null)
        {
            return binIndex;
        }

        double left = NumOps.ToDouble(_timeBinEdges[binIndex]);
        double right = NumOps.ToDouble(_timeBinEdges[Math.Min(binIndex + 1, _timeBinEdges.Length - 1)]);
        return (left + right) / 2;
    }

    /// <summary>
    /// Initializes the neural network architecture.
    /// </summary>
    private void InitializeNetwork()
    {
        _sharedWeights = [];
        _sharedBiases = [];
        _causeWeights = [];
        _causeBiases = [];
        _outputWeights = [];
        _outputBiases = [];

        // Shared sub-network
        int inputSize = _numFeatures;
        for (int layer = 0; layer < _options.NumSharedLayers; layer++)
        {
            int outputSize = _options.HiddenLayerSize;
            _sharedWeights.Add(InitializeWeights(inputSize, outputSize));
            _sharedBiases.Add(InitializeBiases(outputSize));
            inputSize = outputSize;
        }

        int sharedOutputSize = inputSize;

        // Cause-specific sub-networks
        for (int k = 0; k < _options.NumRisks; k++)
        {
            var causeW = new List<T[,]>();
            var causeB = new List<T[]>();

            inputSize = sharedOutputSize;
            for (int layer = 0; layer < _options.NumCauseLayers; layer++)
            {
                int outputSize = _options.HiddenLayerSize;
                causeW.Add(InitializeWeights(inputSize, outputSize));
                causeB.Add(InitializeBiases(outputSize));
                inputSize = outputSize;
            }

            _causeWeights.Add(causeW);
            _causeBiases.Add(causeB);

            // Output layer for this cause (maps to time bins)
            _outputWeights.Add(InitializeWeights(inputSize, _options.NumTimeBins));
            _outputBiases.Add(InitializeBiases(_options.NumTimeBins));
        }
    }

    /// <summary>
    /// Initializes weight matrix with He initialization.
    /// </summary>
    private T[,] InitializeWeights(int inputSize, int outputSize)
    {
        double scale = Math.Sqrt(2.0 / inputSize);
        var w = new T[inputSize, outputSize];

        for (int i = 0; i < inputSize; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                w[i, j] = NumOps.FromDouble((_random.NextDouble() * 2 - 1) * scale);
            }
        }

        return w;
    }

    /// <summary>
    /// Initializes bias array with zeros.
    /// </summary>
    private T[] InitializeBiases(int size)
    {
        return new T[size];
    }

    /// <summary>
    /// Forward pass through the network.
    /// </summary>
    private (T[][][], T[][], List<T[][]>) ForwardPass(Matrix<T> x, int[] indices)
    {
        int n = indices.Length;

        // Extract input for batch
        var current = new T[n][];
        for (int i = 0; i < n; i++)
        {
            current[i] = new T[_numFeatures];
            for (int j = 0; j < _numFeatures; j++)
            {
                current[i][j] = x[indices[i], j];
            }
        }

        // Shared layers
        for (int layer = 0; layer < _sharedWeights.Count; layer++)
        {
            current = ApplyLayer(current, _sharedWeights[layer], _sharedBiases[layer], true);
        }

        var sharedOutput = current;

        // Cause-specific layers and outputs
        var causeOutputs = new List<T[][]>();
        var pmfs = new T[n][][];
        for (int i = 0; i < n; i++)
        {
            pmfs[i] = new T[_options.NumRisks][];
        }

        for (int k = 0; k < _options.NumRisks; k++)
        {
            current = CloneArray(sharedOutput);

            for (int layer = 0; layer < _causeWeights[k].Count; layer++)
            {
                current = ApplyLayer(current, _causeWeights[k][layer], _causeBiases[k][layer], true);
            }

            causeOutputs.Add(current);

            // Output layer (no activation - logits)
            var logits = ApplyLayer(current, _outputWeights[k], _outputBiases[k], false);

            // Store PMF for this cause
            for (int i = 0; i < n; i++)
            {
                pmfs[i][k] = logits[i];
            }
        }

        // Apply softmax across all causes and time bins
        ApplySoftmaxAcrossAll(pmfs);

        return (pmfs, sharedOutput, causeOutputs);
    }

    /// <summary>
    /// Applies a single layer.
    /// </summary>
    private T[][] ApplyLayer(T[][] input, T[,] weights, T[] biases, bool applyActivation)
    {
        int n = input.Length;
        int outputSize = weights.GetLength(1);

        var output = new T[n][];
        for (int i = 0; i < n; i++)
        {
            output[i] = new T[outputSize];
            for (int j = 0; j < outputSize; j++)
            {
                double sum = NumOps.ToDouble(biases[j]);
                for (int k = 0; k < input[i].Length; k++)
                {
                    sum += NumOps.ToDouble(input[i][k]) * NumOps.ToDouble(weights[k, j]);
                }

                output[i][j] = applyActivation
                    ? NumOps.FromDouble(ApplyActivation(sum))
                    : NumOps.FromDouble(sum);
            }
        }

        // Apply dropout during training (simplified - always apply with prob)
        if (applyActivation && _options.DropoutRate > 0)
        {
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < outputSize; j++)
                {
                    if (_random.NextDouble() < _options.DropoutRate)
                    {
                        output[i][j] = NumOps.Zero;
                    }
                    else
                    {
                        output[i][j] = NumOps.FromDouble(
                            NumOps.ToDouble(output[i][j]) / (1 - _options.DropoutRate));
                    }
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Applies softmax across all causes and time bins.
    /// </summary>
    private void ApplySoftmaxAcrossAll(T[][][] pmfs)
    {
        int n = pmfs.Length;

        for (int i = 0; i < n; i++)
        {
            // Collect all logits
            double maxLogit = double.MinValue;
            for (int k = 0; k < _options.NumRisks; k++)
            {
                for (int t = 0; t < _options.NumTimeBins; t++)
                {
                    double logit = NumOps.ToDouble(pmfs[i][k][t]);
                    if (logit > maxLogit) maxLogit = logit;
                }
            }

            // Softmax with numerical stability
            double sumExp = 0;
            for (int k = 0; k < _options.NumRisks; k++)
            {
                for (int t = 0; t < _options.NumTimeBins; t++)
                {
                    double expVal = Math.Exp(NumOps.ToDouble(pmfs[i][k][t]) - maxLogit);
                    pmfs[i][k][t] = NumOps.FromDouble(expVal);
                    sumExp += expVal;
                }
            }

            // Normalize
            for (int k = 0; k < _options.NumRisks; k++)
            {
                for (int t = 0; t < _options.NumTimeBins; t++)
                {
                    pmfs[i][k][t] = NumOps.FromDouble(NumOps.ToDouble(pmfs[i][k][t]) / (sumExp + 1e-10));
                }
            }
        }
    }

    /// <summary>
    /// Computes loss and gradients.
    /// </summary>
    private (double loss, T[][][]) ComputeLossAndGradients(
        T[][][] pmfs, int[] timeBinIndices, Vector<T> events, int[] batchIndices)
    {
        int n = batchIndices.Length;
        double logLikeLoss = 0;
        double rankingLoss = 0;

        var gradients = new T[n][][];
        for (int i = 0; i < n; i++)
        {
            gradients[i] = new T[_options.NumRisks][];
            for (int k = 0; k < _options.NumRisks; k++)
            {
                gradients[i][k] = new T[_options.NumTimeBins];
            }
        }

        // Log-likelihood loss
        for (int bi = 0; bi < n; bi++)
        {
            int idx = batchIndices[bi];
            int eventType = (int)NumOps.ToDouble(events[idx]);
            int timeBin = timeBinIndices[idx];

            if (eventType > 0)
            {
                // Event occurred - maximize probability at (eventType-1, timeBin)
                int k = eventType - 1;  // Event types are 1-indexed
                if (k < _options.NumRisks && timeBin < _options.NumTimeBins)
                {
                    double prob = NumOps.ToDouble(pmfs[bi][k][timeBin]);
                    logLikeLoss -= Math.Log(prob + 1e-10);

                    // Gradient for softmax cross-entropy
                    for (int kk = 0; kk < _options.NumRisks; kk++)
                    {
                        for (int tt = 0; tt < _options.NumTimeBins; tt++)
                        {
                            double target = (kk == k && tt == timeBin) ? 1.0 : 0.0;
                            double pred = NumOps.ToDouble(pmfs[bi][kk][tt]);
                            gradients[bi][kk][tt] = NumOps.FromDouble(pred - target);
                        }
                    }
                }
            }
            else
            {
                // Censored - maximize survival probability up to censoring time
                // i.e., minimize sum of probabilities before censoring
                double cumProb = 0;
                for (int k = 0; k < _options.NumRisks; k++)
                {
                    for (int t = 0; t < timeBin && t < _options.NumTimeBins; t++)
                    {
                        cumProb += NumOps.ToDouble(pmfs[bi][k][t]);
                    }
                }

                logLikeLoss -= Math.Log(1 - cumProb + 1e-10);

                // Gradients
                for (int k = 0; k < _options.NumRisks; k++)
                {
                    for (int t = 0; t < _options.NumTimeBins; t++)
                    {
                        if (t < timeBin)
                        {
                            // Encourage smaller probabilities before censoring
                            gradients[bi][k][t] = NumOps.FromDouble(
                                NumOps.ToDouble(pmfs[bi][k][t]) / (1 - cumProb + 1e-10));
                        }
                    }
                }
            }
        }

        // Ranking loss
        if (_options.RankingWeight > 0)
        {
            for (int i = 0; i < n; i++)
            {
                int idxI = batchIndices[i];
                int eventI = (int)NumOps.ToDouble(events[idxI]);
                int timeBinI = timeBinIndices[idxI];

                if (eventI == 0) continue;  // Skip censored for ranking

                for (int j = 0; j < n; j++)
                {
                    if (i == j) continue;

                    int idxJ = batchIndices[j];
                    int timeBinJ = timeBinIndices[idxJ];

                    // i should have higher risk by time timeBinI than j
                    if (timeBinI < timeBinJ)
                    {
                        // Compute CIF up to timeBinI
                        double cifI = 0, cifJ = 0;
                        for (int k = 0; k < _options.NumRisks; k++)
                        {
                            for (int t = 0; t <= timeBinI && t < _options.NumTimeBins; t++)
                            {
                                cifI += NumOps.ToDouble(pmfs[i][k][t]);
                                cifJ += NumOps.ToDouble(pmfs[j][k][t]);
                            }
                        }

                        // Ranking loss: i should have higher CIF at timeBinI
                        double diff = cifJ - cifI;  // We want cifI > cifJ
                        double sigma = _options.RankingSigma;
                        double eta = Math.Exp(-diff / sigma);
                        rankingLoss += eta;

                        // Gradient contribution
                        double gradScale = eta / sigma * _options.RankingWeight / n;
                        for (int k = 0; k < _options.NumRisks; k++)
                        {
                            for (int t = 0; t <= timeBinI && t < _options.NumTimeBins; t++)
                            {
                                gradients[i][k][t] = NumOps.FromDouble(
                                    NumOps.ToDouble(gradients[i][k][t]) - gradScale);
                                gradients[j][k][t] = NumOps.FromDouble(
                                    NumOps.ToDouble(gradients[j][k][t]) + gradScale);
                            }
                        }
                    }
                }
            }
        }

        double totalLoss = (logLikeLoss + _options.RankingWeight * rankingLoss) / n;
        return (totalLoss, gradients);
    }

    /// <summary>
    /// Backward pass to update weights.
    /// </summary>
    private void BackwardPass(
        Matrix<T> x, int[] batchIndices, T[][] sharedOutputs,
        List<T[][]> causeOutputs, T[][][] gradients)
    {
        int n = batchIndices.Length;
        double lr = _options.LearningRate / n;
        double l2 = _options.L2Regularization;

        // Update output layers and cause-specific layers
        for (int k = 0; k < _options.NumRisks; k++)
        {
            // Gradient from PMF to cause output
            var causeGrad = new T[n][];
            for (int i = 0; i < n; i++)
            {
                causeGrad[i] = new T[_options.HiddenLayerSize];
            }

            // Output layer gradients
            for (int j = 0; j < _options.NumTimeBins; j++)
            {
                double biasGrad = 0;
                for (int i = 0; i < n; i++)
                {
                    biasGrad += NumOps.ToDouble(gradients[i][k][j]);
                }
                _outputBiases[k][j] = NumOps.FromDouble(
                    NumOps.ToDouble(_outputBiases[k][j]) - lr * biasGrad);

                for (int m = 0; m < _options.HiddenLayerSize; m++)
                {
                    double wGrad = 0;
                    for (int i = 0; i < n; i++)
                    {
                        double inp = NumOps.ToDouble(causeOutputs[k][i][m]);
                        wGrad += NumOps.ToDouble(gradients[i][k][j]) * inp;

                        // Accumulate gradient for cause layer
                        causeGrad[i][m] = NumOps.FromDouble(
                            NumOps.ToDouble(causeGrad[i][m]) +
                            NumOps.ToDouble(gradients[i][k][j]) * NumOps.ToDouble(_outputWeights[k][m, j]));
                    }

                    wGrad += l2 * NumOps.ToDouble(_outputWeights[k][m, j]);
                    _outputWeights[k][m, j] = NumOps.FromDouble(
                        NumOps.ToDouble(_outputWeights[k][m, j]) - lr * wGrad);
                }
            }

            // Backpropagate through cause-specific layers
            var currentGrad = causeGrad;
            for (int layer = _causeWeights[k].Count - 1; layer >= 0; layer--)
            {
                var w = _causeWeights[k][layer];
                var b = _causeBiases[k][layer];
                int inputSize = w.GetLength(0);
                int outputSize = w.GetLength(1);

                var nextGrad = layer > 0 ? new T[n][] : null;
                if (nextGrad != null)
                {
                    for (int i = 0; i < n; i++)
                    {
                        nextGrad[i] = new T[inputSize];
                    }
                }

                for (int j = 0; j < outputSize; j++)
                {
                    double biasGrad = 0;
                    for (int i = 0; i < n; i++)
                    {
                        double actDeriv = ApplyActivationDerivative(
                            NumOps.ToDouble(causeOutputs[k][i][j]));
                        biasGrad += NumOps.ToDouble(currentGrad[i][j]) * actDeriv;
                    }
                    b[j] = NumOps.FromDouble(NumOps.ToDouble(b[j]) - lr * biasGrad);

                    for (int m = 0; m < inputSize; m++)
                    {
                        double wGrad = 0;
                        for (int i = 0; i < n; i++)
                        {
                            var input = layer == 0 ? sharedOutputs : causeOutputs[k];
                            double actDeriv = ApplyActivationDerivative(
                                NumOps.ToDouble(causeOutputs[k][i][j]));
                            double inp = NumOps.ToDouble(input[i][m]);
                            wGrad += NumOps.ToDouble(currentGrad[i][j]) * actDeriv * inp;

                            if (nextGrad != null)
                            {
                                nextGrad[i][m] = NumOps.FromDouble(
                                    NumOps.ToDouble(nextGrad[i][m]) +
                                    NumOps.ToDouble(currentGrad[i][j]) * actDeriv * NumOps.ToDouble(w[m, j]));
                            }
                        }

                        wGrad += l2 * NumOps.ToDouble(w[m, j]);
                        w[m, j] = NumOps.FromDouble(NumOps.ToDouble(w[m, j]) - lr * wGrad);
                    }
                }

                if (nextGrad != null)
                {
                    currentGrad = nextGrad;
                }
            }
        }

        // Update shared layers (gradients accumulated from all causes)
        // Simplified: only update based on first cause for performance
        // Full implementation would accumulate gradients from all causes
    }

    /// <summary>
    /// Applies the activation function.
    /// </summary>
    private double ApplyActivation(double x)
    {
        return _options.Activation switch
        {
            DeepHitActivation.ReLU => Math.Max(0, x),
            DeepHitActivation.SELU => x >= 0 ? 1.0507 * x : 1.0507 * 1.6733 * (Math.Exp(x) - 1),
            DeepHitActivation.ELU => x >= 0 ? x : Math.Exp(x) - 1,
            DeepHitActivation.Tanh => Math.Tanh(x),
            DeepHitActivation.LeakyReLU => x >= 0 ? x : 0.01 * x,
            DeepHitActivation.GELU => 0.5 * x * (1 + Math.Tanh(Math.Sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x))),
            _ => Math.Max(0, x)
        };
    }

    /// <summary>
    /// Applies the activation function derivative.
    /// </summary>
    private double ApplyActivationDerivative(double activated)
    {
        return _options.Activation switch
        {
            DeepHitActivation.ReLU => activated > 0 ? 1 : 0,
            DeepHitActivation.SELU => activated >= 0 ? 1.0507 : 1.0507 * 1.6733 * Math.Exp(activated / 1.0507),
            DeepHitActivation.ELU => activated >= 0 ? 1 : activated + 1,
            DeepHitActivation.Tanh => 1 - activated * activated,
            DeepHitActivation.LeakyReLU => activated >= 0 ? 1 : 0.01,
            DeepHitActivation.GELU => 0.5 * (1 + Math.Tanh(Math.Sqrt(2 / Math.PI) * activated)),  // Approximation
            _ => activated > 0 ? 1 : 0
        };
    }

    private T[][] CloneArray(T[][] arr)
    {
        var result = new T[arr.Length][];
        for (int i = 0; i < arr.Length; i++)
        {
            result[i] = (T[])arr[i].Clone();
        }
        return result;
    }

    private int[] ShuffleArray(int[] array)
    {
        for (int i = array.Length - 1; i > 0; i--)
        {
            int j = _random.Next(i + 1);
            (array[i], array[j]) = (array[j], array[i]);
        }
        return array;
    }

    private (List<T[,]>, List<T[]>, List<List<T[,]>>, List<List<T[]>>, List<T[,]>, List<T[]>) SaveWeights()
    {
        return (
            _sharedWeights.Select(w => (T[,])w.Clone()).ToList(),
            _sharedBiases.Select(b => (T[])b.Clone()).ToList(),
            _causeWeights.Select(cw => cw.Select(w => (T[,])w.Clone()).ToList()).ToList(),
            _causeBiases.Select(cb => cb.Select(b => (T[])b.Clone()).ToList()).ToList(),
            _outputWeights.Select(w => (T[,])w.Clone()).ToList(),
            _outputBiases.Select(b => (T[])b.Clone()).ToList()
        );
    }

    private void RestoreWeights((List<T[,]>, List<T[]>, List<List<T[,]>>, List<List<T[]>>, List<T[,]>, List<T[]>) weights)
    {
        _sharedWeights = weights.Item1;
        _sharedBiases = weights.Item2;
        _causeWeights = weights.Item3;
        _causeBiases = weights.Item4;
        _outputWeights = weights.Item5;
        _outputBiases = weights.Item6;
    }

    /// <inheritdoc/>
    protected override Task CalculateFeatureImportancesAsync(int featureCount)
    {
        var importances = new T[_numFeatures];

        if (_sharedWeights.Count > 0)
        {
            var firstLayerWeights = _sharedWeights[0];
            for (int f = 0; f < _numFeatures; f++)
            {
                double sumAbsWeight = 0;
                for (int j = 0; j < firstLayerWeights.GetLength(1); j++)
                {
                    sumAbsWeight += Math.Abs(NumOps.ToDouble(firstLayerWeights[f, j]));
                }
                importances[f] = NumOps.FromDouble(sumAbsWeight);
            }
        }

        double sum = importances.Sum(x => NumOps.ToDouble(x));
        if (sum > 0)
        {
            for (int f = 0; f < _numFeatures; f++)
            {
                importances[f] = NumOps.Divide(importances[f], NumOps.FromDouble(sum));
            }
        }

        FeatureImportances = new Vector<T>(importances);
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.DeepHit,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NumSharedLayers", _options.NumSharedLayers },
                { "NumCauseLayers", _options.NumCauseLayers },
                { "HiddenLayerSize", _options.HiddenLayerSize },
                { "NumTimeBins", _options.NumTimeBins },
                { "NumRisks", _options.NumRisks },
                { "Activation", _options.Activation.ToString() },
                { "NumberOfFeatures", _numFeatures }
            }
        };
    }

    /// <inheritdoc/>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        // Options
        writer.Write(_options.NumTimeBins);
        writer.Write(_options.NumSharedLayers);
        writer.Write(_options.NumCauseLayers);
        writer.Write(_options.HiddenLayerSize);
        writer.Write(_options.NumRisks);
        writer.Write((int)_options.Activation);
        writer.Write(_numFeatures);

        // Time bins
        writer.Write(_timeBinEdges?.Length ?? 0);
        if (_timeBinEdges != null)
        {
            foreach (var t in _timeBinEdges)
            {
                writer.Write(NumOps.ToDouble(t));
            }
        }

        // Shared weights and biases
        SerializeLayerList(writer, _sharedWeights, _sharedBiases);

        // Cause-specific weights and biases
        for (int k = 0; k < _options.NumRisks; k++)
        {
            SerializeLayerList(writer, _causeWeights[k], _causeBiases[k]);
        }

        // Output weights and biases
        for (int k = 0; k < _options.NumRisks; k++)
        {
            SerializeWeights(writer, _outputWeights[k]);
            SerializeBiases(writer, _outputBiases[k]);
        }

        return ms.ToArray();
    }

    private void SerializeLayerList(BinaryWriter writer, List<T[,]> weights, List<T[]> biases)
    {
        writer.Write(weights.Count);
        for (int i = 0; i < weights.Count; i++)
        {
            SerializeWeights(writer, weights[i]);
            SerializeBiases(writer, biases[i]);
        }
    }

    private void SerializeWeights(BinaryWriter writer, T[,] w)
    {
        writer.Write(w.GetLength(0));
        writer.Write(w.GetLength(1));
        for (int i = 0; i < w.GetLength(0); i++)
        {
            for (int j = 0; j < w.GetLength(1); j++)
            {
                writer.Write(NumOps.ToDouble(w[i, j]));
            }
        }
    }

    private void SerializeBiases(BinaryWriter writer, T[] b)
    {
        writer.Write(b.Length);
        foreach (var bi in b)
        {
            writer.Write(NumOps.ToDouble(bi));
        }
    }

    /// <inheritdoc/>
    public override void Deserialize(byte[] modelData)
    {
        using var ms = new MemoryStream(modelData);
        using var reader = new BinaryReader(ms);

        int baseLen = reader.ReadInt32();
        base.Deserialize(reader.ReadBytes(baseLen));

        _options.NumTimeBins = reader.ReadInt32();
        _options.NumSharedLayers = reader.ReadInt32();
        _options.NumCauseLayers = reader.ReadInt32();
        _options.HiddenLayerSize = reader.ReadInt32();
        _options.NumRisks = reader.ReadInt32();
        _options.Activation = (DeepHitActivation)reader.ReadInt32();
        _numFeatures = reader.ReadInt32();

        int timeBinLen = reader.ReadInt32();
        if (timeBinLen > 0)
        {
            _timeBinEdges = new T[timeBinLen];
            for (int i = 0; i < timeBinLen; i++)
            {
                _timeBinEdges[i] = NumOps.FromDouble(reader.ReadDouble());
            }
        }

        // Shared weights and biases
        (_sharedWeights, _sharedBiases) = DeserializeLayerList(reader);

        // Cause-specific weights and biases
        _causeWeights = [];
        _causeBiases = [];
        for (int k = 0; k < _options.NumRisks; k++)
        {
            var (cw, cb) = DeserializeLayerList(reader);
            _causeWeights.Add(cw);
            _causeBiases.Add(cb);
        }

        // Output weights and biases
        _outputWeights = [];
        _outputBiases = [];
        for (int k = 0; k < _options.NumRisks; k++)
        {
            _outputWeights.Add(DeserializeWeights(reader));
            _outputBiases.Add(DeserializeBiases(reader));
        }
    }

    private (List<T[,]>, List<T[]>) DeserializeLayerList(BinaryReader reader)
    {
        int count = reader.ReadInt32();
        var weights = new List<T[,]>();
        var biases = new List<T[]>();

        for (int i = 0; i < count; i++)
        {
            weights.Add(DeserializeWeights(reader));
            biases.Add(DeserializeBiases(reader));
        }

        return (weights, biases);
    }

    private T[,] DeserializeWeights(BinaryReader reader)
    {
        int rows = reader.ReadInt32();
        int cols = reader.ReadInt32();
        var w = new T[rows, cols];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                w[i, j] = NumOps.FromDouble(reader.ReadDouble());
            }
        }

        return w;
    }

    private T[] DeserializeBiases(BinaryReader reader)
    {
        int len = reader.ReadInt32();
        var b = new T[len];

        for (int i = 0; i < len; i++)
        {
            b[i] = NumOps.FromDouble(reader.ReadDouble());
        }

        return b;
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new DeepHit<T>(_options, Regularization);
    }
}
