using AiDotNet.Helpers;
using AiDotNet.Models.Options;

namespace AiDotNet.Regression;

/// <summary>
/// DeepSurv: A deep learning approach to survival analysis using Cox proportional hazards.
/// </summary>
/// <remarks>
/// <para>
/// DeepSurv extends the classical Cox Proportional Hazards model by using a deep neural
/// network to model the log-risk function. It optimizes the negative partial log-likelihood
/// of the Cox model while learning complex non-linear relationships.
/// </para>
/// <para>
/// <b>For Beginners:</b> Survival analysis predicts "time until an event occurs." DeepSurv
/// is a neural network that learns to predict risk scores from your features:
///
/// - Higher risk score = event is likely to happen sooner
/// - Lower risk score = event is likely to happen later
///
/// What makes survival analysis unique is "censoring": some subjects haven't experienced
/// the event yet when the study ends. DeepSurv properly handles this by using the Cox
/// partial likelihood, which only compares subjects who are "at risk" at each event time.
///
/// Example applications:
/// - Medical: Predict patient survival time based on clinical features
/// - Business: Predict customer churn time based on usage patterns
/// - Engineering: Predict equipment failure time based on sensor data
///
/// Key outputs:
/// - Risk scores: Relative risk for each subject
/// - Survival curves: Probability of surviving past time t
/// - Hazard ratios: How much each feature affects risk
/// </para>
/// <para>
/// Reference: Katzman, J.L. et al. (2018). "DeepSurv: Personalized Treatment Recommender
/// System Using A Cox Proportional Hazards Deep Neural Network". BMC Medical Research Methodology.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class DeepSurv<T> : AsyncDecisionTreeRegressionBase<T>
{
    /// <summary>
    /// Network weights for each layer.
    /// </summary>
    private List<Matrix<T>> _weights;

    /// <summary>
    /// Network biases for each layer.
    /// </summary>
    private List<Vector<T>> _biases;

    /// <summary>
    /// Number of features.
    /// </summary>
    private int _numFeatures;

    /// <summary>
    /// Baseline cumulative hazard function times.
    /// </summary>
    private Vector<T>? _baselineHazardTimes;

    /// <summary>
    /// Baseline cumulative hazard function values.
    /// </summary>
    private Vector<T>? _baselineHazardValues;

    /// <summary>
    /// Configuration options.
    /// </summary>
    private readonly DeepSurvOptions _options;

    /// <summary>
    /// Random number generator.
    /// </summary>
    private readonly Random _random;

    /// <inheritdoc/>
    public override int NumberOfTrees => 1;

    /// <summary>
    /// Initializes a new instance of DeepSurv.
    /// </summary>
    /// <param name="options">Configuration options.</param>
    /// <param name="regularization">Optional regularization.</param>
    public DeepSurv(DeepSurvOptions? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(null, regularization)
    {
        _options = options ?? new DeepSurvOptions();
        _weights = [];
        _biases = [];
        _numFeatures = 0;
        _random = _options.Seed.HasValue ? RandomHelper.CreateSeededRandom(_options.Seed.Value) : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Trains the DeepSurv model.
    /// </summary>
    /// <param name="x">Feature matrix.</param>
    /// <param name="times">Observed times (time to event or censoring).</param>
    /// <param name="events">Event indicators (1 = event occurred, 0 = censored).</param>
    public async Task TrainAsync(Matrix<T> x, Vector<T> times, Vector<T> events)
    {
        _numFeatures = x.Columns;
        int n = x.Rows;

        // Initialize network
        InitializeNetwork();

        // Sort by time (required for Cox partial likelihood)
        var sortedIndices = GetSortedIndices(times);

        double bestLoss = double.MaxValue;
        int patienceCounter = 0;

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
                var (riskScores, hiddenOutputs) = ForwardPass(x, batchIndices);

                // Compute Cox loss and gradients
                var (loss, gradients) = ComputeCoxLossAndGradients(
                    riskScores, times, events, batchIndices, sortedIndices);

                epochLoss += loss;
                numBatches++;

                // Backward pass and update
                BackwardPass(x, batchIndices, hiddenOutputs, gradients);
            }

            epochLoss /= numBatches;

            // Early stopping
            if (_options.EarlyStoppingPatience.HasValue)
            {
                if (epochLoss < bestLoss - 1e-6)
                {
                    bestLoss = epochLoss;
                    patienceCounter = 0;
                }
                else
                {
                    patienceCounter++;
                    if (patienceCounter >= _options.EarlyStoppingPatience.Value)
                    {
                        break;
                    }
                }
            }
        }

        // Compute baseline hazard
        ComputeBaselineHazard(x, times, events, sortedIndices);

        await CalculateFeatureImportancesAsync(x.Columns);
    }

    /// <inheritdoc/>
    public override async Task TrainAsync(Matrix<T> x, Vector<T> y)
    {
        // For standard interface, assume all events occurred (no censoring)
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
        // Return risk scores
        return await Task.Run(() => PredictRiskScores(input));
    }

    /// <summary>
    /// Predicts risk scores for input samples.
    /// </summary>
    /// <param name="input">Input feature matrix.</param>
    /// <returns>Vector of risk scores (higher = higher risk).</returns>
    public Vector<T> PredictRiskScores(Matrix<T> input)
    {
        var indices = Enumerable.Range(0, input.Rows).ToArray();
        var (riskScores, _) = ForwardPass(input, indices);

        var result = new Vector<T>(input.Rows);
        for (int i = 0; i < input.Rows; i++)
        {
            result[i] = riskScores[i];
        }

        return result;
    }

    /// <summary>
    /// Predicts survival probability at specified times.
    /// </summary>
    /// <param name="input">Input feature matrix.</param>
    /// <param name="times">Times at which to evaluate survival probability.</param>
    /// <returns>Matrix where [i,j] is P(T > times[j] | X_i).</returns>
    public Matrix<T> PredictSurvival(Matrix<T> input, Vector<T> times)
    {
        var riskScores = PredictRiskScores(input);
        var survivalProbs = new Matrix<T>(input.Rows, times.Length);

        if (_baselineHazardTimes == null || _baselineHazardValues == null)
        {
            // If baseline hazard not computed, use exponential model
            for (int i = 0; i < input.Rows; i++)
            {
                double risk = NumOps.ToDouble(riskScores[i]);
                for (int j = 0; j < times.Length; j++)
                {
                    double t = NumOps.ToDouble(times[j]);
                    survivalProbs[i, j] = NumOps.FromDouble(Math.Exp(-Math.Exp(risk) * t));
                }
            }
        }
        else
        {
            // Use baseline cumulative hazard
            for (int i = 0; i < input.Rows; i++)
            {
                double risk = Math.Exp(NumOps.ToDouble(riskScores[i]));
                for (int j = 0; j < times.Length; j++)
                {
                    double t = NumOps.ToDouble(times[j]);
                    double H0 = InterpolateBaselineHazard(t);
                    survivalProbs[i, j] = NumOps.FromDouble(Math.Exp(-H0 * risk));
                }
            }
        }

        return survivalProbs;
    }

    /// <summary>
    /// Predicts median survival time for each sample.
    /// </summary>
    /// <param name="input">Input feature matrix.</param>
    /// <returns>Vector of median survival times.</returns>
    public Vector<T> PredictMedianSurvivalTime(Matrix<T> input)
    {
        var riskScores = PredictRiskScores(input);
        var medianTimes = new Vector<T>(input.Rows);

        for (int i = 0; i < input.Rows; i++)
        {
            double risk = Math.Exp(NumOps.ToDouble(riskScores[i]));

            // Find time where S(t) = 0.5
            if (_baselineHazardTimes != null && _baselineHazardValues != null && _baselineHazardTimes.Length > 0)
            {
                double targetH0 = -Math.Log(0.5) / risk;  // H0 such that S = exp(-H0 * risk) = 0.5

                // Binary search for time
                double medianTime = FindTimeForHazard(targetH0);
                medianTimes[i] = NumOps.FromDouble(medianTime);
            }
            else
            {
                // Exponential model: median = ln(2) / (exp(risk))
                medianTimes[i] = NumOps.FromDouble(Math.Log(2) / risk);
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
        var riskScores = PredictRiskScores(x);
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
                    double ri = NumOps.ToDouble(riskScores[i]);
                    double rj = NumOps.ToDouble(riskScores[j]);

                    if (ri > rj) concordant++;
                    else if (ri < rj) discordant++;
                }
            }
        }

        int total = concordant + discordant;
        return total > 0 ? (double)concordant / total : 0.5;
    }

    /// <summary>
    /// Initializes the neural network weights.
    /// </summary>
    private void InitializeNetwork()
    {
        _weights = [];
        _biases = [];

        int inputSize = _numFeatures;

        for (int layer = 0; layer < _options.NumHiddenLayers; layer++)
        {
            int outputSize = _options.HiddenLayerSize;

            // Xavier/He initialization
            double scale = Math.Sqrt(2.0 / inputSize);
            var w = new Matrix<T>(inputSize, outputSize);
            var b = new Vector<T>(outputSize);

            for (int i = 0; i < inputSize; i++)
            {
                for (int j = 0; j < outputSize; j++)
                {
                    w[i, j] = NumOps.FromDouble((_random.NextDouble() * 2 - 1) * scale);
                }
            }

            _weights.Add(w);
            _biases.Add(b);

            inputSize = outputSize;
        }

        // Output layer (single risk score)
        double outputScale = Math.Sqrt(2.0 / inputSize);
        var wOutput = new Matrix<T>(inputSize, 1);
        var bOutput = new Vector<T>(1);

        for (int i = 0; i < inputSize; i++)
        {
            wOutput[i, 0] = NumOps.FromDouble((_random.NextDouble() * 2 - 1) * outputScale);
        }

        _weights.Add(wOutput);
        _biases.Add(bOutput);
    }

    /// <summary>
    /// Forward pass through the network.
    /// </summary>
    private (Vector<T>, List<Vector<T>[]>) ForwardPass(Matrix<T> x, int[] indices)
    {
        int n = indices.Length;
        var hiddenOutputs = new List<Vector<T>[]>();

        // Current layer input
        var current = new Vector<T>[n];
        for (int i = 0; i < n; i++)
        {
            current[i] = new Vector<T>(_numFeatures);
            for (int j = 0; j < _numFeatures; j++)
            {
                current[i][j] = x[indices[i], j];
            }
        }

        // Hidden layers
        for (int layer = 0; layer < _weights.Count - 1; layer++)
        {
            var w = _weights[layer];
            var b = _biases[layer];
            int outputSize = w.Columns;

            var next = new Vector<T>[n];
            for (int i = 0; i < n; i++)
            {
                next[i] = new Vector<T>(outputSize);
                for (int j = 0; j < outputSize; j++)
                {
                    double sum = NumOps.ToDouble(b[j]);
                    for (int k = 0; k < current[i].Length; k++)
                    {
                        sum += NumOps.ToDouble(current[i][k]) * NumOps.ToDouble(w[k, j]);
                    }

                    // Apply activation
                    next[i][j] = NumOps.FromDouble(ApplyActivation(sum));
                }
            }

            hiddenOutputs.Add(current);
            current = next;
        }

        // Output layer (no activation - linear risk score)
        var wOut = _weights[^1];
        var bOut = _biases[^1];
        var riskScores = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            double sum = NumOps.ToDouble(bOut[0]);
            for (int k = 0; k < current[i].Length; k++)
            {
                sum += NumOps.ToDouble(current[i][k]) * NumOps.ToDouble(wOut[k, 0]);
            }
            riskScores[i] = NumOps.FromDouble(sum);
        }

        hiddenOutputs.Add(current);
        return (riskScores, hiddenOutputs);
    }

    /// <summary>
    /// Computes Cox partial log-likelihood loss and gradients.
    /// </summary>
    private (double loss, Vector<T> gradients) ComputeCoxLossAndGradients(
        Vector<T> riskScores, Vector<T> times, Vector<T> events,
        int[] batchIndices, int[] sortedIndices)
    {
        int n = batchIndices.Length;
        var gradients = new Vector<T>(n);

        // Create mapping from global to batch index
        var batchSet = new HashSet<int>(batchIndices);
        var batchIndexMap = new Dictionary<int, int>();
        for (int i = 0; i < batchIndices.Length; i++)
        {
            batchIndexMap[batchIndices[i]] = i;
        }

        double loss = 0;

        // Process events in time order
        double riskSum = 0;
        for (int i = sortedIndices.Length - 1; i >= 0; i--)
        {
            int idx = sortedIndices[i];
            if (!batchSet.Contains(idx)) continue;

            int batchIdx = batchIndexMap[idx];
            double ri = NumOps.ToDouble(riskScores[batchIdx]);
            double expRi = Math.Exp(ri);

            riskSum += expRi;

            if (NumOps.ToDouble(events[idx]) == 1)
            {
                // Event occurred
                loss -= ri - Math.Log(riskSum + 1e-10);

                // Gradient: event contribution
                gradients[batchIdx] = NumOps.FromDouble(
                    NumOps.ToDouble(gradients[batchIdx]) - 1 + expRi / (riskSum + 1e-10));
            }
            else
            {
                // Censored - only contributes to risk set
                gradients[batchIdx] = NumOps.FromDouble(
                    NumOps.ToDouble(gradients[batchIdx]) + expRi / (riskSum + 1e-10));
            }
        }

        return (loss / n, gradients);
    }

    /// <summary>
    /// Backward pass to update weights.
    /// </summary>
    private void BackwardPass(Matrix<T> x, int[] batchIndices, List<Vector<T>[]> hiddenOutputs, Vector<T> gradients)
    {
        int n = batchIndices.Length;
        double lr = _options.LearningRate / n;
        double l2 = _options.L2Regularization;

        // Start from output layer
        var currentGrad = new Vector<T>[n];
        for (int i = 0; i < n; i++)
        {
            currentGrad[i] = new Vector<T>(new[] { gradients[i] });
        }

        // Backpropagate through layers
        for (int layer = _weights.Count - 1; layer >= 0; layer--)
        {
            var w = _weights[layer];
            var b = _biases[layer];
            var input = layer > 0 ? hiddenOutputs[layer - 1] : null;

            int inputSize = w.Rows;
            int outputSize = w.Columns;

            // Gradient w.r.t. weights and biases
            for (int j = 0; j < outputSize; j++)
            {
                double biasGrad = 0;
                for (int i = 0; i < n; i++)
                {
                    biasGrad += NumOps.ToDouble(currentGrad[i][j]);
                }
                b[j] = NumOps.FromDouble(NumOps.ToDouble(b[j]) - lr * biasGrad);

                for (int k = 0; k < inputSize; k++)
                {
                    double wGrad = 0;
                    for (int i = 0; i < n; i++)
                    {
                        double inp;
                        if (layer == 0)
                        {
                            inp = NumOps.ToDouble(x[batchIndices[i], k]);
                        }
                        else
                        {
                            inp = NumOps.ToDouble(hiddenOutputs[layer - 1][i][k]);
                        }
                        wGrad += NumOps.ToDouble(currentGrad[i][j]) * inp;
                    }

                    // L2 regularization
                    wGrad += l2 * NumOps.ToDouble(w[k, j]);

                    w[k, j] = NumOps.FromDouble(NumOps.ToDouble(w[k, j]) - lr * wGrad);
                }
            }

            // Gradient w.r.t. input (for previous layer)
            if (layer > 0)
            {
                var nextGrad = new Vector<T>[n];
                for (int i = 0; i < n; i++)
                {
                    nextGrad[i] = new Vector<T>(inputSize);
                    for (int k = 0; k < inputSize; k++)
                    {
                        double sum = 0;
                        for (int j = 0; j < outputSize; j++)
                        {
                            sum += NumOps.ToDouble(currentGrad[i][j]) * NumOps.ToDouble(w[k, j]);
                        }

                        // Activation derivative
                        double actDeriv = ApplyActivationDerivative(NumOps.ToDouble(hiddenOutputs[layer - 1][i][k]));
                        nextGrad[i][k] = NumOps.FromDouble(sum * actDeriv);
                    }
                }
                currentGrad = nextGrad;
            }
        }
    }

    /// <summary>
    /// Computes baseline cumulative hazard using Breslow estimator.
    /// </summary>
    private void ComputeBaselineHazard(Matrix<T> x, Vector<T> times, Vector<T> events, int[] sortedIndices)
    {
        var riskScores = PredictRiskScores(x);

        var uniqueTimes = new List<double>();
        var hazardValues = new List<double>();

        double cumulativeHazard = 0;
        double riskSum = 0;

        // Compute sum of exp(risk) for all at risk at end
        for (int i = 0; i < x.Rows; i++)
        {
            riskSum += Math.Exp(NumOps.ToDouble(riskScores[i]));
        }

        double lastTime = double.MinValue;

        for (int i = 0; i < sortedIndices.Length; i++)
        {
            int idx = sortedIndices[i];
            double t = NumOps.ToDouble(times[idx]);
            double e = NumOps.ToDouble(events[idx]);

            if (e == 1 && t > lastTime)
            {
                // Add to baseline hazard
                cumulativeHazard += 1.0 / (riskSum + 1e-10);
                uniqueTimes.Add(t);
                hazardValues.Add(cumulativeHazard);
                lastTime = t;
            }

            // Remove from risk set
            riskSum -= Math.Exp(NumOps.ToDouble(riskScores[idx]));
        }

        var timesArr = uniqueTimes.Select(t => NumOps.FromDouble(t)).ToArray();
        _baselineHazardTimes = new Vector<T>(timesArr);
        var hazardArr = hazardValues.Select(h => NumOps.FromDouble(h)).ToArray();
        _baselineHazardValues = new Vector<T>(hazardArr);
    }

    /// <summary>
    /// Interpolates baseline hazard at a given time.
    /// </summary>
    private double InterpolateBaselineHazard(double t)
    {
        if (_baselineHazardTimes == null || _baselineHazardTimes.Length == 0)
            return 0;

        for (int i = 0; i < _baselineHazardTimes.Length; i++)
        {
            if (t <= NumOps.ToDouble(_baselineHazardTimes[i]))
            {
                return i > 0 ? NumOps.ToDouble(_baselineHazardValues![i - 1]) : 0;
            }
        }

        return NumOps.ToDouble(_baselineHazardValues![^1]);
    }

    /// <summary>
    /// Finds time for a given cumulative hazard value.
    /// </summary>
    private double FindTimeForHazard(double targetH0)
    {
        if (_baselineHazardTimes == null || _baselineHazardTimes.Length == 0)
            return double.PositiveInfinity;

        for (int i = 0; i < _baselineHazardValues!.Length; i++)
        {
            if (NumOps.ToDouble(_baselineHazardValues[i]) >= targetH0)
            {
                return NumOps.ToDouble(_baselineHazardTimes[i]);
            }
        }

        return double.PositiveInfinity;
    }

    /// <summary>
    /// Applies the activation function.
    /// </summary>
    private double ApplyActivation(double x)
    {
        return _options.Activation switch
        {
            DeepSurvActivation.ReLU => Math.Max(0, x),
            DeepSurvActivation.SELU => x >= 0 ? 1.0507 * x : 1.0507 * 1.6733 * (Math.Exp(x) - 1),
            DeepSurvActivation.ELU => x >= 0 ? x : Math.Exp(x) - 1,
            DeepSurvActivation.Tanh => Math.Tanh(x),
            DeepSurvActivation.LeakyReLU => x >= 0 ? x : 0.01 * x,
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
            DeepSurvActivation.ReLU => activated > 0 ? 1 : 0,
            DeepSurvActivation.SELU => activated >= 0 ? 1.0507 : 1.0507 * 1.6733 * Math.Exp(activated / 1.0507),
            DeepSurvActivation.ELU => activated >= 0 ? 1 : activated + 1,
            DeepSurvActivation.Tanh => 1 - activated * activated,
            DeepSurvActivation.LeakyReLU => activated >= 0 ? 1 : 0.01,
            _ => activated > 0 ? 1 : 0
        };
    }

    private int[] GetSortedIndices(Vector<T> times)
    {
        return Enumerable.Range(0, times.Length)
            .OrderBy(i => NumOps.ToDouble(times[i]))
            .ToArray();
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

    /// <inheritdoc/>
    protected override Task CalculateFeatureImportancesAsync(int featureCount)
    {
        // Use first layer weights as importance proxy
        var importances = new Vector<T>(_numFeatures);

        if (_weights.Count > 0)
        {
            var firstLayerWeights = _weights[0];
            for (int f = 0; f < _numFeatures; f++)
            {
                double sumAbsWeight = 0;
                for (int j = 0; j < firstLayerWeights.Columns; j++)
                {
                    sumAbsWeight += Math.Abs(NumOps.ToDouble(firstLayerWeights[f, j]));
                }
                importances[f] = NumOps.FromDouble(sumAbsWeight);
            }
        }

        double sum = 0;
        for (int f = 0; f < _numFeatures; f++)
        {
            sum += NumOps.ToDouble(importances[f]);
        }
        if (sum > 0)
        {
            for (int f = 0; f < _numFeatures; f++)
            {
                importances[f] = NumOps.Divide(importances[f], NumOps.FromDouble(sum));
            }
        }

        FeatureImportances = importances;
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.DeepSurv,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NumHiddenLayers", _options.NumHiddenLayers },
                { "HiddenLayerSize", _options.HiddenLayerSize },
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
        writer.Write(_options.NumHiddenLayers);
        writer.Write(_options.HiddenLayerSize);
        writer.Write((int)_options.Activation);
        writer.Write(_numFeatures);

        // Weights and biases
        writer.Write(_weights.Count);
        foreach (var w in _weights)
        {
            writer.Write(w.Rows);
            writer.Write(w.Columns);
            for (int i = 0; i < w.Rows; i++)
            {
                for (int j = 0; j < w.Columns; j++)
                {
                    writer.Write(NumOps.ToDouble(w[i, j]));
                }
            }
        }

        foreach (var b in _biases)
        {
            writer.Write(b.Length);
            for (int i = 0; i < b.Length; i++)
            {
                writer.Write(NumOps.ToDouble(b[i]));
            }
        }

        // Baseline hazard
        writer.Write(_baselineHazardTimes != null);
        if (_baselineHazardTimes != null)
        {
            writer.Write(_baselineHazardTimes.Length);
            foreach (var t in _baselineHazardTimes)
            {
                writer.Write(NumOps.ToDouble(t));
            }
            foreach (var h in _baselineHazardValues!)
            {
                writer.Write(NumOps.ToDouble(h));
            }
        }

        return ms.ToArray();
    }

    /// <inheritdoc/>
    public override void Deserialize(byte[] modelData)
    {
        using var ms = new MemoryStream(modelData);
        using var reader = new BinaryReader(ms);

        int baseLen = reader.ReadInt32();
        base.Deserialize(reader.ReadBytes(baseLen));

        _options.NumHiddenLayers = reader.ReadInt32();
        _options.HiddenLayerSize = reader.ReadInt32();
        _options.Activation = (DeepSurvActivation)reader.ReadInt32();
        _numFeatures = reader.ReadInt32();

        int numLayers = reader.ReadInt32();
        _weights = [];
        _biases = [];

        for (int l = 0; l < numLayers; l++)
        {
            int rows = reader.ReadInt32();
            int cols = reader.ReadInt32();
            var w = new Matrix<T>(rows, cols);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    w[i, j] = NumOps.FromDouble(reader.ReadDouble());
                }
            }
            _weights.Add(w);
        }

        for (int l = 0; l < numLayers; l++)
        {
            int len = reader.ReadInt32();
            var b = new Vector<T>(len);
            for (int i = 0; i < len; i++)
            {
                b[i] = NumOps.FromDouble(reader.ReadDouble());
            }
            _biases.Add(b);
        }

        bool hasBaseline = reader.ReadBoolean();
        if (hasBaseline)
        {
            int len = reader.ReadInt32();
            _baselineHazardTimes = new Vector<T>(len);
            _baselineHazardValues = new Vector<T>(len);
            for (int i = 0; i < len; i++)
            {
                _baselineHazardTimes[i] = NumOps.FromDouble(reader.ReadDouble());
            }
            for (int i = 0; i < len; i++)
            {
                _baselineHazardValues[i] = NumOps.FromDouble(reader.ReadDouble());
            }
        }
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new DeepSurv<T>(_options, Regularization);
    }
}
