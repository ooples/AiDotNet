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

        T bestLoss = NumOps.FromDouble(double.MaxValue);
        int patienceCounter = 0;
        T earlyStopThreshold = NumOps.FromDouble(1e-6);

        for (int epoch = 0; epoch < _options.Epochs; epoch++)
        {
            // Mini-batch training
            var shuffledIndices = ShuffleArray(Enumerable.Range(0, n).ToArray());
            T epochLoss = NumOps.Zero;
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

                epochLoss = NumOps.Add(epochLoss, loss);
                numBatches++;

                // Backward pass and update
                BackwardPass(x, batchIndices, hiddenOutputs, gradients);
            }

            epochLoss = NumOps.Divide(epochLoss, NumOps.FromDouble(numBatches));

            // Early stopping
            if (_options.EarlyStoppingPatience.HasValue)
            {
                if (NumOps.LessThan(epochLoss, NumOps.Subtract(bestLoss, earlyStopThreshold)))
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
            // If baseline hazard not computed, use exponential model: S(t) = exp(-exp(risk) * t)
            for (int i = 0; i < input.Rows; i++)
            {
                T expRisk = NumOps.Exp(riskScores[i]);
                for (int j = 0; j < times.Length; j++)
                {
                    survivalProbs[i, j] = NumOps.Exp(NumOps.Negate(NumOps.Multiply(expRisk, times[j])));
                }
            }
        }
        else
        {
            // Use baseline cumulative hazard: S(t) = exp(-H0(t) * exp(risk))
            for (int i = 0; i < input.Rows; i++)
            {
                T expRisk = NumOps.Exp(riskScores[i]);
                for (int j = 0; j < times.Length; j++)
                {
                    T h0 = InterpolateBaselineHazard(times[j]);
                    survivalProbs[i, j] = NumOps.Exp(NumOps.Negate(NumOps.Multiply(h0, expRisk)));
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

        T ln2 = NumOps.Log(NumOps.FromDouble(2.0));

        for (int i = 0; i < input.Rows; i++)
        {
            T expRisk = NumOps.Exp(riskScores[i]);

            // Find time where S(t) = 0.5
            if (_baselineHazardTimes != null && _baselineHazardValues != null && _baselineHazardTimes.Length > 0)
            {
                // H0 such that S = exp(-H0 * risk) = 0.5 => H0 = ln(2) / exp(risk)
                T targetH0 = NumOps.Divide(ln2, expRisk);

                // Search for time
                medianTimes[i] = FindTimeForHazard(targetH0);
            }
            else
            {
                // Exponential model: median = ln(2) / exp(risk)
                medianTimes[i] = NumOps.Divide(ln2, expRisk);
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
                    T sum = b[j];
                    for (int k = 0; k < current[i].Length; k++)
                    {
                        sum = NumOps.Add(sum, NumOps.Multiply(current[i][k], w[k, j]));
                    }

                    // Apply activation (stays double - special math functions)
                    next[i][j] = NumOps.FromDouble(ApplyActivation(NumOps.ToDouble(sum)));
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
            T sum = bOut[0];
            for (int k = 0; k < current[i].Length; k++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(current[i][k], wOut[k, 0]));
            }
            riskScores[i] = sum;
        }

        hiddenOutputs.Add(current);
        return (riskScores, hiddenOutputs);
    }

    /// <summary>
    /// Computes Cox partial log-likelihood loss and gradients.
    /// </summary>
    private (T loss, Vector<T> gradients) ComputeCoxLossAndGradients(
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

        T loss = NumOps.Zero;
        T epsilon = NumOps.FromDouble(1e-10);

        // Process events in time order
        T riskSum = NumOps.Zero;
        for (int i = sortedIndices.Length - 1; i >= 0; i--)
        {
            int idx = sortedIndices[i];
            if (!batchSet.Contains(idx)) continue;

            int batchIdx = batchIndexMap[idx];
            T ri = riskScores[batchIdx];
            T expRi = NumOps.Exp(ri);

            riskSum = NumOps.Add(riskSum, expRi);

            T riskSumSafe = NumOps.Add(riskSum, epsilon);
            T expRiOverRiskSum = NumOps.Divide(expRi, riskSumSafe);

            if (NumOps.Compare(events[idx], NumOps.One) == 0)
            {
                // Event occurred: loss -= ri - log(riskSum + eps)
                loss = NumOps.Subtract(loss, NumOps.Subtract(ri, NumOps.Log(riskSumSafe)));

                // Gradient: event contribution: grad - 1 + expRi / (riskSum + eps)
                gradients[batchIdx] = NumOps.Add(
                    NumOps.Subtract(gradients[batchIdx], NumOps.One),
                    expRiOverRiskSum);
            }
            else
            {
                // Censored - only contributes to risk set
                gradients[batchIdx] = NumOps.Add(gradients[batchIdx], expRiOverRiskSum);
            }
        }

        return (NumOps.Divide(loss, NumOps.FromDouble(n)), gradients);
    }

    /// <summary>
    /// Backward pass to update weights.
    /// </summary>
    private void BackwardPass(Matrix<T> x, int[] batchIndices, List<Vector<T>[]> hiddenOutputs, Vector<T> gradients)
    {
        int n = batchIndices.Length;
        T lr = NumOps.Divide(NumOps.FromDouble(_options.LearningRate), NumOps.FromDouble(n));
        T l2 = NumOps.FromDouble(_options.L2Regularization);

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

            int inputSize = w.Rows;
            int outputSize = w.Columns;

            // Gradient w.r.t. weights and biases
            for (int j = 0; j < outputSize; j++)
            {
                T biasGrad = NumOps.Zero;
                for (int i = 0; i < n; i++)
                {
                    biasGrad = NumOps.Add(biasGrad, currentGrad[i][j]);
                }
                b[j] = NumOps.Subtract(b[j], NumOps.Multiply(lr, biasGrad));

                for (int k = 0; k < inputSize; k++)
                {
                    T wGrad = NumOps.Zero;
                    for (int i = 0; i < n; i++)
                    {
                        T inp;
                        if (layer == 0)
                        {
                            inp = x[batchIndices[i], k];
                        }
                        else
                        {
                            inp = hiddenOutputs[layer - 1][i][k];
                        }
                        wGrad = NumOps.Add(wGrad, NumOps.Multiply(currentGrad[i][j], inp));
                    }

                    // L2 regularization
                    wGrad = NumOps.Add(wGrad, NumOps.Multiply(l2, w[k, j]));

                    w[k, j] = NumOps.Subtract(w[k, j], NumOps.Multiply(lr, wGrad));
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
                        T sum = NumOps.Zero;
                        for (int j = 0; j < outputSize; j++)
                        {
                            sum = NumOps.Add(sum, NumOps.Multiply(currentGrad[i][j], w[k, j]));
                        }

                        // Activation derivative (stays double - special math functions)
                        double actDeriv = ApplyActivationDerivative(NumOps.ToDouble(hiddenOutputs[layer - 1][i][k]));
                        nextGrad[i][k] = NumOps.Multiply(sum, NumOps.FromDouble(actDeriv));
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

        var uniqueTimes = new List<T>();
        var hazardValues = new List<T>();

        T cumulativeHazard = NumOps.Zero;
        T riskSum = NumOps.Zero;
        T epsilon = NumOps.FromDouble(1e-10);

        // Compute sum of exp(risk) for all at risk at end
        for (int i = 0; i < x.Rows; i++)
        {
            riskSum = NumOps.Add(riskSum, NumOps.Exp(riskScores[i]));
        }

        T lastTime = NumOps.FromDouble(double.MinValue);

        for (int i = 0; i < sortedIndices.Length; i++)
        {
            int idx = sortedIndices[i];
            T t = times[idx];

            if (NumOps.Compare(events[idx], NumOps.One) == 0 && NumOps.GreaterThan(t, lastTime))
            {
                // Add to baseline hazard
                cumulativeHazard = NumOps.Add(cumulativeHazard,
                    NumOps.Divide(NumOps.One, NumOps.Add(riskSum, epsilon)));
                uniqueTimes.Add(t);
                hazardValues.Add(cumulativeHazard);
                lastTime = t;
            }

            // Remove from risk set
            riskSum = NumOps.Subtract(riskSum, NumOps.Exp(riskScores[idx]));
        }

        _baselineHazardTimes = new Vector<T>(uniqueTimes.ToArray());
        _baselineHazardValues = new Vector<T>(hazardValues.ToArray());
    }

    /// <summary>
    /// Interpolates baseline hazard at a given time.
    /// </summary>
    private T InterpolateBaselineHazard(T t)
    {
        if (_baselineHazardTimes == null || _baselineHazardTimes.Length == 0)
            return NumOps.Zero;

        for (int i = 0; i < _baselineHazardTimes.Length; i++)
        {
            if (!NumOps.GreaterThan(t, _baselineHazardTimes[i]))
            {
                if (_baselineHazardValues is null)
                {
                    throw new InvalidOperationException("Baseline hazard values have not been computed.");
                }
                return i > 0 ? _baselineHazardValues[i - 1] : NumOps.Zero;
            }
        }

        if (_baselineHazardValues is null)
        {
            throw new InvalidOperationException("Baseline hazard values have not been computed.");
        }
        return _baselineHazardValues[^1];
    }

    /// <summary>
    /// Finds time for a given cumulative hazard value.
    /// </summary>
    private T FindTimeForHazard(T targetH0)
    {
        if (_baselineHazardTimes == null || _baselineHazardTimes.Length == 0)
            return NumOps.FromDouble(double.PositiveInfinity);

        if (_baselineHazardValues is null)
        {
            throw new InvalidOperationException("Baseline hazard values have not been computed.");
        }

        for (int i = 0; i < _baselineHazardValues.Length; i++)
        {
            if (!NumOps.LessThan(_baselineHazardValues[i], targetH0))
            {
                return _baselineHazardTimes[i];
            }
        }

        return NumOps.FromDouble(double.PositiveInfinity);
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
                T sumAbsWeight = NumOps.Zero;
                for (int j = 0; j < firstLayerWeights.Columns; j++)
                {
                    sumAbsWeight = NumOps.Add(sumAbsWeight, NumOps.Abs(firstLayerWeights[f, j]));
                }
                importances[f] = sumAbsWeight;
            }
        }

        T sum = NumOps.Zero;
        for (int f = 0; f < _numFeatures; f++)
        {
            sum = NumOps.Add(sum, importances[f]);
        }
        if (NumOps.GreaterThan(sum, NumOps.Zero))
        {
            for (int f = 0; f < _numFeatures; f++)
            {
                importances[f] = NumOps.Divide(importances[f], sum);
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
        writer.Write(_baselineHazardTimes is not null);
        if (_baselineHazardTimes is not null && _baselineHazardValues is not null)
        {
            writer.Write(_baselineHazardTimes.Length);
            foreach (var t in _baselineHazardTimes)
            {
                writer.Write(NumOps.ToDouble(t));
            }
            foreach (var h in _baselineHazardValues)
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
