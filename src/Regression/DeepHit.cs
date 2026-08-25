using AiDotNet.Attributes;
using AiDotNet.Enums;
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
/// <example>
/// <code>
/// // Create a DeepHit model for survival analysis with competing risks
/// var options = new DeepHitOptions&lt;double&gt;();
/// var model = new DeepHit&lt;double&gt;(options);
///
/// // Prepare training data: 6 samples with 3 features (clinical covariates)
/// var features = Matrix&lt;double&gt;.Build.Dense(6, 3, new double[] {
///     55, 1, 2.1,  60, 0, 3.5,  45, 1, 1.8,
///     70, 0, 4.2,  50, 1, 2.9,  65, 0, 3.1 });
/// var targets = new Vector&lt;double&gt;(new double[] { 12, 24, 36, 6, 18, 30 });
///
/// // Train the survival model
/// model.Train(features, targets);
///
/// // Predict survival probabilities for a new patient
/// var newPatient = Matrix&lt;double&gt;.Build.Dense(1, 3, new double[] { 58, 1, 2.5 });
/// var prediction = model.Predict(newPatient);
/// </code>
/// </example>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Healthcare)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.SurvivalModel)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("DeepHit: A Deep Learning Approach to Survival Analysis with Competing Risks", "https://ojs.aaai.org/index.php/AAAI/article/view/11842", Year = 2018, Authors = "Changhee Lee, William R. Zame, Jinsung Yoon, Mihaela van der Schaar")]
public class DeepHit<T> : AsyncDecisionTreeRegressionBase<T>
{
    /// <summary>
    /// Shared network weights.
    /// </summary>
    private List<Matrix<T>> _sharedWeights;

    /// <summary>
    /// Shared network biases.
    /// </summary>
    private List<Vector<T>> _sharedBiases;

    /// <summary>
    /// Cause-specific network weights (one list per cause).
    /// </summary>
    private List<List<Matrix<T>>> _causeWeights;

    /// <summary>
    /// Cause-specific network biases (one list per cause).
    /// </summary>
    private List<List<Vector<T>>> _causeBiases;

    /// <summary>
    /// Output layer weights (for each cause, maps to time bins).
    /// </summary>
    private List<Matrix<T>> _outputWeights;

    /// <summary>
    /// Output layer biases (for each cause).
    /// </summary>
    private List<Vector<T>> _outputBiases;

    /// <summary>
    /// Number of features.
    /// </summary>
    private int _numFeatures;

    /// <summary>
    /// Per-feature training mean, used to standardize inputs before the shared trunk.
    /// </summary>
    private Vector<T>? _featureMean;

    /// <summary>
    /// Per-feature training standard deviation, used to standardize inputs before the shared trunk.
    /// </summary>
    private Vector<T>? _featureStd;



    /// <summary>
    /// Time bin edges (discretization of time axis).
    /// </summary>
    private Vector<T>? _timeBinEdges;

    /// <summary>
    /// The number of time bins actually used, which may be fewer than
    /// <see cref="DeepHitOptions{T}.NumTimeBins"/> when the training set cannot support that many.
    /// </summary>
    private int _effectiveTimeBins;

    /// <summary>
    /// The number of time bins the model is currently using.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Every consumer reads this rather than the option, because the option is an upper bound and the grid
    /// is sized to the data. The output layer has one cell per (cause, bin), and the log-likelihood puts
    /// each subject's mass on exactly one of them, so a grid finer than the data can fill leaves most
    /// cells with no training signal at all: with the default 100 bins and 100 subjects there is about one
    /// event per bin, the softmax stays near uniform, and the predicted expectation collapses to the mean
    /// observed time regardless of the covariates. Discrete-time survival practice sizes the grid to the
    /// sample for exactly this reason.
    /// </para>
    /// </remarks>
    private int NumTimeBins => _effectiveTimeBins > 0 ? _effectiveTimeBins : _options.NumTimeBins;

    /// <summary>
    /// Configuration options.
    /// </summary>
    private readonly DeepHitOptions<T> _options;

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
    public DeepHit(DeepHitOptions<T>? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(null, regularization)
    {
        _options = options ?? new DeepHitOptions<T>();
        _sharedWeights = [];
        _sharedBiases = [];
        _causeWeights = [];
        _causeBiases = [];
        _outputWeights = [];
        _outputBiases = [];
        _numFeatures = 0;
        _random = _options.Seed.HasValue ? RandomHelper.CreateSeededRandom(_options.Seed.Value) : RandomHelper.CreateSecureRandom();
    }


    /// <inheritdoc/>
    public override async Task TrainAsync(Matrix<T> x, Vector<T> y)
    {
        // The general regression interface carries one target vector and cannot express censoring or a
        // cause of failure, so every observation is treated as an observed event of cause 1. Use
        // TrainAsync(x, times, events) for censored or competing-risks data.
        var events = Vector<T>.CreateDefault(y.Length, NumOps.One);
        await TrainAsync(x, y, events);
    }

    /// <summary>
    /// Trains the network on right-censored, possibly competing-risks survival data.
    /// </summary>
    /// <param name="x">Feature matrix, one row per subject.</param>
    /// <param name="times">Observed time for each subject: the event time, or the censoring time.</param>
    /// <param name="events">
    /// 0 when the subject was censored, otherwise the 1-based index of the cause that occurred.
    /// </param>
    /// <remarks>
    /// <para>
    /// This is the procedure from Lee et al. (2018): a shared trunk feeds one branch per competing cause,
    /// and a single softmax over every (cause, time bin) cell gives the joint distribution of when the
    /// event happens and which cause causes it. The objective is the log-likelihood term plus the ranking
    /// term, both of which <c>ComputeLossAndGradients</c> already computed -- and which nothing called,
    /// because this method fitted ordinary least squares and returned.
    /// </para>
    /// <para>
    /// Optimization is mini-batch Adam with decoupled L2 decay and early stopping, using the Epochs,
    /// BatchSize, LearningRate, L2Regularization, DropoutRate and EarlyStoppingPatience settings that were
    /// previously read by nothing.
    /// </para>
    /// </remarks>
    public async Task TrainAsync(Matrix<T> x, Vector<T> times, Vector<T> events)
    {
        ValidationHelper<T>.ValidateInputData(x, times);

        if (events.Length != times.Length)
        {
            throw new ArgumentException(
                $"The event vector must have one entry per subject: got {events.Length} entries for " +
                $"{times.Length} observed times.",
                nameof(events));
        }

        for (int i = 0; i < times.Length; i++)
        {
            if (!NumOps.GreaterThan(times[i], NumOps.Zero))
            {
                throw new ArgumentException(
                    $"Survival times must be strictly positive; got {NumOps.ToDouble(times[i]):G6} at index {i}.",
                    nameof(times));
            }
        }

        for (int i = 0; i < events.Length; i++)
        {
            double e = NumOps.ToDouble(events[i]);
            if (e < 0 || e > _options.NumRisks || Math.Abs(e - Math.Round(e)) > 1e-9)
            {
                throw new ArgumentException(
                    $"Event codes must be 0 for censored or a 1-based cause index up to NumRisks " +
                    $"({_options.NumRisks}); got {e:G6} at index {i}.",
                    nameof(events));
            }
        }

        _numFeatures = x.Columns;
        ComputeFeatureStandardization(x);
        InitializeTimeBins(times);
        InitializeNetwork();

        int[] timeBinIndices = ConvertTimesToBins(times);

        await Task.Run(() => FitNetwork(x, timeBinIndices, events));

        await CalculateFeatureImportancesAsync(x.Columns);
    }

    /// <summary>
    /// Predicts the expected event time for each subject.
    /// </summary>
    /// <param name="input">Feature matrix, one row per subject.</param>
    /// <returns>Expected event time per subject, on the same scale as the training times.</returns>
    /// <remarks>
    /// The model was trained against observed times, so the general regression interface returns a time,
    /// keeping <c>Predict</c> on the scale of the <c>y</c> passed to <c>Train</c>. The expectation is taken
    /// over the predicted joint distribution of cause and time bin. <see cref="PredictPMF"/>,
    /// <see cref="PredictCIF"/> and <see cref="PredictSurvival"/> expose the full distribution.
    /// </remarks>
    public override async Task<Vector<T>> PredictAsync(Matrix<T> input)
    {
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
        var (pmfs, _) = ForwardPass(input, indices, training: false);

        var result = new T[input.Rows, _options.NumRisks, NumTimeBins];
        for (int i = 0; i < input.Rows; i++)
        {
            for (int k = 0; k < _options.NumRisks; k++)
            {
                for (int t = 0; t < NumTimeBins; t++)
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
    public Matrix<T> PredictSurvival(Matrix<T> input, Vector<T> times)
    {
        var pmf = PredictPMF(input);
        var survivalProbs = new Matrix<T>(input.Rows, times.Length);

        for (int i = 0; i < input.Rows; i++)
        {
            for (int j = 0; j < times.Length; j++)
            {
                int binIndex = GetTimeBinIndex(times[j]);

                // S(t) = 1 - sum of PMF up to time bin
                T cumProb = NumOps.Zero;
                for (int k = 0; k < _options.NumRisks; k++)
                {
                    for (int b = 0; b <= binIndex && b < NumTimeBins; b++)
                    {
                        cumProb = NumOps.Add(cumProb, pmf[i, k, b]);
                    }
                }

                T survival = NumOps.Subtract(NumOps.One, cumProb);
                survivalProbs[i, j] = NumOps.LessThan(survival, NumOps.Zero) ? NumOps.Zero : survival;
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
    public Matrix<T> PredictCIF(Matrix<T> input, Vector<T> times, int riskIndex = 0)
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
                int binIndex = GetTimeBinIndex(times[j]);

                // CIF(t, k) = sum of PMF_k up to time bin
                T cumProb = NumOps.Zero;
                for (int b = 0; b <= binIndex && b < NumTimeBins; b++)
                {
                    cumProb = NumOps.Add(cumProb, pmf[i, riskIndex, b]);
                }

                cif[i, j] = cumProb;
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
            T expected = NumOps.Zero;
            T totalProb = NumOps.Zero;

            for (int k = 0; k < _options.NumRisks; k++)
            {
                for (int t = 0; t < NumTimeBins; t++)
                {
                    T prob = pmf[i, k, t];
                    T time = GetTimeBinCenterT(t);
                    expected = NumOps.Add(expected, NumOps.Multiply(prob, time));
                    totalProb = NumOps.Add(totalProb, prob);
                }
            }

            // Normalize by total probability (may be < 1 for censored observations)
            expectedTimes[i] = NumOps.GreaterThan(totalProb, NumOps.Zero)
                ? NumOps.Divide(expected, totalProb)
                : GetTimeBinCenterT(NumTimeBins - 1);
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

        T half = NumOps.FromDouble(0.5);
        for (int i = 0; i < input.Rows; i++)
        {
            T cumProb = NumOps.Zero;
            bool found = false;

            // Find time bin where cumulative probability crosses 0.5
            for (int t = 0; t < NumTimeBins; t++)
            {
                for (int k = 0; k < _options.NumRisks; k++)
                {
                    cumProb = NumOps.Add(cumProb, pmf[i, k, t]);
                }

                if (!NumOps.LessThan(cumProb, half))
                {
                    medianTimes[i] = GetTimeBinCenterT(t);
                    found = true;
                    break;
                }
            }

            if (!found)
            {
                medianTimes[i] = GetTimeBinCenterT(NumTimeBins - 1);
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
        var horizonVec = new Vector<T>(new[] { horizon });
        var survivalProbs = PredictSurvival(x, horizonVec);

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
        // Size the grid to the sample. Aiming for roughly five observations per bin keeps every cell of
        // the softmax supported by real events; asking for more bins than that produces a distribution the
        // data cannot estimate. See the NumTimeBins property for what happens when this is ignored.
        const int targetObservationsPerBin = 5;
        int supportable = Math.Max(2, times.Length / targetObservationsPerBin);
        _effectiveTimeBins = Math.Max(2, Math.Min(_options.NumTimeBins, supportable));

        T minTime = times[0];
        T maxTime = times[0];
        for (int i = 1; i < times.Length; i++)
        {
            if (NumOps.LessThan(times[i], minTime)) minTime = times[i];
            if (NumOps.GreaterThan(times[i], maxTime)) maxTime = times[i];
        }

        // Add small buffer
        T range = NumOps.Subtract(maxTime, minTime);
        maxTime = NumOps.Add(maxTime, NumOps.Multiply(range, NumOps.FromDouble(0.01)));

        _timeBinEdges = new Vector<T>(NumTimeBins + 1);
        T binWidth = NumOps.Divide(NumOps.Subtract(maxTime, minTime), NumOps.FromDouble(NumTimeBins));

        for (int i = 0; i <= NumTimeBins; i++)
        {
            _timeBinEdges[i] = NumOps.Add(minTime, NumOps.Multiply(NumOps.FromDouble(i), binWidth));
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
            binIndices[i] = GetTimeBinIndex(times[i]);
        }

        return binIndices;
    }

    /// <summary>
    /// Gets the bin index for a given time.
    /// </summary>
    private int GetTimeBinIndex(T time)
    {
        if (_timeBinEdges == null)
        {
            return 0;
        }

        for (int i = 1; i < _timeBinEdges.Length; i++)
        {
            if (NumOps.LessThan(time, _timeBinEdges[i]))
            {
                return i - 1;
            }
        }

        return NumTimeBins - 1;
    }

    /// <summary>
    /// Gets the center time of a bin as T.
    /// </summary>
    private T GetTimeBinCenterT(int binIndex)
    {
        if (_timeBinEdges == null)
        {
            return NumOps.FromDouble(binIndex);
        }

        T left = _timeBinEdges[binIndex];
        T right = _timeBinEdges[Math.Min(binIndex + 1, _timeBinEdges.Length - 1)];
        return NumOps.Divide(NumOps.Add(left, right), NumOps.FromDouble(2));
    }

    /// <summary>
    /// Gets the center time of a bin as double (for evaluation metrics).
    /// </summary>
    private double GetTimeBinCenter(int binIndex)
    {
        return NumOps.ToDouble(GetTimeBinCenterT(binIndex));
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
            var causeW = new List<Matrix<T>>();
            var causeB = new List<Vector<T>>();

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
            _outputWeights.Add(InitializeWeights(inputSize, NumTimeBins));
            _outputBiases.Add(InitializeBiases(NumTimeBins));
        }
    }

    /// <summary>
    /// Initializes weight matrix with He initialization.
    /// </summary>
    private Matrix<T> InitializeWeights(int inputSize, int outputSize)
    {
        double scale = Math.Sqrt(2.0 / inputSize);
        var w = new Matrix<T>(inputSize, outputSize);

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
    /// Initializes bias vector with zeros.
    /// </summary>
    private Vector<T> InitializeBiases(int size)
    {
        return new Vector<T>(size);
    }

    /// <summary>
    /// What one layer computed on the way forward, kept so the backward pass can retrace it.
    /// </summary>
    private sealed class LayerCache
    {
        public Vector<T>[] Input { get; set; } = [];
        public Vector<T>[] PreActivation { get; set; } = [];
        public Vector<T>[] DropoutMask { get; set; } = [];
        public bool HasActivation { get; set; }
    }

    /// <summary>
    /// Every cache from one forward pass, in the order the network applies them.
    /// </summary>
    private sealed class DeepHitForwardCache
    {
        public List<LayerCache> Shared { get; } = new List<LayerCache>();
        public List<List<LayerCache>> Cause { get; } = new List<List<LayerCache>>();
        public List<LayerCache> Output { get; } = new List<LayerCache>();
    }

    /// <summary>
    /// Forward pass through the shared trunk and the per-cause heads.
    /// </summary>
    /// <param name="x">Feature matrix.</param>
    /// <param name="indices">Rows of <paramref name="x"/> to run.</param>
    /// <param name="training">
    /// When true, dropout is applied. When false no units are dropped, so predictions are deterministic.
    /// </param>
    /// <returns>The per-sample PMF over (cause, time bin), and the caches the backward pass needs.</returns>
    private (Vector<T>[][] Pmfs, DeepHitForwardCache Cache) ForwardPass(Matrix<T> x, int[] indices, bool training)
    {
        int n = indices.Length;
        var cache = new DeepHitForwardCache();

        var current = new Vector<T>[n];
        for (int i = 0; i < n; i++)
        {
            current[i] = new Vector<T>(_numFeatures);
            for (int j = 0; j < _numFeatures; j++)
            {
                current[i][j] = StandardizeFeature(x[indices[i], j], j);
            }
        }

        // Shared trunk.
        for (int layer = 0; layer < _sharedWeights.Count; layer++)
        {
            current = ApplyLayer(current, _sharedWeights[layer], _sharedBiases[layer],
                applyActivation: true, training: training, caches: cache.Shared);
        }

        var sharedOutput = current;

        var pmfs = new Vector<T>[n][];
        for (int i = 0; i < n; i++)
        {
            pmfs[i] = new Vector<T>[_options.NumRisks];
        }

        for (int k = 0; k < _options.NumRisks; k++)
        {
            var causeCaches = new List<LayerCache>();
            current = CloneArray(sharedOutput);

            for (int layer = 0; layer < _causeWeights[k].Count; layer++)
            {
                current = ApplyLayer(current, _causeWeights[k][layer], _causeBiases[k][layer],
                    applyActivation: true, training: training, caches: causeCaches);
            }

            cache.Cause.Add(causeCaches);

            // Output layer produces logits, with no activation.
            var outputCaches = new List<LayerCache>();
            var logits = ApplyLayer(current, _outputWeights[k], _outputBiases[k],
                applyActivation: false, training: false, caches: outputCaches);
            cache.Output.Add(outputCaches[0]);

            for (int i = 0; i < n; i++)
            {
                pmfs[i][k] = logits[i];
            }
        }

        // One softmax over every (cause, time bin) cell, as DeepHit specifies.
        ApplySoftmaxAcrossAll(pmfs);

        return (pmfs, cache);
    }

    /// <summary>
    /// Applies a single fully-connected layer, recording what the backward pass will need.
    /// </summary>
    /// <remarks>
    /// Dropout is applied only when <paramref name="training"/> is true. It previously ran on EVERY call,
    /// including from Predict, which made predictions random: the same model and the same input returned
    /// different answers each time. Inverted dropout also means the surviving units are scaled at training
    /// time, so inference needs no rescaling of its own.
    /// </remarks>
    private Vector<T>[] ApplyLayer(Vector<T>[] input, Matrix<T> weights, Vector<T> biases,
        bool applyActivation, bool training, List<LayerCache> caches)
    {
        int n = input.Length;
        int outputSize = weights.Columns;
        var weightTensor = Tensor<T>.FromMatrix(weights);
        var biasTensor = Tensor<T>.FromVector(biases).Reshape(1, outputSize);

        var preActivation = new Vector<T>[n];
        var output = new Vector<T>[n];

        for (int i = 0; i < n; i++)
        {
            var inputTensor = Tensor<T>.FromVector(input[i]).Reshape(1, input[i].Length);
            var result = Engine.TensorAdd(
                Engine.TensorMatMul(inputTensor, weightTensor), biasTensor);

            preActivation[i] = result.Reshape(outputSize).ToVector();

            if (applyActivation)
            {
                var activated = _options.Activation.Activate(
                    Tensor<T>.FromVector(preActivation[i]).Reshape(1, outputSize));
                output[i] = activated.Reshape(outputSize).ToVector();
            }
            else
            {
                output[i] = preActivation[i];
            }
        }

        var masks = new Vector<T>[n];
        if (applyActivation && training && _options.DropoutRate > 0)
        {
            T keepScale = NumOps.FromDouble(1.0 / (1.0 - _options.DropoutRate));
            for (int i = 0; i < n; i++)
            {
                masks[i] = new Vector<T>(outputSize);
                for (int j = 0; j < outputSize; j++)
                {
                    bool keep = _random.NextDouble() >= _options.DropoutRate;
                    masks[i][j] = keep ? keepScale : NumOps.Zero;
                    output[i][j] = NumOps.Multiply(output[i][j], masks[i][j]);
                }
            }
        }
        else
        {
            for (int i = 0; i < n; i++)
            {
                masks[i] = Vector<T>.CreateDefault(outputSize, NumOps.One);
            }
        }

        caches.Add(new LayerCache
        {
            Input = input,
            PreActivation = preActivation,
            DropoutMask = masks,
            HasActivation = applyActivation
        });

        return output;
    }

    /// <summary>
    /// Backpropagates one layer, accumulating its parameter gradients and returning the delta for its input.
    /// </summary>
    /// <param name="cache">What the layer recorded on the way forward.</param>
    /// <param name="delta">d(loss)/d(layer output).</param>
    /// <param name="weights">The layer's weight matrix.</param>
    /// <param name="gradWeights">Accumulator for d(loss)/d(weights).</param>
    /// <param name="gradBiases">Accumulator for d(loss)/d(biases).</param>
    /// <returns>d(loss)/d(layer input).</returns>
    private Vector<T>[] BackwardLayer(LayerCache cache, Vector<T>[] delta, Matrix<T> weights,
        Matrix<T> gradWeights, Vector<T> gradBiases)
    {
        int n = delta.Length;
        int outputSize = weights.Columns;
        int inputSize = weights.Rows;

        var inputDelta = new Vector<T>[n];
        for (int i = 0; i < n; i++)
        {
            inputDelta[i] = new Vector<T>(inputSize);
        }

        for (int i = 0; i < n; i++)
        {
            for (int k = 0; k < outputSize; k++)
            {
                // Back through dropout, then the activation.
                T d = NumOps.Multiply(delta[i][k], cache.DropoutMask[i][k]);
                if (cache.HasActivation)
                {
                    d = NumOps.Multiply(d, _options.Activation.Derivative(cache.PreActivation[i][k]));
                }

                gradBiases[k] = NumOps.Add(gradBiases[k], d);

                for (int j = 0; j < inputSize; j++)
                {
                    gradWeights[j, k] = NumOps.Add(gradWeights[j, k], NumOps.Multiply(d, cache.Input[i][j]));
                    inputDelta[i][j] = NumOps.Add(inputDelta[i][j], NumOps.Multiply(d, weights[j, k]));
                }
            }
        }

        return inputDelta;
    }

    private void ApplySoftmaxAcrossAll(Vector<T>[][] pmfs)
    {
        int n = pmfs.Length;
        T tiny = NumOps.FromDouble(1e-10);

        for (int i = 0; i < n; i++)
        {
            // Find max logit for numerical stability
            T maxLogit = pmfs[i][0][0];
            for (int k = 0; k < _options.NumRisks; k++)
            {
                for (int t = 0; t < NumTimeBins; t++)
                {
                    if (NumOps.GreaterThan(pmfs[i][k][t], maxLogit))
                        maxLogit = pmfs[i][k][t];
                }
            }

            // Softmax with numerical stability
            T sumExp = NumOps.Zero;
            for (int k = 0; k < _options.NumRisks; k++)
            {
                for (int t = 0; t < NumTimeBins; t++)
                {
                    T expVal = NumOps.Exp(NumOps.Subtract(pmfs[i][k][t], maxLogit));
                    pmfs[i][k][t] = expVal;
                    sumExp = NumOps.Add(sumExp, expVal);
                }
            }

            // Normalize
            T denom = NumOps.Add(sumExp, tiny);
            for (int k = 0; k < _options.NumRisks; k++)
            {
                for (int t = 0; t < NumTimeBins; t++)
                {
                    pmfs[i][k][t] = NumOps.Divide(pmfs[i][k][t], denom);
                }
            }
        }
    }

    /// <summary>
    /// Computes loss and gradients.
    /// </summary>
    private (T loss, Vector<T>[][]) ComputeLossAndGradients(
        Vector<T>[][] pmfs, int[] timeBinIndices, Vector<T> events, int[] batchIndices)
    {
        int n = batchIndices.Length;
        T logLikeLoss = NumOps.Zero;
        T rankingLoss = NumOps.Zero;
        T tiny = NumOps.FromDouble(1e-10);

        var gradients = new Vector<T>[n][];
        for (int i = 0; i < n; i++)
        {
            gradients[i] = new Vector<T>[_options.NumRisks];
            for (int k = 0; k < _options.NumRisks; k++)
            {
                gradients[i][k] = new Vector<T>(NumTimeBins);
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
                if (k < _options.NumRisks && timeBin < NumTimeBins)
                {
                    T prob = pmfs[bi][k][timeBin];
                    logLikeLoss = NumOps.Subtract(logLikeLoss, NumOps.Log(NumOps.Add(prob, tiny)));

                    // Gradient for softmax cross-entropy
                    for (int kk = 0; kk < _options.NumRisks; kk++)
                    {
                        for (int tt = 0; tt < NumTimeBins; tt++)
                        {
                            T target = (kk == k && tt == timeBin) ? NumOps.One : NumOps.Zero;
                            gradients[bi][kk][tt] = NumOps.Subtract(pmfs[bi][kk][tt], target);
                        }
                    }
                }
            }
            else
            {
                // Censored - maximize survival probability up to censoring time
                T cumProb = NumOps.Zero;
                for (int k = 0; k < _options.NumRisks; k++)
                {
                    for (int t = 0; t < timeBin && t < NumTimeBins; t++)
                    {
                        cumProb = NumOps.Add(cumProb, pmfs[bi][k][t]);
                    }
                }

                logLikeLoss = NumOps.Subtract(logLikeLoss,
                    NumOps.Log(NumOps.Add(NumOps.Subtract(NumOps.One, cumProb), tiny)));

                // Gradient with respect to the LOGITS, not the probabilities. With one softmax over all
                // cells, dp[a]/dz[b] = p[a](delta_ab - p[b]), so for L = -log(1 - C) with
                // C = sum of p over the cells before the censoring bin:
                //
                //   dL/dz[b] = p[b] * (1{b before bin} - C) / (1 - C)
                //
                // which is p[b] before the bin and -p[b]*C/(1-C) after it. The previous version used
                // p[b]/(1-C) before the bin and ZERO after, treating the probability derivative as if it
                // were the logit derivative. A softmax logit gradient must sum to zero over all cells,
                // because adding a constant to every logit leaves the probabilities unchanged; that one
                // summed to C/(1-C), so every censored subject pushed the whole distribution one way.
                T survDenom = NumOps.Add(NumOps.Subtract(NumOps.One, cumProb), tiny);
                T tailScale = NumOps.Divide(cumProb, survDenom);
                for (int k = 0; k < _options.NumRisks; k++)
                {
                    for (int t = 0; t < NumTimeBins; t++)
                    {
                        gradients[bi][k][t] = t < timeBin
                            ? pmfs[bi][k][t]
                            : NumOps.Negate(NumOps.Multiply(pmfs[bi][k][t], tailScale));
                    }
                }
            }
        }

        // Ranking loss
        if (_options.RankingWeight > 0)
        {
            T sigma = NumOps.FromDouble(_options.RankingSigma);
            T rankWeight = NumOps.FromDouble(_options.RankingWeight);
            T nT = NumOps.FromDouble(n);

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
                        T cifI = NumOps.Zero;
                        T cifJ = NumOps.Zero;
                        for (int k = 0; k < _options.NumRisks; k++)
                        {
                            for (int t = 0; t <= timeBinI && t < NumTimeBins; t++)
                            {
                                cifI = NumOps.Add(cifI, pmfs[i][k][t]);
                                cifJ = NumOps.Add(cifJ, pmfs[j][k][t]);
                            }
                        }

                        // DeepHit eq. 8: eta = exp(-(F_i(T_i) - F_j(T_i)) / sigma). Subject i failed first,
                        // so its cumulative incidence at that time SHOULD exceed j's; the penalty must
                        // therefore shrink as F_i - F_j grows.
                        //
                        // The sign was inverted here. `diff` was computed as cifJ - cifI and then negated,
                        // giving exp(+(F_i - F_j)/sigma) -- a penalty that grows when the ranking is
                        // RIGHT. Minimizing it drove F_i below F_j, so the ranking term actively fought
                        // the log-likelihood term. That is why the loss stopped improving within a few
                        // epochs and early stopping cut training short: the two terms had reached a
                        // stalemate, not a fit.
                        T diff = NumOps.Subtract(cifI, cifJ);
                        T eta = NumOps.Exp(NumOps.Negate(NumOps.Divide(diff, sigma)));
                        rankingLoss = NumOps.Add(rankingLoss, eta);

                        // As in the censored branch, this must be a gradient with respect to the LOGITS.
                        // d(eta)/d(F_i) = -eta/sigma over the cells at or before timeBinI, so pushing that
                        // through the softmax Jacobian gives, for subject i,
                        //
                        //   d(eta)/dz_i[b] = (-eta/sigma) * p_i[b] * (1{b <= timeBinI} - F_i)
                        //
                        // and the same with the opposite sign and F_j for subject j. The old code added a
                        // bare constant to the leading cells instead, which does not sum to zero over the
                        // cells and so injected a spurious uniform shift into every update.
                        T scale = NumOps.Divide(NumOps.Multiply(NumOps.Divide(eta, sigma), rankWeight), nT);
                        for (int k = 0; k < _options.NumRisks; k++)
                        {
                            for (int t = 0; t < NumTimeBins; t++)
                            {
                                T indicator = t <= timeBinI ? NumOps.One : NumOps.Zero;

                                T centeredI = NumOps.Subtract(indicator, cifI);
                                gradients[i][k][t] = NumOps.Subtract(
                                    gradients[i][k][t],
                                    NumOps.Multiply(scale, NumOps.Multiply(pmfs[i][k][t], centeredI)));

                                T centeredJ = NumOps.Subtract(indicator, cifJ);
                                gradients[j][k][t] = NumOps.Add(
                                    gradients[j][k][t],
                                    NumOps.Multiply(scale, NumOps.Multiply(pmfs[j][k][t], centeredJ)));
                            }
                        }
                    }
                }
            }
        }

        T totalLoss = NumOps.Divide(
            NumOps.Add(logLikeLoss, NumOps.Multiply(NumOps.FromDouble(_options.RankingWeight), rankingLoss)),
            NumOps.FromDouble(n));
        return (totalLoss, gradients);
    }


     /// <summary>
    /// Records the per-feature mean and standard deviation of the training inputs.
    /// </summary>
    /// <remarks>
    /// The trunk is fed raw covariates otherwise, which pushes the output logits far enough apart that the
    /// shared softmax saturates: one cell holds essentially all the mass, its gradient vanishes, and the
    /// network stops learning. Standardizing keeps the logits in a range where the softmax has gradient.
    /// </remarks>
    private void ComputeFeatureStandardization(Matrix<T> x)
    {
        int n = x.Rows;
        _featureMean = new Vector<T>(x.Columns);
        _featureStd = new Vector<T>(x.Columns);

        for (int j = 0; j < x.Columns; j++)
        {
            double sum = 0.0;
            for (int i = 0; i < n; i++)
            {
                sum += NumOps.ToDouble(x[i, j]);
            }
            double mean = sum / n;

            double sq = 0.0;
            for (int i = 0; i < n; i++)
            {
                double d = NumOps.ToDouble(x[i, j]) - mean;
                sq += d * d;
            }
            double std = Math.Sqrt(sq / n);
            if (std < 1e-10) std = 1.0;

            _featureMean[j] = NumOps.FromDouble(mean);
            _featureStd[j] = NumOps.FromDouble(std);
        }
    }

    /// <summary>
    /// Applies the training standardization to one feature value.
    /// </summary>
    private T StandardizeFeature(T value, int featureIndex)
    {
        if (_featureMean is null || _featureStd is null ||
            featureIndex >= _featureMean.Length || featureIndex >= _featureStd.Length)
        {
            return value;
        }

        return NumOps.Divide(NumOps.Subtract(value, _featureMean[featureIndex]), _featureStd[featureIndex]);
    }

    /// <summary>
    /// Collects every weight matrix and bias vector in a fixed order.
    /// </summary>
    /// <remarks>
    /// The lists hold the live objects, not copies, so an optimizer that writes through them updates the
    /// network directly. Gradient accumulators and Adam moments are built in this same order, which is what
    /// lets one flat loop drive shared, cause-specific and output layers alike.
    /// </remarks>
    private (List<Matrix<T>> Weights, List<Vector<T>> Biases) FlattenParameters()
    {
        var weights = new List<Matrix<T>>();
        var biases = new List<Vector<T>>();

        for (int layer = 0; layer < _sharedWeights.Count; layer++)
        {
            weights.Add(_sharedWeights[layer]);
            biases.Add(_sharedBiases[layer]);
        }

        for (int k = 0; k < _options.NumRisks; k++)
        {
            for (int layer = 0; layer < _causeWeights[k].Count; layer++)
            {
                weights.Add(_causeWeights[k][layer]);
                biases.Add(_causeBiases[k][layer]);
            }
        }

        for (int k = 0; k < _options.NumRisks; k++)
        {
            weights.Add(_outputWeights[k]);
            biases.Add(_outputBiases[k]);
        }

        return (weights, biases);
    }

    /// <summary>
    /// Allocates zeroed gradient accumulators shaped like the parameters.
    /// </summary>
    private (List<Matrix<T>> Weights, List<Vector<T>> Biases) AllocateLike(
        List<Matrix<T>> weights, List<Vector<T>> biases)
    {
        var gw = new List<Matrix<T>>(weights.Count);
        var gb = new List<Vector<T>>(biases.Count);

        for (int i = 0; i < weights.Count; i++)
        {
            gw.Add(new Matrix<T>(weights[i].Rows, weights[i].Columns));
            gb.Add(new Vector<T>(biases[i].Length));
        }

        return (gw, gb);
    }

    /// <summary>
    /// Fits the network by minimizing the DeepHit loss: the log-likelihood term plus the ranking term.
    /// </summary>
    /// <param name="x">Feature matrix.</param>
    /// <param name="timeBinIndices">Discretized event time per subject.</param>
    /// <param name="events">Event type per subject: 0 for censored, 1..NumRisks for a cause.</param>
    /// <remarks>
    /// Mini-batch Adam with decoupled L2 decay and early stopping. SaveWeights and RestoreWeights already
    /// existed for exactly this purpose and had no callers, because training never ran.
    /// </remarks>
    private void FitNetwork(Matrix<T> x, int[] timeBinIndices, Vector<T> events)
    {
        int n = x.Rows;
        int batchSize = Math.Max(1, Math.Min(_options.BatchSize, n));
        var (weights, biases) = FlattenParameters();
        var (mW, mB) = AllocateLike(weights, biases);
        var (vW, vB) = AllocateLike(weights, biases);
        int adamStep = 0;

        double bestLoss = double.MaxValue;
        int epochsWithoutImprovement = 0;
        var bestWeights = SaveWeights();

        var order = Enumerable.Range(0, n).ToArray();

        for (int epoch = 0; epoch < _options.Epochs; epoch++)
        {
            order = ShuffleArray(order);
            double epochLoss = 0.0;
            int batches = 0;

            for (int start = 0; start < n; start += batchSize)
            {
                int count = Math.Min(batchSize, n - start);
                var batchIndices = new int[count];
                Array.Copy(order, start, batchIndices, 0, count);

                var (pmfs, cache) = ForwardPass(x, batchIndices, training: true);
                var (loss, logitGradients) = ComputeLossAndGradients(pmfs, timeBinIndices, events, batchIndices);

                var (gradW, gradB) = AllocateLike(weights, biases);
                Backpropagate(cache, logitGradients, gradW, gradB);

                adamStep++;
                ApplyAdam(weights, biases, gradW, gradB, mW, mB, vW, vB, adamStep);

                epochLoss += NumOps.ToDouble(loss);
                batches++;
            }

            if (batches == 0)
            {
                continue;
            }

            double meanLoss = epochLoss / batches;

            if (meanLoss < bestLoss - 1e-7)
            {
                bestLoss = meanLoss;
                epochsWithoutImprovement = 0;
                bestWeights = SaveWeights();
            }
            else
            {
                epochsWithoutImprovement++;
                if (_options.EarlyStoppingPatience.HasValue &&
                    epochsWithoutImprovement >= _options.EarlyStoppingPatience.Value)
                {
                    break;
                }
            }
        }

        // Keep the best epoch, not the last one.
        RestoreWeights(bestWeights);
    }

    /// <summary>
    /// Propagates the logit gradients back through the output heads, the cause branches and the shared trunk.
    /// </summary>
    /// <param name="cache">Caches recorded by the forward pass.</param>
    /// <param name="logitGradients">d(loss)/d(logit) per sample, cause and time bin.</param>
    /// <param name="gradWeights">Accumulators in <see cref="FlattenParameters"/> order.</param>
    /// <param name="gradBiases">Accumulators in <see cref="FlattenParameters"/> order.</param>
    /// <remarks>
    /// The shared trunk feeds every cause branch, so its delta is the SUM of what each branch sends back.
    /// Overwriting rather than summing there would train the trunk on one cause only.
    /// </remarks>
    private void Backpropagate(DeepHitForwardCache cache, Vector<T>[][] logitGradients,
        List<Matrix<T>> gradWeights, List<Vector<T>> gradBiases)
    {
        int n = logitGradients.Length;
        int sharedCount = _sharedWeights.Count;
        int causeLayersPerRisk = _options.NumRisks > 0 ? _causeWeights[0].Count : 0;

        // Delta arriving at the shared trunk's output, summed over causes.
        int sharedOutputSize = sharedCount > 0
            ? _sharedWeights[sharedCount - 1].Columns
            : _numFeatures;

        var sharedDelta = new Vector<T>[n];
        for (int i = 0; i < n; i++)
        {
            sharedDelta[i] = new Vector<T>(sharedOutputSize);
        }

        for (int k = 0; k < _options.NumRisks; k++)
        {
            // d(loss)/d(logit) for this cause.
            var delta = new Vector<T>[n];
            for (int i = 0; i < n; i++)
            {
                delta[i] = logitGradients[i][k];
            }

            int outputParamIndex = sharedCount + _options.NumRisks * causeLayersPerRisk + k;
            delta = BackwardLayer(cache.Output[k], delta, _outputWeights[k],
                gradWeights[outputParamIndex], gradBiases[outputParamIndex]);

            for (int layer = causeLayersPerRisk - 1; layer >= 0; layer--)
            {
                int paramIndex = sharedCount + k * causeLayersPerRisk + layer;
                delta = BackwardLayer(cache.Cause[k][layer], delta, _causeWeights[k][layer],
                    gradWeights[paramIndex], gradBiases[paramIndex]);
            }

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < sharedOutputSize && j < delta[i].Length; j++)
                {
                    sharedDelta[i][j] = NumOps.Add(sharedDelta[i][j], delta[i][j]);
                }
            }
        }

        var trunkDelta = sharedDelta;
        for (int layer = sharedCount - 1; layer >= 0; layer--)
        {
            trunkDelta = BackwardLayer(cache.Shared[layer], trunkDelta, _sharedWeights[layer],
                gradWeights[layer], gradBiases[layer]);
        }
    }

    /// <summary>
    /// Applies one Adam step with decoupled L2 decay to every parameter, in place.
    /// </summary>
    /// <remarks>
    /// Weight decay applies to the weight matrices only; biases are left undecayed, which is the standard
    /// convention because shrinking them removes the layer's ability to shift its output.
    /// </remarks>
    private void ApplyAdam(List<Matrix<T>> weights, List<Vector<T>> biases,
        List<Matrix<T>> gradWeights, List<Vector<T>> gradBiases,
        List<Matrix<T>> mW, List<Vector<T>> mB, List<Matrix<T>> vW, List<Vector<T>> vB, int step)
    {
        double lr = _options.LearningRate;
        const double beta1 = 0.9;
        const double beta2 = 0.999;
        const double eps = 1e-8;
        double decay = _options.L2Regularization;
        double correction1 = 1.0 - Math.Pow(beta1, step);
        double correction2 = 1.0 - Math.Pow(beta2, step);

        for (int p = 0; p < weights.Count; p++)
        {
            var w = weights[p];
            var g = gradWeights[p];
            var m = mW[p];
            var v = vW[p];

            for (int i = 0; i < w.Rows; i++)
            {
                for (int j = 0; j < w.Columns; j++)
                {
                    double grad = NumOps.ToDouble(g[i, j]) + decay * NumOps.ToDouble(w[i, j]);
                    double mv = beta1 * NumOps.ToDouble(m[i, j]) + (1 - beta1) * grad;
                    double vv = beta2 * NumOps.ToDouble(v[i, j]) + (1 - beta2) * grad * grad;
                    m[i, j] = NumOps.FromDouble(mv);
                    v[i, j] = NumOps.FromDouble(vv);

                    double stepSize = lr * (mv / correction1) / (Math.Sqrt(vv / correction2) + eps);
                    w[i, j] = NumOps.FromDouble(NumOps.ToDouble(w[i, j]) - stepSize);
                }
            }

            var b = biases[p];
            var gbv = gradBiases[p];
            var mb = mB[p];
            var vb = vB[p];

            for (int i = 0; i < b.Length; i++)
            {
                double grad = NumOps.ToDouble(gbv[i]);
                double mv = beta1 * NumOps.ToDouble(mb[i]) + (1 - beta1) * grad;
                double vv = beta2 * NumOps.ToDouble(vb[i]) + (1 - beta2) * grad * grad;
                mb[i] = NumOps.FromDouble(mv);
                vb[i] = NumOps.FromDouble(vv);

                double stepSize = lr * (mv / correction1) / (Math.Sqrt(vv / correction2) + eps);
                b[i] = NumOps.FromDouble(NumOps.ToDouble(b[i]) - stepSize);
            }
        }
    }

   private Vector<T>[] CloneArray(Vector<T>[] arr)
    {
        var result = new Vector<T>[arr.Length];
        for (int i = 0; i < arr.Length; i++)
        {
            var src = arr[i];
            var dst = new Vector<T>(src.Length);
            for (int j = 0; j < src.Length; j++)
            {
                dst[j] = src[j];
            }
            result[i] = dst;
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

    private (List<Matrix<T>>, List<Vector<T>>, List<List<Matrix<T>>>, List<List<Vector<T>>>, List<Matrix<T>>, List<Vector<T>>) SaveWeights()
    {
        return (
            _sharedWeights.Select(CloneMatrix).ToList(),
            _sharedBiases.Select(CloneVector).ToList(),
            _causeWeights.Select(cw => cw.Select(CloneMatrix).ToList()).ToList(),
            _causeBiases.Select(cb => cb.Select(CloneVector).ToList()).ToList(),
            _outputWeights.Select(CloneMatrix).ToList(),
            _outputBiases.Select(CloneVector).ToList()
        );
    }

    private void RestoreWeights((List<Matrix<T>>, List<Vector<T>>, List<List<Matrix<T>>>, List<List<Vector<T>>>, List<Matrix<T>>, List<Vector<T>>) weights)
    {
        _sharedWeights = weights.Item1;
        _sharedBiases = weights.Item2;
        _causeWeights = weights.Item3;
        _causeBiases = weights.Item4;
        _outputWeights = weights.Item5;
        _outputBiases = weights.Item6;
    }

    private static Matrix<T> CloneMatrix(Matrix<T> src)
    {
        var dst = new Matrix<T>(src.Rows, src.Columns);
        for (int i = 0; i < src.Rows; i++)
            for (int j = 0; j < src.Columns; j++)
                dst[i, j] = src[i, j];
        return dst;
    }

    private static Vector<T> CloneVector(Vector<T> src)
    {
        var dst = new Vector<T>(src.Length);
        for (int i = 0; i < src.Length; i++)
            dst[i] = src[i];
        return dst;
    }

    /// <inheritdoc/>
    protected override Task CalculateFeatureImportancesAsync(int featureCount)
    {
        var importances = new Vector<T>(_numFeatures);

        if (_sharedWeights.Count > 0)
        {
            var firstLayerWeights = _sharedWeights[0];
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
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NumSharedLayers", _options.NumSharedLayers },
                { "NumCauseLayers", _options.NumCauseLayers },
                { "HiddenLayerSize", _options.HiddenLayerSize },
                { "NumTimeBins", NumTimeBins },
                { "NumRisks", _options.NumRisks },
                { "Activation", _options.Activation.GetType().Name },
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
        writer.Write(_effectiveTimeBins);
        writer.Write(_options.NumSharedLayers);
        writer.Write(_options.NumCauseLayers);
        writer.Write(_options.HiddenLayerSize);
        writer.Write(_options.NumRisks);
        writer.Write(_options.Activation.GetType().AssemblyQualifiedName ?? _options.Activation.GetType().FullName ?? _options.Activation.GetType().Name);
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

        // Feature standardization, which replaces the OLS coefficient block written here before. It is
        // part of the fitted model: a restored network fed raw features would see inputs on a completely
        // different scale from the ones it was trained on.
        writer.Write(_featureMean is not null && _featureStd is not null);
        if (_featureMean is not null && _featureStd is not null)
        {
            SerializeBiases(writer, _featureMean);
            SerializeBiases(writer, _featureStd);
        }

        return ms.ToArray();
    }

    private void SerializeLayerList(BinaryWriter writer, List<Matrix<T>> weights, List<Vector<T>> biases)
    {
        writer.Write(weights.Count);
        for (int i = 0; i < weights.Count; i++)
        {
            SerializeWeights(writer, weights[i]);
            SerializeBiases(writer, biases[i]);
        }
    }

    private void SerializeWeights(BinaryWriter writer, Matrix<T> w)
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

    private void SerializeBiases(BinaryWriter writer, Vector<T> b)
    {
        writer.Write(b.Length);
        for (int i = 0; i < b.Length; i++)
        {
            writer.Write(NumOps.ToDouble(b[i]));
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
        _effectiveTimeBins = reader.ReadInt32();
        _options.NumSharedLayers = reader.ReadInt32();
        _options.NumCauseLayers = reader.ReadInt32();
        _options.HiddenLayerSize = reader.ReadInt32();
        _options.NumRisks = reader.ReadInt32();
        string activationTypeName = reader.ReadString();
        var activationType = Type.GetType(activationTypeName);
        if (activationType is not null
            && typeof(IActivationFunction<T>).IsAssignableFrom(activationType)
            && activationType.Namespace is not null
            && activationType.Namespace.StartsWith("AiDotNet.", StringComparison.Ordinal))
        {
            _options.Activation = (IActivationFunction<T>)(Activator.CreateInstance(activationType) ?? new ReLUActivation<T>());
        }
        else
        {
            _options.Activation = new ReLUActivation<T>();
        }
        _numFeatures = reader.ReadInt32();

        int timeBinLen = reader.ReadInt32();
        if (timeBinLen > 0)
        {
            _timeBinEdges = new Vector<T>(timeBinLen);
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

        // Feature standardization (see Serialize).
        if (reader.ReadBoolean())
        {
            _featureMean = DeserializeBiases(reader);
            _featureStd = DeserializeBiases(reader);
        }
        else
        {
            _featureMean = null;
            _featureStd = null;
        }
    }

    private (List<Matrix<T>>, List<Vector<T>>) DeserializeLayerList(BinaryReader reader)
    {
        int count = reader.ReadInt32();
        var weights = new List<Matrix<T>>();
        var biases = new List<Vector<T>>();

        for (int i = 0; i < count; i++)
        {
            weights.Add(DeserializeWeights(reader));
            biases.Add(DeserializeBiases(reader));
        }

        return (weights, biases);
    }

    private Matrix<T> DeserializeWeights(BinaryReader reader)
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

        return w;
    }

    private Vector<T> DeserializeBiases(BinaryReader reader)
    {
        int len = reader.ReadInt32();
        var b = new Vector<T>(len);

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

    public override IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        var clone = new DeepHit<T>(_options, Regularization);
        clone.Deserialize(Serialize());
        return clone;
    }
}
