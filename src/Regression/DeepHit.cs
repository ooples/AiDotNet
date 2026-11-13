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
///
/// // Prepare training data: 6 samples with 3 features (clinical covariates)
/// var features = new Matrix&lt;double&gt;(new double[,] { { 55, 1, 2.1 }, { 60, 0, 3.5 }, { 45, 1, 1.8 }, { 70, 0, 4.2 }, { 50, 1, 2.9 }, { 65, 0, 3.1 } });
/// var targets = new Vector&lt;double&gt;(new double[] { 12, 24, 36, 6, 18, 30 });
///
/// // Train the survival model
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new DeepHit&lt;double&gt;(options))
///     .Build(features, targets);
///
/// // Predict survival probabilities for a new patient
/// var newPatient = new Matrix&lt;double&gt;(new double[,] { { 58, 1, 2.5 } });
/// var prediction = result.Predict(newPatient);
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
public partial class DeepHit<T> : AsyncDecisionTreeRegressionBase<T>
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
    [AiDotNet.Attributes.FittedParameter]
    private List<List<Matrix<T>>> _causeWeights;

    /// <summary>
    /// Cause-specific network biases (one list per cause).
    /// </summary>
    [AiDotNet.Attributes.FittedParameter]
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
    /// Optimization is mini-batch Adam with an L2 weight penalty and early stopping, using the Epochs,
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

        int[] nonPositive = RegressionEngineOps.NonzeroIndices(
            Engine.ScalarMinusTensor(NumOps.One, Engine.TensorGreaterThan(Tensor<T>.FromVector(times), NumOps.Zero)));
        if (nonPositive.Length > 0)
        {
            int i = nonPositive[0];
            throw new ArgumentException(
                $"Survival times must be strictly positive; got {NumOps.ToDouble(times[i]):G6} at index {i}.",
                nameof(times));
        }

        var eventTensor = Tensor<T>.FromVector(events);
        var roundedEvents = Engine.TensorRound(eventTensor);
        var invalidEvents = Engine.TensorAdd(
            Engine.TensorAdd(
                Engine.TensorLessThan(eventTensor, NumOps.Zero),
                Engine.TensorGreaterThan(eventTensor, NumOps.FromDouble(_options.NumRisks))),
            Engine.TensorGreaterThan(
                Engine.TensorAbs(Engine.TensorSubtract(eventTensor, roundedEvents)), NumOps.FromDouble(1e-9)));
        int[] invalid = RegressionEngineOps.NonzeroIndices(invalidEvents);
        if (invalid.Length > 0)
        {
            int i = invalid[0];
            throw new ArgumentException(
                $"Event codes must be 0 for censored or a 1-based cause index up to NumRisks " +
                $"({_options.NumRisks}); got {NumOps.ToDouble(events[i]):G6} at index {i}.",
                nameof(events));
        }

        _numFeatures = x.Columns;
        ComputeFeatureStandardization(x);
        InitializeTimeBins(times);
        InitializeNetwork();

        var timeBins = TimeBinsOf(times);
        var eventCodes = roundedEvents.ToVector();

        await Task.Run(() => FitNetwork(x, timeBins, eventCodes));

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
    /// <returns>
    /// A [samples, NumRisks, time bins] tensor, where pmf[sample, risk, timeBin] is the probability that
    /// the event of that cause happens in that bin. Each sample's cells sum to one.
    /// </returns>
    public Tensor<T> PredictPMF(Matrix<T> input)
    {
        return PredictFlatPmf(input).Reshape(input.Rows, _options.NumRisks, NumTimeBins);
    }

    /// <summary>
    /// Predicts survival probability S(t) = P(T > t) at specified times.
    /// </summary>
    /// <param name="input">Input feature matrix.</param>
    /// <param name="times">Times at which to evaluate survival probability.</param>
    /// <returns>Matrix where [i,j] is P(T > times[j] | X_i).</returns>
    public Matrix<T> PredictSurvival(Matrix<T> input, Vector<T> times)
    {
        // S(t) = 1 - sum of the PMF over every cause up to and including t's bin, floored at zero.
        var incidence = Engine.TensorCumSum(AllCausePmf(PredictFlatPmf(input), input.Rows), 1);
        var atTimes = Engine.TensorIndexSelect(incidence, TimeBinIndicesOf(times), 1);
        return Engine.TensorMax(Engine.ScalarMinusTensor(NumOps.One, atTimes), NumOps.Zero).ToMatrix();
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

        // CIF(t, k) = sum of PMF_k up to and including t's bin.
        var causePmf = Engine.TensorSliceAxis(PredictPMF(input), 1, riskIndex);
        var incidence = Engine.TensorCumSum(causePmf, 1);
        return Engine.TensorIndexSelect(incidence, TimeBinIndicesOf(times), 1).ToMatrix();
    }

    /// <summary>
    /// Predicts expected time to event.
    /// </summary>
    /// <param name="input">Input feature matrix.</param>
    /// <returns>Vector of expected event times.</returns>
    public Vector<T> PredictExpectedTime(Matrix<T> input)
    {
        int n = input.Rows;
        var mass = AllCausePmf(PredictFlatPmf(input), n);
        var centers = TimeBinCenters();

        var weighted = Engine.TensorMatMul(mass, new Tensor<T>(new[] { centers.Length, 1 }, centers)).Reshape(n);
        var totalProb = Engine.ReduceSum(mass, new[] { 1 }, keepDims: false);

        // Normalize by total probability (may be < 1 for censored observations)
        return Engine.TensorWhere(
            Engine.TensorGreaterThan(totalProb, NumOps.Zero),
            Engine.TensorDivide(weighted, totalProb),
            Tensor<T>.CreateDefault(new[] { n }, centers[centers.Length - 1])).ToVector();
    }

    /// <summary>
    /// Predicts median survival time for each sample.
    /// </summary>
    /// <param name="input">Input feature matrix.</param>
    /// <returns>Vector of median survival times.</returns>
    /// <remarks>
    /// The median is the center of the first bin where the all-cause cumulative incidence reaches 0.5, or
    /// the last bin's center when it never does. The incidence is a running sum of probabilities, so it
    /// never decreases and "reached 0.5" flips from 0 to 1 exactly once; the difference of that flag from
    /// its own one-bin shift is therefore a one-hot pick of the crossing bin.
    /// </remarks>
    public Vector<T> PredictMedianSurvivalTime(Matrix<T> input)
    {
        int n = input.Rows;
        int bins = NumTimeBins;
        var centers = TimeBinCenters();
        var incidence = Engine.TensorCumSum(AllCausePmf(PredictFlatPmf(input), n), 1);

        var reached = Engine.ScalarMinusTensor(NumOps.One, Engine.TensorLessThan(incidence, NumOps.FromDouble(0.5)));
        var reachedBefore = bins > 1
            ? Engine.TensorConcatenate(new[]
            {
                Tensor<T>.CreateDefault(new[] { n, 1 }, NumOps.Zero),
                Engine.TensorSlice(reached, new[] { 0, 0 }, new[] { n, bins - 1 })
            }, 1)
            : Tensor<T>.CreateDefault(new[] { n, 1 }, NumOps.Zero);
        var crossing = Engine.TensorSubtract(reached, reachedBefore);

        var atCrossing = Engine.TensorMatMul(crossing, new Tensor<T>(new[] { bins, 1 }, centers)).Reshape(n);
        var neverReached = Engine.ScalarMinusTensor(
            NumOps.One, Engine.ReduceSum(crossing, new[] { 1 }, keepDims: false));
        return Engine.TensorAdd(atCrossing, Engine.TensorMultiplyScalar(neverReached, centers[bins - 1])).ToVector();
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
        // DeepHit predicts expected time - lower expected time = higher risk
        var expectedTimes = PredictExpectedTime(x);
        var (concordant, discordant) = SurvivalNetworkOps.ConcordanceCounts(
            times, events, expectedTimes, higherScoreFailsFirst: false);

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
    /// <remarks>
    /// Cases had an event by the horizon; controls survived past it. A case/control pair counts 1 when the
    /// case's predicted risk 1 - S(horizon) is higher, 1/2 when the two risks are within 1e-10, and 0
    /// otherwise.
    /// </remarks>
    public double ComputeTimeDependentAUC(Matrix<T> x, Vector<T> times, Vector<T> events, T horizon)
    {
        var horizonVec = new Vector<T>(new[] { horizon });
        var survival = Tensor<T>.FromMatrix(PredictSurvival(x, horizonVec)).Reshape(x.Rows);
        var risk = Engine.ScalarMinusTensor(NumOps.One, survival).ToVector();

        var timeTensor = Tensor<T>.FromVector(times);
        var cases = Engine.TensorMultiply(
            Engine.TensorLessEqual(timeTensor, Tensor<T>.CreateDefault(new[] { times.Length }, horizon)),
            Engine.TensorGreaterThan(Tensor<T>.FromVector(events), NumOps.Zero)).ToVector();
        var controls = Engine.TensorGreaterThan(timeTensor, horizon).ToVector();

        if (Engine.TensorCountNonzero(Tensor<T>.FromVector(cases)) == 0 ||
            Engine.TensorCountNonzero(Tensor<T>.FromVector(controls)) == 0)
        {
            return 0.5;
        }

        int n = x.Rows;
        T tieTolerance = NumOps.FromDouble(1e-10);
        T one = NumOps.One;
        int blockRows = (int)Math.Max(1L, Math.Min(n, (1L << 24) / Math.Max(1, n)));
        int concordant = 0;
        int total = 0;

        for (int start = 0; start < n; start += blockRows)
        {
            int count = Math.Min(blockRows, n - start);
            var caseRisk = SurvivalNetworkOps.BroadcastColumn(risk.GetSubVector(start, count), n);
            var controlRisk = SurvivalNetworkOps.BroadcastRow(risk, count);
            var pairs = Engine.TensorOuter(
                Tensor<T>.FromVector(cases.GetSubVector(start, count)), Tensor<T>.FromVector(controls));

            var higher = Engine.TensorGreaterThan(caseRisk, controlRisk);
            var tie = Engine.TensorMultiply(
                Engine.ScalarMinusTensor(one, higher),
                Engine.TensorLessThan(Engine.TensorAbs(Engine.TensorSubtract(caseRisk, controlRisk)), tieTolerance));

            int pairCount = Engine.TensorCountNonzero(pairs);
            int ties = Engine.TensorCountNonzero(Engine.TensorMultiply(pairs, tie));
            concordant += Engine.TensorCountNonzero(Engine.TensorMultiply(pairs, higher)) + ties;
            total += pairCount + ties;
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

        var timeTensor = Tensor<T>.FromVector(times);
        T minTime = Engine.TensorMinValue(timeTensor);
        T maxTime = Engine.TensorMaxValue(timeTensor);

        // Add small buffer
        T range = NumOps.Subtract(maxTime, minTime);
        maxTime = NumOps.Add(maxTime, NumOps.Multiply(range, NumOps.FromDouble(0.01)));

        // edge[i] = min + i * width, for i = 0..NumTimeBins.
        T binWidth = NumOps.Divide(NumOps.Subtract(maxTime, minTime), NumOps.FromDouble(NumTimeBins));
        _timeBinEdges = Engine.TensorAddScalar(
            Engine.TensorMultiplyScalar(Engine.TensorArange<T>(0, NumTimeBins + 1, 1), binWidth), minTime).ToVector();
    }

    /// <summary>
    /// The interior bin edges edge[1] .. edge[NumTimeBins - 1]: a time's bin is the number of them at or
    /// below it, so times below the grid land in bin 0 and times at or past the last edge in the last bin.
    /// </summary>
    private Vector<T>? InteriorBinEdges()
    {
        if (_timeBinEdges is null || _timeBinEdges.Length < 3)
        {
            return null;
        }

        return RegressionEngineOps2.Slice(_timeBinEdges, 1, _timeBinEdges.Length - 2);
    }

    /// <summary>
    /// The bin index of each time, for gathering bin columns.
    /// </summary>
    private Tensor<int> TimeBinIndicesOf(Vector<T> times)
    {
        var interior = InteriorBinEdges();
        if (interior is null)
        {
            return new Tensor<int>(new int[times.Length], new[] { times.Length });
        }

        return Engine.TensorSearchSorted(Tensor<T>.FromVector(interior), Tensor<T>.FromVector(times), true);
    }

    /// <summary>
    /// The bin index of each time as a value of T, for building per-subject bin masks.
    /// </summary>
    private Vector<T> TimeBinsOf(Vector<T> times)
    {
        var interior = InteriorBinEdges();
        if (interior is null)
        {
            return new Vector<T>(times.Length);
        }

        var atOrBelow = Engine.TensorLessEqual(
            SurvivalNetworkOps.BroadcastRow(interior, times.Length),
            SurvivalNetworkOps.BroadcastColumn(times, interior.Length));
        return Engine.ReduceSum(atOrBelow, new[] { 1 }, keepDims: false).ToVector();
    }

    /// <summary>
    /// The center time of every bin; the bin index itself when no grid has been fitted.
    /// </summary>
    private Vector<T> TimeBinCenters()
    {
        int bins = NumTimeBins;
        if (_timeBinEdges is null || _timeBinEdges.Length < bins + 1)
        {
            return Engine.TensorArange<T>(0, bins, 1).ToVector();
        }

        return Engine.Divide(
            Engine.Add(_timeBinEdges.GetSubVector(0, bins), RegressionEngineOps2.Slice(_timeBinEdges, 1, bins)),
            NumOps.FromDouble(2));
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
    /// Initializes a weight matrix uniformly in (-s, s), s = sqrt(2 / fan-in), drawn row-major from the
    /// model's seeded generator.
    /// </summary>
    private Matrix<T> InitializeWeights(int inputSize, int outputSize)
    {
        return SurvivalNetworkOps.UniformWeights(
            _random, inputSize, outputSize, SurvivalNetworkOps.HeScale<T>(inputSize));
    }

    /// <summary>
    /// Initializes bias vector with zeros.
    /// </summary>
    private Vector<T> InitializeBiases(int size)
    {
        return new Vector<T>(size);
    }

    /// <summary>
    /// What one layer computed on the way forward for the whole batch, kept so the backward pass can
    /// retrace it. Every tensor is [batch, width].
    /// </summary>
    private sealed class LayerCache
    {
        public LayerCache(Tensor<T> input, Tensor<T> preActivation, Tensor<T>? dropoutMask, bool hasActivation)
        {
            Input = input;
            PreActivation = preActivation;
            DropoutMask = dropoutMask;
            HasActivation = hasActivation;
        }

        public Tensor<T> Input { get; }
        public Tensor<T> PreActivation { get; }

        /// <summary>The inverted-dropout mask, or null when no unit was dropped.</summary>
        public Tensor<T>? DropoutMask { get; }
        public bool HasActivation { get; }
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
    /// The PMF of every input row over (cause, time bin) cells, flattened to [rows, NumRisks * bins] with
    /// cell index cause * bins + bin.
    /// </summary>
    private Tensor<T> PredictFlatPmf(Matrix<T> input)
    {
        var indices = Enumerable.Range(0, input.Rows).ToArray();
        var (pmf, _) = ForwardPass(input, indices, training: false);
        return pmf;
    }

    /// <summary>
    /// The PMF summed over causes, [rows, bins].
    /// </summary>
    private Tensor<T> AllCausePmf(Tensor<T> flatPmf, int rows)
    {
        return Engine.ReduceSum(
            flatPmf.Reshape(rows, _options.NumRisks, NumTimeBins), new[] { 1 }, keepDims: false);
    }

    /// <summary>
    /// Forward pass through the shared trunk and the per-cause heads.
    /// </summary>
    /// <param name="x">Feature matrix.</param>
    /// <param name="indices">Rows of <paramref name="x"/> to run.</param>
    /// <param name="training">
    /// When true, dropout is applied. When false no units are dropped, so predictions are deterministic.
    /// </param>
    /// <returns>
    /// The PMF over (cause, time bin) per sample as [n, NumRisks * bins], and the caches the backward pass
    /// needs.
    /// </returns>
    private (Tensor<T> Pmf, DeepHitForwardCache Cache) ForwardPass(Matrix<T> x, int[] indices, bool training)
    {
        int n = indices.Length;
        var cache = new DeepHitForwardCache();
        T dropoutRate = NumOps.FromDouble(_options.DropoutRate);
        T keepScale = NumOps.Divide(NumOps.One, NumOps.Subtract(NumOps.One, dropoutRate));
        var dropout = new DropoutSettings(training && _options.DropoutRate > 0, dropoutRate, keepScale);

        var current = SurvivalNetworkOps.StandardizedRows(x, indices, _numFeatures, _featureMean, _featureStd);

        // Shared trunk.
        for (int layer = 0; layer < _sharedWeights.Count; layer++)
        {
            current = ApplyLayer(current, n, _sharedWeights[layer], _sharedBiases[layer],
                applyActivation: true, dropout, cache.Shared);
        }

        var sharedOutput = current;
        var logits = new Tensor<T>[_options.NumRisks];
        var noDropout = new DropoutSettings(false, dropoutRate, keepScale);

        for (int k = 0; k < _options.NumRisks; k++)
        {
            var causeCaches = new List<LayerCache>();
            current = sharedOutput;

            for (int layer = 0; layer < _causeWeights[k].Count; layer++)
            {
                current = ApplyLayer(current, n, _causeWeights[k][layer], _causeBiases[k][layer],
                    applyActivation: true, dropout, causeCaches);
            }

            cache.Cause.Add(causeCaches);

            // Output layer produces logits, with no activation.
            var outputCaches = new List<LayerCache>();
            logits[k] = ApplyLayer(current, n, _outputWeights[k], _outputBiases[k],
                applyActivation: false, noDropout, outputCaches);
            cache.Output.Add(outputCaches[0]);
        }

        // One softmax over every (cause, time bin) cell, as DeepHit specifies.
        var allLogits = logits.Length == 1 ? logits[0] : Engine.TensorConcatenate(logits, 1);
        return (SoftmaxAcrossAll(allLogits, n), cache);
    }

    /// <summary>
    /// Whether a layer applies dropout, and with what rate and inverted-dropout scale.
    /// </summary>
    private readonly struct DropoutSettings
    {
        public DropoutSettings(bool enabled, T rate, T keepScale)
        {
            Enabled = enabled;
            Rate = rate;
            KeepScale = keepScale;
        }

        public bool Enabled { get; }
        public T Rate { get; }
        public T KeepScale { get; }
    }

    /// <summary>
    /// Applies a single fully-connected layer to the whole batch, recording what the backward pass will need.
    /// </summary>
    /// <remarks>
    /// Dropout is applied only when training. It previously ran on EVERY call, including from Predict,
    /// which made predictions random: the same model and the same input returned different answers each
    /// time. Inverted dropout also means the surviving units are scaled at training time, so inference
    /// needs no rescaling of its own.
    /// </remarks>
    private Tensor<T> ApplyLayer(Tensor<T> input, int n, Matrix<T> weights, Vector<T> biases,
        bool applyActivation, DropoutSettings dropout, List<LayerCache> caches)
    {
        var preActivation = SurvivalNetworkOps.Linear(input, n, weights, biases);
        var output = applyActivation ? _options.Activation.Activate(preActivation) : preActivation;

        Tensor<T>? mask = null;
        if (applyActivation && dropout.Enabled)
        {
            mask = SurvivalNetworkOps.DropoutMask(_random, n, weights.Columns, dropout.Rate, dropout.KeepScale);
            output = Engine.TensorMultiply(output, mask);
        }

        caches.Add(new LayerCache(input, preActivation, mask, applyActivation));
        return output;
    }

    /// <summary>
    /// Backpropagates one layer, returning its parameter gradients and the delta for its input.
    /// </summary>
    /// <param name="cache">What the layer recorded on the way forward.</param>
    /// <param name="delta">d(loss)/d(layer output), [batch, out].</param>
    /// <param name="weights">The layer's weight matrix.</param>
    /// <returns>d(loss)/d(weights), d(loss)/d(biases) and d(loss)/d(layer input).</returns>
    private (Matrix<T> WeightGradient, Vector<T> BiasGradient, Tensor<T> InputDelta) BackwardLayer(
        LayerCache cache, Tensor<T> delta, Matrix<T> weights)
    {
        // Back through dropout, then the activation.
        var d = cache.DropoutMask is null ? delta : Engine.TensorMultiply(delta, cache.DropoutMask);
        if (cache.HasActivation)
        {
            d = Engine.TensorMultiply(d, _options.Activation.Derivative(cache.PreActivation));
        }

        return SurvivalNetworkOps.LinearBackward(cache.Input, d, weights);
    }

    /// <summary>
    /// One softmax per row over every (cause, time bin) cell: exp(z - max) / (sum + 1e-10).
    /// </summary>
    private Tensor<T> SoftmaxAcrossAll(Tensor<T> logits, int n)
    {
        int cells = _options.NumRisks * NumTimeBins;
        T tiny = NumOps.FromDouble(1e-10);

        // Subtract the row max for numerical stability
        var maxLogit = Engine.ReduceMax(logits, new[] { 1 }, keepDims: false).ToVector();
        var exps = Engine.TensorExp(Engine.TensorSubtract(logits, SurvivalNetworkOps.BroadcastColumn(maxLogit, cells)));
        var denominator = Engine.TensorAddScalar(Engine.ReduceSum(exps, new[] { 1 }, keepDims: false), tiny).ToVector();
        return Engine.TensorDivide(exps, SurvivalNetworkOps.BroadcastColumn(denominator, cells));
    }

    /// <summary>
    /// The [n, causes * bins] cell mask a[i, cause] * b[i, bin], laid out like the flattened PMF.
    /// </summary>
    private Tensor<T> CellProduct(Tensor<T> perCause, Tensor<T> perBin, int n)
    {
        int causes = _options.NumRisks;
        int bins = NumTimeBins;
        var shape = new[] { n, causes, bins };
        var product = Engine.TensorMultiply(
            Engine.TensorBroadcastTo(perCause.Reshape(n, causes, 1), shape),
            Engine.TensorBroadcastTo(perBin.Reshape(n, 1, bins), shape));
        return product.Reshape(n, causes * bins);
    }

    /// <summary>
    /// Computes the DeepHit loss of one batch and its gradient with respect to the logits.
    /// </summary>
    /// <param name="pmf">The batch PMF, [n, NumRisks * bins].</param>
    /// <param name="timeBins">Each subject's time bin.</param>
    /// <param name="eventCodes">Each subject's event: 0 censored, otherwise the 1-based cause.</param>
    /// <returns>The loss, and d(loss)/d(logit) as [n, NumRisks * bins].</returns>
    /// <remarks>
    /// <para>
    /// Lee et al. (2018), eqs. 2-3. The log-likelihood term is -log P(cause k, bin s) for an observed event
    /// and -log(1 - sum over causes of F_k(s)) for a subject censored in bin s, where
    /// F_k(s) = sum of P(k, m) over m &lt;= s includes the censoring bin itself: the subject is known to
    /// survive that bin. The previous version stopped before the censoring bin, so it credited the subject
    /// with surviving one bin less than it did.
    /// </para>
    /// <para>
    /// The ranking term compares every event subject i with every subject j still event-free at i's time
    /// (s_i &lt; s_j) through eta = exp(-(F_{k_i}(s_i | x_i) - F_{k_i}(s_i | x_j)) / sigma), using the
    /// cumulative incidence of i's own cause k_i as the paper's A_{k,i,j} specifies. The previous version
    /// compared the incidence summed over every cause, which differs from the paper whenever there is
    /// more than one competing risk.
    /// </para>
    /// <para>
    /// Normalization: the reported loss is mean NLL + RankingWeight * (pair sum / n^2) -- the mean over
    /// anchors of the mean over partners, as pycox does -- and the returned gradient is exactly its
    /// derivative. The previous version reported (NLL + RankingWeight * pair sum) / n while descending a
    /// gradient whose ranking part carried 1/n of that weight, so early stopping tracked a different
    /// objective from the one being optimized.
    /// </para>
    /// <para>
    /// All gradients are with respect to the LOGITS. With one softmax over all cells,
    /// dp[a]/dz[b] = p[a](delta_ab - p[b]), so the derivative of a cell-set mass M = sum over A of p is
    /// p[b](1{b in A} - M). Every gradient here therefore sums to zero over a subject's cells, as a softmax
    /// logit gradient must.
    /// </para>
    /// </remarks>
    private (T Loss, Tensor<T> Gradients) ComputeLossAndGradients(
        Tensor<T> pmf, Vector<T> timeBins, Vector<T> eventCodes)
    {
        int n = timeBins.Length;
        int causes = _options.NumRisks;
        int bins = NumTimeBins;
        int cells = causes * bins;
        T tiny = NumOps.FromDouble(1e-10);
        T batchSize = NumOps.FromDouble(n);

        // Per-subject masks: one-hot of the subject's bin, bins at or before it, one-hot of its cause.
        var binColumn = SurvivalNetworkOps.BroadcastColumn(timeBins, bins);
        var binPositions = SurvivalNetworkOps.BroadcastRow(Engine.TensorArange<T>(0, bins, 1).ToVector(), n);
        var ownBin = Engine.TensorEquals(binPositions, binColumn);
        var atOrBeforeBin = Engine.TensorLessEqual(binPositions, binColumn);
        var causeOneHot = Engine.TensorEquals(
            SurvivalNetworkOps.BroadcastRow(Engine.TensorArange<T>(1, causes + 1, 1).ToVector(), n),
            SurvivalNetworkOps.BroadcastColumn(eventCodes, causes));
        var isEvent = Engine.ReduceSum(causeOneHot, new[] { 1 }, keepDims: false).ToVector();
        var censored = Engine.Subtract(Engine.Fill(n, NumOps.One), isEvent);

        var target = CellProduct(causeOneHot, ownBin, n);
        var ownCauseToBin = CellProduct(causeOneHot, atOrBeforeBin, n);
        var allCausesToBin = CellProduct(Tensor<T>.CreateDefault(new[] { n, causes }, NumOps.One), atOrBeforeBin, n);

        // Log-likelihood term.
        var eventProbability = Engine.ReduceSum(Engine.TensorMultiply(pmf, target), new[] { 1 }, keepDims: false).ToVector();
        var incidenceToBin = Engine.ReduceSum(Engine.TensorMultiply(pmf, allCausesToBin), new[] { 1 }, keepDims: false).ToVector();
        var survival = Engine.Add(Engine.Subtract(Engine.Fill(n, NumOps.One), incidenceToBin), Engine.Fill(n, tiny));
        var eventNll = Engine.Negate(Engine.Log(Engine.Add(eventProbability, Engine.Fill(n, tiny))));
        var censoredNll = Engine.Negate(Engine.Log(survival));
        T logLikeLoss = Engine.Sum(Engine.Add(Engine.Multiply(isEvent, eventNll), Engine.Multiply(censored, censoredNll)));

        // Event: softmax cross-entropy, p - onehot. Censored: L = -log(1 - C + tiny), so
        // dL/dz[b] = p[b] * (1{b at or before the bin} - C) / (1 - C + tiny).
        var eventGradient = Engine.TensorSubtract(pmf, target);
        var censoredGradient = Engine.TensorMultiply(pmf, Engine.TensorDivide(
            Engine.TensorSubtract(allCausesToBin, SurvivalNetworkOps.BroadcastColumn(incidenceToBin, cells)),
            SurvivalNetworkOps.BroadcastColumn(survival, cells)));
        var gradients = Engine.TensorAdd(
            Engine.TensorMultiply(eventGradient, SurvivalNetworkOps.BroadcastColumn(isEvent, cells)),
            Engine.TensorMultiply(censoredGradient, SurvivalNetworkOps.BroadcastColumn(censored, cells)));

        // Ranking term.
        T rankingLoss = NumOps.Zero;
        T rankWeight = NumOps.FromDouble(_options.RankingWeight);
        if (_options.RankingWeight > 0)
        {
            T sigma = NumOps.FromDouble(_options.RankingSigma);

            // Cause-specific cumulative incidence per cell, then F_{k_i}(s_i) for i itself and for every j.
            var incidence = Engine.TensorCumSum(pmf.Reshape(n, causes, bins), 2).Reshape(n, cells);
            var ownIncidence = Engine.ReduceSum(
                Engine.TensorMultiply(target, incidence), new[] { 1 }, keepDims: false).ToVector();
            var partnerIncidence = Engine.TensorMatMul(target, Engine.TensorTranspose(incidence));

            // A[i, j] = 1 when i had an event and s_i < s_j.
            var comparable = Engine.TensorMultiply(
                SurvivalNetworkOps.BroadcastColumn(isEvent, n),
                Engine.TensorLessThan(
                    SurvivalNetworkOps.BroadcastColumn(timeBins, n), SurvivalNetworkOps.BroadcastRow(timeBins, n)));

            // DeepHit eq. 8: eta = exp(-(F_i(s_i) - F_j(s_i)) / sigma). Subject i failed first, so its
            // cumulative incidence at that time SHOULD exceed j's, and the penalty shrinks as it does.
            var eta = Engine.TensorExp(Engine.TensorNegate(Engine.TensorDivideScalar(
                Engine.TensorSubtract(SurvivalNetworkOps.BroadcastColumn(ownIncidence, n), partnerIncidence), sigma)));
            var pairPenalty = Engine.TensorMultiply(comparable, eta);
            rankingLoss = Engine.TensorSum(pairPenalty);

            // d(eta)/dz_i[b] = (-eta/sigma) p_i[b] (1{b in A_i} - F_i) and, for the partner,
            // d(eta)/dz_j[b] = (eta/sigma) p_j[b] (1{b in A_i} - F_j), with A_i the cells of cause k_i at or
            // before s_i. Summed over partners (rows) and over anchors (columns) respectively.
            var pairWeight = Engine.TensorMultiplyScalar(pairPenalty, NumOps.Divide(rankWeight, NumOps.Multiply(sigma, batchSize)));
            var anchorWeight = Engine.ReduceSum(pairWeight, new[] { 1 }, keepDims: false).ToVector();
            var anchorGradient = Engine.TensorNegate(Engine.TensorMultiply(
                SurvivalNetworkOps.BroadcastColumn(anchorWeight, cells),
                Engine.TensorMultiply(pmf, Engine.TensorSubtract(
                    ownCauseToBin, SurvivalNetworkOps.BroadcastColumn(ownIncidence, cells)))));

            var partnerCells = Engine.TensorMatMul(Engine.TensorTranspose(pairWeight), ownCauseToBin);
            var partnerOffset = Engine.ReduceSum(
                Engine.TensorMultiply(pairWeight, partnerIncidence), new[] { 0 }, keepDims: false).ToVector();
            var partnerGradient = Engine.TensorMultiply(pmf, Engine.TensorSubtract(
                partnerCells, SurvivalNetworkOps.BroadcastColumn(partnerOffset, cells)));

            gradients = Engine.TensorAdd(gradients, Engine.TensorAdd(anchorGradient, partnerGradient));
        }

        // loss = mean NLL + rankWeight * pairSum / n^2; the gradient above is n times its derivative.
        T totalLoss = NumOps.Divide(
            NumOps.Add(logLikeLoss, NumOps.Divide(NumOps.Multiply(rankWeight, rankingLoss), batchSize)),
            batchSize);
        return (totalLoss, Engine.TensorDivideScalar(gradients, batchSize));
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
        var (mean, std) = SurvivalNetworkOps.ColumnMeanAndStd(x);
        _featureMean = mean;
        _featureStd = std;
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

        weights.AddRange(_sharedWeights);
        biases.AddRange(_sharedBiases);

        for (int k = 0; k < _options.NumRisks; k++)
        {
            weights.AddRange(_causeWeights[k]);
            biases.AddRange(_causeBiases[k]);
        }

        weights.AddRange(_outputWeights);
        biases.AddRange(_outputBiases);

        return (weights, biases);
    }

    /// <summary>
    /// Allocates zeroed accumulators shaped like the parameters.
    /// </summary>
    private (List<Matrix<T>> Weights, List<Vector<T>> Biases) AllocateLike(
        List<Matrix<T>> weights, List<Vector<T>> biases)
    {
        var gw = weights.Select(w => new Matrix<T>(w.Rows, w.Columns)).ToList();
        var gb = biases.Select(b => new Vector<T>(b.Length)).ToList();
        return (gw, gb);
    }

    /// <summary>
    /// Fits the network by minimizing the DeepHit loss: the log-likelihood term plus the ranking term.
    /// </summary>
    /// <param name="x">Feature matrix.</param>
    /// <param name="timeBins">Discretized event time per subject.</param>
    /// <param name="eventCodes">Event type per subject: 0 for censored, 1..NumRisks for a cause.</param>
    /// <remarks>
    /// Mini-batch Adam with the L2 penalty on the weight gradients, and early stopping. The best epoch's
    /// weights are kept, not the last epoch's.
    /// </remarks>
    private void FitNetwork(Matrix<T> x, Vector<T> timeBins, Vector<T> eventCodes)
    {
        int n = x.Rows;
        int batchSize = Math.Max(1, Math.Min(_options.BatchSize, n));
        var (weights, biases) = FlattenParameters();
        var (mW, mB) = AllocateLike(weights, biases);
        var (vW, vB) = AllocateLike(weights, biases);
        int adamStep = 0;
        var binTensor = Tensor<T>.FromVector(timeBins);
        var codeTensor = Tensor<T>.FromVector(eventCodes);

        T bestLoss = NumOps.MaxValue;
        int epochsWithoutImprovement = 0;
        var bestWeights = SaveWeights();

        var order = Enumerable.Range(0, n).ToArray();

        for (int epoch = 0; epoch < _options.Epochs; epoch++)
        {
            order = ShuffleArray(order);
            var batchLosses = new List<T>();

            for (int start = 0; start < n; start += batchSize)
            {
                int count = Math.Min(batchSize, n - start);
                var batchIndices = new int[count];
                Array.Copy(order, start, batchIndices, 0, count);
                var selector = new Tensor<int>(batchIndices, new[] { count });

                var (pmf, cache) = ForwardPass(x, batchIndices, training: true);
                var (loss, logitGradients) = ComputeLossAndGradients(
                    pmf,
                    Engine.TensorIndexSelect(binTensor, selector, 0).ToVector(),
                    Engine.TensorIndexSelect(codeTensor, selector, 0).ToVector());

                var (gradW, gradB) = Backpropagate(cache, logitGradients, count, weights.Count);

                adamStep++;
                ApplyAdam(weights, biases, gradW, gradB, mW, mB, vW, vB, adamStep);

                batchLosses.Add(loss);
            }

            if (batchLosses.Count == 0)
            {
                continue;
            }

            T meanLoss = Engine.Mean(new Vector<T>(batchLosses));

            if (IsImprovement(meanLoss, bestLoss))
            {
                bestLoss = meanLoss;
                epochsWithoutImprovement = 0;
                bestWeights = SaveWeights(bestWeights);
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
    /// Whether an epoch's mean loss beats the best so far by more than 1e-7.
    /// </summary>
    private bool IsImprovement(T meanLoss, T bestLoss)
    {
        return NumOps.LessThan(meanLoss, NumOps.Subtract(bestLoss, NumOps.FromDouble(1e-7)));
    }

    /// <summary>
    /// Propagates the logit gradients back through the output heads, the cause branches and the shared trunk.
    /// </summary>
    /// <param name="cache">Caches recorded by the forward pass.</param>
    /// <param name="logitGradients">d(loss)/d(logit), [n, NumRisks * bins].</param>
    /// <param name="n">Batch size.</param>
    /// <param name="parameterCount">Number of layers in <see cref="FlattenParameters"/> order.</param>
    /// <returns>Weight and bias gradients in <see cref="FlattenParameters"/> order.</returns>
    /// <remarks>
    /// The shared trunk feeds every cause branch, so its delta is the SUM of what each branch sends back.
    /// Overwriting rather than summing there would train the trunk on one cause only.
    /// </remarks>
    private (List<Matrix<T>> Weights, List<Vector<T>> Biases) Backpropagate(
        DeepHitForwardCache cache, Tensor<T> logitGradients, int n, int parameterCount)
    {
        int sharedCount = _sharedWeights.Count;
        int causeLayersPerRisk = _options.NumRisks > 0 ? _causeWeights[0].Count : 0;
        int bins = NumTimeBins;
        var weightGradients = new Matrix<T>[parameterCount];
        var biasGradients = new Vector<T>[parameterCount];

        // Delta arriving at the shared trunk's output, summed over causes.
        Tensor<T>? sharedDelta = null;

        for (int k = 0; k < _options.NumRisks; k++)
        {
            // d(loss)/d(logit) for this cause.
            var delta = Engine.TensorSlice(logitGradients, new[] { 0, k * bins }, new[] { n, bins });

            int outputParamIndex = sharedCount + _options.NumRisks * causeLayersPerRisk + k;
            var (outputWeightGradient, outputBiasGradient, outputDelta) =
                BackwardLayer(cache.Output[k], delta, _outputWeights[k]);
            weightGradients[outputParamIndex] = outputWeightGradient;
            biasGradients[outputParamIndex] = outputBiasGradient;
            delta = outputDelta;

            for (int layer = causeLayersPerRisk - 1; layer >= 0; layer--)
            {
                int paramIndex = sharedCount + k * causeLayersPerRisk + layer;
                var (weightGradient, biasGradient, inputDelta) =
                    BackwardLayer(cache.Cause[k][layer], delta, _causeWeights[k][layer]);
                weightGradients[paramIndex] = weightGradient;
                biasGradients[paramIndex] = biasGradient;
                delta = inputDelta;
            }

            sharedDelta = sharedDelta is null ? delta : Engine.TensorAdd(sharedDelta, delta);
        }

        if (sharedDelta is not null)
        {
            var trunkDelta = sharedDelta;
            for (int layer = sharedCount - 1; layer >= 0; layer--)
            {
                var (weightGradient, biasGradient, inputDelta) =
                    BackwardLayer(cache.Shared[layer], trunkDelta, _sharedWeights[layer]);
                weightGradients[layer] = weightGradient;
                biasGradients[layer] = biasGradient;
                trunkDelta = inputDelta;
            }
        }

        return (weightGradients.ToList(), biasGradients.ToList());
    }

    /// <summary>
    /// Applies one Adam step to every parameter, in place.
    /// </summary>
    /// <remarks>
    /// The L2 penalty applies to the weight matrices only; biases are left unpenalized, which is the
    /// standard convention because shrinking them removes the layer's ability to shift its output.
    /// </remarks>
    private void ApplyAdam(List<Matrix<T>> weights, List<Vector<T>> biases,
        List<Matrix<T>> gradWeights, List<Vector<T>> gradBiases,
        List<Matrix<T>> mW, List<Vector<T>> mB, List<Matrix<T>> vW, List<Vector<T>> vB, int step)
    {
        var coefficients = new SurvivalNetworkOps.AdamCoefficients<T>(
            _options.LearningRate, _options.L2Regularization, step);

        for (int p = 0; p < weights.Count; p++)
        {
            SurvivalNetworkOps.AdamStep(weights[p], gradWeights[p], mW[p], vW[p], coefficients, applyL2: true);
            SurvivalNetworkOps.AdamStep(biases[p], gradBiases[p], mB[p], vB[p], coefficients, applyL2: false);
        }
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

    private (List<Matrix<T>>, List<Vector<T>>, List<List<Matrix<T>>>, List<List<Vector<T>>>, List<Matrix<T>>, List<Vector<T>>) SaveWeights((List<Matrix<T>>, List<Vector<T>>, List<List<Matrix<T>>>, List<List<Vector<T>>>, List<Matrix<T>>, List<Vector<T>>)? snapshot = null)
    {
        return (
            AiDotNet.Training.HostStateCheckpoint<T>.Capture(_sharedWeights, snapshot?.Item1),
            AiDotNet.Training.HostStateCheckpoint<T>.Capture(_sharedBiases, snapshot?.Item2),
            _causeWeights.Select((cw, i) => AiDotNet.Training.HostStateCheckpoint<T>.Capture(cw, snapshot?.Item3[i])).ToList(),
            _causeBiases.Select((cb, i) => AiDotNet.Training.HostStateCheckpoint<T>.Capture(cb, snapshot?.Item4[i])).ToList(),
            AiDotNet.Training.HostStateCheckpoint<T>.Capture(_outputWeights, snapshot?.Item5),
            AiDotNet.Training.HostStateCheckpoint<T>.Capture(_outputBiases, snapshot?.Item6)
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

    /// <inheritdoc/>
    protected override Task CalculateFeatureImportancesAsync(int featureCount)
    {
        FeatureImportances = SurvivalNetworkOps.FirstLayerImportances(
            _sharedWeights.Count > 0 ? _sharedWeights[0] : null, _numFeatures);
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
}
