using AiDotNet.Attributes;
using AiDotNet.Enums;
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
/// <example>
/// <code>
/// // Create a DeepSurv model for Cox proportional hazards survival analysis
/// var options = new DeepSurvOptions&lt;double&gt;();
///
/// // Prepare training data: 6 samples with 3 clinical features
/// var features = new Matrix&lt;double&gt;(new double[,] { { 55, 1, 2.1 }, { 60, 0, 3.5 }, { 45, 1, 1.8 }, { 70, 0, 4.2 }, { 50, 1, 2.9 }, { 65, 0, 3.1 } });
/// var targets = new Vector&lt;double&gt;(new double[] { 12, 24, 36, 6, 18, 30 });
///
/// // Train the neural network for Cox regression
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new DeepSurv&lt;double&gt;(options))
///     .Build(features, targets);
///
/// // Predict risk scores for a new patient
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
[ResearchPaper("DeepSurv: Personalized Treatment Recommender System Using a Cox Proportional Hazards Deep Neural Network", "https://doi.org/10.1186/s12874-018-0482-1", Year = 2018, Authors = "Jared L. Katzman, Uri Shaham, Alexander Cloninger, Jonathan Bates, Tingting Jiang, Yuval Kluger")]
public partial class DeepSurv<T> : AsyncDecisionTreeRegressionBase<T>
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
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T>? _baselineHazardTimes;

    /// <summary>
    /// Baseline cumulative hazard function values.
    /// </summary>
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T>? _baselineHazardValues;

    /// <summary>
    /// Configuration options.
    /// </summary>
    private readonly DeepSurvOptions<T> _options;

    /// <summary>
    /// Random number generator.
    /// </summary>
    private readonly Random _random;
    private bool _useOLS;
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T>? _olsCoefficients;

    /// <summary>
    /// Per-layer batch-normalization scale, one entry per hidden layer. Empty when
    /// <see cref="DeepSurvOptions{T}.UseBatchNormalization"/> is false.
    /// </summary>
    private List<Vector<T>> _bnGamma = new List<Vector<T>>();

    /// <summary>
    /// Per-layer batch-normalization shift, one entry per hidden layer.
    /// </summary>
    private List<Vector<T>> _bnBeta = new List<Vector<T>>();

    /// <summary>
    /// Running mean per hidden layer, accumulated during training and used at inference time.
    /// </summary>
    private List<Vector<T>> _bnRunningMean = new List<Vector<T>>();

    /// <summary>
    /// Running variance per hidden layer, accumulated during training and used at inference time.
    /// </summary>
    private List<Vector<T>> _bnRunningVariance = new List<Vector<T>>();

    /// <summary>
    /// The largest event time seen during training, used to bound the survival-time integration.
    /// </summary>
    private T _maxObservedTime;

    /// <summary>
    /// Per-feature training mean, used to standardize inputs before the first layer.
    /// </summary>
    private Vector<T>? _featureMean;

    /// <summary>
    /// Per-feature training standard deviation, used to standardize inputs before the first layer.
    /// </summary>
    private Vector<T>? _featureStd;

    /// <summary>
    /// Largest magnitude a risk score is allowed to reach before being exponentiated.
    /// </summary>
    /// <remarks>
    /// The risk score is an unbounded linear output, and every use of it in this model goes through
    /// <c>exp</c>: the partial likelihood, the Breslow baseline hazard, and the survival function. A score
    /// of 750 overflows a double, which turns the cumulative hazard into infinity and then, once it is
    /// subtracted from itself in the Breslow loop, into NaN. Clamping at the point of exponentiation keeps
    /// every one of those finite. exp(20) is about 4.9e8, far beyond any meaningful hazard ratio.
    /// </remarks>
    private const double RiskScoreClamp = 20.0;



    /// <inheritdoc/>
    public override int NumberOfTrees => 1;

    /// <summary>
    /// Initializes a new instance of DeepSurv.
    /// </summary>
    /// <param name="options">Configuration options.</param>
    /// <param name="regularization">Optional regularization.</param>
    public DeepSurv(DeepSurvOptions<T>? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(null, regularization)
    {
        _maxObservedTime = NumOps.One;
        _options = options ?? new DeepSurvOptions<T>();
        _weights = [];
        _biases = [];
        _numFeatures = 0;
        _random = _options.Seed.HasValue ? RandomHelper.CreateSeededRandom(_options.Seed.Value) : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Trains the network on observed times, treating every observation as an event (uncensored).
    /// </summary>
    /// <param name="x">Feature matrix, one row per subject.</param>
    /// <param name="y">Observed time to event for each subject. Must be positive.</param>
    /// <remarks>
    /// <para>
    /// The general regression interface carries a single target vector, which cannot express censoring.
    /// This overload therefore treats every observation as an observed event. Use
    /// <see cref="TrainAsync(Matrix{T}, Vector{T}, Vector{T})"/> when some subjects are censored --
    /// which, for real survival data, is the usual case.
    /// </para>
    /// <para><b>For Beginners:</b> Censoring means you stopped watching a subject before the event
    /// happened, so all you know is that it survived at least that long. This method assumes you saw the
    /// event for everyone. If some of your subjects were censored, use the three-argument version and pass
    /// a 1 for observed events and a 0 for censored ones.
    /// </para>
    /// </remarks>
    public override async Task TrainAsync(Matrix<T> x, Vector<T> y)
    {
        var events = Vector<T>.CreateDefault(y.Length, NumOps.One);
        await TrainAsync(x, y, events);
    }

    /// <summary>
    /// Trains the network on right-censored survival data by maximizing the Cox partial likelihood.
    /// </summary>
    /// <param name="x">Feature matrix, one row per subject.</param>
    /// <param name="times">Observed time for each subject: the event time, or the censoring time.</param>
    /// <param name="events">1 when the event was observed, 0 when the subject was censored.</param>
    /// <remarks>
    /// <para>
    /// This is the training procedure from Katzman et al. (2018): a fully-connected network produces a
    /// scalar risk score per subject, and the network is fitted by minimizing the negative Cox partial
    /// log-likelihood. Optimization is mini-batch Adam with L2 weight decay and early stopping, using the
    /// epoch, batch size, learning rate, regularization and patience settings already present on
    /// <see cref="DeepSurvOptions{T}"/> -- every one of which was previously ignored, because this method
    /// fitted ordinary least squares and returned before the network was ever used.
    /// </para>
    /// <para>
    /// The risk set for each mini-batch is the batch itself, the standard mini-batch approximation to the
    /// partial likelihood; with the default batch size of 32 and a smaller training set, the batch is the
    /// full sample and the approximation is exact.
    /// </para>
    /// </remarks>
    public async Task TrainAsync(Matrix<T> x, Vector<T> times, Vector<T> events)
    {
        ValidationHelper<T>.ValidateInputData(x, times);

        if (events.Length != times.Length)
        {
            throw new ArgumentException(
                $"The event indicator vector must have one entry per subject: got {events.Length} " +
                $"indicators for {times.Length} observed times.",
                nameof(events));
        }

        var timeTensor = Tensor<T>.FromVector(times);
        int[] nonPositive = RegressionEngineOps.NonzeroIndices(
            Engine.ScalarMinusTensor(NumOps.One, Engine.TensorGreaterThan(timeTensor, NumOps.Zero)));
        if (nonPositive.Length > 0)
        {
            int i = nonPositive[0];
            throw new ArgumentException(
                $"Survival times must be strictly positive; got {NumOps.ToDouble(times[i]):G6} at index {i}.",
                nameof(times));
        }

        if (Engine.TensorCountNonzero(EventIndicator(events)) == 0)
        {
            throw new ArgumentException(
                "The Cox partial likelihood is defined by the observed events, but every subject is " +
                "censored, so there is nothing to fit. Pass 1 in the event vector for subjects whose " +
                "event was observed.",
                nameof(events));
        }

        _numFeatures = x.Columns;
        ComputeFeatureStandardization(x);
        InitializeNetwork();

        _maxObservedTime = Engine.TensorMaxValue(timeTensor);

        await Task.Run(() => FitNetwork(x, times, events));

        ComputeBaselineHazard(x, times, events);

        await CalculateFeatureImportancesAsync(x.Columns);
    }

    /// <summary>
    /// Predicts the expected survival time for each subject.
    /// </summary>
    /// <param name="input">Feature matrix, one row per subject.</param>
    /// <returns>Expected survival time for each subject, on the same scale as the training times.</returns>
    /// <remarks>
    /// <para>
    /// The model was trained against observed times, so the general regression interface returns a time,
    /// keeping <c>Predict</c> on the same scale as the <c>y</c> that was passed to <c>Train</c>. The
    /// expectation is obtained by integrating the predicted survival curve, which is what lifelines'
    /// <c>predict_expectation</c> reports; <see cref="PredictRiskScores"/> exposes the network's raw
    /// linear predictor, and <see cref="PredictMedianSurvivalTime"/> the median, for callers who want those.
    /// </para>
    /// <para>
    /// The expectation is preferred over the median here because the median is undefined whenever the
    /// predicted survival curve never falls to 0.5 within the observed follow-up -- common for low-risk
    /// subjects -- whereas the integral is always finite.
    /// </para>
    /// </remarks>
    public override async Task<Vector<T>> PredictAsync(Matrix<T> input)
    {
        return await Task.Run(() => PredictExpectedSurvivalTime(input));
    }

    /// <summary>
    /// Predicts risk scores for input samples.
    /// </summary>
    /// <param name="input">Input feature matrix.</param>
    /// <returns>Vector of risk scores (higher = higher risk).</returns>
    public Vector<T> PredictRiskScores(Matrix<T> input)
    {
        var indices = Enumerable.Range(0, input.Rows).ToArray();
        var (riskScores, _) = ForwardPass(input, indices, training: false);
        return riskScores;
    }

    /// <summary>
    /// Predicts survival probability at specified times.
    /// </summary>
    /// <param name="input">Input feature matrix.</param>
    /// <param name="times">Times at which to evaluate survival probability.</param>
    /// <returns>Matrix where [i,j] is P(T > times[j] | X_i).</returns>
    /// <remarks>
    /// S(t | x) = exp(-H0(t) * exp(r(x))), with H0 the Breslow step function, which is right-continuous:
    /// at an event time it already includes that time's increment. Without a fitted baseline the
    /// exponential model S(t) = exp(-exp(r) * t) is used.
    /// </remarks>
    public Matrix<T> PredictSurvival(Matrix<T> input, Vector<T> times)
    {
        var expRisk = ClampedExpRisk(PredictRiskScores(input));
        var hazard = _baselineHazardTimes is null || _baselineHazardValues is null
            ? times
            : BaselineHazardAt(times);

        return Engine.TensorExp(Engine.TensorNegate(
            Engine.TensorOuter(Tensor<T>.FromVector(expRisk), Tensor<T>.FromVector(hazard)))).ToMatrix();
    }

    /// <summary>
    /// Predicts median survival time for each sample.
    /// </summary>
    /// <param name="input">Input feature matrix.</param>
    /// <returns>Vector of median survival times.</returns>
    public Vector<T> PredictMedianSurvivalTime(Matrix<T> input)
    {
        var expRisk = ClampedExpRisk(PredictRiskScores(input));

        // S = exp(-H0 * exp(r)) = 0.5  =>  H0 = ln(2) / exp(r); the exponential model's median is the same.
        T ln2 = NumOps.Log(NumOps.FromDouble(2.0));
        var targetHazard = Engine.Divide(Engine.Fill(expRisk.Length, ln2), expRisk);

        if (_baselineHazardTimes is not null && _baselineHazardValues is not null && _baselineHazardTimes.Length > 0)
        {
            return FindTimesForHazard(targetHazard);
        }

        return targetHazard;
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
        var (concordant, discordant) = SurvivalNetworkOps.ConcordanceCounts(
            times, events, riskScores, higherScoreFailsFirst: true);

        int total = concordant + discordant;
        return total > 0 ? (double)concordant / total : 0.5;
    }

    /// <summary>
    /// Initializes the neural network weights.
    /// </summary>
    /// <remarks>
    /// Weights are drawn uniformly from (-s, s) with s = sqrt(2 / fan-in), layer by layer in row-major
    /// order from the model's seeded generator; biases start at zero.
    /// </remarks>
    private void InitializeNetwork()
    {
        _weights = [];
        _biases = [];
        _bnGamma = new List<Vector<T>>();
        _bnBeta = new List<Vector<T>>();
        _bnRunningMean = new List<Vector<T>>();
        _bnRunningVariance = new List<Vector<T>>();

        int inputSize = _numFeatures;
        T one = NumOps.One;

        for (int layer = 0; layer < _options.NumHiddenLayers; layer++)
        {
            int outputSize = _options.HiddenLayerSize;

            _weights.Add(SurvivalNetworkOps.UniformWeights(
                _random, inputSize, outputSize, SurvivalNetworkOps.HeScale<T>(inputSize)));
            _biases.Add(new Vector<T>(outputSize));

            // Batch-normalization parameters start at the identity transform (gamma = 1, beta = 0), so an
            // untrained normalized layer behaves like the un-normalized one.
            _bnGamma.Add(Vector<T>.CreateDefault(outputSize, one));
            _bnBeta.Add(new Vector<T>(outputSize));
            _bnRunningMean.Add(new Vector<T>(outputSize));
            _bnRunningVariance.Add(Vector<T>.CreateDefault(outputSize, one));

            inputSize = outputSize;
        }

        // Output layer (single risk score)
        _weights.Add(SurvivalNetworkOps.UniformWeights(
            _random, inputSize, 1, SurvivalNetworkOps.HeScale<T>(inputSize)));
        _biases.Add(new Vector<T>(1));
    }

    /// <summary>
    /// Records the per-feature mean and standard deviation of the training inputs.
    /// </summary>
    /// <param name="x">Training feature matrix.</param>
    /// <remarks>
    /// <para>
    /// Katzman et al. standardize the covariates before fitting, and it is not cosmetic here. Raw features
    /// on the order of 10, fanned into a linear layer, produce risk scores large enough that exp overflows
    /// in the Breslow baseline hazard; once the cumulative hazard is infinite, subtracting a subject out of
    /// the risk set makes it NaN and every prediction downstream is lost. Standardizing puts the linear
    /// predictor in a range where the network trains stably and the hazard stays finite.
    /// </para>
    /// <para>
    /// A feature with zero variance gets a standard deviation of 1, so it maps to a constant 0 rather than
    /// dividing by zero.
    /// </para>
    /// </remarks>
    private void ComputeFeatureStandardization(Matrix<T> x)
    {
        var (mean, std) = SurvivalNetworkOps.ColumnMeanAndStd(x);
        _featureMean = mean;
        _featureStd = std;
    }

    /// <summary>
    /// Exponentiates risk scores after clamping them into a range where the result stays finite.
    /// </summary>
    /// <param name="riskScores">The raw linear predictors.</param>
    /// <returns>exp of the clamped risk scores; a NaN score is treated as 0.</returns>
    /// <remarks>
    /// See <see cref="RiskScoreClamp"/>. Every exponentiation of a risk score in this class goes through
    /// here, so the loss, the baseline hazard and the survival function all agree on the same bound --
    /// they previously did not, and the baseline hazard was the one that overflowed.
    /// </remarks>
    private Vector<T> ClampedExpRisk(Vector<T> riskScores)
    {
        return Engine.TensorExp(ClampRisk(riskScores)).ToVector();
    }

    /// <summary>
    /// Clamps risk scores to [-RiskScoreClamp, RiskScoreClamp], mapping NaN to 0.
    /// </summary>
    private Tensor<T> ClampRisk(Vector<T> riskScores)
    {
        var scores = Tensor<T>.FromVector(riskScores);
        var finite = Engine.TensorWhere(
            Engine.TensorIsNan(scores), Tensor<T>.CreateDefault(new[] { riskScores.Length }, NumOps.Zero), scores);
        T bound = NumOps.FromDouble(RiskScoreClamp);
        return Engine.TensorClamp(finite, NumOps.Negate(bound), bound);
    }

    /// <summary>
    /// A 0/1 tensor that is 1 where the event indicator equals 1 (an observed event).
    /// </summary>
    private Tensor<T> EventIndicator(Vector<T> events)
    {
        return Engine.TensorEquals(Tensor<T>.FromVector(events), NumOps.One);
    }

    /// <summary>
    /// Everything the backward pass needs from one forward pass, one [n, width] tensor per layer.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Working out how to improve the network requires remembering what it computed
    /// on the way forward, not just the final answer. This carries those intermediate values back.
    /// </remarks>
    private sealed class ForwardCache
    {
        /// <summary>Input to each layer; entry l feeds layer l. Length is layer count.</summary>
        public List<Tensor<T>> Inputs { get; } = new List<Tensor<T>>();

        /// <summary>Pre-activation of each hidden layer, before batch norm and activation.</summary>
        public List<Tensor<T>> PreActivations { get; } = new List<Tensor<T>>();

        /// <summary>Batch-normalized pre-activation of each hidden layer, before the activation.</summary>
        public List<Tensor<T>> NormalizedPreActivations { get; } = new List<Tensor<T>>();

        /// <summary>Per-hidden-layer batch mean, empty when batch norm is off.</summary>
        public List<Vector<T>> BatchMean { get; } = new List<Vector<T>>();

        /// <summary>Per-hidden-layer batch variance, empty when batch norm is off.</summary>
        public List<Vector<T>> BatchVariance { get; } = new List<Vector<T>>();

        /// <summary>
        /// Dropout mask per hidden layer: 0 for a dropped unit, the inverted-dropout scale otherwise; null
        /// when no unit was dropped.
        /// </summary>
        public List<Tensor<T>?> DropoutMasks { get; } = new List<Tensor<T>?>();
    }

    /// <summary>
    /// Forward pass through the network.
    /// </summary>
    /// <param name="x">Feature matrix.</param>
    /// <param name="indices">Rows of <paramref name="x"/> to run, in order.</param>
    /// <param name="training">
    /// When true, batch norm uses the batch statistics and updates its running estimates, and dropout is
    /// applied. When false, the running estimates are used and no units are dropped.
    /// </param>
    /// <returns>The risk score per row, and the intermediate values the backward pass needs.</returns>
    private (Vector<T> RiskScores, ForwardCache Cache) ForwardPass(Matrix<T> x, int[] indices, bool training)
    {
        int n = indices.Length;
        var cache = new ForwardCache();
        bool useBatchNorm = _options.UseBatchNormalization && n > 1;
        T epsilon = NumOps.FromDouble(1e-5);
        T momentum = NumOps.FromDouble(0.9);
        T oneMinusMomentum = NumOps.FromDouble(0.1);
        T dropoutRate = NumOps.FromDouble(_options.DropoutRate);
        T keepScale = NumOps.Divide(NumOps.One, NumOps.Subtract(NumOps.One, dropoutRate));

        var current = SurvivalNetworkOps.StandardizedRows(x, indices, _numFeatures, _featureMean, _featureStd);

        // Hidden layers: linear -> batch norm -> activation -> dropout.
        for (int layer = 0; layer < _weights.Count - 1; layer++)
        {
            int outputSize = _weights[layer].Columns;
            var preActivation = SurvivalNetworkOps.Linear(current, n, _weights[layer], _biases[layer]);

            // Batch normalization (Ioffe & Szegedy 2015) over the batch dimension.
            var normalized = preActivation;
            if (useBatchNorm)
            {
                var mean = Engine.ReduceMean(preActivation, new[] { 0 }, keepDims: false).ToVector();
                var centered = Engine.TensorSubtract(preActivation, SurvivalNetworkOps.BroadcastRow(mean, n));
                var variance = Engine.ReduceMean(Engine.TensorSquare(centered), new[] { 0 }, keepDims: false).ToVector();

                Vector<T> useMean = training ? mean : _bnRunningMean[layer];
                Vector<T> useVariance = training ? variance : _bnRunningVariance[layer];

                var zhat = Standardize(preActivation, n, useMean, useVariance, epsilon);
                normalized = Engine.TensorAdd(
                    Engine.TensorMultiply(zhat, SurvivalNetworkOps.BroadcastRow(_bnGamma[layer], n)),
                    SurvivalNetworkOps.BroadcastRow(_bnBeta[layer], n));

                if (training)
                {
                    // Running estimates for inference, exponential moving average with momentum 0.9,
                    // written into the existing storage rather than rebinding it to an engine result.
                    Engine.Add(Engine.Multiply(_bnRunningMean[layer], momentum), Engine.Multiply(mean, oneMinusMomentum))
                        .AsSpan().CopyTo(_bnRunningMean[layer].AsWritableSpan());
                    Engine.Add(Engine.Multiply(_bnRunningVariance[layer], momentum), Engine.Multiply(variance, oneMinusMomentum))
                        .AsSpan().CopyTo(_bnRunningVariance[layer].AsWritableSpan());
                }

                cache.BatchMean.Add(training ? mean : _bnRunningMean[layer]);
                cache.BatchVariance.Add(training ? variance : _bnRunningVariance[layer]);
            }

            var activated = _options.Activation.Activate(normalized);

            // Inverted dropout: scale the survivors at training time so inference needs no rescaling.
            Tensor<T>? mask = null;
            if (training && _options.DropoutRate > 0.0)
            {
                mask = SurvivalNetworkOps.DropoutMask(_random, n, outputSize, dropoutRate, keepScale);
                activated = Engine.TensorMultiply(activated, mask);
            }

            cache.Inputs.Add(current);
            cache.PreActivations.Add(preActivation);
            cache.NormalizedPreActivations.Add(normalized);
            cache.DropoutMasks.Add(mask);

            current = activated;
        }

        // Output layer: a single linear risk score, no activation.
        var riskScores = SurvivalNetworkOps.Linear(current, n, _weights[^1], _biases[^1]).Reshape(n).ToVector();

        cache.Inputs.Add(current);
        return (riskScores, cache);
    }

    /// <summary>
    /// (z - mean) / sqrt(variance + epsilon) for every row of an [n, width] tensor.
    /// </summary>
    private Tensor<T> Standardize(Tensor<T> values, int n, Vector<T> mean, Vector<T> variance, T epsilon)
    {
        var std = Engine.Sqrt(Engine.Add(variance, Engine.Fill(variance.Length, epsilon)));
        return Engine.TensorDivide(
            Engine.TensorSubtract(values, SurvivalNetworkOps.BroadcastRow(mean, n)),
            SurvivalNetworkOps.BroadcastRow(std, n));
    }

    /// <summary>
    /// Computes the negative Cox partial log-likelihood and its derivative with respect to each risk score.
    /// </summary>
    /// <param name="riskScores">Risk score per batch member.</param>
    /// <param name="times">Observed time per batch member.</param>
    /// <param name="events">Event indicator per batch member.</param>
    /// <returns>The mean loss over the events, and d(loss)/d(risk score) per batch member.</returns>
    /// <remarks>
    /// <para>
    /// Katzman et al. (2018), eq. 4. With the risk set R(t) = { j : t_j >= t }, the negative partial
    /// log-likelihood is
    /// </para>
    /// <para>
    ///   L = - (1/N_E) sum over events i of [ r_i - log S_i ],   S_i = sum over j in R(t_i) of exp(r_j)
    /// </para>
    /// <para>
    /// and its derivative with respect to one risk score is
    /// </para>
    /// <para>
    ///   dL/dr_k = (1/N_E) [ -d_k + exp(r_k) * sum over events i with t_i &lt;= t_k of 1/S_i ]
    /// </para>
    /// <para>
    /// where d_k is the event indicator. Both sums are evaluated in ascending time order: S is a suffix sum
    /// read at the first position of each tied time, and the reciprocal sum a prefix sum read at the last
    /// position of each tied time. Tied subjects are therefore all in one another's risk set, as R(t)
    /// requires; the previous position-by-position scan dropped the tied subjects that happened to sort
    /// earlier from the risk set, which made the loss depend on the order of tied rows.
    /// </para>
    /// </remarks>
    private (T loss, Vector<T> gradients) ComputeCoxLossAndGradients(
        Vector<T> riskScores, Vector<T> times, Vector<T> events)
    {
        int n = riskScores.Length;
        T epsilon = NumOps.FromDouble(1e-10);

        // Clamp risk scores before exponentiating, to keep exp finite.
        var clamped = ClampRisk(riskScores);
        var expRisk = Engine.TensorExp(clamped);

        // Ascending time order within this batch; the risk set is the batch itself.
        var (sortedTimes, order) = Engine.TensorSort(Tensor<T>.FromVector(times), 0, false);
        var expSorted = Engine.TensorIndexSelect(expRisk, order, 0);
        var clampedSorted = Engine.TensorIndexSelect(clamped, order, 0);
        var eventSorted = Engine.TensorIndexSelect(EventIndicator(events), order, 0);

        var firstOfTie = Engine.TensorSearchSorted(sortedTimes, sortedTimes, false);
        var endOfTie = Engine.TensorSearchSorted(sortedTimes, sortedTimes, true);

        // S at each position: the suffix sum of exp(r) from the first subject tied with it.
        var riskSetSum = Engine.TensorAddScalar(
            Engine.TensorIndexSelect(SuffixSum(expSorted), firstOfTie, 0), epsilon);

        T loss = NumOps.Negate(Engine.TensorSum(Engine.TensorMultiply(
            eventSorted, Engine.TensorSubtract(clampedSorted, Engine.TensorLog(riskSetSum)))));
        T eventCount = Engine.TensorSum(eventSorted);

        // sum of 1/S over the events at or before each position, ties included.
        var reciprocal = Engine.TensorDivide(eventSorted, riskSetSum);
        var reciprocalSum = Engine.TensorIndexSelect(ExclusivePrefixSum(reciprocal), endOfTie, 0);
        var sortedGradients = Engine.TensorSubtract(Engine.TensorMultiply(expSorted, reciprocalSum), eventSorted);

        // Back to batch order.
        var gradients = Engine.TensorScatter(
            Tensor<T>.CreateDefault(new[] { n }, NumOps.Zero), order, sortedGradients, 0);

        // Normalize by the number of events, which is what the likelihood actually sums over.
        T scale = NumOps.GreaterThan(eventCount, NumOps.Zero) ? eventCount : NumOps.One;
        return (NumOps.Divide(loss, scale), Engine.TensorDivideScalar(gradients, scale).ToVector());
    }

    /// <summary>
    /// suffix[p] = sum of values[q] for q &gt;= p, accumulated from the end so small late terms are not
    /// lost against large early ones.
    /// </summary>
    private Tensor<T> SuffixSum(Tensor<T> values)
    {
        var axis = new[] { 0 };
        return Engine.TensorFlip(Engine.TensorCumSum(Engine.TensorFlip(values, axis), 0), axis);
    }

    /// <summary>
    /// A length n+1 tensor whose entry p is the sum of the first p values (entry 0 is 0).
    /// </summary>
    private Tensor<T> ExclusivePrefixSum(Tensor<T> values)
    {
        return Engine.TensorConcatenate(
            new[] { Tensor<T>.CreateDefault(new[] { 1 }, NumOps.Zero), Engine.TensorCumSum(values, 0) }, 0);
    }


    /// <summary>
    /// Accumulated gradients for one backward pass, laid out to match the network's parameter lists.
    /// </summary>
    private sealed class ParameterGradients
    {
        public List<Matrix<T>> Weights { get; } = new List<Matrix<T>>();
        public List<Vector<T>> Biases { get; } = new List<Vector<T>>();
        public List<Vector<T>> Gamma { get; } = new List<Vector<T>>();
        public List<Vector<T>> Beta { get; } = new List<Vector<T>>();
    }

    /// <summary>
    /// Adam moment estimates, one entry per parameter tensor.
    /// </summary>
    private sealed class AdamState
    {
        public List<Matrix<T>> WeightM { get; } = new List<Matrix<T>>();
        public List<Matrix<T>> WeightV { get; } = new List<Matrix<T>>();
        public List<Vector<T>> BiasM { get; } = new List<Vector<T>>();
        public List<Vector<T>> BiasV { get; } = new List<Vector<T>>();
        public List<Vector<T>> GammaM { get; } = new List<Vector<T>>();
        public List<Vector<T>> GammaV { get; } = new List<Vector<T>>();
        public List<Vector<T>> BetaM { get; } = new List<Vector<T>>();
        public List<Vector<T>> BetaV { get; } = new List<Vector<T>>();
        public int Step { get; set; }
    }

    /// <summary>
    /// Fits the network by minimizing the negative Cox partial log-likelihood.
    /// </summary>
    /// <param name="x">Feature matrix.</param>
    /// <param name="times">Observed times.</param>
    /// <param name="events">Event indicators.</param>
    /// <remarks>
    /// Mini-batch Adam with the L2 penalty of Katzman et al. added to the weight gradients, and early
    /// stopping on the epoch loss. The batch is its own risk set, so each batch must be large enough to
    /// contain events; a batch that happens to hold no event contributes no gradient and is skipped rather
    /// than dividing by zero.
    /// </remarks>
    private void FitNetwork(Matrix<T> x, Vector<T> times, Vector<T> events)
    {
        int n = x.Rows;
        int batchSize = Math.Max(2, Math.Min(_options.BatchSize, n));
        var adam = CreateAdamState();
        var timeTensor = Tensor<T>.FromVector(times);
        var eventTensor = Tensor<T>.FromVector(events);
        var indicator = EventIndicator(events);

        T bestLoss = NumOps.MaxValue;
        int epochsWithoutImprovement = 0;
        List<Matrix<T>>? bestWeights = null;
        List<Vector<T>>? bestBiases = null;
        List<Vector<T>>? bestBnGamma = null;
        List<Vector<T>>? bestBnBeta = null;
        List<Vector<T>>? bestBnRunningMean = null;
        List<Vector<T>>? bestBnRunningVariance = null;

        var order = Enumerable.Range(0, n).ToArray();

        for (int epoch = 0; epoch < _options.Epochs; epoch++)
        {
            order = ShuffleArray(order);
            var batchLosses = new List<T>();

            for (int start = 0; start < n; start += batchSize)
            {
                int count = Math.Min(batchSize, n - start);
                if (count < 2)
                {
                    // A single-subject risk set carries no information: the partial likelihood term is
                    // log(exp(r)/exp(r)) = 0 for it. Skip rather than emit a zero-variance update.
                    continue;
                }

                var batchIndices = new int[count];
                Array.Copy(order, start, batchIndices, 0, count);
                var batchSelector = new Tensor<int>(batchIndices, new[] { count });

                if (Engine.TensorCountNonzero(Engine.TensorIndexSelect(indicator, batchSelector, 0)) == 0)
                {
                    continue;
                }

                var batchTimes = Engine.TensorIndexSelect(timeTensor, batchSelector, 0).ToVector();
                var batchEvents = Engine.TensorIndexSelect(eventTensor, batchSelector, 0).ToVector();

                var (riskScores, cache) = ForwardPass(x, batchIndices, training: true);
                var (loss, riskGradients) = ComputeCoxLossAndGradients(riskScores, batchTimes, batchEvents);
                var gradients = Backpropagate(cache, riskGradients);
                ApplyAdamUpdate(gradients, adam);

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
                bestWeights = AiDotNet.Training.HostStateCheckpoint<T>.Capture(_weights, bestWeights);
                bestBiases = AiDotNet.Training.HostStateCheckpoint<T>.Capture(_biases, bestBiases);
                bestBnGamma = AiDotNet.Training.HostStateCheckpoint<T>.Capture(_bnGamma, bestBnGamma);
                bestBnBeta = AiDotNet.Training.HostStateCheckpoint<T>.Capture(_bnBeta, bestBnBeta);
                bestBnRunningMean = AiDotNet.Training.HostStateCheckpoint<T>.Capture(_bnRunningMean, bestBnRunningMean);
                bestBnRunningVariance = AiDotNet.Training.HostStateCheckpoint<T>.Capture(_bnRunningVariance, bestBnRunningVariance);
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

        // Restore one coherent best-epoch checkpoint. Batch-normalization scale, shift and running
        // statistics are part of the model state just as much as the dense weights are. Restoring only
        // weights/biases paired the best dense checkpoint with normalization state from a later, possibly
        // unstable epoch; that produced initialization-dependent NaN/Infinity predictions after training.
        if (bestWeights is not null && bestBiases is not null &&
            bestBnGamma is not null && bestBnBeta is not null &&
            bestBnRunningMean is not null && bestBnRunningVariance is not null)
        {
            _weights = bestWeights;
            _biases = bestBiases;
            _bnGamma = bestBnGamma;
            _bnBeta = bestBnBeta;
            _bnRunningMean = bestBnRunningMean;
            _bnRunningVariance = bestBnRunningVariance;
        }
    }

    /// <summary>
    /// Whether an epoch's mean loss beats the best so far by more than 1e-7.
    /// </summary>
    private bool IsImprovement(T meanLoss, T bestLoss)
    {
        return NumOps.LessThan(meanLoss, NumOps.Subtract(bestLoss, NumOps.FromDouble(1e-7)));
    }

    /// <summary>
    /// Creates zeroed Adam moment estimates matching the current parameter shapes.
    /// </summary>
    private AdamState CreateAdamState()
    {
        var state = new AdamState();
        for (int layer = 0; layer < _weights.Count; layer++)
        {
            state.WeightM.Add(new Matrix<T>(_weights[layer].Rows, _weights[layer].Columns));
            state.WeightV.Add(new Matrix<T>(_weights[layer].Rows, _weights[layer].Columns));
            state.BiasM.Add(new Vector<T>(_biases[layer].Length));
            state.BiasV.Add(new Vector<T>(_biases[layer].Length));
        }

        for (int layer = 0; layer < _bnGamma.Count; layer++)
        {
            state.GammaM.Add(new Vector<T>(_bnGamma[layer].Length));
            state.GammaV.Add(new Vector<T>(_bnGamma[layer].Length));
            state.BetaM.Add(new Vector<T>(_bnBeta[layer].Length));
            state.BetaV.Add(new Vector<T>(_bnBeta[layer].Length));
        }

        return state;
    }

    /// <summary>
    /// Propagates d(loss)/d(risk score) back through the network into parameter gradients.
    /// </summary>
    /// <param name="cache">Intermediate values recorded by the forward pass.</param>
    /// <param name="riskGradients">d(loss)/d(risk score) for each batch member.</param>
    /// <returns>Gradients for every weight, bias and batch-norm parameter.</returns>
    /// <remarks>
    /// The chain runs output layer, then each hidden layer in reverse: dropout mask, activation
    /// derivative, batch norm, then the linear map. The batch-norm step uses the standard three-term
    /// derivative from Ioffe and Szegedy (2015), which accounts for the batch mean and variance both
    /// depending on every sample in the batch.
    /// </remarks>
    private ParameterGradients Backpropagate(ForwardCache cache, Vector<T> riskGradients)
    {
        int layerCount = _weights.Count;
        int n = riskGradients.Length;
        bool useBatchNorm = _options.UseBatchNormalization && cache.BatchMean.Count > 0;
        T epsilon = NumOps.FromDouble(1e-5);
        T batchCount = NumOps.FromDouble(n);

        var weightGradients = new Matrix<T>[layerCount];
        var biasGradients = new Vector<T>[layerCount];
        var gammaGradients = new Vector<T>[_bnGamma.Count];
        var betaGradients = new Vector<T>[_bnGamma.Count];

        // Output layer: r_i = w . h_i + b, so dL/dw = H^T g and dL/db = sum_i g_i.
        var (outputWeightGradient, outputBiasGradient, delta) = SurvivalNetworkOps.LinearBackward(
            cache.Inputs[layerCount - 1], new Tensor<T>(new[] { n, 1 }, riskGradients), _weights[layerCount - 1]);
        weightGradients[layerCount - 1] = outputWeightGradient;
        biasGradients[layerCount - 1] = outputBiasGradient;

        // Hidden layers, in reverse.
        for (int layer = layerCount - 2; layer >= 0; layer--)
        {
            int outputSize = _weights[layer].Columns;
            var mask = cache.DropoutMasks[layer];

            // Through dropout and the activation, giving dL/d(normalized pre-activation).
            var afterDropout = mask is null ? delta : Engine.TensorMultiply(delta, mask);
            var dNorm = Engine.TensorMultiply(
                afterDropout, _options.Activation.Derivative(cache.NormalizedPreActivations[layer]));

            // Through batch norm, giving dL/d(pre-activation).
            var dPre = dNorm;
            if (useBatchNorm)
            {
                var variance = cache.BatchVariance[layer];
                var zhat = Standardize(cache.PreActivations[layer], n, cache.BatchMean[layer], variance, epsilon);

                // dL/dgamma and dL/dbeta accumulate over the batch.
                gammaGradients[layer] = Engine.ReduceSum(
                    Engine.TensorMultiply(dNorm, zhat), new[] { 0 }, keepDims: false).ToVector();
                betaGradients[layer] = Engine.ReduceSum(dNorm, new[] { 0 }, keepDims: false).ToVector();

                // dL/dz = gamma/(N*std) * (N*dzhat - sum(dzhat) - zhat * sum(dzhat*zhat))
                var dzhat = Engine.TensorMultiply(dNorm, SurvivalNetworkOps.BroadcastRow(_bnGamma[layer], n));
                var sumDzhat = Engine.ReduceSum(dzhat, new[] { 0 }, keepDims: false).ToVector();
                var sumDzhatZhat = Engine.ReduceSum(
                    Engine.TensorMultiply(dzhat, zhat), new[] { 0 }, keepDims: false).ToVector();

                var inner = Engine.TensorSubtract(
                    Engine.TensorSubtract(
                        Engine.TensorMultiplyScalar(dzhat, batchCount),
                        SurvivalNetworkOps.BroadcastRow(sumDzhat, n)),
                    Engine.TensorMultiply(zhat, SurvivalNetworkOps.BroadcastRow(sumDzhatZhat, n)));

                var std = Engine.Sqrt(Engine.Add(variance, Engine.Fill(outputSize, epsilon)));
                var invNStd = Engine.Reciprocal(Engine.Multiply(std, batchCount));
                dPre = Engine.TensorMultiply(inner, SurvivalNetworkOps.BroadcastRow(invNStd, n));
            }
            else if (layer < gammaGradients.Length)
            {
                gammaGradients[layer] = new Vector<T>(outputSize);
                betaGradients[layer] = new Vector<T>(outputSize);
            }

            // Through the linear map, giving parameter gradients and the delta for the layer below.
            var (weightGradient, biasGradient, nextDelta) = SurvivalNetworkOps.LinearBackward(
                cache.Inputs[layer], dPre, _weights[layer]);
            weightGradients[layer] = weightGradient;
            biasGradients[layer] = biasGradient;
            delta = nextDelta;
        }

        var grads = new ParameterGradients();
        grads.Weights.AddRange(weightGradients);
        grads.Biases.AddRange(biasGradients);
        grads.Gamma.AddRange(gammaGradients);
        grads.Beta.AddRange(betaGradients);
        return grads;
    }

    /// <summary>
    /// Applies one Adam step to every parameter, with the L2 penalty on the weight matrices.
    /// </summary>
    /// <remarks>
    /// The penalty is added to the weight gradients only, which is the L2 term of the DeepSurv objective.
    /// Biases and the batch-norm scale and shift are left unpenalized, which is the standard convention:
    /// shrinking them toward zero removes the layer's ability to shift its output rather than controlling
    /// model complexity.
    /// </remarks>
    private void ApplyAdamUpdate(ParameterGradients grads, AdamState adam)
    {
        adam.Step++;
        var coefficients = new SurvivalNetworkOps.AdamCoefficients<T>(
            _options.LearningRate, _options.L2Regularization, adam.Step);

        for (int layer = 0; layer < _weights.Count; layer++)
        {
            SurvivalNetworkOps.AdamStep(_weights[layer], grads.Weights[layer],
                adam.WeightM[layer], adam.WeightV[layer], coefficients, applyL2: true);
            SurvivalNetworkOps.AdamStep(_biases[layer], grads.Biases[layer],
                adam.BiasM[layer], adam.BiasV[layer], coefficients, applyL2: false);
        }

        for (int layer = 0; layer < grads.Gamma.Count; layer++)
        {
            SurvivalNetworkOps.AdamStep(_bnGamma[layer], grads.Gamma[layer],
                adam.GammaM[layer], adam.GammaV[layer], coefficients, applyL2: false);
            SurvivalNetworkOps.AdamStep(_bnBeta[layer], grads.Beta[layer],
                adam.BetaM[layer], adam.BetaV[layer], coefficients, applyL2: false);
        }
    }

    /// <summary>
    /// Predicts the expected survival time for each subject by integrating the predicted survival curve.
    /// </summary>
    /// <param name="input">Feature matrix, one row per subject.</param>
    /// <returns>Expected survival time per subject.</returns>
    /// <remarks>
    /// <para>
    /// E[T] = integral of S(t) dt over t >= 0. The integral is evaluated on the trapezoid rule over the
    /// baseline hazard's event times, which is where S changes, and the tail beyond the last observed time
    /// is added in closed form: past that point the Cox model has no further baseline increments, so S is
    /// treated as decaying exponentially at the rate implied by the subject's own hazard. That keeps the
    /// result finite for every subject, including low-risk ones whose survival curve never reaches 0.5.
    /// </para>
    /// </remarks>
    public Vector<T> PredictExpectedSurvivalTime(Matrix<T> input)
    {
        var expRisk = ClampedExpRisk(PredictRiskScores(input));
        int n = expRisk.Length;

        if (_baselineHazardTimes is null || _baselineHazardValues is null || _baselineHazardTimes.Length == 0)
        {
            // No baseline hazard: fall back to the exponential model S(t) = exp(-exp(r) t), whose mean is
            // 1/exp(r). This is the same model PredictSurvival uses in that situation.
            return Engine.Reciprocal(expRisk);
        }

        int m = _baselineHazardTimes.Length;
        var eventTimes = _baselineHazardTimes;
        var hazard = _baselineHazardValues;
        var zero = new Vector<T>(1);
        var previousTimes = Vector<T>.Concatenate(new[] { zero, eventTimes.GetSubVector(0, m - 1) });
        var previousHazard = Vector<T>.Concatenate(new[] { zero, hazard.GetSubVector(0, m - 1) });
        var risk = Tensor<T>.FromVector(expRisk);

        // S at each event time and at the one before it (S = 1 before the first), per subject: [n, m].
        var survival = Engine.TensorExp(Engine.TensorNegate(Engine.TensorOuter(risk, Tensor<T>.FromVector(hazard))));
        var previousSurvival = Engine.TensorExp(Engine.TensorNegate(
            Engine.TensorOuter(risk, Tensor<T>.FromVector(previousHazard))));
        var widths = Engine.Subtract(eventTimes, previousTimes);

        var trapezoids = Engine.TensorMultiplyScalar(
            Engine.TensorMultiply(Engine.TensorAdd(previousSurvival, survival), SurvivalNetworkOps.BroadcastRow(widths, n)),
            NumOps.FromDouble(0.5));
        var area = Engine.ReduceSum(trapezoids, new[] { 1 }, keepDims: false);

        // Tail beyond the last observed event time. The instantaneous hazard there is unidentified, so
        // extend at the average rate the subject accumulated over follow-up; with S = exp(-H) that
        // gives a remaining expectation of S(last) / rate.
        T lastHazard = hazard[m - 1];
        T lastTime = eventTimes[m - 1];
        var rate = NumOps.GreaterThan(lastTime, NumOps.Zero)
            ? Engine.TensorMultiplyScalar(risk, NumOps.Divide(lastHazard, lastTime))
            : risk;
        var lastSurvival = Engine.TensorExp(Engine.TensorNegate(Engine.TensorMultiplyScalar(risk, lastHazard)));

        var tail = Engine.TensorWhere(
            Engine.TensorGreaterThan(rate, NumOps.FromDouble(1e-12)),
            Engine.TensorDivide(lastSurvival, rate),
            Engine.TensorMultiplyScalar(lastSurvival, _maxObservedTime));

        return Engine.TensorAdd(area, tail).ToVector();
    }

    /// <summary>
    /// Fits the Breslow estimate of the baseline cumulative hazard on the training data.
    /// </summary>
    /// <remarks>
    /// H0(t) = sum over distinct event times t_i &lt;= t of d(t_i) / sum over j in R(t_i) of exp(r_j). The
    /// risk-set sums are suffix sums in ascending time order, read at the first subject of each tied time
    /// so all tied subjects stay in that time's risk set. Starting from the full sum and subtracting
    /// exp(risk) is numerically unsafe -- the clamp still permits an exp(40) dynamic range -- so the suffix
    /// sum is accumulated from the end, building the small late-time risk sets before the large early ones.
    /// </remarks>
    private void ComputeBaselineHazard(Matrix<T> x, Vector<T> times, Vector<T> events)
    {
        int n = times.Length;
        T epsilon = NumOps.FromDouble(1e-10);
        var expRisk = Tensor<T>.FromVector(ClampedExpRisk(PredictRiskScores(x)));

        var (sortedTimes, order) = Engine.TensorSort(Tensor<T>.FromVector(times), 0, false);
        var expSorted = Engine.TensorIndexSelect(expRisk, order, 0);
        var eventSorted = Engine.TensorIndexSelect(EventIndicator(events), order, 0);

        var firstOfTie = Engine.TensorSearchSorted(sortedTimes, sortedTimes, false);
        var endOfTie = Engine.TensorSearchSorted(sortedTimes, sortedTimes, true);

        // Each event's share d / (risk-set sum + eps); summed over a tied group that is the group's increment.
        var riskSetSum = Engine.TensorAddScalar(Engine.TensorIndexSelect(SuffixSum(expSorted), firstOfTie, 0), epsilon);
        var cumulativeHazard = Engine.TensorIndexSelect(
            ExclusivePrefixSum(Engine.TensorDivide(eventSorted, riskSetSum)), endOfTie, 0);

        // One entry per distinct time that has at least one event, taken at the group's first subject.
        var eventsBefore = ExclusivePrefixSum(eventSorted);
        var groupEvents = Engine.TensorSubtract(
            Engine.TensorIndexSelect(eventsBefore, endOfTie, 0),
            Engine.TensorIndexSelect(eventsBefore, firstOfTie, 0));
        var sortedVector = sortedTimes.ToVector();
        var groupStart = n > 1
            ? Engine.TensorConcatenate(new[]
            {
                Tensor<T>.CreateDefault(new[] { 1 }, NumOps.One),
                Engine.ScalarMinusTensor(NumOps.One, Engine.TensorEquals(
                    Tensor<T>.FromVector(RegressionEngineOps2.Slice(sortedVector, 1, n - 1)),
                    Tensor<T>.FromVector(sortedVector.GetSubVector(0, n - 1))))
            }, 0)
            : Tensor<T>.CreateDefault(new[] { n }, NumOps.One);
        var keep = Engine.TensorEqScalar(
            Engine.TensorMultiply(groupStart, Engine.TensorGreaterThan(groupEvents, NumOps.Zero)), NumOps.One);

        _baselineHazardTimes = Engine.TensorMaskedSelect(sortedTimes, keep).ToVector();
        _baselineHazardValues = Engine.TensorMaskedSelect(cumulativeHazard, keep).ToVector();
    }

    /// <summary>
    /// The Breslow baseline cumulative hazard at each of <paramref name="times"/>: the value at the last
    /// event time at or before t, and 0 before the first event time.
    /// </summary>
    /// <remarks>
    /// The estimate is a right-continuous step function. The previous lookup returned the step before an
    /// event time when asked at exactly that time, so S(t_i) omitted the event at t_i itself.
    /// </remarks>
    private Vector<T> BaselineHazardAt(Vector<T> times)
    {
        if (_baselineHazardTimes is null || _baselineHazardTimes.Length == 0)
        {
            return new Vector<T>(times.Length);
        }

        if (_baselineHazardValues is null)
        {
            throw new InvalidOperationException("Baseline hazard values have not been computed.");
        }

        // Number of event times <= t indexes [0, H...] directly: 0 before the first event time.
        var steps = Engine.TensorSearchSorted(
            Tensor<T>.FromVector(_baselineHazardTimes), Tensor<T>.FromVector(times), true);
        var padded = Vector<T>.Concatenate(new[] { new Vector<T>(1), _baselineHazardValues });
        return Engine.TensorIndexSelect(Tensor<T>.FromVector(padded), steps, 0).ToVector();
    }

    /// <summary>
    /// For each target cumulative hazard, the first event time at which the baseline hazard reaches it,
    /// or <c>MaxValue</c> when it never does.
    /// </summary>
    private Vector<T> FindTimesForHazard(Vector<T> targetHazard)
    {
        if (_baselineHazardTimes is null || _baselineHazardTimes.Length == 0)
        {
            return Engine.Fill(targetHazard.Length, NumOps.MaxValue);
        }

        if (_baselineHazardValues is null)
        {
            throw new InvalidOperationException("Baseline hazard values have not been computed.");
        }

        // H0 is non-decreasing, so the first index with H0 >= target is its left insertion point.
        var positions = Engine.TensorSearchSorted(
            Tensor<T>.FromVector(_baselineHazardValues), Tensor<T>.FromVector(targetHazard), false);
        var padded = Vector<T>.Concatenate(new[] { _baselineHazardTimes, Engine.Fill(1, NumOps.MaxValue) });
        return Engine.TensorIndexSelect(Tensor<T>.FromVector(padded), positions, 0).ToVector();
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
        FeatureImportances = SurvivalNetworkOps.FirstLayerImportances(
            _weights.Count > 0 ? _weights[0] : null, _numFeatures);
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NumHiddenLayers", _options.NumHiddenLayers },
                { "HiddenLayerSize", _options.HiddenLayerSize },
                { "Activation", _options.Activation.GetType().Name },
                { "NumberOfFeatures", _numFeatures }
            }
        };
    }
}
