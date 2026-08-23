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
/// var model = new DeepSurv&lt;double&gt;(options);
///
/// // Prepare training data: 6 samples with 3 clinical features
/// var features = Matrix&lt;double&gt;.Build.Dense(6, 3, new double[] {
///     55, 1, 2.1,  60, 0, 3.5,  45, 1, 1.8,
///     70, 0, 4.2,  50, 1, 2.9,  65, 0, 3.1 });
/// var targets = new Vector&lt;double&gt;(new double[] { 12, 24, 36, 6, 18, 30 });
///
/// // Train the neural network for Cox regression
/// model.Train(features, targets);
///
/// // Predict risk scores for a new patient
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
[ResearchPaper("DeepSurv: Personalized Treatment Recommender System Using a Cox Proportional Hazards Deep Neural Network", "https://doi.org/10.1186/s12874-018-0482-1", Year = 2018, Authors = "Jared L. Katzman, Uri Shaham, Alexander Cloninger, Jonathan Bates, Tingting Jiang, Yuval Kluger")]
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
    private readonly DeepSurvOptions<T> _options;

    /// <summary>
    /// Random number generator.
    /// </summary>
    private readonly Random _random;

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

        for (int i = 0; i < times.Length; i++)
        {
            if (!NumOps.GreaterThan(times[i], NumOps.Zero))
            {
                throw new ArgumentException(
                    $"Survival times must be strictly positive; got {NumOps.ToDouble(times[i]):G6} at index {i}.",
                    nameof(times));
            }
        }

        bool anyEvent = false;
        for (int i = 0; i < events.Length; i++)
        {
            if (NumOps.Compare(events[i], NumOps.One) == 0)
            {
                anyEvent = true;
                break;
            }
        }

        if (!anyEvent)
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

        T maxTime = times[0];
        for (int i = 1; i < times.Length; i++)
        {
            if (NumOps.GreaterThan(times[i], maxTime)) maxTime = times[i];
        }
        _maxObservedTime = maxTime;

        await Task.Run(() => FitNetwork(x, times, events));

        int[] sorted = GetSortedIndices(times);
        ComputeBaselineHazard(x, times, events, sorted);

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
                T expRisk = ClampedExpRisk(riskScores[i]);
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
                T expRisk = ClampedExpRisk(riskScores[i]);
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
            T expRisk = ClampedExpRisk(riskScores[i]);

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
        _bnGamma = new List<Vector<T>>();
        _bnBeta = new List<Vector<T>>();
        _bnRunningMean = new List<Vector<T>>();
        _bnRunningVariance = new List<Vector<T>>();

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

            // Batch-normalization parameters start at the identity transform (gamma = 1, beta = 0), so an
            // untrained normalized layer behaves like the un-normalized one.
            _bnGamma.Add(Vector<T>.CreateDefault(outputSize, NumOps.One));
            _bnBeta.Add(new Vector<T>(outputSize));
            _bnRunningMean.Add(new Vector<T>(outputSize));
            _bnRunningVariance.Add(Vector<T>.CreateDefault(outputSize, NumOps.One));

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
    /// Exponentiates a risk score after clamping it into a range where the result stays finite.
    /// </summary>
    /// <param name="riskScore">The raw linear predictor.</param>
    /// <returns>exp of the clamped risk score.</returns>
    /// <remarks>
    /// See <see cref="RiskScoreClamp"/>. Every exponentiation of a risk score in this class goes through
    /// here, so the loss, the baseline hazard and the survival function all agree on the same bound --
    /// they previously did not, and the baseline hazard was the one that overflowed.
    /// </remarks>
    private T ClampedExpRisk(T riskScore)
    {
        double v = NumOps.ToDouble(riskScore);
        if (double.IsNaN(v)) v = 0.0;
        v = Math.Max(-RiskScoreClamp, Math.Min(RiskScoreClamp, v));
        return NumOps.FromDouble(Math.Exp(v));
    }

    /// <summary>
    /// Everything the backward pass needs from one forward pass.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Working out how to improve the network requires remembering what it computed
    /// on the way forward, not just the final answer. This carries those intermediate values back.
    /// </remarks>
    private sealed class ForwardCache
    {
        /// <summary>Input to each layer; entry l feeds layer l. Length is layer count.</summary>
        public List<Vector<T>[]> Inputs { get; } = new List<Vector<T>[]>();

        /// <summary>Pre-activation of each hidden layer, before batch norm and activation.</summary>
        public List<Vector<T>[]> PreActivations { get; } = new List<Vector<T>[]>();

        /// <summary>Batch-normalized pre-activation of each hidden layer, before the activation.</summary>
        public List<Vector<T>[]> NormalizedPreActivations { get; } = new List<Vector<T>[]>();

        /// <summary>Per-hidden-layer batch mean, empty when batch norm is off.</summary>
        public List<Vector<T>> BatchMean { get; } = new List<Vector<T>>();

        /// <summary>Per-hidden-layer batch variance, empty when batch norm is off.</summary>
        public List<Vector<T>> BatchVariance { get; } = new List<Vector<T>>();

        /// <summary>Dropout mask per hidden layer: 0 for a dropped unit, the inverted-dropout scale otherwise.</summary>
        public List<Vector<T>[]> DropoutMasks { get; } = new List<Vector<T>[]>();
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

        var current = new Vector<T>[n];
        for (int i = 0; i < n; i++)
        {
            current[i] = new Vector<T>(_numFeatures);
            for (int j = 0; j < _numFeatures; j++)
            {
                current[i][j] = StandardizeFeature(x[indices[i], j], j);
            }
        }

        // Hidden layers: linear -> batch norm -> activation -> dropout.
        for (int layer = 0; layer < _weights.Count - 1; layer++)
        {
            var w = _weights[layer];
            var b = _biases[layer];
            int outputSize = w.Columns;

            var weightTensor = Tensor<T>.FromMatrix(w);
            var biasTensor = Tensor<T>.FromVector(b).Reshape(1, outputSize);

            // Linear part, vectorized per sample.
            var preActivation = new Vector<T>[n];
            for (int i = 0; i < n; i++)
            {
                var inputTensor = Tensor<T>.FromVector(current[i]).Reshape(1, current[i].Length);
                var result = Engine.TensorAdd(
                    Engine.TensorMatMul(inputTensor, weightTensor), biasTensor);
                preActivation[i] = result.Reshape(outputSize).ToVector();
            }

            // Batch normalization (Ioffe & Szegedy 2015) over the batch dimension.
            var normalized = new Vector<T>[n];
            if (useBatchNorm)
            {
                var mean = new Vector<T>(outputSize);
                var variance = new Vector<T>(outputSize);

                for (int k = 0; k < outputSize; k++)
                {
                    T sum = NumOps.Zero;
                    for (int i = 0; i < n; i++)
                    {
                        sum = NumOps.Add(sum, preActivation[i][k]);
                    }
                    mean[k] = NumOps.Divide(sum, NumOps.FromDouble(n));

                    T sqSum = NumOps.Zero;
                    for (int i = 0; i < n; i++)
                    {
                        T d = NumOps.Subtract(preActivation[i][k], mean[k]);
                        sqSum = NumOps.Add(sqSum, NumOps.Multiply(d, d));
                    }
                    variance[k] = NumOps.Divide(sqSum, NumOps.FromDouble(n));
                }

                Vector<T> useMean = training ? mean : _bnRunningMean[layer];
                Vector<T> useVariance = training ? variance : _bnRunningVariance[layer];

                for (int i = 0; i < n; i++)
                {
                    normalized[i] = new Vector<T>(outputSize);
                    for (int k = 0; k < outputSize; k++)
                    {
                        T std = NumOps.Sqrt(NumOps.Add(useVariance[k], epsilon));
                        T zhat = NumOps.Divide(NumOps.Subtract(preActivation[i][k], useMean[k]), std);
                        normalized[i][k] = NumOps.Add(
                            NumOps.Multiply(_bnGamma[layer][k], zhat), _bnBeta[layer][k]);
                    }
                }

                if (training)
                {
                    // Running estimates for inference, exponential moving average with momentum 0.9.
                    T momentum = NumOps.FromDouble(0.9);
                    T oneMinus = NumOps.FromDouble(0.1);
                    for (int k = 0; k < outputSize; k++)
                    {
                        _bnRunningMean[layer][k] = NumOps.Add(
                            NumOps.Multiply(momentum, _bnRunningMean[layer][k]),
                            NumOps.Multiply(oneMinus, mean[k]));
                        _bnRunningVariance[layer][k] = NumOps.Add(
                            NumOps.Multiply(momentum, _bnRunningVariance[layer][k]),
                            NumOps.Multiply(oneMinus, variance[k]));
                    }
                }

                cache.BatchMean.Add(training ? mean : _bnRunningMean[layer]);
                cache.BatchVariance.Add(training ? variance : _bnRunningVariance[layer]);
            }
            else
            {
                for (int i = 0; i < n; i++)
                {
                    normalized[i] = preActivation[i];
                }
            }

            // Activation.
            var activated = new Vector<T>[n];
            for (int i = 0; i < n; i++)
            {
                var t = Tensor<T>.FromVector(normalized[i]).Reshape(1, outputSize);
                activated[i] = _options.Activation.Activate(t).Reshape(outputSize).ToVector();
            }

            // Inverted dropout: scale the survivors at training time so inference needs no rescaling.
            var masks = new Vector<T>[n];
            if (training && _options.DropoutRate > 0.0)
            {
                T keepScale = NumOps.FromDouble(1.0 / (1.0 - _options.DropoutRate));
                for (int i = 0; i < n; i++)
                {
                    masks[i] = new Vector<T>(outputSize);
                    for (int k = 0; k < outputSize; k++)
                    {
                        bool keep = _random.NextDouble() >= _options.DropoutRate;
                        masks[i][k] = keep ? keepScale : NumOps.Zero;
                        activated[i][k] = NumOps.Multiply(activated[i][k], masks[i][k]);
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

            cache.Inputs.Add(current);
            cache.PreActivations.Add(preActivation);
            cache.NormalizedPreActivations.Add(normalized);
            cache.DropoutMasks.Add(masks);

            current = activated;
        }

        // Output layer: a single linear risk score, no activation.
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

        cache.Inputs.Add(current);
        return (riskScores, cache);
    }

    /// <summary>
    /// Computes the negative Cox partial log-likelihood and its derivative with respect to each risk score.
    /// </summary>
    /// <param name="riskScores">Risk score per batch member.</param>
    /// <param name="times">Observed time per batch member.</param>
    /// <param name="events">Event indicator per batch member.</param>
    /// <returns>The mean loss over the batch, and d(loss)/d(risk score) per batch member.</returns>
    /// <remarks>
    /// <para>
    /// With the risk set R(t) = { j : t_j >= t }, the negative partial log-likelihood is
    /// </para>
    /// <para>
    ///   L = - sum over events i of [ r_i - log S_i ],   S_i = sum over j in R(t_i) of exp(r_j)
    /// </para>
    /// <para>
    /// and its derivative with respect to one risk score is
    /// </para>
    /// <para>
    ///   dL/dr_k = -d_k + exp(r_k) * sum over events i with t_i &lt;= t_k of 1/S_i
    /// </para>
    /// <para>
    /// where d_k is the event indicator. That trailing sum runs over every earlier event time, because
    /// subject k sits in the risk set of all of them. The previous implementation kept only the single
    /// i = k term and added it for censored subjects too, where the correct sum runs over events alone --
    /// so the reported gradient was not the derivative of the reported loss. Both directions are computed
    /// in one pass each: descending time accumulates S, ascending time accumulates the reciprocal sum.
    /// </para>
    /// </remarks>
    private (T loss, Vector<T> gradients) ComputeCoxLossAndGradients(
        Vector<T> riskScores, Vector<T> times, Vector<T> events)
    {
        int n = riskScores.Length;
        var gradients = new Vector<T>(n);
        T epsilon = NumOps.FromDouble(1e-10);

        // Ascending time order within this batch; the risk set is the batch itself.
        int[] order = Enumerable.Range(0, n)
            .OrderBy(i => NumOps.ToDouble(times[i]))
            .ToArray();

        // Clamp risk scores before exponentiating, as the original code did, to keep exp finite.
        var expRisk = new T[n];
        var clamped = new T[n];
        for (int i = 0; i < n; i++)
        {
            double v = Math.Max(-20.0, Math.Min(20.0, NumOps.ToDouble(riskScores[i])));
            clamped[i] = NumOps.FromDouble(v);
            expRisk[i] = NumOps.Exp(clamped[i]);
        }

        // Descending time: S[p] is the risk-set sum at the time of the subject in position p.
        var riskSetSum = new T[n];
        T running = NumOps.Zero;
        for (int p = n - 1; p >= 0; p--)
        {
            running = NumOps.Add(running, expRisk[order[p]]);
            riskSetSum[p] = NumOps.Add(running, epsilon);
        }

        T loss = NumOps.Zero;
        int eventCount = 0;
        for (int p = 0; p < n; p++)
        {
            int idx = order[p];
            if (NumOps.Compare(events[idx], NumOps.One) == 0)
            {
                loss = NumOps.Subtract(loss, NumOps.Subtract(clamped[idx], NumOps.Log(riskSetSum[p])));
                eventCount++;
            }
        }

        // Ascending time: accumulate sum of 1/S over the events at or before each position.
        T reciprocalSum = NumOps.Zero;
        for (int p = 0; p < n; p++)
        {
            int idx = order[p];
            if (NumOps.Compare(events[idx], NumOps.One) == 0)
            {
                reciprocalSum = NumOps.Add(reciprocalSum, NumOps.Divide(NumOps.One, riskSetSum[p]));
            }

            T term = NumOps.Multiply(expRisk[idx], reciprocalSum);
            gradients[idx] = NumOps.Compare(events[idx], NumOps.One) == 0
                ? NumOps.Subtract(term, NumOps.One)
                : term;
        }

        // Normalize by the number of events, which is what the likelihood actually sums over.
        T scale = NumOps.FromDouble(eventCount > 0 ? eventCount : 1);
        for (int i = 0; i < n; i++)
        {
            gradients[i] = NumOps.Divide(gradients[i], scale);
        }

        return (NumOps.Divide(loss, scale), gradients);
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
    /// Mini-batch Adam with decoupled L2 weight decay and early stopping on the epoch loss. The batch is
    /// its own risk set, so each batch must be large enough to contain events; a batch that happens to hold
    /// no event contributes no gradient and is skipped rather than dividing by zero.
    /// </remarks>
    private void FitNetwork(Matrix<T> x, Vector<T> times, Vector<T> events)
    {
        int n = x.Rows;
        int batchSize = Math.Max(2, Math.Min(_options.BatchSize, n));
        var adam = CreateAdamState();

        double bestLoss = double.MaxValue;
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
            double epochLoss = 0.0;
            int batches = 0;

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

                var batchTimes = new Vector<T>(count);
                var batchEvents = new Vector<T>(count);
                for (int i = 0; i < count; i++)
                {
                    batchTimes[i] = times[batchIndices[i]];
                    batchEvents[i] = events[batchIndices[i]];
                }

                bool batchHasEvent = false;
                for (int i = 0; i < count; i++)
                {
                    if (NumOps.Compare(batchEvents[i], NumOps.One) == 0)
                    {
                        batchHasEvent = true;
                        break;
                    }
                }

                if (!batchHasEvent)
                {
                    continue;
                }

                var (riskScores, cache) = ForwardPass(x, batchIndices, training: true);
                var (loss, riskGradients) = ComputeCoxLossAndGradients(riskScores, batchTimes, batchEvents);
                var gradients = Backpropagate(cache, riskGradients);
                ApplyAdamUpdate(gradients, adam);

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
                bestWeights = CloneWeights(_weights);
                bestBiases = CloneVectors(_biases);
                bestBnGamma = CloneVectors(_bnGamma);
                bestBnBeta = CloneVectors(_bnBeta);
                bestBnRunningMean = CloneVectors(_bnRunningMean);
                bestBnRunningVariance = CloneVectors(_bnRunningVariance);
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

        var grads = new ParameterGradients();
        for (int layer = 0; layer < layerCount; layer++)
        {
            grads.Weights.Add(new Matrix<T>(_weights[layer].Rows, _weights[layer].Columns));
            grads.Biases.Add(new Vector<T>(_biases[layer].Length));
        }

        for (int layer = 0; layer < _bnGamma.Count; layer++)
        {
            grads.Gamma.Add(new Vector<T>(_bnGamma[layer].Length));
            grads.Beta.Add(new Vector<T>(_bnBeta[layer].Length));
        }

        // Output layer: r_i = w . h_i + b, so dL/dw_k = sum_i g_i h_i[k] and dL/db = sum_i g_i.
        var outputInput = cache.Inputs[layerCount - 1];
        int outputInputSize = _weights[layerCount - 1].Rows;

        // delta carries dL/d(activation output) of the layer below.
        var delta = new Vector<T>[n];
        for (int i = 0; i < n; i++)
        {
            delta[i] = new Vector<T>(outputInputSize);
            for (int k = 0; k < outputInputSize; k++)
            {
                grads.Weights[layerCount - 1][k, 0] = NumOps.Add(
                    grads.Weights[layerCount - 1][k, 0],
                    NumOps.Multiply(riskGradients[i], outputInput[i][k]));
                delta[i][k] = NumOps.Multiply(riskGradients[i], _weights[layerCount - 1][k, 0]);
            }
            grads.Biases[layerCount - 1][0] = NumOps.Add(grads.Biases[layerCount - 1][0], riskGradients[i]);
        }

        // Hidden layers, in reverse.
        for (int layer = layerCount - 2; layer >= 0; layer--)
        {
            int outputSize = _weights[layer].Columns;
            int inputSize = _weights[layer].Rows;
            var layerInput = cache.Inputs[layer];
            var preActivation = cache.PreActivations[layer];
            var normalized = cache.NormalizedPreActivations[layer];
            var masks = cache.DropoutMasks[layer];

            // Through dropout and the activation, giving dL/d(normalized pre-activation).
            var dNorm = new Vector<T>[n];
            for (int i = 0; i < n; i++)
            {
                dNorm[i] = new Vector<T>(outputSize);
                for (int k = 0; k < outputSize; k++)
                {
                    T afterDropout = NumOps.Multiply(delta[i][k], masks[i][k]);
                    T actDeriv = _options.Activation.Derivative(normalized[i][k]);
                    dNorm[i][k] = NumOps.Multiply(afterDropout, actDeriv);
                }
            }

            // Through batch norm, giving dL/d(pre-activation).
            var dPre = new Vector<T>[n];
            for (int i = 0; i < n; i++)
            {
                dPre[i] = new Vector<T>(outputSize);
            }

            if (useBatchNorm)
            {
                var mean = cache.BatchMean[layer];
                var variance = cache.BatchVariance[layer];

                for (int k = 0; k < outputSize; k++)
                {
                    T std = NumOps.Sqrt(NumOps.Add(variance[k], epsilon));

                    // dL/dgamma and dL/dbeta accumulate over the batch.
                    T dGamma = NumOps.Zero;
                    T dBeta = NumOps.Zero;
                    var zhat = new T[n];
                    for (int i = 0; i < n; i++)
                    {
                        zhat[i] = NumOps.Divide(NumOps.Subtract(preActivation[i][k], mean[k]), std);
                        dGamma = NumOps.Add(dGamma, NumOps.Multiply(dNorm[i][k], zhat[i]));
                        dBeta = NumOps.Add(dBeta, dNorm[i][k]);
                    }

                    grads.Gamma[layer][k] = NumOps.Add(grads.Gamma[layer][k], dGamma);
                    grads.Beta[layer][k] = NumOps.Add(grads.Beta[layer][k], dBeta);

                    // dL/dz = gamma/(N*std) * (N*dzhat - sum(dzhat) - zhat * sum(dzhat*zhat))
                    T sumDzhat = NumOps.Zero;
                    T sumDzhatZhat = NumOps.Zero;
                    for (int i = 0; i < n; i++)
                    {
                        T dzhat = NumOps.Multiply(dNorm[i][k], _bnGamma[layer][k]);
                        sumDzhat = NumOps.Add(sumDzhat, dzhat);
                        sumDzhatZhat = NumOps.Add(sumDzhatZhat, NumOps.Multiply(dzhat, zhat[i]));
                    }

                    T invNStd = NumOps.Divide(NumOps.One, NumOps.Multiply(NumOps.FromDouble(n), std));
                    for (int i = 0; i < n; i++)
                    {
                        T dzhat = NumOps.Multiply(dNorm[i][k], _bnGamma[layer][k]);
                        T inner = NumOps.Subtract(
                            NumOps.Subtract(NumOps.Multiply(NumOps.FromDouble(n), dzhat), sumDzhat),
                            NumOps.Multiply(zhat[i], sumDzhatZhat));
                        dPre[i][k] = NumOps.Multiply(invNStd, inner);
                    }
                }
            }
            else
            {
                for (int i = 0; i < n; i++)
                {
                    for (int k = 0; k < outputSize; k++)
                    {
                        dPre[i][k] = dNorm[i][k];
                    }
                }
            }

            // Through the linear map, giving parameter gradients and the delta for the layer below.
            var nextDelta = new Vector<T>[n];
            for (int i = 0; i < n; i++)
            {
                nextDelta[i] = new Vector<T>(inputSize);
            }

            for (int i = 0; i < n; i++)
            {
                for (int k = 0; k < outputSize; k++)
                {
                    T d = dPre[i][k];
                    grads.Biases[layer][k] = NumOps.Add(grads.Biases[layer][k], d);
                    for (int j = 0; j < inputSize; j++)
                    {
                        grads.Weights[layer][j, k] = NumOps.Add(
                            grads.Weights[layer][j, k], NumOps.Multiply(d, layerInput[i][j]));
                        nextDelta[i][j] = NumOps.Add(
                            nextDelta[i][j], NumOps.Multiply(d, _weights[layer][j, k]));
                    }
                }
            }

            delta = nextDelta;
        }

        return grads;
    }

    /// <summary>
    /// Applies one Adam step with decoupled L2 weight decay to every parameter.
    /// </summary>
    /// <remarks>
    /// Weight decay is applied to the weight matrices only. Biases and the batch-norm scale and shift are
    /// left undecayed, which is the standard convention: shrinking them toward zero removes the layer's
    /// ability to shift its output rather than controlling model complexity.
    /// </remarks>
    private void ApplyAdamUpdate(ParameterGradients grads, AdamState adam)
    {
        adam.Step++;
        double lr = _options.LearningRate;
        double beta1 = 0.9;
        double beta2 = 0.999;
        double eps = 1e-8;
        double decay = _options.L2Regularization;
        double biasCorrection1 = 1.0 - Math.Pow(beta1, adam.Step);
        double biasCorrection2 = 1.0 - Math.Pow(beta2, adam.Step);

        for (int layer = 0; layer < _weights.Count; layer++)
        {
            var w = _weights[layer];
            var gw = grads.Weights[layer];
            var mw = adam.WeightM[layer];
            var vw = adam.WeightV[layer];

            for (int i = 0; i < w.Rows; i++)
            {
                for (int j = 0; j < w.Columns; j++)
                {
                    double g = NumOps.ToDouble(gw[i, j]) + decay * NumOps.ToDouble(w[i, j]);
                    double m = beta1 * NumOps.ToDouble(mw[i, j]) + (1 - beta1) * g;
                    double v = beta2 * NumOps.ToDouble(vw[i, j]) + (1 - beta2) * g * g;
                    mw[i, j] = NumOps.FromDouble(m);
                    vw[i, j] = NumOps.FromDouble(v);

                    double step = lr * (m / biasCorrection1) / (Math.Sqrt(v / biasCorrection2) + eps);
                    w[i, j] = NumOps.FromDouble(NumOps.ToDouble(w[i, j]) - step);
                }
            }

            var b = _biases[layer];
            var gb = grads.Biases[layer];
            var mb = adam.BiasM[layer];
            var vb = adam.BiasV[layer];

            for (int k = 0; k < b.Length; k++)
            {
                double g = NumOps.ToDouble(gb[k]);
                double m = beta1 * NumOps.ToDouble(mb[k]) + (1 - beta1) * g;
                double v = beta2 * NumOps.ToDouble(vb[k]) + (1 - beta2) * g * g;
                mb[k] = NumOps.FromDouble(m);
                vb[k] = NumOps.FromDouble(v);

                double step = lr * (m / biasCorrection1) / (Math.Sqrt(v / biasCorrection2) + eps);
                b[k] = NumOps.FromDouble(NumOps.ToDouble(b[k]) - step);
            }
        }

        for (int layer = 0; layer < grads.Gamma.Count; layer++)
        {
            UpdateAdamVector(_bnGamma[layer], grads.Gamma[layer], adam.GammaM[layer], adam.GammaV[layer],
                lr, beta1, beta2, eps, biasCorrection1, biasCorrection2);
            UpdateAdamVector(_bnBeta[layer], grads.Beta[layer], adam.BetaM[layer], adam.BetaV[layer],
                lr, beta1, beta2, eps, biasCorrection1, biasCorrection2);
        }
    }

    /// <summary>
    /// Applies one Adam step to a single parameter vector.
    /// </summary>
    private void UpdateAdamVector(Vector<T> parameter, Vector<T> gradient, Vector<T> m, Vector<T> v,
        double lr, double beta1, double beta2, double eps, double biasCorrection1, double biasCorrection2)
    {
        for (int k = 0; k < parameter.Length; k++)
        {
            double g = NumOps.ToDouble(gradient[k]);
            double mk = beta1 * NumOps.ToDouble(m[k]) + (1 - beta1) * g;
            double vk = beta2 * NumOps.ToDouble(v[k]) + (1 - beta2) * g * g;
            m[k] = NumOps.FromDouble(mk);
            v[k] = NumOps.FromDouble(vk);

            double step = lr * (mk / biasCorrection1) / (Math.Sqrt(vk / biasCorrection2) + eps);
            parameter[k] = NumOps.FromDouble(NumOps.ToDouble(parameter[k]) - step);
        }
    }

    /// <summary>
    /// Deep-copies the weight matrices, so early stopping can restore the best epoch.
    /// </summary>
    private static List<Matrix<T>> CloneWeights(List<Matrix<T>> source)
    {
        var copy = new List<Matrix<T>>(source.Count);
        foreach (var w in source)
        {
            var c = new Matrix<T>(w.Rows, w.Columns);
            for (int i = 0; i < w.Rows; i++)
            {
                for (int j = 0; j < w.Columns; j++)
                {
                    c[i, j] = w[i, j];
                }
            }
            copy.Add(c);
        }

        return copy;
    }

    /// <summary>
    /// Deep-copies model-state vectors, so early stopping can restore the best epoch.
    /// </summary>
    private static List<Vector<T>> CloneVectors(List<Vector<T>> source)
    {
        var copy = new List<Vector<T>>(source.Count);
        foreach (var b in source)
        {
            var c = new Vector<T>(b.Length);
            for (int i = 0; i < b.Length; i++)
            {
                c[i] = b[i];
            }
            copy.Add(c);
        }

        return copy;
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
        var riskScores = PredictRiskScores(input);
        var expected = new Vector<T>(input.Rows);

        if (_baselineHazardTimes is null || _baselineHazardValues is null || _baselineHazardTimes.Length == 0)
        {
            // No baseline hazard: fall back to the exponential model S(t) = exp(-exp(r) t), whose mean is
            // 1/exp(r). This is the same model PredictSurvival uses in that situation.
            for (int i = 0; i < input.Rows; i++)
            {
                double r = Math.Max(-20.0, Math.Min(20.0, NumOps.ToDouble(riskScores[i])));
                expected[i] = NumOps.FromDouble(1.0 / Math.Exp(r));
            }

            return expected;
        }

        int m = _baselineHazardTimes.Length;
        for (int i = 0; i < input.Rows; i++)
        {
            double r = Math.Max(-20.0, Math.Min(20.0, NumOps.ToDouble(riskScores[i])));
            double expRisk = Math.Exp(r);

            double area = 0.0;
            double prevTime = 0.0;
            double prevSurvival = 1.0;

            for (int j = 0; j < m; j++)
            {
                double t = NumOps.ToDouble(_baselineHazardTimes[j]);
                double h0 = NumOps.ToDouble(_baselineHazardValues[j]);
                double survival = Math.Exp(-h0 * expRisk);

                area += 0.5 * (prevSurvival + survival) * (t - prevTime);
                prevTime = t;
                prevSurvival = survival;
            }

            // Tail beyond the last observed event time. The instantaneous hazard there is unidentified, so
            // extend at the average rate the subject accumulated over follow-up; with S = exp(-H) that
            // gives a remaining expectation of S(last) / rate.
            double lastH0 = NumOps.ToDouble(_baselineHazardValues[m - 1]);
            double lastTime = NumOps.ToDouble(_baselineHazardTimes[m - 1]);
            double rate = lastTime > 0.0 ? (lastH0 * expRisk) / lastTime : expRisk;
            if (rate > 1e-12)
            {
                area += prevSurvival / rate;
            }
            else
            {
                area += prevSurvival * NumOps.ToDouble(_maxObservedTime);
            }

            expected[i] = NumOps.FromDouble(area);
        }

        return expected;
    }

   private void ComputeBaselineHazard(Matrix<T> x, Vector<T> times, Vector<T> events, int[] sortedIndices)
    {
        var riskScores = PredictRiskScores(x);

        var uniqueTimes = new List<T>();
        var hazardValues = new List<T>();

        T cumulativeHazard = NumOps.Zero;
        T epsilon = NumOps.FromDouble(1e-10);

        // Build each risk set with a reverse cumulative sum. Starting with the full sum and repeatedly
        // subtracting exp(risk) is numerically unsafe: the clamp still permits an exp(40) dynamic range,
        // so small tail terms can disappear when added to the full sum. Subtracting the large terms later
        // then leaves a zero or negative denominator and corrupts the cumulative hazard. Reverse summation
        // constructs the small late-time risk sets before adding the large early-time terms.
        var riskSetSums = new T[sortedIndices.Length];
        T runningRiskSum = NumOps.Zero;
        for (int position = sortedIndices.Length - 1; position >= 0; position--)
        {
            runningRiskSum = NumOps.Add(
                runningRiskSum,
                ClampedExpRisk(riskScores[sortedIndices[position]]));
            riskSetSums[position] = runningRiskSum;
        }

        // Breslow's estimator adds d(t) / sum_{j in R(t)} exp(r_j) once per distinct event time.
        // Grouping ties is both the correct estimator and ensures all subjects at a tied time remain in
        // that time's risk set.
        for (int groupStart = 0; groupStart < sortedIndices.Length;)
        {
            T t = times[sortedIndices[groupStart]];
            int groupEnd = groupStart + 1;
            while (groupEnd < sortedIndices.Length &&
                   NumOps.Compare(times[sortedIndices[groupEnd]], t) == 0)
            {
                groupEnd++;
            }

            int eventCount = 0;
            for (int position = groupStart; position < groupEnd; position++)
            {
                if (NumOps.Compare(events[sortedIndices[position]], NumOps.One) == 0)
                {
                    eventCount++;
                }
            }

            if (eventCount > 0)
            {
                cumulativeHazard = NumOps.Add(cumulativeHazard,
                    NumOps.Divide(
                        NumOps.FromDouble(eventCount),
                        NumOps.Add(riskSetSums[groupStart], epsilon)));
                uniqueTimes.Add(t);
                hazardValues.Add(cumulativeHazard);
            }

            groupStart = groupEnd;
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
            return NumOps.MaxValue;

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

        return NumOps.MaxValue;
    }

    /// <summary>
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
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NumHiddenLayers", _options.NumHiddenLayers },
                { "HiddenLayerSize", _options.HiddenLayerSize },
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
        writer.Write(_options.NumHiddenLayers);
        writer.Write(_options.HiddenLayerSize);
        writer.Write(_options.Activation.GetType().AssemblyQualifiedName ?? _options.Activation.GetType().FullName ?? _options.Activation.GetType().Name);
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

        // Batch-normalization state. This replaces the OLS coefficient block that used to be written
        // here: the model no longer fits least squares, and the running mean and variance ARE model
        // parameters -- a round-tripped network that lost them would normalize with the initial
        // mean 0 / variance 1 and predict differently from the model that was saved.
        writer.Write(NumOps.ToDouble(_maxObservedTime));

        // Feature standardization is part of the fitted model: a restored network fed raw features would
        // see inputs on a completely different scale from the ones it was trained on.
        writer.Write(_featureMean is not null && _featureStd is not null);
        if (_featureMean is not null && _featureStd is not null)
        {
            WriteBatchNormVector(writer, _featureMean);
            WriteBatchNormVector(writer, _featureStd);
        }

        writer.Write(_bnGamma.Count);
        for (int layer = 0; layer < _bnGamma.Count; layer++)
        {
            WriteBatchNormVector(writer, _bnGamma[layer]);
            WriteBatchNormVector(writer, _bnBeta[layer]);
            WriteBatchNormVector(writer, _bnRunningMean[layer]);
            WriteBatchNormVector(writer, _bnRunningVariance[layer]);
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
        string activationTypeName = reader.ReadString();
        var activationType = Type.GetType(activationTypeName);
        if (activationType is not null
            && typeof(IActivationFunction<T>).IsAssignableFrom(activationType)
            && activationType.Namespace is not null
            && activationType.Namespace.StartsWith("AiDotNet.", StringComparison.Ordinal))
        {
            _options.Activation = (IActivationFunction<T>)(Activator.CreateInstance(activationType) ?? new SELUActivation<T>());
        }
        else
        {
            _options.Activation = new SELUActivation<T>();
        }
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

        // Batch-normalization state (see Serialize).
        _maxObservedTime = NumOps.FromDouble(reader.ReadDouble());

        // Feature standardization (see Serialize).
        if (reader.ReadBoolean())
        {
            _featureMean = ReadBatchNormVector(reader);
            _featureStd = ReadBatchNormVector(reader);
        }
        else
        {
            _featureMean = null;
            _featureStd = null;
        }

        int bnLayers = reader.ReadInt32();
        _bnGamma = new List<Vector<T>>(bnLayers);
        _bnBeta = new List<Vector<T>>(bnLayers);
        _bnRunningMean = new List<Vector<T>>(bnLayers);
        _bnRunningVariance = new List<Vector<T>>(bnLayers);
        for (int layer = 0; layer < bnLayers; layer++)
        {
            _bnGamma.Add(ReadBatchNormVector(reader));
            _bnBeta.Add(ReadBatchNormVector(reader));
            _bnRunningMean.Add(ReadBatchNormVector(reader));
            _bnRunningVariance.Add(ReadBatchNormVector(reader));
        }
    }

    /// <summary>
    /// Writes one batch-normalization parameter vector.
    /// </summary>
    private void WriteBatchNormVector(BinaryWriter writer, Vector<T> v)
    {
        writer.Write(v.Length);
        for (int i = 0; i < v.Length; i++)
        {
            writer.Write(NumOps.ToDouble(v[i]));
        }
    }

    /// <summary>
    /// Reads one batch-normalization parameter vector.
    /// </summary>
    private Vector<T> ReadBatchNormVector(BinaryReader reader)
    {
        int length = reader.ReadInt32();
        var v = new Vector<T>(length);
        for (int i = 0; i < length; i++)
        {
            v[i] = NumOps.FromDouble(reader.ReadDouble());
        }

        return v;
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new DeepSurv<T>(_options, Regularization);
    }

    public override IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        var clone = new DeepSurv<T>(_options, Regularization);
        clone.Deserialize(Serialize());
        return clone;
    }
}
