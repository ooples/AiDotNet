using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.SyntheticData;

/// <summary>
/// PATE-GAN (Jordon, Yoon and van der Schaar, ICLR 2019): a GAN made differentially private by replacing its
/// discriminator with the PATE mechanism. k teachers each see one disjoint partition of the data; a student learns
/// from the teachers' Laplace-noised votes on generated rows; the generator learns only through the student.
/// </summary>
/// <remarks>
/// <para>
/// <code>
///  Partition D_1 ──► Teacher T_1 ──┐
///  Partition D_2 ──► Teacher T_2 ──┼──► PATE_λ (noisy vote) ──► labels r_j for generated rows
///       ...              ...       │
///  Partition D_k ──► Teacher T_k ──┘
///  z ~ Unif([0, 1]^d) ──► Generator G ──► Student S (trained on (G(z_j), r_j)) ──► gradient for G
/// </code>
/// </para>
/// <para>
/// This follows section 4, Algorithm 1 and the appendix:
/// <list type="bullet">
/// <item>The data are normalized into [0, 1]^d (continuous columns min-max scaled, categorical columns one-hot) and
/// the noise is z ~ Unif([0, 1]^d), so that the generator's support covers the data's (section 4.2.2).</item>
/// <item>Generator and student are depth 3 with d and d/2 hidden units, ReLU hidden layers and sigmoid outputs; the
/// teachers are depth 1 (a logistic regression).</item>
/// <item>Each generator iteration makes n_T updates of every teacher on its own partition and generated rows
/// (Eq. 6), then n_S student updates on generated rows labelled by PATE_λ (Eqs. 4 and 7), updating the moments
/// accountant for every label, then one generator update on Σ log(1 − S(G(z))). Training stops when the
/// accountant's ε̂ reaches ε.</item>
/// </list>
/// The paper adds Lap(λ) noise to the vote counts, which makes one vote (1/λ, 0)-differentially private (section 3),
/// yet writes its accountant bound (Algorithm 1, lines 17-19) with λ where Papernot et al. 2017 have the privacy
/// parameter γ of Lap(1/γ) noise. This class keeps Lap(λ) noise and evaluates the bound at γ = 1/λ, so ε̂ is the
/// guarantee the added noise actually provides.
/// </para>
/// <para>
/// <b>For Beginners:</b> PATE-GAN learns to produce realistic rows while protecting the people in the data. No single
/// network sees the whole table, and the only signal that leaves the real data is a noisy vote, so the generated rows
/// cannot reveal much about any one person. Training stops automatically once the privacy budget is used up.
/// <code>
/// var generator = new PATEGANGenerator&lt;double&gt;(architecture, new PATEGANOptions&lt;double&gt; { Epsilon = 1.0 });
/// generator.Fit(data, columns);
/// var synthetic = generator.Generate(1000);
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelCategory(ModelCategory.SyntheticDataGenerator)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("PATE-GAN: Generating Synthetic Data with Differential Privacy Guarantees",
    "https://openreview.net/forum?id=S1zk9iRqF7",
    Year = 2019,
    Authors = "James Jordon, Jinsung Yoon, Mihaela van der Schaar")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 1e-4,
                Source = "Jordon et al. 2019, appendix (Hyper-parameter optimization): \"The learning rate is "
                        + "10^-4 and we use Adam Optimizer\", batch size 64 for the generator and discriminator.")]
public partial class PATEGANGenerator<T> : NeuralSyntheticTabularGeneratorBase<T>, ISyntheticTabularGenerator<T>
{
    private readonly PATEGANOptions<T> _options;
    // One optimizer per network so no two share Adam moment state.
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _generatorOptimizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _studentOptimizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>[] _teacherOptimizers = System.Array.Empty<IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>>();

    private List<ColumnMetadata> _columns = new();
    // The [0, 1]^d encoding: per column its minimum and maximum (continuous) or category count (categorical).
    private double[] _columnMin = System.Array.Empty<double>();
    private double[] _columnMax = System.Array.Empty<double>();
    private int[] _columnWidth = System.Array.Empty<int>();
    private int _dataWidth;
    private Random _random;
    private bool _usingCustomLayers;

    // Teachers T_1..T_k: hidden layers then a sigmoid output; the student S likewise.
    private readonly List<List<FullyConnectedLayer<T>>> _teacherLayers = new();
    private readonly List<FullyConnectedLayer<T>> _studentLayers = new();

    // The moments accountant α(l), l = 1..L (Algorithm 1, line 2).
    private double[] _moments = System.Array.Empty<double>();

    /// <summary>Gets the PATE-GAN-specific options.</summary>
    public new PATEGANOptions<T> Options => _options;

    /// <inheritdoc />
    public IReadOnlyList<ColumnMetadata> Columns => _columns.AsReadOnly();

    /// <inheritdoc />
    public bool IsFitted { get; private set; }

    /// <summary>The privacy spent so far, ε̂ = min_l (α(l) + log(1/δ)) / l (Algorithm 1, line 25).</summary>
    public double PrivacySpent { get; private set; }

    /// <summary>The number of generator iterations the last <see cref="Fit"/> ran before ε was spent.</summary>
    public int IterationsTrained { get; private set; }

    /// <summary>Width of the encoded row d.</summary>
    internal int EncodedWidth => _dataWidth;

    /// <summary>The mean probability the teachers assign to <paramref name="encoded"/> rows ([n, d]) being real.</summary>
    internal double MeanTeacherScore(Tensor<T> encoded)
    {
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        return _teacherLayers.Average(teacher =>
        {
            var p = Classify(teacher, encoded);
            return Enumerable.Range(0, p.Length).Average(i => NumOps.ToDouble(p[i]));
        });
    }

    /// <summary>The encoding of <paramref name="data"/> into [0, 1]^d.</summary>
    internal Tensor<T> EncodeRows(Matrix<T> data) => Encode(data);

    /// <summary>Generated encodings for <paramref name="n"/> fresh noise draws.</summary>
    internal Tensor<T> GenerateEncoded(int n) => GenerateDetached(n);

    /// <summary>The number of teachers built.</summary>
    internal int TeacherCount => _teacherLayers.Count;

    /// <summary>Initializes a new instance with default architecture settings.</summary>
    public PATEGANGenerator()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 10,
            outputSize: 10))
    {
    }

    /// <summary>Initializes a new PATE-GAN generator with the specified architecture.</summary>
    /// <param name="architecture">Before <see cref="Fit"/>, its input size is the noise width and its output size the
    /// row width d; custom layers replace the generator. <see cref="Fit"/> rebuilds the networks for the table.</param>
    /// <param name="options">The options; defaults are the paper's.</param>
    /// <param name="optimizer">The generator's optimizer; null selects the paper's Adam.</param>
    /// <param name="lossFunction">The loss of the supervised <see cref="NeuralNetworkBase{T}.Train"/> entry point.</param>
    public PATEGANGenerator(NeuralNetworkArchitecture<T> architecture,
        PATEGANOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), (options ??= new PATEGANOptions<T>()).MaxGradNorm)
    {
        _options = options;
        base.Options = _options;
        if (_options.NumTeachers < 1)
            throw new ArgumentOutOfRangeException(nameof(options), "PATE-GAN needs at least one teacher.");
        if (_options.LaplaceScale <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "The Laplace noise scale must be positive.");
        _generatorOptimizer = optimizer ?? PaperOptimizerFactory.VerifyHandBuilt(this, CreateAdam());
        _studentOptimizer = CreateAdam();
        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();
        InitializeLayers();
    }

    private AdamOptimizer<T, Tensor<T>, Tensor<T>> CreateAdam() =>
        new(this, new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = _options.LearningRate,
            UseAdaptiveLearningRate = false,
            UseAMSGrad = false,
        });

    #region Layer Initialization

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        // The base constructor calls this before the options are set; the constructor calls it again.
        if (_options is null) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            _usingCustomLayers = true;
            Layers.Clear();
            Layers.AddRange(Architecture.Layers);
            return;
        }

        _usingCustomLayers = false;
        BuildGenerator(Architecture.OutputSize > 0 ? Architecture.OutputSize : 1);
    }

    // The paper's d and d/2 hidden units, or the configured widths.
    private static int[] HiddenWidths(int[]? configured, int d) => configured ?? new[] { d, Math.Max(1, d / 2) };

    private void BuildGenerator(int d)
    {
        Layers.Clear();
        int fanIn = NoiseWidth;
        foreach (var layer in BuildChain(fanIn, HiddenWidths(_options.GeneratorDimensions, d), d))
        {
            // Section 4.2.2: the generator is Xavier-initialized so that G([0, 1]^d) covers [0, 1]^d and some generated
            // rows look real to the teachers. The authors' code draws N(0, 2 / fan_in) weights and zero biases; a
            // smaller draw leaves every output near sigmoid(0) = 0.5 and the student never sees a realistic row.
            var weights = layer.GetWeights();
            if (weights is not null)
            {
                double std = Math.Sqrt(2.0 / fanIn);
                for (int i = 0; i < weights.Length; i++)
                {
                    double u1 = 1.0 - _random.NextDouble(), u2 = _random.NextDouble();
                    weights[i] = NumOps.FromDouble(std * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
                }
            }
            var biases = layer.GetBiases();
            if (biases is not null)
                for (int i = 0; i < biases.Length; i++) biases[i] = NumOps.Zero;
            fanIn = layer.GetOutputShape()[^1];
            Layers.Add(layer);
        }
    }

    // Explicitly sized FC layers: ReLU hidden layers, then a sigmoid output.
    private static List<FullyConnectedLayer<T>> BuildChain(int input, IEnumerable<int> hidden, int output)
    {
        var layers = new List<FullyConnectedLayer<T>>();
        int previous = input;
        foreach (int width in hidden)
        {
            layers.Add(new FullyConnectedLayer<T>(previous, width, (IActivationFunction<T>)new ReLUActivation<T>()));
            previous = width;
        }
        layers.Add(new FullyConnectedLayer<T>(previous, output, (IActivationFunction<T>)new SigmoidActivation<T>()));
        return layers;
    }

    private void BuildDiscriminators(int d)
    {
        _teacherLayers.Clear();
        for (int i = 0; i < _options.NumTeachers; i++) _teacherLayers.Add(BuildChain(d, _options.TeacherDimensions, 1));
        _studentLayers.Clear();
        _studentLayers.AddRange(BuildChain(d, HiddenWidths(_options.StudentDimensions, d), 1));
    }

    #endregion

    #region ISyntheticTabularGenerator Implementation

    /// <inheritdoc />
    /// <param name="data">The training rows.</param>
    /// <param name="columns">Column metadata; categorical columns hold category indices.</param>
    /// <param name="epochs">The maximum number of generator iterations; training stops earlier once ε is spent.
    /// Null uses <see cref="PATEGANOptions{T}.Epochs"/>.</param>
    public void Fit(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int? epochs = null)
    {
        if (data is null) throw new ArgumentNullException(nameof(data));
        if (columns is null) throw new ArgumentNullException(nameof(columns));
        int maxIterations = epochs ?? _options.Epochs;
        if (maxIterations < 0) throw new ArgumentOutOfRangeException(nameof(epochs), "Iterations cannot be negative.");
        int k = _options.NumTeachers;
        if (data.Rows < k)
            throw new ArgumentException($"PATE-GAN partitions the data among {k} teachers and needs at least that many rows.", nameof(data));
        _columns = columns.ToList();

        FitEncoding(data);
        var encoded = Encode(data);
        BuildFromConstructionSequence(() =>
        {
            if (!_usingCustomLayers) BuildGenerator(_dataWidth);
            BuildDiscriminators(_dataWidth);
        });
        _generatorOptimizer.Reset();
        _studentOptimizer = CreateAdam();
        _teacherOptimizers = Enumerable.Range(0, k).Select(_ => (IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>)CreateAdam()).ToArray();
        _moments = new double[Math.Max(1, _options.MomentOrders)];
        PrivacySpent = 0.0;
        IterationsTrained = 0;

        // Algorithm 1, line 3: k disjoint partitions of size |D| / k.
        var order = Enumerable.Range(0, data.Rows).OrderBy(_ => _random.Next()).ToArray();
        int partitionSize = data.Rows / k;
        var partitions = Enumerable.Range(0, k).Select(i => order.Skip(i * partitionSize).Take(partitionSize).ToArray()).ToArray();
        int n = Math.Max(1, _options.BatchSize);

        SetTrainingMode(true);
        try
        {
            // Lines 4-25: iterate until the privacy budget is spent.
            while (IterationsTrained < maxIterations && PrivacySpent < _options.Epsilon)
            {
                for (int step = 0; step < _options.TeacherSteps; step++)
                {
                    var fake = GenerateDetached(n);
                    for (int i = 0; i < k; i++) TrainTeacher(i, Rows(encoded, partitions[i], n), fake);
                }
                for (int step = 0; step < _options.StudentSteps; step++) TrainStudent(n);
                TrainGenerator(n);
                IterationsTrained++;
                PrivacySpent = Enumerable.Range(1, _moments.Length)
                    .Min(l => (_moments[l - 1] + Math.Log(1.0 / _options.Delta)) / l);
            }
        }
        finally
        {
            SetTrainingMode(false);
        }

        IsFitted = true;
    }

    /// <inheritdoc />
    public async Task FitAsync(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns,
        int? epochs = null, CancellationToken cancellationToken = default)
    {
        await Task.Run(() =>
        {
            cancellationToken.ThrowIfCancellationRequested();
            Fit(data, columns, epochs);
        }, cancellationToken).ConfigureAwait(false);
    }

    /// <inheritdoc />
    /// <remarks>PATE-GAN is unconditional: every row is G(z) for fresh z ~ Unif([0, 1]^n), decoded back to the
    /// table's columns, and the condition arguments are not used.</remarks>
    public Matrix<T> Generate(int numSamples, Vector<T>? conditionColumn = null, Vector<T>? conditionValue = null)
    {
        if (!IsFitted)
            throw new InvalidOperationException("Generator is not fitted. Call Fit() first.");
        if (numSamples <= 0)
            throw new ArgumentOutOfRangeException(nameof(numSamples), "Number of samples must be positive.");
        SetTrainingMode(false);
        return Decode(GenerateDetached(numSamples));
    }

    #endregion

    #region Training Steps

    /// <summary>Eq. 6: teacher i separates rows of its own partition from generated rows (cross-entropy).</summary>
    private void TrainTeacher(int teacher, Tensor<T> real, Tensor<T> fake)
    {
        var layers = _teacherLayers[teacher];
        var target = Labels(real.Shape[0], 1.0);
        var fakeTarget = Labels(fake.Shape[0], 0.0);
        AiDotNet.Training.TapeTrainingStepper<T>.EagerObjectiveStep(
            () => AiDotNet.Training.TapeTrainingStep<T>.CollectParameters(layers.Cast<ILayer<T>>().ToList()),
            () => Engine.TensorAdd(
                CrossEntropy(Classify(layers, real), target),
                CrossEntropy(Classify(layers, fake), fakeTarget)),
            _teacherOptimizers[teacher]);
    }

    /// <summary>
    /// Lines 12-21: label generated rows with PATE_λ, update the moments accountant for each label, and train the
    /// student on the labelled rows (Eq. 7).
    /// </summary>
    private void TrainStudent(int n)
    {
        var fake = GenerateDetached(n);
        var votes = new int[n];
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
        {
            foreach (var teacher in _teacherLayers)
            {
                var p = Classify(teacher, fake);
                for (int j = 0; j < n; j++)
                    if (NumOps.ToDouble(p[j, 0]) > 0.5) votes[j]++;
            }
        }

        var labels = new Tensor<T>(new[] { n, 1 });
        int k = _teacherLayers.Count;
        for (int j = 0; j < n; j++)
        {
            int n1 = votes[j], n0 = k - n1;
            // Eq. 4: argmax over the classes of the Laplace-noised counts.
            labels[j, 0] = NumOps.FromDouble(n1 + Laplace(_options.LaplaceScale) > n0 + Laplace(_options.LaplaceScale) ? 1.0 : 0.0);
            AccountQuery(Math.Abs(n0 - n1));
        }

        AiDotNet.Training.TapeTrainingStepper<T>.EagerObjectiveStep(
            () => AiDotNet.Training.TapeTrainingStep<T>.CollectParameters(_studentLayers.Cast<ILayer<T>>().ToList()),
            () => CrossEntropy(Classify(_studentLayers, fake), labels),
            _studentOptimizer);
    }

    /// <summary>Line 24: the generator descends Σ log(1 − S(G(z))).</summary>
    private void TrainGenerator(int n)
    {
        var z = Noise(n);
        var generatorLayers = Layers.ToList();
        var one = NumOps.One;
        LastLoss = AiDotNet.Training.TapeTrainingStepper<T>.EagerObjectiveStep(
            () => AiDotNet.Training.TapeTrainingStep<T>.CollectParameters(generatorLayers),
            () =>
            {
                var s = Classify(_studentLayers, Forward(z));
                var log = Engine.TensorLog(Engine.TensorAddScalar(Engine.ScalarMinusTensor(one, s), NumOps.FromDouble(1e-8)));
                return Engine.ReduceMean(log, new[] { 0, 1 }, keepDims: false);
            },
            _generatorOptimizer,
            modelGradientClip: _options.MaxGradNorm);
    }

    /// <summary>
    /// Lines 17-19 for one query with vote gap |n0 − n1|, at the privacy parameter γ = 1/λ of the Lap(λ) noise:
    /// α(l) += min{2γ²l(l+1), log((1−q)((1−q)/(1−e^{2γ}q))^l + q e^{2γl})}, q = (2 + γ|n0−n1|) / (4 exp(γ|n0−n1|)).
    /// The data-dependent term only applies while 1 − e^{2γ}q is positive.
    /// </summary>
    private void AccountQuery(int gap)
    {
        double gamma = 1.0 / _options.LaplaceScale;
        for (int l = 1; l <= _moments.Length; l++) _moments[l - 1] += MomentBound(gamma, gap, l);
    }

    /// <summary>The bound on the l-th moment of one noisy vote with gap |n0 − n1| at privacy parameter γ.</summary>
    internal static double MomentBound(double gamma, int gap, int l)
    {
        double q = (2.0 + gamma * gap) / (4.0 * Math.Exp(gamma * gap));
        double bound = 2.0 * gamma * gamma * l * (l + 1);
        double denominator = 1.0 - Math.Exp(2.0 * gamma) * q;
        if (q < 1.0 && denominator > 0)
        {
            double dataDependent = Math.Log((1 - q) * Math.Pow((1 - q) / denominator, l) + q * Math.Exp(2.0 * gamma * l));
            if (!double.IsNaN(dataDependent)) bound = Math.Min(bound, dataDependent);
        }
        return bound;
    }

    #endregion

    #region Networks

    private Tensor<T> Forward(Tensor<T> input)
    {
        var current = input;
        foreach (var layer in Layers) current = layer.Forward(current);
        return current;
    }

    private static Tensor<T> Classify(IEnumerable<FullyConnectedLayer<T>> layers, Tensor<T> input)
    {
        var current = input;
        foreach (var layer in layers) current = layer.Forward(current);
        return current;
    }

    // −mean(y log p + (1 − y) log(1 − p)).
    private Tensor<T> CrossEntropy(Tensor<T> p, Tensor<T> y)
    {
        var eps = NumOps.FromDouble(1e-8);
        var one = NumOps.One;
        var positive = Engine.TensorMultiply(y, Engine.TensorLog(Engine.TensorAddScalar(p, eps)));
        var negative = Engine.TensorMultiply(Engine.ScalarMinusTensor(one, y),
            Engine.TensorLog(Engine.TensorAddScalar(Engine.ScalarMinusTensor(one, p), eps)));
        return Engine.TensorNegate(Engine.ReduceMean(Engine.TensorAdd(positive, negative), new[] { 0, 1 }, keepDims: false));
    }

    private Tensor<T> GenerateDetached(int n)
    {
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        return Forward(Noise(n));
    }

    private int NoiseWidth => _options.EmbeddingDimension
        ?? (_dataWidth > 0 ? _dataWidth : (Architecture.CalculatedInputSize > 0 ? Architecture.CalculatedInputSize : 1));

    // z ~ Unif([0, 1]^n) (section 4.2.2).
    private Tensor<T> Noise(int n)
    {
        var z = new Tensor<T>(new[] { n, NoiseWidth });
        for (int i = 0; i < z.Length; i++) z[i] = NumOps.FromDouble(_random.NextDouble());
        return z;
    }

    private Tensor<T> Labels(int n, double value)
    {
        var y = new Tensor<T>(new[] { n, 1 });
        for (int i = 0; i < n; i++) y[i, 0] = NumOps.FromDouble(value);
        return y;
    }

    // Laplace(0, scale) by inverse transform sampling.
    private double Laplace(double scale)
    {
        double u = _random.NextDouble() - 0.5;
        return -scale * Math.Sign(u) * Math.Log(1.0 - 2.0 * Math.Abs(u) + double.Epsilon);
    }

    // n rows drawn with replacement from one partition (Algorithm 1, line 8).
    private Tensor<T> Rows(Tensor<T> encoded, int[] partition, int n)
    {
        var rows = new Tensor<T>(new[] { n, _dataWidth });
        for (int b = 0; b < n; b++)
        {
            int row = partition[_random.Next(partition.Length)];
            for (int j = 0; j < _dataWidth; j++) rows[b, j] = encoded[row, j];
        }
        return rows;
    }

    #endregion

    #region [0, 1] Encoding

    private bool IsCategorical(int col) => _columns[col].DataType == ColumnDataType.Categorical;

    private void FitEncoding(Matrix<T> data)
    {
        int count = _columns.Count;
        _columnMin = new double[count];
        _columnMax = new double[count];
        _columnWidth = new int[count];
        for (int c = 0; c < count; c++)
        {
            int source = SourceColumn(c, data);
            var values = Enumerable.Range(0, data.Rows).Select(i => NumOps.ToDouble(data[i, source])).ToArray();
            _columnMin[c] = values.Min();
            _columnMax[c] = values.Max();
            _columnWidth[c] = IsCategorical(c)
                ? Math.Max(_columns[c].Categories.Count, (int)Math.Round(_columnMax[c]) + 1)
                : 1;
        }
        _dataWidth = _columnWidth.Sum();
    }

    private int SourceColumn(int c, Matrix<T> data)
        => _columns[c].ColumnIndex >= 0 && _columns[c].ColumnIndex < data.Columns ? _columns[c].ColumnIndex : c;

    private Tensor<T> Encode(Matrix<T> data)
    {
        var encoded = new Tensor<T>(new[] { data.Rows, _dataWidth });
        for (int i = 0; i < data.Rows; i++)
        {
            int offset = 0;
            for (int c = 0; c < _columns.Count; c++)
            {
                double value = NumOps.ToDouble(data[i, SourceColumn(c, data)]);
                if (IsCategorical(c))
                {
                    int category = Math.Min(Math.Max((int)Math.Round(value), 0), _columnWidth[c] - 1);
                    encoded[i, offset + category] = NumOps.One;
                }
                else
                {
                    double range = _columnMax[c] - _columnMin[c];
                    encoded[i, offset] = NumOps.FromDouble(range > 0 ? (value - _columnMin[c]) / range : 0.5);
                }
                offset += _columnWidth[c];
            }
        }
        return encoded;
    }

    private Matrix<T> Decode(Tensor<T> encoded)
    {
        int n = encoded.Shape[0];
        var rows = new Matrix<T>(n, _columns.Count);
        for (int i = 0; i < n; i++)
        {
            int offset = 0;
            for (int c = 0; c < _columns.Count; c++)
            {
                if (IsCategorical(c))
                {
                    int best = 0;
                    for (int j = 1; j < _columnWidth[c]; j++)
                        if (NumOps.ToDouble(encoded[i, offset + j]) > NumOps.ToDouble(encoded[i, offset + best])) best = j;
                    rows[i, c] = NumOps.FromDouble(best);
                }
                else
                {
                    double v = Math.Min(Math.Max(NumOps.ToDouble(encoded[i, offset]), 0.0), 1.0);
                    rows[i, c] = NumOps.FromDouble(_columnMin[c] + v * (_columnMax[c] - _columnMin[c]));
                }
                offset += _columnWidth[c];
            }
        }
        return rows;
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                ["GeneratorType"] = "PATEGANGenerator",
                ["NumTeachers"] = _options.NumTeachers,
                ["LaplaceScale"] = _options.LaplaceScale,
                ["Epsilon"] = _options.Epsilon,
                ["Delta"] = _options.Delta,
                ["PrivacySpent"] = PrivacySpent,
                ["IterationsTrained"] = IterationsTrained,
                ["BatchSize"] = _options.BatchSize,
                ["LearningRate"] = _options.LearningRate,
                ["IsFitted"] = IsFitted,
                ["DataWidth"] = _dataWidth,
                ["UsingCustomLayers"] = _usingCustomLayers,
            }
        };
    }

    /// <inheritdoc />
    public override Dictionary<string, T> GetFeatureImportance()
    {
        var importance = new Dictionary<string, T>();
        T equal = NumOps.FromDouble(1.0 / Math.Max(_columns.Count, 1));
        foreach (var column in _columns) importance[column.Name] = equal;
        return importance;
    }

    /// <inheritdoc />
    /// <remarks>The input is the noise z, <c>[n]</c> or <c>[batch, n]</c>; the output is the encoded row in [0, 1]^d.</remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        bool single = input.Rank == 1;
        var x = single ? Engine.Reshape(input, new[] { 1, input.Length }) : input;
        var output = Forward(x);
        return single ? Engine.Reshape(output, new[] { output.Length }) : output;
    }

    /// <inheritdoc />
    /// <remarks>
    /// The private adversarial training of Algorithm 1 runs in <see cref="Fit"/>. This single-step entry point
    /// regresses the generator's output for the given noise onto the target (mean squared error), for supervised use.
    /// </remarks>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expectedOutput)
    {
        var generatorLayers = Layers.ToList();
        SetTrainingMode(true);
        try
        {
            LastLoss = AiDotNet.Training.TapeTrainingStepper<T>.EagerObjectiveStep(
                () => AiDotNet.Training.TapeTrainingStep<T>.CollectParameters(generatorLayers),
                () =>
                {
                    var output = PredictCore(input);
                    var diff = Engine.TensorSubtract(
                        Engine.Reshape(output, new[] { output.Length }),
                        Engine.Reshape(expectedOutput, new[] { expectedOutput.Length }));
                    return Engine.ReduceMean(Engine.TensorSquare(diff), new[] { 0 }, keepDims: false);
                },
                _generatorOptimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
        return true;
    }

    #endregion
}
