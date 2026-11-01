using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.SyntheticData;

/// <summary>
/// OCT-GAN (Kim et al. 2021): a conditional tabular GAN whose generator first maps the noise and condition vector
/// <c>z ⊕ c</c> through a neural ODE, and whose discriminator classifies the trajectory of a neural ODE sampled at
/// learnable time points.
/// </summary>
/// <remarks>
/// <para>
/// The networks are the paper's (section 3, dimensions from section 4.1.4):
/// <list type="bullet">
/// <item><b>Generator</b> (Eqs. 10-17): <c>p(0) = z ⊕ c</c>, <c>z' = p(0) + ∫₀¹ g(p(t), t) dt</c> with
/// <c>g = Leaky(FC13(... Leaky(FC6(Norm(p) ⊕ t))))</c> (eight layers, <c>Norm(p) = p / ‖p‖₂</c>); then two residual
/// blocks <c>h(0) = z' ⊕ ReLU(BN(FC1 z'))</c>, <c>h(1) = h(0) ⊕ ReLU(BN(FC2 h(0)))</c>; then tanh heads for the
/// mode-normalized values and Gumbel-softmax heads for the modes and the discrete columns.</item>
/// <item><b>Discriminator</b> (Eqs. 2-9): <c>h(0) = Drop(Leaky(FC2(Drop(Leaky(FC1 x)))))</c>; <c>h(t_i)</c> integrates
/// <c>f = ReLU(BN(FC7(ReLU(BN(FC6(ReLU(BN(h)) ⊕ t))))))</c> from <c>t_{i-1}</c> to <c>t_i</c>; the trajectory
/// <c>h_x = h(0) ⊕ h(t_1) ⊕ ... ⊕ h(t_m)</c> is scored by <c>FC5(Leaky(FC4(Leaky(FC3 h_x))))</c>.</item>
/// </list>
/// </para>
/// <para>
/// Training is Algorithm 1: per mini-batch, a WGAN-GP discriminator step that also moves the time points along
/// <c>∇_{t_i} L_D</c>, then a generator step on <c>L_G + L_matching</c> (the cross-entropy between the condition and
/// the generated discrete column); every learning rate decays by ψ every ξ epochs. As in the authors' released code,
/// t_0 = 0 and t_m = 1 are fixed, the interior t_1..t_{m-1} share the critic's Adam, and each is clamped between its
/// neighbours after the step.
/// The authors solve the ODEs with Dormand-Prince and train the time points with the adjoint method; this class uses
/// a fixed-step RK4 (<see cref="OCTGANOptions{T}.OdeSolverSteps"/>) and differentiates through it, which gives the
/// gradient of Proposition 3.1 for the discretized trajectory. The discriminator reads the row with its condition
/// vector, as in the CTGAN code the paper builds on.
/// </para>
/// <para>
/// <b>For Beginners:</b> A generator invents table rows and a discriminator judges whether rows are real. OCT-GAN
/// lets a hidden vector "flow" smoothly over time in both networks. The generator's flow turns random noise into a
/// better starting point, so similar noise gives similar rows; the discriminator watches its flow at a few moments it
/// picks itself and judges the row from the whole path.
/// <code>
/// var generator = new OCTGANGenerator&lt;double&gt;(architecture, new OCTGANOptions&lt;double&gt; { Epochs = 300 });
/// generator.Fit(data, columns);
/// var synthetic = generator.Generate(1000);
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.SyntheticDataGenerator)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("OCT-GAN: Neural ODE-based Conditional Tabular GANs",
    "https://arxiv.org/abs/2105.14969",
    Year = 2021,
    Authors = "Jayoung Kim, Jinsung Jeon, Jaehoon Lee, Jihyeon Hyeong, Noseong Park")]
public partial class OCTGANGenerator<T> : NeuralSyntheticTabularGeneratorBase<T>, ISyntheticTabularGenerator<T>
{
    /// <summary>The generator ODE function g has eight FC + Leaky layers, FC6 to FC13 (Eq. 17).</summary>
    private const int GeneratorOdeDepth = 8;
    private const double LeakySlope = 0.2;
    private const int DefaultDataDimension = 128;

    private readonly OCTGANOptions<T> _options;
    // Separate G/D optimizers so the two players never share Adam moment state.
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _generatorOptimizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _discriminatorOptimizer;

    private TabularDataTransformer<T>? _transformer;
    private CTGANDataSampler<T>? _sampler;
    private List<ColumnMetadata> _columns = new();
    private readonly List<(int Offset, int Width)> _catOutputBlocks = new();
    private int _dataWidth;
    private int _condWidth;
    private Random _random;
    private bool _usingCustomLayers;

    // Generator: Layers holds FC6..FC13 of g, FC1, FC2 and the output head; the residual blocks' BN live here.
    private readonly List<BatchNormalizationLayer<T>> _genBNLayers = new();

    // Discriminator: FC1, FC2 (Eq. 2) and FC3, FC4, FC5 (Eq. 8); FC6, FC7 and the three BN of f (Eq. 9).
    private readonly List<FullyConnectedLayer<T>> _discLayers = new();
    private readonly List<DropoutLayer<T>> _discDropoutLayers = new();
    private readonly List<FullyConnectedLayer<T>> _discOdeLayers = new();
    private readonly List<BatchNormalizationLayer<T>> _discOdeNormLayers = new();

    /// <summary>The learnable interior time points t_1..t_{m-1} of the discriminator's ODE (t_0 = 0, t_m = 1).</summary>
    [TrainableParameter]
    private Tensor<T> _timePoints;

    /// <summary>Gets the OCT-GAN-specific options.</summary>
    public new OCTGANOptions<T> Options => _options;

    /// <inheritdoc />
    public IReadOnlyList<ColumnMetadata> Columns => _columns.AsReadOnly();

    /// <inheritdoc />
    public bool IsFitted { get; private set; }

    /// <summary>The discriminator's current interior time points t_1..t_{m-1}.</summary>
    internal double[] TimePointValues => Enumerable.Range(0, _timePoints.Length).Select(i => NumOps.ToDouble(_timePoints[i])).ToArray();

    /// <summary>Width of the discriminator input, a row with its condition vector.</summary>
    internal int DiscriminatorInputWidth => _dataWidth + _condWidth;

    /// <summary>The generator optimizer's current learning rate λ_G.</summary>
    internal double GeneratorLearningRate => _generatorOptimizer.GetCurrentLearningRate();

    /// <summary>
    /// The derivative of the mean evaluation-mode score of <paramref name="rows"/> (rows with their conditions) with
    /// respect to time point <paramref name="index"/>: through the solver, and by central differences.
    /// </summary>
    internal (double Analytic, double Numeric) TimePointGradientCheck(Tensor<T> rows, int index, double epsilon)
    {
        double Score()
        {
            using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
            var s = DiscriminatorForward(rows, training: false);
            return Enumerable.Range(0, s.Length).Average(i => NumOps.ToDouble(s[i]));
        }

        double analytic;
        using (var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<T>())
        {
            var score = Engine.ReduceMean(DiscriminatorForward(rows, training: false), new[] { 0, 1 }, keepDims: false);
            var gradients = tape.ComputeGradients(score, new[] { _timePoints });
            analytic = gradients.TryGetValue(_timePoints, out var g) ? NumOps.ToDouble(g[index]) : 0.0;
        }
        var original = _timePoints[index];
        _timePoints[index] = NumOps.Add(original, NumOps.FromDouble(epsilon));
        double plus = Score();
        _timePoints[index] = NumOps.Subtract(original, NumOps.FromDouble(epsilon));
        double minus = Score();
        _timePoints[index] = original;
        return (analytic, (plus - minus) / (2 * epsilon));
    }

    /// <summary>Width of the generator input <c>p(0) = z ⊕ c</c>.</summary>
    internal int GeneratorInputWidth => _usingCustomLayers || Layers.Count < GeneratorOdeDepth
        ? Architecture.CalculatedInputSize
        : Layers[GeneratorOdeDepth - 1].GetOutputShape()[^1];

    /// <summary>Initializes a new OCT-GAN generator with default configuration.</summary>
    public OCTGANGenerator()
        : this(new NeuralNetworkArchitecture<T>(InputType.OneDimensional, NeuralNetworkTaskType.Generative, inputSize: DefaultDataDimension, outputSize: DefaultDataDimension))
    {
    }

    /// <summary>Initializes a new OCT-GAN generator with the specified architecture.</summary>
    /// <param name="architecture">Before <see cref="Fit"/>, its input size is the width of <c>z ⊕ c</c> and its
    /// output size the row width; <see cref="Fit"/> rebuilds both networks for the table.</param>
    /// <param name="options">The options; defaults are the paper's.</param>
    /// <param name="optimizer">The generator's optimizer; null selects Adam with the paper's learning-rate decay.</param>
    /// <param name="lossFunction">The loss of the supervised <see cref="NeuralNetworkBase{T}.Train"/> entry point.</param>
    public OCTGANGenerator(NeuralNetworkArchitecture<T> architecture,
        OCTGANOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), (options ??= new OCTGANOptions<T>()).MaxGradNorm)
    {
        _options = options;
        if (_options.TimePoints < 2)
            throw new ArgumentOutOfRangeException(nameof(options), "OCT-GAN needs at least two ODE segments (one learnable time point).");
        if (_options.OdeSolverSteps < 1)
            throw new ArgumentOutOfRangeException(nameof(options), "Each ODE integral needs at least one solver step.");
        _generatorOptimizer = optimizer ?? CreateAdam();
        _discriminatorOptimizer = CreateAdam();
        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();
        _timePoints = EvenlySpacedTimePoints(_options.TimePoints);
        InitializeLayers();
    }

    // Adam with CTGAN's betas; λ decays by ψ every ξ epochs (section 4.1.4), stepped by OnEpochEnd.
    private AdamOptimizer<T, Tensor<T>, Tensor<T>> CreateAdam() =>
        new(this, new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = _options.LearningRate,
            Beta1 = 0.5,
            Beta2 = 0.9,
            UseAdaptiveLearningRate = false,
            UseAMSGrad = false,
            LearningRateScheduler = new StepLRScheduler(_options.LearningRate, Math.Max(1, _options.LearningRateDecayPeriod), _options.LearningRateDecay),
            SchedulerStepMode = SchedulerStepMode.StepPerEpoch,
        });

    // The interior points t_1..t_{m-1}; t_0 = 0 and t_m = 1 are fixed. OCT-GAN(fixed)'s t_i = i/m is also the
    // initialization of the learned points.
    private Tensor<T> EvenlySpacedTimePoints(int m)
    {
        var t = new Tensor<T>(new[] { m - 1 });
        for (int i = 0; i < m - 1; i++) t[i] = NumOps.FromDouble((i + 1.0) / m);
        return t;
    }

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
        int inputWidth = Architecture.CalculatedInputSize > 0 ? Architecture.CalculatedInputSize : _options.EmbeddingDimension;
        BuildGenerator(inputWidth, Architecture.OutputSize > 0 ? Architecture.OutputSize : 1);
    }

    private void BuildGenerator(int inputWidth, int outputWidth)
    {
        Layers.Clear();
        _genBNLayers.Clear();
        var identity = (IActivationFunction<T>)new IdentityActivation<T>();
        // g: FC6 (dim p + 1 -> dim p), FC7..FC13 (dim p -> dim p).
        for (int i = 0; i < GeneratorOdeDepth; i++)
            Layers.Add(new FullyConnectedLayer<T>(i == 0 ? inputWidth + 1 : inputWidth, inputWidth, identity));
        foreach (int width in _options.GeneratorDimensions.Take(2))
        {
            Layers.Add(new FullyConnectedLayer<T>(width, identity));
            _genBNLayers.Add(new BatchNormalizationLayer<T>());
        }
        // FC3, FC4 and FC5 read the same h(1); one layer holds all three heads, sliced per column.
        Layers.Add(new FullyConnectedLayer<T>(outputWidth, identity));
    }

    private void BuildDiscriminator(int inputWidth)
    {
        _discLayers.Clear();
        _discDropoutLayers.Clear();
        _discOdeLayers.Clear();
        _discOdeNormLayers.Clear();
        var identity = (IActivationFunction<T>)new IdentityActivation<T>();
        var dims = _options.DiscriminatorDimensions.Length > 0 ? _options.DiscriminatorDimensions : new[] { 256 };
        int previous = inputWidth;
        foreach (int width in dims)
        {
            _discLayers.Add(new FullyConnectedLayer<T>(previous, width, identity));
            _discDropoutLayers.Add(new DropoutLayer<T>(_options.DiscriminatorDropout));
            previous = width;
        }
        int stateWidth = previous;
        _discOdeNormLayers.Add(new BatchNormalizationLayer<T>());
        _discOdeLayers.Add(new FullyConnectedLayer<T>(stateWidth + 1, stateWidth, identity));
        _discOdeNormLayers.Add(new BatchNormalizationLayer<T>());
        _discOdeLayers.Add(new FullyConnectedLayer<T>(stateWidth, stateWidth, identity));
        _discOdeNormLayers.Add(new BatchNormalizationLayer<T>());
        int trajectoryWidth = (_options.TimePoints + 1) * stateWidth;
        _discLayers.Add(new FullyConnectedLayer<T>(trajectoryWidth, 2 * trajectoryWidth, identity));
        _discLayers.Add(new FullyConnectedLayer<T>(2 * trajectoryWidth, trajectoryWidth, identity));
        _discLayers.Add(new FullyConnectedLayer<T>(trajectoryWidth, 1, identity));
    }

    /// <inheritdoc />
    /// <remarks>The generator is not a chain: g is integrated and the residual blocks concatenate.</remarks>
    protected override void ResolveLazyLayerShapes()
    {
        if (_usingCustomLayers) return;
        PredictCore(new Tensor<T>(new[] { 1, GeneratorInputWidth }));
    }

    #endregion

    #region ISyntheticTabularGenerator Implementation

    /// <inheritdoc />
    public void Fit(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int? epochs = null)
    {
        if (data is null) throw new ArgumentNullException(nameof(data));
        if (columns is null) throw new ArgumentNullException(nameof(columns));
        if (data.Rows == 0) throw new ArgumentException("OCT-GAN needs at least one training row.", nameof(data));
        int epochCount = epochs ?? _options.Epochs;
        if (epochCount < 0) throw new ArgumentOutOfRangeException(nameof(epochs), "Epochs cannot be negative.");
        _columns = columns.ToList();

        // Section 3.1: mode-specific normalization; section 3.3: conditions from the discrete columns.
        _transformer = new TabularDataTransformer<T>(_options.VGMModes, _random);
        _transformer.Fit(data, _columns);
        _dataWidth = _transformer.TransformedWidth;
        BuildCategoricalOutputBlocks();
        _sampler = new CTGANDataSampler<T>(_random);
        _sampler.Fit(data, _columns);
        _condWidth = _sampler.ConditionalVectorWidth;

        if (!_usingCustomLayers) BuildGenerator(_options.EmbeddingDimension + _condWidth, _dataWidth);
        BuildDiscriminator(_dataWidth + _condWidth);
        _timePoints = EvenlySpacedTimePoints(_options.TimePoints);
        _generatorOptimizer.Reset();
        _discriminatorOptimizer = CreateAdam();

        var transformed = _transformer.Transform(data);
        int batchSize = Math.Max(1, Math.Min(_options.BatchSize, data.Rows));
        int stepsPerEpoch = Math.Max(1, data.Rows / batchSize);
        SetTrainingMode(true);
        try
        {
            for (int epoch = 0; epoch < epochCount; epoch++)
            {
                for (int step = 0; step < stepsPerEpoch; step++)
                {
                    // Algorithm 1, lines 5-7.
                    for (int d = 0; d < Math.Max(1, _options.DiscriminatorSteps); d++)
                        TrainDiscriminatorStep(transformed, batchSize);
                    TrainGeneratorStep(batchSize);
                }

                // Line 9: anneal λ_G and λ_D (whose optimizer also steps the time points) by ψ every ξ epochs.
                if (_generatorOptimizer is GradientBasedOptimizerBase<T, Tensor<T>, Tensor<T>> generatorBase) generatorBase.OnEpochEnd();
                if (_discriminatorOptimizer is GradientBasedOptimizerBase<T, Tensor<T>, Tensor<T>> discriminatorBase) discriminatorBase.OnEpochEnd();
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
    /// <remarks>Each row gets fresh noise and a condition: the requested category when <paramref name="conditionColumn"/>
    /// (discrete-column index) and <paramref name="conditionValue"/> (category index) are given for it, otherwise one
    /// drawn from the training distribution. Modes and categories are Gumbel-softmax samples (Eqs. 15-16).</remarks>
    public Matrix<T> Generate(int numSamples, Vector<T>? conditionColumn = null, Vector<T>? conditionValue = null)
    {
        if (!IsFitted || _transformer is null || _sampler is null)
            throw new InvalidOperationException("Generator is not fitted. Call Fit() first.");
        if (numSamples <= 0)
            throw new ArgumentOutOfRangeException(nameof(numSamples), "Number of samples must be positive.");

        var rows = new Matrix<T>(numSamples, _dataWidth);
        int chunk = Math.Max(1, _options.BatchSize);
        SetTrainingMode(false);
        for (int start = 0; start < numSamples; start += chunk)
        {
            int count = Math.Min(chunk, numSamples - start);
            var cond = new Tensor<T>(new[] { count, _condWidth });
            for (int b = 0; b < count; b++)
            {
                int i = start + b;
                var cv = conditionColumn is not null && conditionValue is not null && i < conditionColumn.Length && i < conditionValue.Length
                    ? _sampler.CreateConditionVector((int)NumOps.ToDouble(conditionColumn[i]), (int)NumOps.ToDouble(conditionValue[i]))
                    : _sampler.SampleRandomConditionVector();
                for (int j = 0; j < _condWidth && j < cv.Length; j++) cond[b, j] = cv[j];
            }
            var input = _condWidth > 0 ? Engine.TensorConcatenate(new[] { Noise(count), cond }, axis: 1) : Noise(count);
            var fake = GeneratorForward(input, gumbel: true);
            for (int b = 0; b < count; b++)
                for (int j = 0; j < _dataWidth; j++) rows[start + b, j] = fake[b, j];
        }
        return _transformer.InverseTransform(rows);
    }

    #endregion

    #region Forward Passes

    /// <inheritdoc />
    public override void SetTrainingMode(bool isTraining)
    {
        base.SetTrainingMode(isTraining);
        foreach (var bn in _genBNLayers) bn.SetTrainingMode(isTraining);
        foreach (var bn in _discOdeNormLayers) bn.SetTrainingMode(isTraining);
        foreach (var drop in _discDropoutLayers) drop.SetTrainingMode(isTraining);
    }

    /// <summary>The generator from <c>p(0) = z ⊕ c</c> ([batch, dim p]) to an activated row ([batch, row width]).</summary>
    private Tensor<T> GeneratorForward(Tensor<T> input, bool gumbel)
    {
        if (_usingCustomLayers)
        {
            var current = input;
            foreach (var layer in Layers) current = layer.Forward(current);
            return ApplyOutputActivations(current, gumbel);
        }

        // Eq. 11: z' = p(0) + ∫₀¹ g(p(t), t) dt.
        var zPrime = Integrate(input, Constant(0.0), Constant(1.0), GeneratorOdeFunction);
        // Eqs. 12-13: two residual blocks that concatenate their input.
        var h = zPrime;
        for (int i = 0; i < _genBNLayers.Count; i++)
        {
            var block = Engine.ReLU(_genBNLayers[i].Forward(Layers[GeneratorOdeDepth + i].Forward(h)));
            h = Engine.TensorConcatenate(new[] { h, block }, axis: 1);
        }
        // Eqs. 14-16.
        return ApplyOutputActivations(Layers[^1].Forward(h), gumbel);
    }

    /// <summary>Eq. 17: g(p, t) = Leaky(FC13(... Leaky(FC6(Norm(p) ⊕ t)))) with Norm(p) = p / ‖p‖₂.</summary>
    private Tensor<T> GeneratorOdeFunction(Tensor<T> p, Tensor<T> timeColumn)
    {
        int width = p.Shape[1];
        var norm = Engine.TensorSqrt(Engine.TensorAddScalar(
            Engine.ReduceSum(Engine.TensorSquare(p), new[] { 1 }, keepDims: true), NumOps.FromDouble(1e-12)));
        var x = Engine.TensorConcatenate(new[] { Engine.TensorDivide(p, Engine.TensorTile(norm, new[] { 1, width })), timeColumn }, axis: 1);
        for (int i = 0; i < GeneratorOdeDepth; i++)
            x = Engine.TensorLeakyReLU(Layers[i].Forward(x), NumOps.FromDouble(LeakySlope));
        return x;
    }

    /// <summary>The discriminator from a row with its condition ([batch, row + condition width]) to a score.</summary>
    private Tensor<T> DiscriminatorForward(Tensor<T> input, bool training)
    {
        foreach (var drop in _discDropoutLayers) drop.SetTrainingMode(training);
        foreach (var bn in _discOdeNormLayers) bn.SetTrainingMode(training);
        int hidden = _discDropoutLayers.Count;

        // Eq. 2.
        var h = input;
        for (int i = 0; i < hidden; i++)
        {
            h = Engine.TensorLeakyReLU(_discLayers[i].Forward(h), NumOps.FromDouble(LeakySlope));
            if (training) h = _discDropoutLayers[i].Forward(h);
        }

        // Eqs. 3-7: one ODE system on [0, 1], read at the learnable t_1..t_{m-1} and at t_m = 1.
        var trajectory = new List<Tensor<T>> { h };
        var previous = Constant(0.0);
        for (int i = 0; i <= _timePoints.Length; i++)
        {
            var next = i < _timePoints.Length ? Engine.TensorSlice(_timePoints, new[] { i }, new[] { 1 }) : Constant(1.0);
            h = Integrate(h, previous, next, DiscriminatorOdeFunction);
            trajectory.Add(h);
            previous = next;
        }

        // Eq. 8.
        var score = Engine.TensorConcatenate(trajectory.ToArray(), axis: 1);
        for (int i = hidden; i < _discLayers.Count; i++)
        {
            score = _discLayers[i].Forward(score);
            if (i < _discLayers.Count - 1) score = Engine.TensorLeakyReLU(score, NumOps.FromDouble(LeakySlope));
        }
        return score;
    }

    /// <summary>Eq. 9: f(h, t) = ReLU(BN(FC7(ReLU(BN(FC6(ReLU(BN(h)) ⊕ t)))))).</summary>
    private Tensor<T> DiscriminatorOdeFunction(Tensor<T> h, Tensor<T> timeColumn)
    {
        var x = Engine.ReLU(_discOdeNormLayers[0].Forward(h));
        x = Engine.ReLU(_discOdeNormLayers[1].Forward(_discOdeLayers[0].Forward(Engine.TensorConcatenate(new[] { x, timeColumn }, axis: 1))));
        return Engine.ReLU(_discOdeNormLayers[2].Forward(_discOdeLayers[1].Forward(x)));
    }

    /// <summary>
    /// <c>state(t1) = state(t0) + ∫ field(state, t) dt</c> by <see cref="OCTGANOptions{T}.OdeSolverSteps"/> classical
    /// RK4 steps. The bounds are one-element tensors, so the result is differentiable with respect to them.
    /// </summary>
    private Tensor<T> Integrate(Tensor<T> state, Tensor<T> t0, Tensor<T> t1, Func<Tensor<T>, Tensor<T>, Tensor<T>> field)
    {
        int steps = _options.OdeSolverSteps;
        int batch = state.Shape[0], width = state.Shape[1];
        var dt = Engine.TensorMultiplyScalar(Engine.TensorSubtract(t1, t0), NumOps.FromDouble(1.0 / steps));
        var dtState = Engine.TensorTile(Engine.Reshape(dt, new[] { 1, 1 }), new[] { batch, width });
        var halfDtState = Engine.TensorMultiplyScalar(dtState, NumOps.FromDouble(0.5));
        Tensor<T> Column(Tensor<T> time) => Engine.TensorTile(Engine.Reshape(time, new[] { 1, 1 }), new[] { batch, 1 });

        var t = t0;
        for (int s = 0; s < steps; s++)
        {
            var tHalf = Engine.TensorAdd(t, Engine.TensorMultiplyScalar(dt, NumOps.FromDouble(0.5)));
            var tNext = Engine.TensorAdd(t, dt);
            var k1 = field(state, Column(t));
            var k2 = field(Engine.TensorAdd(state, Engine.TensorMultiply(halfDtState, k1)), Column(tHalf));
            var k3 = field(Engine.TensorAdd(state, Engine.TensorMultiply(halfDtState, k2)), Column(tHalf));
            var k4 = field(Engine.TensorAdd(state, Engine.TensorMultiply(dtState, k3)), Column(tNext));
            var sum = Engine.TensorAdd(Engine.TensorAdd(k1, k4),
                Engine.TensorMultiplyScalar(Engine.TensorAdd(k2, k3), NumOps.FromDouble(2.0)));
            state = Engine.TensorAdd(state, Engine.TensorMultiplyScalar(Engine.TensorMultiply(dtState, sum), NumOps.FromDouble(1.0 / 6.0)));
            t = tNext;
        }
        return state;
    }

    /// <summary>
    /// Eqs. 14-16: tanh on each mode-normalized value α, Gumbel-softmax (temperature τ) on each mode block β and each
    /// discrete column d. Without <paramref name="gumbel"/> the blocks are the noise-free softmax(logits / τ), which
    /// keeps <see cref="NeuralNetworkBase{T}.Predict"/> deterministic. Before <see cref="Fit"/> there are no columns
    /// and the head is returned as is.
    /// </summary>
    private Tensor<T> ApplyOutputActivations(Tensor<T> output, bool gumbel)
    {
        if (_transformer is null) return output;
        int batch = output.Shape[0], total = output.Shape[1];
        var inverseTemperature = NumOps.FromDouble(1.0 / _options.GumbelTemperature);
        Tensor<T> Categorical(int offset, int width)
        {
            var logits = Engine.TensorSlice(output, new[] { 0, offset }, new[] { batch, width });
            if (gumbel) logits = Engine.TensorAdd(logits, GumbelNoise(batch, width));
            return Engine.Softmax(Engine.TensorMultiplyScalar(logits, inverseTemperature), axis: 1);
        }

        var blocks = new List<Tensor<T>>();
        int idx = 0;
        for (int col = 0; col < _columns.Count && idx < total; col++)
        {
            var info = _transformer.GetTransformInfo(col);
            if (info.IsContinuous)
            {
                blocks.Add(Engine.TensorTanh(Engine.TensorSlice(output, new[] { 0, idx }, new[] { batch, 1 })));
                idx++;
                int modes = Math.Min(info.Width - 1, total - idx);
                if (modes > 0) { blocks.Add(Categorical(idx, modes)); idx += modes; }
            }
            else
            {
                int width = Math.Min(info.Width, total - idx);
                blocks.Add(Categorical(idx, width));
                idx += width;
            }
        }
        if (idx < total) blocks.Add(Engine.TensorSlice(output, new[] { 0, idx }, new[] { batch, total - idx }));
        return blocks.Count == 1 ? blocks[0] : Engine.TensorConcatenate(blocks.ToArray(), axis: 1);
    }

    #endregion

    #region Training Steps

    /// <summary>
    /// The WGAN-GP discriminator step (Eq. 1) on a training-by-sampling mini-batch. The same step moves the time points
    /// (Algorithm 1, line 7), as in the authors' code: one backward of L_D reaches both, and the points are then
    /// clamped strictly inside their neighbours, 0 &lt; t_1 &lt; ... &lt; t_{m-1} &lt; 1.
    /// </summary>
    private void TrainDiscriminatorStep(Matrix<T> transformed, int batchSize)
    {
        var (cond, _, rows) = SampleConditions(batchSize);
        var real = new Tensor<T>(new[] { batchSize, _dataWidth });
        for (int b = 0; b < batchSize; b++)
            for (int j = 0; j < _dataWidth; j++) real[b, j] = transformed[rows[b], j];
        Tensor<T> fake;
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
            fake = GeneratorForward(GeneratorInput(cond), gumbel: true);
        var realInput = WithCondition(real, cond);
        var fakeInput = WithCondition(fake, cond);

        var engine = Engine;
        var discriminatorLayers = DiscriminatorLayers();
        int penaltyState = _random.Next();
        Tensor<T> Critic(Tensor<T> x) => DiscriminatorForward(x, training: true);
        LastLoss = AiDotNet.Training.TapeTrainingStepper<T>.EagerObjectiveStep(
            () =>
            {
                var parameters = AiDotNet.Training.TapeTrainingStep<T>.CollectParameters(discriminatorLayers).ToList();
                parameters.Add(_timePoints);
                return parameters;
            },
            () =>
            {
                var wasserstein = engine.TensorSubtract(
                    engine.ReduceMean(Critic(fakeInput), new[] { 0, 1 }, keepDims: false),
                    engine.ReduceMean(Critic(realInput), new[] { 0, 1 }, keepDims: false));
                // One draw of the interpolation weights per step, replayed if the objective is re-evaluated.
                var penalty = AiDotNet.Training.WassersteinCriticStep<T>.GradientPenalty(
                    engine, realInput, fakeInput, Critic, RandomHelper.CreateSeededRandom(penaltyState));
                return engine.TensorAdd(wasserstein, engine.TensorMultiplyScalar(penalty, NumOps.FromDouble(_options.GradientPenaltyWeight)));
            },
            _discriminatorOptimizer);

        const double margin = 1e-5;
        for (int i = 0; i < _timePoints.Length; i++)
        {
            double low = i == 0 ? margin : NumOps.ToDouble(_timePoints[i - 1]) + margin;
            double high = i == _timePoints.Length - 1 ? 1.0 - margin : NumOps.ToDouble(_timePoints[i + 1]) - margin;
            _timePoints[i] = NumOps.FromDouble(Math.Min(Math.Max(NumOps.ToDouble(_timePoints[i]), low), Math.Max(low, high)));
        }
    }

    /// <summary>The generator step on L_G + L_matching: −E[D(G(z, c) ⊕ c)] + H(c_s, d̂_s) (section 3.3).</summary>
    private void TrainGeneratorStep(int batchSize)
    {
        var (cond, mask, _) = SampleConditions(batchSize);
        var input = GeneratorInput(cond);
        // One draw of the Gumbel noise per step, replayed if the optimizer re-evaluates the objective.
        var noiseState = _random.Next();
        var generatorLayers = GeneratorLayers();
        LastLoss = AiDotNet.Training.TapeTrainingStepper<T>.EagerObjectiveStep(
            () => AiDotNet.Training.TapeTrainingStep<T>.CollectParameters(generatorLayers),
            () =>
            {
                var saved = _random;
                _random = RandomHelper.CreateSeededRandom(noiseState);
                try
                {
                    var fake = GeneratorForward(input, gumbel: true);
                    var scores = DiscriminatorForward(WithCondition(fake, cond), training: true);
                    var loss = Engine.TensorNegate(Engine.ReduceMean(scores, new[] { 0, 1 }, keepDims: false));
                    if (_condWidth > 0 && _catOutputBlocks.Count > 0)
                        loss = Engine.TensorAdd(loss, ConditionalCrossEntropy(fake, cond, mask));
                    return loss;
                }
                finally
                {
                    _random = saved;
                }
            },
            _generatorOptimizer,
            modelGradientClip: _options.MaxGradNorm);
    }

    /// <summary>
    /// L_matching = H(c_s, d̂_s): the cross-entropy between the one-hot condition and the generated distribution of
    /// the conditioned discrete column, averaged over the batch.
    /// </summary>
    private Tensor<T> ConditionalCrossEntropy(Tensor<T> fake, Tensor<T> cond, Tensor<T> mask)
    {
        int batch = fake.Shape[0];
        var blocks = _catOutputBlocks.Select(b => Engine.TensorSlice(fake, new[] { 0, b.Offset }, new[] { batch, b.Width })).ToArray();
        var aligned = blocks.Length == 1 ? blocks[0] : Engine.TensorConcatenate(blocks, axis: 1);
        var logp = Engine.TensorLog(Engine.TensorAddScalar(aligned, NumOps.FromDouble(1e-8)));
        var perRow = Engine.ReduceSum(Engine.TensorMultiply(Engine.TensorMultiply(mask, cond), logp), new[] { 1 }, keepDims: false);
        return Engine.TensorNegate(Engine.ReduceMean(perRow, new[] { 0 }, keepDims: false));
    }

    // Offset and width of each discrete column's block in the row, in the condition vector's column order.
    private void BuildCategoricalOutputBlocks()
    {
        _catOutputBlocks.Clear();
        if (_transformer is null) return;
        for (int col = 0; col < _columns.Count; col++)
        {
            var info = _transformer.GetTransformInfo(col);
            if (!info.IsContinuous) _catOutputBlocks.Add((info.StartOffset, info.Width));
        }
    }

    // Section 3.3: one random discrete column s gets a random one-hot c_s (training-by-sampling), with the real row
    // drawn from those holding that category.
    private (Tensor<T> Cond, Tensor<T> Mask, int[] Rows) SampleConditions(int batchSize)
    {
        var cond = new Tensor<T>(new[] { batchSize, _condWidth });
        var mask = new Tensor<T>(new[] { batchSize, _condWidth });
        var rows = new int[batchSize];
        if (_sampler is null) return (cond, mask, rows);
        for (int b = 0; b < batchSize; b++)
        {
            var (cv, mv, _, _, row) = _sampler.SampleCondVecWithMask();
            for (int j = 0; j < _condWidth; j++) { cond[b, j] = cv[j]; mask[b, j] = mv[j]; }
            rows[b] = row;
        }
        return (cond, mask, rows);
    }

    private Tensor<T> GeneratorInput(Tensor<T> cond)
        => _condWidth > 0 ? Engine.TensorConcatenate(new[] { Noise(cond.Shape[0]), cond }, axis: 1) : Noise(cond.Shape[0]);

    private Tensor<T> WithCondition(Tensor<T> row, Tensor<T> cond)
        => _condWidth > 0 ? Engine.TensorConcatenate(new[] { row, cond }, axis: 1) : row;

    #endregion

    #region Helpers

    private Tensor<T> Constant(double value)
    {
        var t = new Tensor<T>(new[] { 1 });
        t[0] = NumOps.FromDouble(value);
        return t;
    }

    private Tensor<T> Noise(int batch)
    {
        var z = new Tensor<T>(new[] { batch, _options.EmbeddingDimension });
        for (int i = 0; i < z.Length; i++)
        {
            double u1 = 1.0 - _random.NextDouble();
            double u2 = _random.NextDouble();
            z[i] = NumOps.FromDouble(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }
        return z;
    }

    // Standard Gumbel noise −log(−log U).
    private Tensor<T> GumbelNoise(int batch, int width)
    {
        var g = new Tensor<T>(new[] { batch, width });
        for (int i = 0; i < g.Length; i++)
        {
            double u = Math.Min(Math.Max(_random.NextDouble(), 1e-10), 1.0 - 1e-10);
            g[i] = NumOps.FromDouble(-Math.Log(-Math.Log(u)));
        }
        return g;
    }

    private IReadOnlyList<ILayer<T>> GeneratorLayers()
    {
        var all = new List<ILayer<T>>(Layers);
        all.AddRange(_genBNLayers);
        return all;
    }

    private IReadOnlyList<ILayer<T>> DiscriminatorLayers()
    {
        var all = new List<ILayer<T>>(_discLayers);
        all.AddRange(_discOdeLayers);
        all.AddRange(_discOdeNormLayers);
        all.AddRange(_discDropoutLayers);
        return all;
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
                ["GeneratorType"] = "OCTGANGenerator",
                ["EmbeddingDimension"] = _options.EmbeddingDimension,
                ["GeneratorDimensions"] = _options.GeneratorDimensions,
                ["DiscriminatorDimensions"] = _options.DiscriminatorDimensions,
                ["TimePoints"] = TimePointValues,
                ["OdeSolverSteps"] = _options.OdeSolverSteps,
                ["BatchSize"] = _options.BatchSize,
                ["Epochs"] = _options.Epochs,
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
    /// <remarks>
    /// The input is <c>p(0) = z ⊕ c</c>, <c>[dim p]</c> or <c>[batch, dim p]</c>; a bare noise vector <c>z</c> is read
    /// with an all-zero (unconditional) condition. Modes and categories are the noise-free softmax, so the output is
    /// deterministic.
    /// </remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        bool single = input.Rank == 1;
        var x = single ? Engine.Reshape(input, new[] { 1, input.Length }) : input;
        if (!_usingCustomLayers && _condWidth > 0 && x.Shape[1] == _options.EmbeddingDimension
            && GeneratorInputWidth == _options.EmbeddingDimension + _condWidth)
            x = Engine.TensorConcatenate(new[] { x, new Tensor<T>(new[] { x.Shape[0], _condWidth }) }, axis: 1);
        var output = GeneratorForward(x, gumbel: false);
        return single ? Engine.Reshape(output, new[] { output.Length }) : output;
    }

    /// <inheritdoc />
    /// <remarks>
    /// The adversarial training of Algorithm 1 runs in <see cref="Fit"/>. This single-step entry point regresses the
    /// generator's output for the given <c>z ⊕ c</c> onto the target (mean squared error), for supervised use.
    /// </remarks>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expectedOutput)
    {
        var generatorLayers = GeneratorLayers();
        SetTrainingMode(true);
        try
        {
            LastLoss = AiDotNet.Training.TapeTrainingStepper<T>.EagerObjectiveStep(
                () => AiDotNet.Training.TapeTrainingStep<T>.CollectParameters(generatorLayers),
                () =>
                {
                    var output = PredictCore(input);
                    var flatOut = Engine.Reshape(output, new[] { output.Length });
                    var target = Engine.Reshape(expectedOutput, new[] { expectedOutput.Length });
                    var diff = Engine.TensorSubtract(flatOut, target);
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
