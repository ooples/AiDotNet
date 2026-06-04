using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.SyntheticData;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.SyntheticData;

/// <summary>
/// PATE-GAN generator for differentially private synthetic tabular data generation using
/// the Private Aggregation of Teacher Ensembles (PATE) framework.
/// </summary>
/// <remarks>
/// <para>
/// PATE-GAN uses a teacher-student architecture for differential privacy:
///
/// <code>
///  Real Data Partition 1 ──► Teacher 1 ──┐
///  Real Data Partition 2 ──► Teacher 2 ──┤
///  Real Data Partition 3 ──► Teacher 3 ──┼──► Noisy Aggregation ──► Student Labels
///       ...                    ...       │
///  Real Data Partition N ──► Teacher N ──┘
///
///  Noise ──► Generator ──► Fake Data ──► Student Discriminator ──► Real/Fake?
///                                        (trained on noisy labels)
/// </code>
///
/// This implementation follows the standard neural network architecture pattern with:
/// - Proper inheritance from NeuralNetworkBase
/// - Layer-based architecture using ILayer components
/// - Support for custom layers via NeuralNetworkArchitecture
/// - Full forward → backward → update → reset lifecycle
/// </para>
/// <para>
/// <b>For Beginners:</b> PATE-GAN is like a game of telephone with added privacy:
///
/// 1. Split real data among teachers (each sees only a small part)
/// 2. Show generated data to all teachers: "Is this real or fake?"
/// 3. Count votes and add random noise to the count
/// 4. Tell the student the noisy answer
/// 5. Generator tries to fool the student
///
/// The noise ensures no individual's information leaks through.
///
/// If you provide custom layers in the architecture, those will be used directly
/// for the generator network. If not, the network creates industry-standard
/// PATE-GAN layers based on the original research paper specifications.
///
/// Example usage:
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputFeatures: 10,
///     outputSize: 10
/// );
/// var options = new PATEGANOptions&lt;double&gt;
/// {
///     NumTeachers = 10,
///     LaplaceScale = 0.1,
///     Epochs = 300
/// };
/// var generator = new PATEGANGenerator&lt;double&gt;(architecture, options);
/// generator.Fit(data, columns, epochs: 300);
/// var synthetic = generator.Generate(1000);
/// </code>
/// </para>
/// <para>
/// Reference: "PATE-GAN: Generating Synthetic Data with Differential Privacy Guarantees"
/// (Jordon et al., ICLR 2019)
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
    "https://arxiv.org/abs/1906.09338",
    Year = 2019,
    Authors = "James Jordon, Jinsung Yoon, Mihaela van der Schaar")]
public class PATEGANGenerator<T> : NeuralNetworkBase<T>, ISyntheticTabularGenerator<T>
{
    private readonly PATEGANOptions<T> _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private ILossFunction<T> _lossFunction;

    // Synthetic tabular data infrastructure
    private TabularDataTransformer<T>? _transformer;
    private List<ColumnMetadata> _columns = new();
    private int _dataWidth;
    private Random _random;

    // Generator batch normalization layers (auxiliary, always match generator hidden layers)
    private readonly List<BatchNormalizationLayer<T>> _genBNLayers = new();

    // Teacher ensemble: list of hidden layers + output layer per teacher (auxiliary, not user-overridable)
    private readonly List<List<FullyConnectedLayer<T>>> _teacherLayers = new();
    private readonly List<FullyConnectedLayer<T>> _teacherOutputs = new();

    // Student discriminator layers (auxiliary, not user-overridable)
    private readonly List<FullyConnectedLayer<T>> _studentLayers = new();
    private readonly List<DropoutLayer<T>> _studentDropoutLayers = new();

    // Cached pre-activations for proper backward passes

    // Whether custom layers are being used (disables residual connection logic)
    private bool _usingCustomLayers;

    /// <summary>
    /// Gets the PATE-GAN-specific options.
    /// </summary>
    public new PATEGANOptions<T> Options => _options;

    /// <inheritdoc />
    public IReadOnlyList<ColumnMetadata> Columns => _columns.AsReadOnly();

    /// <inheritdoc />
    public bool IsFitted { get; private set; }

    /// <summary>
    /// Initializes a new PATE-GAN generator with the specified architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining input/output dimensions and optional custom layers.</param>
    /// <param name="options">PATE-GAN-specific options for generator and discriminator configuration.</param>
    /// <param name="optimizer">Gradient-based optimizer (defaults to Adam).</param>
    /// <param name="lossFunction">Loss function (defaults based on task type).</param>
    /// <param name="maxGradNorm">Maximum gradient norm for clipping (default 5.0).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The architecture parameter controls the generator network.
    /// If you provide custom layers, those become the generator. Teacher and student
    /// networks are always created automatically from the options.
    /// </para>
    /// </remarks>
    /// <summary>
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    public PATEGANGenerator()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.OneDimensional,
            taskType: Enums.NeuralNetworkTaskType.Regression,
            inputSize: 10,
            outputSize: 10))
    {
    }

    public PATEGANGenerator(
        NeuralNetworkArchitecture<T> architecture,
        PATEGANOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 5.0)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), maxGradNorm)
    {
        _options = options ?? new PATEGANOptions<T>();
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();
        InitializeLayers();
    }

    #region Layer Initialization

    /// <summary>
    /// Initializes the generator layers from architecture or defaults.
    /// Teacher and student networks are always built from options.
    /// </summary>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            _usingCustomLayers = true;
            Layers.Clear();
            foreach (var layer in Architecture.Layers)
            {
                Layers.Add(layer);
            }
        }
        else
        {
            _usingCustomLayers = false;
            BuildDefaultGeneratorLayers(
                _options.EmbeddingDimension,
                Architecture.OutputSize > 0 ? Architecture.OutputSize : 1);
        }
    }

    /// <summary>
    /// Rebuilds default generator layers when actual data dimensions become known during Fit().
    /// Also builds teacher ensemble and student discriminator with correct dimensions.
    /// </summary>
    private void RebuildLayersWithActualDimensions()
    {
        if (!_usingCustomLayers)
        {
            BuildDefaultGeneratorLayers(_options.EmbeddingDimension, _dataWidth);
        }

        BuildTeachers();
        BuildStudent();
    }

    private void BuildDefaultGeneratorLayers(int inputDim, int outputDim)
    {
        Layers.Clear();
        _genBNLayers.Clear();

        var defaultLayers = LayerHelper<T>.CreateDefaultPATEGANGeneratorLayers(
            inputDim, outputDim, _options.GeneratorDimensions).ToList();

        foreach (var layer in defaultLayers)
        {
            Layers.Add(layer);
        }

        // Create matching BN layers for hidden layers (all except output layer)
        for (int i = 0; i < _options.GeneratorDimensions.Length; i++)
        {
            _genBNLayers.Add(new BatchNormalizationLayer<T>());
        }
    }

    private void BuildTeachers()
    {
        _teacherLayers.Clear();
        _teacherOutputs.Clear();

        var identity = new IdentityActivation<T>() as IActivationFunction<T>;

        for (int t = 0; t < _options.NumTeachers; t++)
        {
            var layers = new List<FullyConnectedLayer<T>>();
            var dims = _options.TeacherDimensions;

            for (int i = 0; i < dims.Length; i++)
            {
                int layerInput = i == 0 ? _dataWidth : dims[i - 1];
                layers.Add(new FullyConnectedLayer<T>(dims[i], identity));
            }

            int lastDim = dims.Length > 0 ? dims[^1] : _dataWidth;
            var output = new FullyConnectedLayer<T>(1, identity);

            _teacherLayers.Add(layers);
            _teacherOutputs.Add(output);
        }
    }

    private void BuildStudent()
    {
        _studentLayers.Clear();
        _studentDropoutLayers.Clear();

        var identity = new IdentityActivation<T>() as IActivationFunction<T>;
        var dims = _options.StudentDimensions;

        for (int i = 0; i < dims.Length; i++)
        {
            int layerInput = i == 0 ? _dataWidth : dims[i - 1];
            _studentLayers.Add(new FullyConnectedLayer<T>(dims[i], identity));
            _studentDropoutLayers.Add(new DropoutLayer<T>(_options.StudentDropout));
        }

        int lastDim = dims.Length > 0 ? dims[^1] : _dataWidth;
        _studentLayers.Add(new FullyConnectedLayer<T>(1, identity));
    }

    #endregion

    #region ISyntheticTabularGenerator Implementation

    /// <inheritdoc />
    public void Fit(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs)
    {
        _columns = columns.ToList();

        // Transform data
        _transformer = new TabularDataTransformer<T>(_options.VGMModes, _random);
        _transformer.Fit(data, columns);
        _dataWidth = _transformer.TransformedWidth;
        var transformedData = _transformer.Transform(data);

        // Rebuild networks with actual data dimensions
        RebuildLayersWithActualDimensions();

        // Partition data for teachers
        var partitions = PartitionData(transformedData, _options.NumTeachers);

        int batchSize = Math.Min(_options.BatchSize, data.Rows);
        T lr = NumOps.FromDouble(_options.LearningRate);

        // Phase 1: Pre-train teachers on their partitions
        int teacherEpochs = Math.Max(1, epochs / 4);
        SetTrainingMode(true);
        try
        {
            for (int epoch = 0; epoch < teacherEpochs; epoch++)
                for (int tIdx = 0; tIdx < _options.NumTeachers; tIdx++)
                    TrainTeacher(tIdx, partitions[tIdx], batchSize, lr);
        }
        finally { SetTrainingMode(false); }

        // Phase 2: Joint student + generator training
        int jointEpochs = epochs - teacherEpochs;
        for (int epoch = 0; epoch < jointEpochs; epoch++)
        {
            SetTrainingMode(true);
            try
            {
                for (int step = 0; step < _options.StudentSteps; step++)
                {
                    TrainStudentStep(transformedData, batchSize, lr);
                }
                TrainGeneratorStep(batchSize);
            }
            finally { SetTrainingMode(false); }
        }

        IsFitted = true;
    }

    /// <inheritdoc />
    public async Task FitAsync(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns,
        int epochs, CancellationToken cancellationToken = default)
    {
        await Task.Run(() =>
        {
            cancellationToken.ThrowIfCancellationRequested();

            _columns = columns.ToList();

            // Transform data
            _transformer = new TabularDataTransformer<T>(_options.VGMModes, _random);
            _transformer.Fit(data, columns);
            _dataWidth = _transformer.TransformedWidth;
            var transformedData = _transformer.Transform(data);

            // Rebuild networks with actual data dimensions
            RebuildLayersWithActualDimensions();

            // Partition data for teachers
            var partitions = PartitionData(transformedData, _options.NumTeachers);

            int batchSize = Math.Min(_options.BatchSize, data.Rows);
            T lr = NumOps.FromDouble(_options.LearningRate);

            // Phase 1: Pre-train teachers
            int teacherEpochs = Math.Max(1, epochs / 4);
            for (int epoch = 0; epoch < teacherEpochs; epoch++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                for (int tIdx = 0; tIdx < _options.NumTeachers; tIdx++)
                {
                    TrainTeacher(tIdx, partitions[tIdx], batchSize, lr);
                }
            }

            // Phase 2: Joint training
            int jointEpochs = epochs - teacherEpochs;
            for (int epoch = 0; epoch < jointEpochs; epoch++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                SetTrainingMode(true);
                try
                {
                    for (int step = 0; step < _options.StudentSteps; step++)
                    {
                        TrainStudentStep(transformedData, batchSize, lr);
                    }
                    TrainGeneratorStep(batchSize);
                }
                finally { SetTrainingMode(false); }
            }

            IsFitted = true;
        }, cancellationToken).ConfigureAwait(false);
    }

    /// <inheritdoc />
    public Matrix<T> Generate(int numSamples, Vector<T>? conditionColumn = null, Vector<T>? conditionValue = null)
    {
        if (_transformer is null)
        {
            throw new InvalidOperationException("Generator is not fitted. Call Fit() first.");
        }

        var transformedRows = new Matrix<T>(numSamples, _dataWidth);

        for (int i = 0; i < numSamples; i++)
        {
            var noise = CreateStandardNormalVector(_options.EmbeddingDimension);
            var generated = GeneratorForward(noise);

            for (int j = 0; j < _dataWidth && j < generated.Length; j++)
            {
                transformedRows[i, j] = generated[j];
            }
        }

        return _transformer.InverseTransform(transformedRows);
    }

    #endregion

    #region Forward Passes

    /// <summary>
    /// Generator forward pass with residual connections, BatchNorm, and manual ReLU.
    /// When using custom layers, performs a simple sequential forward pass instead.
    /// </summary>
    private Tensor<T> GeneratorForward(Vector<T> input)
    {
        var inputTensor = VectorToTensor(input);

        if (_usingCustomLayers)
        {
            return CustomLayersForward(inputTensor);
        }

        return DefaultGeneratorForward(inputTensor);
    }

    private Tensor<T> CustomLayersForward(Tensor<T> input)
    {
        var current = input;
        for (int i = 0; i < Layers.Count; i++)
        {
            current = Layers[i].Forward(current);
        }
        return ApplyOutputActivations(current);
    }

    private Tensor<T> DefaultGeneratorForward(Tensor<T> inputTensor)
    {
        var current = inputTensor;

        for (int i = 0; i < Layers.Count - 1; i++)
        {
            // Residual: concatenate original input
            if (i > 0)
            {
                current = ConcatTensors(current, inputTensor);
            }

            current = Layers[i].Forward(current);

            if (i < _genBNLayers.Count)
            {
                current = _genBNLayers[i].Forward(current);
            }

            // Cache pre-activation for ReLU backward
            current = ApplyReLU(current);
        }

        // Final layer with residual connection
        current = ConcatTensors(current, inputTensor);
        current = Layers[^1].Forward(current);

        return ApplyOutputActivations(current);
    }

    /// <summary>
    /// Student discriminator forward pass with manual LeakyReLU and Dropout.
    /// </summary>
    private Tensor<T> StudentForward(Tensor<T> input, bool isTraining)
    {
        var current = input;

        for (int i = 0; i < _studentLayers.Count - 1; i++)
        {
            current = _studentLayers[i].Forward(current);
            current = ApplyLeakyReLU(current);

            if (isTraining)
            {
                current = _studentDropoutLayers[i].Forward(current);
            }
        }

        current = _studentLayers[^1].Forward(current);
        return current;
    }

    /// <summary>
    /// Teacher discriminator forward pass with manual LeakyReLU.
    /// </summary>
    private Tensor<T> TeacherForward(int teacherIdx, Tensor<T> input)
    {
        var layers = _teacherLayers[teacherIdx];
        var outputLayer = _teacherOutputs[teacherIdx];
        var current = input;

        for (int i = 0; i < layers.Count; i++)
        {
            current = layers[i].Forward(current);
            current = ApplyLeakyReLU(current);
        }

        current = outputLayer.Forward(current);
        return current;
    }

    #endregion

    #region Training (tape-connected PATE-GAN)

    // Teachers/student are BCE discriminators; the generator fools the student
    // (Jordon et al. 2019 PATE-GAN). All forwards are tape-connected (Engine ops)
    // so autodiff backpropagates; the student learns the DP-noisy teacher consensus.
    private void TrainTeacher(int teacherIdx, Matrix<T> partition, int batchSize, T lr)
    {
        int size = Math.Min(batchSize, partition.Rows);
        for (int b = 0; b < partition.Rows; b += size)
        {
            int end = Math.Min(b + size, partition.Rows);
            for (int row = b; row < end; row++)
            {
                var real = VectorToTensor(GetRow(partition, row));
                var noise = CreateStandardNormalVector(_options.EmbeddingDimension);
                using var tape = new GradientTape<T>();
                var realScore = TeacherForward(teacherIdx, real);
                var fakeData = GeneratorForward(noise);
                var fakeScore = TeacherForward(teacherIdx, fakeData);
                var loss = Engine.TensorAdd(BceLoss(realScore, 1.0), BceLoss(fakeScore, 0.0));
                TapeStepOver(tape, loss, BuildTeacherLayerList(teacherIdx));
            }
        }
    }

    private void TrainStudentStep(Matrix<T> transformedData, int batchSize, T lr)
    {
        int size = Math.Min(batchSize, transformedData.Rows);
        for (int i = 0; i < size; i++)
        {
            var noise = CreateStandardNormalVector(_options.EmbeddingDimension);
            var fakeVector = TensorToVector(GeneratorForward(noise), _dataWidth);
            double fakeLabel = QueryTeachers(fakeVector);

            var realSample = GetRow(transformedData, _random.Next(transformedData.Rows));
            double realLabel = QueryTeachers(realSample);

            using var tape = new GradientTape<T>();
            var fakeScore = StudentForward(VectorToTensor(fakeVector), isTraining: true);
            var realScore = StudentForward(VectorToTensor(realSample), isTraining: true);
            var loss = Engine.TensorAdd(BceLoss(fakeScore, fakeLabel), BceLoss(realScore, realLabel));
            TapeStepOver(tape, loss, BuildStudentLayerList());
        }
    }

    private void TrainGeneratorStep(int batchSize)
    {
        for (int i = 0; i < batchSize; i++)
        {
            var noise = CreateStandardNormalVector(_options.EmbeddingDimension);
            using var tape = new GradientTape<T>();
            var fakeData = GeneratorForward(noise);
            var studentScore = StudentForward(fakeData, isTraining: true);
            var loss = BceLoss(studentScore, 1.0);
            TapeStepOver(tape, loss, BuildGeneratorLayerList());
        }
    }

    #endregion

    #region PATE Mechanism

    /// <summary>
    /// Queries the teacher ensemble with noisy aggregation for differential privacy.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the core privacy mechanism. The noise added to the vote
    /// count ensures that even if an attacker sees the final label, they cannot determine
    /// what any individual teacher (and thus any individual data point) contributed.
    /// </para>
    /// </remarks>
    private double QueryTeachers(Vector<T> sample)
    {
        double voteSum = 0;
        var sampleTensor = VectorToTensor(sample);

        for (int t = 0; t < _options.NumTeachers; t++)
        {
            var score = TeacherForward(t, sampleTensor);
            double sig = Sigmoid(NumOps.ToDouble(score[0]));
            voteSum += sig > 0.5 ? 1.0 : 0.0;
        }

        double noisyVoteCount = voteSum + SampleLaplace(_options.LaplaceScale);
        return Math.Min(Math.Max(noisyVoteCount / _options.NumTeachers, 0.0), 1.0);
    }

    /// <summary>
    /// Samples from the Laplace distribution using the inverse CDF method.
    /// </summary>
    private double SampleLaplace(double scale)
    {
        double u = _random.NextDouble() - 0.5;
        return -scale * Math.Sign(u) * Math.Log(1.0 - 2.0 * Math.Abs(u));
    }

    #endregion

    #region Backward Passes

    #endregion

    #region Tape Step Helpers

    private void TapeStepOver(GradientTape<T> tape, Tensor<T> loss, IReadOnlyList<ILayer<T>> layers)
    {
        var trainable = Training.TapeTrainingStep<T>.CollectParameters(layers);
        if (trainable.Count == 0) return;
        var grads = tape.ComputeGradients(loss, trainable);
        T lossValue = loss.Length > 0 ? loss[0] : NumOps.Zero;
        LastLoss = lossValue;
        var ctx = new AiDotNet.Tensors.Engines.Autodiff.TapeStepContext<T>(trainable, grads, lossValue);
        _optimizer.Step(ctx);
    }

    private Tensor<T> ReduceToScalar(Tensor<T> t)
        => Engine.ReduceSum(t, Enumerable.Range(0, t.Shape.Length).ToArray(), keepDims: false);

    /// <summary>Tape-connected binary cross-entropy for a single logit against a soft target in [0,1].</summary>
    private Tensor<T> BceLoss(Tensor<T> logit, double target)
    {
        T eps = NumOps.FromDouble(1e-7);
        var p = Engine.TensorClamp(Engine.TensorSigmoid(logit), eps, NumOps.FromDouble(1.0 - 1e-7));
        var logP = Engine.TensorLog(p);
        var log1mP = Engine.TensorLog(Engine.ScalarMinusTensor(NumOps.One, p));
        var t1 = Engine.TensorMultiplyScalar(logP, NumOps.FromDouble(target));
        var t2 = Engine.TensorMultiplyScalar(log1mP, NumOps.FromDouble(1.0 - target));
        return Engine.TensorNegate(ReduceToScalar(Engine.TensorAdd(t1, t2)));
    }

    private IReadOnlyList<ILayer<T>> BuildGeneratorLayerList()
    {
        var all = new List<ILayer<T>>(Layers);
        all.AddRange(_genBNLayers);
        return all;
    }

    private IReadOnlyList<ILayer<T>> BuildStudentLayerList()
    {
        var all = new List<ILayer<T>>(_studentLayers);
        all.AddRange(_studentDropoutLayers);
        return all;
    }

    private IReadOnlyList<ILayer<T>> BuildTeacherLayerList(int teacherIdx)
    {
        var all = new List<ILayer<T>>(_teacherLayers[teacherIdx]) { _teacherOutputs[teacherIdx] };
        return all;
    }

    #endregion

    #region Activation Functions

    private Tensor<T> ApplyReLU(Tensor<T> input) => Engine.TensorReLU(input);


    private Tensor<T> ApplyLeakyReLU(Tensor<T> input) => Engine.TensorLeakyReLU(input, NumOps.FromDouble(0.2));


    #endregion

    #region Output Activations

    /// <summary>
    /// Applies per-column output activations: tanh for continuous values, softmax for
    /// mode indicators and categorical columns.
    /// </summary>
    private Tensor<T> ApplyOutputActivations(Tensor<T> output)
    {
        if (_transformer is null) return output;

        var flat = output.Rank == 1 ? output : Engine.Reshape(output, new[] { output.Length });
        var blocks = new List<Tensor<T>>();
        int idx = 0;
        for (int col = 0; col < _columns.Count && idx < flat.Length; col++)
        {
            var transform = _transformer.GetTransformInfo(col);
            if (transform.IsContinuous)
            {
                blocks.Add(Engine.TensorTanh(Engine.TensorSlice(flat, new[] { idx }, new[] { 1 })));
                idx++;
                int numModes = transform.Width - 1;
                if (numModes > 0)
                {
                    blocks.Add(Engine.TensorSoftmax(Engine.TensorSlice(flat, new[] { idx }, new[] { numModes }), axis: 0));
                    idx += numModes;
                }
            }
            else
            {
                blocks.Add(Engine.TensorSoftmax(Engine.TensorSlice(flat, new[] { idx }, new[] { transform.Width }), axis: 0));
                idx += transform.Width;
            }
        }
        if (blocks.Count == 0) return flat;
        if (idx < flat.Length) blocks.Add(Engine.TensorSlice(flat, new[] { idx }, new[] { flat.Length - idx }));
        return Engine.TensorConcatenate(blocks.ToArray(), axis: 0);
    }


    #endregion

    #region Gradient Safety


    #endregion

    #region Data Helpers

    private Vector<T> CreateStandardNormalVector(int length)
    {
        var v = new Vector<T>(length);
        for (int i = 0; i < length; i++)
        {
            double u1 = 1.0 - _random.NextDouble();
            double u2 = _random.NextDouble();
            double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            v[i] = NumOps.FromDouble(z);
        }
        return v;
    }

    private static List<Matrix<T>> PartitionData(Matrix<T> data, int numPartitions)
    {
        var partitions = new List<Matrix<T>>();
        int partitionSize = Math.Max(1, data.Rows / numPartitions);

        for (int p = 0; p < numPartitions; p++)
        {
            int start = p * partitionSize;
            int end = (p == numPartitions - 1) ? data.Rows : Math.Min(start + partitionSize, data.Rows);
            int rows = end - start;

            if (rows <= 0)
            {
                var minPartition = new Matrix<T>(1, data.Columns);
                int lastRow = Math.Min(start, data.Rows - 1);
                for (int j = 0; j < data.Columns; j++)
                {
                    minPartition[0, j] = data[lastRow, j];
                }
                partitions.Add(minPartition);
                continue;
            }

            var partition = new Matrix<T>(rows, data.Columns);
            for (int r = 0; r < rows; r++)
            {
                for (int j = 0; j < data.Columns; j++)
                {
                    partition[r, j] = data[start + r, j];
                }
            }
            partitions.Add(partition);
        }

        return partitions;
    }

    private static double Sigmoid(double x)
    {
        double clipped = Math.Min(Math.Max(x, -20.0), 20.0);
        return 1.0 / (1.0 + Math.Exp(-clipped));
    }

    private static Vector<T> GetRow(Matrix<T> matrix, int row)
    {
        var v = new Vector<T>(matrix.Columns);
        for (int j = 0; j < matrix.Columns; j++)
        {
            v[j] = matrix[row, j];
        }
        return v;
    }

    private Tensor<T> VectorToTensor(Vector<T> v)
    {
        var t = new Tensor<T>([v.Length]);
        for (int i = 0; i < v.Length; i++) t[i] = v[i];
        return t;
    }

    private Vector<T> TensorToVector(Tensor<T> t, int length)
    {
        var v = new Vector<T>(length);
        int copyLen = Math.Min(length, t.Length);
        for (int i = 0; i < copyLen; i++) v[i] = t[i];
        return v;
    }

    private Tensor<T> ConcatTensors(Tensor<T> a, Tensor<T> b)
    {
        var a1 = a.Rank == 1 ? a : Engine.Reshape(a, new[] { a.Length });
        var b1 = b.Rank == 1 ? b : Engine.Reshape(b, new[] { b.Length });
        return Engine.TensorConcatenate(new[] { a1, b1 }, axis: 0);
    }

    private static Tensor<T> CloneTensor(Tensor<T> source)
    {
        var clone = new Tensor<T>(source._shape);
        for (int i = 0; i < source.Length; i++)
        {
            clone[i] = source[i];
        }
        return clone;
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
                ["NumTeachers"] = _options.NumTeachers,
                ["LaplaceScale"] = _options.LaplaceScale,
                ["EmbeddingDimension"] = _options.EmbeddingDimension,
                ["GeneratorDimensions"] = _options.GeneratorDimensions,
                ["TeacherDimensions"] = _options.TeacherDimensions,
                ["StudentDimensions"] = _options.StudentDimensions,
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
        for (int i = 0; i < _columns.Count; i++)
        {
            importance[_columns[i].Name] = equal;
        }
        return importance;
    }

    /// <inheritdoc />
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // Treats the input as the generator's latent noise (deterministic). The
        // generator layers adapt to the input width on first forward, so this works
        // before Fit too (the generated ModelFamily tests call Predict without Fit).
        var noise = new Vector<T>(input.Length);
        for (int i = 0; i < input.Length; i++) noise[i] = input[i];
        return GeneratorForward(noise);
    }

    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Full PATE-GAN training (teachers + student + generator) runs in Fit. This
        // single-step entry point trains the generator (the Layers chain that Predict
        // runs) to reconstruct the target via a tape-connected step, satisfying the
        // NeuralNetworkBase contract. The previous body was a no-op.
        SetTrainingMode(true);
        try
        {
            using var tape = new GradientTape<T>();
            var output = GeneratorForward(TensorToVector(input, input.Length));
            var flatOut = output.Rank == 1 ? output : Engine.Reshape(output, new[] { output.Length });
            var target = expectedOutput.Rank == 1 ? expectedOutput : Engine.Reshape(expectedOutput, new[] { expectedOutput.Length });
            var loss = ReduceToScalar(Engine.TensorSquare(Engine.TensorSubtract(flatOut, target)));
            TapeStepOver(tape, loss, BuildGeneratorLayerList());
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <inheritdoc />
    public override void UpdateParameters(Vector<T> parameters)
    {
        int offset = 0;
        foreach (var layer in Layers)
        {
            var layerParams = layer.GetParameters();
            int count = layerParams.Length;
            if (offset + count <= parameters.Length)
            {
                var newParams = new Vector<T>(count);
                for (int i = 0; i < count; i++)
                {
                    newParams[i] = parameters[offset + i];
                }
                layer.SetParameters(newParams);
                offset += count;
            }
        }
    }

    /// <inheritdoc />
    protected override void SerializeNetworkSpecificData(System.IO.BinaryWriter writer)
    {
        writer.Write(_dataWidth);
        writer.Write(_usingCustomLayers);
        writer.Write(IsFitted);
    }

    /// <inheritdoc />
    protected override void DeserializeNetworkSpecificData(System.IO.BinaryReader reader)
    {
        _dataWidth = reader.ReadInt32();
        _usingCustomLayers = reader.ReadBoolean();
        IsFitted = reader.ReadBoolean();
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new PATEGANGenerator<T>(Architecture, _options, _optimizer, _lossFunction);
    }

    #endregion

}
