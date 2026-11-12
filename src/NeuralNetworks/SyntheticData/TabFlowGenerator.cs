using AiDotNet.LearningRateSchedulers;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.SyntheticData;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.SyntheticData;

/// <summary>
/// TabFlow generator using flow matching with optimal transport conditional paths
/// for high-quality, fast synthetic tabular data generation.
/// </summary>
/// <remarks>
/// <para>
/// TabFlow learns a velocity field v(x, t) that defines an ODE: dx/dt = v(x, t).
/// The ODE transports samples from noise (t=0) to data (t=1) along straight paths.
/// This implementation follows the standard neural network architecture pattern with:
/// - Proper inheritance from NeuralNetworkBase
/// - Layer-based architecture using ILayer components
/// - Engine-based tensor operations for CPU/GPU acceleration
/// - Full autodiff and JIT compilation support
///
/// <b>Training</b>: Given data x1 and noise x0, the optimal transport path is:
///   xt = (1 - t) * x0 + t * x1  (linear interpolation)
///   Target velocity: v* = x1 - x0  (direction from noise to data)
///   Loss: ||v(xt, t) - v*||^2     (learn to predict the direction)
///
/// <b>Generation</b>: Start at x0 ~ N(0,1), solve ODE from t=0 to t=1 using Euler/RK4.
/// </para>
/// <para>
/// <b>For Beginners:</b> TabFlow works by learning "which direction data should move":
///
/// <code>
/// Training: noise x0 ----[straight line]----> data x1
///           At time t, point is at: xt = mix of x0 and x1
///           MLP learns: "from xt at time t, which direction is x1?"
///
/// Generation: Start at random noise
///             Repeatedly ask MLP "which way?" and take a small step
///             After ~100 steps, arrive at realistic data
/// </code>
///
/// If you provide custom layers in the architecture, those will be used directly
/// for the velocity field MLP. If not, the network creates industry-standard
/// TabFlow layers based on the original research paper specifications.
///
/// Example usage:
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputFeatures: 10,
///     outputSize: 10
/// );
/// var options = new TabFlowOptions&lt;double&gt;
/// {
///     MLPDimensions = new[] { 256, 256, 256 },
///     NumSteps = 100
/// };
/// var generator = new TabFlowGenerator&lt;double&gt;(architecture, options);
/// generator.Fit(data, columns, epochs: 500);
/// var synthetic = generator.Generate(1000);
/// </code>
/// </para>
/// <para>
/// Reference: "Flow Matching for Tabular Data" (2024)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.SyntheticDataGenerator)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Flow Matching for Generative Modeling",
    "https://arxiv.org/abs/2210.02747",
    Year = 2023,
    Authors = "Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le")]
[PaperOptimizer(OptimizerKind.Adam, Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-8,
                WeightDecay = 0,
                Source = "Lipman et al. 2023, Sec. 5: all models are trained using the Adam optimizer "
                        + "with beta1 0.9, beta2 0.999, a weight decay of 0.0 and epsilon 1e-8. The zero "
                        + "weight decay is the paper value rather than an omission.")]
public partial class TabFlowGenerator<T> : NeuralSyntheticTabularGeneratorBase<T>, ISyntheticTabularGenerator<T>
{
    private readonly TabFlowOptions<T> _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private ILossFunction<T> _lossFunction;

    // Synthetic tabular data infrastructure
    private TabularDataTransformer<T>? _transformer;
    private List<ColumnMetadata> _columns = new();
    private int _dataWidth;
    private Random _random;

    // Time projection layers (always created by LayerHelper, not user-overridable)
    private readonly List<ILayer<T>> _timeProjectionLayers = new();

    /// <summary>
    /// Gets the TabFlow-specific options.
    /// </summary>
    public new TabFlowOptions<T> Options => _options;

    /// <inheritdoc />
    public IReadOnlyList<ColumnMetadata> Columns => _columns.AsReadOnly();

    /// <inheritdoc />
    public bool IsFitted { get; private set; }

    /// <summary>
    /// Initializes a new TabFlow generator with the specified architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining input/output dimensions and optional custom layers.</param>
    /// <param name="options">TabFlow-specific options for velocity field configuration.</param>
    /// <param name="optimizer">Gradient-based optimizer (defaults to Adam).</param>
    /// <param name="lossFunction">Loss function (defaults based on task type).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a TabFlow network based on the architecture you provide.
    ///
    /// If you provide custom layers in the architecture, those will be used directly
    /// for the velocity field MLP. If not, the network will create industry-standard
    /// TabFlow layers based on the original research paper specifications.
    ///
    /// Example usage:
    /// <code>
    /// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
    ///     inputFeatures: 10,
    ///     outputSize: 10
    /// );
    /// var options = new TabFlowOptions&lt;double&gt; { NumSteps = 100 };
    /// var generator = new TabFlowGenerator&lt;double&gt;(architecture, options);
    /// </code>
    /// </para>
    /// </remarks>
    /// <summary>
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    public TabFlowGenerator()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.OneDimensional,
            taskType: Enums.NeuralNetworkTaskType.Regression,
            inputSize: 10,
            outputSize: 10))
    {
    }

    public TabFlowGenerator(NeuralNetworkArchitecture<T> architecture,
        TabFlowOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), (options ??= new TabFlowOptions<T>()).MaxGradNorm)
    {
        _options = options;
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);
        _optimizer = optimizer ?? PaperOptimizerFactory.VerifyHandBuilt(this,
            new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
                new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
                {
                    InitialLearningRate = _options.LearningRate
                }));
        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        InitializeLayers();
    }

    #region Layer Initialization (GANDALF Pattern)

    /// <summary>
    /// Initializes the layers of the TabFlow network based on the provided architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method either uses custom layers provided in the architecture or creates
    /// default TabFlow layers following the original paper specifications.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method sets up the network structure:
    /// - If you provided custom layers, those are used for the velocity field MLP
    /// - Otherwise, it creates the standard TabFlow architecture:
    ///   1. Time projection (sinusoidal embedding processed by a Dense layer)
    ///   2. Velocity field MLP (Dense(SiLU) layers ending with Dense(Identity))
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            // Use the layers provided by the user for the velocity field MLP
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
        }
        else
        {
            // Use default TabFlow layer configuration based on original paper specs
            Layers.AddRange(LayerHelper<T>.CreateDefaultTabFlowVelocityLayers(
                inputDim: Architecture.CalculatedInputSize + _options.TimeEmbeddingDimension,
                outputDim: Architecture.OutputSize,
                hiddenDims: _options.MLPDimensions,
                dropoutRate: _options.DropoutRate));
        }

        // Time projection layers are always created (not user-overridable)
        _timeProjectionLayers.Clear();
        _timeProjectionLayers.AddRange(
            LayerHelper<T>.CreateDefaultTabFlowTimeProjectionLayers(
                _options.TimeEmbeddingDimension));
    }

    /// <summary>
    /// Rebuilds MLP layers using the actual transformed data width (which may differ from
    /// Architecture.OutputSize due to VGMM mode-encoding of continuous columns).
    /// </summary>
    private void RebuildLayersForDataWidth()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            return; // User-provided layers are not rebuilt
        }

        Layers.Clear();
        Layers.AddRange(LayerHelper<T>.CreateDefaultTabFlowVelocityLayers(
            inputDim: _dataWidth + _options.TimeEmbeddingDimension,
            outputDim: _dataWidth,
            hiddenDims: _options.MLPDimensions,
            dropoutRate: _options.DropoutRate));

        _timeProjectionLayers.Clear();
        _timeProjectionLayers.AddRange(
            LayerHelper<T>.CreateDefaultTabFlowTimeProjectionLayers(
                _options.TimeEmbeddingDimension));
    }

    #endregion

    #region Neural Network Methods (GANDALF Pattern)

    /// <summary>
    /// Makes a prediction using the TabFlow velocity field for the given input tensor.
    /// </summary>
    /// <param name="input">The input tensor (concatenated data + time embedding).</param>
    /// <returns>The predicted velocity tensor after passing through all layers.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This predicts the velocity (direction) that data should move
    /// at a given point and time. During generation, this is called repeatedly to follow
    /// the flow from noise to data.
    /// </para>
    /// </remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {

        // CPU path: forward pass through each layer sequentially
        Tensor<T> currentOutput = input;
        foreach (var layer in Layers)
        {
            currentOutput = layer.Forward(currentOutput);
        }

        return currentOutput;
    }

    /// <summary>
    /// Updates the parameters of all layers in the network based on computed gradients.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> After backpropagation computes how much each parameter
    /// contributed to the error, this method uses the optimizer to adjust all parameters
    /// to improve predictions.
    /// </para>
    /// </remarks>
    private void UpdateNetworkParameters()
    {
        _optimizer.UpdateParameters(Layers);
    }

    // UpdateParameters re-sliced the flat vector across Layers by hand -- the base walks
    // exactly the same enumeration, so this said nothing the base does not already say.
    #endregion

    #region ISyntheticTabularGenerator<T> Implementation

    /// <summary>
    /// Fits the TabFlow generator to the provided real tabular data.
    /// </summary>
    /// <param name="data">The real data matrix where each row is a sample and each column is a feature.</param>
    /// <param name="columns">Metadata describing each column (type, categories, etc.).</param>
    /// <param name="epochs">Number of training epochs. When null, the model's published Epochs from its options is used.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the "learning" step. The generator studies your real data
    /// to understand its patterns and distributions. After fitting, call Generate() to create
    /// new synthetic rows.
    /// </para>
    /// </remarks>
    public void Fit(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int? epochs = null)
    {
        int epochCount = epochs ?? _options.Epochs;
        ValidateFitInputs(data, columns, epochCount);

        _columns = PrepareColumns(data, columns);

        // Step 1: Fit transformer
        _transformer = new TabularDataTransformer<T>(_options.VGMModes, _random);
        _transformer.Fit(data, _columns);
        _dataWidth = _transformer.TransformedWidth;

        // Rebuild layers with correct transformed data dimensions
        BuildFromConstructionSequence(() => RebuildLayersForDataWidth());

        var transformedData = _transformer.Transform(data);

        // Step 2: Training loop
        int batchSize = Math.Min(_options.BatchSize, data.Rows);
        int numBatches = Math.Max(1, data.Rows / batchSize);

        for (int epoch = 0; epoch < epochCount; epoch++)
        {
            for (int batch = 0; batch < numBatches; batch++)
            {
                int startRow = batch * batchSize;
                int endRow = Math.Min(startRow + batchSize, data.Rows);
                TrainBatch(transformedData, startRow, endRow);
            }
        }

        IsFitted = true;
    }

    /// <inheritdoc />
    public async Task FitAsync(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int? epochs = null, CancellationToken ct = default)
    {
        int epochCount = epochs ?? _options.Epochs;
        ValidateFitInputs(data, columns, epochCount);

        _columns = PrepareColumns(data, columns);

        await Task.Run(() =>
        {
            ct.ThrowIfCancellationRequested();

            _transformer = new TabularDataTransformer<T>(_options.VGMModes, _random);
            _transformer.Fit(data, _columns);
            _dataWidth = _transformer.TransformedWidth;

            BuildFromConstructionSequence(() => RebuildLayersForDataWidth());

            var transformedData = _transformer.Transform(data);

            int batchSize = Math.Min(_options.BatchSize, data.Rows);
            int numBatches = Math.Max(1, data.Rows / batchSize);

            for (int epoch = 0; epoch < epochCount; epoch++)
            {
                ct.ThrowIfCancellationRequested();
                for (int batch = 0; batch < numBatches; batch++)
                {
                    int startRow = batch * batchSize;
                    int endRow = Math.Min(startRow + batchSize, data.Rows);
                    TrainBatch(transformedData, startRow, endRow);
                }
            }
        }, ct).ConfigureAwait(false);

        IsFitted = true;
    }

    /// <summary>
    /// Generates new synthetic tabular data rows.
    /// </summary>
    /// <param name="numSamples">The number of synthetic rows to generate.</param>
    /// <param name="conditionColumn">Optional conditioning column indices.</param>
    /// <param name="conditionValue">Optional conditioning values.</param>
    /// <returns>A matrix of synthetic data with the same column structure as the training data.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> After fitting, this creates new fake-but-realistic rows by
    /// starting from random noise and following the learned velocity field using an ODE solver.
    /// </para>
    /// </remarks>
    public Matrix<T> Generate(int numSamples, Vector<T>? conditionColumn = null, Vector<T>? conditionValue = null)
    {
        if (!IsFitted || _transformer is null)
        {
            throw new InvalidOperationException(
                "The generator must be fitted before generating data. Call Fit() first.");
        }

        if (numSamples <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numSamples), "Number of samples must be positive.");
        }

        var transformedRows = new Matrix<T>(numSamples, _dataWidth);
        int numSteps = _options.NumSteps;
        double dt = 1.0 / numSteps;
        T dtT = NumOps.FromDouble(dt);

        for (int i = 0; i < numSamples; i++)
        {
            // Start from noise at t=0
            var x = CreateStandardNormalVector(_dataWidth);

            if (_options.Solver == "rk4")
            {
                // RK4 ODE solver
                for (int step = 0; step < numSteps; step++)
                {
                    double t = (double)step / numSteps;
                    x = RK4Step(x, t, dt);
                }
            }
            else
            {
                // Euler ODE solver
                for (int step = 0; step < numSteps; step++)
                {
                    double t = (double)step / numSteps;
                    var velocity = PredictVelocity(x, t);

                    // x <- x + dt * v
                    x = Engine.Add(x, Engine.Multiply(velocity, dtT));
                }
            }

            // Apply output activations to get valid data
            var xTensor = VectorToTensor(x);
            var activated = ApplyOutputActivations(xTensor);
            transformedRows.SetRow(i, TensorToVector(activated, _dataWidth));
        }

        return _transformer.InverseTransform(transformedRows);
    }

    #endregion

    #region Velocity Prediction

    private Vector<T> PredictVelocity(Vector<T> x, double t)
    {
        var timeEmbed = CreateTimeEmbedding(t);

        // Concatenate data and time embedding
        // Forward through velocity field layers (GANDALF pattern: layers handle Forward)
        var current = Engine.TensorConcatenate(new[] { VectorToTensor(x), VectorToTensor(timeEmbed) }, 0);
        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }

        return TensorToVector(current, _dataWidth);
    }

    private Vector<T> CreateTimeEmbedding(double t)
    {
        int dim = _options.TimeEmbeddingDimension;
        int halfDim = dim / 2;

        // [sin(angle_i), cos(angle_i)] with angle_i = 1000 t * exp(-ln(10000) i / halfDim); t is scaled by
        // 1000 for better frequency coverage. An odd dim leaves the last entry 0.
        Tensor<T> current;
        if (halfDim == 0)
        {
            current = new Tensor<T>([dim]);
        }
        else
        {
            var frequencies = Engine.TensorExp(Engine.TensorMultiplyScalar(
                Engine.TensorArange<T>(0, halfDim, 1), NumOps.FromDouble(-Math.Log(10000.0) / halfDim)));
            var angles = Engine.TensorMultiplyScalar(frequencies, NumOps.FromDouble(t * 1000.0));
            var parts = dim % 2 == 0
                ? new[] { Engine.TensorSin(angles), Engine.TensorCos(angles) }
                : new[] { Engine.TensorSin(angles), Engine.TensorCos(angles), new Tensor<T>([1]) };
            current = Engine.TensorConcatenate(parts, 0);
        }

        // Forward through time projection layers
        foreach (var layer in _timeProjectionLayers)
        {
            current = layer.Forward(current);
        }

        return TensorToVector(current, dim);
    }

    #endregion

    #region Training

    private void TrainBatch(Matrix<T> transformedData, int startRow, int endRow)
    {
        for (int row = startRow; row < endRow; row++)
        {
            TrainOnRow(transformedData.GetRow(row));
        }
    }

    /// <summary>One flow-matching step on data point <paramref name="x1"/>.</summary>
    private void TrainOnRow(Vector<T> x1)
    {
        // Sample random time t ~ U(0, 1)
        double t = _random.NextDouble();
        T tT = NumOps.FromDouble(t);

        // Sample noise x0 ~ N(0, 1)
        var x0 = CreateStandardNormalVector(_dataWidth);

        // Compute the point on the conditional probability path:
        //   xt = (1-t)*x0 + t*x1 + sigma*eps      (Lipman et al. 2023, Eq. 22)
        // Sigma is the width of that path. At sigma = 0 the path is the bare
        // straight line and the option is inert, which is what it was before.
        var pathNoise = CreateStandardNormalVector(_dataWidth);
        var xt = Engine.Add(
            Engine.Add(Engine.Multiply(x0, NumOps.Subtract(NumOps.One, tT)), Engine.Multiply(x1, tT)),
            Engine.Multiply(pathNoise, NumOps.FromDouble(_options.Sigma)));

        // Target velocity: v* = x1 - x0 (optimal transport direction)
        var targetVelocity = Tensor<T>.FromVector(Engine.Subtract(x1, x0));

        // Create time embedding and build input tensor
        var timeEmbed = CreateTimeEmbedding(t);
        var input = Engine.TensorConcatenate(new[] { VectorToTensor(xt), VectorToTensor(timeEmbed) }, 0);

        // Use Train() method (GANDALF pattern: forward → loss → backward → update)
        Train(input, targetVelocity);
    }

    #endregion

    #region ODE Solvers

    private Vector<T> RK4Step(Vector<T> x, double t, double dt)
    {
        // k1 = v(x, t)
        var k1 = PredictVelocity(x, t);

        T halfDt = NumOps.FromDouble(0.5 * dt);
        T fullDt = NumOps.FromDouble(dt);

        // k2 = v(x + dt/2 * k1, t + dt/2)
        var k2 = PredictVelocity(Engine.Add(x, Engine.Multiply(k1, halfDt)), t + 0.5 * dt);

        // k3 = v(x + dt/2 * k2, t + dt/2)
        var k3 = PredictVelocity(Engine.Add(x, Engine.Multiply(k2, halfDt)), t + 0.5 * dt);

        // k4 = v(x + dt * k3, t + dt)
        var k4 = PredictVelocity(Engine.Add(x, Engine.Multiply(k3, fullDt)), t + dt);

        // x_new = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        T two = NumOps.FromDouble(2.0);
        var weighted = Engine.Add(
            Engine.Add(k1, Engine.Multiply(k2, two)),
            Engine.Add(Engine.Multiply(k3, two), k4));
        return Engine.Add(x, Engine.Multiply(weighted, NumOps.FromDouble(dt / 6.0)));
    }

    #endregion

    #region Output Activations

    private Tensor<T> ApplyOutputActivations(Tensor<T> output)
    {
        if (_transformer is null) return output;

        var result = new Tensor<T>(output._shape);
        var squashed = Engine.TensorTanh(output);
        int idx = 0;

        for (int col = 0; col < _columns.Count && idx < output.Length; col++)
        {
            var transform = _transformer.GetTransformInfo(col);
            if (transform.IsContinuous)
            {
                if (idx < output.Length)
                {
                    // The scalar value goes through tanh.
                    squashed.Data.Span.Slice(idx, 1).CopyTo(result.Data.Span.Slice(idx, 1));
                    idx++;
                }
                int numModes = transform.Width - 1;
                if (numModes > 0) ApplySoftmax(output, result, ref idx, numModes);
            }
            else
            {
                ApplySoftmax(output, result, ref idx, transform.Width);
            }
        }
        return result;
    }

    private void ApplySoftmax(Tensor<T> input, Tensor<T> output, ref int idx, int count)
    {
        if (count <= 0) return;
        int actualCount = Math.Min(count, input.Length - idx);
        if (actualCount <= 0) return;
        var slice = new Tensor<T>([actualCount]);
        input.Data.Span.Slice(idx, actualCount).CopyTo(slice.Data.Span);
        var result = Engine.Softmax(slice, -1);
        result.Data.Span.CopyTo(output.Data.Span.Slice(idx, actualCount));
        idx += actualCount;
    }

    #endregion

    #region Metadata and Serialization (GANDALF Pattern)

    /// <inheritdoc/>
    public override Dictionary<string, T> GetFeatureImportance()
    {
        var importance = new Dictionary<string, T>();
        int numFeatures = Architecture.CalculatedInputSize;
        var uniformValue = NumOps.FromDouble(1.0 / Math.Max(numFeatures, 1));
        for (int f = 0; f < numFeatures; f++)
        {
            importance[$"feature_{f}"] = uniformValue;
        }
        return importance;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "Architecture", "TabFlow" },
                { "InputSize", Architecture.CalculatedInputSize },
                { "OutputSize", Architecture.OutputSize },
                { "TimeEmbeddingDim", _options.TimeEmbeddingDimension },
                { "NumSteps", _options.NumSteps },
                { "Solver", _options.Solver },
                { "MLPDimensions", _options.MLPDimensions },
                { "LayerCount", Layers.Count },
                { "LayerTypes", Layers.Select(l => l.GetType().Name).ToArray() }
            },
            ModelDataProvider = () => SerializeForMetadata()
        };
    }

    /// <inheritdoc/>


    /// <inheritdoc/>


    #endregion

    #region Input Validation and Column Management

    private static void ValidateFitInputs(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs)
    {
        if (data.Rows == 0 || data.Columns == 0)
        {
            throw new ArgumentException("Data matrix must not be empty.", nameof(data));
        }

        if (columns.Count == 0)
        {
            throw new ArgumentException("Column metadata list must not be empty.", nameof(columns));
        }

        if (columns.Count != data.Columns)
        {
            throw new ArgumentException(
                $"Column metadata count ({columns.Count}) must match data column count ({data.Columns}).",
                nameof(columns));
        }

        if (epochs <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(epochs), "Epochs must be positive.");
        }
    }

    private List<ColumnMetadata> PrepareColumns(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns)
    {
        var prepared = new List<ColumnMetadata>(columns.Count);

        for (int col = 0; col < columns.Count; col++)
        {
            var meta = columns[col].Clone();
            meta.ColumnIndex = col;

            if (meta.IsNumerical)
            {
                ComputeColumnStatistics(data, col, meta);
            }
            else if (meta.IsCategorical && meta.NumCategories == 0)
            {
                var categories = new HashSet<string>();
                for (int row = 0; row < data.Rows; row++)
                {
                    var val = NumOps.ToDouble(data[row, col]);
                    categories.Add(val.ToString(System.Globalization.CultureInfo.InvariantCulture));
                }
                meta.Categories = categories.OrderBy(c => c, StringComparer.Ordinal).ToList().AsReadOnly();
            }

            prepared.Add(meta);
        }

        return prepared;
    }

    private void ComputeColumnStatistics(Matrix<T> data, int colIndex, ColumnMetadata meta)
    {
        int n = data.Rows;
        if (n == 0)
        {
            // An empty column keeps the empty-scan values: no min/max, mean 0/0, unit std.
            meta.Min = double.MaxValue;
            meta.Max = double.MinValue;
            meta.Mean = 0.0 / n;
            meta.Std = 1.0;
            return;
        }

        var column = Tensor<T>.FromVector(data.GetColumn(colIndex));
        T meanT = Engine.TensorMean(column);
        double mean = NumOps.ToDouble(meanT);

        // Sample standard deviation (n - 1), floored at 1e-10.
        T sumSqDiff = Engine.TensorSumOfSquares(Engine.TensorSubtractScalar(column, meanT));
        double std = n > 1 ? Math.Sqrt(NumOps.ToDouble(sumSqDiff) / (n - 1)) : 1.0;
        if (std < 1e-10) std = 1e-10;

        meta.Min = NumOps.ToDouble(Engine.TensorMinValue(column));
        meta.Max = NumOps.ToDouble(Engine.TensorMaxValue(column));
        meta.Mean = mean;
        meta.Std = std;
    }

    #endregion

    #region Random Sampling Utilities

    private T SampleStandardNormal()
    {
        double u1 = 1.0 - _random.NextDouble();
        double u2 = _random.NextDouble();
        double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        return NumOps.FromDouble(z);
    }

    private Vector<T> CreateStandardNormalVector(int length)
    {
        var v = new Vector<T>(length);
        for (int i = 0; i < length; i++)
        {
            v[i] = SampleStandardNormal();
        }
        return v;
    }

    #endregion

    #region Helpers

    private static Tensor<T> VectorToTensor(Vector<T> v) => Tensor<T>.FromVector(v);

    /// <summary>The first min(length, t.Length) values of <paramref name="t"/>, zero-padded to <paramref name="length"/>.</summary>
    private static Vector<T> TensorToVector(Tensor<T> t, int length)
    {
        int copyLen = Math.Min(length, t.Length);
        if (copyLen == 0)
            return new Vector<T>(length);
        var engine = AiDotNetEngine.Current;
        var flat = engine.Reshape(t, new[] { t.Length });
        var head = copyLen == t.Length ? flat : engine.TensorSlice(flat, new[] { 0 }, new[] { copyLen });
        if (copyLen < length)
            head = engine.TensorConcatenate(new[] { head, new Tensor<T>(new[] { length - copyLen }) }, 0);
        return head.ToVector();
    }

    #endregion

}
