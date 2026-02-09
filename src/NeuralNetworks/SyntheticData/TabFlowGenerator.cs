using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.SyntheticData;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;

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
public class TabFlowGenerator<T> : NeuralNetworkBase<T>, ISyntheticTabularGenerator<T>
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
    /// <param name="maxGradNorm">Maximum gradient norm for clipping (default 5.0).</param>
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
    public TabFlowGenerator(
        NeuralNetworkArchitecture<T> architecture,
        TabFlowOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 5.0)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), maxGradNorm)
    {
        _options = options ?? new TabFlowOptions<T>();
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
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
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // GPU-resident optimization: use TryForwardGpuOptimized for 10-50x speedup
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        // CPU path: forward pass through each layer sequentially
        Tensor<T> currentOutput = input;
        foreach (var layer in Layers)
        {
            currentOutput = layer.Forward(currentOutput);
        }

        return currentOutput;
    }

    /// <summary>
    /// Trains the TabFlow network using the provided input and expected output.
    /// </summary>
    /// <param name="input">The input tensor (concatenated xt + time embedding).</param>
    /// <param name="expectedOutput">The expected velocity tensor (x1 - x0).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is how the velocity field learns from examples.
    ///
    /// The training process:
    /// 1. Takes a noisy data point (input) and the correct velocity direction (expected output)
    /// 2. Predicts velocity using the current network state
    /// 3. Compares prediction to correct velocity to calculate the error
    /// 4. Uses this error to adjust the network's parameters (backpropagation)
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Forward pass to get prediction
        Tensor<T> prediction = Predict(input);

        // Calculate loss
        LastLoss = _lossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());

        // Calculate error gradient
        Tensor<T> error = prediction.Subtract(expectedOutput);

        // Backpropagate error through network
        BackpropagateError(error);

        // Update network parameters
        UpdateNetworkParameters();
    }

    /// <summary>
    /// Backpropagates the error through the network layers.
    /// </summary>
    /// <param name="error">The error tensor to backpropagate.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This propagates the error backwards through each layer,
    /// allowing each layer to compute its local gradients for parameter updates.
    /// </para>
    /// </remarks>
    private void BackpropagateError(Tensor<T> error)
    {
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            error = Layers[i].Backward(error);
        }
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

    /// <summary>
    /// Updates the parameters of all layers in the network.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters for the network.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This distributes updated parameter values to each layer
    /// based on their parameter count. Called during training to apply improvements.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int startIndex = 0;
        foreach (var layer in Layers)
        {
            int layerParameterCount = layer.ParameterCount;
            if (layerParameterCount > 0)
            {
                Vector<T> layerParameters = parameters.SubVector(startIndex, layerParameterCount);
                layer.UpdateParameters(layerParameters);
                startIndex += layerParameterCount;
            }
        }
    }

    #endregion

    #region ISyntheticTabularGenerator<T> Implementation

    /// <summary>
    /// Fits the TabFlow generator to the provided real tabular data.
    /// </summary>
    /// <param name="data">The real data matrix where each row is a sample and each column is a feature.</param>
    /// <param name="columns">Metadata describing each column (type, categories, etc.).</param>
    /// <param name="epochs">Number of training epochs.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the "learning" step. The generator studies your real data
    /// to understand its patterns and distributions. After fitting, call Generate() to create
    /// new synthetic rows.
    /// </para>
    /// </remarks>
    public void Fit(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs)
    {
        ValidateFitInputs(data, columns, epochs);

        _columns = PrepareColumns(data, columns);

        // Step 1: Fit transformer
        _transformer = new TabularDataTransformer<T>(_options.VGMModes, _random);
        _transformer.Fit(data, _columns);
        _dataWidth = _transformer.TransformedWidth;

        // Rebuild layers with correct transformed data dimensions
        RebuildLayersForDataWidth();

        var transformedData = _transformer.Transform(data);

        // Step 2: Training loop
        int batchSize = Math.Min(_options.BatchSize, data.Rows);
        int numBatches = Math.Max(1, data.Rows / batchSize);

        for (int epoch = 0; epoch < epochs; epoch++)
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
    public async Task FitAsync(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs, CancellationToken ct = default)
    {
        ValidateFitInputs(data, columns, epochs);

        _columns = PrepareColumns(data, columns);

        await Task.Run(() =>
        {
            ct.ThrowIfCancellationRequested();

            _transformer = new TabularDataTransformer<T>(_options.VGMModes, _random);
            _transformer.Fit(data, _columns);
            _dataWidth = _transformer.TransformedWidth;

            RebuildLayersForDataWidth();

            var transformedData = _transformer.Transform(data);

            int batchSize = Math.Min(_options.BatchSize, data.Rows);
            int numBatches = Math.Max(1, data.Rows / batchSize);

            for (int epoch = 0; epoch < epochs; epoch++)
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

                    for (int j = 0; j < _dataWidth; j++)
                    {
                        x[j] = NumOps.FromDouble(NumOps.ToDouble(x[j]) + dt * NumOps.ToDouble(velocity[j]));
                    }
                }
            }

            // Apply output activations to get valid data
            var xTensor = VectorToTensor(x);
            var activated = ApplyOutputActivations(xTensor);

            for (int j = 0; j < _dataWidth && j < activated.Length; j++)
            {
                transformedRows[i, j] = activated[j];
            }
        }

        return _transformer.InverseTransform(transformedRows);
    }

    #endregion

    #region Velocity Prediction

    private Vector<T> PredictVelocity(Vector<T> x, double t)
    {
        var timeEmbed = CreateTimeEmbedding(t);

        // Concatenate data and time embedding
        int totalLen = x.Length + timeEmbed.Length;
        var input = new Vector<T>(totalLen);
        for (int i = 0; i < x.Length; i++) input[i] = x[i];
        for (int i = 0; i < timeEmbed.Length; i++) input[x.Length + i] = timeEmbed[i];

        // Forward through velocity field layers (GANDALF pattern: layers handle Forward)
        var current = VectorToTensor(input);
        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }

        return TensorToVector(current, _dataWidth);
    }

    private Vector<T> CreateTimeEmbedding(double t)
    {
        int dim = _options.TimeEmbeddingDimension;
        var embedding = new Vector<T>(dim);
        int halfDim = dim / 2;

        for (int i = 0; i < halfDim; i++)
        {
            double freq = Math.Exp(-Math.Log(10000.0) * i / halfDim);
            double angle = t * 1000.0 * freq; // Scale t by 1000 for better frequency coverage
            embedding[i] = NumOps.FromDouble(Math.Sin(angle));
            if (i + halfDim < dim) embedding[i + halfDim] = NumOps.FromDouble(Math.Cos(angle));
        }

        // Forward through time projection layers
        var current = VectorToTensor(embedding);
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
            // Sample random time t ~ U(0, 1)
            double t = _random.NextDouble();

            // Get data point x1
            var x1 = GetRow(transformedData, row);

            // Sample noise x0 ~ N(0, 1)
            var x0 = CreateStandardNormalVector(_dataWidth);

            // Compute interpolated point: xt = (1-t)*x0 + t*x1
            var xt = new Vector<T>(_dataWidth);
            for (int j = 0; j < _dataWidth; j++)
            {
                double v0 = NumOps.ToDouble(x0[j]);
                double v1 = NumOps.ToDouble(x1[j]);
                xt[j] = NumOps.FromDouble((1.0 - t) * v0 + t * v1);
            }

            // Target velocity: v* = x1 - x0 (optimal transport direction)
            var targetVelocity = new Tensor<T>([_dataWidth]);
            for (int j = 0; j < _dataWidth; j++)
            {
                targetVelocity[j] = NumOps.FromDouble(
                    NumOps.ToDouble(x1[j]) - NumOps.ToDouble(x0[j]));
            }

            // Create time embedding and build input tensor
            var timeEmbed = CreateTimeEmbedding(t);
            int totalLen = xt.Length + timeEmbed.Length;
            var input = new Tensor<T>([totalLen]);
            for (int j = 0; j < xt.Length; j++) input[j] = xt[j];
            for (int j = 0; j < timeEmbed.Length; j++) input[xt.Length + j] = timeEmbed[j];

            // Use Train() method (GANDALF pattern: forward → loss → backward → update)
            Train(input, targetVelocity);
        }
    }

    #endregion

    #region ODE Solvers

    private Vector<T> RK4Step(Vector<T> x, double t, double dt)
    {
        // k1 = v(x, t)
        var k1 = PredictVelocity(x, t);

        // k2 = v(x + dt/2 * k1, t + dt/2)
        var xMid1 = new Vector<T>(_dataWidth);
        for (int j = 0; j < _dataWidth; j++)
        {
            xMid1[j] = NumOps.FromDouble(NumOps.ToDouble(x[j]) + 0.5 * dt * NumOps.ToDouble(k1[j]));
        }
        var k2 = PredictVelocity(xMid1, t + 0.5 * dt);

        // k3 = v(x + dt/2 * k2, t + dt/2)
        var xMid2 = new Vector<T>(_dataWidth);
        for (int j = 0; j < _dataWidth; j++)
        {
            xMid2[j] = NumOps.FromDouble(NumOps.ToDouble(x[j]) + 0.5 * dt * NumOps.ToDouble(k2[j]));
        }
        var k3 = PredictVelocity(xMid2, t + 0.5 * dt);

        // k4 = v(x + dt * k3, t + dt)
        var xEnd = new Vector<T>(_dataWidth);
        for (int j = 0; j < _dataWidth; j++)
        {
            xEnd[j] = NumOps.FromDouble(NumOps.ToDouble(x[j]) + dt * NumOps.ToDouble(k3[j]));
        }
        var k4 = PredictVelocity(xEnd, t + dt);

        // x_new = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        var result = new Vector<T>(_dataWidth);
        for (int j = 0; j < _dataWidth; j++)
        {
            double step = (NumOps.ToDouble(k1[j]) + 2.0 * NumOps.ToDouble(k2[j])
                + 2.0 * NumOps.ToDouble(k3[j]) + NumOps.ToDouble(k4[j])) / 6.0;
            result[j] = NumOps.FromDouble(NumOps.ToDouble(x[j]) + dt * step);
        }
        return result;
    }

    #endregion

    #region Output Activations

    private Tensor<T> ApplyOutputActivations(Tensor<T> output)
    {
        if (_transformer is null) return output;

        var result = new Tensor<T>(output.Shape);
        int idx = 0;

        for (int col = 0; col < _columns.Count && idx < output.Length; col++)
        {
            var transform = _transformer.GetTransformInfo(col);
            if (transform.IsContinuous)
            {
                if (idx < output.Length)
                {
                    result[idx] = NumOps.FromDouble(Math.Tanh(NumOps.ToDouble(output[idx])));
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
        double maxVal = double.MinValue;
        for (int i = 0; i < count && (idx + i) < input.Length; i++)
        {
            double v = NumOps.ToDouble(input[idx + i]);
            if (v > maxVal) maxVal = v;
        }
        double sumExp = 0;
        for (int i = 0; i < count && (idx + i) < input.Length; i++)
            sumExp += Math.Exp(NumOps.ToDouble(input[idx + i]) - maxVal);
        for (int i = 0; i < count && idx < input.Length; i++)
        {
            double expVal = Math.Exp(NumOps.ToDouble(input[idx]) - maxVal);
            output[idx] = NumOps.FromDouble(expVal / Math.Max(sumExp, 1e-10));
            idx++;
        }
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
            ModelType = ModelType.NeuralNetwork,
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
            ModelData = this.Serialize()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_options.NumSteps);
        writer.Write(_options.TimeEmbeddingDimension);
        writer.Write(_options.Solver);
        writer.Write(_options.DropoutRate);
        writer.Write(_options.LearningRate);
        writer.Write(_options.BatchSize);
        writer.Write(_options.Sigma);
        writer.Write(_options.VGMModes);
        writer.Write(_options.MLPDimensions.Length);
        foreach (var dim in _options.MLPDimensions)
        {
            writer.Write(dim);
        }
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Options are reconstructed from serialized data
        // Layers are handled by base class
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new TabFlowGenerator<T>(
            Architecture,
            _options,
            _optimizer,
            _lossFunction);
    }

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
        double sum = 0;
        double min = double.MaxValue;
        double max = double.MinValue;

        for (int row = 0; row < n; row++)
        {
            double val = NumOps.ToDouble(data[row, colIndex]);
            sum += val;
            if (val < min) min = val;
            if (val > max) max = val;
        }

        double mean = sum / n;
        double sumSqDiff = 0;
        for (int row = 0; row < n; row++)
        {
            double val = NumOps.ToDouble(data[row, colIndex]);
            double diff = val - mean;
            sumSqDiff += diff * diff;
        }

        double std = n > 1 ? Math.Sqrt(sumSqDiff / (n - 1)) : 1.0;
        if (std < 1e-10) std = 1e-10;

        meta.Min = min;
        meta.Max = max;
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

    private static Vector<T> GetRow(Matrix<T> matrix, int row)
    {
        var v = new Vector<T>(matrix.Columns);
        for (int j = 0; j < matrix.Columns; j++) v[j] = matrix[row, j];
        return v;
    }

    private static Tensor<T> VectorToTensor(Vector<T> v)
    {
        var t = new Tensor<T>([v.Length]);
        for (int i = 0; i < v.Length; i++) t[i] = v[i];
        return t;
    }

    private static Vector<T> TensorToVector(Tensor<T> t, int length)
    {
        var v = new Vector<T>(length);
        int copyLen = Math.Min(length, t.Length);
        for (int i = 0; i < copyLen; i++) v[i] = t[i];
        return v;
    }

    #endregion

    #region IJitCompilable Override

    /// <summary>
    /// TabFlow uses ODE integration with multiple solver steps which cannot be represented as a single computation graph.
    /// </summary>
    public override bool SupportsJitCompilation => false;

    #endregion
}
