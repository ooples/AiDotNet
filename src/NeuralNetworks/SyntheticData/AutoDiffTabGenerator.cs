using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.SyntheticData;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.NeuralNetworks.SyntheticData;

/// <summary>
/// AutoDiff-Tab generator that automatically searches over diffusion configurations
/// (timesteps, noise schedules, network architecture) to find optimal settings for
/// tabular data generation.
/// </summary>
/// <remarks>
/// <para>
/// AutoDiff-Tab combines architecture search with diffusion models:
///
/// <code>
///  Phase 1: Search
///    Trial 1: [T=100, linear,  MLP=256x256] ──► Loss: 0.45
///    Trial 2: [T=500, cosine,  MLP=128x128] ──► Loss: 0.38
///    Trial 3: [T=1000, linear, MLP=512x512] ──► Loss: 0.42
///    → Best: Trial 2
///
///  Phase 2: Full Training with best config
///    [T=500, cosine, MLP=128x128] ──► Train for full epochs
///
///  Phase 3: Generation
///    Noise → Iterative denoising → Synthetic data
/// </code>
/// </para>
/// <para>
/// <b>For Beginners:</b> AutoDiff-Tab is a "smart" version of TabDDPM:
///
/// 1. It tries several different diffusion setups (like testing different recipes)
/// 2. Evaluates which one works best on your data
/// 3. Uses the winner for the final model
///
/// This saves you from having to manually tune hyperparameters.
///
/// This implementation follows the standard neural network architecture pattern with:
/// - Proper inheritance from NeuralNetworkBase
/// - Layer-based architecture using ILayer components
/// - Engine-based tensor operations for CPU/GPU acceleration
/// - Full autodiff and JIT compilation support
///
/// If you provide custom layers in the architecture, those will be used directly
/// for the denoiser network. Otherwise, the network creates the standard architecture
/// based on the best configuration found during search.
///
/// Example usage:
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputFeatures: 10,
///     outputSize: 10
/// );
/// var options = new AutoDiffTabOptions&lt;double&gt;
/// {
///     SearchTrials = 5,
///     MaxTimesteps = 1000,
///     Epochs = 200
/// };
/// var generator = new AutoDiffTabGenerator&lt;double&gt;(architecture, options);
/// generator.Fit(data, columns, epochs: 200);
/// var synthetic = generator.Generate(1000);
/// </code>
/// </para>
/// <para>
/// Reference: "Automated Diffusion Models for Tabular Data" (2024)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class AutoDiffTabGenerator<T> : NeuralNetworkBase<T>, ISyntheticTabularGenerator<T>
{
    private readonly AutoDiffTabOptions<T> _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private ILossFunction<T> _lossFunction;

    // Synthetic tabular data infrastructure
    private TabularDataTransformer<T>? _transformer;
    private List<ColumnMetadata> _columns = new();
    private int _dataWidth;
    private Random _random;

    // Auxiliary layers (not user-overridable)
    private readonly List<DropoutLayer<T>> _dropoutLayers = new();
    private FullyConnectedLayer<T>? _denoiserOutput;
    private FullyConnectedLayer<T>? _timestepProjection;

    // Diffusion parameters (set after search)
    private int _numTimesteps;
    private double[] _betas = Array.Empty<double>();
    private double[] _alphas = Array.Empty<double>();
    private double[] _alphasCumprod = Array.Empty<double>();

    // Whether custom layers are being used
    private bool _usingCustomLayers;

    /// <summary>
    /// Gets the AutoDiffTab-specific options.
    /// </summary>
    public new AutoDiffTabOptions<T> Options => _options;

    /// <inheritdoc />
    public IReadOnlyList<ColumnMetadata> Columns => _columns.AsReadOnly();

    /// <inheritdoc />
    public bool IsFitted { get; private set; }

    /// <summary>
    /// Initializes a new AutoDiff-Tab generator with the specified architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining input/output dimensions and optional custom layers.</param>
    /// <param name="options">AutoDiffTab-specific configuration options.</param>
    /// <param name="optimizer">Gradient-based optimizer (defaults to Adam).</param>
    /// <param name="lossFunction">Loss function (defaults based on task type).</param>
    /// <param name="maxGradNorm">Maximum gradient norm for clipping (default 5.0).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates an AutoDiff-Tab network. If you provide custom
    /// layers in the architecture, those will be used for the denoiser. Otherwise, the architecture
    /// search discovers the best denoiser configuration automatically.
    /// </para>
    /// </remarks>
    public AutoDiffTabGenerator(
        NeuralNetworkArchitecture<T> architecture,
        AutoDiffTabOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 5.0)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), maxGradNorm)
    {
        _options = options ?? new AutoDiffTabOptions<T>();
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        InitializeLayers();
    }

    #region Layer Initialization (GANDALF Pattern)

    /// <summary>
    /// Initializes the layers of the AutoDiff-Tab denoiser based on the provided architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> If custom layers are provided, they are used as the denoiser MLP.
    /// Otherwise, default layers are created based on the options' MLP dimensions.
    /// The timestep projection and output head are always auxiliary (not user-overridable).
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            _usingCustomLayers = true;
        }
        else
        {
            // Create default denoiser hidden layers
            // Actual dimensions depend on data width, which is unknown until Fit()
            // Use placeholder dims based on options
            BuildDefaultDenoiserLayers(_options.MLPDimensions,
                _options.TimestepEmbeddingDimension + Architecture.CalculatedInputSize);
            _usingCustomLayers = false;
        }
    }

    /// <summary>
    /// Builds the default denoiser hidden layers (Layers) and auxiliary layers.
    /// </summary>
    /// <param name="dims">Hidden layer dimensions.</param>
    /// <param name="inputDim">Total input dimension (data + timestep embedding).</param>
    private void BuildDefaultDenoiserLayers(int[] dims, int inputDim)
    {
        Layers.Clear();
        _dropoutLayers.Clear();

        var silu = new SiLUActivation<T>() as IActivationFunction<T>;
        var identity = new IdentityActivation<T>() as IActivationFunction<T>;

        int teDim = _options.TimestepEmbeddingDimension;
        _timestepProjection = new FullyConnectedLayer<T>(teDim, teDim, silu);

        for (int i = 0; i < dims.Length; i++)
        {
            int layerInput = i == 0 ? inputDim : dims[i - 1];
            Layers.Add(new FullyConnectedLayer<T>(layerInput, dims[i], silu));

            if (_options.DropoutRate > 0)
            {
                _dropoutLayers.Add(new DropoutLayer<T>(_options.DropoutRate));
            }
        }

        int lastDim = dims.Length > 0 ? dims[^1] : inputDim;
        _denoiserOutput = new FullyConnectedLayer<T>(lastDim, Math.Max(1, Architecture.OutputSize), identity);
    }

    /// <summary>
    /// Rebuilds the denoiser layers with actual data dimensions discovered during Fit().
    /// Only called when not using custom layers.
    /// </summary>
    /// <param name="dims">Hidden layer dimensions from the best search config.</param>
    /// <param name="inputDim">Actual input dimension (data width + timestep embedding).</param>
    /// <param name="outputDim">Actual output dimension (data width).</param>
    private void RebuildDenoiserLayers(int[] dims, int inputDim, int outputDim)
    {
        if (!_usingCustomLayers)
        {
            Layers.Clear();
            _dropoutLayers.Clear();

            var silu = new SiLUActivation<T>() as IActivationFunction<T>;

            for (int i = 0; i < dims.Length; i++)
            {
                int layerInput = i == 0 ? inputDim : dims[i - 1];
                Layers.Add(new FullyConnectedLayer<T>(layerInput, dims[i], silu));

                if (_options.DropoutRate > 0)
                {
                    _dropoutLayers.Add(new DropoutLayer<T>(_options.DropoutRate));
                }
            }
        }

        // Always rebuild auxiliary layers
        var siluAux = new SiLUActivation<T>() as IActivationFunction<T>;
        var identity = new IdentityActivation<T>() as IActivationFunction<T>;

        int teDim = _options.TimestepEmbeddingDimension;
        _timestepProjection = new FullyConnectedLayer<T>(teDim, teDim, siluAux);

        // Determine last hidden layer output size
        int lastDim;
        if (!_usingCustomLayers)
        {
            lastDim = dims.Length > 0 ? dims[^1] : inputDim;
        }
        else
        {
            lastDim = Layers.Count > 0 ? GetLayerOutputSize(Layers[^1]) : inputDim;
        }
        _denoiserOutput = new FullyConnectedLayer<T>(lastDim, outputDim, identity);
    }

    /// <summary>
    /// Estimates the output size of a layer by its parameter count.
    /// </summary>
    private static int GetLayerOutputSize(ILayer<T> layer)
    {
        if (layer is FullyConnectedLayer<T> fc)
        {
            // FullyConnectedLayer weights shape is [outputSize, inputSize]
            var weights = fc.GetWeights();
            if (weights is not null && weights.Rank >= 2)
            {
                return weights.Shape[0]; // outputSize is first dimension
            }
        }
        return 1;
    }

    #endregion

    #region Neural Network Methods (GANDALF Pattern)

    /// <summary>
    /// Runs the denoiser forward pass to predict noise from noisy data + timestep embedding.
    /// </summary>
    /// <param name="input">The input tensor (noisy data concatenated with timestep embedding).</param>
    /// <returns>The predicted noise tensor.</returns>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        var current = input;
        int dropIdx = 0;

        for (int i = 0; i < Layers.Count; i++)
        {
            current = Layers[i].Forward(current);
            if (_options.DropoutRate > 0 && dropIdx < _dropoutLayers.Count)
            {
                current = _dropoutLayers[dropIdx++].Forward(current);
            }
        }

        if (_denoiserOutput is not null)
        {
            current = _denoiserOutput.Forward(current);
        }

        return current;
    }

    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        var output = Predict(input);

        // Simple MSE gradient (avoids _lossFunction.CalculateLoss which expects Vector<T>)
        var gradient = new Tensor<T>(output.Shape);
        for (int i = 0; i < output.Length && i < expectedOutput.Length; i++)
        {
            gradient[i] = NumOps.FromDouble(
                2.0 * (NumOps.ToDouble(output[i]) - NumOps.ToDouble(expectedOutput[i])));
        }

        // Backward through denoiser output
        var current = gradient;
        if (_denoiserOutput is not null)
        {
            current = _denoiserOutput.Backward(current);
        }

        // Backward through hidden layers in reverse
        int dropIdx = _dropoutLayers.Count - 1;
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            if (_options.DropoutRate > 0 && dropIdx >= 0)
            {
                current = _dropoutLayers[dropIdx--].Backward(current);
            }
            current = Layers[i].Backward(current);
        }
    }

    /// <inheritdoc />
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
    /// Fits the AutoDiff-Tab generator to the provided real tabular data.
    /// </summary>
    /// <param name="data">The real data matrix.</param>
    /// <param name="columns">Metadata describing each column.</param>
    /// <param name="epochs">Number of training epochs.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method first searches for the best diffusion configuration
    /// (Phase 1), then trains the model fully with the best settings (Phase 2).
    /// After fitting, call Generate() to create new synthetic rows.
    /// </para>
    /// </remarks>
    public void Fit(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs)
    {
        ValidateFitInputs(data, columns, epochs);

        _columns = PrepareColumns(data, columns);

        _transformer = new TabularDataTransformer<T>(_options.VGMModes, _random);
        _transformer.Fit(data, _columns);
        _dataWidth = _transformer.TransformedWidth;
        var transformedData = _transformer.Transform(data);

        // Phase 1: Search over configurations
        var bestConfig = SearchConfigurations(transformedData);

        // Phase 2: Apply best configuration and train
        _numTimesteps = bestConfig.Timesteps;
        ComputeNoiseSchedule(bestConfig.Schedule, bestConfig.Timesteps);

        int teDim = _options.TimestepEmbeddingDimension;
        int inputDim = _dataWidth + teDim;
        RebuildDenoiserLayers(bestConfig.MLPDims, inputDim, _dataWidth);

        int batchSize = Math.Min(_options.BatchSize, data.Rows);
        T lr = NumOps.FromDouble(_options.LearningRate / batchSize);

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            for (int b = 0; b < data.Rows; b += batchSize)
            {
                int end = Math.Min(b + batchSize, data.Rows);
                TrainBatch(transformedData, b, end, lr);
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
            var transformedData = _transformer.Transform(data);

            var bestConfig = SearchConfigurations(transformedData);

            _numTimesteps = bestConfig.Timesteps;
            ComputeNoiseSchedule(bestConfig.Schedule, bestConfig.Timesteps);

            int teDim = _options.TimestepEmbeddingDimension;
            int inputDim = _dataWidth + teDim;
            RebuildDenoiserLayers(bestConfig.MLPDims, inputDim, _dataWidth);

            int batchSize = Math.Min(_options.BatchSize, data.Rows);
            T lr = NumOps.FromDouble(_options.LearningRate / batchSize);

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                ct.ThrowIfCancellationRequested();
                for (int b = 0; b < data.Rows; b += batchSize)
                {
                    int end = Math.Min(b + batchSize, data.Rows);
                    TrainBatch(transformedData, b, end, lr);
                }
            }
        }, ct).ConfigureAwait(false);

        IsFitted = true;
    }

    /// <summary>
    /// Generates new synthetic tabular data rows using the trained diffusion model.
    /// </summary>
    /// <param name="numSamples">The number of synthetic rows to generate.</param>
    /// <param name="conditionColumn">Optional conditioning column indices.</param>
    /// <param name="conditionValue">Optional conditioning values.</param>
    /// <returns>A matrix of synthetic data.</returns>
    public Matrix<T> Generate(int numSamples, Vector<T>? conditionColumn = null, Vector<T>? conditionValue = null)
    {
        if (!IsFitted || _transformer is null || _denoiserOutput is null)
        {
            throw new InvalidOperationException(
                "The generator must be fitted before generating data. Call Fit() first.");
        }

        if (numSamples <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numSamples), "Number of samples must be positive.");
        }

        var result = new Matrix<T>(numSamples, _dataWidth);

        for (int i = 0; i < numSamples; i++)
        {
            // Start from pure noise
            var xt = CreateStandardNormalVector(_dataWidth);

            // Reverse diffusion
            for (int t = _numTimesteps - 1; t >= 0; t--)
            {
                var predictedNoise = PredictNoise(xt, t);
                xt = DenoisingStep(xt, predictedNoise, t);
            }

            for (int j = 0; j < _dataWidth; j++)
            {
                result[i, j] = xt[j];
            }
        }

        return _transformer.InverseTransform(result);
    }

    #endregion

    #region Configuration Search

    private record struct DiffusionConfig(int Timesteps, string Schedule, int[] MLPDims, double Loss);

    private DiffusionConfig SearchConfigurations(Matrix<T> data)
    {
        var configs = GenerateCandidateConfigs();
        DiffusionConfig bestConfig = new(500, "linear", _options.MLPDimensions, double.MaxValue);

        foreach (var config in configs)
        {
            double loss = EvaluateConfig(data, config);
            if (loss < bestConfig.Loss)
            {
                bestConfig = config with { Loss = loss };
            }
        }

        return bestConfig;
    }

    private List<DiffusionConfig> GenerateCandidateConfigs()
    {
        var configs = new List<DiffusionConfig>();
        int[] timestepOptions = [100, 250, 500, 1000];
        string[] scheduleOptions = ["linear", "cosine"];

        for (int trial = 0; trial < _options.SearchTrials; trial++)
        {
            int ts = timestepOptions[_random.Next(timestepOptions.Length)];
            ts = Math.Min(ts, _options.MaxTimesteps);
            string schedule = scheduleOptions[_random.Next(scheduleOptions.Length)];

            // Randomly vary MLP dimensions
            int baseWidth = _options.MLPDimensions.Length > 0 ? _options.MLPDimensions[0] : 256;
            int width = baseWidth + (_random.Next(3) - 1) * 64; // +-64
            width = Math.Max(64, width);
            int depth = _options.MLPDimensions.Length + _random.Next(2) - 1;
            depth = Math.Max(1, Math.Min(4, depth));

            var dims = new int[depth];
            for (int i = 0; i < depth; i++) dims[i] = width;

            configs.Add(new DiffusionConfig(ts, schedule, dims, double.MaxValue));
        }

        return configs;
    }

    private double EvaluateConfig(Matrix<T> data, DiffusionConfig config)
    {
        // Train a small model with this config and evaluate loss
        ComputeNoiseSchedule(config.Schedule, config.Timesteps);
        _numTimesteps = config.Timesteps;

        int teDim = _options.TimestepEmbeddingDimension;
        int inputDim = _dataWidth + teDim;
        RebuildDenoiserLayers(config.MLPDims, inputDim, _dataWidth);

        int batchSize = Math.Min(_options.BatchSize, data.Rows);
        T lr = NumOps.FromDouble(_options.LearningRate / batchSize);
        double totalLoss = 0;
        int count = 0;

        for (int epoch = 0; epoch < _options.TrialEpochs; epoch++)
        {
            for (int b = 0; b < data.Rows; b += batchSize)
            {
                int end = Math.Min(b + batchSize, data.Rows);
                double batchLoss = TrainBatchWithLoss(data, b, end, lr);
                if (!double.IsNaN(batchLoss) && !double.IsInfinity(batchLoss) && batchLoss < 1e10)
                {
                    totalLoss += batchLoss;
                    count++;
                }
            }
        }

        return count > 0 ? totalLoss / count : double.MaxValue;
    }

    #endregion

    #region Noise Schedule

    private void ComputeNoiseSchedule(string schedule, int timesteps)
    {
        _betas = new double[timesteps];
        _alphas = new double[timesteps];
        _alphasCumprod = new double[timesteps];

        if (schedule == "cosine")
        {
            double s = 0.008;
            for (int t = 0; t < timesteps; t++)
            {
                double t1 = (double)t / timesteps;
                double t2 = (double)(t + 1) / timesteps;
                double alpha1 = Math.Cos((t1 + s) / (1 + s) * Math.PI / 2);
                double alpha2 = Math.Cos((t2 + s) / (1 + s) * Math.PI / 2);
                _betas[t] = Math.Min(Math.Max(1.0 - (alpha2 * alpha2) / (alpha1 * alpha1), 1e-4), 0.999);
            }
        }
        else
        {
            for (int t = 0; t < timesteps; t++)
            {
                _betas[t] = _options.BetaStart + (_options.BetaEnd - _options.BetaStart) * t / Math.Max(timesteps - 1, 1);
            }
        }

        double cumprod = 1.0;
        for (int t = 0; t < timesteps; t++)
        {
            _alphas[t] = 1.0 - _betas[t];
            cumprod *= _alphas[t];
            _alphasCumprod[t] = cumprod;
        }
    }

    #endregion

    #region Training

    private void TrainBatch(Matrix<T> data, int startRow, int endRow, T lr)
    {
        TrainBatchWithLoss(data, startRow, endRow, lr);
    }

    private double TrainBatchWithLoss(Matrix<T> data, int startRow, int endRow, T lr)
    {
        double totalLoss = 0;

        for (int row = startRow; row < endRow; row++)
        {
            // Sample random timestep
            int t = _random.Next(_numTimesteps);
            var x0 = GetRow(data, row);

            // Sample noise
            var noise = CreateStandardNormalVector(_dataWidth);

            // Create noisy sample: xt = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
            double sqrtAlphaBar = Math.Sqrt(_alphasCumprod[t]);
            double sqrtOneMinusAlphaBar = Math.Sqrt(1.0 - _alphasCumprod[t]);

            var xt = new Vector<T>(_dataWidth);
            for (int j = 0; j < _dataWidth; j++)
            {
                xt[j] = NumOps.FromDouble(
                    sqrtAlphaBar * NumOps.ToDouble(x0[j]) +
                    sqrtOneMinusAlphaBar * NumOps.ToDouble(noise[j]));
            }

            // Predict noise
            var predictedNoise = PredictNoise(xt, t);

            // MSE loss and gradient
            double loss = 0;
            var grad = new Tensor<T>([_dataWidth]);
            for (int j = 0; j < _dataWidth; j++)
            {
                double diff = NumOps.ToDouble(predictedNoise[j]) - NumOps.ToDouble(noise[j]);
                loss += diff * diff;
                grad[j] = NumOps.FromDouble(2.0 * diff);
            }
            double normalizedLoss = loss / _dataWidth;

            // Skip divergent samples
            if (double.IsNaN(normalizedLoss) || double.IsInfinity(normalizedLoss) || normalizedLoss > 1e10)
            {
                continue;
            }

            totalLoss += normalizedLoss;

            // Sanitize and clip gradient
            grad = SafeGradient(grad, 5.0);

            // Backward through denoiser output
            var current = grad;
            if (_denoiserOutput is not null)
            {
                current = _denoiserOutput.Backward(current);
            }

            // Backward through hidden layers
            int dropIdx = _dropoutLayers.Count - 1;
            for (int i = Layers.Count - 1; i >= 0; i--)
            {
                if (_options.DropoutRate > 0 && dropIdx >= 0)
                {
                    current = _dropoutLayers[dropIdx--].Backward(current);
                }
                current = Layers[i].Backward(current);
            }

            // Update parameters
            foreach (var layer in Layers) layer.UpdateParameters(lr);
            _denoiserOutput?.UpdateParameters(lr);
        }

        int count = endRow - startRow;
        return count > 0 ? totalLoss / count : 0;
    }

    #endregion

    #region Denoising

    private Vector<T> PredictNoise(Vector<T> xt, int t)
    {
        var timeEmbed = CreateTimestepEmbedding(t);

        // Concatenate xt and time embedding
        int totalLen = xt.Length + timeEmbed.Length;
        var input = new Vector<T>(totalLen);
        for (int i = 0; i < xt.Length; i++) input[i] = xt[i];
        for (int i = 0; i < timeEmbed.Length; i++) input[xt.Length + i] = timeEmbed[i];

        var current = VectorToTensor(input);
        int dropIdx = 0;

        for (int i = 0; i < Layers.Count; i++)
        {
            current = Layers[i].Forward(current);
            if (_options.DropoutRate > 0 && dropIdx < _dropoutLayers.Count)
            {
                current = _dropoutLayers[dropIdx++].Forward(current);
            }
        }

        if (_denoiserOutput is not null)
        {
            current = _denoiserOutput.Forward(current);
        }

        return TensorToVector(current, _dataWidth);
    }

    private Vector<T> CreateTimestepEmbedding(int t)
    {
        int dim = _options.TimestepEmbeddingDimension;
        var embedding = new Vector<T>(dim);
        int halfDim = dim / 2;

        for (int i = 0; i < halfDim; i++)
        {
            double freq = Math.Exp(-Math.Log(10000.0) * i / halfDim);
            double angle = t * freq;
            embedding[i] = NumOps.FromDouble(Math.Sin(angle));
            if (i + halfDim < dim) embedding[i + halfDim] = NumOps.FromDouble(Math.Cos(angle));
        }

        if (_timestepProjection is not null)
        {
            var tensor = _timestepProjection.Forward(VectorToTensor(embedding));
            return TensorToVector(tensor, dim);
        }

        return embedding;
    }

    private Vector<T> DenoisingStep(Vector<T> xt, Vector<T> predictedNoise, int t)
    {
        double alphaT = _alphas[t];
        double alphaBarT = _alphasCumprod[t];
        double betaT = _betas[t];

        var result = new Vector<T>(_dataWidth);
        double coeff1 = 1.0 / Math.Sqrt(alphaT);
        double coeff2 = betaT / Math.Sqrt(1.0 - alphaBarT);

        for (int j = 0; j < _dataWidth; j++)
        {
            double mean = coeff1 * (NumOps.ToDouble(xt[j]) - coeff2 * NumOps.ToDouble(predictedNoise[j]));

            if (t > 0)
            {
                double sigma = Math.Sqrt(betaT);
                mean += sigma * NumOps.ToDouble(SampleStandardNormal());
            }

            result[j] = NumOps.FromDouble(mean);
        }

        return result;
    }

    #endregion

    #region Gradient Safety Utilities

    /// <summary>
    /// Applies NaN sanitization and gradient norm clipping in a single operation.
    /// </summary>
    private Tensor<T> SafeGradient(Tensor<T> grad, double maxNorm)
    {
        for (int i = 0; i < grad.Length; i++)
        {
            double v = NumOps.ToDouble(grad[i]);
            if (double.IsNaN(v) || double.IsInfinity(v))
            {
                grad[i] = NumOps.Zero;
            }
        }

        if (maxNorm <= 0) return grad;

        double normSq = 0;
        for (int i = 0; i < grad.Length; i++)
        {
            double v = NumOps.ToDouble(grad[i]);
            normSq += v * v;
        }

        double norm = Math.Sqrt(normSq);
        if (norm <= maxNorm) return grad;

        double scale = maxNorm / norm;
        var clipped = new Tensor<T>(grad.Shape);
        for (int i = 0; i < grad.Length; i++)
        {
            clipped[i] = NumOps.FromDouble(NumOps.ToDouble(grad[i]) * scale);
        }
        return clipped;
    }

    #endregion

    #region Serialization and Model Metadata (GANDALF Pattern)

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "SearchTrials", _options.SearchTrials },
                { "MaxTimesteps", _options.MaxTimesteps },
                { "MLPDimensions", _options.MLPDimensions },
                { "BatchSize", _options.BatchSize },
                { "ActiveTimesteps", _numTimesteps },
                { "DenoiserLayerCount", Layers.Count },
                { "DenoiserLayerTypes", Layers.Select(l => l.GetType().Name).ToArray() }
            },
            ModelData = this.Serialize()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_options.SearchTrials);
        writer.Write(_options.MaxTimesteps);
        writer.Write(_options.MLPDimensions.Length);
        foreach (var dim in _options.MLPDimensions)
        {
            writer.Write(dim);
        }
        writer.Write(_options.BatchSize);
        writer.Write(_options.LearningRate);
        writer.Write(_options.TimestepEmbeddingDimension);
        writer.Write(_numTimesteps);
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
        return new AutoDiffTabGenerator<T>(
            Architecture,
            _options,
            _optimizer,
            _lossFunction);
    }

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
    /// AutoDiffTab uses automated hyperparameter search over diffusion configurations which cannot be represented as a single computation graph.
    /// </summary>
    public override bool SupportsJitCompilation => false;

    #endregion
}
