using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.NeuralNetworks.SyntheticData;

/// <summary>
/// TabDDPM (Tabular Denoising Diffusion Probabilistic Model) for generating synthetic tabular data.
/// </summary>
/// <remarks>
/// <para>
/// TabDDPM applies diffusion models to tabular data with separate processes for different feature types:
/// - <b>Gaussian diffusion</b> for continuous/numerical features (noise prediction)
/// - <b>Multinomial diffusion</b> for categorical features (category probability prediction)
/// - <b>Shared MLP denoiser</b> with sinusoidal timestep embedding processes both types jointly
/// - <b>Simple preprocessing</b>: Quantile normalization for continuous, integer encoding for categorical
/// </para>
/// <para>
/// This implementation follows the standard neural network architecture pattern with:
/// - Proper inheritance from NeuralNetworkBase
/// - Layer-based architecture using ILayer components
/// - Engine-based tensor operations for CPU/GPU acceleration
/// - Full autodiff and JIT compilation support
/// </para>
/// <para>
/// <b>For Beginners:</b> TabDDPM generates data by learning to "undo noise."
///
/// <b>Training:</b>
/// <code>
/// 1. Take a real data row
/// 2. Pick a random noise level (timestep t)
/// 3. Add noise: numbers get Gaussian noise, categories get randomly flipped
/// 4. Feed noisy data + timestep to the MLP
/// 5. MLP predicts: what noise was added (for numbers) / what original category (for categories)
/// 6. Compare predictions to truth and update the MLP
/// </code>
///
/// <b>Generation:</b>
/// <code>
/// 1. Start with pure random noise (numbers = random, categories = random)
/// 2. For t = T, T-1, T-2, ..., 1, 0:
///    a. Feed noisy data + timestep to MLP
///    b. MLP predicts the noise/categories
///    c. Remove a small amount of noise
/// 3. Final result is a clean, realistic data row
/// </code>
///
/// If you provide custom layers in the architecture, those will be used directly
/// for the denoiser MLP. If not, the network creates industry-standard
/// TabDDPM layers based on the original research paper specifications.
///
/// Example usage:
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputFeatures: 10,
///     outputSize: 10
/// );
/// var options = new TabDDPMOptions&lt;double&gt;
/// {
///     NumTimesteps = 1000,
///     MLPDimensions = new[] { 256, 256, 256 },
///     BatchSize = 4096,
///     Epochs = 1000
/// };
/// var tabddpm = new TabDDPMGenerator&lt;double&gt;(architecture, options);
/// tabddpm.Fit(data, columns, epochs: 1000);
/// var synthetic = tabddpm.Generate(1000);
/// </code>
///
/// TabDDPM often produces higher-quality synthetic data than CTGAN/TVAE, especially
/// for complex distributions, at the cost of slower generation (many denoising steps).
/// </para>
/// <para>
/// Reference: "TabDDPM: Modelling Tabular Data with Diffusion Models" (Kotelnikov et al., ICML 2023)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TabDDPMGenerator<T> : NeuralNetworkBase<T>, ISyntheticTabularGenerator<T>
{
    private readonly TabDDPMOptions<T> _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private ILossFunction<T> _lossFunction;

    // Synthetic tabular data infrastructure
    private List<ColumnMetadata> _columns = new();
    private Random _random;

    // Diffusion processes
    private GaussianDiffusion<T>? _gaussianDiffusion;
    private MultinomialDiffusion<T>? _multinomialDiffusion;

    // Output heads (auxiliary, not user-overridable)
    private FullyConnectedLayer<T>? _numericalOutputHead;
    private FullyConnectedLayer<T>? _categoricalOutputHead;

    // Timestep embedding layer (auxiliary)
    private FullyConnectedLayer<T>? _timestepProjection;

    // Feature dimensions
    private int _numNumericalFeatures;
    private int _numCategoricalFeatures;
    private int _totalCategoricalWidth;
    private int _inputWidth;
    private int _lastHiddenDim;
    private Tensor<T>? _lastMLPOutput;

    // Column layout tracking
    private readonly List<int> _numericalColumnIndices = new();
    private readonly List<int> _categoricalColumnIndices = new();
    private readonly List<int> _categoricalColumnWidths = new();

    // Preprocessing stats
    private double[] _quantileMeans = Array.Empty<double>();
    private double[] _quantileStds = Array.Empty<double>();

    // Whether custom layers are being used
    private bool _usingCustomLayers;

    /// <summary>
    /// Gets the TabDDPM-specific options.
    /// </summary>
    public new TabDDPMOptions<T> Options => _options;

    /// <inheritdoc />
    public IReadOnlyList<ColumnMetadata> Columns => _columns.AsReadOnly();

    /// <inheritdoc />
    public bool IsFitted { get; private set; }

    /// <summary>
    /// Initializes a new TabDDPM generator with the specified architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining input/output dimensions and optional custom layers.</param>
    /// <param name="options">TabDDPM-specific options for diffusion and MLP configuration.</param>
    /// <param name="optimizer">Gradient-based optimizer (defaults to Adam).</param>
    /// <param name="lossFunction">Loss function (defaults based on task type).</param>
    /// <param name="maxGradNorm">Maximum gradient norm for clipping (default 5.0).</param>
    public TabDDPMGenerator(
        NeuralNetworkArchitecture<T> architecture,
        TabDDPMOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 5.0)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), maxGradNorm)
    {
        _options = options ?? new TabDDPMOptions<T>();
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        int? seed = _options.Seed;
        _random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();

        InitializeLayers();
    }

    #region Layer Initialization (GANDALF Pattern)

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            _usingCustomLayers = true;
        }
        else
        {
            // Create default denoiser MLP layers
            int inputDim = Architecture.CalculatedInputSize + _options.TimestepEmbeddingDimension;
            Layers.AddRange(LayerHelper<T>.CreateDefaultTabDDPMDenoiserLayers(
                inputDim, _options.MLPDimensions, _options.DropoutRate));
            _usingCustomLayers = false;
        }

        // Timestep projection is always internal
        _timestepProjection = new FullyConnectedLayer<T>(
            _options.TimestepEmbeddingDimension, _options.TimestepEmbeddingDimension,
            new SiLUActivation<T>() as IActivationFunction<T>);
    }

    /// <summary>
    /// Rebuilds denoiser layers with actual data dimensions discovered during Fit().
    /// </summary>
    private void RebuildLayersWithActualDimensions()
    {
        _inputWidth = _numNumericalFeatures + _totalCategoricalWidth + _options.TimestepEmbeddingDimension;
        _lastHiddenDim = _options.MLPDimensions.Length > 0
            ? _options.MLPDimensions[^1]
            : _inputWidth;

        if (!_usingCustomLayers)
        {
            Layers.Clear();
            Layers.AddRange(LayerHelper<T>.CreateDefaultTabDDPMDenoiserLayers(
                _inputWidth, _options.MLPDimensions, _options.DropoutRate));
        }

        // Always rebuild output heads with actual dimensions
        var identity = new IdentityActivation<T>() as IActivationFunction<T>;
        int lastHidden = _usingCustomLayers
            ? GetLastLayerOutputSize()
            : _lastHiddenDim;

        if (_numNumericalFeatures > 0)
        {
            _numericalOutputHead = new FullyConnectedLayer<T>(lastHidden, _numNumericalFeatures, identity);
        }

        if (_totalCategoricalWidth > 0)
        {
            _categoricalOutputHead = new FullyConnectedLayer<T>(lastHidden, _totalCategoricalWidth, identity);
        }
    }

    private int GetLastLayerOutputSize()
    {
        if (Layers.Count == 0) return _inputWidth;
        var shape = Layers[^1].GetOutputShape();
        if (shape is not null && shape.Length > 0) return shape[0];
        return _options.MLPDimensions.Length > 0 ? _options.MLPDimensions[^1] : _inputWidth;
    }

    #endregion

    #region Neural Network Methods (GANDALF Pattern)

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        Tensor<T> current = input;
        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }
        return current;
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        Tensor<T> prediction = Predict(input);
        LastLoss = _lossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());
        Tensor<T> error = prediction.Subtract(expectedOutput);

        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            error = Layers[i].Backward(error);
        }
        _optimizer.UpdateParameters(Layers);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int startIndex = 0;
        foreach (var layer in Layers)
        {
            int count = layer.ParameterCount;
            if (count > 0)
            {
                layer.UpdateParameters(parameters.SubVector(startIndex, count));
                startIndex += count;
            }
        }
    }

    #endregion

    #region ISyntheticTabularGenerator<T> Implementation

    /// <inheritdoc/>
    public void Fit(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs)
    {
        ValidateFitInputs(data, columns, epochs);

        _columns = PrepareColumns(data, columns);

        // Step 1: Analyze column layout
        AnalyzeColumns(_columns);

        // Step 2: Preprocess data
        var (numericalData, categoricalData) = PreprocessData(data, _columns);

        // Step 3: Initialize diffusion processes
        _gaussianDiffusion = new GaussianDiffusion<T>(
            _options.NumTimesteps, _options.BetaStart, _options.BetaEnd,
            _options.BetaSchedule, _random);

        _multinomialDiffusion = new MultinomialDiffusion<T>(
            _options.NumCategoricalDiffusionSteps, _options.BetaStart, _options.BetaEnd, _random);

        // Step 4: Build layers with actual dimensions
        RebuildLayersWithActualDimensions();

        // Step 5: Training loop
        int batchSize = Math.Min(_options.BatchSize, data.Rows);
        int numBatches = Math.Max(1, data.Rows / batchSize);
        T scaledLr = NumOps.FromDouble(_options.LearningRate / batchSize);

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            for (int batch = 0; batch < numBatches; batch++)
            {
                int startRow = batch * batchSize;
                int endRow = Math.Min(startRow + batchSize, data.Rows);
                TrainBatch(numericalData, categoricalData, startRow, endRow, scaledLr);
            }
        }

        IsFitted = true;
    }

    /// <inheritdoc/>
    public async Task FitAsync(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs, CancellationToken ct = default)
    {
        ValidateFitInputs(data, columns, epochs);

        _columns = PrepareColumns(data, columns);

        await Task.Run(() =>
        {
            ct.ThrowIfCancellationRequested();

            AnalyzeColumns(_columns);
            var (numericalData, categoricalData) = PreprocessData(data, _columns);

            _gaussianDiffusion = new GaussianDiffusion<T>(
                _options.NumTimesteps, _options.BetaStart, _options.BetaEnd,
                _options.BetaSchedule, _random);

            _multinomialDiffusion = new MultinomialDiffusion<T>(
                _options.NumCategoricalDiffusionSteps, _options.BetaStart, _options.BetaEnd, _random);

            RebuildLayersWithActualDimensions();

            int batchSize = Math.Min(_options.BatchSize, data.Rows);
            int numBatches = Math.Max(1, data.Rows / batchSize);
            T scaledLr = NumOps.FromDouble(_options.LearningRate / batchSize);

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                ct.ThrowIfCancellationRequested();
                for (int batch = 0; batch < numBatches; batch++)
                {
                    int startRow = batch * batchSize;
                    int endRow = Math.Min(startRow + batchSize, data.Rows);
                    TrainBatch(numericalData, categoricalData, startRow, endRow, scaledLr);
                }
            }

            IsFitted = true;
        }, ct).ConfigureAwait(false);
    }

    /// <inheritdoc/>
    public Matrix<T> Generate(int numSamples, Vector<T>? conditionColumn = null, Vector<T>? conditionValue = null)
    {
        if (_gaussianDiffusion is null || _multinomialDiffusion is null || !IsFitted)
        {
            throw new InvalidOperationException("Generator must be fitted before generating data. Call Fit() first.");
        }

        // Start from pure noise
        var numericalSamples = new Matrix<T>(numSamples, _numNumericalFeatures);
        var categoricalSamples = InitializeRandomCategories(numSamples);

        // Initialize numerical features with standard normal noise
        for (int i = 0; i < numSamples; i++)
        {
            for (int j = 0; j < _numNumericalFeatures; j++)
            {
                numericalSamples[i, j] = SampleStandardNormal();
            }
        }

        // Reverse diffusion: denoise step by step
        int numSteps = _options.NumTimesteps;
        int catSteps = _options.NumCategoricalDiffusionSteps;

        for (int t = numSteps - 1; t >= 0; t--)
        {
            for (int i = 0; i < numSamples; i++)
            {
                var numFeats = GetMatrixRow(numericalSamples, i);
                var catFeats = GetMatrixRow(categoricalSamples, i);
                var timeEmbed = CreateTimestepEmbedding(t);

                var (predictedNoise, predictedLogits) = DenoiserForward(numFeats, catFeats, timeEmbed);

                numericalSamples = SetMatrixRow(numericalSamples, i,
                    _gaussianDiffusion.DenoisingStep(numFeats, predictedNoise, t));

                int catT = (int)((long)t * catSteps / numSteps);
                if (catT < catSteps && _totalCategoricalWidth > 0)
                {
                    var denoisedCat = DenoiseCategorical(catFeats, predictedLogits, catT);
                    categoricalSamples = SetMatrixRow(categoricalSamples, i, denoisedCat);
                }
            }
        }

        return ReconstructData(numericalSamples, categoricalSamples, numSamples);
    }

    #endregion

    #region Column Analysis and Preprocessing

    private void AnalyzeColumns(IReadOnlyList<ColumnMetadata> columns)
    {
        _numericalColumnIndices.Clear();
        _categoricalColumnIndices.Clear();
        _categoricalColumnWidths.Clear();

        _numNumericalFeatures = 0;
        _totalCategoricalWidth = 0;

        for (int col = 0; col < columns.Count; col++)
        {
            if (columns[col].IsNumerical)
            {
                _numericalColumnIndices.Add(col);
                _numNumericalFeatures++;
            }
            else if (columns[col].IsCategorical)
            {
                _categoricalColumnIndices.Add(col);
                int numCats = Math.Max(columns[col].NumCategories, 2);
                _categoricalColumnWidths.Add(numCats);
                _totalCategoricalWidth += numCats;
            }
        }

        _numCategoricalFeatures = _categoricalColumnIndices.Count;
    }

    private (Matrix<T> Numerical, Matrix<T> Categorical) PreprocessData(
        Matrix<T> data, IReadOnlyList<ColumnMetadata> columns)
    {
        int n = data.Rows;

        // Quantile normalization for numerical columns
        var numerical = new Matrix<T>(n, _numNumericalFeatures);
        _quantileMeans = new double[_numNumericalFeatures];
        _quantileStds = new double[_numNumericalFeatures];

        for (int j = 0; j < _numNumericalFeatures; j++)
        {
            int colIdx = _numericalColumnIndices[j];
            double sum = 0;
            for (int i = 0; i < n; i++)
            {
                sum += NumOps.ToDouble(data[i, colIdx]);
            }
            double mean = sum / n;

            double sumSq = 0;
            for (int i = 0; i < n; i++)
            {
                double diff = NumOps.ToDouble(data[i, colIdx]) - mean;
                sumSq += diff * diff;
            }
            double std = n > 1 ? Math.Sqrt(sumSq / (n - 1)) : 1.0;
            if (std < 1e-10) std = 1.0;

            _quantileMeans[j] = mean;
            _quantileStds[j] = std;

            for (int i = 0; i < n; i++)
            {
                double val = (NumOps.ToDouble(data[i, colIdx]) - mean) / std;
                numerical[i, j] = NumOps.FromDouble(val);
            }
        }

        // One-hot encoding for categorical columns
        var categorical = new Matrix<T>(n, _totalCategoricalWidth);
        int catOffset = 0;

        for (int j = 0; j < _numCategoricalFeatures; j++)
        {
            int colIdx = _categoricalColumnIndices[j];
            int numCats = _categoricalColumnWidths[j];

            for (int i = 0; i < n; i++)
            {
                int catVal = (int)Math.Round(NumOps.ToDouble(data[i, colIdx]));
                if (catVal >= 0 && catVal < numCats)
                {
                    categorical[i, catOffset + catVal] = NumOps.One;
                }
            }

            catOffset += numCats;
        }

        return (numerical, categorical);
    }

    #endregion

    #region Timestep Embedding

    private Vector<T> CreateTimestepEmbedding(int timestep)
    {
        int dim = _options.TimestepEmbeddingDimension;
        var embedding = new Vector<T>(dim);

        int halfDim = dim / 2;
        for (int i = 0; i < halfDim; i++)
        {
            double freq = Math.Exp(-Math.Log(10000.0) * i / halfDim);
            double angle = timestep * freq;
            embedding[i] = NumOps.FromDouble(Math.Sin(angle));
            if (i + halfDim < dim)
            {
                embedding[i + halfDim] = NumOps.FromDouble(Math.Cos(angle));
            }
        }

        if (_timestepProjection is not null)
        {
            var embedTensor = VectorToTensor(embedding);
            var projected = _timestepProjection.Forward(embedTensor);
            return TensorToVector(projected, dim);
        }

        return embedding;
    }

    #endregion

    #region Denoiser Forward

    private (Vector<T> NoisePred, Vector<T> CatLogits) DenoiserForward(
        Vector<T> numericalFeatures, Vector<T> categoricalFeatures, Vector<T> timestepEmbed)
    {
        int totalLen = _numNumericalFeatures + _totalCategoricalWidth + _options.TimestepEmbeddingDimension;
        var input = new Vector<T>(totalLen);

        int offset = 0;
        for (int i = 0; i < numericalFeatures.Length; i++) input[offset++] = numericalFeatures[i];
        for (int i = 0; i < categoricalFeatures.Length; i++) input[offset++] = categoricalFeatures[i];
        for (int i = 0; i < timestepEmbed.Length; i++) input[offset++] = timestepEmbed[i];

        // Forward through MLP layers
        var current = VectorToTensor(input);
        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }
        _lastMLPOutput = current;

        // Split output via separate heads
        var noisePred = new Vector<T>(_numNumericalFeatures);
        var catLogits = new Vector<T>(_totalCategoricalWidth);

        if (_numericalOutputHead is not null)
        {
            var numOut = _numericalOutputHead.Forward(current);
            for (int i = 0; i < _numNumericalFeatures && i < numOut.Length; i++)
            {
                noisePred[i] = numOut[i];
            }
        }

        if (_categoricalOutputHead is not null)
        {
            var catOut = _categoricalOutputHead.Forward(current);
            for (int i = 0; i < _totalCategoricalWidth && i < catOut.Length; i++)
            {
                catLogits[i] = catOut[i];
            }
        }

        return (noisePred, catLogits);
    }

    #endregion

    #region Training

    private void TrainBatch(Matrix<T> numericalData, Matrix<T> categoricalData,
        int startRow, int endRow, T scaledLearningRate)
    {
        if (_gaussianDiffusion is null || _multinomialDiffusion is null) return;

        for (int row = startRow; row < endRow; row++)
        {
            int t = _gaussianDiffusion.SampleTimestep();
            int catT = (int)((long)t * _options.NumCategoricalDiffusionSteps / _options.NumTimesteps);

            var numClean = GetMatrixRow(numericalData, row);
            var catClean = GetMatrixRow(categoricalData, row);

            Vector<T> numNoisy;
            Vector<T> actualNoise;
            if (_numNumericalFeatures > 0)
            {
                (numNoisy, actualNoise) = _gaussianDiffusion.AddNoise(numClean, t);
            }
            else
            {
                numNoisy = numClean;
                actualNoise = new Vector<T>(0);
            }

            Vector<T> catNoisy;
            if (_totalCategoricalWidth > 0 && catT < _multinomialDiffusion.NumTimesteps)
            {
                catNoisy = _multinomialDiffusion.AddNoise(catClean, catT);
            }
            else
            {
                catNoisy = catClean;
            }

            var timeEmbed = CreateTimestepEmbedding(t);
            var (predictedNoise, predictedLogits) = DenoiserForward(numNoisy, catNoisy, timeEmbed);

            var numGrad = new Vector<T>(_numNumericalFeatures);
            if (_numNumericalFeatures > 0)
            {
                numGrad = _gaussianDiffusion.ComputeLossGradient(predictedNoise, actualNoise);
            }

            var catGrad = new Vector<T>(_totalCategoricalWidth);
            if (_totalCategoricalWidth > 0)
            {
                catGrad = ComputeCategoricalGradient(predictedLogits, catClean);
            }

            numGrad = SafeGradient(numGrad, 5.0);
            catGrad = SafeGradient(catGrad, 5.0);

            BackwardDenoiser(numGrad, catGrad);
            UpdateDenoiserParameters(scaledLearningRate);
        }
    }

    private Vector<T> ComputeCategoricalGradient(Vector<T> predictedLogits, Vector<T> trueOneHot)
    {
        var grad = new Vector<T>(_totalCategoricalWidth);
        int offset = 0;

        for (int j = 0; j < _numCategoricalFeatures; j++)
        {
            int numCats = _categoricalColumnWidths[j];

            double maxVal = double.MinValue;
            for (int c = 0; c < numCats && (offset + c) < predictedLogits.Length; c++)
            {
                double v = NumOps.ToDouble(predictedLogits[offset + c]);
                if (v > maxVal) maxVal = v;
            }

            double sumExp = 0;
            for (int c = 0; c < numCats && (offset + c) < predictedLogits.Length; c++)
            {
                sumExp += Math.Exp(NumOps.ToDouble(predictedLogits[offset + c]) - maxVal);
            }

            for (int c = 0; c < numCats; c++)
            {
                if (offset + c < predictedLogits.Length)
                {
                    double softmax = Math.Exp(NumOps.ToDouble(predictedLogits[offset + c]) - maxVal) / Math.Max(sumExp, 1e-10);
                    double target = offset + c < trueOneHot.Length ? NumOps.ToDouble(trueOneHot[offset + c]) : 0;
                    grad[offset + c] = NumOps.FromDouble(softmax - target);
                }
            }

            offset += numCats;
        }

        return grad;
    }

    #endregion

    #region Backward Pass

    private void BackwardDenoiser(Vector<T> numGrad, Vector<T> catGrad)
    {
        Tensor<T>? gradFromNumHead = null;
        Tensor<T>? gradFromCatHead = null;

        if (_numericalOutputHead is not null && numGrad.Length > 0)
        {
            gradFromNumHead = _numericalOutputHead.Backward(VectorToTensor(numGrad));
        }

        if (_categoricalOutputHead is not null && catGrad.Length > 0)
        {
            gradFromCatHead = _categoricalOutputHead.Backward(VectorToTensor(catGrad));
        }

        if (Layers.Count == 0 || _lastMLPOutput is null) return;

        var mlpGrad = new Tensor<T>(_lastMLPOutput.Shape);
        if (gradFromNumHead is not null)
        {
            for (int i = 0; i < mlpGrad.Length && i < gradFromNumHead.Length; i++)
            {
                mlpGrad[i] = NumOps.FromDouble(
                    NumOps.ToDouble(mlpGrad[i]) + NumOps.ToDouble(gradFromNumHead[i]));
            }
        }
        if (gradFromCatHead is not null)
        {
            for (int i = 0; i < mlpGrad.Length && i < gradFromCatHead.Length; i++)
            {
                mlpGrad[i] = NumOps.FromDouble(
                    NumOps.ToDouble(mlpGrad[i]) + NumOps.ToDouble(gradFromCatHead[i]));
            }
        }

        var current = mlpGrad;
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            current = Layers[i].Backward(current);
        }
    }

    private void UpdateDenoiserParameters(T learningRate)
    {
        foreach (var layer in Layers)
        {
            layer.UpdateParameters(learningRate);
        }

        _numericalOutputHead?.UpdateParameters(learningRate);
        _categoricalOutputHead?.UpdateParameters(learningRate);
    }

    #endregion

    #region Sampling Helpers

    private Matrix<T> InitializeRandomCategories(int numSamples)
    {
        var cats = new Matrix<T>(numSamples, _totalCategoricalWidth);
        int offset = 0;

        for (int j = 0; j < _numCategoricalFeatures; j++)
        {
            int numCats = _categoricalColumnWidths[j];
            double uniform = 1.0 / numCats;

            for (int i = 0; i < numSamples; i++)
            {
                for (int c = 0; c < numCats; c++)
                {
                    cats[i, offset + c] = NumOps.FromDouble(uniform);
                }
            }

            offset += numCats;
        }

        return cats;
    }

    private Vector<T> DenoiseCategorical(Vector<T> noisyFeats, Vector<T> predictedLogits, int catT)
    {
        if (_multinomialDiffusion is null) return noisyFeats;

        var result = new Vector<T>(_totalCategoricalWidth);
        int offset = 0;

        for (int j = 0; j < _numCategoricalFeatures; j++)
        {
            int numCats = _categoricalColumnWidths[j];

            var colNoisy = new Vector<T>(numCats);
            var colLogits = new Vector<T>(numCats);

            for (int c = 0; c < numCats; c++)
            {
                colNoisy[c] = offset + c < noisyFeats.Length ? noisyFeats[offset + c] : NumOps.Zero;
                colLogits[c] = offset + c < predictedLogits.Length ? predictedLogits[offset + c] : NumOps.Zero;
            }

            var denoised = _multinomialDiffusion.DenoisingStep(colNoisy, colLogits, catT);

            for (int c = 0; c < numCats; c++)
            {
                result[offset + c] = c < denoised.Length ? denoised[c] : NumOps.Zero;
            }

            offset += numCats;
        }

        return result;
    }

    #endregion

    #region Data Reconstruction

    private Matrix<T> ReconstructData(Matrix<T> numericalSamples, Matrix<T> categoricalSamples, int numSamples)
    {
        var result = new Matrix<T>(numSamples, _columns.Count);

        for (int i = 0; i < numSamples; i++)
        {
            int numIdx = 0;
            int catColIdx = 0;
            int catOffset = 0;

            for (int col = 0; col < _columns.Count; col++)
            {
                if (_columns[col].IsNumerical)
                {
                    double normalized = NumOps.ToDouble(numericalSamples[i, numIdx]);
                    double original = normalized * _quantileStds[numIdx] + _quantileMeans[numIdx];
                    original = Math.Max(_columns[col].Min, Math.Min(_columns[col].Max, original));

                    if (_columns[col].DataType == ColumnDataType.Discrete)
                    {
                        original = Math.Round(original);
                    }

                    result[i, col] = NumOps.FromDouble(original);
                    numIdx++;
                }
                else if (_columns[col].IsCategorical)
                {
                    int numCats = _categoricalColumnWidths[catColIdx];
                    int bestCat = 0;
                    double bestVal = double.MinValue;

                    for (int c = 0; c < numCats; c++)
                    {
                        double v = NumOps.ToDouble(categoricalSamples[i, catOffset + c]);
                        if (v > bestVal)
                        {
                            bestVal = v;
                            bestCat = c;
                        }
                    }

                    result[i, col] = NumOps.FromDouble(bestCat);
                    catOffset += numCats;
                    catColIdx++;
                }
            }
        }

        return result;
    }

    #endregion

    #region Gradient Utilities

    private Vector<T> SafeGradient(Vector<T> grad, double maxNorm)
    {
        var result = new Vector<T>(grad.Length);
        double sumSq = 0;

        for (int i = 0; i < grad.Length; i++)
        {
            double val = NumOps.ToDouble(grad[i]);
            if (double.IsNaN(val) || double.IsInfinity(val)) val = 0;
            result[i] = NumOps.FromDouble(val);
            sumSq += val * val;
        }

        double norm = Math.Sqrt(sumSq);
        if (norm > maxNorm && norm > 0)
        {
            double scale = maxNorm / norm;
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = NumOps.FromDouble(NumOps.ToDouble(result[i]) * scale);
            }
        }

        return result;
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
                { "NumTimesteps", _options.NumTimesteps },
                { "MLPDimensions", _options.MLPDimensions },
                { "TimestepEmbeddingDimension", _options.TimestepEmbeddingDimension },
                { "BatchSize", _options.BatchSize },
                { "NumNumericalFeatures", _numNumericalFeatures },
                { "TotalCategoricalWidth", _totalCategoricalWidth },
                { "DenoiserLayerCount", Layers.Count },
                { "DenoiserLayerTypes", Layers.Select(l => l.GetType().Name).ToArray() }
            },
            ModelData = this.Serialize()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_options.NumTimesteps);
        writer.Write(_options.MLPDimensions.Length);
        foreach (var dim in _options.MLPDimensions)
        {
            writer.Write(dim);
        }
        writer.Write(_options.TimestepEmbeddingDimension);
        writer.Write(_options.BatchSize);
        writer.Write(_options.LearningRate);
        writer.Write(_options.DropoutRate);
        writer.Write(_options.BetaStart);
        writer.Write(_options.BetaEnd);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Options are reconstructed from serialized data
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new TabDDPMGenerator<T>(
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
            throw new ArgumentException("Data matrix must not be empty.", nameof(data));
        if (columns.Count == 0)
            throw new ArgumentException("Column metadata list must not be empty.", nameof(columns));
        if (columns.Count != data.Columns)
            throw new ArgumentException(
                $"Column metadata count ({columns.Count}) must match data column count ({data.Columns}).", nameof(columns));
        if (epochs <= 0)
            throw new ArgumentOutOfRangeException(nameof(epochs), "Epochs must be positive.");
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

    #endregion

    #region Helpers

    private static Vector<T> GetMatrixRow(Matrix<T> data, int row)
    {
        var v = new Vector<T>(data.Columns);
        for (int j = 0; j < data.Columns; j++) v[j] = data[row, j];
        return v;
    }

    private static Matrix<T> SetMatrixRow(Matrix<T> data, int row, Vector<T> values)
    {
        for (int j = 0; j < Math.Min(data.Columns, values.Length); j++)
        {
            data[row, j] = values[j];
        }
        return data;
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
    /// TabDDPM uses iterative diffusion denoising which cannot be represented as a single computation graph.
    /// </summary>
    public override bool SupportsJitCompilation => false;

    #endregion
}
