using System.IO;
using AiDotNet.Finance.Base;
using AiDotNet.Finance.Interfaces;
using AiDotNet.Finance.Options;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Finance.Forecasting.Transformers;

/// <summary>
/// PatchTST (Patch Time Series Transformer) for long-term time series forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// PatchTST is a state-of-the-art transformer model for long-term time series forecasting.
/// It introduces two key innovations:
///
/// 1. <b>Patching:</b> Divides the input time series into patches (segments), which serves as
///    input tokens to the transformer. This reduces computation complexity from O(L²) to O(N²)
///    where N = L/P (number of patches) is much smaller than L (sequence length).
///
/// 2. <b>Channel Independence:</b> Processes each channel (variable) independently through
///    the same transformer, sharing parameters across channels. This improves generalization
///    and reduces overfitting on multivariate datasets.
/// </para>
/// <para>
/// <b>For Beginners:</b> PatchTST is like reading a book by looking at groups of words (patches)
/// instead of individual letters. This makes it much faster and often more accurate for understanding
/// the overall meaning.
///
/// Key concepts:
/// - <b>Patches:</b> Groups of consecutive time steps treated as single "tokens"
/// - <b>Self-attention:</b> The model learns which patches are most relevant for forecasting
/// - <b>Channel independence:</b> Each variable (stock price, volume, etc.) is processed separately
///
/// Example usage:
/// <code>
/// // Create options
/// var options = new PatchTSTOptions&lt;double&gt;
/// {
///     SequenceLength = 96,
///     PredictionHorizon = 24,
///     NumFeatures = 7,
///     PatchSize = 16,
///     Stride = 8
/// };
///
/// // Create model
/// var model = new PatchTST&lt;double&gt;(options);
///
/// // Train (shape: [batch, sequence, features])
/// model.Train(inputs, targets);
///
/// // Forecast (shape: [batch, horizon, features])
/// var forecast = model.Forecast(newData);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting
/// with Transformers", ICLR 2023. https://arxiv.org/abs/2211.14730
/// </para>
/// </remarks>
public class PatchTST<T> : FinancialModelBase<T>, IForecastingModel<T>
{
    #region Configuration

    /// <summary>
    /// The options used to configure this model.
    /// </summary>
    private readonly PatchTSTOptions<T> _options;

    /// <summary>
    /// Random number generator for dropout and initialization.
    /// </summary>
    private readonly Random _random;

    #endregion

    #region Model Components

    /// <summary>
    /// Patch embedding layer that projects patches to model dimension.
    /// </summary>
    private readonly DenseLayer<T> _patchEmbedding;

    /// <summary>
    /// Positional encoding for patch positions.
    /// </summary>
    private readonly Tensor<T> _positionalEncoding;

    /// <summary>
    /// Transformer encoder layers.
    /// </summary>
    private readonly List<TransformerEncoderLayer<T>> _encoderLayers;

    /// <summary>
    /// Final layer normalization.
    /// </summary>
    private readonly LayerNormalizationLayer<T> _finalNorm;

    /// <summary>
    /// Output projection layer.
    /// </summary>
    private readonly DenseLayer<T> _outputProjection;

    /// <summary>
    /// Instance normalization mean (for RevIN).
    /// </summary>
    private Tensor<T>? _instanceMean;

    /// <summary>
    /// Instance normalization standard deviation (for RevIN).
    /// </summary>
    private Tensor<T>? _instanceStd;

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override string ModelName => "PatchTST";

    /// <inheritdoc/>
    public int PatchSize => _options.PatchSize;

    /// <inheritdoc/>
    public int Stride => _options.Stride;

    /// <inheritdoc/>
    public bool IsChannelIndependent => _options.ChannelIndependent;

    /// <summary>
    /// Gets the number of patches.
    /// </summary>
    public int NumPatches => _options.CalculateNumPatches();

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new PatchTST model using the specified options.
    /// </summary>
    /// <param name="options">Configuration options for the model.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the easiest way to create a PatchTST model:
    /// <code>
    /// var options = new PatchTSTOptions&lt;double&gt;
    /// {
    ///     SequenceLength = 96,
    ///     PredictionHorizon = 24,
    ///     NumFeatures = 7
    /// };
    /// var model = new PatchTST&lt;double&gt;(options);
    /// </code>
    /// </para>
    /// </remarks>
    public PatchTST(PatchTSTOptions<T> options)
        : base(
            CreateArchitecture(options),
            options.SequenceLength,
            options.PredictionHorizon,
            options.NumFeatures,
            options.LossFunction)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));

        // Validate options
        var errors = _options.Validate();
        if (errors.Count > 0)
            throw new ArgumentException($"Invalid PatchTST options: {string.Join(", ", errors)}");

        // Initialize random generator
        _random = _options.RandomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.RandomSeed.Value)
            : RandomHelper.CreateSecureRandom();

        // Initialize model components
        int numPatches = _options.CalculateNumPatches();

        // Patch embedding: [patch_size] -> [model_dim]
        _patchEmbedding = new DenseLayer<T>(
            _options.PatchSize,
            _options.ModelDimension);

        // Positional encoding for patches
        _positionalEncoding = CreatePositionalEncoding(numPatches, _options.ModelDimension);

        // Transformer encoder layers
        _encoderLayers = new List<TransformerEncoderLayer<T>>();
        for (int i = 0; i < _options.NumLayers; i++)
        {
            _encoderLayers.Add(new TransformerEncoderLayer<T>(
                embeddingSize: _options.ModelDimension,
                numHeads: _options.NumHeads,
                feedForwardDim: _options.FeedForwardDimension));
        }

        // Final layer normalization
        _finalNorm = new LayerNormalizationLayer<T>(_options.ModelDimension);

        // Output projection: [num_patches * model_dim] -> [prediction_horizon]
        _outputProjection = new DenseLayer<T>(
            numPatches * _options.ModelDimension,
            _options.PredictionHorizon);
    }

    /// <summary>
    /// Initializes a new PatchTST model using a pretrained ONNX model.
    /// </summary>
    /// <param name="onnxModelPath">Path to the pretrained ONNX model file.</param>
    /// <param name="options">Configuration options matching the ONNX model.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this when you have a pretrained ONNX model:
    /// <code>
    /// var options = new PatchTSTOptions&lt;float&gt;
    /// {
    ///     SequenceLength = 96,
    ///     PredictionHorizon = 24,
    ///     NumFeatures = 7
    /// };
    /// var model = new PatchTST&lt;float&gt;("patchtst_etth1.onnx", options);
    ///
    /// // Make predictions (training not supported in ONNX mode)
    /// var forecast = model.Forecast(historicalData);
    /// </code>
    /// </para>
    /// </remarks>
    public PatchTST(string onnxModelPath, PatchTSTOptions<T> options)
        : base(
            CreateArchitecture(options),
            onnxModelPath,
            options.SequenceLength,
            options.PredictionHorizon,
            options.NumFeatures)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));

        // Initialize dummy components for ONNX mode (not used but needed for property access)
        _random = RandomHelper.CreateSecureRandom();
        _patchEmbedding = new DenseLayer<T>(_options.PatchSize, _options.ModelDimension);
        _positionalEncoding = new Tensor<T>(new[] { 1 });
        _encoderLayers = new List<TransformerEncoderLayer<T>>();
        _finalNorm = new LayerNormalizationLayer<T>(_options.ModelDimension);
        _outputProjection = new DenseLayer<T>(1, 1);
    }

    /// <summary>
    /// Creates the neural network architecture for PatchTST.
    /// </summary>
    private static NeuralNetworkArchitecture<T> CreateArchitecture(PatchTSTOptions<T> options)
    {
        if (options is null)
            throw new ArgumentNullException(nameof(options));

        return new NeuralNetworkArchitecture<T>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Deep,
            inputSize: 0,
            inputHeight: options.SequenceLength,
            inputWidth: options.NumFeatures,
            inputDepth: 1,
            outputSize: options.PredictionHorizon * options.NumFeatures);
    }

    #endregion

    #region IForecastingModel Implementation

    /// <inheritdoc/>
    public Tensor<T> AutoregressiveForecast(Tensor<T> input, int steps)
    {
        if (input is null)
            throw new ArgumentNullException(nameof(input));
        if (steps < 1)
            throw new ArgumentOutOfRangeException(nameof(steps), "Steps must be at least 1.");

        // Start with the initial input
        var currentInput = input;
        var allPredictions = new List<Tensor<T>>();

        int stepsRemaining = steps;
        while (stepsRemaining > 0)
        {
            // Generate forecast for current horizon
            var forecast = Forecast(currentInput);

            // Determine how many steps to use from this forecast
            int stepsToUse = Math.Min(stepsRemaining, _predictionHorizon);
            allPredictions.Add(forecast);

            stepsRemaining -= stepsToUse;

            if (stepsRemaining > 0)
            {
                // Shift input window and append predictions
                currentInput = ShiftInputWithPredictions(currentInput, forecast, stepsToUse);
            }
        }

        // Concatenate all predictions
        return ConcatenatePredictions(allPredictions, steps);
    }

    /// <inheritdoc/>
    public Dictionary<string, T> Evaluate(Tensor<T> inputs, Tensor<T> targets)
    {
        if (inputs is null)
            throw new ArgumentNullException(nameof(inputs));
        if (targets is null)
            throw new ArgumentNullException(nameof(targets));

        var predictions = Forecast(inputs);
        var metrics = new Dictionary<string, T>();

        // Calculate MAE (Mean Absolute Error)
        T maeSum = NumOps.Zero;
        int count = 0;
        for (int i = 0; i < predictions.Length; i++)
        {
            T diff = NumOps.Subtract(predictions.Data.Span[i], targets.Data.Span[i]);
            maeSum = NumOps.Add(maeSum, NumOps.Abs(diff));
            count++;
        }
        metrics["MAE"] = NumOps.Divide(maeSum, NumOps.FromDouble(count));

        // Calculate MSE (Mean Squared Error)
        T mseSum = NumOps.Zero;
        for (int i = 0; i < predictions.Length; i++)
        {
            T diff = NumOps.Subtract(predictions.Data.Span[i], targets.Data.Span[i]);
            mseSum = NumOps.Add(mseSum, NumOps.Multiply(diff, diff));
        }
        metrics["MSE"] = NumOps.Divide(mseSum, NumOps.FromDouble(count));

        // Calculate RMSE (Root Mean Squared Error)
        metrics["RMSE"] = NumOps.Sqrt(metrics["MSE"]);

        return metrics;
    }

    /// <inheritdoc/>
    public Tensor<T> ApplyInstanceNormalization(Tensor<T> input)
    {
        if (!_options.UseInstanceNormalization)
            return input;

        return ApplyRevIN(input, normalize: true);
    }

    #endregion

    #region FinancialModelBase Implementation

    /// <inheritdoc/>
    protected override Tensor<T> ForecastNative(Tensor<T> input, double[]? quantiles)
    {
        // Apply instance normalization if enabled
        Tensor<T> normalizedInput = _options.UseInstanceNormalization
            ? ApplyRevIN(input, normalize: true)
            : input;

        Tensor<T> output;

        if (_options.ChannelIndependent)
        {
            // Process each channel independently
            output = ProcessChannelIndependent(normalizedInput);
        }
        else
        {
            // Process all channels together
            output = ProcessAllChannels(normalizedInput);
        }

        // Apply reverse instance normalization if enabled
        if (_options.UseInstanceNormalization)
        {
            output = ApplyRevIN(output, normalize: false);
        }

        return output;
    }

    /// <inheritdoc/>
    protected override void ValidateInputShape(Tensor<T> input)
    {
        // Expected shape: [batch_size, sequence_length, num_features] or [sequence_length, num_features]
        if (input.Rank < 2 || input.Rank > 3)
        {
            throw new ArgumentException(
                $"Expected input rank 2 or 3, got {input.Rank}. " +
                $"Shape should be [batch_size, {_sequenceLength}, {_numFeatures}] or [{_sequenceLength}, {_numFeatures}].");
        }

        int seqDim = input.Rank == 3 ? 1 : 0;
        int featDim = input.Rank == 3 ? 2 : 1;

        if (input.Shape[seqDim] != _sequenceLength)
        {
            throw new ArgumentException(
                $"Expected sequence length {_sequenceLength}, got {input.Shape[seqDim]}.");
        }

        if (input.Shape[featDim] != _numFeatures)
        {
            throw new ArgumentException(
                $"Expected {_numFeatures} features, got {input.Shape[featDim]}.");
        }
    }

    /// <inheritdoc/>
    protected override void TrainCore(Tensor<T> input, Tensor<T> target, Tensor<T> output)
    {
        // Calculate loss gradient
        var lossGradient = LossFunction.CalculateDerivative(output.ToVector(), target.ToVector());

        // Convert gradient to tensor shape matching output
        var gradTensor = new Tensor<T>(output.Shape, lossGradient);

        // Backpropagate through output projection
        var flatGrad = _outputProjection.Backward(gradTensor.Reshape(new[] { -1, _predictionHorizon }));

        // Backpropagate through final normalization
        var normGrad = _finalNorm.Backward(flatGrad);

        // Backpropagate through encoder layers (reverse order)
        var encoderGrad = normGrad;
        for (int i = _encoderLayers.Count - 1; i >= 0; i--)
        {
            encoderGrad = _encoderLayers[i].Backward(encoderGrad);
        }

        // Backpropagate through patch embedding
        _patchEmbedding.Backward(encoderGrad);

        // Update parameters with learning rate
        T learningRate = NumOps.FromDouble(_options.LearningRate);
        UpdateAllParameters(learningRate);
    }

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        // Layers are initialized in constructor
        ClearLayers();
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new PatchTST<T>(_options);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters)
    {
        if (parameters is null)
            throw new ArgumentNullException(nameof(parameters));

        int offset = 0;

        // Set patch embedding parameters
        var patchParams = _patchEmbedding.GetParameters();
        var newPatchParams = parameters.Slice(offset, patchParams.Length);
        _patchEmbedding.SetParameters(newPatchParams);
        offset += patchParams.Length;

        // Set encoder layer parameters
        foreach (var layer in _encoderLayers)
        {
            var layerParams = layer.GetParameters();
            var newLayerParams = parameters.Slice(offset, layerParams.Length);
            layer.SetParameters(newLayerParams);
            offset += layerParams.Length;
        }

        // Set final normalization parameters
        var normParams = _finalNorm.GetParameters();
        var newNormParams = parameters.Slice(offset, normParams.Length);
        _finalNorm.SetParameters(newNormParams);
        offset += normParams.Length;

        // Set output projection parameters
        var outputParams = _outputProjection.GetParameters();
        var newOutputParams = parameters.Slice(offset, outputParams.Length);
        _outputProjection.SetParameters(newOutputParams);
    }

    #endregion

    #region Private Methods - Forward Pass

    /// <summary>
    /// Processes input in channel-independent mode.
    /// </summary>
    private Tensor<T> ProcessChannelIndependent(Tensor<T> input)
    {
        int batchSize = input.Rank == 3 ? input.Shape[0] : 1;
        int numChannels = _numFeatures;

        // Output shape: [batch, horizon, channels]
        var outputShape = new[] { batchSize, _predictionHorizon, numChannels };
        var outputData = new T[batchSize * _predictionHorizon * numChannels];

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < numChannels; c++)
            {
                // Extract single channel sequence
                var channelSeq = ExtractChannel(input, b, c);

                // Process through transformer
                var channelOutput = ProcessSingleChannel(channelSeq);

                // Store output
                for (int h = 0; h < _predictionHorizon; h++)
                {
                    int outIdx = (b * _predictionHorizon * numChannels) + (h * numChannels) + c;
                    outputData[outIdx] = channelOutput.Data.Span[h];
                }
            }
        }

        return new Tensor<T>(outputShape, new Vector<T>(outputData));
    }

    /// <summary>
    /// Processes a single channel through the transformer.
    /// </summary>
    private Tensor<T> ProcessSingleChannel(Tensor<T> channelSeq)
    {
        // 1. Create patches from the sequence
        var patches = CreatePatches(channelSeq);

        // 2. Embed patches
        var embedded = EmbedPatches(patches);

        // 3. Add positional encoding
        var withPosition = AddPositionalEncoding(embedded);

        // 4. Apply dropout
        var dropped = ApplyDropout(withPosition);

        // 5. Pass through transformer encoder layers
        var encoded = dropped;
        for (int i = 0; i < _encoderLayers.Count; i++)
        {
            encoded = _encoderLayers[i].Forward(encoded);
        }

        // 6. Apply final layer normalization
        var normalized = _finalNorm.Forward(encoded);

        // 7. Flatten and project to prediction horizon
        var flattened = Flatten(normalized);
        var output = _outputProjection.Forward(flattened);

        return output;
    }

    /// <summary>
    /// Processes all channels together (non-CI mode).
    /// </summary>
    private Tensor<T> ProcessAllChannels(Tensor<T> input)
    {
        int batchSize = input.Rank == 3 ? input.Shape[0] : 1;

        var outputShape = new[] { batchSize, _predictionHorizon, _numFeatures };
        var outputData = new T[batchSize * _predictionHorizon * _numFeatures];

        for (int b = 0; b < batchSize; b++)
        {
            // Extract batch sample
            var sample = ExtractBatchSample(input, b);

            // Create patches for all channels combined
            var patches = CreateMultiChannelPatches(sample);

            // Process through transformer (same as single channel but larger input)
            var embedded = EmbedPatches(patches);
            var withPosition = AddPositionalEncoding(embedded);
            var dropped = ApplyDropout(withPosition);

            var encoded = dropped;
            for (int i = 0; i < _encoderLayers.Count; i++)
            {
                encoded = _encoderLayers[i].Forward(encoded);
            }

            var normalized = _finalNorm.Forward(encoded);
            var flattened = Flatten(normalized);
            var output = _outputProjection.Forward(flattened);

            // Store output
            Array.Copy(output.Data.ToArray(), 0, outputData, b * _predictionHorizon * _numFeatures,
                _predictionHorizon * _numFeatures);
        }

        return new Tensor<T>(outputShape, new Vector<T>(outputData));
    }

    /// <summary>
    /// Creates patches from a 1D sequence.
    /// </summary>
    private Tensor<T> CreatePatches(Tensor<T> sequence)
    {
        int numPatches = _options.CalculateNumPatches();
        var patchData = new T[numPatches * _options.PatchSize];

        for (int p = 0; p < numPatches; p++)
        {
            int startIdx = p * _options.Stride;
            for (int i = 0; i < _options.PatchSize; i++)
            {
                patchData[p * _options.PatchSize + i] = sequence.Data.Span[startIdx + i];
            }
        }

        return new Tensor<T>(new[] { numPatches, _options.PatchSize }, new Vector<T>(patchData));
    }

    /// <summary>
    /// Creates patches from multi-channel input.
    /// </summary>
    private Tensor<T> CreateMultiChannelPatches(Tensor<T> input)
    {
        int numPatches = _options.CalculateNumPatches();
        int patchDim = _options.PatchSize * _numFeatures;
        var patchData = new T[numPatches * patchDim];

        for (int p = 0; p < numPatches; p++)
        {
            int startIdx = p * _options.Stride;
            for (int t = 0; t < _options.PatchSize; t++)
            {
                for (int f = 0; f < _numFeatures; f++)
                {
                    int srcIdx = (startIdx + t) * _numFeatures + f;
                    int dstIdx = p * patchDim + t * _numFeatures + f;
                    patchData[dstIdx] = input.Data.Span[srcIdx];
                }
            }
        }

        return new Tensor<T>(new[] { numPatches, patchDim }, new Vector<T>(patchData));
    }

    /// <summary>
    /// Embeds patches using the patch embedding layer.
    /// </summary>
    private Tensor<T> EmbedPatches(Tensor<T> patches)
    {
        return _patchEmbedding.Forward(patches);
    }

    /// <summary>
    /// Adds positional encoding to embedded patches.
    /// </summary>
    private Tensor<T> AddPositionalEncoding(Tensor<T> embedded)
    {
        var result = new Tensor<T>(embedded.Shape);
        int numPatches = embedded.Shape[0];
        int modelDim = embedded.Shape[1];

        for (int p = 0; p < numPatches; p++)
        {
            for (int d = 0; d < modelDim; d++)
            {
                int idx = p * modelDim + d;
                result.Data.Span[idx] = NumOps.Add(
                    embedded.Data.Span[idx],
                    _positionalEncoding.Data.Span[idx]);
            }
        }

        return result;
    }

    /// <summary>
    /// Applies dropout during training.
    /// </summary>
    private Tensor<T> ApplyDropout(Tensor<T> input)
    {
        if (!IsTrainingMode || _options.Dropout <= 0)
            return input;

        var output = new Tensor<T>(input.Shape);
        T scale = NumOps.FromDouble(1.0 / (1.0 - _options.Dropout));

        for (int i = 0; i < input.Length; i++)
        {
            if (_random.NextDouble() < _options.Dropout)
            {
                output.Data.Span[i] = NumOps.Zero;
            }
            else
            {
                output.Data.Span[i] = NumOps.Multiply(input.Data.Span[i], scale);
            }
        }

        return output;
    }

    /// <summary>
    /// Flattens a tensor to 1D (preserving batch dimension if present).
    /// </summary>
    private Tensor<T> Flatten(Tensor<T> input)
    {
        return new Tensor<T>(new[] { input.Length }, new Vector<T>(input.Data.ToArray()));
    }

    #endregion

    #region Private Methods - Instance Normalization (RevIN)

    /// <summary>
    /// Applies Reversible Instance Normalization (RevIN).
    /// </summary>
    private Tensor<T> ApplyRevIN(Tensor<T> input, bool normalize)
    {
        var result = new Tensor<T>(input.Shape);
        T epsilon = NumOps.FromDouble(1e-5);

        if (normalize)
        {
            // Calculate and store mean and std, then normalize
            _instanceMean = CalculateInstanceMean(input);
            _instanceStd = CalculateInstanceStd(input, _instanceMean);

            // Normalize: (x - mean) / std
            for (int i = 0; i < input.Length; i++)
            {
                int statIdx = i % _numFeatures;
                T centered = NumOps.Subtract(input.Data.Span[i], _instanceMean.Data.Span[statIdx]);
                T stdVal = NumOps.Add(_instanceStd.Data.Span[statIdx], epsilon);
                result.Data.Span[i] = NumOps.Divide(centered, stdVal);
            }
        }
        else
        {
            // Denormalize: x * std + mean
            if (_instanceMean is null || _instanceStd is null)
                return input;

            for (int i = 0; i < input.Length; i++)
            {
                int statIdx = i % _numFeatures;
                T scaled = NumOps.Multiply(input.Data.Span[i], _instanceStd.Data.Span[statIdx]);
                result.Data.Span[i] = NumOps.Add(scaled, _instanceMean.Data.Span[statIdx]);
            }
        }

        return result;
    }

    /// <summary>
    /// Calculates instance mean for each feature.
    /// </summary>
    private Tensor<T> CalculateInstanceMean(Tensor<T> input)
    {
        var mean = new T[_numFeatures];
        int samplesPerFeature = input.Length / _numFeatures;

        for (int f = 0; f < _numFeatures; f++)
        {
            T sum = NumOps.Zero;
            for (int i = f; i < input.Length; i += _numFeatures)
            {
                sum = NumOps.Add(sum, input.Data.Span[i]);
            }
            mean[f] = NumOps.Divide(sum, NumOps.FromDouble(samplesPerFeature));
        }

        return new Tensor<T>(new[] { _numFeatures }, new Vector<T>(mean));
    }

    /// <summary>
    /// Calculates instance standard deviation for each feature.
    /// </summary>
    private Tensor<T> CalculateInstanceStd(Tensor<T> input, Tensor<T> mean)
    {
        var std = new T[_numFeatures];
        int samplesPerFeature = input.Length / _numFeatures;

        for (int f = 0; f < _numFeatures; f++)
        {
            T sumSq = NumOps.Zero;
            for (int i = f; i < input.Length; i += _numFeatures)
            {
                T diff = NumOps.Subtract(input.Data.Span[i], mean.Data.Span[f]);
                sumSq = NumOps.Add(sumSq, NumOps.Multiply(diff, diff));
            }
            T variance = NumOps.Divide(sumSq, NumOps.FromDouble(samplesPerFeature));
            std[f] = NumOps.Sqrt(variance);
        }

        return new Tensor<T>(new[] { _numFeatures }, new Vector<T>(std));
    }

    #endregion

    #region Private Methods - Utilities

    /// <summary>
    /// Creates sinusoidal positional encoding.
    /// </summary>
    private Tensor<T> CreatePositionalEncoding(int numPatches, int modelDim)
    {
        var pe = new T[numPatches * modelDim];

        for (int pos = 0; pos < numPatches; pos++)
        {
            for (int i = 0; i < modelDim; i++)
            {
                double angle = pos / Math.Pow(10000, (2.0 * (i / 2)) / modelDim);
                double value = (i % 2 == 0) ? Math.Sin(angle) : Math.Cos(angle);
                pe[pos * modelDim + i] = NumOps.FromDouble(value);
            }
        }

        return new Tensor<T>(new[] { numPatches, modelDim }, new Vector<T>(pe));
    }

    /// <summary>
    /// Extracts a single channel from the input tensor.
    /// </summary>
    private Tensor<T> ExtractChannel(Tensor<T> input, int batchIdx, int channelIdx)
    {
        var channelData = new T[_sequenceLength];

        if (input.Rank == 3)
        {
            for (int t = 0; t < _sequenceLength; t++)
            {
                int idx = (batchIdx * _sequenceLength * _numFeatures) + (t * _numFeatures) + channelIdx;
                channelData[t] = input.Data.Span[idx];
            }
        }
        else
        {
            for (int t = 0; t < _sequenceLength; t++)
            {
                int idx = (t * _numFeatures) + channelIdx;
                channelData[t] = input.Data.Span[idx];
            }
        }

        return new Tensor<T>(new[] { _sequenceLength }, new Vector<T>(channelData));
    }

    /// <summary>
    /// Extracts a single batch sample from the input tensor.
    /// </summary>
    private Tensor<T> ExtractBatchSample(Tensor<T> input, int batchIdx)
    {
        if (input.Rank == 2)
            return input;

        var sampleData = new T[_sequenceLength * _numFeatures];
        int offset = batchIdx * _sequenceLength * _numFeatures;

        Array.Copy(input.Data.ToArray(), offset, sampleData, 0, sampleData.Length);

        return new Tensor<T>(new[] { _sequenceLength, _numFeatures }, new Vector<T>(sampleData));
    }

    /// <summary>
    /// Shifts input window and appends predictions for autoregressive forecasting.
    /// </summary>
    private Tensor<T> ShiftInputWithPredictions(Tensor<T> input, Tensor<T> predictions, int stepsToShift)
    {
        var newData = new T[input.Length];
        int shiftAmount = stepsToShift * _numFeatures;

        // Shift existing data
        Array.Copy(input.Data.ToArray(), shiftAmount, newData, 0, input.Length - shiftAmount);

        // Append predictions
        Array.Copy(predictions.Data.ToArray(), 0, newData, input.Length - shiftAmount, shiftAmount);

        return new Tensor<T>(input.Shape, new Vector<T>(newData));
    }

    /// <summary>
    /// Concatenates multiple prediction tensors into a single output.
    /// </summary>
    private Tensor<T> ConcatenatePredictions(List<Tensor<T>> predictions, int totalSteps)
    {
        var outputData = new T[totalSteps * _numFeatures];
        int currentIdx = 0;

        foreach (var pred in predictions)
        {
            int stepsToCopy = Math.Min(_predictionHorizon, totalSteps - currentIdx / _numFeatures);
            int elementsToCopy = stepsToCopy * _numFeatures;

            Array.Copy(pred.Data.ToArray(), 0, outputData, currentIdx, elementsToCopy);
            currentIdx += elementsToCopy;

            if (currentIdx >= totalSteps * _numFeatures)
                break;
        }

        return new Tensor<T>(new[] { totalSteps, _numFeatures }, new Vector<T>(outputData));
    }

    /// <summary>
    /// Updates all model parameters with the given learning rate.
    /// </summary>
    private void UpdateAllParameters(T learningRate)
    {
        _patchEmbedding.UpdateParameters(learningRate);

        foreach (var layer in _encoderLayers)
        {
            layer.UpdateParameters(learningRate);
        }

        _finalNorm.UpdateParameters(learningRate);
        _outputProjection.UpdateParameters(learningRate);
    }

    #endregion

    #region Serialization

    /// <inheritdoc/>
    protected override void SerializeModelSpecificData(BinaryWriter writer)
    {
        // Write options
        writer.Write(_options.PatchSize);
        writer.Write(_options.Stride);
        writer.Write(_options.NumLayers);
        writer.Write(_options.NumHeads);
        writer.Write(_options.ModelDimension);
        writer.Write(_options.FeedForwardDimension);
        writer.Write(_options.ChannelIndependent);
        writer.Write(_options.Dropout);
        writer.Write(_options.AttentionDropout);
        writer.Write(_options.UseInstanceNormalization);
        writer.Write(_options.UsePreNorm);
        writer.Write(_options.LearningRate);

        // Write layer parameters
        WriteLayerParameters(writer, _patchEmbedding.GetParameters());
        WriteLayerParameters(writer, _outputProjection.GetParameters());

        foreach (var layer in _encoderLayers)
        {
            WriteLayerParameters(writer, layer.GetParameters());
        }
    }

    /// <inheritdoc/>
    protected override void DeserializeModelSpecificData(BinaryReader reader)
    {
        // Read options (already set in constructor, just advance reader)
        _ = reader.ReadInt32(); // PatchSize
        _ = reader.ReadInt32(); // Stride
        _ = reader.ReadInt32(); // NumLayers
        _ = reader.ReadInt32(); // NumHeads
        _ = reader.ReadInt32(); // ModelDimension
        _ = reader.ReadInt32(); // FeedForwardDimension
        _ = reader.ReadBoolean(); // ChannelIndependent
        _ = reader.ReadDouble(); // Dropout
        _ = reader.ReadDouble(); // AttentionDropout
        _ = reader.ReadBoolean(); // UseInstanceNormalization
        _ = reader.ReadBoolean(); // UsePreNorm
        _ = reader.ReadDouble(); // LearningRate

        // Read layer parameters
        _patchEmbedding.SetParameters(ReadLayerParameters(reader));
        _outputProjection.SetParameters(ReadLayerParameters(reader));

        foreach (var layer in _encoderLayers)
        {
            layer.SetParameters(ReadLayerParameters(reader));
        }
    }

    /// <summary>
    /// Writes layer parameters to binary writer.
    /// </summary>
    private void WriteLayerParameters(BinaryWriter writer, Vector<T> parameters)
    {
        writer.Write(parameters.Length);
        for (int i = 0; i < parameters.Length; i++)
        {
            writer.Write(NumOps.ToDouble(parameters[i]));
        }
    }

    /// <summary>
    /// Reads layer parameters from binary reader.
    /// </summary>
    private Vector<T> ReadLayerParameters(BinaryReader reader)
    {
        int length = reader.ReadInt32();
        var data = new T[length];
        for (int i = 0; i < length; i++)
        {
            data[i] = NumOps.FromDouble(reader.ReadDouble());
        }
        return new Vector<T>(data);
    }

    #endregion
}
