using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Validation;
using AiDotNet.Finance.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

using AiDotNet.Finance.Base;
namespace AiDotNet.Finance.Forecasting.Foundation;

/// <summary>
/// TOTEM — TOkenized Time Series EMbeddings via VQ-VAE.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// TOTEM learns discrete tokenized representations for time series via VQ-VAE,
/// enabling the use of discrete token-based methods (like LLMs) on continuous time series data.
/// It uses an encoder-decoder architecture with vector quantization bottleneck.
/// </para>
/// <para>
/// <b>Reference:</b> Talukder et al., "TOTEM: TOkenized Time Series EMbeddings for General Time Series Analysis", 2024.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Finance)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Autoencoder)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Forecasting)]
[ModelTask(ModelTask.Embedding)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
public class TOTEM<T> : TimeSeriesFoundationModelBase<T>
{
    #region Fields

    private readonly bool _useNativeMode;
    private ILayer<T>? _encoder;
    private readonly List<ILayer<T>> _transformerLayers = [];
    private ILayer<T>? _quantizationProjection;
    private ILayer<T>? _decoder;
    private ILayer<T>? _forecastHead;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly TOTEMOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private int _contextLength;
    private int _forecastHorizon;
    private int _hiddenDimension;
    private int _numLayers;
    private int _numHeads;
    private int _codebookSize;
    private int _codebookDimension;
    private int _numCodebooks;
    private double _dropout;
    private double _commitmentWeight;

    // VQ codebook: [numCodebooks x codebookSize x codebookDimension]
    private Tensor<T>? _codebooks;
    private T _lastCommitmentLoss;

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override int SequenceLength => _contextLength;
    /// <inheritdoc/>
    public override int PredictionHorizon => _forecastHorizon;
    /// <inheritdoc/>
    public override int NumFeatures => 1;
    /// <inheritdoc/>
    public override int PatchSize => 1;
    /// <inheritdoc/>
    public override int Stride => 1;
    /// <inheritdoc/>
    public override bool IsChannelIndependent => true;
    /// <inheritdoc/>
    public override bool UseNativeMode => _useNativeMode;
    /// <inheritdoc/>
    public override FoundationModelSize ModelSize => FoundationModelSize.Base;
    /// <inheritdoc/>
    public override int MaxContextLength => _contextLength;
    /// <inheritdoc/>
    public override int MaxPredictionHorizon => _forecastHorizon;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a TOTEM model using a pretrained ONNX model.
    /// </summary>
    public TOTEM(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        TOTEMOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        options ??= new TOTEMOptions<T>();
        _options = options;
        Options = _options;

        _useNativeMode = false;
        OnnxModelPath = onnxModelPath;
        OnnxSession = new InferenceSession(onnxModelPath);

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _lastCommitmentLoss = NumOps.Zero;

        CopyOptionsToFields(options);
    }

    /// <summary>
    /// Creates a TOTEM model in native mode for training or fine-tuning.
    /// </summary>
    public TOTEM(
        NeuralNetworkArchitecture<T> architecture,
        TOTEMOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        options ??= new TOTEMOptions<T>();
        _options = options;
        Options = _options;

        _useNativeMode = true;
        OnnxSession = null;
        OnnxModelPath = null;

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _lastCommitmentLoss = NumOps.Zero;

        CopyOptionsToFields(options);
        InitializeLayers();
    }

    private void CopyOptionsToFields(TOTEMOptions<T> options)
    {
        Guard.Positive(options.ContextLength, nameof(options.ContextLength));
        Guard.Positive(options.ForecastHorizon, nameof(options.ForecastHorizon));
        Guard.Positive(options.HiddenDimension, nameof(options.HiddenDimension));
        Guard.Positive(options.NumLayers, nameof(options.NumLayers));
        Guard.Positive(options.NumHeads, nameof(options.NumHeads));
        Guard.Positive(options.CodebookSize, nameof(options.CodebookSize));
        Guard.Positive(options.CodebookDimension, nameof(options.CodebookDimension));
        Guard.Positive(options.NumCodebooks, nameof(options.NumCodebooks));

        _contextLength = options.ContextLength;
        _forecastHorizon = options.ForecastHorizon;
        _hiddenDimension = options.HiddenDimension;
        _numLayers = options.NumLayers;
        _numHeads = options.NumHeads;
        _codebookSize = options.CodebookSize;
        _codebookDimension = options.CodebookDimension;
        _numCodebooks = options.NumCodebooks;
        _dropout = options.DropoutRate;
        _commitmentWeight = options.CommitmentWeight;
        InitializeCodebooks();
    }

    /// <summary>
    /// Initializes the VQ codebook embeddings with random values from N(0, 1/dim).
    /// </summary>
    private void InitializeCodebooks()
    {
        _codebooks = new Tensor<T>(new[] { _numCodebooks, _codebookSize, _codebookDimension });
        var rand = RandomHelper.CreateSecureRandom();
        T scale = NumOps.Divide(NumOps.One, NumOps.FromDouble(Math.Sqrt(_codebookDimension)));
        for (int c = 0; c < _numCodebooks; c++)
            for (int k = 0; k < _codebookSize; k++)
                for (int d = 0; d < _codebookDimension; d++)
                {
                    double u1 = 1.0 - rand.NextDouble();
                    double u2 = 1.0 - rand.NextDouble();
                    T sample = NumOps.FromDouble(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
                    int idx = c * _codebookSize * _codebookDimension + k * _codebookDimension + d;
                    _codebooks.Data.Span[idx] = NumOps.Multiply(scale, sample);
                }
    }

    /// <summary>Gets a codebook value at the given indices.</summary>
    private T GetCodebookValue(int codebook, int entry, int dim)
    {
        int idx = codebook * _codebookSize * _codebookDimension + entry * _codebookDimension + dim;
        return _codebooks![idx];
    }

    /// <summary>Sets a codebook value at the given indices.</summary>
    private void SetCodebookValue(int codebook, int entry, int dim, T value)
    {
        int idx = codebook * _codebookSize * _codebookDimension + entry * _codebookDimension + dim;
        _codebooks!.Data.Span[idx] = value;
    }

    #endregion

    #region Initialization

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ExtractLayerReferences();
        }
        else if (_useNativeMode)
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultTOTEMLayers(
                Architecture, _contextLength, _forecastHorizon, _hiddenDimension,
                _numLayers, _numHeads, _codebookDimension, _dropout));
            ExtractLayerReferences();
        }
    }

    private void ExtractLayerReferences()
    {
        int idx = 0;

        if (idx < Layers.Count)
            _encoder = Layers[idx++];

        _transformerLayers.Clear();
        int layersPerBlock = _dropout > 0 ? 9 : 7;
        int totalTransformerLayers = _numLayers * layersPerBlock;

        for (int i = 0; i < totalTransformerLayers && idx < Layers.Count; i++)
            _transformerLayers.Add(Layers[idx++]);

        if (idx < Layers.Count)
            _quantizationProjection = Layers[idx++];

        if (idx < Layers.Count)
            _decoder = Layers[idx++];

        if (idx < Layers.Count)
            _forecastHead = Layers[idx++];
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <inheritdoc/>
    public override bool SupportsTraining => _useNativeMode;

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return _useNativeMode ? ForwardNative(input) : ForecastOnnx(input);
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> target)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is only supported in native mode.");

        SetTrainingMode(true);
        try
        {
            // VQ-VAE training: reconstruction loss + commitment loss
            var output = ForwardNative(input);
            T reconstructionLoss = _lossFunction.CalculateLoss(output.ToVector(), target.ToVector());

            // Total loss = reconstruction + commitment (commitment computed during VectorQuantize)
            // Note: commitment loss gradient flows via the straight-through estimator (STE) in
            // VectorQuantize — the encoder receives reconstruction gradients, and codebook entries
            // are updated via exponential moving average during the forward pass.
            LastLoss = NumOps.Add(reconstructionLoss, _lastCommitmentLoss);

            var gradient = _lossFunction.CalculateDerivative(output.ToVector(), target.ToVector());
            BackwardNative(Tensor<T>.FromVector(gradient, output.Shape));

            _optimizer.UpdateParameters(Layers);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> gradients)
    {
        // Parameters are updated through the optimizer in Train()
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "TOTEM" },
                { "ContextLength", _contextLength },
                { "ForecastHorizon", _forecastHorizon },
                { "HiddenDimension", _hiddenDimension },
                { "NumLayers", _numLayers },
                { "CodebookSize", _codebookSize },
                { "NumCodebooks", _numCodebooks },
                { "UseNativeMode", _useNativeMode },
                { "ParameterCount", GetParameterCount() }
            },
            ModelData = _useNativeMode ? this.Serialize() : Array.Empty<byte>()
        };
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new TOTEM<T>(Architecture, new TOTEMOptions<T>
        {
            ContextLength = _contextLength,
            ForecastHorizon = _forecastHorizon,
            HiddenDimension = _hiddenDimension,
            NumLayers = _numLayers,
            NumHeads = _numHeads,
            CodebookSize = _codebookSize,
            CodebookDimension = _codebookDimension,
            NumCodebooks = _numCodebooks,
            DropoutRate = _dropout,
            CommitmentWeight = _commitmentWeight
        });
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_contextLength);
        writer.Write(_forecastHorizon);
        writer.Write(_hiddenDimension);
        writer.Write(_numLayers);
        writer.Write(_numHeads);
        writer.Write(_codebookSize);
        writer.Write(_codebookDimension);
        writer.Write(_numCodebooks);
        writer.Write(_dropout);
        writer.Write(_commitmentWeight);

        // Serialize codebook embeddings
        if (_codebooks is not null)
        {
            writer.Write(true);
            for (int c = 0; c < _numCodebooks; c++)
                for (int k = 0; k < _codebookSize; k++)
                    for (int d = 0; d < _codebookDimension; d++)
                        writer.Write(NumOps.ToDouble(GetCodebookValue(c, k, d)));
        }
        else
        {
            writer.Write(false);
        }
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _contextLength = reader.ReadInt32();
        _forecastHorizon = reader.ReadInt32();
        _hiddenDimension = reader.ReadInt32();
        _numLayers = reader.ReadInt32();
        _numHeads = reader.ReadInt32();
        _codebookSize = reader.ReadInt32();
        _codebookDimension = reader.ReadInt32();
        _numCodebooks = reader.ReadInt32();
        _dropout = reader.ReadDouble();
        _commitmentWeight = reader.ReadDouble();

        // Deserialize codebook embeddings
        bool hasCodebooks = reader.ReadBoolean();
        if (hasCodebooks)
        {
            _codebooks = new Tensor<T>(new[] { _numCodebooks, _codebookSize, _codebookDimension });
            for (int c = 0; c < _numCodebooks; c++)
                for (int k = 0; k < _codebookSize; k++)
                    for (int d = 0; d < _codebookDimension; d++)
                        SetCodebookValue(c, k, d, NumOps.FromDouble(reader.ReadDouble()));
        }
        else
        {
            InitializeCodebooks();
        }
    }

    #endregion

    #region IForecastingModel Implementation

    /// <inheritdoc/>
    public override Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null)
    {
        if (quantiles is not null && quantiles.Length > 0)
            throw new NotSupportedException("TOTEM does not support quantile forecasting. Pass null for point forecasts.");

        return _useNativeMode ? ForwardNative(historicalData) : ForecastOnnx(historicalData);
    }

    /// <inheritdoc/>
    public override Tensor<T> AutoregressiveForecast(Tensor<T> input, int steps)
    {
        var predictions = new List<Tensor<T>>();
        var currentInput = input;
        int stepsRemaining = steps;

        while (stepsRemaining > 0)
        {
            var prediction = Forecast(currentInput, null);
            predictions.Add(prediction);
            int stepsUsed = Math.Min(_forecastHorizon, stepsRemaining);
            stepsRemaining -= stepsUsed;

            if (stepsRemaining > 0)
                currentInput = ShiftInputWithPredictions(currentInput, prediction, stepsUsed);
        }

        return ConcatenatePredictions(predictions, steps);
    }

    /// <inheritdoc/>
    public override Dictionary<string, T> Evaluate(Tensor<T> predictions, Tensor<T> actuals)
    {
        var metrics = new Dictionary<string, T>();
        T mse = NumOps.Zero;
        T mae = NumOps.Zero;
        int count = 0;

        for (int i = 0; i < predictions.Length && i < actuals.Length; i++)
        {
            var diff = NumOps.Subtract(predictions[i], actuals[i]);
            mse = NumOps.Add(mse, NumOps.Multiply(diff, diff));
            mae = NumOps.Add(mae, NumOps.Abs(diff));
            count++;
        }

        if (count > 0)
        {
            mse = NumOps.Divide(mse, NumOps.FromDouble(count));
            mae = NumOps.Divide(mae, NumOps.FromDouble(count));
        }

        metrics["MSE"] = mse;
        metrics["MAE"] = mae;
        metrics["RMSE"] = NumOps.Sqrt(mse);
        return metrics;
    }

    /// <inheritdoc/>
    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input)
    {
        int batchSize = input.Shape[0];
        int seqLen = input.Shape.Length > 1 ? input.Shape[1] : input.Length;
        var result = new Tensor<T>(input.Shape);

        for (int b = 0; b < batchSize; b++)
        {
            T mean = NumOps.Zero;
            for (int t = 0; t < seqLen; t++)
            {
                int idx = b * seqLen + t;
                if (idx < input.Length)
                    mean = NumOps.Add(mean, input[idx]);
            }
            mean = NumOps.Divide(mean, NumOps.FromDouble(seqLen));

            T variance = NumOps.Zero;
            for (int t = 0; t < seqLen; t++)
            {
                int idx = b * seqLen + t;
                if (idx < input.Length)
                {
                    var diff = NumOps.Subtract(input[idx], mean);
                    variance = NumOps.Add(variance, NumOps.Multiply(diff, diff));
                }
            }
            variance = NumOps.Divide(variance, NumOps.FromDouble(seqLen));
            T std = NumOps.Sqrt(NumOps.Add(variance, NumOps.FromDouble(1e-5)));

            for (int t = 0; t < seqLen; t++)
            {
                int idx = b * seqLen + t;
                if (idx < input.Length && idx < result.Length)
                    result.Data.Span[idx] = NumOps.Divide(NumOps.Subtract(input[idx], mean), std);
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public override Dictionary<string, T> GetFinancialMetrics()
    {
        T lastLoss = LastLoss is not null ? LastLoss : NumOps.Zero;
        return new Dictionary<string, T>
        {
            ["ContextLength"] = NumOps.FromDouble(_contextLength),
            ["ForecastHorizon"] = NumOps.FromDouble(_forecastHorizon),
            ["HiddenDimension"] = NumOps.FromDouble(_hiddenDimension),
            ["CodebookSize"] = NumOps.FromDouble(_codebookSize),
            ["NumCodebooks"] = NumOps.FromDouble(_numCodebooks),
            ["LastLoss"] = lastLoss
        };
    }

    #endregion

    #region Forward/Backward Pass

    /// <summary>
    /// VQ-VAE forward pass: encode → transformer → project to codebook dim →
    /// vector quantize (nearest neighbor lookup) → decode → forecast.
    /// The quantization bottleneck forces discrete token representations.
    /// </summary>
    private Tensor<T> ForwardNative(Tensor<T> input)
    {
        var normalized = ApplyInstanceNormalization(input);
        var current = normalized;

        bool addedBatchDim = false;
        if (current.Rank == 1)
        {
            current = current.Reshape(new[] { 1, current.Length });
            addedBatchDim = true;
        }

        // Encoder
        if (_encoder is not null)
            current = _encoder.Forward(current);

        // Transformer layers
        foreach (var layer in _transformerLayers)
            current = layer.Forward(current);

        // Project to codebook dimension
        if (_quantizationProjection is not null)
            current = _quantizationProjection.Forward(current);

        // Vector Quantization: find nearest codebook entry for each position
        var quantized = VectorQuantize(current);

        // Straight-through estimator: use quantized for forward, but pass gradient through
        // In this simplified version, we just use the quantized output
        if (_decoder is not null)
            quantized = _decoder.Forward(quantized);

        if (_forecastHead is not null)
            quantized = _forecastHead.Forward(quantized);

        if (addedBatchDim && quantized.Rank == 2 && quantized.Shape[0] == 1)
            quantized = quantized.Reshape(new[] { quantized.Shape[1] });

        return quantized;
    }

    /// <summary>
    /// Vector quantization: for each codebook, find nearest embedding to each input vector.
    /// Uses product quantization when numCodebooks > 1 (splits features across codebooks).
    /// Also computes commitment loss: ||z_e - sg(e_k)||^2.
    /// </summary>
    private Tensor<T> VectorQuantize(Tensor<T> encoderOutput)
    {
        if (_codebooks is null) InitializeCodebooks();

        int totalLen = encoderOutput.Length;
        var quantized = new Tensor<T>(encoderOutput.Shape);
        T commitmentLoss = NumOps.Zero;

        // Product quantization: split encoded vector across codebooks
        int dimPerCodebook = Math.Max(1, _codebookDimension);
        int numPositions = Math.Max(1, totalLen / Math.Max(1, dimPerCodebook * _numCodebooks));

        for (int pos = 0; pos < numPositions; pos++)
        {
            for (int c = 0; c < _numCodebooks; c++)
            {
                int startIdx = pos * dimPerCodebook * _numCodebooks + c * dimPerCodebook;
                if (startIdx >= totalLen) break;

                int effectiveDim = Math.Min(dimPerCodebook, totalLen - startIdx);

                // Find nearest codebook entry
                int bestIdx = 0;
                T bestDist = NumOps.MaxValue;
                for (int k = 0; k < _codebookSize; k++)
                {
                    T dist = NumOps.Zero;
                    for (int d = 0; d < effectiveDim; d++)
                    {
                        int idx = startIdx + d;
                        if (idx >= totalLen) break;
                        T diff = NumOps.Subtract(encoderOutput[idx], GetCodebookValue(c, k, d % _codebookDimension));
                        dist = NumOps.Add(dist, NumOps.Multiply(diff, diff));
                    }
                    if (NumOps.LessThan(dist, bestDist)) { bestDist = dist; bestIdx = k; }
                }

                // Replace encoder output with nearest codebook entry (straight-through)
                for (int d = 0; d < effectiveDim; d++)
                {
                    int idx = startIdx + d;
                    if (idx >= totalLen) break;
                    T codebookVal = GetCodebookValue(c, bestIdx, d % _codebookDimension);

                    // Straight-through: quantized = encoder_output + sg(codebook - encoder_output)
                    // This means forward uses codebook values, backward passes gradient through encoder
                    quantized.Data.Span[idx] = codebookVal;

                    // Commitment loss: ||z_e - sg(e_k)||^2
                    T diff = NumOps.Subtract(encoderOutput[idx], codebookVal);
                    commitmentLoss = NumOps.Add(commitmentLoss, NumOps.Multiply(diff, diff));
                }

                // EMA codebook update (during training): move codebook entry toward encoder output
                if (IsTrainingMode)
                {
                    T emaDecay = NumOps.FromDouble(0.99);
                    T oneMinusDecay = NumOps.FromDouble(0.01);
                    for (int d = 0; d < effectiveDim && d < _codebookDimension; d++)
                    {
                        int idx = startIdx + d;
                        if (idx >= totalLen) break;
                        T currentVal = GetCodebookValue(c, bestIdx, d);
                        T newVal = NumOps.Add(NumOps.Multiply(emaDecay, currentVal), NumOps.Multiply(oneMinusDecay, encoderOutput[idx]));
                        SetCodebookValue(c, bestIdx, d, newVal);
                    }
                }
            }
        }

        // Copy through any remaining elements that don't fit product quantization
        int quantizedElements = numPositions * _numCodebooks * dimPerCodebook;
        for (int i = quantizedElements; i < totalLen; i++)
            quantized.Data.Span[i] = encoderOutput[i];

        T commitWeightT = NumOps.FromDouble(_commitmentWeight);
        T invTotalLen = NumOps.Divide(NumOps.One, NumOps.FromDouble(Math.Max(1, totalLen)));
        _lastCommitmentLoss = NumOps.Multiply(NumOps.Multiply(commitmentLoss, commitWeightT), invTotalLen);
        return quantized;
    }

    private Tensor<T> BackwardNative(Tensor<T> gradOutput)
    {
        var current = gradOutput;

        bool addedBatchDim = false;
        if (current.Rank == 1)
        {
            current = current.Reshape(new[] { 1, current.Length });
            addedBatchDim = true;
        }

        if (_forecastHead is not null)
            current = _forecastHead.Backward(current);

        if (_decoder is not null)
            current = _decoder.Backward(current);

        if (_quantizationProjection is not null)
            current = _quantizationProjection.Backward(current);

        for (int i = _transformerLayers.Count - 1; i >= 0; i--)
            current = _transformerLayers[i].Backward(current);

        if (_encoder is not null)
            current = _encoder.Backward(current);

        if (addedBatchDim && current.Rank == 2 && current.Shape[0] == 1)
            current = current.Reshape(new[] { current.Shape[1] });

        return current;
    }

    protected override Tensor<T> ForecastOnnx(Tensor<T> input)
    {
        if (OnnxSession == null)
            throw new InvalidOperationException("ONNX session is not initialized.");

        int batchSize = input.Rank > 1 ? input.Shape[0] : 1;
        int seqLen = input.Rank > 1 ? input.Shape[1] : input.Length;
        int features = input.Rank > 2 ? input.Shape[2] : 1;

        var inputData = new float[batchSize * seqLen * features];
        for (int i = 0; i < input.Length && i < inputData.Length; i++)
            inputData[i] = (float)NumOps.ToDouble(input[i]);

        var inputTensor = new OnnxTensors.DenseTensor<float>(
            inputData, new[] { batchSize, seqLen, features });

        string inputName = OnnxSession.InputMetadata.Keys.FirstOrDefault() ?? "input";
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
        };

        using var results = OnnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        var outputShape = outputTensor.Dimensions.ToArray();
        var output = new Tensor<T>(outputShape);

        int totalElements = 1;
        foreach (var dim in outputShape) totalElements *= dim;

        for (int i = 0; i < totalElements && i < output.Length; i++)
            output.Data.Span[i] = NumOps.FromDouble(outputTensor.GetValue(i));

        return output;
    }

    #endregion

    #region Parameter Estimation

    private new int GetParameterCount()
    {
        long total = (long)_contextLength * _hiddenDimension + _hiddenDimension;
        long perLayer = 4L * _hiddenDimension * _hiddenDimension + 4 * _hiddenDimension;
        perLayer += 2L * _hiddenDimension * (_hiddenDimension * 4) + _hiddenDimension + (_hiddenDimension * 4);
        perLayer += 4L * _hiddenDimension;
        total += perLayer * _numLayers;
        total += (long)_codebookSize * _codebookDimension * _numCodebooks;
        total += (long)_hiddenDimension * _codebookDimension + _codebookDimension;
        total += (long)_codebookDimension * _hiddenDimension + _hiddenDimension;
        total += (long)_hiddenDimension * _forecastHorizon + _forecastHorizon;
        return (int)Math.Min(total, int.MaxValue);
    }

    #endregion
}
