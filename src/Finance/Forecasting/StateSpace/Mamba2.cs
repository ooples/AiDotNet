using System.IO;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Layers.SSM;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

using AiDotNet.Finance.Base;
namespace AiDotNet.Finance.Forecasting.StateSpace;

/// <summary>
/// Mamba-2 (State Space Duality) implementation for time series forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// Mamba-2 improves upon Mamba by discovering the connection between selective state space models
/// and structured masked attention (State Space Duality). This enables a more efficient SSD algorithm
/// using matrix multiplications and multi-head structure, achieving 2-8x faster training.
/// </para>
/// <para><b>For Beginners:</b> Mamba-2 is the next generation of the Mamba architecture:
///
/// <b>Key Improvements over Mamba-1:</b>
/// 1. <b>SSD Algorithm:</b> Uses matrix multiply instead of associative scan — much faster on GPUs
/// 2. <b>Multi-head Structure:</b> Like multi-head attention, enabling better capacity per parameter
/// 3. <b>Chunk-wise Processing:</b> Processes sequences in chunks for better hardware utilization
/// 4. <b>2-8x Faster Training:</b> Due to better hardware mapping
///
/// <b>For Time Series:</b>
/// - Efficient handling of long historical windows
/// - Multi-head captures different temporal patterns simultaneously
/// - Linear complexity enables real-time forecasting on long sequences
/// </para>
/// <para>
/// <b>Reference:</b> Dao and Gu, "Transformers are SSMs: Generalized Models and Efficient Algorithms
/// Through Structured State Space Duality", 2024.
/// </para>
/// </remarks>
public class Mamba2<T> : ForecastingModelBase<T>
{
    #region Execution Mode
    private bool _useNativeMode;
    #endregion

    #region Native Mode Fields
    private DenseLayer<T>? _inputEmbedding;
    private List<Mamba2Block<T>>? _mamba2Blocks;
    private List<DenseLayer<T>>? _outputProjectionLayers;
    #endregion

    #region Shared Fields
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly Mamba2Options<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private int _contextLength;
    private int _lastForwardSeqLen;
    private int _lastForwardBatchSize;
    private int _forecastHorizon;
    private int _modelDimension;
    private int _stateDimension;
    private int _numHeads;
    private int _expandFactor;
    private int _convKernelSize;
    private int _chunkSize;
    private int _numLayers;
    private double _dropout;
    private int _numFeatures;
    #endregion

    #region IForecastingModel Properties
    /// <inheritdoc/>
    public override int SequenceLength => _contextLength;
    /// <inheritdoc/>
    public override int PredictionHorizon => _forecastHorizon;
    /// <inheritdoc/>
    public override int NumFeatures => _numFeatures;
    /// <inheritdoc/>
    public override int PatchSize => 1;
    /// <inheritdoc/>
    public override int Stride => 1;
    /// <inheritdoc/>
    public override bool IsChannelIndependent => true;
    /// <inheritdoc/>
    public override bool UseNativeMode => _useNativeMode;

    /// <summary>Gets the state dimension per head.</summary>
    public int StateDimension => _stateDimension;
    /// <summary>Gets the number of heads for multi-head SSD.</summary>
    public int NumHeads => _numHeads;
    /// <summary>Gets the chunk size for SSD computation.</summary>
    public int ChunkSize => _chunkSize;
    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance using an ONNX pretrained model.
    /// </summary>
    public Mamba2(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        Mamba2Options<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        options ??= new Mamba2Options<T>();
        _options = options;
        Options = _options;
        _useNativeMode = false;
        OnnxModelPath = onnxModelPath;
        OnnxSession = new InferenceSession(onnxModelPath);
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        ApplyOptions(options);
        _numFeatures = 1;
        InitializeLayers();
    }

    /// <summary>
    /// Initializes a new instance in native mode for training.
    /// </summary>
    public Mamba2(
        NeuralNetworkArchitecture<T> architecture,
        Mamba2Options<T>? options = null,
        int numFeatures = 1,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        options ??= new Mamba2Options<T>();
        _options = options;
        Options = _options;
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        ApplyOptions(options);
        _numFeatures = numFeatures;
        InitializeLayers();
    }

    private void ApplyOptions(Mamba2Options<T> options)
    {
        _contextLength = options.ContextLength;
        _forecastHorizon = options.ForecastHorizon;
        _modelDimension = options.ModelDimension;
        _stateDimension = options.StateDimension;
        _numHeads = options.NumHeads;
        _expandFactor = options.ExpandFactor;
        _convKernelSize = options.ConvKernelSize;
        _chunkSize = options.ChunkSize;
        _numLayers = options.NumLayers;
        _dropout = options.DropoutRate;
    }

    #endregion

    #region Initialization

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
            ExtractLayerReferences();
        }
        else if (_useNativeMode)
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultMamba2Layers(
                Architecture, _contextLength, _forecastHorizon, _numFeatures,
                _modelDimension, _stateDimension, _numHeads, _expandFactor, _numLayers, _dropout,
                _convKernelSize, _chunkSize));
            ExtractLayerReferences();
        }
    }

    private void ExtractLayerReferences()
    {
        _inputEmbedding = Layers.OfType<DenseLayer<T>>().FirstOrDefault();
        _mamba2Blocks = Layers.OfType<Mamba2Block<T>>().ToList();
        _outputProjectionLayers = Layers.OfType<DenseLayer<T>>().Skip(1).ToList();
    }

    /// <inheritdoc/>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);

        if (layers.OfType<Mamba2Block<T>>().Count() < 1)
            throw new ArgumentException("Mamba2 requires at least one Mamba2Block layer.");

        if (layers.OfType<DenseLayer<T>>().Count() < 2)
            throw new ArgumentException("Mamba2 requires at least input embedding and output projection DenseLayer layers.");
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <inheritdoc/>
    public override bool SupportsTraining => _useNativeMode;

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return _useNativeMode ? Forward(input) : ForecastOnnx(input);
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> target)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is only supported in native mode.");

        SetTrainingMode(true);
        var output = Forward(input);

        var outputVec = output.ToVector();
        var targetVec = target.ToVector();
        int minLength = Math.Min(outputVec.Length, targetVec.Length);

        var matchedOutput = new T[minLength];
        var matchedTarget = new T[minLength];
        for (int i = 0; i < minLength; i++)
        {
            matchedOutput[i] = outputVec[i];
            matchedTarget[i] = targetVec[i];
        }

        var matchedOutputVec = new Vector<T>(matchedOutput);
        var matchedTargetVec = new Vector<T>(matchedTarget);
        LastLoss = _lossFunction.CalculateLoss(matchedOutputVec, matchedTargetVec);

        var gradient = _lossFunction.CalculateDerivative(matchedOutputVec, matchedTargetVec);
        var fullGradient = new T[output.Length];
        for (int i = 0; i < Math.Min(gradient.Length, fullGradient.Length); i++)
            fullGradient[i] = gradient[i];

        var gradTensor = new Tensor<T>(new[] { 1, output.Length }, new Vector<T>(fullGradient));
        Backward(gradTensor);
        _optimizer.UpdateParameters(Layers);
        SetTrainingMode(false);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> gradients) { }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "Mamba2" },
                { "ContextLength", _contextLength },
                { "ForecastHorizon", _forecastHorizon },
                { "ModelDimension", _modelDimension },
                { "StateDimension", _stateDimension },
                { "NumHeads", _numHeads },
                { "ChunkSize", _chunkSize },
                { "NumLayers", _numLayers },
                { "UseNativeMode", _useNativeMode },
                { "ParameterCount", GetParameterCount() }
            },
            ModelData = _useNativeMode ? this.Serialize() : Array.Empty<byte>()
        };
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new Mamba2<T>(Architecture, new Mamba2Options<T>(_options), _numFeatures);
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_contextLength);
        writer.Write(_forecastHorizon);
        writer.Write(_modelDimension);
        writer.Write(_stateDimension);
        writer.Write(_numHeads);
        writer.Write(_expandFactor);
        writer.Write(_convKernelSize);
        writer.Write(_chunkSize);
        writer.Write(_numLayers);
        writer.Write(_dropout);
        writer.Write(_numFeatures);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _contextLength = reader.ReadInt32();
        _forecastHorizon = reader.ReadInt32();
        _modelDimension = reader.ReadInt32();
        _stateDimension = reader.ReadInt32();
        _numHeads = reader.ReadInt32();
        _expandFactor = reader.ReadInt32();
        _convKernelSize = reader.ReadInt32();
        _chunkSize = reader.ReadInt32();
        _numLayers = reader.ReadInt32();
        _dropout = reader.ReadDouble();
        _numFeatures = reader.ReadInt32();
    }

    #endregion

    #region IForecastingModel Implementation

    /// <inheritdoc/>
    public override Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null)
    {
        var output = _useNativeMode ? Forward(historicalData) : ForecastOnnx(historicalData);
        if (quantiles is not null && quantiles.Length > 0)
            return GenerateQuantilePredictions(historicalData, quantiles);
        return output;
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
        T mse = NumOps.Zero, mae = NumOps.Zero;
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
    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input) => input;

    /// <inheritdoc/>
    public override Dictionary<string, T> GetFinancialMetrics()
    {
        T lastLoss = LastLoss is not null ? LastLoss : NumOps.Zero;
        return new Dictionary<string, T>
        {
            ["ContextLength"] = NumOps.FromDouble(_contextLength),
            ["ForecastHorizon"] = NumOps.FromDouble(_forecastHorizon),
            ["ModelDimension"] = NumOps.FromDouble(_modelDimension),
            ["StateDimension"] = NumOps.FromDouble(_stateDimension),
            ["NumHeads"] = NumOps.FromDouble(_numHeads),
            ["NumLayers"] = NumOps.FromDouble(_numLayers),
            ["LastLoss"] = lastLoss
        };
    }

    #endregion

    #region Forward/Backward Pass

    private Tensor<T> Forward(Tensor<T> input)
    {
        var current = NormalizeInputTo3D(input);
        int batchSize = current.Shape[0];
        int seqLen = current.Shape[1];

        if (seqLen != _contextLength)
            throw new ArgumentException(
                $"Input sequence length ({seqLen}) does not match expected context length ({_contextLength}).");
        int featureDim = current.Shape[2];
        if (featureDim != _numFeatures)
            throw new ArgumentException(
                $"Input feature dimension ({featureDim}) does not match expected numFeatures ({_numFeatures}).");

        _lastForwardSeqLen = seqLen;
        _lastForwardBatchSize = batchSize;

        if (_inputEmbedding is not null)
        {
            current = current.Reshape(new[] { batchSize * seqLen, _numFeatures });
            current = _inputEmbedding.Forward(current);
            current = current.Reshape(new[] { batchSize, seqLen, _modelDimension });
        }

        if (_mamba2Blocks is not null)
        {
            foreach (var block in _mamba2Blocks)
                current = block.Forward(current);
        }

        current = current.Reshape(new[] { batchSize, seqLen * _modelDimension });

        if (_outputProjectionLayers is not null)
        {
            foreach (var layer in _outputProjectionLayers)
                current = layer.Forward(current);
        }

        return current;
    }

    private Tensor<T> NormalizeInputTo3D(Tensor<T> input)
    {
        if (input.Rank == 3) return input;
        if (input.Rank == 2) return input.Reshape(new[] { 1, input.Shape[0], input.Shape[1] });
        if (input.Rank == 1)
        {
            int seqLen, features;
            if (_numFeatures > 1 && input.Length % _numFeatures == 0) { seqLen = input.Length / _numFeatures; features = _numFeatures; }
            else if (_contextLength > 0 && input.Length % _contextLength == 0) { seqLen = _contextLength; features = input.Length / _contextLength; }
            else { seqLen = input.Length; features = 1; }
            return input.Reshape(new[] { 1, seqLen, features });
        }
        int batchDims = 1;
        for (int i = 0; i < input.Rank - 2; i++) batchDims *= input.Shape[i];
        return input.Reshape(new[] { batchDims, input.Shape[input.Rank - 2], input.Shape[input.Rank - 1] });
    }

    private Tensor<T> Backward(Tensor<T> outputGradient)
    {
        var current = outputGradient;
        if (current.Rank == 1) current = current.Reshape(new[] { 1, current.Length });
        int batchSize = _lastForwardBatchSize;

        if (_outputProjectionLayers is not null)
        {
            for (int i = _outputProjectionLayers.Count - 1; i >= 0; i--)
                current = _outputProjectionLayers[i].Backward(current);
        }

        current = current.Reshape(new[] { batchSize, _lastForwardSeqLen, _modelDimension });

        if (_mamba2Blocks is not null)
        {
            for (int i = _mamba2Blocks.Count - 1; i >= 0; i--)
                current = _mamba2Blocks[i].Backward(current);
        }

        int seqLen = current.Shape[1];
        current = current.Reshape(new[] { batchSize * seqLen, _modelDimension });
        if (_inputEmbedding is not null)
            current = _inputEmbedding.Backward(current);
        current = current.Reshape(new[] { batchSize, seqLen, _numFeatures });
        return current;
    }

    /// <inheritdoc/>
    protected override Tensor<T> ForecastOnnx(Tensor<T> input)
    {
        if (OnnxSession is null) throw new InvalidOperationException("ONNX session not initialized.");

        var inputData = new float[input.Length];
        for (int i = 0; i < input.Length; i++)
            inputData[i] = Convert.ToSingle(NumOps.ToDouble(input.Data.Span[i]));

        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input.Shape);
        string inputName = OnnxSession.InputMetadata.Keys.First();
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, onnxInput) };

        using var results = OnnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();
        var outputData = new T[outputTensor.Length];
        for (int i = 0; i < outputTensor.Length; i++)
            outputData[i] = NumOps.FromDouble(outputTensor.GetValue(i));
        return new Tensor<T>(outputTensor.Dimensions.ToArray(), new Vector<T>(outputData));
    }

    #endregion

    #region Model-Specific Processing

    private Tensor<T> GenerateQuantilePredictions(Tensor<T> input, double[] quantiles)
    {
        int numSamples = 100;
        var samples = new List<Tensor<T>>();
        SetTrainingMode(true);
        for (int s = 0; s < numSamples; s++) samples.Add(Forward(input));
        SetTrainingMode(false);

        var result = new Tensor<T>(new[] { 1, _forecastHorizon, quantiles.Length });
        for (int t = 0; t < _forecastHorizon; t++)
        {
            var values = new List<double>();
            foreach (var sample in samples)
                if (t < sample.Length) values.Add(NumOps.ToDouble(sample.Data.Span[t]));
            values.Sort();
            for (int q = 0; q < quantiles.Length; q++)
            {
                int idx = Math.Min((int)(quantiles[q] * values.Count), values.Count - 1);
                result.Data.Span[t * quantiles.Length + q] = NumOps.FromDouble(values[idx]);
            }
        }
        return result;
    }

    /// <inheritdoc/>
    protected override Tensor<T> ShiftInputWithPredictions(Tensor<T> input, Tensor<T> predictions, int stepsUsed)
    {
        var result = new Tensor<T>(input.Shape);
        for (int i = 0; i < _contextLength - stepsUsed; i++)
            result.Data.Span[i] = input.Data.Span[i + stepsUsed];
        for (int i = 0; i < stepsUsed && i < predictions.Length; i++)
            result.Data.Span[_contextLength - stepsUsed + i] = predictions.Data.Span[i];
        return result;
    }

    /// <inheritdoc/>
    protected override Tensor<T> ConcatenatePredictions(List<Tensor<T>> predictions, int totalSteps)
    {
        var result = new Tensor<T>(new[] { 1, totalSteps, 1 });
        int position = 0;
        foreach (var pred in predictions)
        {
            int toCopy = Math.Min(pred.Length, totalSteps - position);
            for (int i = 0; i < toCopy; i++) result.Data.Span[position + i] = pred.Data.Span[i];
            position += toCopy;
        }
        return result;
    }

    #endregion

    #region IDisposable

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing) OnnxSession?.Dispose();
        base.Dispose(disposing);
    }

    #endregion
}
