using System.IO;
using AiDotNet.Enums;
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
/// TF-C — Time-Frequency Consistency for Self-Supervised Time Series.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// TF-C learns time series representations by enforcing consistency between time-domain
/// and frequency-domain representations via contrastive learning, capturing both
/// temporal and spectral patterns. It uses dual CNN encoders with a shared projection head.
/// </para>
/// <para>
/// <b>Reference:</b> Zhang et al., "Self-Supervised Contrastive Pre-Training For Time Series via Time-Frequency Consistency", NeurIPS 2022.
/// </para>
/// </remarks>
public class TFC<T> : TimeSeriesFoundationModelBase<T>
{
    #region Fields

    private readonly bool _useNativeMode;
    private ILayer<T>? _timeInputProjection;
    private readonly List<ILayer<T>> _timeEncoderLayers = [];
    private ILayer<T>? _freqInputProjection;
    private readonly List<ILayer<T>> _freqEncoderLayers = [];
    private ILayer<T>? _projectionHead;
    private ILayer<T>? _forecastHead;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly TFCOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private int _contextLength;
    private int _forecastHorizon;
    private int _hiddenDimension;
    private int _projectionDimension;
    private int _numTimeLayers;
    private int _numFreqLayers;
    private double _dropout;
    private double _contrastiveTemperature;

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
    public override FoundationModelSize ModelSize => FoundationModelSize.Small;
    /// <inheritdoc/>
    public override int MaxContextLength => _contextLength;
    /// <inheritdoc/>
    public override int MaxPredictionHorizon => _forecastHorizon;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a TF-C model using a pretrained ONNX model.
    /// </summary>
    public TFC(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        TFCOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        options ??= new TFCOptions<T>();
        _options = options;
        Options = _options;

        _useNativeMode = false;
        OnnxModelPath = onnxModelPath;
        OnnxSession = new InferenceSession(onnxModelPath);

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        CopyOptionsToFields(options);
    }

    /// <summary>
    /// Creates a TF-C model in native mode for training or fine-tuning.
    /// </summary>
    public TFC(
        NeuralNetworkArchitecture<T> architecture,
        TFCOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        options ??= new TFCOptions<T>();
        _options = options;
        Options = _options;

        _useNativeMode = true;
        OnnxSession = null;
        OnnxModelPath = null;

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        CopyOptionsToFields(options);
        InitializeLayers();
    }

    private void CopyOptionsToFields(TFCOptions<T> options)
    {
        _contextLength = options.ContextLength;
        _forecastHorizon = options.ForecastHorizon;
        _hiddenDimension = options.HiddenDimension;
        _projectionDimension = options.ProjectionDimension;
        _numTimeLayers = options.NumTimeLayers;
        _numFreqLayers = options.NumFreqLayers;
        _dropout = options.DropoutRate;
        _contrastiveTemperature = options.ContrastiveTemperature;
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultTFCLayers(
                Architecture, _contextLength, _forecastHorizon, _hiddenDimension,
                _projectionDimension, _numTimeLayers, _numFreqLayers, _dropout));
            ExtractLayerReferences();
        }
    }

    private void ExtractLayerReferences()
    {
        int idx = 0;
        int layersPerBlock = _dropout > 0 ? 3 : 2;

        // Time encoder input projection
        if (idx < Layers.Count)
            _timeInputProjection = Layers[idx++];

        // Time encoder layers
        _timeEncoderLayers.Clear();
        int totalTimeLayers = _numTimeLayers * layersPerBlock;
        for (int i = 0; i < totalTimeLayers && idx < Layers.Count; i++)
            _timeEncoderLayers.Add(Layers[idx++]);

        // Frequency encoder input projection
        if (idx < Layers.Count)
            _freqInputProjection = Layers[idx++];

        // Frequency encoder layers
        _freqEncoderLayers.Clear();
        int totalFreqLayers = _numFreqLayers * layersPerBlock;
        for (int i = 0; i < totalFreqLayers && idx < Layers.Count; i++)
            _freqEncoderLayers.Add(Layers[idx++]);

        // Shared projection head
        if (idx < Layers.Count)
            _projectionHead = Layers[idx++];

        // Forecast head
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
            // Joint training: forecasting loss + time-frequency contrastive loss
            var output = ForwardNative(input);
            T forecastLoss = _lossFunction.CalculateLoss(output.ToVector(), target.ToVector());

            // Compute time-frequency contrastive loss (InfoNCE)
            T contrastiveLoss = ComputeContrastiveLoss(input);

            // Combined loss: forecast_loss + contrastive_loss
            LastLoss = NumOps.Add(forecastLoss, contrastiveLoss);

            var gradient = _lossFunction.CalculateDerivative(output.ToVector(), target.ToVector());
            BackwardNative(Tensor<T>.FromVector(gradient, output.Shape));

            _optimizer.UpdateParameters(Layers);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <summary>
    /// InfoNCE contrastive loss between time and frequency encoder outputs.
    /// Maximizes agreement between time and frequency representations of the same sample.
    /// </summary>
    private T ComputeContrastiveLoss(Tensor<T> input)
    {
        var normalized = ApplyInstanceNormalization(input);
        var timeCurrent = normalized;
        if (timeCurrent.Rank == 1) timeCurrent = timeCurrent.Reshape(new[] { 1, timeCurrent.Length });

        // Time encoder
        if (_timeInputProjection is not null) timeCurrent = _timeInputProjection.Forward(timeCurrent);
        foreach (var layer in _timeEncoderLayers) timeCurrent = layer.Forward(timeCurrent);

        // Frequency encoder
        var freqInput = ComputeFrequencyRepresentation(normalized);
        if (freqInput.Rank == 1) freqInput = freqInput.Reshape(new[] { 1, freqInput.Length });
        var freqCurrent = freqInput;
        if (_freqInputProjection is not null) freqCurrent = _freqInputProjection.Forward(freqCurrent);
        foreach (var layer in _freqEncoderLayers) freqCurrent = layer.Forward(freqCurrent);

        // Project both to shared space
        Tensor<T> timeProj = timeCurrent, freqProj = freqCurrent;
        if (_projectionHead is not null)
        {
            timeProj = _projectionHead.Forward(timeCurrent);
            freqProj = _projectionHead.Forward(freqCurrent);
        }

        // Cosine similarity / temperature
        T dotProduct = NumOps.Zero;
        T normTime = NumOps.Zero;
        T normFreq = NumOps.Zero;
        int len = Math.Min(timeProj.Length, freqProj.Length);
        for (int i = 0; i < len; i++)
        {
            dotProduct = NumOps.Add(dotProduct, NumOps.Multiply(timeProj[i], freqProj[i]));
            normTime = NumOps.Add(normTime, NumOps.Multiply(timeProj[i], timeProj[i]));
            normFreq = NumOps.Add(normFreq, NumOps.Multiply(freqProj[i], freqProj[i]));
        }
        T eps8 = NumOps.FromDouble(1e-8);
        T normProduct = NumOps.Add(NumOps.Multiply(NumOps.Sqrt(normTime), NumOps.Sqrt(normFreq)), eps8);
        T cosSim = NumOps.Divide(dotProduct, normProduct);
        T tempT = NumOps.FromDouble(Math.Max(1e-8, _contrastiveTemperature));
        T logit = NumOps.Divide(cosSim, tempT);

        // -log(sigmoid(logit)) for positive pair — use log-sum-exp for numerical stability
        // -log(sigmoid(x)) = log(1 + exp(-x))
        double logitD = NumOps.ToDouble(logit);
        double loss = Math.Log(1.0 + Math.Exp(-logitD));
        return NumOps.FromDouble(loss);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> gradients)
    {
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "TFC" },
                { "ContextLength", _contextLength },
                { "ForecastHorizon", _forecastHorizon },
                { "HiddenDimension", _hiddenDimension },
                { "ProjectionDimension", _projectionDimension },
                { "NumTimeLayers", _numTimeLayers },
                { "NumFreqLayers", _numFreqLayers },
                { "UseNativeMode", _useNativeMode },
                { "ParameterCount", GetParameterCount() }
            },
            ModelData = _useNativeMode ? this.Serialize() : Array.Empty<byte>()
        };
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new TFC<T>(Architecture, new TFCOptions<T>
        {
            ContextLength = _contextLength,
            ForecastHorizon = _forecastHorizon,
            HiddenDimension = _hiddenDimension,
            ProjectionDimension = _projectionDimension,
            NumTimeLayers = _numTimeLayers,
            NumFreqLayers = _numFreqLayers,
            DropoutRate = _dropout,
            ContrastiveTemperature = _contrastiveTemperature
        });
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_contextLength);
        writer.Write(_forecastHorizon);
        writer.Write(_hiddenDimension);
        writer.Write(_projectionDimension);
        writer.Write(_numTimeLayers);
        writer.Write(_numFreqLayers);
        writer.Write(_dropout);
        writer.Write(_contrastiveTemperature);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _contextLength = reader.ReadInt32();
        _forecastHorizon = reader.ReadInt32();
        _hiddenDimension = reader.ReadInt32();
        _projectionDimension = reader.ReadInt32();
        _numTimeLayers = reader.ReadInt32();
        _numFreqLayers = reader.ReadInt32();
        _dropout = reader.ReadDouble();
        _contrastiveTemperature = reader.ReadDouble();
    }

    #endregion

    #region IForecastingModel Implementation

    /// <inheritdoc/>
    public override Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null)
    {
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
            ["ProjectionDimension"] = NumOps.FromDouble(_projectionDimension),
            ["NumTimeLayers"] = NumOps.FromDouble(_numTimeLayers),
            ["NumFreqLayers"] = NumOps.FromDouble(_numFreqLayers),
            ["LastLoss"] = lastLoss
        };
    }

    #endregion

    #region Forward/Backward Pass

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

        // Time-domain encoder path
        if (_timeInputProjection is not null)
            current = _timeInputProjection.Forward(current);

        foreach (var layer in _timeEncoderLayers)
            current = layer.Forward(current);

        // Frequency-domain path: compute DFT magnitude spectrum as input
        var freqInput = ComputeFrequencyRepresentation(normalized);
        if (freqInput.Rank == 1)
            freqInput = freqInput.Reshape(new[] { 1, freqInput.Length });

        var freqCurrent = freqInput;
        if (_freqInputProjection is not null)
            freqCurrent = _freqInputProjection.Forward(freqCurrent);

        foreach (var layer in _freqEncoderLayers)
            freqCurrent = layer.Forward(freqCurrent);

        // Average time and frequency representations (contrastive fusion)
        for (int i = 0; i < current.Length && i < freqCurrent.Length; i++)
        {
            current.Data.Span[i] = NumOps.Divide(
                NumOps.Add(current[i], freqCurrent[i]),
                NumOps.FromDouble(2.0));
        }

        if (_projectionHead is not null)
            current = _projectionHead.Forward(current);

        if (_forecastHead is not null)
            current = _forecastHead.Forward(current);

        if (addedBatchDim && current.Rank == 2 && current.Shape[0] == 1)
            current = current.Reshape(new[] { current.Shape[1] });

        return current;
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

        if (_projectionHead is not null)
            current = _projectionHead.Backward(current);

        // Scale gradient for dual-encoder averaging
        var scaledGrad = new Tensor<T>(current.Shape);
        for (int i = 0; i < current.Length; i++)
            scaledGrad.Data.Span[i] = NumOps.Divide(current[i], NumOps.FromDouble(2.0));

        // Backward through frequency encoder
        var freqGrad = scaledGrad;
        for (int i = _freqEncoderLayers.Count - 1; i >= 0; i--)
            freqGrad = _freqEncoderLayers[i].Backward(freqGrad);

        if (_freqInputProjection is not null)
            freqGrad = _freqInputProjection.Backward(freqGrad);

        // Backward through time encoder
        var timeGrad = scaledGrad;
        for (int i = _timeEncoderLayers.Count - 1; i >= 0; i--)
            timeGrad = _timeEncoderLayers[i].Backward(timeGrad);

        if (_timeInputProjection is not null)
            timeGrad = _timeInputProjection.Backward(timeGrad);

        // Sum gradients from both paths
        current = new Tensor<T>(timeGrad.Shape);
        for (int i = 0; i < timeGrad.Length && i < freqGrad.Length; i++)
            current.Data.Span[i] = NumOps.Add(timeGrad[i], freqGrad[i]);

        if (addedBatchDim && current.Rank == 2 && current.Shape[0] == 1)
            current = current.Reshape(new[] { current.Shape[1] });

        return current;
    }

    protected override Tensor<T> ForecastOnnx(Tensor<T> input)
    {
        if (OnnxSession == null)
            throw new InvalidOperationException("ONNX session is not initialized.");

        int batchSize = input.Shape[0];
        int seqLen = input.Shape.Length > 1 ? input.Shape[1] : input.Length;
        int features = input.Shape.Length > 2 ? input.Shape[2] : 1;

        var inputData = new float[batchSize * seqLen * features];
        for (int i = 0; i < input.Length && i < inputData.Length; i++)
            inputData[i] = (float)NumOps.ToDouble(input[i]);

        var inputTensor = new OnnxTensors.DenseTensor<float>(
            inputData, new[] { batchSize, seqLen, features });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input", inputTensor)
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

    #region Frequency Transform

    /// <summary>
    /// Computes the DFT magnitude spectrum of the input time series.
    /// Returns |X[k]| for k = 0..N/2 (one-sided spectrum), same length as input via zero-padding.
    /// </summary>
    private Tensor<T> ComputeFrequencyRepresentation(Tensor<T> input)
    {
        int n = input.Length;
        int halfN = n / 2 + 1;
        var magnitudes = new Vector<T>(n); // same length as input, zero-padded after Nyquist
        T invN = NumOps.Divide(NumOps.One, NumOps.FromDouble(n));

        // DFT: X[k] = sum_{t=0}^{N-1} x[t] * exp(-2*pi*i*k*t/N)
        for (int k = 0; k < halfN; k++)
        {
            T realPart = NumOps.Zero;
            T imagPart = NumOps.Zero;
            for (int t = 0; t < n; t++)
            {
                double angle = -2.0 * Math.PI * k * t / n;
                T cosT = NumOps.FromDouble(Math.Cos(angle));
                T sinT = NumOps.FromDouble(Math.Sin(angle));
                realPart = NumOps.Add(realPart, NumOps.Multiply(input[t], cosT));
                imagPart = NumOps.Add(imagPart, NumOps.Multiply(input[t], sinT));
            }
            T magSquared = NumOps.Add(NumOps.Multiply(realPart, realPart), NumOps.Multiply(imagPart, imagPart));
            magnitudes[k] = NumOps.Multiply(NumOps.Sqrt(magSquared), invN);
        }

        // Mirror the one-sided spectrum for symmetric representation
        for (int k = halfN; k < n; k++)
            magnitudes[k] = magnitudes[n - k];

        var result = new Tensor<T>(input.Shape);
        for (int i = 0; i < n; i++)
            result.Data.Span[i] = magnitudes[i];

        return result;
    }

    #endregion

    #region Parameter Estimation

    private new int GetParameterCount()
    {
        // Time encoder
        long total = (long)_contextLength * _hiddenDimension + _hiddenDimension;
        long perTimeLayer = 2L * _hiddenDimension * _hiddenDimension + 2 * _hiddenDimension;
        total += perTimeLayer * _numTimeLayers;

        // Frequency encoder (same size)
        total += (long)_contextLength * _hiddenDimension + _hiddenDimension;
        long perFreqLayer = 2L * _hiddenDimension * _hiddenDimension + 2 * _hiddenDimension;
        total += perFreqLayer * _numFreqLayers;

        // Projection head
        total += (long)_hiddenDimension * _projectionDimension + _projectionDimension;

        // Forecast head
        total += (long)_projectionDimension * _forecastHorizon + _forecastHorizon;

        return (int)Math.Min(total, int.MaxValue);
    }

    #endregion
}
