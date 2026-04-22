using System.IO;
using AiDotNet.Attributes;
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
/// SimMTM — Simple Pre-Training Framework for Masked Time-Series Modeling.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SimMTM combines masked time series modeling with series-level similarity learning,
/// recovering masked series by aggregating from similar unmasked series in the batch.
/// It uses a patch-based transformer with a similarity-weighted reconstruction objective.
/// </para>
/// <para><b>For Beginners:</b> SimMTM learns about time series by playing a fill-in-the-blank
/// game. Parts of the data are hidden (masked), and the model must reconstruct them. What makes
/// it special is that it looks at similar series in the training batch for clues, like asking
/// a friend who has seen similar patterns. This pre-training approach helps the model learn
/// robust representations that transfer well to forecasting tasks.</para>
/// <para>
/// <b>Reference:</b> Dong et al., "SimMTM: A Simple Pre-Training Framework for Masked Time-Series Modeling", NeurIPS 2023.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a SimMTM model for masked time series pre-training with similarity learning
/// // Reconstructs masked series using similar unmasked series from the training batch
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 512, inputWidth: 1, inputDepth: 1, outputSize: 24);
///
/// // Training mode with similarity-weighted masked reconstruction
/// var model = new SimMTM&lt;double&gt;(architecture);
///
/// // ONNX inference mode with pre-trained model
/// var onnxModel = new SimMTM&lt;double&gt;(architecture, "simmtm.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Finance)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Forecasting)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("SimMTM: A Simple Pre-Training Framework for Masked Time-Series Modeling", "https://arxiv.org/abs/2302.00861", Year = 2023, Authors = "Jiaxiang Dong, Haixu Wu, Haoran Zhang, Li Zhang, Jianmin Wang, Mingsheng Long")]
public class SimMTM<T> : TimeSeriesFoundationModelBase<T>
{
    #region Fields

    private readonly bool _useNativeMode;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly SimMTMOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private int _contextLength;
    private int _forecastHorizon;
    private int _patchLength;
    private int _hiddenDimension;
    private int _numLayers;
    private int _numHeads;
    private double _maskRatio;
    private double _dropout;
    private double _similarityTemperature;

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override int SequenceLength => _contextLength;
    /// <inheritdoc/>
    public override int PredictionHorizon => _forecastHorizon;
    /// <inheritdoc/>
    public override int NumFeatures => 1;
    /// <inheritdoc/>
    public override int PatchSize => _patchLength;
    /// <inheritdoc/>
    public override int Stride => _patchLength;
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
    /// Creates a SimMTM model using a pretrained ONNX model.
    /// </summary>
    public SimMTM(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        SimMTMOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        options ??= new SimMTMOptions<T>();
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
    /// Creates a SimMTM model in native mode for training or fine-tuning.
    /// </summary>
    public SimMTM(
        NeuralNetworkArchitecture<T> architecture,
        SimMTMOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        options ??= new SimMTMOptions<T>();
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

    private void CopyOptionsToFields(SimMTMOptions<T> options)
    {
        _contextLength = options.ContextLength;
        _forecastHorizon = options.ForecastHorizon;
        _patchLength = options.PatchLength;
        _hiddenDimension = options.HiddenDimension;
        _numLayers = options.NumLayers;
        _numHeads = options.NumHeads;
        _maskRatio = options.MaskRatio;
        _dropout = options.DropoutRate;
        _similarityTemperature = options.SimilarityTemperature;
    }

    #endregion

    #region Initialization

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else if (_useNativeMode)
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultSimMTMLayers(
                Architecture, _contextLength, _forecastHorizon, _patchLength,
                _hiddenDimension, _numLayers, _numHeads, _dropout));
        }
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

        // Issue #1166: the old body computed a loss + gradient and then
        // called _optimizer.UpdateParameters(Layers) without a backward
        // pass, so every layer's UpdateParameters threw "Backward pass
        // must be called before updating parameters." Delegate to
        // FinancialModelBase.Train — it routes through the tape-based
        // NeuralNetworkBase.TrainWithTape flow (GradientTape forward +
        // tape.ComputeGradients + optimizer.Step) that every other
        // NeuralNetworkBase subclass uses.
        base.Train(input, target);
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
                { "NetworkType", "SimMTM" },
                { "ContextLength", _contextLength },
                { "ForecastHorizon", _forecastHorizon },
                { "PatchLength", _patchLength },
                { "HiddenDimension", _hiddenDimension },
                { "NumLayers", _numLayers },
                { "MaskRatio", _maskRatio },
                { "UseNativeMode", _useNativeMode },
                { "ParameterCount", GetParameterCount() }
            },
            ModelData = _useNativeMode ? this.Serialize() : Array.Empty<byte>()
        };
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var opts = new SimMTMOptions<T>
        {
            ContextLength = _contextLength,
            ForecastHorizon = _forecastHorizon,
            PatchLength = _patchLength,
            HiddenDimension = _hiddenDimension,
            NumLayers = _numLayers,
            NumHeads = _numHeads,
            MaskRatio = _maskRatio,
            DropoutRate = _dropout,
            SimilarityTemperature = _similarityTemperature
        };

        if (!_useNativeMode && OnnxModelPath is not null)
            return new SimMTM<T>(Architecture, OnnxModelPath, opts);

        return new SimMTM<T>(Architecture, opts);
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_contextLength);
        writer.Write(_forecastHorizon);
        writer.Write(_patchLength);
        writer.Write(_hiddenDimension);
        writer.Write(_numLayers);
        writer.Write(_numHeads);
        writer.Write(_maskRatio);
        writer.Write(_dropout);
        writer.Write(_similarityTemperature);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _contextLength = reader.ReadInt32();
        _forecastHorizon = reader.ReadInt32();
        _patchLength = reader.ReadInt32();
        _hiddenDimension = reader.ReadInt32();
        _numLayers = reader.ReadInt32();
        _numHeads = reader.ReadInt32();
        _maskRatio = reader.ReadDouble();
        _dropout = reader.ReadDouble();
        _similarityTemperature = reader.ReadDouble();
    }

    #endregion

    #region IForecastingModel Implementation

    /// <inheritdoc/>
    public override Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null)
    {
        if (quantiles is not null && quantiles.Length > 0)
            throw new NotSupportedException("SimMTM does not support quantile forecasting. Pass null for point forecasts.");

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
        int batchSize = input.Rank > 1 ? input.Shape[0] : 1;
        int seqLen = input.Rank > 1 ? input.Shape[1] : input.Length;
        var result = new Tensor<T>(input._shape);

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
            ["PatchLength"] = NumOps.FromDouble(_patchLength),
            ["HiddenDimension"] = NumOps.FromDouble(_hiddenDimension),
            ["NumLayers"] = NumOps.FromDouble(_numLayers),
            ["LastLoss"] = lastLoss
        };
    }

    #endregion

    #region Similarity-Weighted Masked Pretraining (Dong et al. 2023)

    /// <summary>
    /// Performs one SimMTM similarity-weighted masked-reconstruction pretraining forward
    /// pass per Dong et al. 2023. Randomly masks a fraction of input patches, runs the
    /// transformer encoder, then reconstructs each masked patch as a similarity-weighted
    /// aggregation over UNMASKED patches' hidden states (rather than a plain dense
    /// projection). Similarity is cosine-similarity with softmax-temperature scaling.
    /// </summary>
    /// <param name="input">Input time series of shape [B, contextLength] or [contextLength].</param>
    /// <param name="seed">Optional seed for reproducible masking.</param>
    /// <returns>
    /// A tuple <c>(reconstructed, patchMask)</c> where <c>reconstructed</c> is the full
    /// signal predicted by similarity-weighted aggregation and <c>patchMask</c> is a
    /// binary [B, numPatches] tensor with 1 at masked positions.
    /// </returns>
    public (Tensor<T> reconstructed, Tensor<T> patchMask) PretrainSimilarityWeightedReconstruction(
        Tensor<T> input, int? seed = null)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException(
                "Similarity pretraining requires native mode.");
        if (input is null) throw new ArgumentNullException(nameof(input));

        // Validate input geometry before we index with b * _contextLength + ...
        // A bad shape would otherwise walk past input.Data.Span.
        if (_patchLength <= 0)
            throw new InvalidOperationException("PatchLength must be positive.");
        if (_contextLength <= 0 || _contextLength % _patchLength != 0)
            throw new InvalidOperationException(
                $"ContextLength ({_contextLength}) must be positive and divisible by PatchLength ({_patchLength}).");
        if (input.Rank != 1 && input.Rank != 2)
            throw new ArgumentException(
                $"SimMTM pretraining expects rank-1 or rank-2 input; got rank {input.Rank}, shape "
                + $"[{string.Join(", ", input.Shape.ToArray())}].",
                nameof(input));
        int trailingDim = input.Shape[input.Rank - 1];
        if (trailingDim != _contextLength)
            throw new ArgumentException(
                $"SimMTM pretraining expects each sample to have length {_contextLength}; got shape "
                + $"[{string.Join(", ", input.Shape.ToArray())}].",
                nameof(input));

        bool addedBatch = false;
        if (input.Rank == 1)
        {
            input = input.Reshape(new[] { 1, input.Length });
            addedBatch = true;
        }
        int batchSize = input.Shape[0];
        int numPatches = _contextLength / _patchLength;

        // Build patch-level mask [B, numPatches]: 1 = masked, 0 = visible.
        var rng = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
        var patchMask = new Tensor<T>(new[] { batchSize, numPatches });
        int maskedCount = Math.Max(1, Math.Min(numPatches - 1, (int)Math.Round(numPatches * _maskRatio)));
        for (int b = 0; b < batchSize; b++)
        {
            var indices = Enumerable.Range(0, numPatches).ToList();
            for (int i = indices.Count - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }
            for (int m = 0; m < maskedCount; m++)
                patchMask.Data.Span[b * numPatches + indices[m]] = NumOps.One;
        }

        // Apply mask: zero out masked patches.
        var masked = new Tensor<T>(input._shape);
        for (int b = 0; b < batchSize; b++)
        {
            for (int p = 0; p < numPatches; p++)
            {
                bool isMasked = !NumOps.Equals(patchMask.Data.Span[b * numPatches + p], NumOps.Zero);
                for (int t = 0; t < _patchLength; t++)
                {
                    int idx = b * _contextLength + p * _patchLength + t;
                    masked.Data.Span[idx] = isMasked ? NumOps.Zero : input.Data.Span[idx];
                }
            }
        }

        // Walk encoder up to (but not including) the final two heads (Flatten + Dense
        // reconstruction + Dense forecast). The layer just before Flatten is the last
        // TransformerEncoderLayer output — our [B, numPatches, hiddenDim] hidden states.
        var current = ApplyInstanceNormalization(masked);
        int encoderEnd = Layers.Count - 3; // skip Flatten, Dense(recon), Dense(forecast)
        for (int i = 0; i < encoderEnd; i++)
            current = Layers[i].Forward(current);
        var hidden = current; // [B, numPatches, hiddenDim]

        // Similarity-weighted aggregation per Dong et al. 2023 §3.2: for each
        // masked patch i in series b, reconstruct its hidden state as a
        // softmax-weighted sum over VISIBLE peers drawn from the full batch
        // (cross-series when batchSize > 1, degenerating to intra-series when
        // batchSize == 1). Each candidate (bj, pj) contributes its hidden
        // vector weighted by cosine(h_{b,pi}, h_{bj,pj}) / tau.
        double tau = _similarityTemperature > 0 ? _similarityTemperature : 0.1;
        var aggregated = new Tensor<T>(hidden._shape);
        int totalPositions = batchSize * numPatches;
        var scores = new double[totalPositions];
        var weights = new double[totalPositions];
        for (int b = 0; b < batchSize; b++)
        {
            for (int pi = 0; pi < numPatches; pi++)
            {
                // Only MASKED patches get reconstructed from visible peers per
                // Dong et al. 2023. Visible patches pass through unchanged so
                // the reconstruction loss doesn't drive the model to overwrite
                // uncorrupted evidence with a similarity-weighted blur.
                int queryIdx = b * numPatches + pi;
                bool isMasked = !NumOps.Equals(patchMask.Data.Span[queryIdx], NumOps.Zero);
                if (!isMasked)
                {
                    for (int h = 0; h < _hiddenDimension; h++)
                        aggregated.Data.Span[queryIdx * _hiddenDimension + h] =
                            hidden.Data.Span[queryIdx * _hiddenDimension + h];
                    continue;
                }

                double normI = 0;
                for (int h = 0; h < _hiddenDimension; h++)
                {
                    double v = NumOps.ToDouble(hidden.Data.Span[(b * numPatches + pi) * _hiddenDimension + h]);
                    normI += v * v;
                }
                normI = Math.Sqrt(normI) + 1e-8;

                double maxScore = double.NegativeInfinity;
                for (int bj = 0; bj < batchSize; bj++)
                {
                    for (int pj = 0; pj < numPatches; pj++)
                    {
                        int idx = bj * numPatches + pj;
                        // Skip self-position so a masked query does not attend to itself.
                        if (bj == b && pj == pi) { scores[idx] = double.NegativeInfinity; continue; }
                        // Only aggregate over VISIBLE patches (mask == 0) across the full batch.
                        if (!NumOps.Equals(patchMask.Data.Span[idx], NumOps.Zero))
                        {
                            scores[idx] = double.NegativeInfinity;
                            continue;
                        }
                        double dot = 0, normJ = 0;
                        for (int h = 0; h < _hiddenDimension; h++)
                        {
                            double vi = NumOps.ToDouble(hidden.Data.Span[(b * numPatches + pi) * _hiddenDimension + h]);
                            double vj = NumOps.ToDouble(hidden.Data.Span[idx * _hiddenDimension + h]);
                            dot += vi * vj;
                            normJ += vj * vj;
                        }
                        normJ = Math.Sqrt(normJ) + 1e-8;
                        scores[idx] = dot / (normI * normJ) / tau;
                        if (scores[idx] > maxScore) maxScore = scores[idx];
                    }
                }

                // Softmax over visible peers.
                double denom = 0;
                for (int k = 0; k < totalPositions; k++)
                {
                    if (double.IsNegativeInfinity(scores[k])) { weights[k] = 0; continue; }
                    weights[k] = Math.Exp(scores[k] - maxScore);
                    denom += weights[k];
                }
                if (denom < 1e-12)
                {
                    // All peers masked (pathological). Fall back to self-hidden.
                    for (int h = 0; h < _hiddenDimension; h++)
                        aggregated.Data.Span[(b * numPatches + pi) * _hiddenDimension + h] =
                            hidden.Data.Span[(b * numPatches + pi) * _hiddenDimension + h];
                    continue;
                }
                for (int k = 0; k < totalPositions; k++)
                    weights[k] /= denom;

                // Weighted sum of visible peers' hidden states (cross-batch + cross-patch).
                for (int h = 0; h < _hiddenDimension; h++)
                {
                    double agg = 0;
                    for (int k = 0; k < totalPositions; k++)
                    {
                        if (weights[k] == 0) continue;
                        agg += weights[k] * NumOps.ToDouble(hidden.Data.Span[k * _hiddenDimension + h]);
                    }
                    aggregated.Data.Span[(b * numPatches + pi) * _hiddenDimension + h] = NumOps.FromDouble(agg);
                }
            }
        }

        // Feed the aggregated hidden states through the reconstruction head (the
        // three remaining layers: Flatten + Dense + Dense). We want just the
        // reconstruction (skip the last forecast Dense).
        var reconIn = Layers[encoderEnd].Forward(aggregated);       // Flatten
        var reconOut = Layers[encoderEnd + 1].Forward(reconIn);     // Dense(reconstruction)

        if (addedBatch && reconOut.Rank == 2 && reconOut.Shape[0] == 1)
            reconOut = reconOut.Reshape(new[] { reconOut.Shape[1] });
        if (addedBatch)
            patchMask = patchMask.Reshape(new[] { numPatches });

        return (reconOut, patchMask);
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

        // Helper emits a flat Layers list: Reshape → Dense(patch) → N ×
        // TransformerEncoderLayer (+ optional Dropout) → Flatten →
        // Dense(reconstruction head) → Dense(forecast head).
        foreach (var layer in Layers)
            current = layer.Forward(current);

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
        if (_patchLength <= 0)
            return 0;

        int numPatches = _contextLength / _patchLength;
        long total = (long)_patchLength * _hiddenDimension + _hiddenDimension;

        long perLayer = 4L * _hiddenDimension * _hiddenDimension + 4 * _hiddenDimension;
        perLayer += 2L * _hiddenDimension * (_hiddenDimension * 4) + _hiddenDimension + (_hiddenDimension * 4);
        perLayer += 4L * _hiddenDimension;
        total += perLayer * _numLayers;

        total += (long)_hiddenDimension * _patchLength + _patchLength;
        total += (long)numPatches * _patchLength * _forecastHorizon;

        return (int)Math.Min(total, int.MaxValue);
    }

    #endregion
}
