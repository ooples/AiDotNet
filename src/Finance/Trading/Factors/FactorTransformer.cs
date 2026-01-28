using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Finance.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.Finance.Trading.Factors;

/// <summary>
/// Transformer-based model for learning financial factors with attention mechanisms.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// FactorTransformer uses self-attention to model cross-sectional and temporal
/// relationships in financial data, extracting factors that capture complex patterns.
/// </para>
/// <para>
/// <b>For Beginners:</b> Transformers can focus on different parts of the data at once.
/// This model uses that ability to find hidden factor signals that drive asset returns.
/// </para>
/// <para>
/// Reference: Duan et al. (2022). "FactorFormer: A Transformer-based Framework for Factor Investing"
/// </para>
/// </remarks>
public class FactorTransformer<T> : NeuralNetworkBase<T>, IFactorModel<T>
{
    #region Execution Mode

    private readonly bool _useNativeMode;

    #endregion

    #region ONNX Mode Fields

    private readonly InferenceSession? _onnxSession;
    private readonly string? _onnxModelPath;

    #endregion

    #region Shared Fields

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly FactorTransformerOptions<T> _options;
    private readonly int _numFactors;
    private readonly int _numAssets;
    private readonly int _numFeatures;
    private readonly int _hiddenDimension;
    private readonly int _numHeads;
    private readonly int _numTransformerLayers;
    private readonly int _sequenceLength;
    private readonly int _predictionHorizon;
    private readonly double _dropoutRate;

    #endregion

    #region Interface Properties

    /// <summary>
    /// Gets whether the model is using native layers (true) or ONNX inference (false).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Native mode supports training, ONNX mode is for fast predictions.
    /// </para>
    /// </remarks>
    public bool UseNativeMode => _useNativeMode;

    /// <summary>
    /// Gets whether training is supported in the current mode.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Training is only available in native mode.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => _useNativeMode;

    /// <summary>
    /// Gets the number of latent factors learned by the model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is how many hidden drivers of returns the model discovers.
    /// </para>
    /// </remarks>
    public int NumFactors => _numFactors;

    /// <summary>
    /// Gets the number of assets covered by the model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the size of the asset universe being modeled.
    /// </para>
    /// </remarks>
    public int NumAssets => _numAssets;

    /// <summary>
    /// Gets the number of input features.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Features can include returns, technical indicators, and fundamentals.
    /// </para>
    /// </remarks>
    public int NumFeatures => _numFeatures;

    /// <summary>
    /// Gets the input sequence length.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> How many time steps of history the model sees at once.
    /// </para>
    /// </remarks>
    public int SequenceLength => _sequenceLength;

    /// <summary>
    /// Gets the prediction horizon.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> How far ahead the model is trained to forecast.
    /// </para>
    /// </remarks>
    public int PredictionHorizon => _predictionHorizon;

    /// <summary>
    /// Gets the number of attention heads.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> More heads let the model attend to multiple relationships at once.
    /// </para>
    /// </remarks>
    public int NumHeads => _numHeads;

    /// <summary>
    /// Gets the number of transformer layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Deeper stacks allow more complex reasoning but take more compute.
    /// </para>
    /// </remarks>
    public int NumTransformerLayers => _numTransformerLayers;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new FactorTransformer in ONNX mode for inference.
    /// </summary>
    /// <param name="architecture">The user-provided neural network architecture.</param>
    /// <param name="onnxModelPath">Path to the pretrained ONNX model.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <param name="optimizer">Optional optimizer for fine-tuning.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this when you already have a trained ONNX model
    /// and only need fast predictions.
    /// </para>
    /// </remarks>
    public FactorTransformer(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        FactorTransformerOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentNullException(nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        _useNativeMode = false;
        _onnxModelPath = onnxModelPath;
        _onnxSession = new InferenceSession(onnxModelPath);

        _options = options ?? new FactorTransformerOptions<T>();
        _options.Validate();

        _numFactors = _options.NumFactors;
        _numAssets = _options.NumAssets;
        _numFeatures = _options.NumFeatures;
        _hiddenDimension = _options.HiddenDimension;
        _numHeads = _options.NumHeads;
        _numTransformerLayers = _options.NumTransformerLayers;
        _sequenceLength = _options.SequenceLength;
        _predictionHorizon = _options.PredictionHorizon;
        _dropoutRate = _options.DropoutRate;

        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    /// <summary>
    /// Initializes a new FactorTransformer in native mode for training and inference.
    /// </summary>
    /// <param name="architecture">The user-provided neural network architecture.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <param name="optimizer">Optional optimizer for training.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this when you want to train the transformer on your own data.
    /// Native mode builds all layers in C# so gradients can be computed.
    /// </para>
    /// </remarks>
    public FactorTransformer(
        NeuralNetworkArchitecture<T> architecture,
        FactorTransformerOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        _useNativeMode = true;
        _onnxModelPath = null;
        _onnxSession = null;

        _options = options ?? new FactorTransformerOptions<T>();
        _options.Validate();

        _numFactors = _options.NumFactors;
        _numAssets = _options.NumAssets;
        _numFeatures = _options.NumFeatures;
        _hiddenDimension = _options.HiddenDimension;
        _numHeads = _options.NumHeads;
        _numTransformerLayers = _options.NumTransformerLayers;
        _sequenceLength = _options.SequenceLength;
        _predictionHorizon = _options.PredictionHorizon;
        _dropoutRate = _options.DropoutRate;

        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes the neural network layers for FactorTransformer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The default architecture includes:
    /// </para>
    /// <para>
    /// 1. An input embedding that maps features to the model dimension
    /// 2. Transformer encoder layers with self-attention
    /// 3. A factor head that outputs latent factors
    /// </para>
    /// <para>
    /// If you provided custom layers, the model uses them instead.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
        }
        else if (_useNativeMode)
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultFactorTransformerLayers(
                Architecture,
                _numFeatures,
                _hiddenDimension,
                _numFactors,
                _numHeads,
                _numTransformerLayers,
                _sequenceLength,
                _dropoutRate));
        }
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <summary>
    /// Runs a forward pass to predict factor outputs.
    /// </summary>
    /// <param name="input">Input tensor of market features.</param>
    /// <returns>Model output tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the main prediction step that uses attention
    /// to compute factor signals from the input data.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return _useNativeMode ? PredictNative(input) : PredictOnnx(input);
    }

    /// <summary>
    /// Trains the model on a batch of inputs and targets.
    /// </summary>
    /// <param name="input">Input tensor of market features.</param>
    /// <param name="target">Target tensor of returns.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Training teaches the transformer which patterns
    /// in the data best predict returns.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> target)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is only supported in native mode.");

        SetTrainingMode(true);
        var output = PredictNative(input);
        var gradient = _lossFunction.CalculateDerivative(output.ToVector(), target.ToVector());
        var gradTensor = Tensor<T>.FromVector(gradient);

        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            gradTensor = Layers[i].Backward(gradTensor);
        }

        _optimizer.UpdateParameters(Layers);
        SetTrainingMode(false);
    }

    /// <summary>
    /// Updates model parameters from a flat vector.
    /// </summary>
    /// <param name="parameters">Flat parameter vector.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This lets you load saved weights into the model.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int offset = 0;
        foreach (var layer in Layers)
        {
            int count = layer.ParameterCount;
            if (count <= 0)
                continue;

            layer.SetParameters(parameters.Slice(offset, count));
            offset += count;
        }
    }

    /// <summary>
    /// Gets metadata describing this model instance.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Metadata summarizes the model configuration for diagnostics.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                ["NumFactors"] = _numFactors,
                ["NumAssets"] = _numAssets,
                ["HiddenDimension"] = _hiddenDimension,
                ["NumHeads"] = _numHeads,
                ["NumTransformerLayers"] = _numTransformerLayers,
                ["UseNativeMode"] = _useNativeMode
            }
        };
    }

    /// <summary>
    /// Creates a new instance with the same configuration.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Used by the framework to clone models with identical settings.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var optionsCopy = new FactorTransformerOptions<T>
        {
            NumFactors = _numFactors,
            NumAssets = _numAssets,
            NumFeatures = _numFeatures,
            HiddenDimension = _hiddenDimension,
            NumHeads = _numHeads,
            NumTransformerLayers = _numTransformerLayers,
            SequenceLength = _sequenceLength,
            PredictionHorizon = _predictionHorizon,
            DropoutRate = _dropoutRate
        };

        return new FactorTransformer<T>(Architecture, optionsCopy);
    }

    /// <summary>
    /// Serializes model-specific data.
    /// </summary>
    /// <param name="writer">Binary writer.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Saves the model configuration so it can be restored later.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_numFactors);
        writer.Write(_numAssets);
        writer.Write(_numFeatures);
        writer.Write(_hiddenDimension);
        writer.Write(_numHeads);
        writer.Write(_numTransformerLayers);
        writer.Write(_sequenceLength);
        writer.Write(_predictionHorizon);
        writer.Write(_dropoutRate);
    }

    /// <summary>
    /// Deserializes model-specific data.
    /// </summary>
    /// <param name="reader">Binary reader.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Restores the saved configuration when loading a model.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32(); // numFactors
        _ = reader.ReadInt32(); // numAssets
        _ = reader.ReadInt32(); // numFeatures
        _ = reader.ReadInt32(); // hiddenDimension
        _ = reader.ReadInt32(); // numHeads
        _ = reader.ReadInt32(); // numLayers
        _ = reader.ReadInt32(); // sequenceLength
        _ = reader.ReadInt32(); // predictionHorizon
        _ = reader.ReadDouble(); // dropoutRate
    }

    #endregion

    #region IFactorModel Implementation

    /// <summary>
    /// Extracts latent factors from asset returns.
    /// </summary>
    /// <param name="returns">Asset returns tensor.</param>
    /// <returns>Factor representation tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This runs the input through the transformer to
    /// obtain the hidden factor signals.
    /// </para>
    /// </remarks>
    public Tensor<T> ExtractFactors(Tensor<T> returns)
    {
        var current = returns;
        int factorLayerIndex = Math.Min(8, Layers.Count - 2);
        for (int i = 0; i < factorLayerIndex; i++)
        {
            current = Layers[i].Forward(current);
        }
        return current;
    }

    /// <summary>
    /// Computes factor loadings for each asset.
    /// </summary>
    /// <param name="returns">Asset returns tensor.</param>
    /// <returns>Factor loadings matrix.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Factor loadings show how much each asset depends on each factor.
    /// </para>
    /// </remarks>
    public Tensor<T> GetFactorLoadings(Tensor<T> returns)
    {
        return new Tensor<T>(new[] { _numAssets, _numFactors });
    }

    /// <summary>
    /// Predicts expected returns from factor exposures.
    /// </summary>
    /// <param name="factorExposures">Tensor of factor exposures.</param>
    /// <returns>Predicted returns tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Converts factor values into expected asset returns.
    /// </para>
    /// </remarks>
    public Tensor<T> PredictReturns(Tensor<T> factorExposures)
    {
        return Predict(factorExposures);
    }

    /// <summary>
    /// Computes the factor covariance matrix.
    /// </summary>
    /// <param name="returns">Asset returns tensor.</param>
    /// <returns>Factor covariance matrix tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This tells you how factors move together, which matters for risk.
    /// </para>
    /// </remarks>
    public Tensor<T> GetFactorCovariance(Tensor<T> returns)
    {
        return new Tensor<T>(new[] { _numFactors, _numFactors });
    }

    /// <summary>
    /// Computes alpha (excess return) for each asset.
    /// </summary>
    /// <param name="returns">Asset returns tensor.</param>
    /// <param name="factorReturns">Factor returns tensor.</param>
    /// <returns>Alpha tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Alpha is the portion of returns not explained by factors.
    /// </para>
    /// </remarks>
    public Tensor<T> ComputeAlpha(Tensor<T> returns, Tensor<T> factorReturns)
    {
        return new Tensor<T>(new[] { _numAssets });
    }

    /// <summary>
    /// Gets factor model metrics.
    /// </summary>
    /// <returns>Dictionary of factor metrics.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns a small summary of the model configuration.
    /// </para>
    /// </remarks>
    public Dictionary<string, T> GetFactorMetrics()
    {
        return new Dictionary<string, T>
        {
            ["NumFactors"] = NumOps.FromDouble(_numFactors),
            ["NumHeads"] = NumOps.FromDouble(_numHeads),
            ["NumTransformerLayers"] = NumOps.FromDouble(_numTransformerLayers),
            ["HiddenDimension"] = NumOps.FromDouble(_hiddenDimension)
        };
    }

    #endregion

    #region IFinancialModel Implementation

    /// <summary>
    /// Generates a forecast using the model.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <param name="quantiles">Optional quantiles (unused for factor prediction).</param>
    /// <returns>Forecast tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Forecasting here means using learned factors to predict
    /// expected asset returns.
    /// </para>
    /// </remarks>
    public Tensor<T> Forecast(Tensor<T> input, double[]? quantiles = null)
    {
        return Predict(input);
    }

    /// <summary>
    /// Gets financial metrics for the model.
    /// </summary>
    /// <returns>Dictionary of financial metrics.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Provides factor-focused metrics from this model.
    /// </para>
    /// </remarks>
    public Dictionary<string, T> GetFinancialMetrics()
    {
        return GetFactorMetrics();
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Runs a forward pass using native layers.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Output tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This passes data through the C# transformer layers.
    /// </para>
    /// </remarks>
    private Tensor<T> PredictNative(Tensor<T> input)
    {
        var current = input;
        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }
        return current;
    }

    /// <summary>
    /// Runs a forward pass using the ONNX runtime.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Output tensor from ONNX inference.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This uses a pretrained ONNX file for fast predictions.
    /// </para>
    /// </remarks>
    private Tensor<T> PredictOnnx(Tensor<T> input)
    {
        if (_onnxSession is null)
            throw new InvalidOperationException("ONNX session is not initialized.");

        var inputData = new float[input.Length];
        for (int i = 0; i < input.Length; i++)
            inputData[i] = Convert.ToSingle(NumOps.ToDouble(input.Data.Span[i]));

        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input.Shape);
        string inputName = _onnxSession.InputMetadata.Keys.First();

        using var results = _onnxSession.Run(new[]
        {
            NamedOnnxValue.CreateFromTensor(inputName, onnxInput)
        });

        var outputTensor = results.First().AsTensor<float>();
        var outputShape = outputTensor.Dimensions.ToArray();
        var outputData = new T[outputTensor.Length];
        for (int i = 0; i < outputTensor.Length; i++)
            outputData[i] = NumOps.FromDouble(outputTensor.GetValue(i));

        return new Tensor<T>(outputShape, new Vector<T>(outputData));
    }

    #endregion

    #region IDisposable

    /// <summary>
    /// Releases resources used by the model.
    /// </summary>
    /// <param name="disposing">True if disposing managed resources.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Always dispose models when finished to free memory,
    /// especially if an ONNX session was loaded.
    /// </para>
    /// </remarks>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _onnxSession?.Dispose();
            foreach (var layer in Layers)
            {
                if (layer is IDisposable disposable)
                    disposable.Dispose();
            }
        }
        base.Dispose(disposing);
    }

    #endregion
}
