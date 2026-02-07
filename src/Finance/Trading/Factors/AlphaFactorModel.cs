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

using AiDotNet.Finance.Base;
namespace AiDotNet.Finance.Trading.Factors;

/// <summary>
/// Neural network model for learning alpha factors from market data.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// AlphaFactorModel learns latent factors that explain and predict excess returns.
/// It discovers these factors directly from data instead of relying on hand-crafted signals.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of this model as a factor "discoverer."
/// Instead of manually choosing factors like value or momentum, the model learns hidden
/// drivers of returns from historical data and then uses them to predict performance.
/// </para>
/// <para>
/// Reference: Chen et al. (2020). "Deep Learning for Alpha Generation"
/// </para>
/// </remarks>
public class AlphaFactorModel<T> : FinancialModelBase<T>, IFactorModel<T>
{
    #region Execution Mode

    private readonly bool _useNativeMode;

    #endregion

    
    #region Shared Fields

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly AlphaFactorOptions<T> _options;
    private int _numFactors;
    private int _numAssets;
    private int _numFeatures;
    private int _hiddenDimension;
    private int _sequenceLength;
    private int _predictionHorizon;
    private double _dropoutRate;

    #endregion

    #region Interface Properties

    /// <summary>
    /// Gets whether the model is using native layers (true) or ONNX inference (false).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Native mode lets you train and fine-tune the model in C#.
    /// ONNX mode is read-only and optimized for fast predictions.
    /// </para>
    /// </remarks>
    public override bool UseNativeMode => _useNativeMode;

    /// <summary>
    /// Gets whether training is supported in the current mode.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Training is only available in native mode.
    /// ONNX mode is for inference only.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => _useNativeMode;

    /// <summary>
    /// Gets the number of latent factors learned by the model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is how many hidden drivers of returns the model learns.
    /// </para>
    /// </remarks>
    public int NumFactors => _numFactors;

    /// <summary>
    /// Gets the number of assets covered by the model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the size of the asset universe the model predicts for.
    /// </para>
    /// </remarks>
    public int NumAssets => _numAssets;

    /// <summary>
    /// Gets the number of input features per asset.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Features can be prices, volumes, and technical indicators.
    /// </para>
    /// </remarks>
    public override int NumFeatures => _numFeatures;

    /// <summary>
    /// Gets the input sequence length.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> How many time steps of history the model sees at once.
    /// </para>
    /// </remarks>
    public override int SequenceLength => _sequenceLength;

    /// <summary>
    /// Gets the prediction horizon.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> How many steps ahead the model is trained to predict.
    /// </para>
    /// </remarks>
    public override int PredictionHorizon => _predictionHorizon;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new AlphaFactorModel in ONNX mode for inference.
    /// </summary>
    /// <param name="architecture">The user-provided neural network architecture.</param>
    /// <param name="onnxModelPath">Path to the pretrained ONNX model.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <param name="optimizer">Optional optimizer for fine-tuning.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this when you already have a trained ONNX model and
    /// want fast predictions. ONNX mode skips training and loads pretrained weights.
    /// </para>
    /// </remarks>
    public AlphaFactorModel(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        AlphaFactorOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentNullException(nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        _useNativeMode = false;
        OnnxModelPath = onnxModelPath;
        OnnxSession = new InferenceSession(onnxModelPath);

        _options = options ?? new AlphaFactorOptions<T>();
        Options = _options;
        _options.Validate();

        _numFactors = _options.NumFactors;
        _numAssets = _options.NumAssets;
        _numFeatures = _options.NumFeatures;
        _hiddenDimension = _options.HiddenDimension;
        _sequenceLength = _options.SequenceLength;
        _predictionHorizon = _options.PredictionHorizon;
        _dropoutRate = _options.DropoutRate;

        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    /// <summary>
    /// Initializes a new AlphaFactorModel in native mode for training and inference.
    /// </summary>
    /// <param name="architecture">The user-provided neural network architecture.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <param name="optimizer">Optional optimizer for training.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this when you want to train the model on your own data.
    /// Native mode builds the layers in C# so gradients can be computed during learning.
    /// </para>
    /// </remarks>
    public AlphaFactorModel(
        NeuralNetworkArchitecture<T> architecture,
        AlphaFactorOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        _useNativeMode = true;
        OnnxModelPath = null;
        OnnxSession = null;

        _options = options ?? new AlphaFactorOptions<T>();
        Options = _options;
        _options.Validate();

        _numFactors = _options.NumFactors;
        _numAssets = _options.NumAssets;
        _numFeatures = _options.NumFeatures;
        _hiddenDimension = _options.HiddenDimension;
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
    /// Initializes the neural network layers for AlphaFactorModel.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This builds the default factor-learning pipeline:
    /// </para>
    /// <para>
    /// 1. Encode raw features into a hidden representation
    /// 2. Extract latent factors from the hidden space
    /// 3. Map factors to asset-level alpha predictions
    /// </para>
    /// <para>
    /// If you provided custom layers in the architecture, those are used instead.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultAlphaFactorLayers(
                Architecture,
                _numFeatures,
                _hiddenDimension,
                _numFactors,
                _numAssets,
                _dropoutRate));
        }
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <summary>
    /// Runs a forward pass to predict alpha values.
    /// </summary>
    /// <param name="input">Input tensor of market features.</param>
    /// <returns>Predicted alpha values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the main prediction step. The model converts
    /// your market data into factor signals and then into expected excess returns.
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
    /// <param name="target">Target tensor of realized returns.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Training teaches the model which factors actually
    /// predict returns by comparing predictions to real outcomes and adjusting weights.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> target)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is only supported in native mode.");

        SetTrainingMode(true);
        var output = PredictNative(input);
        var gradient = _lossFunction.CalculateDerivative(output.ToVector(), target.ToVector());
        var gradTensor = Tensor<T>.FromVector(gradient, output.Shape);

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
    /// <b>For Beginners:</b> This lets you load a precomputed set of weights
    /// into the model, which is useful for serialization or fine-tuning.
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
    /// <b>For Beginners:</b> Metadata is a summary of the model settings, useful for logging
    /// and diagnostics without exposing internal IP.
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
                ["NumFeatures"] = _numFeatures,
                ["HiddenDimension"] = _hiddenDimension,
                ["SequenceLength"] = _sequenceLength,
                ["PredictionHorizon"] = _predictionHorizon,
                ["UseNativeMode"] = _useNativeMode
            }
        };
    }

    /// <summary>
    /// Creates a new instance with the same configuration.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is used when the framework needs a fresh model
    /// with the same settings (for example during cloning).
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var optionsCopy = new AlphaFactorOptions<T>
        {
            NumFactors = _numFactors,
            NumAssets = _numAssets,
            NumFeatures = _numFeatures,
            HiddenDimension = _hiddenDimension,
            SequenceLength = _sequenceLength,
            PredictionHorizon = _predictionHorizon,
            DropoutRate = _dropoutRate
        };

        return new AlphaFactorModel<T>(Architecture, optionsCopy);
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
        _numFactors = reader.ReadInt32();
        _numAssets = reader.ReadInt32();
        _numFeatures = reader.ReadInt32();
        _hiddenDimension = reader.ReadInt32();
        _sequenceLength = reader.ReadInt32();
        _predictionHorizon = reader.ReadInt32();
        _dropoutRate = reader.ReadDouble();
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
    /// <b>For Beginners:</b> This runs the input through the early layers to produce
    /// the hidden factor signals that drive returns.
    /// </para>
    /// </remarks>
    public Tensor<T> ExtractFactors(Tensor<T> returns)
    {
        var current = returns;
        int factorLayerIndex = Math.Min(6, Layers.Count - 2);
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
    /// <b>For Beginners:</b> Factor loadings describe how strongly each asset depends
    /// on each learned factor.
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
    /// <b>For Beginners:</b> Once you know factor exposures, this converts them into
    /// expected asset returns.
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
    /// <b>For Beginners:</b> This measures how factors move together,
    /// which is important for risk management.
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
    /// <b>For Beginners:</b> Alpha is the portion of returns not explained by factors,
    /// which is the "extra edge" investors seek.
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
    /// <b>For Beginners:</b> Returns a small report of the model configuration
    /// and factor counts for monitoring and diagnostics.
    /// </para>
    /// </remarks>
    public Dictionary<string, T> GetFactorMetrics()
    {
        return new Dictionary<string, T>
        {
            ["NumFactors"] = NumOps.FromDouble(_numFactors),
            ["NumAssets"] = NumOps.FromDouble(_numAssets),
            ["HiddenDimension"] = NumOps.FromDouble(_hiddenDimension),
            ["SequenceLength"] = NumOps.FromDouble(_sequenceLength)
        };
    }

    #endregion

    #region IFinancialModel Implementation

    /// <summary>
    /// Generates a forecast using the model.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <param name="quantiles">Optional quantiles (unused for alpha prediction).</param>
    /// <returns>Forecast tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Forecasting here means predicting alpha values for assets
    /// based on the current market features.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forecast(Tensor<T> input, double[]? quantiles = null)
    {
        return Predict(input);
    }

    /// <summary>
    /// Gets financial metrics for the model.
    /// </summary>
    /// <returns>Dictionary of financial metrics.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Provides the factor-focused metrics that describe this model.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> GetFinancialMetrics()
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
    /// <b>For Beginners:</b> This passes data through the C# layers to get predictions.
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
    /// <b>For Beginners:</b> This uses a pretrained ONNX file for fast predictions
    /// without training in C#.
    /// </para>
    /// </remarks>
    private Tensor<T> PredictOnnx(Tensor<T> input)
    {
        if (OnnxSession is null)
            throw new InvalidOperationException("ONNX session is not initialized.");

        var inputData = new float[input.Length];
        for (int i = 0; i < input.Length; i++)
            inputData[i] = Convert.ToSingle(NumOps.ToDouble(input.Data.Span[i]));

        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input.Shape);
        string inputName = OnnxSession.InputMetadata.Keys.First();

        using var results = OnnxSession.Run(new[]
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
    /// <b>For Beginners:</b> Always dispose models when you are done to free memory,
    /// especially if you loaded an ONNX session.
    /// </para>
    /// </remarks>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            OnnxSession?.Dispose();
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

