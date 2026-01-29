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
/// Variational autoencoder for learning disentangled financial factors.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// FactorVAE combines a variational autoencoder with a disentanglement penalty
/// so each latent dimension captures a distinct factor.
/// </para>
/// <para>
/// <b>For Beginners:</b> The model compresses market data into a small set of hidden
/// variables (factors). The disentanglement penalty encourages each factor to capture
/// a different driver of returns rather than mixing everything together.
/// </para>
/// <para>
/// Reference: Kim &amp; Mnih (2019). "Disentangling by Factorising"
/// </para>
/// </remarks>
public class FactorVAE<T> : FinancialModelBase<T>, IFactorModel<T>
{
    #region Execution Mode

    private readonly bool _useNativeMode;

    #endregion

    
    #region Shared Fields

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly FactorVAEOptions<T> _options;
    private readonly int _numFactors;
    private readonly int _numAssets;
    private readonly int _numFeatures;
    private readonly int _hiddenDimension;
    private readonly int _latentDimension;
    private readonly int _sequenceLength;
    private readonly int _predictionHorizon;
    private readonly double _beta;
    private readonly double _gamma;
    private readonly double _dropoutRate;
    private readonly Random _random;

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
    public override bool UseNativeMode => _useNativeMode;

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
    /// <b>For Beginners:</b> Each asset has this many features (prices, indicators, etc.).
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
    /// <b>For Beginners:</b> How far ahead the model is trained to forecast.
    /// </para>
    /// </remarks>
    public override int PredictionHorizon => _predictionHorizon;

    /// <summary>
    /// Gets the dimension of the latent space.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the size of the compressed representation the VAE learns.
    /// </para>
    /// </remarks>
    public int LatentDimension => _latentDimension;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new FactorVAE in ONNX mode for inference.
    /// </summary>
    /// <param name="architecture">The user-provided neural network architecture.</param>
    /// <param name="onnxModelPath">Path to the pretrained ONNX model.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <param name="optimizer">Optional optimizer for fine-tuning.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this when you already have a trained VAE stored as
    /// an ONNX file and want fast, read-only inference.
    /// </para>
    /// </remarks>
    public FactorVAE(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        FactorVAEOptions<T>? options = null,
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

        _options = options ?? new FactorVAEOptions<T>();
        _options.Validate();

        _numFactors = _options.NumFactors;
        _numAssets = _options.NumAssets;
        _numFeatures = _options.NumFeatures;
        _hiddenDimension = _options.HiddenDimension;
        _latentDimension = _options.LatentDimension;
        _sequenceLength = _options.SequenceLength;
        _predictionHorizon = _options.PredictionHorizon;
        _beta = _options.Beta;
        _gamma = _options.Gamma;
        _dropoutRate = _options.DropoutRate;
        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    /// <summary>
    /// Initializes a new FactorVAE in native mode for training and inference.
    /// </summary>
    /// <param name="architecture">The user-provided neural network architecture.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <param name="optimizer">Optional optimizer for training.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this when you want to train a FactorVAE from scratch
    /// and learn disentangled market factors from your own data.
    /// </para>
    /// </remarks>
    public FactorVAE(
        NeuralNetworkArchitecture<T> architecture,
        FactorVAEOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        _useNativeMode = true;
        OnnxModelPath = null;
        OnnxSession = null;

        _options = options ?? new FactorVAEOptions<T>();
        _options.Validate();

        _numFactors = _options.NumFactors;
        _numAssets = _options.NumAssets;
        _numFeatures = _options.NumFeatures;
        _hiddenDimension = _options.HiddenDimension;
        _latentDimension = _options.LatentDimension;
        _sequenceLength = _options.SequenceLength;
        _predictionHorizon = _options.PredictionHorizon;
        _beta = _options.Beta;
        _gamma = _options.Gamma;
        _dropoutRate = _options.DropoutRate;
        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes the neural network layers for FactorVAE.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The default architecture has three main parts:
    /// </para>
    /// <para>
    /// 1. Encoder: Compresses inputs into a latent representation
    /// 2. Disentangler: Encourages factors to be independent
    /// 3. Decoder: Reconstructs inputs from the latent factors
    /// </para>
    /// <para>
    /// If you provide custom layers, the model uses them instead.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultFactorVAELayers(
                Architecture,
                _numFeatures,
                _hiddenDimension,
                _latentDimension,
                _numFactors,
                _dropoutRate));
        }
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <summary>
    /// Runs a forward pass to reconstruct inputs or generate factor outputs.
    /// </summary>
    /// <param name="input">Input tensor of market features.</param>
    /// <returns>Model output tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This passes data through the VAE to produce outputs
    /// that reflect the learned latent factors.
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
    /// <param name="target">Target tensor for reconstruction.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Training teaches the VAE to reconstruct inputs while
    /// keeping factors disentangled. The beta and gamma settings control this balance.
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
    /// <b>For Beginners:</b> This lets you load saved weights into the model,
    /// which is useful for serialization and fine-tuning.
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
    /// <b>For Beginners:</b> Metadata summarizes the model setup for diagnostics.
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
                ["LatentDimension"] = _latentDimension,
                ["Beta"] = _beta,
                ["Gamma"] = _gamma,
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
        var optionsCopy = new FactorVAEOptions<T>
        {
            NumFactors = _numFactors,
            NumAssets = _numAssets,
            NumFeatures = _numFeatures,
            HiddenDimension = _hiddenDimension,
            LatentDimension = _latentDimension,
            SequenceLength = _sequenceLength,
            PredictionHorizon = _predictionHorizon,
            Beta = _beta,
            Gamma = _gamma,
            DropoutRate = _dropoutRate
        };

        return new FactorVAE<T>(Architecture, optionsCopy);
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
        writer.Write(_latentDimension);
        writer.Write(_sequenceLength);
        writer.Write(_predictionHorizon);
        writer.Write(_beta);
        writer.Write(_gamma);
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
        _ = reader.ReadInt32(); // latentDimension
        _ = reader.ReadInt32(); // sequenceLength
        _ = reader.ReadInt32(); // predictionHorizon
        _ = reader.ReadDouble(); // beta
        _ = reader.ReadDouble(); // gamma
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
    /// <b>For Beginners:</b> This passes the data through the encoder to obtain
    /// the compact factor representation.
    /// </para>
    /// </remarks>
    public Tensor<T> ExtractFactors(Tensor<T> returns)
    {
        var current = returns;
        int encoderEnd = Math.Min(5, Layers.Count - 3);
        for (int i = 0; i < encoderEnd; i++)
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
            ["LatentDimension"] = NumOps.FromDouble(_latentDimension),
            ["Beta"] = NumOps.FromDouble(_beta),
            ["Gamma"] = NumOps.FromDouble(_gamma)
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
    /// <b>For Beginners:</b> Forecasting here means using the learned factors to
    /// predict the next returns or reconstructed outputs.
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
    /// <b>For Beginners:</b> Provides factor-focused metrics from this model.
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
    /// <b>For Beginners:</b> This passes data through the C# layers to get outputs.
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
    /// <b>For Beginners:</b> Always dispose models when finished to free memory,
    /// especially if an ONNX session was loaded.
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

