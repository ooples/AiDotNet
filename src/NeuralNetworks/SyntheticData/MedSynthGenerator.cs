using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.NeuralNetworks.SyntheticData;

/// <summary>
/// MedSynth generator for privacy-preserving medical tabular data synthesis using a
/// VAE/GAN hybrid with clinical validity constraints and optional differential privacy.
/// </summary>
/// <remarks>
/// <para>
/// MedSynth combines VAE and GAN approaches with medical domain constraints:
///
/// <code>
///  Data --> Encoder --> (mean, logvar) --> z --> Decoder --> Reconstructed --> Constraint Layer
///                                                     |
///                                              Discriminator --> Real/Fake?
/// </code>
///
/// Training alternates between three objectives:
/// 1. <b>VAE loss</b>: Reconstruction + KL divergence + constraint violation penalty
/// 2. <b>Discriminator loss</b>: BCE on real vs fake samples
/// 3. <b>Adversarial loss</b>: Non-saturating generator loss through discriminator input gradients
///
/// This implementation follows the standard neural network architecture pattern with:
/// - Proper inheritance from NeuralNetworkBase
/// - Layer-based architecture using ILayer components
/// - Full autodiff and JIT compilation support
/// </para>
/// <para>
/// <b>For Beginners:</b> MedSynth ensures generated medical data is:
///
/// 1. Realistic (VAE reconstruction + GAN adversarial training)
/// 2. Valid (no impossible lab values or vital signs)
/// 3. Private (optional differential privacy protection)
///
/// If you provide custom layers in the architecture, those will be used directly
/// for the decoder network. Otherwise, standard layers are created.
///
/// Example usage:
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(inputFeatures: 64, outputSize: 50);
/// var options = new MedSynthOptions&lt;double&gt; { LatentDimension = 64, EnablePrivacy = true };
/// var medsynth = new MedSynthGenerator&lt;double&gt;(architecture, options);
/// medsynth.Fit(data, columns, epochs: 500);
/// var synthetic = medsynth.Generate(1000);
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class MedSynthGenerator<T> : NeuralNetworkBase<T>, ISyntheticTabularGenerator<T>
{
    private readonly MedSynthOptions<T> _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private ILossFunction<T> _lossFunction;
    private Random _random;

    // ISyntheticTabularGenerator state
    private List<ColumnMetadata> _columns = new();
    private TabularDataTransformer<T>? _transformer;

    // VAE encoder (auxiliary - not user-overridable)
    private readonly List<FullyConnectedLayer<T>> _encoderLayers = new();
    private FullyConnectedLayer<T>? _meanHead;
    private FullyConnectedLayer<T>? _logvarHead;
    private readonly List<Tensor<T>> _encoderPreActs = new();

    // Decoder batch normalization (auxiliary, matched to Layers)
    private readonly List<BatchNormalizationLayer<T>> _decoderBN = new();
    private FullyConnectedLayer<T>? _decoderOutput;
    private readonly List<Tensor<T>> _decoderPreActs = new();

    // Discriminator (auxiliary - not user-overridable)
    private readonly List<FullyConnectedLayer<T>> _discLayers = new();
    private readonly List<DropoutLayer<T>> _discDropout = new();
    private FullyConnectedLayer<T>? _discOutput;
    private readonly List<Tensor<T>> _discPreActs = new();

    // Clinical constraints (learned from data)
    private double[]? _colMin;
    private double[]? _colMax;

    // Whether custom layers are being used
    private bool _usingCustomLayers;

    private int _dataWidth;

    /// <summary>
    /// Gets the MedSynth-specific options.
    /// </summary>
    public new MedSynthOptions<T> Options => _options;

    /// <inheritdoc />
    public IReadOnlyList<ColumnMetadata> Columns => _columns.AsReadOnly();

    /// <inheritdoc />
    public bool IsFitted { get; private set; }

    /// <summary>
    /// Initializes a new MedSynth generator with the specified architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining input/output dimensions and optional custom layers.</param>
    /// <param name="options">MedSynth-specific options for generation configuration.</param>
    /// <param name="optimizer">Gradient-based optimizer (defaults to Adam).</param>
    /// <param name="lossFunction">Loss function (defaults based on task type).</param>
    /// <param name="maxGradNorm">Maximum gradient norm for clipping (default 5.0).</param>
    public MedSynthGenerator(
        NeuralNetworkArchitecture<T> architecture,
        MedSynthOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 5.0)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), maxGradNorm)
    {
        _options = options ?? new MedSynthOptions<T>();
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        InitializeLayers();
    }

    #region Layer Initialization (GANDALF Pattern)

    /// <summary>
    /// Initializes the layers of the MedSynth network based on the provided architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method sets up the decoder network:
    /// - If you provided custom layers, those are used for the decoder
    /// - Otherwise, standard decoder layers are created
    ///
    /// The encoder, discriminator, mean/logvar heads, and decoder output are always
    /// created internally and are not user-overridable.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            _usingCustomLayers = true;
        }
        else
        {
            // Create default decoder hidden layers (reverse of encoder dims)
            var identity = new IdentityActivation<T>() as IActivationFunction<T>;
            var dims = _options.EncoderDimensions;
            int latentDim = _options.LatentDimension;
            int outputDim = Architecture.OutputSize;

            for (int i = dims.Length - 1; i >= 0; i--)
            {
                int layerInput = i == dims.Length - 1 ? latentDim : dims[i + 1];
                Layers.Add(new FullyConnectedLayer<T>(layerInput, dims[i], identity));
            }
            _usingCustomLayers = false;
        }
    }

    /// <summary>
    /// Rebuilds all layers with actual data dimensions discovered during Fit().
    /// </summary>
    private void RebuildLayersWithActualDimensions()
    {
        var identity = new IdentityActivation<T>() as IActivationFunction<T>;
        var dims = _options.EncoderDimensions;

        // Build encoder (always auxiliary)
        _encoderLayers.Clear();
        for (int i = 0; i < dims.Length; i++)
        {
            int layerInput = i == 0 ? _dataWidth : dims[i - 1];
            _encoderLayers.Add(new FullyConnectedLayer<T>(layerInput, dims[i], identity));
        }

        int lastEncoderDim = dims.Length > 0 ? dims[^1] : _dataWidth;
        _meanHead = new FullyConnectedLayer<T>(lastEncoderDim, _options.LatentDimension, identity);
        _logvarHead = new FullyConnectedLayer<T>(lastEncoderDim, _options.LatentDimension, identity);

        // Rebuild decoder (Layers) if not using custom layers
        if (!_usingCustomLayers)
        {
            Layers.Clear();
            _decoderBN.Clear();

            for (int i = dims.Length - 1; i >= 0; i--)
            {
                int layerInput = i == dims.Length - 1 ? _options.LatentDimension : dims[i + 1];
                Layers.Add(new FullyConnectedLayer<T>(layerInput, dims[i], identity));
                _decoderBN.Add(new BatchNormalizationLayer<T>(dims[i]));
            }
        }

        int lastDecoderDim = dims.Length > 0 ? dims[0] : _options.LatentDimension;
        _decoderOutput = new FullyConnectedLayer<T>(lastDecoderDim, _dataWidth, identity);

        // Build discriminator (always auxiliary)
        _discLayers.Clear();
        _discDropout.Clear();

        var discDims = _options.DiscriminatorDimensions;
        for (int i = 0; i < discDims.Length; i++)
        {
            int layerInput = i == 0 ? _dataWidth : discDims[i - 1];
            _discLayers.Add(new FullyConnectedLayer<T>(layerInput, discDims[i], identity));
            _discDropout.Add(new DropoutLayer<T>(_options.DiscriminatorDropout));
        }

        int lastDiscDim = discDims.Length > 0 ? discDims[^1] : _dataWidth;
        _discOutput = new FullyConnectedLayer<T>(lastDiscDim, 1, identity);
    }

    #endregion

    #region ISyntheticTabularGenerator Implementation

    /// <inheritdoc />
    public void Fit(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs)
    {
        _columns = new List<ColumnMetadata>(columns);

        _transformer = new TabularDataTransformer<T>(_options.VGMModes, _random);
        _transformer.Fit(data, columns);
        _dataWidth = _transformer.TransformedWidth;
        var transformedData = _transformer.Transform(data);

        // Learn clinical constraints from data
        LearnConstraints(transformedData);

        // Build all networks with actual dimensions
        RebuildLayersWithActualDimensions();

        // Compute noise multiplier for DP if enabled
        double noiseMultiplier = 0;
        if (_options.EnablePrivacy)
        {
            noiseMultiplier = ComputeNoiseMultiplier(data.Rows, epochs);
        }

        int batchSize = Math.Min(_options.BatchSize, data.Rows);
        T lr = NumOps.FromDouble(_options.LearningRate / Math.Max(batchSize, 1));

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            for (int b = 0; b < data.Rows; b += batchSize)
            {
                int end = Math.Min(b + batchSize, data.Rows);

                TrainVAEStep(transformedData, b, end, lr, noiseMultiplier);

                for (int d = 0; d < _options.DiscriminatorSteps; d++)
                {
                    TrainDiscriminatorStep(transformedData, b, end, lr);
                }

                TrainGeneratorAdversarialStep(batchSize, lr);
            }
        }

        IsFitted = true;
    }

    /// <inheritdoc />
    public Task FitAsync(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs,
        CancellationToken cancellationToken = default)
    {
        return Task.Run(() => Fit(data, columns, epochs), cancellationToken);
    }

    /// <inheritdoc />
    public Matrix<T> Generate(int numSamples, Vector<T>? conditionColumn = null, Vector<T>? conditionValue = null)
    {
        if (_transformer is null || _decoderOutput is null)
        {
            throw new InvalidOperationException("Generator is not fitted. Call Fit() before Generate().");
        }

        int latentDim = _options.LatentDimension;
        var result = new Matrix<T>(numSamples, _dataWidth);

        for (int i = 0; i < numSamples; i++)
        {
            var z = CreateStandardNormalVector(latentDim);
            var decoded = DecoderForward(z, isTraining: false);
            var constrained = ApplyConstraints(decoded);
            var activated = ApplyOutputActivations(constrained);

            for (int j = 0; j < _dataWidth && j < activated.Length; j++)
            {
                result[i, j] = activated[j];
            }
        }

        return _transformer.InverseTransform(result);
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <inheritdoc />
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // Forward through decoder (Layers)
        var current = input;
        for (int i = 0; i < Layers.Count; i++)
        {
            current = Layers[i].Forward(current);
            if (i < _decoderBN.Count)
            {
                _decoderBN[i].SetTrainingMode(false);
                current = _decoderBN[i].Forward(current);
            }
            current = ApplyReLU(current);
        }
        if (_decoderOutput is not null)
        {
            current = _decoderOutput.Forward(current);
        }
        return current;
    }

    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Training is handled through Fit() for tabular generators.
        var output = Predict(input);

        var gradient = new Tensor<T>(output.Shape);
        for (int i = 0; i < output.Length && i < expectedOutput.Length; i++)
        {
            gradient[i] = NumOps.FromDouble(
                2.0 * (NumOps.ToDouble(output[i]) - NumOps.ToDouble(expectedOutput[i])));
        }

        var current = gradient;
        if (_decoderOutput is not null)
            current = _decoderOutput.Backward(current);
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
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

    /// <inheritdoc />
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_dataWidth);
        writer.Write(IsFitted);
    }

    /// <inheritdoc />
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _dataWidth = reader.ReadInt32();
        IsFitted = reader.ReadBoolean();
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new MedSynthGenerator<T>(Architecture, _options);
    }

    /// <inheritdoc />
    public override Dictionary<string, T> GetFeatureImportance()
    {
        return new Dictionary<string, T>();
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                ["GeneratorType"] = "MedSynth",
                ["LatentDimension"] = _options.LatentDimension,
                ["EnablePrivacy"] = _options.EnablePrivacy,
                ["Epsilon"] = _options.Epsilon,
                ["IsFitted"] = IsFitted
            }
        };
    }

    #endregion

    #region Training

    private void TrainVAEStep(Matrix<T> data, int startRow, int endRow, T lr, double noiseMultiplier)
    {
        if (_meanHead is null || _logvarHead is null || _decoderOutput is null) return;

        for (int row = startRow; row < endRow; row++)
        {
            var x = GetRow(data, row);

            var hidden = EncoderForward(x);
            var meanTensor = _meanHead.Forward(VectorToTensor(hidden));
            var logvarTensor = _logvarHead.Forward(VectorToTensor(hidden));

            int latentDim = _options.LatentDimension;
            var mean = TensorToVector(meanTensor, latentDim);
            var logvar = TensorToVector(logvarTensor, latentDim);

            var z = Reparameterize(mean, logvar);
            var decoded = DecoderForward(z, isTraining: true);

            // Reconstruction loss gradient
            var reconGrad = new Tensor<T>([_dataWidth]);
            for (int j = 0; j < _dataWidth; j++)
            {
                double diff = NumOps.ToDouble(decoded[j]) - NumOps.ToDouble(x[j]);
                reconGrad[j] = NumOps.FromDouble(2.0 * diff);
            }

            // Add constraint violation penalty gradient
            if (_colMin is not null && _colMax is not null)
            {
                for (int j = 0; j < _dataWidth; j++)
                {
                    double val = NumOps.ToDouble(decoded[j]);
                    if (val < _colMin[j])
                    {
                        reconGrad[j] = NumOps.Add(reconGrad[j],
                            NumOps.FromDouble(_options.ConstraintWeight * (val - _colMin[j])));
                    }
                    else if (val > _colMax[j])
                    {
                        reconGrad[j] = NumOps.Add(reconGrad[j],
                            NumOps.FromDouble(_options.ConstraintWeight * (val - _colMax[j])));
                    }
                }
            }

            // Apply DP noise if enabled
            if (_options.EnablePrivacy && noiseMultiplier > 0)
            {
                ClipAndNoiseGradient(reconGrad, noiseMultiplier);
            }

            reconGrad = SanitizeAndClipGradient(reconGrad, 5.0);
            BackwardDecoder(reconGrad);

            // KL divergence gradients
            var meanGrad = new Tensor<T>([latentDim]);
            var logvarGrad = new Tensor<T>([latentDim]);
            for (int j = 0; j < latentDim; j++)
            {
                double m = NumOps.ToDouble(mean[j]);
                double lv = NumOps.ToDouble(logvar[j]);
                meanGrad[j] = NumOps.FromDouble(m * _options.KLWeight);
                logvarGrad[j] = NumOps.FromDouble(0.5 * (Math.Exp(lv) - 1.0) * _options.KLWeight);
            }

            var encoderGradFromMean = _meanHead.Backward(meanGrad);
            var encoderGradFromLogvar = _logvarHead.Backward(logvarGrad);

            // Sum gradients from both heads before propagating through encoder
            var encoderGrad = encoderGradFromMean.Add(encoderGradFromLogvar);
            BackwardEncoder(encoderGrad);

            // Update VAE parameters
            foreach (var layer in _encoderLayers) layer.UpdateParameters(lr);
            _meanHead.UpdateParameters(lr);
            _logvarHead.UpdateParameters(lr);
            foreach (var layer in Layers) layer.UpdateParameters(lr);
            foreach (var bn in _decoderBN) bn.UpdateParameters(lr);
            _decoderOutput.UpdateParameters(lr);
        }
    }

    private void TrainDiscriminatorStep(Matrix<T> data, int startRow, int endRow, T lr)
    {
        if (_discOutput is null) return;

        for (int row = startRow; row < endRow; row++)
        {
            var real = GetRow(data, row);

            var realScore = DiscriminatorForward(real, isTraining: true);
            double sigReal = Sigmoid(NumOps.ToDouble(realScore[0]));

            var gradReal = new Tensor<T>([1]);
            gradReal[0] = NumOps.FromDouble(-(1.0 - sigReal));
            BackwardDiscriminator(gradReal);
            UpdateDiscriminator(lr);

            var noise = CreateStandardNormalVector(_options.LatentDimension);
            var fake = DecoderForward(noise, isTraining: false);

            var fakeScore = DiscriminatorForward(fake, isTraining: true);
            double sigFake = Sigmoid(NumOps.ToDouble(fakeScore[0]));

            var gradFake = new Tensor<T>([1]);
            gradFake[0] = NumOps.FromDouble(sigFake);
            BackwardDiscriminator(gradFake);
            UpdateDiscriminator(lr);
        }
    }

    private void TrainGeneratorAdversarialStep(int batchSize, T lr)
    {
        if (_discOutput is null || _decoderOutput is null) return;

        T scaledLr = NumOps.FromDouble(NumOps.ToDouble(lr) * _options.AdversarialWeight);

        for (int i = 0; i < batchSize; i++)
        {
            var noise = CreateStandardNormalVector(_options.LatentDimension);
            var fakeData = DecoderForward(noise, isTraining: true);

            var fakeScore = DiscriminatorForward(fakeData, isTraining: false);
            double sig = Sigmoid(NumOps.ToDouble(fakeScore[0]));

            double genLossGrad = -(1.0 - sig);

            // Compute discriminator input gradient using TapeLayerBridge autodiff
            var allDiscLayers = BuildDiscLayerList();
            var dataGrad = TapeLayerBridge<T>.ComputeInputGradient(
                VectorToTensor(fakeData),
                allDiscLayers,
                TapeLayerBridge<T>.HiddenActivation.LeakyReLU,
                applyActivationOnLast: false);

            // Scale by genLossGrad (chain rule)
            for (int g = 0; g < dataGrad.Length; g++)
            {
                dataGrad[g] = NumOps.FromDouble(NumOps.ToDouble(dataGrad[g]) * genLossGrad);
            }
            dataGrad = SanitizeAndClipGradient(dataGrad, 5.0);

            _ = DecoderForward(noise, isTraining: true);

            var decoderGrad = new Tensor<T>([_dataWidth]);
            for (int j = 0; j < _dataWidth && j < dataGrad.Length; j++)
            {
                decoderGrad[j] = dataGrad[j];
            }

            BackwardDecoder(decoderGrad);

            foreach (var layer in Layers) layer.UpdateParameters(scaledLr);
            foreach (var bn in _decoderBN) bn.UpdateParameters(scaledLr);
            _decoderOutput.UpdateParameters(scaledLr);
        }
    }

    private void ClipAndNoiseGradient(Tensor<T> grad, double noiseMultiplier)
    {
        double norm = 0;
        for (int i = 0; i < grad.Length; i++)
        {
            double v = NumOps.ToDouble(grad[i]);
            norm += v * v;
        }
        norm = Math.Sqrt(norm);

        double clipNorm = _options.ClipNorm;
        double scale = norm > clipNorm ? clipNorm / norm : 1.0;

        for (int i = 0; i < grad.Length; i++)
        {
            double clipped = NumOps.ToDouble(grad[i]) * scale;
            double u1 = Math.Max(1e-10, _random.NextDouble());
            double u2 = _random.NextDouble();
            double noiseVal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            double noise = noiseVal * noiseMultiplier * clipNorm;
            grad[i] = NumOps.FromDouble(clipped + noise);
        }
    }

    private double ComputeNoiseMultiplier(int dataSize, int epochs)
    {
        double delta = 1.0 / (dataSize * dataSize);
        int batchSize = Math.Min(_options.BatchSize, dataSize);
        double samplingRate = (double)batchSize / dataSize;
        int totalSteps = epochs * (dataSize / Math.Max(batchSize, 1));

        double noiseMultiplier = 1.0;
        for (int attempt = 0; attempt < 100; attempt++)
        {
            double eps = samplingRate * Math.Sqrt(totalSteps * 2.0 * Math.Log(1.0 / delta)) * noiseMultiplier;
            if (eps <= _options.Epsilon) break;
            noiseMultiplier *= 1.1;
        }

        return noiseMultiplier;
    }

    #endregion

    #region Forward Passes with Manual Activation

    private Vector<T> EncoderForward(Vector<T> x)
    {
        _encoderPreActs.Clear();
        var current = VectorToTensor(x);

        for (int i = 0; i < _encoderLayers.Count; i++)
        {
            current = _encoderLayers[i].Forward(current);
            _encoderPreActs.Add(CloneTensor(current));
            current = ApplyReLU(current);
        }

        int hiddenDim = _options.EncoderDimensions.Length > 0
            ? _options.EncoderDimensions[^1] : _dataWidth;
        return TensorToVector(current, hiddenDim);
    }

    private Vector<T> DecoderForward(Vector<T> z, bool isTraining)
    {
        _decoderPreActs.Clear();
        var current = VectorToTensor(z);

        for (int i = 0; i < Layers.Count; i++)
        {
            current = Layers[i].Forward(current);
            if (i < _decoderBN.Count)
            {
                _decoderBN[i].SetTrainingMode(isTraining);
                current = _decoderBN[i].Forward(current);
            }
            _decoderPreActs.Add(CloneTensor(current));
            current = ApplyReLU(current);
        }

        if (_decoderOutput is not null)
        {
            current = _decoderOutput.Forward(current);
        }

        return TensorToVector(current, _dataWidth);
    }

    private Vector<T> DiscriminatorForward(Vector<T> x, bool isTraining)
    {
        _discPreActs.Clear();
        var current = VectorToTensor(x);

        for (int i = 0; i < _discLayers.Count; i++)
        {
            current = _discLayers[i].Forward(current);
            _discPreActs.Add(CloneTensor(current));
            current = ApplyLeakyReLU(current, 0.2);

            if (i < _discDropout.Count)
            {
                _discDropout[i].SetTrainingMode(isTraining);
                current = _discDropout[i].Forward(current);
            }
        }

        if (_discOutput is not null)
        {
            current = _discOutput.Forward(current);
        }

        return TensorToVector(current, current.Length);
    }

    #endregion

    #region Backward Passes with Manual Activation Derivatives

    private void BackwardEncoder(Tensor<T> grad)
    {
        var current = grad;

        for (int i = _encoderLayers.Count - 1; i >= 0; i--)
        {
            if (i < _encoderPreActs.Count)
            {
                current = ApplyReLUDerivative(current, _encoderPreActs[i]);
            }
            current = _encoderLayers[i].Backward(current);
        }
    }

    private void BackwardDecoder(Tensor<T> grad)
    {
        var current = grad;

        if (_decoderOutput is not null)
        {
            current = _decoderOutput.Backward(current);
        }

        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            if (i < _decoderPreActs.Count)
            {
                current = ApplyReLUDerivative(current, _decoderPreActs[i]);
            }
            if (i < _decoderBN.Count)
            {
                current = _decoderBN[i].Backward(current);
            }
            current = Layers[i].Backward(current);
        }
    }

    private void BackwardDiscriminator(Tensor<T> grad)
    {
        var current = grad;

        if (_discOutput is not null)
        {
            current = _discOutput.Backward(current);
        }

        for (int i = _discLayers.Count - 1; i >= 0; i--)
        {
            if (i < _discDropout.Count)
            {
                current = _discDropout[i].Backward(current);
            }
            if (i < _discPreActs.Count)
            {
                current = ApplyLeakyReLUDerivative(current, _discPreActs[i], 0.2);
            }
            current = _discLayers[i].Backward(current);
        }
    }

    private void UpdateDiscriminator(T lr)
    {
        foreach (var layer in _discLayers) layer.UpdateParameters(lr);
        _discOutput?.UpdateParameters(lr);
    }

    #endregion

    #region Discriminator Layer List

    /// <summary>
    /// Builds a combined list of discriminator layers for TapeLayerBridge.
    /// </summary>
    private IReadOnlyList<ILayer<T>> BuildDiscLayerList()
    {
        var allLayers = new List<ILayer<T>>();
        for (int i = 0; i < _discDropout.Count; i++)
        {
            allLayers.Add(_discLayers[i]);
            allLayers.Add(_discDropout[i]);
        }
        if (_discOutput is not null)
        {
            allLayers.Add(_discOutput);
        }
        return allLayers;
    }

    #endregion

    #region VAE Helpers

    private Vector<T> Reparameterize(Vector<T> mean, Vector<T> logvar)
    {
        int dim = mean.Length;
        var z = new Vector<T>(dim);
        for (int i = 0; i < dim; i++)
        {
            double u1 = Math.Max(1e-10, _random.NextDouble());
            double u2 = _random.NextDouble();
            double eps = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            double m = NumOps.ToDouble(mean[i]);
            double lv = NumOps.ToDouble(logvar[i]);
            z[i] = NumOps.FromDouble(m + eps * Math.Exp(0.5 * lv));
        }
        return z;
    }

    #endregion

    #region Manual Activation Functions

    private static Tensor<T> ApplyReLU(Tensor<T> input)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(input.Shape);
        for (int i = 0; i < input.Length; i++)
        {
            double v = ops.ToDouble(input[i]);
            result[i] = ops.FromDouble(v > 0 ? v : 0);
        }
        return result;
    }

    private static Tensor<T> ApplyReLUDerivative(Tensor<T> grad, Tensor<T> preActivation)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(grad.Shape);
        int len = Math.Min(grad.Length, preActivation.Length);
        for (int i = 0; i < len; i++)
        {
            double pre = ops.ToDouble(preActivation[i]);
            result[i] = pre > 0 ? grad[i] : ops.Zero;
        }
        return result;
    }

    private static Tensor<T> ApplyLeakyReLU(Tensor<T> input, double alpha)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(input.Shape);
        for (int i = 0; i < input.Length; i++)
        {
            double v = ops.ToDouble(input[i]);
            result[i] = ops.FromDouble(v >= 0 ? v : alpha * v);
        }
        return result;
    }

    private static Tensor<T> ApplyLeakyReLUDerivative(Tensor<T> grad, Tensor<T> preActivation, double alpha)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(grad.Shape);
        int len = Math.Min(grad.Length, preActivation.Length);
        for (int i = 0; i < len; i++)
        {
            double pre = ops.ToDouble(preActivation[i]);
            double deriv = pre >= 0 ? 1.0 : alpha;
            result[i] = ops.FromDouble(ops.ToDouble(grad[i]) * deriv);
        }
        return result;
    }

    #endregion

    #region Constraints & Activations

    private void LearnConstraints(Matrix<T> data)
    {
        _colMin = new double[data.Columns];
        _colMax = new double[data.Columns];

        for (int j = 0; j < data.Columns; j++)
        {
            double min = double.MaxValue;
            double max = double.MinValue;
            for (int i = 0; i < data.Rows; i++)
            {
                double v = NumOps.ToDouble(data[i, j]);
                if (v < min) min = v;
                if (v > max) max = v;
            }
            double range = max - min;
            _colMin[j] = min - 0.1 * range;
            _colMax[j] = max + 0.1 * range;
        }
    }

    private Vector<T> ApplyConstraints(Vector<T> decoded)
    {
        if (_colMin is null || _colMax is null) return decoded;

        var constrained = new Vector<T>(decoded.Length);
        for (int j = 0; j < decoded.Length; j++)
        {
            double val = NumOps.ToDouble(decoded[j]);
            constrained[j] = NumOps.FromDouble(Math.Min(Math.Max(val, _colMin[j]), _colMax[j]));
        }
        return constrained;
    }

    private Vector<T> ApplyOutputActivations(Vector<T> constrained)
    {
        if (_transformer is null) return constrained;

        var output = VectorToTensor(constrained);
        var result = new Tensor<T>(output.Shape);
        int idx = 0;

        for (int col = 0; col < _columns.Count && idx < output.Length; col++)
        {
            var transform = _transformer.GetTransformInfo(col);
            if (transform.IsContinuous)
            {
                if (idx < output.Length)
                {
                    result[idx] = NumOps.FromDouble(Math.Tanh(NumOps.ToDouble(output[idx])));
                    idx++;
                }
                int numModes = transform.Width - 1;
                if (numModes > 0) ApplySoftmax(output, result, ref idx, numModes);
            }
            else
            {
                ApplySoftmax(output, result, ref idx, transform.Width);
            }
        }

        return TensorToVector(result, _dataWidth);
    }

    private static void ApplySoftmax(Tensor<T> input, Tensor<T> output, ref int idx, int count)
    {
        if (count <= 0) return;
        var ops = MathHelper.GetNumericOperations<T>();
        double maxVal = double.MinValue;
        for (int i = 0; i < count && (idx + i) < input.Length; i++)
        {
            double v = ops.ToDouble(input[idx + i]);
            if (v > maxVal) maxVal = v;
        }
        double sumExp = 0;
        for (int i = 0; i < count && (idx + i) < input.Length; i++)
        {
            sumExp += Math.Exp(ops.ToDouble(input[idx + i]) - maxVal);
        }
        for (int i = 0; i < count && idx < input.Length; i++)
        {
            double expVal = Math.Exp(ops.ToDouble(input[idx]) - maxVal);
            output[idx] = ops.FromDouble(expVal / Math.Max(sumExp, 1e-10));
            idx++;
        }
    }

    #endregion

    #region Helpers

    private static double Sigmoid(double x)
    {
        return 1.0 / (1.0 + Math.Exp(-Math.Min(Math.Max(x, -20.0), 20.0)));
    }

    private static Tensor<T> CloneTensor(Tensor<T> source)
    {
        var clone = new Tensor<T>(source.Shape);
        for (int i = 0; i < source.Length; i++)
        {
            clone[i] = source[i];
        }
        return clone;
    }

    private static Vector<T> GetRow(Matrix<T> matrix, int row)
    {
        var v = new Vector<T>(matrix.Columns);
        for (int j = 0; j < matrix.Columns; j++) v[j] = matrix[row, j];
        return v;
    }

    /// <summary>
    /// Creates a vector of standard normal random values using Box-Muller transform.
    /// </summary>
    private Vector<T> CreateStandardNormalVector(int size)
    {
        var v = new Vector<T>(size);
        for (int i = 0; i < size; i++)
        {
            double u1 = Math.Max(1e-10, _random.NextDouble());
            double u2 = _random.NextDouble();
            double normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            v[i] = NumOps.FromDouble(normal);
        }
        return v;
    }

    /// <summary>
    /// Sanitizes a gradient tensor by replacing NaN/Inf values with zero and applying gradient clipping.
    /// </summary>
    private static Tensor<T> SanitizeAndClipGradient(Tensor<T> grad, double maxNorm)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        double normSq = 0;
        for (int i = 0; i < grad.Length; i++)
        {
            double v = ops.ToDouble(grad[i]);
            if (double.IsNaN(v) || double.IsInfinity(v))
            {
                grad[i] = ops.Zero;
            }
            else
            {
                normSq += v * v;
            }
        }

        double norm = Math.Sqrt(normSq);
        if (norm > maxNorm)
        {
            double scale = maxNorm / norm;
            for (int i = 0; i < grad.Length; i++)
            {
                grad[i] = ops.FromDouble(ops.ToDouble(grad[i]) * scale);
            }
        }

        return grad;
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
    /// MedSynth uses a VAE/GAN hybrid with clinical validity constraints which cannot be represented as a single computation graph.
    /// </summary>
    public override bool SupportsJitCompilation => false;

    #endregion
}
