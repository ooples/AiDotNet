using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.SyntheticDataGenerator)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Generation and Evaluation of Synthetic Patient Data",
    "https://arxiv.org/abs/1909.02662",
    Year = 2020,
    Authors = "Andrew Yale, Saloni Dash, Ritik Dutta, Isabelle Guyon, Adrien Pavao, Kristin P. Bennett")]
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
    /// <summary>
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    public MedSynthGenerator()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.OneDimensional,
            taskType: Enums.NeuralNetworkTaskType.Regression,
            inputSize: 10,
            outputSize: 10))
    {
    }

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
        Layers.Clear();

        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            _usingCustomLayers = true;
        }
        else
        {
            int dataWidth = Math.Max(1, Architecture.OutputSize);
            var allLayers = LayerHelper<T>.CreateDefaultMedSynthLayers(
                dataWidth, _options.LatentDimension,
                _options.EncoderDimensions, _options.DiscriminatorDimensions,
                _options.DiscriminatorDropout).ToList();
            Layers.AddRange(allLayers);
            _usingCustomLayers = false;
        }

        ExtractMedSynthLayerReferences();
    }

    /// <summary>
    /// Extracts private layer references as aliases from the unified Layers list.
    /// </summary>
    private void ExtractMedSynthLayerReferences()
    {
        _encoderLayers.Clear();
        _decoderBN.Clear();
        _discLayers.Clear();
        _discDropout.Clear();

        int idx = 0;
        var dims = _options.EncoderDimensions;
        var discDims = _options.DiscriminatorDimensions;

        // Encoder layers
        for (int i = 0; i < dims.Length && idx < Layers.Count; i++)
        {
            if (Layers[idx] is FullyConnectedLayer<T> enc)
                _encoderLayers.Add(enc);
            idx++;
        }

        // VAE heads
        if (idx < Layers.Count && Layers[idx] is FullyConnectedLayer<T> mean)
        { _meanHead = mean; idx++; }
        if (idx < Layers.Count && Layers[idx] is FullyConnectedLayer<T> logvar)
        { _logvarHead = logvar; idx++; }

        // Decoder layers (FC + BN pairs)
        for (int i = 0; i < dims.Length && idx < Layers.Count; i++)
        {
            idx++; // FC layer
            if (idx < Layers.Count && Layers[idx] is BatchNormalizationLayer<T> bn)
            { _decoderBN.Add(bn); idx++; }
        }

        // Decoder output
        if (idx < Layers.Count && Layers[idx] is FullyConnectedLayer<T> decOut)
        { _decoderOutput = decOut; idx++; }

        // Discriminator layers (FC + Dropout pairs)
        for (int i = 0; i < discDims.Length && idx < Layers.Count; i++)
        {
            if (Layers[idx] is FullyConnectedLayer<T> disc)
            { _discLayers.Add(disc); idx++; }
            if (idx < Layers.Count && Layers[idx] is DropoutLayer<T> drop)
            { _discDropout.Add(drop); idx++; }
        }

        // Discriminator output
        if (idx < Layers.Count && Layers[idx] is FullyConnectedLayer<T> discOut)
        { _discOutput = discOut; idx++; }
    }

    /// <summary>
    /// Rebuilds all layers with actual data dimensions discovered during Fit().
    /// </summary>
    private void RebuildLayersWithActualDimensions()
    {
        if (!_usingCustomLayers)
        {
            // Rebuild ALL layers via LayerHelper with actual data dimensions
            Layers.Clear();
            var allLayers = LayerHelper<T>.CreateDefaultMedSynthLayers(
                _dataWidth, _options.LatentDimension,
                _options.EncoderDimensions, _options.DiscriminatorDimensions,
                _options.DiscriminatorDropout).ToList();
            Layers.AddRange(allLayers);

            ExtractMedSynthLayerReferences();
        }
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


                for (int d = 0; d < _options.DiscriminatorSteps; d++)
                {
                    TrainDiscriminatorStep(transformedData, b, end, lr);
                }

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
        SetTrainingMode(true);
        try
        {
            TrainWithTape(input, expectedOutput, _optimizer);
        }
        finally
        {
            SetTrainingMode(false);
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
            UpdateDiscriminator(lr);

            var noise = CreateStandardNormalVector(_options.LatentDimension);
            var fake = DecoderForward(noise, isTraining: false);

            var fakeScore = DiscriminatorForward(fake, isTraining: true);
            double sigFake = Sigmoid(NumOps.ToDouble(fakeScore[0]));

            var gradFake = new Tensor<T>([1]);
            gradFake[0] = NumOps.FromDouble(sigFake);
            UpdateDiscriminator(lr);
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

    private void UpdateDiscriminator(T lr)
    {
        foreach (var layer in _discLayers) layer.UpdateParameters(lr);
        _discOutput?.UpdateParameters(lr);
    }

    #endregion

    #region Discriminator Layer List

    /// <summary>
    /// Builds a combined list of discriminator layers (dense + dropout + output)
    /// for gradient-penalty and related analyses.
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
        var result = new Tensor<T>(input._shape);
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
        var result = new Tensor<T>(grad._shape);
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
        var result = new Tensor<T>(input._shape);
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
        var result = new Tensor<T>(grad._shape);
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
        var result = new Tensor<T>(output._shape);
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
        int actualCount = Math.Min(count, input.Length - idx);
        if (actualCount <= 0) return;
        var slice = new Tensor<T>([actualCount]);
        input.Data.Span.Slice(idx, actualCount).CopyTo(slice.Data.Span);
        var result = AiDotNetEngine.Current.Softmax(slice, -1);
        result.Data.Span.CopyTo(output.Data.Span.Slice(idx, actualCount));
        idx += actualCount;
    }

    #endregion

    #region Helpers

    private static double Sigmoid(double x)
    {
        return 1.0 / (1.0 + Math.Exp(-Math.Min(Math.Max(x, -20.0), 20.0)));
    }

    private static Tensor<T> CloneTensor(Tensor<T> source)
    {
        var clone = new Tensor<T>(source._shape);
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

}
