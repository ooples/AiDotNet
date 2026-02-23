using AiDotNet.ActivationFunctions;
using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.NeuralNetworks.SyntheticData;

/// <summary>
/// TableGAN generator using a DCGAN-style architecture with classification and information
/// loss for high-quality synthetic tabular data generation.
/// </summary>
/// <remarks>
/// <para>
/// TableGAN optimizes three losses simultaneously:
///
/// <code>
///  Noise --> Generator (residual+BN) --> Fake Data --> Discriminator (WGAN-GP) --> score
///                                           |
///                                           |---> Classifier --> Label prediction (classification loss)
///                                           |
///                                           +---> Statistics --> Mean/Var match (information loss)
/// </code>
///
/// This implementation follows the standard neural network architecture pattern with:
/// - Proper inheritance from NeuralNetworkBase
/// - Layer-based architecture using ILayer components
/// - Full autodiff and JIT compilation support
/// </para>
/// <para>
/// <b>For Beginners:</b> TableGAN is like a regular GAN with two extra quality checks:
/// 1. <b>Adversarial loss</b>: Does the synthetic data look real? (WGAN-GP for stable training)
/// 2. <b>Classification loss</b>: Are label-feature relationships preserved?
/// 3. <b>Information loss</b>: Do the mean/variance statistics match the real data?
///
/// If you provide custom layers in the architecture, those will be used directly
/// for the generator network. If not, the network creates industry-standard
/// TableGAN layers based on the original research paper specifications.
///
/// Example usage:
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(inputFeatures: 10, outputSize: 10);
/// var options = new TableGANOptions&lt;double&gt;
/// {
///     EmbeddingDimension = 100,
///     LabelColumnIndex = 4,
///     Epochs = 300
/// };
/// var generator = new TableGANGenerator&lt;double&gt;(architecture, options);
/// generator.Fit(data, columns, epochs: 300);
/// var synthetic = generator.Generate(1000);
/// </code>
/// </para>
/// <para>
/// Reference: "Data Synthesis based on Generative Adversarial Networks" (Park et al., 2018)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TableGANGenerator<T> : NeuralNetworkBase<T>, ISyntheticTabularGenerator<T>
{
    private readonly TableGANOptions<T> _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private ILossFunction<T> _lossFunction;

    // Synthetic tabular data infrastructure
    private TabularDataTransformer<T>? _transformer;
    private List<ColumnMetadata> _columns = new();
    private int _dataWidth;
    private Random _random;

    // Generator batch normalization layers (auxiliary, paired with Layers)
    private readonly List<BatchNormalizationLayer<T>> _genBNLayers = new();

    // Discriminator layers (auxiliary, not user-overridable)
    private readonly List<FullyConnectedLayer<T>> _discLayers = new();
    private readonly List<DropoutLayer<T>> _discDropoutLayers = new();
    private readonly List<(int InputSize, int OutputSize)> _discLayerDims = new();

    // Classifier layers (auxiliary, not user-overridable)
    private readonly List<FullyConnectedLayer<T>> _classLayers = new();
    private FullyConnectedLayer<T>? _classOutput;

    // Cached pre-activations for proper backward passes
    private readonly List<Tensor<T>> _genPreActivations = new();
    private readonly List<Tensor<T>> _discPreActivations = new();

    // Real data statistics for information loss
    private Vector<T>? _realMean;
    private Vector<T>? _realVar;

    private int _numClasses;
    private bool _usingCustomLayers;

    /// <summary>
    /// Gets the TableGAN-specific options.
    /// </summary>
    public new TableGANOptions<T> Options => _options;

    /// <inheritdoc />
    public IReadOnlyList<ColumnMetadata> Columns => _columns.AsReadOnly();

    /// <inheritdoc />
    public bool IsFitted { get; private set; }

    /// <summary>
    /// Initializes a new TableGAN generator with the specified architecture.
    /// </summary>
    public TableGANGenerator(
        NeuralNetworkArchitecture<T> architecture,
        TableGANOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 5.0)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), maxGradNorm)
    {
        _options = options ?? new TableGANOptions<T>();
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        InitializeLayers();
    }

    #region Layer Initialization (GANDALF Pattern)

    protected override void InitializeLayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            _usingCustomLayers = true;
        }
        else
        {
            var identity = new IdentityActivation<T>() as IActivationFunction<T>;
            int inputDim = _options.EmbeddingDimension;
            var dims = _options.GeneratorDimensions;

            for (int i = 0; i < dims.Length; i++)
            {
                int layerInput = i == 0 ? inputDim : dims[i - 1] + inputDim;
                Layers.Add(new FullyConnectedLayer<T>(layerInput, dims[i], identity));
                _genBNLayers.Add(new BatchNormalizationLayer<T>(dims[i]));
            }

            int lastHidden = dims.Length > 0 ? dims[^1] + inputDim : inputDim;
            Layers.Add(new FullyConnectedLayer<T>(lastHidden, Architecture.OutputSize, identity));
            _usingCustomLayers = false;
        }
    }

    private void BuildDiscriminator()
    {
        _discLayers.Clear();
        _discDropoutLayers.Clear();
        _discLayerDims.Clear();

        var identity = new IdentityActivation<T>() as IActivationFunction<T>;
        var dims = _options.DiscriminatorDimensions;

        for (int i = 0; i < dims.Length; i++)
        {
            int layerInput = i == 0 ? _dataWidth : dims[i - 1];
            _discLayers.Add(new FullyConnectedLayer<T>(layerInput, dims[i], identity));
            _discLayerDims.Add((layerInput, dims[i]));
            _discDropoutLayers.Add(new DropoutLayer<T>(_options.DiscriminatorDropout));
        }

        int lastHidden = dims.Length > 0 ? dims[^1] : _dataWidth;
        _discLayers.Add(new FullyConnectedLayer<T>(lastHidden, 1, identity));
        _discLayerDims.Add((lastHidden, 1));
    }

    private void BuildClassifier(int numClasses)
    {
        _classLayers.Clear();

        var identity = new IdentityActivation<T>() as IActivationFunction<T>;
        var dims = _options.ClassifierDimensions;

        for (int i = 0; i < dims.Length; i++)
        {
            int layerInput = i == 0 ? _dataWidth : dims[i - 1];
            _classLayers.Add(new FullyConnectedLayer<T>(layerInput, dims[i], identity));
        }

        int lastDim = dims.Length > 0 ? dims[^1] : _dataWidth;
        _classOutput = new FullyConnectedLayer<T>(lastDim, numClasses, identity);
    }

    #endregion

    #region ISyntheticTabularGenerator Implementation

    /// <inheritdoc />
    public void Fit(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs)
    {
        _columns = columns.ToList();
        _transformer = new TabularDataTransformer<T>(_options.VGMModes, _random);
        _transformer.Fit(data, columns);
        _dataWidth = _transformer.TransformedWidth;
        var transformedData = _transformer.Transform(data);

        ComputeRealStatistics(transformedData);

        // Re-initialize generator layers with actual data width
        if (!_usingCustomLayers)
        {
            Layers.Clear();
            _genBNLayers.Clear();

            var identity = new IdentityActivation<T>() as IActivationFunction<T>;
            int inputDim = _options.EmbeddingDimension;
            var dims = _options.GeneratorDimensions;

            for (int i = 0; i < dims.Length; i++)
            {
                int layerInput = i == 0 ? inputDim : dims[i - 1] + inputDim;
                Layers.Add(new FullyConnectedLayer<T>(layerInput, dims[i], identity));
                _genBNLayers.Add(new BatchNormalizationLayer<T>(dims[i]));
            }

            int lastHidden = dims.Length > 0 ? dims[^1] + inputDim : inputDim;
            Layers.Add(new FullyConnectedLayer<T>(lastHidden, _dataWidth, identity));
        }

        BuildDiscriminator();

        _numClasses = 0;
        if (_options.LabelColumnIndex >= 0 && _options.LabelColumnIndex < columns.Count)
        {
            var labelCol = columns[_options.LabelColumnIndex];
            _numClasses = labelCol.IsCategorical ? Math.Max(2, labelCol.Categories.Count) : 1;
            BuildClassifier(_numClasses);
        }

        T lr = NumOps.FromDouble(_options.LearningRate);
        int batchSize = Math.Min(_options.BatchSize, data.Rows);
        int numBatches = Math.Max(1, data.Rows / batchSize);

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            for (int batch = 0; batch < numBatches; batch++)
            {
                for (int dStep = 0; dStep < _options.DiscriminatorSteps; dStep++)
                {
                    TrainDiscriminatorStep(transformedData, batchSize, lr);
                }
                TrainGeneratorStep(transformedData, batchSize, lr);
            }
        }

        IsFitted = true;
    }

    /// <inheritdoc />
    public async Task FitAsync(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs,
        CancellationToken cancellationToken = default)
    {
        await Task.Run(() =>
        {
            cancellationToken.ThrowIfCancellationRequested();
            Fit(data, columns, epochs);
        }, cancellationToken);
    }

    /// <inheritdoc />
    public Matrix<T> Generate(int numSamples, Vector<T>? conditionColumn = null, Vector<T>? conditionValue = null)
    {
        if (_transformer is null || Layers.Count == 0)
        {
            throw new InvalidOperationException("Generator is not fitted. Call Fit() first.");
        }

        var transformedRows = new Matrix<T>(numSamples, _dataWidth);

        for (int i = 0; i < numSamples; i++)
        {
            var noise = CreateStandardNormalVector(_options.EmbeddingDimension);
            var generated = GeneratorForward(noise);
            var activated = ApplyOutputActivations(generated);

            for (int j = 0; j < _dataWidth && j < activated.Length; j++)
            {
                transformedRows[i, j] = activated[j];
            }
        }

        return _transformer.InverseTransform(transformedRows);
    }

    #endregion

    #region Forward Passes

    private Tensor<T> GeneratorForward(Vector<T> noise)
    {
        _genPreActivations.Clear();
        var inputTensor = VectorToTensor(noise);

        if (_usingCustomLayers)
        {
            var current = inputTensor;
            foreach (var layer in Layers) current = layer.Forward(current);
            return current;
        }

        var h = inputTensor;
        for (int i = 0; i < Layers.Count - 1; i++)
        {
            if (i > 0) h = ConcatTensors(h, inputTensor);
            h = Layers[i].Forward(h);
            h = _genBNLayers[i].Forward(h);
            _genPreActivations.Add(CloneTensor(h));
            h = ApplyReLU(h);
        }

        h = ConcatTensors(h, inputTensor);
        h = Layers[^1].Forward(h);
        return h;
    }

    private Tensor<T> DiscriminatorForward(Tensor<T> input, bool isTraining)
    {
        _discPreActivations.Clear();
        var current = input;

        for (int i = 0; i < _discLayers.Count - 1; i++)
        {
            current = _discLayers[i].Forward(current);
            _discPreActivations.Add(CloneTensor(current));
            current = ApplyLeakyReLU(current);
            if (isTraining) current = _discDropoutLayers[i].Forward(current);
        }

        current = _discLayers[^1].Forward(current);
        return current;
    }

    private Tensor<T>? _lastClassOutput;

    private Vector<T> ClassifierForward(Vector<T> input)
    {
        var current = VectorToTensor(input);
        for (int i = 0; i < _classLayers.Count; i++)
        {
            current = _classLayers[i].Forward(current);
            current = ApplyReLU(current);
        }
        if (_classOutput is not null) current = _classOutput.Forward(current);
        _lastClassOutput = current;
        return TensorToVector(current, current.Length);
    }

    #endregion

    #region Training Steps

    private void TrainDiscriminatorStep(Matrix<T> transformedData, int batchSize, T learningRate)
    {
        T scaledLr = NumOps.FromDouble(NumOps.ToDouble(learningRate) / batchSize);

        for (int s = 0; s < batchSize; s++)
        {
            int rowIdx = _random.Next(transformedData.Rows);
            var realRow = GetRow(transformedData, rowIdx);
            var noise = CreateStandardNormalVector(_options.EmbeddingDimension);
            var fakeRaw = GeneratorForward(noise);
            fakeRaw = ApplyOutputActivations(fakeRaw);
            var fakeRow = TensorToVector(fakeRaw, _dataWidth);

            _ = DiscriminatorForward(VectorToTensor(fakeRow), isTraining: true);
            var fakeGrad = new Tensor<T>([1]);
            fakeGrad[0] = NumOps.One;
            BackwardDiscriminator(fakeGrad);
            UpdateDiscriminatorParameters(scaledLr);

            _ = DiscriminatorForward(VectorToTensor(realRow), isTraining: true);
            var realGrad = new Tensor<T>([1]);
            realGrad[0] = NumOps.Negate(NumOps.One);
            BackwardDiscriminator(realGrad);
            UpdateDiscriminatorParameters(scaledLr);

            ApplyGradientPenalty(realRow, fakeRow, scaledLr);
        }
    }

    private void TrainGeneratorStep(Matrix<T> transformedData, int batchSize, T learningRate)
    {
        T scaledLr = NumOps.FromDouble(NumOps.ToDouble(learningRate) / batchSize);

        for (int s = 0; s < batchSize; s++)
        {
            var noise = CreateStandardNormalVector(_options.EmbeddingDimension);
            var fakeRaw = GeneratorForward(noise);
            var fakeActivated = ApplyOutputActivations(fakeRaw);
            var fakeRow = TensorToVector(fakeActivated, _dataWidth);

            // Adversarial loss: compute dD/dInput using GradientTape autodiff, then negate
            var discInputGrad = TapeLayerBridge<T>.ComputeInputGradient(
                VectorToTensor(fakeRow),
                _discLayers,
                TapeLayerBridge<T>.HiddenActivation.LeakyReLU,
                applyActivationOnLast: false);
            for (int g = 0; g < discInputGrad.Length; g++)
            {
                discInputGrad[g] = NumOps.Negate(discInputGrad[g]);
            }

            // Information loss gradient (shape matches generator output)
            var infoGrad = new Tensor<T>(fakeRaw.Shape);
            if (_realMean is not null && _options.InformationWeight > 0)
            {
                for (int j = 0; j < _dataWidth && j < infoGrad.Length; j++)
                {
                    double fakeVal = NumOps.ToDouble(fakeRow[j]);
                    double meanDiff = fakeVal - NumOps.ToDouble(_realMean[j]);
                    infoGrad[j] = NumOps.FromDouble(2.0 * meanDiff * _options.InformationWeight);
                }
            }

            // Classification loss gradient (shape matches generator output)
            var classGrad = new Tensor<T>(fakeRaw.Shape);
            if (_classOutput is not null && _options.LabelColumnIndex >= 0 && _options.ClassificationWeight > 0)
            {
                int rowIdx = _random.Next(transformedData.Rows);
                var realRow = GetRow(transformedData, rowIdx);
                ComputeClassificationGradient(classGrad, fakeRow, realRow);
            }

            // Combine gradients (shape matches generator output)
            var totalGrad = new Tensor<T>(fakeRaw.Shape);
            for (int j = 0; j < _dataWidth && j < totalGrad.Length; j++)
            {
                double adv = j < discInputGrad.Length ? NumOps.ToDouble(discInputGrad[j]) : 0.0;
                double info = NumOps.ToDouble(infoGrad[j]);
                double cls = NumOps.ToDouble(classGrad[j]);
                totalGrad[j] = NumOps.FromDouble(adv + info + cls);
            }

            totalGrad = SafeGradient(totalGrad, 5.0);

            _ = GeneratorForward(noise);
            BackwardGenerator(totalGrad);
            UpdateGeneratorParameters(scaledLr);

            if (_classOutput is not null && _options.LabelColumnIndex >= 0)
            {
                int rowIdx = _random.Next(transformedData.Rows);
                var realRow = GetRow(transformedData, rowIdx);
                TrainClassifierOnSample(fakeRow, realRow, scaledLr);
            }
        }
    }

    private void ComputeClassificationGradient(Tensor<T> classGrad, Vector<T> fakeRow, Vector<T> realRow)
    {
        if (_classOutput is null) return;

        var classLogits = ClassifierForward(fakeRow);
        int labelIdx = Math.Min(_options.LabelColumnIndex, realRow.Length - 1);
        int targetClass = Math.Min(Math.Max(
            (int)Math.Round(NumOps.ToDouble(realRow[Math.Max(labelIdx, 0)])),
            0), classLogits.Length - 1);

        var softmax = ComputeSoftmax(classLogits);
        double totalClassGrad = 0;
        for (int c = 0; c < softmax.Length; c++)
        {
            double target = c == targetClass ? 1.0 : 0.0;
            totalClassGrad += (NumOps.ToDouble(softmax[c]) - target);
        }

        double perDimGrad = totalClassGrad * _options.ClassificationWeight / _dataWidth;
        for (int j = 0; j < _dataWidth && j < classGrad.Length; j++)
        {
            classGrad[j] = NumOps.Add(classGrad[j], NumOps.FromDouble(perDimGrad));
        }
    }

    private void TrainClassifierOnSample(Vector<T> fakeRow, Vector<T> realRow, T lr)
    {
        if (_classOutput is null) return;

        var classLogits = ClassifierForward(fakeRow);
        int labelIdx = Math.Min(_options.LabelColumnIndex, realRow.Length - 1);
        int targetClass = Math.Min(Math.Max(
            (int)Math.Round(NumOps.ToDouble(realRow[Math.Max(labelIdx, 0)])),
            0), classLogits.Length - 1);

        var softmax = ComputeSoftmax(classLogits);
        int[] gradShape = _lastClassOutput is not null ? _lastClassOutput.Shape : [softmax.Length];
        var classOutputGrad = new Tensor<T>(gradShape);
        for (int c = 0; c < softmax.Length && c < classOutputGrad.Length; c++)
        {
            double target = c == targetClass ? 1.0 : 0.0;
            classOutputGrad[c] = NumOps.FromDouble(NumOps.ToDouble(softmax[c]) - target);
        }

        var current = _classOutput.Backward(classOutputGrad);
        for (int i = _classLayers.Count - 1; i >= 0; i--)
        {
            current = _classLayers[i].Backward(current);
        }

        _classOutput.UpdateParameters(lr);
        foreach (var layer in _classLayers) layer.UpdateParameters(lr);
    }

    #endregion

    #region Gradient Penalty

    private void ApplyGradientPenalty(Vector<T> realRow, Vector<T> fakeRow, T scaledLr)
    {
        double alpha = _random.NextDouble();
        int len = Math.Min(realRow.Length, fakeRow.Length);
        var interpolated = new Vector<T>(len);

        for (int i = 0; i < len; i++)
        {
            interpolated[i] = NumOps.Add(
                NumOps.Multiply(NumOps.FromDouble(alpha), realRow[i]),
                NumOps.Multiply(NumOps.FromDouble(1.0 - alpha), fakeRow[i]));
        }

        // Compute gradient penalty using GradientTape autodiff
        var interpolatedTensor = VectorToTensor(interpolated);
        var inputGrad = TapeLayerBridge<T>.ComputeInputGradient(
            interpolatedTensor,
            _discLayers,
            TapeLayerBridge<T>.HiddenActivation.LeakyReLU,
            applyActivationOnLast: false);

        double gradNormSq = 0;
        for (int i = 0; i < inputGrad.Length; i++)
        {
            double g = NumOps.ToDouble(inputGrad[i]);
            gradNormSq += g * g;
        }
        double gradNorm = Math.Sqrt(gradNormSq + 1e-12);
        double penaltyGradScale = 2.0 * _options.GradientPenaltyWeight * (gradNorm - 1.0) / gradNorm;

        if (Math.Abs(penaltyGradScale) > 1e-10)
        {
            _ = DiscriminatorForward(VectorToTensor(interpolated), isTraining: false);
            var penaltyGrad = new Tensor<T>([1]);
            penaltyGrad[0] = NumOps.FromDouble(penaltyGradScale);
            BackwardDiscriminator(penaltyGrad);
            UpdateDiscriminatorParameters(scaledLr);
        }
    }

    #endregion

    #region Backward Passes

    private void BackwardDiscriminator(Tensor<T> gradOutput)
    {
        var current = gradOutput;
        current = _discLayers[^1].Backward(current);
        for (int i = _discLayers.Count - 2; i >= 0; i--)
        {
            if (i < _discPreActivations.Count)
                current = ApplyLeakyReLUDerivative(current, _discPreActivations[i]);
            current = _discLayers[i].Backward(current);
        }
    }

    private void BackwardGenerator(Tensor<T> gradOutput)
    {
        if (_usingCustomLayers)
        {
            var current = gradOutput;
            for (int i = Layers.Count - 1; i >= 0; i--) current = Layers[i].Backward(current);
            return;
        }

        int inputDim = _options.EmbeddingDimension;
        var h = gradOutput;

        h = Layers[^1].Backward(h);

        int lastHiddenDim = h.Length - inputDim;
        if (lastHiddenDim > 0)
        {
            var hiddenGrad = new Tensor<T>([lastHiddenDim]);
            for (int j = 0; j < lastHiddenDim && j < h.Length; j++) hiddenGrad[j] = h[j];
            h = hiddenGrad;
        }

        for (int i = Layers.Count - 2; i >= 0; i--)
        {
            if (i < _genPreActivations.Count)
                h = ApplyReLUDerivative(h, _genPreActivations[i]);
            h = _genBNLayers[i].Backward(h);
            h = Layers[i].Backward(h);

            if (i > 0)
            {
                int prevDim = h.Length - inputDim;
                if (prevDim > 0)
                {
                    var hiddenGrad = new Tensor<T>([prevDim]);
                    for (int j = 0; j < prevDim && j < h.Length; j++) hiddenGrad[j] = h[j];
                    h = hiddenGrad;
                }
            }
        }
    }

    private void UpdateGeneratorParameters(T learningRate)
    {
        foreach (var layer in Layers) layer.UpdateParameters(learningRate);
        foreach (var bn in _genBNLayers) bn.UpdateParameters(learningRate);
    }

    private void UpdateDiscriminatorParameters(T learningRate)
    {
        foreach (var layer in _discLayers) layer.UpdateParameters(learningRate);
    }

    #endregion

    #region NeuralNetworkBase Required Overrides

    /// <inheritdoc />
    public override Tensor<T> Predict(Tensor<T> input)
    {
        var current = input;
        foreach (var layer in Layers) current = layer.Forward(current);
        return current;
    }

    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // TableGAN uses its own specialized training via Fit/FitAsync.
    }

    /// <inheritdoc />
    public override void UpdateParameters(Vector<T> parameters)
    {
        int offset = 0;
        foreach (var layer in Layers)
        {
            var layerParams = layer.GetParameters();
            int count = layerParams.Length;
            if (offset + count <= parameters.Length)
            {
                var slice = new Vector<T>(count);
                for (int i = 0; i < count; i++) slice[i] = parameters[offset + i];
                layer.UpdateParameters(slice);
                offset += count;
            }
        }
    }

    /// <inheritdoc />
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_options.EmbeddingDimension);
        writer.Write(_dataWidth);
        writer.Write(_numClasses);
        writer.Write(IsFitted);
    }

    /// <inheritdoc />
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32();
        _dataWidth = reader.ReadInt32();
        _numClasses = reader.ReadInt32();
        IsFitted = reader.ReadBoolean();
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new TableGANGenerator<T>(Architecture, _options);
    }

    /// <inheritdoc />
    public override Dictionary<string, T> GetFeatureImportance()
    {
        var importance = new Dictionary<string, T>();
        for (int i = 0; i < _columns.Count; i++)
            importance[$"feature_{i}"] = NumOps.One;
        return importance;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                ["generator_type"] = "TableGAN",
                ["embedding_dimension"] = _options.EmbeddingDimension,
                ["label_column_index"] = _options.LabelColumnIndex,
                ["num_classes"] = _numClasses,
                ["is_fitted"] = IsFitted,
                ["data_width"] = _dataWidth,
                ["using_custom_layers"] = _usingCustomLayers
            }
        };
    }

    #endregion

    #region Statistics & Activations

    private void ComputeRealStatistics(Matrix<T> data)
    {
        _realMean = new Vector<T>(data.Columns);
        _realVar = new Vector<T>(data.Columns);

        for (int j = 0; j < data.Columns; j++)
        {
            double sum = 0;
            for (int i = 0; i < data.Rows; i++) sum += NumOps.ToDouble(data[i, j]);
            double mean = sum / data.Rows;

            double varSum = 0;
            for (int i = 0; i < data.Rows; i++)
            {
                double diff = NumOps.ToDouble(data[i, j]) - mean;
                varSum += diff * diff;
            }

            _realMean[j] = NumOps.FromDouble(mean);
            _realVar[j] = NumOps.FromDouble(varSum / Math.Max(1, data.Rows - 1));
        }
    }

    private Vector<T> ComputeSoftmax(Vector<T> logits)
    {
        double maxVal = double.MinValue;
        for (int i = 0; i < logits.Length; i++)
        {
            double v = NumOps.ToDouble(logits[i]);
            if (v > maxVal) maxVal = v;
        }

        var result = new Vector<T>(logits.Length);
        double sumExp = 0;
        for (int i = 0; i < logits.Length; i++)
            sumExp += Math.Exp(NumOps.ToDouble(logits[i]) - maxVal);

        for (int i = 0; i < logits.Length; i++)
            result[i] = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(logits[i]) - maxVal) / Math.Max(sumExp, 1e-10));

        return result;
    }

    #endregion

    #region Activation Functions

    private Tensor<T> ApplyReLU(Tensor<T> input)
    {
        var result = new Tensor<T>(input.Shape);
        for (int i = 0; i < input.Length; i++)
            result[i] = NumOps.ToDouble(input[i]) > 0 ? input[i] : NumOps.Zero;
        return result;
    }

    private Tensor<T> ApplyReLUDerivative(Tensor<T> gradOutput, Tensor<T> preActivation)
    {
        int len = Math.Min(gradOutput.Length, preActivation.Length);
        var result = new Tensor<T>(gradOutput.Shape);
        for (int i = 0; i < len; i++)
            result[i] = NumOps.ToDouble(preActivation[i]) > 0 ? gradOutput[i] : NumOps.Zero;
        return result;
    }

    private Tensor<T> ApplyLeakyReLU(Tensor<T> input)
    {
        var result = new Tensor<T>(input.Shape);
        T slope = NumOps.FromDouble(0.2);
        for (int i = 0; i < input.Length; i++)
        {
            double val = NumOps.ToDouble(input[i]);
            result[i] = val > 0 ? input[i] : NumOps.Multiply(slope, input[i]);
        }
        return result;
    }

    private Tensor<T> ApplyLeakyReLUDerivative(Tensor<T> gradOutput, Tensor<T> preActivation)
    {
        int len = Math.Min(gradOutput.Length, preActivation.Length);
        var result = new Tensor<T>(gradOutput.Shape);
        T slope = NumOps.FromDouble(0.2);
        for (int i = 0; i < len; i++)
        {
            result[i] = NumOps.ToDouble(preActivation[i]) > 0
                ? gradOutput[i]
                : NumOps.Multiply(slope, gradOutput[i]);
        }
        return result;
    }

    private Tensor<T> ApplyOutputActivations(Tensor<T> output)
    {
        if (_transformer is null) return output;

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
                if (numModes > 0) ApplySoftmaxBlock(output, result, ref idx, numModes);
            }
            else
            {
                ApplySoftmaxBlock(output, result, ref idx, transform.Width);
            }
        }

        return result;
    }

    private void ApplySoftmaxBlock(Tensor<T> input, Tensor<T> output, ref int idx, int count)
    {
        if (count <= 0) return;

        double maxVal = double.MinValue;
        for (int m = 0; m < count && (idx + m) < input.Length; m++)
        {
            double v = NumOps.ToDouble(input[idx + m]);
            if (v > maxVal) maxVal = v;
        }

        double sumExp = 0;
        for (int m = 0; m < count && (idx + m) < input.Length; m++)
            sumExp += Math.Exp(NumOps.ToDouble(input[idx + m]) - maxVal);

        for (int m = 0; m < count && idx < input.Length; m++)
        {
            double expVal = Math.Exp(NumOps.ToDouble(input[idx]) - maxVal);
            output[idx] = NumOps.FromDouble(expVal / Math.Max(sumExp, 1e-10));
            idx++;
        }
    }

    #endregion

    #region Helpers

    private Tensor<T> SafeGradient(Tensor<T> grad, double maxNorm)
    {
        var result = new Tensor<T>(grad.Shape);
        double normSq = 0;

        for (int i = 0; i < grad.Length; i++)
        {
            double val = NumOps.ToDouble(grad[i]);
            if (double.IsNaN(val) || double.IsInfinity(val)) val = 0;
            result[i] = NumOps.FromDouble(val);
            normSq += val * val;
        }

        double norm = Math.Sqrt(normSq + 1e-12);
        if (norm > maxNorm)
        {
            double scale = maxNorm / norm;
            for (int i = 0; i < result.Length; i++)
                result[i] = NumOps.FromDouble(NumOps.ToDouble(result[i]) * scale);
        }

        return result;
    }

    private Vector<T> CreateStandardNormalVector(int size)
    {
        var v = new Vector<T>(size);
        for (int i = 0; i < size; i++)
        {
            double u1 = Math.Max(1e-10, _random.NextDouble());
            double u2 = _random.NextDouble();
            double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            v[i] = NumOps.FromDouble(z);
        }
        return v;
    }

    private static Vector<T> GetRow(Matrix<T> matrix, int row)
    {
        var v = new Vector<T>(matrix.Columns);
        for (int j = 0; j < matrix.Columns; j++) v[j] = matrix[row, j];
        return v;
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

    private static Tensor<T> ConcatTensors(Tensor<T> a, Tensor<T> b)
    {
        var result = new Tensor<T>([a.Length + b.Length]);
        for (int i = 0; i < a.Length; i++) result[i] = a[i];
        for (int i = 0; i < b.Length; i++) result[a.Length + i] = b[i];
        return result;
    }

    private static Tensor<T> CloneTensor(Tensor<T> source)
    {
        var clone = new Tensor<T>(source.Shape);
        for (int i = 0; i < source.Length; i++) clone[i] = source[i];
        return clone;
    }

    #endregion

    #region IJitCompilable Override

    /// <inheritdoc />
    public override bool SupportsJitCompilation =>
        IsFitted && Layers.Count > 1 && !_usingCustomLayers &&
        _genBNLayers.Count > 0 &&
        Layers.All(l => l.SupportsJitCompilation) &&
        _genBNLayers.All(l => l.SupportsJitCompilation);

    /// <inheritdoc />
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (!SupportsJitCompilation)
        {
            throw new NotSupportedException(
                $"{GetType().Name} does not support JIT compilation in its current configuration.");
        }

        int genInputDim = _options.EmbeddingDimension;
        var hiddenLayers = Layers.Take(Layers.Count - 1).ToList();

        return TapeLayerBridge<T>.ExportMLPGeneratorGraph(
            inputNodes, genInputDim, hiddenLayers,
            _genBNLayers.Cast<ILayer<T>>().ToList(), Layers[^1],
            TapeLayerBridge<T>.HiddenActivation.ReLU,
            TapeLayerBridge<T>.HiddenActivation.None,
            useResidualConcat: true);
    }

    #endregion
}
