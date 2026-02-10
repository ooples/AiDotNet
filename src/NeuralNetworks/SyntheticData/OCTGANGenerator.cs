using AiDotNet.ActivationFunctions;
using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.SyntheticData;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.NeuralNetworks.SyntheticData;

/// <summary>
/// OCT-GAN (One-Class Tabular GAN) generator for synthesizing minority-class tabular data
/// using a one-class discriminator with Deep SVDD (Support Vector Data Description) objective.
/// </summary>
/// <remarks>
/// <para>
/// OCT-GAN addresses the class imbalance problem by combining:
/// - <b>One-class discriminator</b>: Learns the hypersphere boundary of minority class data
/// - <b>Deep SVDD objective</b>: Maps real data close to a learned center, pushes fakes away
/// - <b>WGAN-GP training</b>: Wasserstein distance with gradient penalty for stable training
/// - <b>Residual generator</b>: Skip connections for better gradient flow
/// - <b>VGM normalization</b>: Handles multi-modal continuous distributions
///
/// This implementation follows the standard neural network architecture pattern with:
/// - Proper inheritance from NeuralNetworkBase
/// - Layer-based architecture using ILayer components
/// - Support for custom layers via NeuralNetworkArchitecture
/// - Full forward, backward, update, reset lifecycle
/// </para>
/// <para>
/// <b>For Beginners:</b> OCT-GAN is designed for imbalanced datasets where one class is rare
/// (e.g., fraud detection with 99% normal transactions and 1% fraud):
///
/// 1. The <b>Generator</b> takes random noise and creates synthetic minority-class rows
/// 2. The <b>Discriminator</b> learns a compact "sphere" around real minority samples
/// 3. Real minority data maps close to the sphere's center (low SVDD score)
/// 4. The generator learns to produce data that also maps near the center
/// 5. A gradient penalty keeps training stable
///
/// If you provide custom layers in the architecture, those will be used directly
/// for the generator network. If not, the network creates standard layers.
///
/// Example usage:
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputFeatures: 10,
///     outputSize: 10
/// );
/// var options = new OCTGANOptions&lt;double&gt;
/// {
///     MinorityClassValue = 1,
///     LabelColumnIndex = 4,
///     Epochs = 300
/// };
/// var generator = new OCTGANGenerator&lt;double&gt;(architecture, options);
/// generator.Fit(data, columns, epochs: 300);
/// var synthetic = generator.Generate(1000);
/// </code>
/// </para>
/// <para>
/// Reference: "OCT-GAN: One-Class Tabular GAN for Imbalanced Data" (2021)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class OCTGANGenerator<T> : NeuralNetworkBase<T>, ISyntheticTabularGenerator<T>
{
    private readonly OCTGANOptions<T> _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private ILossFunction<T> _lossFunction;

    // Synthetic tabular data infrastructure
    private TabularDataTransformer<T>? _transformer;
    private List<ColumnMetadata> _columns = new();
    private int _dataWidth;
    private Random _random;

    // Generator batch normalization layers (auxiliary, always match generator hidden layers)
    private readonly List<BatchNormalizationLayer<T>> _genBNLayers = new();

    // Discriminator layers (auxiliary, not user-overridable)
    private readonly List<FullyConnectedLayer<T>> _discLayers = new();
    private readonly List<DropoutLayer<T>> _discDropoutLayers = new();
    private readonly List<(int InputSize, int OutputSize)> _discLayerDims = new();

    // SVDD embedding layer (final discriminator output before distance computation)
    private FullyConnectedLayer<T>? _svddEmbeddingLayer;

    // Cached pre-activations for proper backward passes
    private readonly List<Tensor<T>> _genPreActivations = new();
    private readonly List<Tensor<T>> _discPreActivations = new();

    // SVDD center in embedding space
    private Tensor<T>? _svddCenter;

    private Matrix<T>? _minorityData;
    private bool _usingCustomLayers;

    /// <summary>
    /// Gets the OCT-GAN-specific options.
    /// </summary>
    public new OCTGANOptions<T> Options => _options;

    /// <inheritdoc />
    public IReadOnlyList<ColumnMetadata> Columns => _columns.AsReadOnly();

    /// <inheritdoc />
    public bool IsFitted { get; private set; }

    /// <summary>
    /// Initializes a new OCT-GAN generator with the specified architecture.
    /// </summary>
    public OCTGANGenerator(
        NeuralNetworkArchitecture<T> architecture,
        OCTGANOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 5.0)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), maxGradNorm)
    {
        _options = options ?? new OCTGANOptions<T>();
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();
        InitializeLayers();
    }

    #region Layer Initialization

    protected override void InitializeLayers()
    {
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            _usingCustomLayers = true;
            Layers.Clear();
            foreach (var layer in Architecture.Layers)
            {
                Layers.Add(layer);
            }
        }
        else
        {
            _usingCustomLayers = false;
            BuildDefaultGeneratorLayers(
                _options.EmbeddingDimension,
                Architecture.OutputSize > 0 ? Architecture.OutputSize : 1);
        }
    }

    private void RebuildLayersWithActualDimensions()
    {
        if (!_usingCustomLayers)
        {
            BuildDefaultGeneratorLayers(_options.EmbeddingDimension, _dataWidth);
        }

        BuildDiscriminator();
    }

    private void BuildDefaultGeneratorLayers(int inputDim, int outputDim)
    {
        Layers.Clear();
        _genBNLayers.Clear();

        var defaultLayers = LayerHelper<T>.CreateDefaultPATEGANGeneratorLayers(
            inputDim, outputDim, _options.GeneratorDimensions).ToList();

        foreach (var layer in defaultLayers)
        {
            Layers.Add(layer);
        }

        for (int i = 0; i < _options.GeneratorDimensions.Length; i++)
        {
            _genBNLayers.Add(new BatchNormalizationLayer<T>(_options.GeneratorDimensions[i]));
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
        _svddEmbeddingLayer = new FullyConnectedLayer<T>(lastHidden, _options.SVDDEmbeddingDimension, identity);
        _discLayerDims.Add((lastHidden, _options.SVDDEmbeddingDimension));
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

        _minorityData = ExtractMinorityData(data, transformedData, columns);
        var trainData = _minorityData.Rows > 0 ? _minorityData : transformedData;

        RebuildLayersWithActualDimensions();
        InitializeSVDDCenter(trainData);

        T lr = NumOps.FromDouble(_options.LearningRate);
        int batchSize = Math.Min(_options.BatchSize, trainData.Rows);
        int numBatches = Math.Max(1, trainData.Rows / batchSize);

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            for (int batch = 0; batch < numBatches; batch++)
            {
                int batchStart = batch * batchSize;
                int batchEnd = Math.Min(batchStart + batchSize, trainData.Rows);
                int actualBatchSize = batchEnd - batchStart;

                T scaledLr = NumOps.FromDouble(_options.LearningRate / actualBatchSize);

                for (int dStep = 0; dStep < _options.DiscriminatorSteps; dStep++)
                {
                    TrainDiscriminatorStep(trainData, batchStart, batchEnd, scaledLr);
                }

                TrainGeneratorStep(actualBatchSize, scaledLr);
                UpdateSVDDCenter(trainData, batchStart, batchEnd);
            }
        }

        IsFitted = true;
    }

    /// <inheritdoc />
    public async Task FitAsync(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns,
        int epochs, CancellationToken cancellationToken = default)
    {
        await Task.Run(() =>
        {
            cancellationToken.ThrowIfCancellationRequested();
            Fit(data, columns, epochs);
        }, cancellationToken).ConfigureAwait(false);
    }

    /// <inheritdoc />
    public Matrix<T> Generate(int numSamples, Vector<T>? conditionColumn = null, Vector<T>? conditionValue = null)
    {
        if (_transformer is null)
        {
            throw new InvalidOperationException("Generator is not fitted. Call Fit() first.");
        }

        var transformedRows = new Matrix<T>(numSamples, _dataWidth);

        for (int i = 0; i < numSamples; i++)
        {
            var noise = CreateStandardNormalVector(_options.EmbeddingDimension);
            var fakeRow = GeneratorForward(VectorToTensor(noise));
            for (int j = 0; j < _dataWidth; j++)
            {
                transformedRows[i, j] = j < fakeRow.Length ? fakeRow[j] : NumOps.Zero;
            }
        }

        return _transformer.InverseTransform(transformedRows);
    }

    #endregion

    #region Minority Data Extraction

    private Matrix<T> ExtractMinorityData(Matrix<T> rawData, Matrix<T> transformedData, IReadOnlyList<ColumnMetadata> columns)
    {
        int labelCol = _options.LabelColumnIndex;
        if (labelCol < 0 || labelCol >= columns.Count)
        {
            labelCol = columns.Count - 1;
        }

        var minorityRows = new List<int>();
        for (int r = 0; r < rawData.Rows; r++)
        {
            int label = (int)Math.Round(NumOps.ToDouble(rawData[r, labelCol]));
            if (label == _options.MinorityClassValue)
            {
                minorityRows.Add(r);
            }
        }

        if (minorityRows.Count == 0)
        {
            return transformedData;
        }

        var result = new Matrix<T>(minorityRows.Count, transformedData.Columns);
        for (int i = 0; i < minorityRows.Count; i++)
        {
            for (int j = 0; j < transformedData.Columns; j++)
            {
                result[i, j] = transformedData[minorityRows[i], j];
            }
        }

        return result;
    }

    #endregion

    #region SVDD Center

    private void InitializeSVDDCenter(Matrix<T> trainData)
    {
        int embDim = _options.SVDDEmbeddingDimension;
        _svddCenter = new Tensor<T>([embDim]);

        int numSamples = Math.Min(trainData.Rows, 500);
        for (int i = 0; i < numSamples; i++)
        {
            var row = GetRow(trainData, i);
            var embedding = DiscriminatorForward(VectorToTensor(row), isTraining: false);
            for (int j = 0; j < embDim && j < embedding.Length; j++)
            {
                _svddCenter[j] = NumOps.Add(_svddCenter[j], embedding[j]);
            }
        }

        T invN = NumOps.FromDouble(1.0 / numSamples);
        for (int j = 0; j < embDim; j++)
        {
            _svddCenter[j] = NumOps.Multiply(_svddCenter[j], invN);
        }
    }

    private void UpdateSVDDCenter(Matrix<T> trainData, int batchStart, int batchEnd)
    {
        if (_svddCenter is null) return;

        int embDim = _options.SVDDEmbeddingDimension;
        double momentum = _options.CenterUpdateMomentum;
        int batchSize = batchEnd - batchStart;

        var batchMean = new Tensor<T>([embDim]);
        for (int i = batchStart; i < batchEnd; i++)
        {
            var row = GetRow(trainData, i);
            var embedding = DiscriminatorForward(VectorToTensor(row), isTraining: false);
            for (int j = 0; j < embDim && j < embedding.Length; j++)
            {
                batchMean[j] = NumOps.Add(batchMean[j], embedding[j]);
            }
        }

        T invN = NumOps.FromDouble(1.0 / batchSize);
        for (int j = 0; j < embDim; j++)
        {
            batchMean[j] = NumOps.Multiply(batchMean[j], invN);
        }

        T momT = NumOps.FromDouble(momentum);
        T oneMinusMom = NumOps.FromDouble(1.0 - momentum);
        for (int j = 0; j < embDim; j++)
        {
            _svddCenter[j] = NumOps.Add(
                NumOps.Multiply(oneMinusMom, _svddCenter[j]),
                NumOps.Multiply(momT, batchMean[j]));
        }
    }

    private double ComputeSVDDDistance(Tensor<T> embedding)
    {
        if (_svddCenter is null) return 0.0;

        double distSq = 0;
        int len = Math.Min(embedding.Length, _svddCenter.Length);
        for (int j = 0; j < len; j++)
        {
            double diff = NumOps.ToDouble(embedding[j]) - NumOps.ToDouble(_svddCenter[j]);
            distSq += diff * diff;
        }

        return distSq;
    }

    private Tensor<T> ComputeSVDDGradient(Tensor<T> embedding)
    {
        if (_svddCenter is null) return new Tensor<T>(embedding.Shape);

        int len = Math.Min(embedding.Length, _svddCenter.Length);
        var grad = new Tensor<T>([len]);
        for (int j = 0; j < len; j++)
        {
            double diff = NumOps.ToDouble(embedding[j]) - NumOps.ToDouble(_svddCenter[j]);
            grad[j] = NumOps.FromDouble(2.0 * diff);
        }

        return grad;
    }

    #endregion

    #region Forward Passes

    private Tensor<T> GeneratorForward(Tensor<T> input)
    {
        if (_usingCustomLayers)
        {
            var current = input;
            for (int i = 0; i < Layers.Count; i++)
            {
                current = Layers[i].Forward(current);
            }
            return ApplyOutputActivations(current);
        }

        return DefaultGeneratorForward(input);
    }

    private Tensor<T> DefaultGeneratorForward(Tensor<T> input)
    {
        _genPreActivations.Clear();
        var current = input;
        var originalInput = CloneTensor(input);

        for (int i = 0; i < Layers.Count - 1; i++)
        {
            if (i > 0)
            {
                current = ConcatTensors(current, originalInput);
            }

            current = Layers[i].Forward(current);

            if (i < _genBNLayers.Count)
            {
                current = _genBNLayers[i].Forward(current);
            }

            _genPreActivations.Add(CloneTensor(current));
            current = ApplyReLU(current);
        }

        current = ConcatTensors(current, originalInput);
        current = Layers[^1].Forward(current);

        return ApplyOutputActivations(current);
    }

    private Tensor<T> DiscriminatorForward(Tensor<T> input, bool isTraining)
    {
        _discPreActivations.Clear();
        var current = input;

        for (int i = 0; i < _discLayers.Count; i++)
        {
            current = _discLayers[i].Forward(current);
            _discPreActivations.Add(CloneTensor(current));
            current = ApplyLeakyReLU(current);

            if (isTraining)
            {
                current = _discDropoutLayers[i].Forward(current);
            }
        }

        if (_svddEmbeddingLayer is not null)
        {
            current = _svddEmbeddingLayer.Forward(current);
        }

        return current;
    }

    #endregion

    #region Training Steps

    private void TrainDiscriminatorStep(Matrix<T> trainData, int batchStart, int batchEnd, T scaledLr)
    {
        for (int i = batchStart; i < batchEnd; i++)
        {
            var realRow = GetRow(trainData, i);
            var realEmbedding = DiscriminatorForward(VectorToTensor(realRow), isTraining: true);

            var realGrad = ComputeSVDDGradient(realEmbedding);
            BackwardDiscriminator(realGrad);
            UpdateDiscriminatorParameters(scaledLr);

            var noise = CreateStandardNormalVector(_options.EmbeddingDimension);
            var fakeRow = GeneratorForward(VectorToTensor(noise));

            var fakeEmbedding = DiscriminatorForward(fakeRow, isTraining: true);

            var fakeGrad = ComputeSVDDGradient(fakeEmbedding);
            var negFakeGrad = new Tensor<T>(fakeGrad.Shape);
            for (int j = 0; j < fakeGrad.Length; j++)
            {
                negFakeGrad[j] = NumOps.Negate(fakeGrad[j]);
            }
            BackwardDiscriminator(negFakeGrad);
            UpdateDiscriminatorParameters(scaledLr);

            ApplyGradientPenalty(realRow, TensorToVector(fakeRow, _dataWidth), scaledLr);
        }
    }

    private void TrainGeneratorStep(int batchSize, T scaledLr)
    {
        for (int i = 0; i < batchSize; i++)
        {
            var noise = CreateStandardNormalVector(_options.EmbeddingDimension);
            var fakeRow = GeneratorForward(VectorToTensor(noise));

            var fakeEmbedding = DiscriminatorForward(fakeRow, isTraining: false);

            var embGrad = ComputeSVDDGradient(fakeEmbedding);
            var discInputGrad = ComputeDiscriminatorInputGradient(embGrad);
            discInputGrad = SafeGradient(discInputGrad, 5.0);

            _ = GeneratorForward(VectorToTensor(noise));

            BackwardGenerator(discInputGrad);
            UpdateGeneratorParameters(scaledLr);
        }
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

        var interpEmbedding = DiscriminatorForward(VectorToTensor(interpolated), isTraining: false);

        var embGrad = ComputeSVDDGradient(interpEmbedding);
        var inputGrad = ComputeDiscriminatorInputGradient(embGrad);

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
            var reEmbedding = DiscriminatorForward(VectorToTensor(interpolated), isTraining: false);
            var penaltyEmbGrad = ComputeSVDDGradient(reEmbedding);

            for (int j = 0; j < penaltyEmbGrad.Length; j++)
            {
                penaltyEmbGrad[j] = NumOps.FromDouble(
                    NumOps.ToDouble(penaltyEmbGrad[j]) * penaltyGradScale / (gradNorm + 1e-12));
            }

            BackwardDiscriminator(penaltyEmbGrad);
            UpdateDiscriminatorParameters(scaledLr);
        }
    }

    private Tensor<T> ComputeDiscriminatorInputGradient(Tensor<T> outputGrad)
    {
        var current = outputGrad;

        if (_svddEmbeddingLayer is not null)
        {
            int lastIdx = _discLayerDims.Count - 1;
            var embParams = _svddEmbeddingLayer.GetParameters();
            int embOutputSize = _discLayerDims[lastIdx].OutputSize;
            int embInputSize = _discLayerDims[lastIdx].InputSize;
            current = ManualLinearBackward(current, embParams, embOutputSize, embInputSize);
        }

        for (int i = _discLayers.Count - 1; i >= 0; i--)
        {
            if (i < _discPreActivations.Count)
            {
                current = ApplyLeakyReLUDerivative(current, _discPreActivations[i]);
            }

            var layerParams = _discLayers[i].GetParameters();
            int layerOutputSize = _discLayerDims[i].OutputSize;
            int layerInputSize = _discLayerDims[i].InputSize;
            current = ManualLinearBackward(current, layerParams, layerOutputSize, layerInputSize);
        }

        return current;
    }

    private Tensor<T> ManualLinearBackward(Tensor<T> outputGrad, Vector<T> layerParams,
        int outputSize, int inputSize)
    {
        var inputGrad = new Tensor<T>([inputSize]);

        for (int j = 0; j < inputSize; j++)
        {
            double sum = 0;
            for (int k = 0; k < outputSize && k < outputGrad.Length; k++)
            {
                int wIdx = k * inputSize + j;
                if (wIdx < layerParams.Length)
                {
                    sum += NumOps.ToDouble(layerParams[wIdx]) * NumOps.ToDouble(outputGrad[k]);
                }
            }
            inputGrad[j] = NumOps.FromDouble(sum);
        }

        return inputGrad;
    }

    #endregion

    #region Backward Passes

    private void BackwardDiscriminator(Tensor<T> gradOutput)
    {
        var current = gradOutput;

        if (_svddEmbeddingLayer is not null)
        {
            current = _svddEmbeddingLayer.Backward(current);
        }

        for (int i = _discLayers.Count - 1; i >= 0; i--)
        {
            if (i < _discPreActivations.Count)
            {
                current = ApplyLeakyReLUDerivative(current, _discPreActivations[i]);
            }
            current = _discLayers[i].Backward(current);
        }
    }

    private void BackwardGenerator(Tensor<T> gradOutput)
    {
        if (_usingCustomLayers)
        {
            var current = gradOutput;
            for (int i = Layers.Count - 1; i >= 0; i--)
            {
                current = Layers[i].Backward(current);
            }
            return;
        }

        BackwardDefaultGenerator(gradOutput);
    }

    private void BackwardDefaultGenerator(Tensor<T> gradOutput)
    {
        int inputDim = _options.EmbeddingDimension;
        var current = gradOutput;

        current = Layers[^1].Backward(current);

        int lastHiddenDim = current.Length - inputDim;
        if (lastHiddenDim > 0)
        {
            var hiddenGrad = new Tensor<T>([lastHiddenDim]);
            for (int j = 0; j < lastHiddenDim && j < current.Length; j++)
            {
                hiddenGrad[j] = current[j];
            }
            current = hiddenGrad;
        }

        for (int i = Layers.Count - 2; i >= 0; i--)
        {
            if (i < _genPreActivations.Count)
            {
                current = ApplyReLUDerivative(current, _genPreActivations[i]);
            }

            if (i < _genBNLayers.Count)
            {
                current = _genBNLayers[i].Backward(current);
            }

            current = Layers[i].Backward(current);

            if (i > 0)
            {
                int prevDim = current.Length - inputDim;
                if (prevDim > 0)
                {
                    var hiddenGrad = new Tensor<T>([prevDim]);
                    for (int j = 0; j < prevDim && j < current.Length; j++)
                    {
                        hiddenGrad[j] = current[j];
                    }
                    current = hiddenGrad;
                }
            }
        }
    }

    private void UpdateGeneratorParameters(T learningRate)
    {
        foreach (var layer in Layers)
        {
            layer.UpdateParameters(learningRate);
        }
        foreach (var bn in _genBNLayers)
        {
            bn.UpdateParameters(learningRate);
        }
    }

    private void UpdateDiscriminatorParameters(T learningRate)
    {
        foreach (var layer in _discLayers)
        {
            layer.UpdateParameters(learningRate);
        }
        _svddEmbeddingLayer?.UpdateParameters(learningRate);
    }

    #endregion

    #region Activation Functions

    private Tensor<T> ApplyReLU(Tensor<T> input)
    {
        var result = new Tensor<T>(input.Shape);
        for (int i = 0; i < input.Length; i++)
        {
            result[i] = NumOps.ToDouble(input[i]) > 0 ? input[i] : NumOps.Zero;
        }
        return result;
    }

    private Tensor<T> ApplyReLUDerivative(Tensor<T> gradOutput, Tensor<T> preActivation)
    {
        int len = Math.Min(gradOutput.Length, preActivation.Length);
        var result = new Tensor<T>(gradOutput.Shape);
        for (int i = 0; i < len; i++)
        {
            result[i] = NumOps.ToDouble(preActivation[i]) > 0 ? gradOutput[i] : NumOps.Zero;
        }
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
            if (NumOps.ToDouble(preActivation[i]) > 0)
            {
                result[i] = gradOutput[i];
            }
            else
            {
                result[i] = NumOps.Multiply(slope, gradOutput[i]);
            }
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
                    double val = NumOps.ToDouble(output[idx]);
                    result[idx] = NumOps.FromDouble(Math.Tanh(val));
                    idx++;
                }

                int numModes = transform.Width - 1;
                if (numModes > 0)
                {
                    ApplySoftmaxBlock(output, result, ref idx, numModes);
                }
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
        {
            sumExp += Math.Exp(NumOps.ToDouble(input[idx + m]) - maxVal);
        }

        for (int m = 0; m < count && idx < input.Length; m++)
        {
            double expVal = Math.Exp(NumOps.ToDouble(input[idx]) - maxVal);
            output[idx] = NumOps.FromDouble(expVal / Math.Max(sumExp, 1e-10));
            idx++;
        }
    }

    #endregion

    #region Gradient Safety

    private Tensor<T> SafeGradient(Tensor<T> grad, double maxNorm)
    {
        double normSq = 0;
        for (int i = 0; i < grad.Length; i++)
        {
            double val = NumOps.ToDouble(grad[i]);
            if (double.IsNaN(val) || double.IsInfinity(val))
            {
                grad[i] = NumOps.Zero;
            }
            else
            {
                normSq += val * val;
            }
        }

        double norm = Math.Sqrt(normSq);
        if (norm > maxNorm && norm > 0)
        {
            double scale = maxNorm / norm;
            for (int i = 0; i < grad.Length; i++)
            {
                grad[i] = NumOps.FromDouble(NumOps.ToDouble(grad[i]) * scale);
            }
        }

        return grad;
    }

    #endregion

    #region Data Helpers

    private Vector<T> CreateStandardNormalVector(int length)
    {
        var v = new Vector<T>(length);
        for (int i = 0; i < length; i++)
        {
            double u1 = 1.0 - _random.NextDouble();
            double u2 = _random.NextDouble();
            double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            v[i] = NumOps.FromDouble(z);
        }
        return v;
    }

    private static Vector<T> GetRow(Matrix<T> matrix, int row)
    {
        var v = new Vector<T>(matrix.Columns);
        for (int j = 0; j < matrix.Columns; j++)
        {
            v[j] = matrix[row, j];
        }
        return v;
    }

    private Tensor<T> VectorToTensor(Vector<T> v)
    {
        var t = new Tensor<T>([v.Length]);
        for (int i = 0; i < v.Length; i++) t[i] = v[i];
        return t;
    }

    private Vector<T> TensorToVector(Tensor<T> t, int length)
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
        for (int i = 0; i < source.Length; i++)
        {
            clone[i] = source[i];
        }
        return clone;
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                ["GeneratorType"] = "OCTGANGenerator",
                ["EmbeddingDimension"] = _options.EmbeddingDimension,
                ["GeneratorDimensions"] = _options.GeneratorDimensions,
                ["DiscriminatorDimensions"] = _options.DiscriminatorDimensions,
                ["SVDDEmbeddingDimension"] = _options.SVDDEmbeddingDimension,
                ["BatchSize"] = _options.BatchSize,
                ["Epochs"] = _options.Epochs,
                ["LearningRate"] = _options.LearningRate,
                ["IsFitted"] = IsFitted,
                ["DataWidth"] = _dataWidth,
                ["UsingCustomLayers"] = _usingCustomLayers,
            }
        };
    }

    /// <inheritdoc />
    public override Dictionary<string, T> GetFeatureImportance()
    {
        var importance = new Dictionary<string, T>();
        T equal = NumOps.FromDouble(1.0 / Math.Max(_columns.Count, 1));
        for (int i = 0; i < _columns.Count; i++)
        {
            importance[_columns[i].Name] = equal;
        }
        return importance;
    }

    /// <inheritdoc />
    public override Tensor<T> Predict(Tensor<T> input)
    {
        if (!IsFitted)
        {
            throw new InvalidOperationException("Generator is not fitted. Call Fit() first.");
        }

        var noise = new Vector<T>(input.Length);
        for (int i = 0; i < input.Length; i++)
        {
            noise[i] = input[i];
        }

        return GeneratorForward(VectorToTensor(noise));
    }

    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Training is handled by Fit() for synthetic tabular generators
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
                var newParams = new Vector<T>(count);
                for (int i = 0; i < count; i++)
                {
                    newParams[i] = parameters[offset + i];
                }
                layer.SetParameters(newParams);
                offset += count;
            }
        }
    }

    /// <inheritdoc />
    protected override void SerializeNetworkSpecificData(System.IO.BinaryWriter writer)
    {
        writer.Write(_dataWidth);
        writer.Write(_usingCustomLayers);
        writer.Write(IsFitted);
    }

    /// <inheritdoc />
    protected override void DeserializeNetworkSpecificData(System.IO.BinaryReader reader)
    {
        _dataWidth = reader.ReadInt32();
        _usingCustomLayers = reader.ReadBoolean();
        IsFitted = reader.ReadBoolean();
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new OCTGANGenerator<T>(Architecture, _options, _optimizer, _lossFunction);
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
            useResidualConcat: false);
    }

    #endregion
}
