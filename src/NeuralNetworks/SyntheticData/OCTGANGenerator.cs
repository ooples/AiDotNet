using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.SyntheticData;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.SyntheticDataGenerator)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("OCT-GAN: Neural ODE-based Conditional Tabular GANs",
    "https://arxiv.org/abs/2105.14969",
    Year = 2021,
    Authors = "Jayoung Kim, Jinsung Jeon, Jaehoon Lee, Jihyeon Hyeong, Noseong Park")]
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
    /// Initializes a new OCT-GAN generator with default configuration.
    /// </summary>
    private const int DefaultDataDimension = 128;

    public OCTGANGenerator()
        : this(new NeuralNetworkArchitecture<T>(InputType.OneDimensional, NeuralNetworkTaskType.Generative, inputSize: DefaultDataDimension, outputSize: DefaultDataDimension))
    {
    }

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
            _genBNLayers.Add(new BatchNormalizationLayer<T>());
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
            _discLayers.Add(new FullyConnectedLayer<T>(dims[i], identity));
            _discLayerDims.Add((layerInput, dims[i]));
            _discDropoutLayers.Add(new DropoutLayer<T>(_options.DiscriminatorDropout));
        }

        int lastHidden = dims.Length > 0 ? dims[^1] : _dataWidth;
        _svddEmbeddingLayer = new FullyConnectedLayer<T>(_options.SVDDEmbeddingDimension, identity);
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

                SetTrainingMode(true);
                try
                {
                    for (int dStep = 0; dStep < _options.DiscriminatorSteps; dStep++)
                    {
                        TrainDiscriminatorStep(trainData, batchStart, batchEnd, scaledLr);
                    }
                    TrainGeneratorStep(actualBatchSize, scaledLr);
                }
                finally { SetTrainingMode(false); }
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


    #endregion

    /// <summary>
    /// Propagates the training/inference mode to the auxiliary sub-networks (generator batch-norm
    /// and discriminator dropout) that live outside the base <c>Layers</c> collection, so generation
    /// runs the generator batch-norm in inference mode with the learned running statistics.
    /// </summary>
    public override void SetTrainingMode(bool isTraining)
    {
        base.SetTrainingMode(isTraining);
        foreach (var bn in _genBNLayers) bn.SetTrainingMode(isTraining);
        foreach (var drop in _discDropoutLayers) drop.SetTrainingMode(isTraining);
    }

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

            current = ApplyReLU(current);
        }

        current = ConcatTensors(current, originalInput);
        current = Layers[^1].Forward(current);

        return ApplyOutputActivations(current);
    }

    private Tensor<T> DiscriminatorForward(Tensor<T> input, bool isTraining)
    {
        var current = input;

        for (int i = 0; i < _discLayers.Count; i++)
        {
            current = _discLayers[i].Forward(current);
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
        // DeepSVDD critic (Xu et al. 2021 OCT-GAN): pull real embeddings toward the
        // SVDD center and push generated ones away. Tape-connected; weight clipping
        // keeps the critic Lipschitz-bounded.
        for (int i = batchStart; i < batchEnd; i++)
        {
            var realRow = VectorToTensor(GetRow(trainData, i));
            var noise = VectorToTensor(CreateStandardNormalVector(_options.EmbeddingDimension));
            using var tape = new GradientTape<T>();
            var realEmb = DiscriminatorForward(realRow, isTraining: true);
            var fakeRow = GeneratorForward(noise);
            var fakeEmb = DiscriminatorForward(fakeRow, isTraining: true);
            var loss = Engine.TensorSubtract(SvddDistSq(realEmb), SvddDistSq(fakeEmb));
            TapeStepOver(tape, loss, BuildDiscriminatorLayerList());
        }
        ClipWeights(BuildDiscriminatorLayerList());
    }

    private void TrainGeneratorStep(int batchSize, T scaledLr)
    {
        // Generator pulls its embeddings toward the SVDD center (look real).
        for (int i = 0; i < batchSize; i++)
        {
            var noise = VectorToTensor(CreateStandardNormalVector(_options.EmbeddingDimension));
            using var tape = new GradientTape<T>();
            var fakeRow = GeneratorForward(noise);
            var fakeEmb = DiscriminatorForward(fakeRow, isTraining: false);
            var loss = SvddDistSq(fakeEmb);
            TapeStepOver(tape, loss, BuildGeneratorLayerList());
        }
    }

    #endregion

    #region Gradient Penalty



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



    #endregion

    #region Activation Functions

    private Tensor<T> ApplyReLU(Tensor<T> input) => Engine.TensorReLU(input);


    private Tensor<T> ApplyLeakyReLU(Tensor<T> input) => Engine.TensorLeakyReLU(input, NumOps.FromDouble(0.2));


    private Tensor<T> ApplyOutputActivations(Tensor<T> output)
    {
        if (_transformer is null) return output;
        var flat = output.Rank == 1 ? output : Engine.Reshape(output, new[] { output.Length });
        var blocks = new List<Tensor<T>>();
        int idx = 0;
        for (int col = 0; col < _columns.Count && idx < flat.Length; col++)
        {
            var transform = _transformer.GetTransformInfo(col);
            if (transform.IsContinuous)
            {
                blocks.Add(Engine.TensorTanh(Engine.TensorSlice(flat, new[] { idx }, new[] { 1 })));
                idx++;
                int numModes = transform.Width - 1;
                if (numModes > 0)
                {
                    blocks.Add(Engine.TensorSoftmax(Engine.TensorSlice(flat, new[] { idx }, new[] { numModes }), axis: 0));
                    idx += numModes;
                }
            }
            else
            {
                blocks.Add(Engine.TensorSoftmax(Engine.TensorSlice(flat, new[] { idx }, new[] { transform.Width }), axis: 0));
                idx += transform.Width;
            }
        }
        if (blocks.Count == 0) return flat;
        if (idx < flat.Length) blocks.Add(Engine.TensorSlice(flat, new[] { idx }, new[] { flat.Length - idx }));
        return Engine.TensorConcatenate(blocks.ToArray(), axis: 0);
    }


    #endregion

    #region Gradient Safety


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

    private Tensor<T> ConcatTensors(Tensor<T> a, Tensor<T> b)
    {
        var a1 = a.Rank == 1 ? a : Engine.Reshape(a, new[] { a.Length });
        var b1 = b.Rank == 1 ? b : Engine.Reshape(b, new[] { b.Length });
        return Engine.TensorConcatenate(new[] { a1, b1 }, axis: 0);
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

    #endregion

    #region NeuralNetworkBase Overrides

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
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
        // Treats the input as the generator's latent noise (deterministic). Works
        // before Fit (the generated ModelFamily tests call Predict without Fit).
        var noise = new Vector<T>(input.Length);
        for (int i = 0; i < input.Length; i++) noise[i] = input[i];
        return GeneratorForward(VectorToTensor(noise));
    }

    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Full OCT-GAN training (SVDD discriminator + generator) runs in Fit. This
        // single-step entry point trains the generator (the Layers chain) to
        // reconstruct the target via a tape step. The previous body was a no-op.
        SetTrainingMode(true);
        try
        {
            using var tape = new GradientTape<T>();
            var output = GeneratorForward(input);
            var flatOut = output.Rank == 1 ? output : Engine.Reshape(output, new[] { output.Length });
            var target = expectedOutput.Rank == 1 ? expectedOutput : Engine.Reshape(expectedOutput, new[] { expectedOutput.Length });
            var loss = ReduceToScalar(Engine.TensorSquare(Engine.TensorSubtract(flatOut, target)));
            TapeStepOver(tape, loss, BuildGeneratorLayerList());
        }
        finally { SetTrainingMode(false); }
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

        // The generator batch-norm layers (running mean/variance included) live outside the base
        // Layers collection, and the fitted VGM transformer drives inverse-transform + per-column
        // output activations. Both must be persisted or a loaded model generates garbage. (The SVDD
        // center is a discriminator-training anchor only, never used by Generate, and is rebuilt by
        // Fit, so it is intentionally not persisted.)
        AuxLayerSerialization.WriteLayerList(writer, _genBNLayers);
        if (_transformer is not null)
        {
            writer.Write(true);
            _transformer.Serialize(writer);
        }
        else
        {
            writer.Write(false);
        }
    }

    /// <inheritdoc />
    protected override void DeserializeNetworkSpecificData(System.IO.BinaryReader reader)
    {
        _dataWidth = reader.ReadInt32();
        _usingCustomLayers = reader.ReadBoolean();
        IsFitted = reader.ReadBoolean();

        AuxLayerSerialization.ReadLayerList<T, BatchNormalizationLayer<T>>(
            reader, _genBNLayers, (inShape, outShape) => new BatchNormalizationLayer<T>());

        bool hasTransformer = reader.ReadBoolean();
        if (hasTransformer)
        {
            _transformer = new TabularDataTransformer<T>(_options.VGMModes, _random);
            _transformer.Deserialize(reader);
            _columns = new List<ColumnMetadata>(_transformer.Columns);
        }
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new OCTGANGenerator<T>(Architecture, _options, _optimizer, _lossFunction);
    }

    #endregion


    #region Tape Step Helpers

    private void TapeStepOver(GradientTape<T> tape, Tensor<T> loss, IReadOnlyList<ILayer<T>> layers)
    {
        var trainable = Training.TapeTrainingStep<T>.CollectParameters(layers);
        if (trainable.Count == 0) return;
        var grads = tape.ComputeGradients(loss, trainable);
        T lossValue = loss.Length > 0 ? loss[0] : NumOps.Zero;
        LastLoss = lossValue;
        var ctx = new AiDotNet.Tensors.Engines.Autodiff.TapeStepContext<T>(trainable, grads, lossValue);
        _optimizer.Step(ctx);
    }

    private Tensor<T> ReduceToScalar(Tensor<T> t)
        => Engine.ReduceSum(t, Enumerable.Range(0, t.Shape.Length).ToArray(), keepDims: false);

    /// <summary>Tape-connected squared distance of an embedding to the (constant) SVDD center.</summary>
    private Tensor<T> SvddDistSq(Tensor<T> embedding)
    {
        if (_svddCenter is null) return ReduceToScalar(Engine.TensorSquare(embedding));
        var emb = embedding.Rank == 1 ? embedding : Engine.Reshape(embedding, new[] { embedding.Length });
        return ReduceToScalar(Engine.TensorSquare(Engine.TensorSubtract(emb, _svddCenter)));
    }

    private const double GanClip = 0.01;
    private void ClipWeights(IReadOnlyList<ILayer<T>> layers)
    {
        T lo = NumOps.FromDouble(-GanClip), hi = NumOps.FromDouble(GanClip);
        foreach (var layer in layers)
        {
            var ps = layer.GetParameters();
            if (ps.Length == 0) continue;
            bool changed = false;
            for (int i = 0; i < ps.Length; i++)
            {
                if (NumOps.GreaterThan(ps[i], hi)) { ps[i] = hi; changed = true; }
                else if (NumOps.LessThan(ps[i], lo)) { ps[i] = lo; changed = true; }
            }
            if (changed) layer.UpdateParameters(ps);
        }
    }

    private IReadOnlyList<ILayer<T>> BuildGeneratorLayerList()
    {
        var all = new List<ILayer<T>>(Layers);
        all.AddRange(_genBNLayers);
        return all;
    }

    private IReadOnlyList<ILayer<T>> BuildDiscriminatorLayerList()
    {
        var all = new List<ILayer<T>>(_discLayers);
        all.AddRange(_discDropoutLayers);
        if (_svddEmbeddingLayer is not null) all.Add(_svddEmbeddingLayer);
        return all;
    }

    #endregion
}
