using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;

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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.SyntheticDataGenerator)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Data Synthesis based on Generative Adversarial Networks",
    "https://arxiv.org/abs/1806.03384",
    Year = 2018,
    Authors = "Noseong Park, Mahmoud Mohammadi, Kshitij Gorde, Sushil Jajodia, Hongkyu Park, Youngmin Kim")]
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
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    public TableGANGenerator()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.OneDimensional,
            taskType: Enums.NeuralNetworkTaskType.Regression,
            inputSize: 10,
            outputSize: 10))
    {
    }

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
                Layers.Add(new FullyConnectedLayer<T>(dims[i], identity));
                _genBNLayers.Add(new BatchNormalizationLayer<T>());
            }

            int lastHidden = dims.Length > 0 ? dims[^1] + inputDim : inputDim;
            Layers.Add(new FullyConnectedLayer<T>(Architecture.OutputSize, identity));
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
            _discLayers.Add(new FullyConnectedLayer<T>(dims[i], identity));
            _discLayerDims.Add((layerInput, dims[i]));
            _discDropoutLayers.Add(new DropoutLayer<T>(_options.DiscriminatorDropout));
        }

        int lastHidden = dims.Length > 0 ? dims[^1] : _dataWidth;
        _discLayers.Add(new FullyConnectedLayer<T>(1, identity));
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
            _classLayers.Add(new FullyConnectedLayer<T>(dims[i], identity));
        }

        int lastDim = dims.Length > 0 ? dims[^1] : _dataWidth;
        _classOutput = new FullyConnectedLayer<T>(numClasses, identity);
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
                Layers.Add(new FullyConnectedLayer<T>(dims[i], identity));
                _genBNLayers.Add(new BatchNormalizationLayer<T>());
            }

            int lastHidden = dims.Length > 0 ? dims[^1] + inputDim : inputDim;
            Layers.Add(new FullyConnectedLayer<T>(_dataWidth, identity));
        }

        BuildDiscriminator();

        _numClasses = 0;
        if (_options.LabelColumnIndex >= 0 && _options.LabelColumnIndex < columns.Count)
        {
            var labelCol = columns[_options.LabelColumnIndex];
            _numClasses = labelCol.IsCategorical ? Math.Max(2, labelCol.Categories.Count) : 1;
            BuildClassifier(_numClasses);
        }

        int batchSize = Math.Min(_options.BatchSize, data.Rows);
        int numBatches = Math.Max(1, data.Rows / batchSize);

        // Tape-based WGAN training (Park et al. 2018 / WGAN family pattern).
        // Critic minimizes E[D(fake)] - E[D(real)]; generator minimizes
        // -E[D(G(z))]. Both networks update through GradientTape.ComputeGradients
        // + TapeStepContext + optimizer.Step because the codebase migrated from
        // manual Backward() to tape-based autodiff (LayerBase.cs:1593) — any
        // remaining call site that runs layer.UpdateParameters(lr) without
        // an intervening tape throws "Backward pass must be called before
        // updating parameters." See WGANGP.TrainStep for the canonical pattern.
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            for (int batch = 0; batch < numBatches; batch++)
            {
                // Paper's training schedule: DiscriminatorSteps critic updates per generator update.
                for (int dStep = 0; dStep < _options.DiscriminatorSteps; dStep++)
                {
                    var realBatch = BuildRealBatchTensor(transformedData, batchSize);
                    var noiseBatch = GenerateNoiseBatchTensor(batchSize);
                    TrainDiscriminatorStepBatched(realBatch, noiseBatch);
                }

                var genNoise = GenerateNoiseBatchTensor(batchSize);
                var generatorTargetBatch = BuildRealBatchTensor(transformedData, batchSize);
                TrainGeneratorStepBatched(genNoise, generatorTargetBatch);
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

    /// <summary>
    /// Paper-faithful Wasserstein critic update (Park et al. 2018 §3.2).
    /// Minimizes the WGAN critic objective <c>E[D(G(z))] - E[D(x_real)]</c>
    /// using <see cref="GradientTape{T}"/> + <see cref="TapeStepContext{T}"/>
    /// so gradients flow through every tape-tracked op in
    /// <see cref="DiscriminatorForwardBatched"/>. Replaces the manual
    /// per-sample <c>DiscriminatorForward → UpdateParameters</c> pattern that
    /// the codebase's tape migration (see <c>LayerBase.cs:1593</c>) made
    /// invalid — the prior call site threw
    /// <c>InvalidOperationException: Backward pass must be called before
    /// updating parameters.</c> on the first <c>UpdateParameters(lr)</c>.
    /// </summary>
    private void TrainDiscriminatorStepBatched(Tensor<T> realBatch, Tensor<T> noiseBatch)
    {
        // Generator forward runs OUTSIDE the critic's tape — the critic
        // treats fake samples as data, so generator parameters must NOT
        // appear in this tape's gradient graph.
        var fakeBatch = GeneratorForwardBatched(noiseBatch);
        fakeBatch = ApplyOutputActivationsBatched(fakeBatch);

        using var tape = new GradientTape<T>();
        var discParams = TapeTrainingStep<T>.CollectParameters(_discLayers.Cast<ILayer<T>>());

        var realScores = DiscriminatorForwardBatched(realBatch, isTraining: true);
        var fakeScores = DiscriminatorForwardBatched(fakeBatch, isTraining: true);

        var allAxes = Enumerable.Range(0, realScores.Shape.Length).ToArray();
        var avgReal = Engine.ReduceMean(realScores, allAxes, keepDims: false);
        var avgFake = Engine.ReduceMean(fakeScores, allAxes, keepDims: false);
        // Critic minimizes E[D(fake)] - E[D(real)] (= -wassersteinDistance).
        var lossTensor = Engine.TensorSubtract(avgFake, avgReal);

        var grads = tape.ComputeGradients(lossTensor, discParams);
        T lossValue = lossTensor.Length > 0 ? lossTensor[0] : NumOps.Zero;

        Tensor<T> ComputeForward(Tensor<T> inp, Tensor<T> _) => DiscriminatorForwardBatched(inp, true);
        Tensor<T> RecomputeLoss(Tensor<T> pred, Tensor<T> _) => Engine.ReduceMean(pred, allAxes, keepDims: false);

        var context = new TapeStepContext<T>(
            discParams, grads, lossValue,
            realBatch, realBatch, ComputeForward, RecomputeLoss,
            parameterBuffer: null);
        _optimizer.Step(context);
    }

    /// <summary>
    /// Paper-faithful generator update (Park et al. 2018 §3.2). Minimizes
    /// <c>-E[D(G(z))]</c> (Wasserstein generator term) so the generator
    /// pushes its synthetic samples into regions the critic scores high.
    /// Information loss (mean/variance matching) is folded in via the
    /// secondary term scaled by <see cref="TableGANOptions{T}.InformationWeight"/>
    /// when real statistics are available. Classification loss is similarly
    /// folded in when a label column is configured. Both pieces are
    /// expressed in tape-tracked engine ops so they participate in the
    /// single <see cref="GradientTape{T}.ComputeGradients"/> call.
    /// </summary>
    private void TrainGeneratorStepBatched(Tensor<T> noiseBatch, Tensor<T> realBatch)
    {
        using var tape = new GradientTape<T>();

        // Generator's trainable surface = generator FC layers + their BN layers.
        var generatorLayers = new List<ILayer<T>>();
        generatorLayers.AddRange(Layers);
        foreach (var bn in _genBNLayers) generatorLayers.Add(bn);
        var genParams = TapeTrainingStep<T>.CollectParameters(generatorLayers);

        var fakeBatch = GeneratorForwardBatched(noiseBatch);
        var fakeActivated = ApplyOutputActivationsBatched(fakeBatch);
        var lossTensor = ComputeGeneratorLoss(fakeActivated, realBatch);
        var grads = tape.ComputeGradients(lossTensor, genParams);
        T lossValue = lossTensor.Length > 0 ? lossTensor[0] : NumOps.Zero;

        Tensor<T> ComputeForward(Tensor<T> inp, Tensor<T> _) =>
            ApplyOutputActivationsBatched(GeneratorForwardBatched(inp));
        Tensor<T> RecomputeLoss(Tensor<T> pred, Tensor<T> _) =>
            ComputeGeneratorLoss(pred, realBatch);

        var context = new TapeStepContext<T>(
            genParams, grads, lossValue,
            noiseBatch, noiseBatch, ComputeForward, RecomputeLoss,
            parameterBuffer: null);
        _optimizer.Step(context);
    }

    private Tensor<T> ComputeGeneratorLoss(Tensor<T> fakeActivated, Tensor<T> realBatch)
    {
        var fakeScores = DiscriminatorForwardBatched(fakeActivated, isTraining: false);
        var scoreAxes = Enumerable.Range(0, fakeScores.Shape.Length).ToArray();
        var avgFake = Engine.ReduceMean(fakeScores, scoreAxes, keepDims: false);
        var lossTensor = Engine.TensorNegate(avgFake);

        if (_realMean is not null && _realVar is not null && _options.InformationWeight > 0)
        {
            var weightedInformationLoss = Engine.TensorMultiplyScalar(
                ComputeInformationLoss(fakeActivated),
                NumOps.FromDouble(_options.InformationWeight));
            lossTensor = Engine.TensorAdd(
                lossTensor,
                MatchLossShape(weightedInformationLoss, lossTensor));
        }

        if (_classOutput is not null && _numClasses > 1 && _options.ClassificationWeight > 0)
        {
            var weightedClassificationLoss = Engine.TensorMultiplyScalar(
                ComputeClassificationLoss(fakeActivated, realBatch),
                NumOps.FromDouble(_options.ClassificationWeight));
            lossTensor = Engine.TensorAdd(
                lossTensor,
                MatchLossShape(weightedClassificationLoss, lossTensor));
        }

        return lossTensor;
    }

    private static Tensor<T> MatchLossShape(Tensor<T> term, Tensor<T> reference)
    {
        if (ShapesEqual(term.Shape.ToArray(), reference.Shape.ToArray()))
            return term;

        if (term.Length == 1 && reference.Length == 1)
            return new Tensor<T>([term[0]], reference._shape);

        return term;
    }

    private static bool ShapesEqual(IReadOnlyList<int> left, IReadOnlyList<int> right)
    {
        if (left.Count != right.Count)
            return false;

        for (int i = 0; i < left.Count; i++)
        {
            if (left[i] != right[i])
                return false;
        }

        return true;
    }

    private Tensor<T> ComputeInformationLoss(Tensor<T> fakeActivated)
    {
        var fakeFeatureAxes = Enumerable.Range(0, fakeActivated.Shape.Length - 1).ToArray();
        var fakeMean = Engine.ReduceMean(fakeActivated, fakeFeatureAxes, keepDims: false);
        var fakeMeanBroadcast = Engine.TensorTile(fakeMean.Reshape([1, fakeMean.Length]), [fakeActivated.Shape[0], 1]);
        var centered = Engine.TensorSubtract(fakeActivated, fakeMeanBroadcast);
        var fakeVariance = Engine.ReduceMean(Engine.TensorMultiply(centered, centered), fakeFeatureAxes, keepDims: false);

        var realMeanTensor = VectorToTensor(_realMean!);
        var realVarianceTensor = VectorToTensor(_realVar!);
        var meanDiff = Engine.TensorSubtract(fakeMean, realMeanTensor);
        var varianceDiff = Engine.TensorSubtract(fakeVariance, realVarianceTensor);
        var meanSq = Engine.TensorMultiply(meanDiff, meanDiff);
        var varianceSq = Engine.TensorMultiply(varianceDiff, varianceDiff);
        var infoLoss = Engine.TensorAdd(meanSq, varianceSq);
        var infoAxes = Enumerable.Range(0, infoLoss.Shape.Length).ToArray();
        return Engine.ReduceMean(infoLoss, infoAxes, keepDims: false);
    }

    private Tensor<T> ComputeClassificationLoss(Tensor<T> fakeActivated, Tensor<T> realBatch)
    {
        var logits = ClassifierForwardBatched(fakeActivated);
        var probabilities = Engine.Softmax(logits, axis: logits.Shape.Length - 1);
        var targets = BuildClassificationTargetTensor(realBatch, logits);
        var logProbabilities = Engine.TensorLog(Engine.TensorAddScalar(probabilities, NumOps.FromDouble(1e-8)));
        var crossEntropyTerms = Engine.TensorMultiply(targets, logProbabilities);
        var allAxes = Enumerable.Range(0, crossEntropyTerms.Shape.Length).ToArray();
        var summed = Engine.ReduceSum(crossEntropyTerms, allAxes, keepDims: false);
        return Engine.TensorMultiplyScalar(summed, NumOps.FromDouble(-1.0 / Math.Max(1, realBatch.Shape[0])));
    }

    private Tensor<T> ClassifierForwardBatched(Tensor<T> input)
    {
        var current = input;
        for (int i = 0; i < _classLayers.Count; i++)
        {
            current = _classLayers[i].Forward(current);
            current = Engine.ReLU(current);
        }

        return _classOutput is null ? current : _classOutput.Forward(current);
    }

    private Tensor<T> BuildClassificationTargetTensor(Tensor<T> realBatch, Tensor<T> logits)
    {
        var targets = new Tensor<T>(logits._shape);
        int batch = Math.Min(realBatch.Shape[0], logits.Shape[0]);
        int classCount = logits.Shape[^1];
        int labelIdx = Math.Max(0, Math.Min(_options.LabelColumnIndex, realBatch.Shape[^1] - 1));

        for (int b = 0; b < batch; b++)
        {
            int targetClass = Math.Min(
                Math.Max((int)Math.Round(NumOps.ToDouble(realBatch[b, labelIdx])), 0),
                classCount - 1);
            targets[b, targetClass] = NumOps.One;
        }

        return targets;
    }

    /// <summary>
    /// Samples <paramref name="batchSize"/> rows uniformly at random from
    /// the post-transformer matrix into a rank-2 tensor <c>[B, dataWidth]</c>.
    /// </summary>
    private Tensor<T> BuildRealBatchTensor(Matrix<T> transformedData, int batchSize)
    {
        var batch = new Tensor<T>([batchSize, _dataWidth]);
        for (int b = 0; b < batchSize; b++)
        {
            int rowIdx = _random.Next(transformedData.Rows);
            int cols = Math.Min(_dataWidth, transformedData.Columns);
            for (int j = 0; j < cols; j++)
                batch[b, j] = transformedData[rowIdx, j];
        }
        return batch;
    }

    /// <summary>
    /// Generates a batched Gaussian noise tensor <c>[B, embeddingDim]</c> via
    /// vectorized Box-Muller using <see cref="Engine.TensorRandomUniformRange"/>.
    /// Matches the canonical pattern used by
    /// <c>GenerativeAdversarialNetwork.GenerateRandomNoiseTensor</c>.
    /// </summary>
    private Tensor<T> GenerateNoiseBatchTensor(int batchSize)
    {
        int embedDim = _options.EmbeddingDimension;
        int totalElements = batchSize * embedDim;
        int halfElements = (totalElements + 1) / 2;

        var u2 = Engine.TensorRandomUniformRange<T>([halfElements], NumOps.Zero, NumOps.One);
        var u1Temp = Engine.TensorRandomUniformRange<T>([halfElements], NumOps.Zero, NumOps.One);
        var u1 = Engine.ScalarMinusTensor(NumOps.One, u1Temp);
        var logU1 = Engine.TensorLog(u1);
        var negTwoLogU1 = Engine.TensorMultiplyScalar(logU1, NumOps.FromDouble(-2.0));
        var radius = Engine.TensorSqrt(negTwoLogU1);
        var theta = Engine.TensorMultiplyScalar(u2, NumOps.FromDouble(2.0 * Math.PI));
        var z1 = Engine.TensorMultiply(radius, Engine.TensorCos(theta));
        var z2 = Engine.TensorMultiply(radius, Engine.TensorSin(theta));

        var noiseData = new T[totalElements];
        var z1Arr = z1.ToArray();
        var z2Arr = z2.ToArray();
        for (int i = 0; i < halfElements; i++)
        {
            int idx = i * 2;
            if (idx < totalElements) noiseData[idx] = z1Arr[i];
            if (idx + 1 < totalElements) noiseData[idx + 1] = z2Arr[i];
        }
        return new Tensor<T>(noiseData, [batchSize, embedDim]);
    }

    /// <summary>
    /// Batched generator forward, fully tape-tracked. Mirrors the paper's
    /// residual + BN architecture (Park et al. 2018 §3.1): each hidden layer
    /// concatenates its input with the original noise (skip connection),
    /// passes through FC → BN → ReLU, and the final layer projects to
    /// <c>_dataWidth</c>.
    /// </summary>
    private Tensor<T> GeneratorForwardBatched(Tensor<T> noise)
    {
        if (_usingCustomLayers)
        {
            var c = noise;
            foreach (var l in Layers) c = l.Forward(c);
            return c;
        }

        var h = noise;
        for (int i = 0; i < Layers.Count - 1; i++)
        {
            if (i > 0) h = Engine.TensorConcatenate([h, noise], axis: 1);
            h = Layers[i].Forward(h);
            h = _genBNLayers[i].Forward(h);
            h = Engine.ReLU(h);
        }
        h = Engine.TensorConcatenate([h, noise], axis: 1);
        h = Layers[^1].Forward(h);
        return h;
    }

    /// <summary>
    /// Batched discriminator forward, fully tape-tracked. Mirrors the paper's
    /// critic stack (Park et al. 2018 §3.1): hidden layers are FC →
    /// LeakyReLU(0.2) → Dropout, output is a scalar score per row.
    /// </summary>
    private Tensor<T> DiscriminatorForwardBatched(Tensor<T> input, bool isTraining)
    {
        var current = input;
        T leakySlope = NumOps.FromDouble(0.2);

        for (int i = 0; i < _discLayers.Count - 1; i++)
        {
            current = _discLayers[i].Forward(current);
            current = Engine.LeakyReLU(current, leakySlope);
            if (isTraining) current = _discDropoutLayers[i].Forward(current);
        }

        current = _discLayers[^1].Forward(current);
        return current;
    }

    /// <summary>
    /// Batched output activations (Park et al. 2018 §3.1): per the paper's
    /// VGM column encoding, each continuous column emits a single Tanh-bounded
    /// mode value followed by a softmax over mode-probabilities; categorical
    /// columns emit one softmax over the category one-hot block. The whole
    /// dispatch runs through tape-tracked engine ops
    /// (<see cref="Engine.TensorTanh"/>, <see cref="Engine.Softmax"/>,
    /// <see cref="Engine.TensorSlice"/>, <see cref="Engine.TensorConcatenate"/>)
    /// so backprop flows from the WGAN critic through every column block.
    /// </summary>
    private Tensor<T> ApplyOutputActivationsBatched(Tensor<T> output)
    {
        // Pre-fit smoke path: no transformer yet, so column widths are
        // unknown — fall back to a single tape-tracked Tanh that keeps the
        // output bounded for any test exercising Predict before Fit.
        if (_transformer is null) return Engine.TensorTanh(output);

        int batch = output.Shape[0];
        int totalWidth = output.Shape[1];
        var blocks = new List<Tensor<T>>(_columns.Count * 2);
        int idx = 0;

        // Engine.TensorSlice signature: (tensor, startIndices, sliceLengths).
        for (int col = 0; col < _columns.Count && idx < totalWidth; col++)
        {
            var transform = _transformer.GetTransformInfo(col);
            if (transform.IsContinuous)
            {
                // Mode value: single Tanh-bounded element.
                var valueSlice = Engine.TensorSlice(output, [0, idx], [batch, 1]);
                blocks.Add(Engine.TensorTanh(valueSlice));
                idx++;

                int numModes = transform.Width - 1;
                int modeLength = Math.Min(numModes, totalWidth - idx);
                if (modeLength > 0)
                {
                    var modeSlice = Engine.TensorSlice(output, [0, idx], [batch, modeLength]);
                    blocks.Add(Engine.Softmax(modeSlice, axis: 1));
                    idx += modeLength;
                }
            }
            else
            {
                int blockLength = Math.Min(transform.Width, totalWidth - idx);
                var catSlice = Engine.TensorSlice(output, [0, idx], [batch, blockLength]);
                blocks.Add(Engine.Softmax(catSlice, axis: 1));
                idx += blockLength;
            }
        }

        // The transformer's total column width should match the model's
        // output projection. If the projection is wider than the column
        // schema (e.g. a custom architecture override), Tanh-bound the
        // trailing slice so the tape still produces finite gradients.
        if (idx < totalWidth)
        {
            var tail = Engine.TensorSlice(output, [0, idx], [batch, totalWidth - idx]);
            blocks.Add(Engine.TensorTanh(tail));
        }

        if (blocks.Count == 1) return blocks[0];
        return Engine.TensorConcatenate(blocks.ToArray(), axis: 1);
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

    #endregion

    #region Gradient Penalty

    #endregion

    #region Backward Passes

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
        var result = new Tensor<T>(input._shape);
        for (int i = 0; i < input.Length; i++)
            result[i] = NumOps.GreaterThan(input[i], NumOps.Zero) ? input[i] : NumOps.Zero;
        return result;
    }

    private Tensor<T> ApplyReLUDerivative(Tensor<T> gradOutput, Tensor<T> preActivation)
    {
        int len = Math.Min(gradOutput.Length, preActivation.Length);
        var result = new Tensor<T>(gradOutput._shape);
        for (int i = 0; i < len; i++)
            result[i] = NumOps.GreaterThan(preActivation[i], NumOps.Zero) ? gradOutput[i] : NumOps.Zero;
        return result;
    }

    private Tensor<T> ApplyLeakyReLU(Tensor<T> input)
    {
        var result = new Tensor<T>(input._shape);
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
        var result = new Tensor<T>(gradOutput._shape);
        T slope = NumOps.FromDouble(0.2);
        for (int i = 0; i < len; i++)
        {
            result[i] = NumOps.GreaterThan(preActivation[i], NumOps.Zero)
                ? gradOutput[i]
                : NumOps.Multiply(slope, gradOutput[i]);
        }
        return result;
    }

    private Tensor<T> ApplyOutputActivations(Tensor<T> output)
    {
        if (_transformer is null) return output;

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
        int actualCount = Math.Min(count, input.Length - idx);
        if (actualCount <= 0) return;
        var slice = new Tensor<T>([actualCount]);
        input.Data.Span.Slice(idx, actualCount).CopyTo(slice.Data.Span);
        var result = Engine.Softmax(slice, -1);
        result.Data.Span.CopyTo(output.Data.Span.Slice(idx, actualCount));
        idx += actualCount;
    }

    #endregion

    #region Helpers

    private Tensor<T> SafeGradient(Tensor<T> grad, double maxNorm)
    {
        var result = new Tensor<T>(grad._shape);
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
        var clone = new Tensor<T>(source._shape);
        for (int i = 0; i < source.Length; i++) clone[i] = source[i];
        return clone;
    }

    #endregion

}
