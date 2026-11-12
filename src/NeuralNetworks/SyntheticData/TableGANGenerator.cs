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
public partial class TableGANGenerator<T> : NeuralSyntheticTabularGeneratorBase<T>, ISyntheticTabularGenerator<T>
{
    /// <inheritdoc/>
    /// <remarks>
    /// TableGAN trains adversarially (generator, discriminator and classifier) through <c>Fit</c>; the supervised
    /// <c>Train(input, target)</c> contract does not map onto its minimax objective, so it takes no step.
    /// </remarks>
    protected override IReadOnlyList<AiDotNet.Training.TrainingObjective<T>>? DefineTrainingObjectives(Tensor<T> input, Tensor<T> target)
        => Array.Empty<AiDotNet.Training.TrainingObjective<T>>();
    // Held across Train calls. The fused plan owns the optimizer state, so a step object built per call
    // re-traced and recompiled every call and restarted the optimizer at t = 1 (Adam ~ lr*sign(g)).
    private AiDotNet.Training.WganGpFusedStep<T>? _wganFusedStep;

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _wganFusedStep?.Dispose();
            _wganFusedStep = null;
        }
        base.Dispose(disposing);
    }

    private readonly TableGANOptions<T> _options;
    // Separate G/D optimizers (see CTGANGenerator for the divergence rationale).
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _generatorOptimizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _discriminatorOptimizer;
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

    // Real rows as a tensor, cached per transformed matrix for batch gathering
    [Scratch]
    private Matrix<T>? _realDataSource;
    [Scratch]
    private Tensor<T>? _realDataTensor;

    // Real data statistics for information loss
    [Buffer(Availability = AiDotNet.Models.Parameters.ParameterAvailability.Conditional)]
    private Vector<T>? _realMean;
    [Buffer(Availability = AiDotNet.Models.Parameters.ParameterAvailability.Conditional)]
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
    public TableGANGenerator(NeuralNetworkArchitecture<T> architecture,
        TableGANOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), (options ??= new TableGANOptions<T>()).MaxGradNorm)
    {
        _options = options;
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);
        AdamOptimizer<T, Tensor<T>, Tensor<T>> MakeAdam() =>
            new(this, new Models.Options.AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate,
                Beta1 = 0.5,
                Beta2 = 0.9,
                UseAdaptiveLearningRate = false,
                UseAMSGrad = false,
            });
        _generatorOptimizer = optimizer ?? MakeAdam();
        _discriminatorOptimizer = MakeAdam();
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
    public void Fit(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int? epochs = null)
    {
        try
        {
            FitCore(data, columns, epochs);
        }
        finally
        {
            // The cached training rows only serve the training loop: a fitted generator, or one whose Fit failed,
            // does not keep a copy of its training data.
            _realDataSource = null;
            _realDataTensor = null;
        }
    }

    private void FitCore(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int? epochs)
    {
        int epochCount = epochs ?? _options.Epochs;
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
        for (int epoch = 0; epoch < epochCount; epoch++)
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
    public async Task FitAsync(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int? epochs = null,
        CancellationToken cancellationToken = default)
    {
        int epochCount = epochs ?? _options.Epochs;
        await Task.Run(() =>
        {
            cancellationToken.ThrowIfCancellationRequested();
            Fit(data, columns, epochCount);
        }, cancellationToken);
    }

    /// <inheritdoc />
    public Matrix<T> Generate(int numSamples, Vector<T>? conditionColumn = null, Vector<T>? conditionValue = null)
    {
        if (_transformer is null || Layers.Count == 0)
        {
            throw new InvalidOperationException("Generator is not fitted. Call Fit() first.");
        }

        var generated = new Tensor<T>(new[] { Math.Max(0, numSamples), _dataWidth });

        // Rows are generated one at a time, in order, so each row's noise draws keep their RNG order. Each activated
        // row is written into one buffer as it is produced.
        for (int i = 0; i < numSamples; i++)
        {
            var noise = CreateStandardNormalVector(_options.EmbeddingDimension);
            SyntheticTensorOps<T>.WriteRow(generated, i, ApplyOutputActivations(GeneratorForward(noise)), _dataWidth);
        }

        TabularDataTransformer<T> transformer = _transformer;
        return SyntheticTensorOps<T>.DecodeInBlocks(generated, block => transformer.InverseTransform(block.ToMatrix()));
    }

    #endregion

    #region Forward Passes

    private Tensor<T> GeneratorForward(Vector<T> noise)
    {
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
            h = ApplyReLU(h);
        }

        h = ConcatTensors(h, inputTensor);
        h = Layers[^1].Forward(h);
        return h;
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

        // The shared WGAN-GP critic step: the GPU-resident primitive, the fused compiled plan or the shared eager
        // tape step, with one replay-safe gradient penalty (Gulrajani et al. 2017; #1844 / #1845).
        AiDotNet.Training.WassersteinCriticStep<T>.Step(
            this,
            _discLayers,
            realBatch,
            fakeBatch,
            input => DiscriminatorForwardBatched(input, isTraining: true),
            _options.GradientPenaltyWeight,
            _discriminatorOptimizer,
            _random,
            ref _wganFusedStep);
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
        // Generator's trainable surface = generator FC layers + their BN layers.
        var generatorLayers = new List<ILayer<T>>();
        generatorLayers.AddRange(Layers);
        foreach (var bn in _genBNLayers) generatorLayers.Add(bn);
        var genParams = TapeTrainingStep<T>.CollectParameters(generatorLayers);

        // GPU-RESIDENT fast path — noise → gen → activation → composite loss
        // (fake-scores + info-loss + optional classification). Fused SGD/Adam
        // step; disc/classifier layers not in trainable set so their weights stay frozen.
        var trainableGenLayers = generatorLayers.OfType<ITrainableLayer<T>>().ToList();
        if (trainableGenLayers.Count > 0)
        {
            Tensor<T> Fwd(Tensor<T> nb) => ApplyOutputActivationsBatched(GeneratorForwardBatched(nb));
            Tensor<T> Loss(Tensor<T> fakeActivated, Tensor<T> real) => ComputeGeneratorLoss(fakeActivated, real);
            // One step through the shared training step: the fused plan when it applies, otherwise the shared
            // eager tape step on the same objective. The critic scores the samples in evaluation mode and is not
            // in the generator's parameter group, so only the generator moves.
            AiDotNet.Training.FusedTrainingStep<T>.Step(
                this, trainableGenLayers, noiseBatch, realBatch,
                forward: Fwd,
                computeLoss: Loss,
                optimizer: _generatorOptimizer,
                extraTensors: genParams);
        }
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

        // Classification auxiliary term is gated on _options.TrainClassifier
        // (default false). Without a dedicated classifier-training routine
        // updating _classLayers / _classOutput against real (row, label)
        // pairs, the classifier logits are effectively random and the
        // weighted classification gradient is noise — enabling
        // ClassificationWeight alone silently degrades label fidelity rather
        // than preserving it. Once a real classifier-training path is
        // implemented, callers can set TrainClassifier=true to opt back in.
        if (_options.TrainClassifier
            && _classOutput is not null && _numClasses > 1 && _options.ClassificationWeight > 0)
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

        var realMeanTensor = VectorToTensor((_realMean ?? throw new InvalidOperationException("_realMean is null where it is required.")));
        var realVarianceTensor = VectorToTensor((_realVar ?? throw new InvalidOperationException("_realVar is null where it is required.")));
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
        if (_transformer is null ||
            _options.LabelColumnIndex < 0 ||
            _options.LabelColumnIndex >= _columns.Count)
        {
            return targets;
        }

        var labelTransform = _transformer.GetTransformInfo(_options.LabelColumnIndex);
        int batch = Math.Min(realBatch.Shape[0], logits.Shape[0]);
        int classCount = logits.Shape[^1];
        int labelWidth = Math.Min(labelTransform.Width, classCount);
        if (batch <= 0 || classCount <= 0)
        {
            return targets;
        }

        var engine = Engine;
        Tensor<T> oneHot;
        if (labelTransform.IsContinuous || labelWidth <= 0)
        {
            // Continuous label: the class is the value rounded half-to-even (Math.Round) and clamped to the
            // class range; a NaN value maps to class 0. An empty label block is class 0.
            var classValue = labelTransform.IsContinuous
                ? engine.TensorClamp(
                    engine.TensorRound(engine.TensorNanToNum(
                        engine.TensorSlice(realBatch, [0, labelTransform.StartOffset], [batch, 1]), 0.0,
                        double.PositiveInfinity, double.NegativeInfinity)),
                    NumOps.Zero, NumOps.FromDouble(classCount - 1))
                : new Tensor<T>([batch, 1]);
            var classIds = engine.TensorArange<T>(0, classCount, 1).Reshape([1, classCount]);
            oneHot = engine.TensorEquals(
                engine.TensorTile(classValue, [1, classCount]),
                engine.TensorTile(classIds, [batch, 1]));
        }
        else
        {
            // Categorical label: the first largest entry of the label block. NaN never wins (it compares false),
            // so it is ranked below every number; an all-NaN block is class 0.
            var block = engine.TensorNanToNum(
                engine.TensorSlice(realBatch, [0, labelTransform.StartOffset], [batch, labelWidth]),
                double.NegativeInfinity, double.PositiveInfinity, double.NegativeInfinity);
            var rowMax = engine.ReduceMax(block, [1], keepDims: true);
            var isMax = engine.TensorEquals(block, engine.TensorTile(rowMax, [1, labelWidth]));
            var firstMax = engine.TensorMultiply(isMax, engine.TensorEquals(engine.TensorCumSum(isMax, 1), NumOps.One));
            oneHot = labelWidth < classCount
                ? engine.TensorConcatenate([firstMax, new Tensor<T>([batch, classCount - labelWidth])], 1)
                : firstMax;
        }

        if (batch < logits.Shape[0])
        {
            oneHot = engine.TensorConcatenate([oneHot, new Tensor<T>([logits.Shape[0] - batch, classCount])], 0);
        }

        return oneHot.Reshape(logits._shape);
    }

    /// <summary>
    /// Samples <paramref name="batchSize"/> rows uniformly at random from
    /// the post-transformer matrix into a rank-2 tensor <c>[B, dataWidth]</c>.
    /// </summary>
    private Tensor<T> BuildRealBatchTensor(Matrix<T> transformedData, int batchSize)
    {
        var rowIndices = new int[batchSize];
        for (int b = 0; b < batchSize; b++)
        {
            rowIndices[b] = _random.Next(transformedData.Rows);
        }

        if (_realDataTensor is null || !ReferenceEquals(_realDataSource, transformedData))
        {
            _realDataTensor = Tensor<T>.FromMatrix(transformedData);
            _realDataSource = transformedData;
        }

        return SyntheticTensorOps<T>.GatherRows(_realDataTensor, rowIndices, _dataWidth);
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

        var u2 = Engine.TensorRandomUniformRange<T>([halfElements], NumOps.Zero, NumOps.One, _random.Next());
        var u1Temp = Engine.TensorRandomUniformRange<T>([halfElements], NumOps.Zero, NumOps.One, _random.Next());
        var u1 = Engine.ScalarMinusTensor(NumOps.One, u1Temp);
        var logU1 = Engine.TensorLog(u1);
        var negTwoLogU1 = Engine.TensorMultiplyScalar(logU1, NumOps.FromDouble(-2.0));
        var radius = Engine.TensorSqrt(negTwoLogU1);
        var theta = Engine.TensorMultiplyScalar(u2, NumOps.FromDouble(2.0 * Math.PI));
        var z1 = Engine.TensorMultiply(radius, Engine.TensorCos(theta));
        var z2 = Engine.TensorMultiply(radius, Engine.TensorSin(theta));
        return SyntheticTensorOps<T>.InterleaveNormalPairs(z1, z2, batchSize, embedDim);
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

    #endregion

    #region Gradient Penalty

    #endregion

    #region Backward Passes

    #endregion

    #region NeuralNetworkBase Required Overrides

    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// Mirrors <see cref="GeneratorForward(Vector{T})"/>'s noise-skip pattern.
    /// Per Park et al. 2018 §3.2 the TableGAN generator concatenates the
    /// original noise back into each hidden layer's input (a residual-style
    /// skip from <c>z</c>): layer 0 takes raw <c>z</c>, but layers 1..N-1
    /// take <c>[h_{i-1}; z]</c>. The previous naïve <c>foreach</c>
    /// implementation worked at construction time (when the layer chain
    /// hadn't yet been re-sized by Fit) but broke after Fit, because Fit
    /// re-creates the chain with the noise-concatenated input dims —
    /// running raw <c>foreach</c> through it produces the shape mismatch
    /// (<c>[1, 256] × [356, 256]</c>) seen on
    /// <c>TableGANGeneratorTests.Fit_TinyDataset_MarksGeneratorAsFitted</c>.
    /// </para>
    /// </remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        if (_usingCustomLayers)
        {
            // User-supplied architecture: no implicit noise-skip
            // contract; respect the caller's chain verbatim.
            var current = input;
            foreach (var layer in Layers) current = layer.Forward(current);
            return current;
        }

        // Default Park et al. 2018 architecture — noise concatenated
        // into every hidden-layer input. Mirrors GeneratorForward.
        var inputTensor = input;
        var h = inputTensor;
        for (int i = 0; i < Layers.Count - 1; i++)
        {
            if (i > 0) h = ConcatTensors(h, inputTensor);
            h = Layers[i].Forward(h);
            // Inference: BatchNorm in eval mode, ReLU activation.
            if (i < _genBNLayers.Count) h = _genBNLayers[i].Forward(h);
            h = ApplyReLU(h);
        }

        h = ConcatTensors(h, inputTensor);
        h = Layers[^1].Forward(h);
        return h;
    }

    // UpdateParameters re-sliced the flat vector across Layers by hand -- the base walks
    // exactly the same enumeration, so this said nothing the base does not already say.
    /// <inheritdoc />


    /// <inheritdoc />


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
        // Column means and unbiased (n - 1) variances.
        var engine = Engine;
        var x = Tensor<T>.FromMatrix(data);
        var mean = engine.ReduceMean(x, [0], keepDims: true);
        var centered = engine.TensorSubtract(x, engine.TensorTile(mean, [data.Rows, 1]));
        var sumSquares = engine.ReduceSum(engine.TensorMultiply(centered, centered), [0], keepDims: false);
        _realMean = mean.Reshape([data.Columns]).ToVector();
        _realVar = engine.TensorDivideScalar(sumSquares, NumOps.FromDouble(Math.Max(1, data.Rows - 1))).ToVector();
    }

    #endregion

    #region Activation Functions

    private Tensor<T> ApplyReLU(Tensor<T> input) => SyntheticTensorOps<T>.ReLU(input);

    private Tensor<T> ApplyOutputActivations(Tensor<T> output)
    {
        if (_transformer is null) return output;
        return SyntheticTensorOps<T>.ApplyOutputActivations(output, _transformer, _columns.Count);
    }

    #endregion

    #region Helpers

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

    private static Tensor<T> VectorToTensor(Vector<T> v) => Tensor<T>.FromVector(v);

    private static Tensor<T> ConcatTensors(Tensor<T> a, Tensor<T> b) => SyntheticTensorOps<T>.Concat1D(a, b);

    #endregion

}
