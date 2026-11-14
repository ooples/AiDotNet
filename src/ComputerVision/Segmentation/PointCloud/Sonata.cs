using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.ComputerVision.Segmentation.PointCloud;

/// <summary>
/// Sonata: self-supervised learning of reliable point representations (Wu et al., CVPR 2025) -- a
/// Point Transformer V3 encoder pretrained by point self-distillation, read out by a linear probe.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Sonata learns what the points of a 3D scan mean without any labels: it
/// shows a network several cropped, rotated and partly hidden copies of the same scene and trains it
/// to describe each point the same way in every copy as a slowly-updated copy of itself (the
/// teacher) does. Call <see cref="Pretrain"/> on unlabelled scans, then train the small classifier
/// on top with a few labelled ones.
/// </para>
/// <para><b>What is reproduced</b> (Pointcept <c>Sonata-v1m1</c>):</para>
/// <list type="bullet">
/// <item>Backbone: PTv3 <c>PT-v3m2</c>, encoder-only (the decoder is removed), with a learnable mask
/// token, LayerNorm, depths (3, 3, 3, 12, 3) and widths (48, 96, 192, 384, 512); xyz, colour and normal
/// as input features.</item>
/// <item>Feature up-casting: pooled features are concatenated back onto their parent points --
/// <see cref="SonataOptions.UpCastLevel"/> levels for the distillation heads, every level (1232
/// channels) for the linear probe.</item>
/// <item>EMA teacher with drop path 0 and a cosine momentum schedule 0.994 -&gt; 1.</item>
/// <item>Views: 2 global and 4 local crops; the global views are masked (cubes 0.1 -&gt; 0.4 m,
/// ratio 0.3 -&gt; 0.7) for the student.</item>
/// <item>Losses: masked view vs its own global view (mask), vs the other global view (roll mask) and
/// local views vs the principal global view (unmask), each a cross-entropy between the student's
/// prototype scores and Sinkhorn-Knopp-centred teacher scores of the nearest matched point, using
/// <c>OnlineCluster</c> prototype heads (4096 prototypes).</item>
/// </list>
/// <para>
/// The losses, the target centering and an optional embedding regulariser (KoLeo) are pluggable
/// through <see cref="ISonataDistillationLoss{T}"/>, <see cref="ISonataTargetCentering{T}"/> and
/// <see cref="ISonataFeatureRegularizer{T}"/>.
/// </para>
/// <para>
/// <b>Input</b>: <c>[points, InputChannels]</c> or <c>[batch, points, InputChannels]</c> whose first
/// three channels are xyz in metres. <b>Output</b>: class logits <c>[..., points, NumClasses]</c>.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.ThreeD)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Segmentation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Sonata: Self-Supervised Learning of Reliable Point Representations", "https://arxiv.org/abs/2503.16429",
    Year = 2025, Authors = "Xiaoyang Wu, Daniel DeTone, Duncan Frost, Tianwei Shen, Chris Xie, Nan Yang, Jakob Engel, Richard Newcombe, Hengshuang Zhao, Julian Straub")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Length, TensorAxis.Features, Direction = TensorLayoutDirection.Input, BatchOptional = true)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Length, TensorAxis.Classes, Direction = TensorLayoutDirection.Output, BatchOptional = true)]
public partial class Sonata<T> : Common.SemanticSegmentationBase<T>
{
    private readonly SonataOptions _options;
    private PointTransformerV3Encoder<T>? _teacher;
    private PtContext<T>? _studentHeadContext;
    private PtContext<T>? _teacherHeadContext;
    private SonataPrototypeHead<T>? _studentMaskHead;
    private SonataPrototypeHead<T>? _studentUnmaskHead;
    private SonataPrototypeHead<T>? _teacherMaskHead;
    private SonataPrototypeHead<T>? _teacherUnmaskHead;
    private Random? _viewRandom;

    /// <summary>
    /// Initializes Sonata in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture; custom layers in it replace the default network.</param>
    /// <param name="optimizer">Optimizer for supervised training of the probe (default: AdamW).</param>
    /// <param name="lossFunction">Loss function (default: per-point cross-entropy over the last axis).</param>
    /// <param name="options">Model options; the defaults are the Sonata v1m1 base configuration.</param>
    public Sonata(NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        SonataOptions? options = null)
        : base(architecture, optimizer, lossFunction ?? new CrossEntropyWithLogitsLoss<T>(-1),
            (options ??= new SonataOptions()).NumClasses)
    {
        _options = options;
        Options = _options;
        _options.Validate();
        _channels = _options.InputChannels;
        InitializeLayers();
    }

    /// <summary>
    /// Initializes Sonata in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="options">Optional model options.</param>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public Sonata(NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        SonataOptions? options = null)
        : base(architecture, onnxModelPath, (options ??= new SonataOptions()).NumClasses)
    {
        _options = options;
        Options = _options;
        _options.Validate();
        _channels = _options.InputChannels;
        InitializeLayers();
    }

    internal bool UseNativeMode => _useNativeMode;

    /// <summary>The student encoder, or null in ONNX mode or with custom architecture layers.</summary>
    /// <remarks>Read from the layer list, so it follows a deserialized or cloned model.</remarks>
    internal PointTransformerV3Encoder<T>? Encoder => _useNativeMode && Layers.Count > 0 ? Layers[0] as PointTransformerV3Encoder<T> : null;

    internal PointTransformerV3Encoder<T>? Teacher => _teacher;

    /// <inheritdoc />
    protected override bool SupportsFusedCompiledTraining => false;

    /// <inheritdoc />
    public override IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
        => PointTransformerV3<T>.PointCloudContract(inputRank, _numClasses);

    /// <summary>Argmax over the class (last) axis: one class index per point.</summary>
    public override Tensor<T> GetClassMap(Tensor<T> image) => PointTransformerV3<T>.LastAxisArgMax(Segment(image));

    /// <summary>Softmax over the class (last) axis.</summary>
    public override Tensor<T> GetProbabilityMap(Tensor<T> image) => Engine.TensorSoftmax(Segment(image), -1);

    /// <inheritdoc />
    protected override Tensor<T> PredictCore(Tensor<T> input) => _useNativeMode ? Forward(input) : PredictOnnx(input);

    /// <inheritdoc />
    protected override Tensor<T> Forward(Tensor<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        var features = input;
        foreach (var layer in Layers) features = layer.Forward(features);
        return features;
    }

    /// <summary>
    /// Runs Sonata's point self-distillation on unlabelled scenes, one scene per step.
    /// </summary>
    /// <param name="scenes">Scenes <c>[points, InputChannels]</c>, xyz first (then colour in [-1, 1] and normal).</param>
    /// <param name="optimizer">Optimizer for the student; null builds AdamW at the configured rate and decay.</param>
    /// <param name="distillationLoss">Per-match loss; null uses <see cref="SonataClusterCrossEntropyLoss{T}"/>.</param>
    /// <param name="targetCentering">Teacher centering; null uses <see cref="SinkhornKnoppCentering{T}"/>.</param>
    /// <param name="featureRegularizer">Embedding regulariser weighted by <see cref="SonataOptions.KoLeoWeight"/>;
    /// null uses <see cref="KoLeoRegularizer{T}"/>.</param>
    /// <returns>The mean loss over the final epoch.</returns>
    /// <remarks>
    /// <para>
    /// The teacher is created from the student on the first call (drop path 0) and persists between
    /// calls. Each call builds the mask-size, mask-ratio, teacher-temperature and momentum schedules
    /// over its own <c>PretrainingEpochs * scenes.Count</c> steps.
    /// </para>
    /// <para>
    /// The teacher runs without gradient and is updated by EMA after every student step.
    /// </para>
    /// </remarks>
    public T Pretrain(
        IReadOnlyList<Tensor<T>> scenes,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ISonataDistillationLoss<T>? distillationLoss = null,
        ISonataTargetCentering<T>? targetCentering = null,
        ISonataFeatureRegularizer<T>? featureRegularizer = null)
    {
        if (scenes is null) throw new ArgumentNullException(nameof(scenes));
        if (scenes.Count == 0) throw new ArgumentException("At least one scene is required.", nameof(scenes));
        var student = RequireEncoder();
        EnsurePretrainingState(student);
        var objectives = new Objectives(
            distillationLoss ?? new SonataClusterCrossEntropyLoss<T>(),
            targetCentering ?? new SinkhornKnoppCentering<T>(_options.SinkhornIterations),
            featureRegularizer ?? new KoLeoRegularizer<T>());
        var opt = optimizer ?? CreateAdamWOptimizer(_options.LearningRate, _options.WeightDecay);
        var random = _viewRandom ?? throw new InvalidOperationException("Pretraining state was not created.");

        int total = _options.PretrainingEpochs * scenes.Count;
        var maskSize = new SonataCosineSchedule(_options.MaskSizeBase, _options.MaskSizeBase, total, _options.MaskSizeStart,
            (int)(total * _options.MaskSizeWarmupRatio));
        var maskRatio = new SonataCosineSchedule(_options.MaskRatioBase, _options.MaskRatioBase, total, _options.MaskRatioStart,
            (int)(total * _options.MaskRatioWarmupRatio));
        var teacherTemperature = new SonataCosineSchedule(_options.TeacherTemperatureBase, _options.TeacherTemperatureBase, total,
            _options.TeacherTemperatureStart, (int)(total * _options.TeacherTemperatureWarmupRatio));
        var momentum = new SonataCosineSchedule(_options.MomentumBase, _options.MomentumFinal, total);

        // Step losses of the current epoch; the last epoch's mean is reduced once after training.
        var epochLosses = new T[scenes.Count];
        int step = 0;
        SetTrainingMode(true);
        try
        {
            for (int epoch = 0; epoch < _options.PretrainingEpochs; epoch++)
            {
                for (int sceneIndex = 0; sceneIndex < scenes.Count; sceneIndex++)
                {
                    var scene = scenes[sceneIndex];
                    var settings = new StepSettings(maskSize.Get(step), maskRatio.Get(step), teacherTemperature.Get(step));
                    var batch = PrepareStep(scene, settings, random);
                    T loss = AiDotNet.Training.TapeTrainingStepper<T>.EagerObjectiveStep(
                        StudentParameters,
                        () => StudentObjective(batch, settings, objectives, training: true),
                        opt,
                        modelGradientClip: _options.GradientClip,
                        beforeUpdate: MarkTrainMutationStarted);
                    InvalidateWeightCachesAfterSuccessfulWeightUpdate();
                    UpdateTeacher(momentum.Get(step));
                    epochLosses[sceneIndex] = loss;
                    step++;
                }
            }
        }
        finally
        {
            SetTrainingMode(false);
        }

        if (_options.PretrainingEpochs <= 0) return NumOps.Zero;

        // Averaged in double so a Half model does not lose precision summing step losses.
        var asFloat = new float[epochLosses.Length];
        NumOps.ToFloatSpan(epochLosses, asFloat);
        double sum = 0.0;
        foreach (float value in asFloat) sum += value;
        return NumOps.FromDouble(sum / asFloat.Length);
    }

    /// <summary>
    /// The pretraining objective on one scene with the start-of-schedule settings, in evaluation mode
    /// and without updating anything; <paramref name="seed"/> fixes the views and masks.
    /// </summary>
    internal double EvaluatePretrainingLoss(Tensor<T> scene, int seed)
    {
        var student = RequireEncoder();
        EnsurePretrainingState(student);
        var settings = new StepSettings(_options.MaskSizeStart, _options.MaskRatioStart, _options.TeacherTemperatureStart);
        var objectives = new Objectives(new SonataClusterCrossEntropyLoss<T>(),
            new SinkhornKnoppCentering<T>(_options.SinkhornIterations), new KoLeoRegularizer<T>());
        var batch = PrepareStep(scene, settings, RandomHelper.CreateSeededRandom(seed));
        using var noGrad = new NoGradScope<T>();
        return NumOps.ToDouble(StudentObjective(batch, settings, objectives, training: false).ToArray()[0]);
    }

    /// <summary>Encoder and head tensors the student step updates.</summary>
    internal IReadOnlyList<Tensor<T>> StudentParameters()
    {
        var list = new List<Tensor<T>>(RequireEncoder().GetTrainableParameters());
        if (_studentHeadContext is not null) list.AddRange(_studentHeadContext.Created);
        return list;
    }

    /// <summary>The views, masks and teacher targets of one step: everything that carries no gradient.</summary>
    private StepBatch PrepareStep(Tensor<T> scene, StepSettings settings, Random random)
    {
        var teacher = _teacher ?? throw new InvalidOperationException("The teacher was not created.");
        if (scene is null) throw new ArgumentException("A scene was null.", nameof(scene));
        int width = _options.InputChannels;
        if (scene.Rank != 2 || scene.Shape[1] != width)
            throw new ArgumentException($"Each scene must be [points, {width}] with xyz first.", nameof(scene));
        var values = new float[scene.Length];
        NumOps.ToFloatSpan(scene.ToArray(), values);
        var (global, local) = SonataPretraining.GenerateViews(values, width, random, _options.GlobalViewCount,
            _options.GlobalViewScaleMin, _options.GlobalViewScaleMax, _options.LocalViewCount, _options.LocalViewScaleMin,
            _options.LocalViewScaleMax, _options.MaxViewSize);

        var mask = SonataPretraining.GenerateMask(global, settings.MaskSize, settings.MaskRatio, random);
        var maskedCoordinates = (double[])global.Coordinates.Clone();
        var maskValues = new float[global.Count];
        for (int i = 0; i < global.Count; i++)
        {
            if (!mask[i]) continue;
            maskValues[i] = 1f;
            for (int a = 0; a < 3; a++)
            {
                // torch.clip(randn * jitter, max=2 * jitter): clipped above only, as in Sonata.
                double jitter = Math.Min(2 * _options.MaskJitter, SonataPretraining.Gaussian(random) * _options.MaskJitter);
                maskedCoordinates[i * 3 + a] += jitter;
            }
        }

        Tensor<T> teacherMask, teacherUnmask;
        double[] teacherOrigin;
        int[] teacherView;
        using (new NoGradScope<T>())
        {
            var level = teacher.BuildLevel(ToTensor(global.Features, global.Count, width), global.Coordinates, global.Origin,
                global.View, global.ViewCount, null, true);
            var up = teacher.UpCast(teacher.Encode(level, true), _options.UpCastLevel);
            teacherOrigin = up.OriginCoordinates;
            teacherView = up.Batch;
            var maskHead = _teacherMaskHead ?? throw new InvalidOperationException("Teacher heads were not created.");
            var unmaskHead = _teacherUnmaskHead ?? throw new InvalidOperationException("Teacher heads were not created.");
            teacherMask = maskHead.Score(maskHead.Embed(up.Features));
            teacherUnmask = unmaskHead.Score(unmaskHead.Embed(up.Features));
        }

        return new StepBatch(global, local, maskedCoordinates, ToTensor(maskValues, global.Count, 1), teacherMask,
            teacherUnmask, teacherOrigin, teacherView);
    }

    /// <summary>The student's weighted loss for one prepared step.</summary>
    private Tensor<T> StudentObjective(StepBatch batch, StepSettings settings, Objectives objectives, bool training)
    {
        var student = RequireEncoder();
        var maskHead = _studentMaskHead ?? throw new InvalidOperationException("Student heads were not created.");
        var unmaskHead = _studentUnmaskHead ?? throw new InvalidOperationException("Student heads were not created.");
        int width = _options.InputChannels;
        var terms = new List<Tensor<T>>();
        Tensor<T>? embeddingForRegularizer = null;

        if (_options.MaskLossWeight > 0 || _options.RollMaskLossWeight > 0)
        {
            var g = batch.Global;
            var level = student.BuildLevel(ToTensor(g.Features, g.Count, width), batch.MaskedCoordinates, g.Origin, g.View,
                g.ViewCount, batch.Mask, training);
            var up = student.UpCast(student.Encode(level, training), _options.UpCastLevel);
            var embedding = maskHead.Embed(up.Features);
            embeddingForRegularizer = embedding;
            var scores = maskHead.Score(embedding);
            var identity = new int[g.ViewCount];
            for (int v = 0; v < identity.Length; v++) identity[v] = v;
            AddTerm(terms, objectives, scores, up, batch.TeacherMask, batch.TeacherOrigin, batch.TeacherView, identity,
                settings, _options.MaskLossWeight);
            if (g.ViewCount == 2)
            {
                AddTerm(terms, objectives, scores, up, batch.TeacherMask, batch.TeacherOrigin, batch.TeacherView,
                    new[] { 1, 0 }, settings, _options.RollMaskLossWeight);
            }
        }

        if (_options.UnmaskLossWeight > 0 && batch.Local.Count > 0)
        {
            var l = batch.Local;
            var level = student.BuildLevel(ToTensor(l.Features, l.Count, width), l.Coordinates, l.Origin, l.View, l.ViewCount,
                null, training);
            var up = student.UpCast(student.Encode(level, training), _options.UpCastLevel);
            var embedding = unmaskHead.Embed(up.Features);
            embeddingForRegularizer ??= embedding;
            // Every local view is matched against the principal (first) global view.
            AddTerm(terms, objectives, unmaskHead.Score(embedding), up, batch.TeacherUnmask, batch.TeacherOrigin,
                batch.TeacherView, new int[l.ViewCount], settings, _options.UnmaskLossWeight);
        }

        if (_options.KoLeoWeight > 0 && embeddingForRegularizer is not null)
        {
            terms.Add(Engine.TensorMultiplyScalar(objectives.Regularizer.Compute(embeddingForRegularizer),
                NumOps.FromDouble(_options.KoLeoWeight)));
        }

        if (terms.Count == 0)
            throw new InvalidOperationException(
                "No points of the views matched within MatchMaxRadius; use larger scenes or a larger radius.");
        var sum = terms[0];
        for (int i = 1; i < terms.Count; i++) sum = Engine.TensorAdd(sum, terms[i]);
        return sum;
    }

    private void AddTerm(List<Tensor<T>> terms, Objectives objectives, Tensor<T> studentScores, PointCloudLevel<T> studentLevel,
        Tensor<T> teacherScores, double[] teacherOrigin, int[] teacherView, int[] pairedView, StepSettings settings,
        double weight)
    {
        if (!(weight > 0)) return;
        var (query, reference) = SonataPretraining.MatchNeighbour(studentLevel.OriginCoordinates, studentLevel.Batch,
            teacherOrigin, teacherView, pairedView, _options.MatchMaxRadius);
        if (query.Length == 0) return;
        Tensor<T> targets;
        using (new NoGradScope<T>())
        {
            targets = objectives.Centering.Center(Engine.TensorIndexSelect(teacherScores, PtOps.Indices(reference), 0),
                settings.TeacherTemperature);
        }

        var sampleOfMatch = new int[query.Length];
        for (int i = 0; i < query.Length; i++) sampleOfMatch[i] = studentLevel.Batch[query[i]];
        var loss = objectives.Loss.Compute(Engine.TensorIndexSelect(studentScores, PtOps.Indices(query), 0), targets,
            _options.StudentTemperature, sampleOfMatch);
        terms.Add(Engine.TensorMultiplyScalar(loss, NumOps.FromDouble(weight)));
    }

    private void UpdateTeacher(double momentum)
    {
        var teacher = _teacher ?? throw new InvalidOperationException("The teacher was not created.");
        SonataPretraining.UpdateTeacher(teacher.GetTrainableParameters(), RequireEncoder().GetTrainableParameters(), momentum);
        if (_teacherHeadContext is not null && _studentHeadContext is not null)
            SonataPretraining.UpdateTeacher(_teacherHeadContext.Created, _studentHeadContext.Created, momentum);
    }

    private void EnsurePretrainingState(PointTransformerV3Encoder<T> student)
    {
        if (_teacher is not null) return;
        var o = _options;
        int seed = o.Seed ?? RandomHelper.CreateSecureRandom().Next();
        _viewRandom = RandomHelper.CreateSeededRandom(seed);
        // Sonata teacher_custom: drop_path=0. Same architecture, values copied from the student.
        _teacher = CreateEncoder(0.0, seed, student.Curves);
        SonataPretraining.CopyInto(_teacher.GetTrainableParameters(), student.GetTrainableParameters());

        int headIn = 0;
        var channels = o.EncoderChannels;
        for (int s = channels.Length - 1 - o.UpCastLevel; s < channels.Length; s++) headIn += channels[s];
        _studentHeadContext = new PtContext<T>(RandomHelper.CreateSeededRandom(seed + 1), truncatedNormalInit: true);
        _teacherHeadContext = new PtContext<T>(RandomHelper.CreateSeededRandom(seed + 2), truncatedNormalInit: true);
        _studentMaskHead = new SonataPrototypeHead<T>(_studentHeadContext, headIn, o.HeadHiddenChannels, o.HeadEmbedChannels, o.PrototypeCount);
        _studentUnmaskHead = new SonataPrototypeHead<T>(_studentHeadContext, headIn, o.HeadHiddenChannels, o.HeadEmbedChannels, o.PrototypeCount);
        _teacherMaskHead = new SonataPrototypeHead<T>(_teacherHeadContext, headIn, o.HeadHiddenChannels, o.HeadEmbedChannels, o.PrototypeCount);
        _teacherUnmaskHead = new SonataPrototypeHead<T>(_teacherHeadContext, headIn, o.HeadHiddenChannels, o.HeadEmbedChannels, o.PrototypeCount);
        SonataPretraining.CopyInto(_teacherHeadContext.Created, _studentHeadContext.Created);
    }

    private PointTransformerV3Encoder<T> RequireEncoder()
    {
        if (!_useNativeMode) throw new InvalidOperationException("Pretraining is not supported in ONNX mode. Use the native mode constructor.");
        return Encoder ?? throw new InvalidOperationException("Pretraining needs the built-in Sonata encoder, not custom architecture layers.");
    }

    private PointTransformerV3Encoder<T> CreateEncoder(double dropPath, int? seed, IReadOnlyList<ISpaceFillingCurve> curves)
    {
        var o = _options;
        return new PointTransformerV3Encoder<T>(o.InputChannels, true, o.GridSize, o.Strides, o.EncoderDepths,
            o.EncoderChannels, o.EncoderNumHeads, o.EncoderPatchSizes, null, null, null, null, o.MlpRatio, dropPath,
            o.EnableConditionalPositionalEncoding, o.PreNorm, o.ShuffleOrders, new SpaceFillingCurveSet(curves),
            PointTransformerV3Revision.M2Sonata, true, true, seed);
    }

    private Tensor<T> ToTensor(float[] values, int rows, int columns)
    {
        var data = new T[values.Length];
        NumOps.FromFloatSpan(values, data);
        return new Tensor<T>(data, new[] { rows, columns });
    }

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
        {
            ClearLayers();
            return;
        }

        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            _encoderLayerEnd = Math.Max(0, Architecture.Layers.Count - 1);
            return;
        }

        Layers.Add(CreateEncoder(_options.DropPathRate, _options.Seed, _options.SerializationCurves));
        _encoderLayerEnd = 1;
        // Linear probe on the fully up-cast features (Sonata linear probing: 1232 channels -> classes).
        Layers.Add(new DenseLayer<T>(_numClasses, (IActivationFunction<T>?)null));
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        AdditionalInfo = new Dictionary<string, object>
        {
            { "ModelName", "Sonata" },
            { "Revision", PointTransformerV3Revision.M2Sonata.ToString() },
            { "InputChannels", _options.InputChannels },
            { "NumClasses", _numClasses },
            { "EncoderDepths", string.Join(",", _options.EncoderDepths) },
            { "EncoderChannels", string.Join(",", _options.EncoderChannels) },
            { "UseNativeMode", _useNativeMode },
            { "NumLayers", Layers.Count }
        },
        ModelDataProvider = () => SerializeForMetadata()
    };

    private readonly record struct StepSettings(double MaskSize, double MaskRatio, double TeacherTemperature);

    private sealed record Objectives(ISonataDistillationLoss<T> Loss, ISonataTargetCentering<T> Centering,
        ISonataFeatureRegularizer<T> Regularizer);

    private sealed record StepBatch(SonataViewSet Global, SonataViewSet Local, double[] MaskedCoordinates, Tensor<T> Mask,
        Tensor<T> TeacherMask, Tensor<T> TeacherUnmask, double[] TeacherOrigin, int[] TeacherView);
}
