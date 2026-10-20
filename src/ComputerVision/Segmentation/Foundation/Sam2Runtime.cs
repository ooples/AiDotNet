using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.ComputerVision.Segmentation.Foundation;

/// <summary>
/// The settings every SAM 2 model exposes: capacity overrides (null follows the released size) and the training recipe.
/// </summary>
internal interface ISam2Settings
{
    int? ImageSize { get; }

    int MemoryBankSize { get; }

    int? HieraEmbeddingDimension { get; }

    int[]? HieraStageDepths { get; }

    int? HieraInitialHeadCount { get; }

    int[]? HieraWindowSizes { get; }

    int[]? HieraGlobalAttentionBlockIndexes { get; }

    int? ModelDimension { get; }

    int? MemoryDimension { get; }

    int? DecoderHeadCount { get; }

    int? MaskDecoderMlpDimension { get; }

    int? MemoryAttentionLayerCount { get; }

    int? MemoryAttentionFeedForwardDimension { get; }

    int? MemoryFuserDimension { get; }

    double LearningRate { get; }

    double ImageEncoderLearningRate { get; }

    double? LayerDecay { get; }

    double WeightDecay { get; }

    double MaxGradientNorm { get; }

    int TrainingSteps { get; }

    int WarmupSteps { get; }

    double? DropPath { get; }

    double MaskFocalWeight { get; }

    double MaskDiceWeight { get; }

    double IouLossWeight { get; }

    double OcclusionLossWeight { get; }

    double FocalGamma { get; }

    double FocalAlpha { get; }

    int CorrectionPoints { get; }

    int? Seed { get; }
}

/// <summary>Validation of the settings every SAM 2 model shares.</summary>
internal static class Sam2SettingsValidation
{
    /// <summary>Throws if a shared setting is out of range.</summary>
    /// <param name="o">The settings.</param>
    /// <param name="imageMultiple">The encoder's size granularity (Hiera: 32; a plain ViT: its 16-pixel patch).</param>
    public static void Validate(ISam2Settings o, int imageMultiple = 32)
    {
        if (o.ImageSize is { } size && (size <= 0 || size % imageMultiple != 0))
            throw new ArgumentOutOfRangeException(nameof(ISam2Settings.ImageSize), $"The image size must be a positive multiple of {imageMultiple}.");
        if (o.MemoryBankSize < 1) throw new ArgumentOutOfRangeException(nameof(o.MemoryBankSize));
        if (o.HieraStageDepths is { } depths && (depths.Length != 4 || depths.Any(d => d <= 0)))
            throw new ArgumentException("HieraStageDepths must hold four positive values.", nameof(ISam2Settings.HieraStageDepths));
        if (o.HieraWindowSizes is { } windows && (windows.Length != 4 || windows.Any(w => w <= 0)))
            throw new ArgumentException("HieraWindowSizes must hold four positive values.", nameof(ISam2Settings.HieraWindowSizes));
        if (o.HieraGlobalAttentionBlockIndexes is { } globals && (globals.Any(g => g < 0) || globals.Distinct().Count() != globals.Length))
            throw new ArgumentException("The global-attention blocks must be distinct and non-negative.", nameof(ISam2Settings.HieraGlobalAttentionBlockIndexes));
        foreach (var (value, name) in new[]
        {
            (o.HieraEmbeddingDimension, nameof(ISam2Settings.HieraEmbeddingDimension)), (o.HieraInitialHeadCount, nameof(ISam2Settings.HieraInitialHeadCount)),
            (o.ModelDimension, nameof(ISam2Settings.ModelDimension)), (o.MemoryDimension, nameof(ISam2Settings.MemoryDimension)), (o.DecoderHeadCount, nameof(ISam2Settings.DecoderHeadCount)),
            (o.MaskDecoderMlpDimension, nameof(ISam2Settings.MaskDecoderMlpDimension)), (o.MemoryAttentionLayerCount, nameof(ISam2Settings.MemoryAttentionLayerCount)),
            (o.MemoryAttentionFeedForwardDimension, nameof(ISam2Settings.MemoryAttentionFeedForwardDimension)), (o.MemoryFuserDimension, nameof(ISam2Settings.MemoryFuserDimension)),
        })
            if (value is <= 0) throw new ArgumentOutOfRangeException(name);
        if (o.LearningRate <= 0 || o.ImageEncoderLearningRate < 0 || o.WeightDecay < 0 || o.MaxGradientNorm < 0)
            throw new ArgumentOutOfRangeException(nameof(ISam2Settings.LearningRate), "Learning rates, weight decay and the clip must be non-negative.");
        if (o.LayerDecay is <= 0 or > 1) throw new ArgumentOutOfRangeException(nameof(ISam2Settings.LayerDecay));
        if (o.DropPath is < 0 or >= 1) throw new ArgumentOutOfRangeException(nameof(ISam2Settings.DropPath));
        if (o.TrainingSteps <= 0 || o.WarmupSteps < 0) throw new ArgumentOutOfRangeException(nameof(ISam2Settings.TrainingSteps));
        if (o.MaskFocalWeight < 0 || o.MaskDiceWeight < 0 || o.IouLossWeight < 0 || o.OcclusionLossWeight < 0 || o.CorrectionPoints < 0)
            throw new ArgumentOutOfRangeException(nameof(ISam2Settings.MaskFocalWeight), "Loss weights and the click count cannot be negative.");
    }
}

/// <summary>
/// One SAM 2 model's network, preprocessing, prompting, tracking and training, shared by the classes that expose SAM 2
/// (<c>Video.Segmentation.SAM2</c>, <see cref="SAM21{T}"/> and <c>Medical.MedSAM2</c>), which differ only in their
/// configuration, recipe and public API.
/// </summary>
internal sealed class Sam2Runtime<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly ISam2Settings _settings;
    private readonly Sam2TrainingSettings _training;
    private readonly int _sizeIndex;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _userOptimizer;
    private readonly Sam2BankSettings? _bank;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;

    /// <param name="engine">The engine.</param>
    /// <param name="settings">The model's settings.</param>
    /// <param name="sizeIndex">The released size (0 Tiny, 1 Small, 2 Base+, 3 Large).</param>
    /// <param name="sam21">Whether the network is SAM 2.1's.</param>
    /// <param name="side">The encoder's square size when the settings leave it open.</param>
    /// <param name="training">The prompt simulation (the loss weights and click count come from the settings).</param>
    /// <param name="optimizer">A caller's optimizer, or null for the recipe's.</param>
    /// <param name="bank">A self-sorting memory bank for image sets (Medical SAM 2), or null.</param>
    public Sam2Runtime(IEngine engine, ISam2Settings settings, int sizeIndex, bool sam21, int side, Sam2TrainingSettings training,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer, Sam2BankSettings? bank = null)
        : this(engine, settings, Configuration(settings, sizeIndex, sam21, side), sizeIndex, training, optimizer, bank)
    {
    }

    /// <summary>A runtime over a given network configuration (EfficientTAM's ViT encoder, for one).</summary>
    public Sam2Runtime(IEngine engine, ISam2Settings settings, Sam2Configuration configuration, int sizeIndex, Sam2TrainingSettings training,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer, Sam2BankSettings? bank = null)
    {
        _bank = bank;
        _engine = engine;
        _settings = settings;
        _sizeIndex = sizeIndex;
        _userOptimizer = optimizer;
        _training = training with
        {
            FocalWeight = settings.MaskFocalWeight,
            DiceWeight = settings.MaskDiceWeight,
            IouWeight = settings.IouLossWeight,
            OcclusionWeight = settings.OcclusionLossWeight,
            FocalGamma = settings.FocalGamma,
            FocalAlpha = settings.FocalAlpha,
            CorrectionPoints = settings.CorrectionPoints,
        };
        Layers = new List<LayerBase<T>>();
        Network = new Sam2Network<T>(engine, Layers, configuration);
        int seed = settings.Seed ?? 42;
        Network.InitializeLikeReference(Tensors.Helpers.RandomHelper.CreateSeededRandom(seed));
        TrainingRandom = Tensors.Helpers.RandomHelper.CreateSeededRandom(seed + 1);
        Network.TrainingRandom = TrainingRandom;
    }

    public Sam2Network<T> Network { get; }

    public List<LayerBase<T>> Layers { get; }

    /// <summary>Draws the training prompts, corrective clicks, drop-path and dropout masks.</summary>
    public Random TrainingRandom { get; }

    public int ImageSize => Network.Configuration.ImageSize;

    /// <summary>The released size's configuration with the settings' overrides.</summary>
    public static Sam2Configuration Configuration(ISam2Settings o, int sizeIndex, bool sam21, int side)
    {
        var preset = Sam2Configuration.Hiera(sizeIndex);
        side = o.ImageSize ?? side;
        if (side <= 0 || side % 32 != 0)
            throw new ArgumentException($"SAM 2's encoder size must be a positive multiple of 32 pixels; got {side}. Set ImageSize.");
        int stem = o.HieraEmbeddingDimension ?? preset.EmbedDims[0];
        int heads = o.HieraInitialHeadCount ?? preset.HeadsPerStage[0];
        var depths = o.HieraStageDepths ?? preset.BlocksPerStage;
        var globals = o.HieraGlobalAttentionBlockIndexes
            ?? (depths.SequenceEqual(preset.BlocksPerStage) ? preset.GlobalAttentionBlocks : new[] { depths[0] + depths[1] + depths[2] - 1 });
        var memory = (sam21 ? Sam2MemoryConfiguration.Sam21 : new Sam2MemoryConfiguration()) with
        {
            MaskMemories = o.MemoryBankSize,
            MemoryDim = o.MemoryDimension ?? 64,
            AttentionLayers = o.MemoryAttentionLayerCount ?? 4,
            AttentionFeedForward = o.MemoryAttentionFeedForwardDimension ?? 2048,
            FuserHidden = o.MemoryFuserDimension ?? 1024,
        };
        return preset with
        {
            ImageSize = side,
            EmbedDims = new[] { stem, 2 * stem, 4 * stem, 8 * stem },
            HeadsPerStage = new[] { heads, 2 * heads, 4 * heads, 8 * heads },
            BlocksPerStage = depths,
            WindowSizes = o.HieraWindowSizes ?? preset.WindowSizes,
            GlobalAttentionBlocks = globals,
            Dim = o.ModelDimension ?? 256,
            DecoderHeads = o.DecoderHeadCount ?? 8,
            DecoderMlp = o.MaskDecoderMlpDimension ?? 2048,
            DropPath = o.DropPath ?? preset.DropPath,
            Memory = memory,
        };
    }

    // ------------------------------------------------------------------ frames and prompts

    /// <summary>A frame <c>[C, H, W]</c> or <c>[1, C, H, W]</c> as <c>[1, C, H, W]</c>.</summary>
    public static Tensor<T> Frame(Tensor<T> frame, string name)
    {
        if (frame.Rank == 4 && frame.Shape[0] == 1) return frame;
        if (frame.Rank != 3) throw new ArgumentException("A frame is [C, H, W] (or [1, C, H, W]).", name);
        return frame.Reshape(new[] { 1, frame.Shape[0], frame.Shape[1], frame.Shape[2] });
    }

    /// <summary>A frame <c>[1, 3, H, W]</c> in 0..1 resized to the encoder's square and normalized: bilinear and antialiased
    /// as the image predictor's torchvision transform resizes, or bicubic as the video predictor's PIL loader does.</summary>
    public Tensor<T> Preprocess(Tensor<T> frame, bool video = false)
    {
        int size = ImageSize;
        if (frame.Shape[1] != 3) throw new ArgumentException("SAM 2 segments RGB frames (3 channels).", nameof(frame));
        var x = frame.Shape[2] == size && frame.Shape[3] == size ? frame
            : video ? SpatialResampling.ResizeBicubicAntialias(_engine, frame, size, size) : SpatialResampling.ResizeAntialias(_engine, frame, size, size);
        return Network.Normalize(x);
    }

    /// <summary>Points <c>[k, 2]</c> (x, y in a frame of the given size) in the encoder's frame, as the official predictors
    /// normalize them.</summary>
    public Tensor<T> ScalePoints(IReadOnlyList<(double X, double Y)> points, int height, int width)
    {
        int size = ImageSize;
        var scaled = new Tensor<T>(new[] { 1, points.Count, 2 });
        for (int i = 0; i < points.Count; i++)
        {
            scaled[0, i, 0] = NumOps.FromDouble(points[i].X * size / width);
            scaled[0, i, 1] = NumOps.FromDouble(points[i].Y * size / height);
        }
        return scaled;
    }

    /// <summary>A click prompt: points (x, y in the frame) with labels 1 (object) or 0 (background).</summary>
    public Sam2Prompt<T> PointPrompt(IReadOnlyList<(double X, double Y)> points, IReadOnlyList<int> labels, int height, int width)
    {
        if (points.Count == 0) throw new ArgumentException("A click prompt needs at least one point.", nameof(points));
        if (labels.Count != points.Count) throw new ArgumentException("Each point needs a label.", nameof(labels));
        if (labels.Any(l => l is not (0 or 1))) throw new ArgumentException("Point labels are 1 (object) or 0 (background).", nameof(labels));
        var l = new Tensor<T>(new[] { 1, labels.Count });
        for (int i = 0; i < labels.Count; i++) l[0, i] = NumOps.FromDouble(labels[i]);
        return new Sam2Prompt<T>(ScalePoints(points, height, width), l, null);
    }

    public Sam2Prompt<T> PointPrompt(float[,] points, int[] labels, int height, int width)
    {
        if (points.GetLength(1) != 2) throw new ArgumentException("Points are [k, 2] (x, y).", nameof(points));
        return PointPrompt(Enumerable.Range(0, points.GetLength(0)).Select(i => ((double)points[i, 0], (double)points[i, 1])).ToList(), labels, height, width);
    }

    /// <summary>A box <c>[x0, y0, x1, y1]</c> as its two corner points (labels 2 and 3), as the official predictors give it.</summary>
    public Sam2Prompt<T> BoxPrompt(IReadOnlyList<double> box, int height, int width)
    {
        if (box.Count != 4) throw new ArgumentException("A box is [x0, y0, x1, y1].", nameof(box));
        var l = new Tensor<T>(new[] { 1, 2 }, new Vector<T>(new[] { NumOps.FromDouble(2), NumOps.FromDouble(3) }));
        return new Sam2Prompt<T>(ScalePoints(new[] { (box[0], box[1]), (box[2], box[3]) }, height, width), l, null);
    }

    /// <summary>A mask prompt <c>[H, W]</c> resized to the encoder's square (bilinear, antialiased) and thresholded at 0.5,
    /// as the official <c>add_new_mask</c> does.</summary>
    public Sam2Prompt<T> MaskPrompt(Tensor<T> mask, int height, int width)
    {
        if (mask.Length != height * width) throw new ArgumentException($"The mask prompt is one {height}x{width} map.", nameof(mask));
        int size = ImageSize;
        var m = _engine.Reshape(mask, new[] { 1, 1, height, width });
        if (height != size || width != size) m = SpatialResampling.ResizeAntialias(_engine, m, size, size);
        var binary = new Tensor<T>(new[] { 1, 1, size, size });
        var values = m.ToArray();
        for (int i = 0; i < values.Length; i++) binary[i] = NumOps.ToDouble(values[i]) >= 0.5 ? NumOps.One : NumOps.Zero;
        return new Sam2Prompt<T>(null, null, binary);
    }

    /// <summary>The model's empty prompt (a padding point).</summary>
    public static Sam2Prompt<T> NoPrompt => new(null, null, null);

    /// <summary>The background holes the video predictor fills in its outputs (<c>fill_hole_area</c>, 8 in the official
    /// predictor builds).</summary>
    public const int FillHoleArea = 8;

    /// <summary>
    /// A tracked frame's mask probabilities <c>[H, W]</c> as the official video predictor outputs them: holes in the
    /// low-resolution logits (8-connected background regions of at most <see cref="FillHoleArea"/> pixels) filled with 0.1,
    /// then the logits resized to the frame (bilinear) once.
    /// </summary>
    public Tensor<T> VideoProbabilities(Sam2FrameOutput<T> output, int height, int width)
        => Probabilities(FillHoles(output.LowRes, FillHoleArea), height, width);

    /// <summary>The official <c>fill_holes_in_mask_scores</c>: 8-connected regions of logits ≤ 0 with at most
    /// <paramref name="maxArea"/> pixels set to 0.1.</summary>
    public static Tensor<T> FillHoles(Tensor<T> logits, int maxArea)
    {
        int h = logits.Shape[^2], w = logits.Shape[^1];
        var values = logits.ToArray();
        var label = new int[h * w];
        var filled = (T[])values.Clone();
        bool changed = false;
        int next = 0;
        var stack = new Stack<int>();
        var members = new List<int>();
        for (int start = 0; start < h * w; start++)
        {
            if (label[start] != 0 || NumOps.ToDouble(values[start]) > 0) continue;
            label[start] = ++next;
            stack.Push(start);
            members.Clear();
            while (stack.Count > 0)
            {
                int i = stack.Pop();
                members.Add(i);
                int y = i / w, x = i % w;
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++)
                    {
                        int ny = y + dy, nx = x + dx;
                        if (ny < 0 || ny >= h || nx < 0 || nx >= w) continue;
                        int j = ny * w + nx;
                        if (label[j] != 0 || NumOps.ToDouble(values[j]) > 0) continue;
                        label[j] = next;
                        stack.Push(j);
                    }
            }
            if (members.Count > maxArea) continue;
            foreach (int i in members) filled[i] = NumOps.FromDouble(0.1);
            changed = true;
        }
        return changed ? new Tensor<T>(logits.Shape.ToArray(), new Vector<T>(filled)) : logits;
    }

    /// <summary>Mask logits <c>[1, 1, h, w]</c> resized to a frame (bilinear) as probabilities <c>[H, W]</c>.</summary>
    public Tensor<T> Probabilities(Tensor<T> logits, int height, int width)
    {
        var resized = logits.Shape[2] == height && logits.Shape[3] == width ? logits : SpatialResampling.Resize(_engine, logits, height, width);
        return _engine.Reshape(_engine.Sigmoid(resized), new[] { height, width });
    }

    // ------------------------------------------------------------------ tracking and images

    /// <summary>Tracks one object through frames <c>[1, 3, H, W]</c> (values in 0..1) from its prompts per frame, as the
    /// official video predictor does (with <paramref name="freshReverse"/>, the backward pass starts again from the prompts
    /// alone, as MedSAM2's inference does); every frame's output.</summary>
    public Sam2FrameOutput<T>[] Track(IReadOnlyList<Tensor<T>> frames, IReadOnlyDictionary<int, Sam2Prompt<T>> prompts, bool freshReverse = false)
    {
        if (frames.Count == 0) throw new ArgumentException("No frames.", nameof(frames));
        using var _ = new NoGradScope<T>();
        var tracker = new Sam2VideoTracker<T>(_engine, Network);
        return tracker.Propagate(frames.Count, t => Network.EncodeFeatures(Preprocess(frames[t], video: true)), prompts, freshReverse);
    }

    /// <summary>Segments one object through a set of images <c>[1, 3, H, W]</c> as a pseudo-video conditioned on a fresh
    /// self-sorting memory bank (Medical SAM 2), its draws seeded by <paramref name="seed"/>; every image's output.</summary>
    public Sam2FrameOutput<T>[] TrackImageSet(IReadOnlyList<Tensor<T>> images, IReadOnlyDictionary<int, Sam2Prompt<T>> prompts, int seed)
    {
        if (_bank is null) throw new InvalidOperationException("This model has no memory bank.");
        if (images.Count == 0) throw new ArgumentException("No images.", nameof(images));
        using var _ = new NoGradScope<T>();
        var tracker = new Sam2VideoTracker<T>(_engine, Network) { Bank = new Sam2SortedMemoryBank<T>(_bank), BankRandom = new Random(seed) };
        return tracker.RunImageSet(images.Count, t => Network.EncodeFeatures(Preprocess(images[t], video: true)), prompts);
    }

    /// <summary>A tracker for streaming one object through frames that arrive one at a time.</summary>
    public Sam2VideoTracker<T> StreamingTracker() => new(_engine, Network) { Frames = int.MaxValue };

    /// <summary>A video frame's features <c>[1, 3, H, W]</c> (values in 0..1) for a tracker.</summary>
    public (Tensor<T> Features, Tensor<T>? HighRes0, Tensor<T>? HighRes1) EncodeVideoFrame(Tensor<T> frame)
        => Network.EncodeFeatures(Preprocess(frame, video: true));

    /// <summary>The image predictor on one frame <c>[1, 3, H, W]</c>: the features with the no-memory embedding.</summary>
    public (Tensor<T> Embedding, Tensor<T>? HighRes0, Tensor<T>? HighRes1) EncodeImage(Tensor<T> frame)
    {
        using var _ = new NoGradScope<T>();
        return Network.EncodeImage(Preprocess(frame));
    }

    /// <summary>The image predictor's masks for N prompts: low-resolution logits <c>[N, M, S/4, S/4]</c>, predicted IoUs
    /// <c>[N, M]</c> and object-score logits <c>[N, 1]</c> (three masks with <paramref name="multimask"/>, else one chosen by
    /// stability).</summary>
    public (Tensor<T> Logits, Tensor<T> Iou, Tensor<T> ObjectScore) DecodeImage((Tensor<T> Embedding, Tensor<T>? HighRes0, Tensor<T>? HighRes1) features,
        Tensor<T>? points, Tensor<T>? labels, Tensor<T>? lowResLogits, bool multimask)
    {
        using var _ = new NoGradScope<T>();
        var (sparse, dense) = Network.EncodePrompts(points, labels, null, lowResLogits);
        return Network.DecodeMasks(features.Embedding, features.HighRes0, features.HighRes1, Network.DensePositional(), sparse, dense, multimask);
    }

    // ------------------------------------------------------------------ training

    /// <summary>The frames (resized and normalized) and masks (resized to the encoder's square, above 0.5 the object) of a
    /// training example: a frame <c>[C, H, W]</c> with its mask <c>[H, W]</c>, or a video <c>[F, C, H, W]</c> with
    /// <c>[F, H, W]</c>.</summary>
    public (List<Tensor<T>> Images, List<double[]> Masks) Example(Tensor<T> input, Tensor<T> target)
    {
        bool single = input.Rank == 3;
        if (!single && input.Rank != 4) throw new ArgumentException("SAM 2 trains on a frame [C, H, W] or a video [F, C, H, W].", nameof(input));
        int frames = single ? 1 : input.Shape[0], height = input.Shape[^2], width = input.Shape[^1], size = ImageSize;
        if (target.Length != frames * height * width)
            throw new ArgumentException($"The target is one {height}x{width} mask per frame ({frames}).", nameof(target));
        var images = new List<Tensor<T>>();
        var masks = new List<double[]>();
        var flat = _engine.Reshape(target, new[] { frames, height * width });
        for (int t = 0; t < frames; t++)
        {
            var frame = single ? Frame(input, nameof(input)) : _engine.TensorSlice(input, new[] { t, 0, 0, 0 }, new[] { 1, input.Shape[1], height, width });
            images.Add(Preprocess(frame));
            var mask = _engine.Reshape(_engine.TensorSlice(flat, new[] { t, 0 }, new[] { 1, height * width }), new[] { 1, 1, height, width });
            if (height != size || width != size) mask = SpatialResampling.ResizeAntialias(_engine, mask, size, size);
            masks.Add(mask.ToArray().Select(v => NumOps.ToDouble(v) > 0.5 ? 1.0 : 0.0).ToArray());
        }
        return (images, masks);
    }

    public Sam2VideoTraining<T> Trainer(bool bank = false)
    {
        if (bank && _bank is null) throw new InvalidOperationException("This model has no memory bank.");
        return new(_engine, Network, bank ? _training with { Bank = _bank } : _training);
    }

    /// <summary>The trained parameters (read every step, as loading weights may replace them) and the optimizer: the
    /// recipe's AdamW unless one was given.</summary>
    private (List<Tensor<T>> Parameters, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> Optimizer) TrainingState()
    {
        var recipe = new Sam2Recipe
        {
            LearningRate = _settings.LearningRate,
            ImageEncoderLearningRate = _settings.ImageEncoderLearningRate,
            LayerDecay = _settings.LayerDecay ?? Sam2Recipe.PaperLayerDecay(_sizeIndex),
            WeightDecay = _settings.WeightDecay,
            MaxGradientNorm = _settings.MaxGradientNorm,
            Steps = _settings.TrainingSteps,
            WarmupSteps = _settings.WarmupSteps,
        };
        _optimizer ??= _userOptimizer ?? Sam2Optimizer.Create(Network, Layers, recipe).Optimizer;
        var parameters = _userOptimizer is null && _optimizer is AdamWOptimizer<T, Tensor<T>, Tensor<T>> paper
            ? Sam2Optimizer.Refresh(paper, Network, Layers, recipe)
            : Training.TapeTrainingStep<T>.CollectParameters(Layers).ToList();
        return (parameters, _optimizer);
    }

    /// <summary>One training step (the caller sets training mode) on examples (frames and their masks, see
    /// <see cref="Example"/>), minimizing their mean loss; returns it.</summary>
    public T Train(IReadOnlyList<(Tensor<T> Input, Tensor<T> Target)> examples, Action<IReadOnlyDictionary<Tensor<T>, Tensor<T>>> onGradients,
        bool bank = false)
    {
        if (examples.Count == 0) throw new ArgumentException("No training examples.", nameof(examples));
        var trainer = Trainer(bank);
        var prepared = examples.Select(e => Example(e.Input, e.Target)).ToList();
        var plans = prepared.Select(e => trainer.Draw(e.Masks, TrainingRandom)).ToList();
        var (parameters, optimizer) = TrainingState();
        Network.TrainingRandom = TrainingRandom;
        Tensor<T> Objective()
        {
            Tensor<T>? total = null;
            for (int i = 0; i < prepared.Count; i++)
            {
                var loss = trainer.Loss(prepared[i].Images, prepared[i].Masks, plans[i], TrainingRandom);
                total = total is null ? loss : _engine.TensorAdd(total, loss);
            }
            return _engine.TensorMultiplyScalar(total!, NumOps.FromDouble(1.0 / prepared.Count));
        }
        return Training.TapeTrainingStepper<T>.EagerObjectiveStep(parameters, Objective, optimizer, onGradients: g => onGradients(g));
    }

    /// <summary>The training objective estimated with common random numbers: its mean over four prompt plans, clicks, drop
    /// path and dropout masks drawn from one fixed seed, so two evaluations compare the same objective.</summary>
    public T Evaluate(IReadOnlyList<(Tensor<T> Input, Tensor<T> Target)> examples, bool bank = false)
    {
        const int draws = 4;
        var prepared = examples.Select(e => Example(e.Input, e.Target)).ToList();
        var trainer = Trainer(bank);
        var random = new Random(_settings.Seed ?? 42);
        bool wasTraining = Network.Training;
        var previous = Network.TrainingRandom;
        Network.Training = true;
        Network.TrainingRandom = random;
        try
        {
            using var _ = new NoGradScope<T>();
            double total = 0;
            foreach (var (images, masks) in prepared)
                for (int i = 0; i < draws; i++)
                    total += NumOps.ToDouble(trainer.Loss(images, masks, trainer.Draw(masks, random), random)[0]);
            return NumOps.FromDouble(total / (draws * prepared.Count));
        }
        finally
        {
            Network.Training = wasTraining;
            Network.TrainingRandom = previous;
        }
    }
}
