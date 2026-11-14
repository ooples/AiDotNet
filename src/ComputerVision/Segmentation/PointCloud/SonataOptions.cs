using AiDotNet.Attributes;

namespace AiDotNet.ComputerVision.Segmentation.PointCloud;

/// <summary>
/// Configuration options for Sonata, self-supervised point self-distillation of a Point Transformer V3
/// encoder (Wu et al., CVPR 2025).
/// </summary>
/// <remarks>
/// <para>
/// The defaults are the official <c>pretrain-sonata-v1m1-0-base</c> configuration: a PT-v3m2
/// encoder-only backbone (depths (3, 3, 3, 12, 3), widths (48, 96, 192, 384, 512), heads
/// (3, 6, 12, 24, 32), patch 1024, drop path 0.3, mask token) fed coordinate, colour and normal
/// (9 channels); <c>OnlineCluster</c> heads of 4096 hidden, 256 embedding and 4096 prototypes on the
/// features up-cast two levels (1088 channels); 2 global views (scale 0.4-1.0) and 4 local views
/// (0.1-0.4) of at most 65536 points; masking of the global views with cube size 0.1 -&gt; 0.4 and
/// ratio 0.3 -&gt; 0.7; teacher temperature 0.04 -&gt; 0.07; student temperature 0.1; loss weights
/// 2/8 (mask), 2/8 (roll mask), 4/8 (unmask); EMA momentum 0.994 -&gt; 1; match radius 0.32 m.
/// </para>
/// <para>
/// The official configuration has no KoLeo term, so <see cref="KoLeoWeight"/> is zero by default.
/// </para>
/// <para><b>For Beginners:</b> The pretraining settings only matter when you call
/// <see cref="Sonata{T}.Pretrain"/>; the backbone settings decide the network.</para>
/// </remarks>
[PaperDefault(nameof(NumClasses), 20, Source = "Wu et al. 2025, Sec. 5: ScanNet v2 linear probing, 20 classes")]
[PaperDefault(nameof(InputChannels), 9, Source = "Pointcept pretrain-sonata-v1m1-0-base: in_channels=9 (coord, color, normal)")]
[PaperDefault(nameof(GridSize), 0.02, Source = "Pointcept pretrain-sonata-v1m1-0-base: GridSample grid_size=0.02")]
[PaperDefault(nameof(Strides), new[] { 2, 2, 2, 2 }, Source = "Pointcept pretrain-sonata-v1m1-0-base: stride=(2, 2, 2, 2)")]
[PaperDefault(nameof(EncoderDepths), new[] { 3, 3, 3, 12, 3 }, Source = "Pointcept pretrain-sonata-v1m1-0-base: enc_depths=(3, 3, 3, 12, 3)")]
[PaperDefault(nameof(EncoderChannels), new[] { 48, 96, 192, 384, 512 }, Source = "Pointcept pretrain-sonata-v1m1-0-base: enc_channels=(48, 96, 192, 384, 512)")]
[PaperDefault(nameof(EncoderNumHeads), new[] { 3, 6, 12, 24, 32 }, Source = "Pointcept pretrain-sonata-v1m1-0-base: enc_num_head=(3, 6, 12, 24, 32)")]
[PaperDefault(nameof(EncoderPatchSizes), new[] { 1024, 1024, 1024, 1024, 1024 }, Source = "Pointcept pretrain-sonata-v1m1-0-base: enc_patch_size=1024")]
[PaperDefault(nameof(MlpRatio), 4.0, Source = "Pointcept pretrain-sonata-v1m1-0-base: mlp_ratio=4")]
[PaperDefault(nameof(DropPathRate), 0.3, Source = "Pointcept pretrain-sonata-v1m1-0-base: drop_path=0.3 (teacher 0)")]
[PaperDefault(nameof(SerializationOrders), new[] { "z", "z-trans", "hilbert", "hilbert-trans" }, Source = "Pointcept pretrain-sonata-v1m1-0-base: order=(z, z-trans, hilbert, hilbert-trans)")]
[PaperDefault(nameof(UpCastLevel), 2, Source = "Pointcept pretrain-sonata-v1m1-0-base: up_cast_level=2")]
[PaperDefault(nameof(HeadHiddenChannels), 4096, Source = "Pointcept pretrain-sonata-v1m1-0-base: head_hidden_channels=4096")]
[PaperDefault(nameof(HeadEmbedChannels), 256, Source = "Pointcept pretrain-sonata-v1m1-0-base: head_embed_channels=256")]
[PaperDefault(nameof(PrototypeCount), 4096, Source = "Pointcept pretrain-sonata-v1m1-0-base: head_num_prototypes=4096")]
[PaperDefault(nameof(GlobalViewCount), 2, Source = "Wu et al. 2025, Sec. 3.2; config num_global_view=2")]
[PaperDefault(nameof(LocalViewCount), 4, Source = "Wu et al. 2025, Sec. 3.2; config num_local_view=4")]
[PaperDefault(nameof(GlobalViewScaleMin), 0.4, Source = "Pointcept pretrain-sonata-v1m1-0-base: global_view_scale=(0.4, 1.0)")]
[PaperDefault(nameof(GlobalViewScaleMax), 1.0, Source = "Pointcept pretrain-sonata-v1m1-0-base: global_view_scale=(0.4, 1.0)")]
[PaperDefault(nameof(LocalViewScaleMin), 0.1, Source = "Pointcept pretrain-sonata-v1m1-0-base: local_view_scale=(0.1, 0.4)")]
[PaperDefault(nameof(LocalViewScaleMax), 0.4, Source = "Pointcept pretrain-sonata-v1m1-0-base: local_view_scale=(0.1, 0.4)")]
[PaperDefault(nameof(MaxViewSize), 65536, Source = "Pointcept pretrain-sonata-v1m1-0-base: max_size=65536")]
[PaperDefault(nameof(MaskSizeStart), 0.1, Source = "Pointcept pretrain-sonata-v1m1-0-base: mask_size_start=0.1")]
[PaperDefault(nameof(MaskSizeBase), 0.4, Source = "Pointcept pretrain-sonata-v1m1-0-base: mask_size_base=0.4")]
[PaperDefault(nameof(MaskSizeWarmupRatio), 0.05, Source = "Pointcept pretrain-sonata-v1m1-0-base: mask_size_warmup_ratio=0.05")]
[PaperDefault(nameof(MaskRatioStart), 0.3, Source = "Pointcept pretrain-sonata-v1m1-0-base: mask_ratio_start=0.3")]
[PaperDefault(nameof(MaskRatioBase), 0.7, Source = "Pointcept pretrain-sonata-v1m1-0-base: mask_ratio_base=0.7")]
[PaperDefault(nameof(MaskRatioWarmupRatio), 0.05, Source = "Pointcept pretrain-sonata-v1m1-0-base: mask_ratio_warmup_ratio=0.05")]
[PaperDefault(nameof(MaskJitter), 0.01, Source = "Pointcept pretrain-sonata-v1m1-0-base: mask_jitter=0.01")]
[PaperDefault(nameof(TeacherTemperatureStart), 0.04, Source = "Pointcept pretrain-sonata-v1m1-0-base: teacher_temp_start=0.04")]
[PaperDefault(nameof(TeacherTemperatureBase), 0.07, Source = "Pointcept pretrain-sonata-v1m1-0-base: teacher_temp_base=0.07")]
[PaperDefault(nameof(TeacherTemperatureWarmupRatio), 0.05, Source = "Pointcept pretrain-sonata-v1m1-0-base: teacher_temp_warmup_ratio=0.05")]
[PaperDefault(nameof(StudentTemperature), 0.1, Source = "Pointcept pretrain-sonata-v1m1-0-base: student_temp=0.1")]
[PaperDefault(nameof(MaskLossWeight), 0.25, Source = "Pointcept pretrain-sonata-v1m1-0-base: mask_loss_weight=2/8")]
[PaperDefault(nameof(RollMaskLossWeight), 0.25, Source = "Pointcept pretrain-sonata-v1m1-0-base: roll_mask_loss_weight=2/8")]
[PaperDefault(nameof(UnmaskLossWeight), 0.5, Source = "Pointcept pretrain-sonata-v1m1-0-base: unmask_loss_weight=4/8")]
[PaperDefault(nameof(MomentumBase), 0.994, Source = "Pointcept pretrain-sonata-v1m1-0-base: momentum_base=0.994")]
[PaperDefault(nameof(MomentumFinal), 1.0, Source = "Pointcept pretrain-sonata-v1m1-0-base: momentum_final=1")]
[PaperDefault(nameof(MatchMaxRadius), 0.32, Source = "Pointcept pretrain-sonata-v1m1-0-base: match_max_r=0.32")]
[PaperDefault(nameof(SinkhornIterations), 3, Source = "Pointcept sonata_v1m1_base.py: sinkhorn_knopp(num_iter=3)")]
[PaperDefault(nameof(KoLeoWeight), 0.0, Source = "Pointcept pretrain-sonata-v1m1-0-base: no KoLeo term")]
[PaperDefault(nameof(LearningRate), 0.004, Source = "Pointcept pretrain-sonata-v1m1-0-base: base_lr=0.004 (AdamW)")]
[PaperDefault(nameof(WeightDecay), 0.04, Source = "Pointcept pretrain-sonata-v1m1-0-base: base_wd=0.04")]
[PaperDefault(nameof(GradientClip), 3.0, Source = "Pointcept pretrain-sonata-v1m1-0-base: clip_grad=3.0")]
public class SonataOptions : PointTransformerV3EncoderOptions
{
    /// <summary>Initializes a new instance with the Sonata v1m1 base values.</summary>
    public SonataOptions()
    {
        NumClasses = 20;
        InputChannels = 9;
        EncoderDepths = new[] { 3, 3, 3, 12, 3 };
        EncoderChannels = new[] { 48, 96, 192, 384, 512 };
        EncoderNumHeads = new[] { 3, 6, 12, 24, 32 };
    }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SonataOptions(SonataOptions other)
        : base(other)
    {
        UpCastLevel = other.UpCastLevel;
        HeadHiddenChannels = other.HeadHiddenChannels;
        HeadEmbedChannels = other.HeadEmbedChannels;
        PrototypeCount = other.PrototypeCount;
        GlobalViewCount = other.GlobalViewCount;
        LocalViewCount = other.LocalViewCount;
        GlobalViewScaleMin = other.GlobalViewScaleMin;
        GlobalViewScaleMax = other.GlobalViewScaleMax;
        LocalViewScaleMin = other.LocalViewScaleMin;
        LocalViewScaleMax = other.LocalViewScaleMax;
        MaxViewSize = other.MaxViewSize;
        MaskSizeStart = other.MaskSizeStart;
        MaskSizeBase = other.MaskSizeBase;
        MaskSizeWarmupRatio = other.MaskSizeWarmupRatio;
        MaskRatioStart = other.MaskRatioStart;
        MaskRatioBase = other.MaskRatioBase;
        MaskRatioWarmupRatio = other.MaskRatioWarmupRatio;
        MaskJitter = other.MaskJitter;
        TeacherTemperatureStart = other.TeacherTemperatureStart;
        TeacherTemperatureBase = other.TeacherTemperatureBase;
        TeacherTemperatureWarmupRatio = other.TeacherTemperatureWarmupRatio;
        StudentTemperature = other.StudentTemperature;
        MaskLossWeight = other.MaskLossWeight;
        RollMaskLossWeight = other.RollMaskLossWeight;
        UnmaskLossWeight = other.UnmaskLossWeight;
        MomentumBase = other.MomentumBase;
        MomentumFinal = other.MomentumFinal;
        MatchMaxRadius = other.MatchMaxRadius;
        SinkhornIterations = other.SinkhornIterations;
        KoLeoWeight = other.KoLeoWeight;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
        GradientClip = other.GradientClip;
        PretrainingEpochs = other.PretrainingEpochs;
    }

    /// <summary>
    /// How many poolings the distillation heads' features are up-cast through (Sonata: 2, giving
    /// 512 + 384 + 192 = 1088 channels).
    /// </summary>
    [OptionBehaviourExempt("Pretraining only: Predict always up-casts every level for the linear probe.")]
    public int UpCastLevel { get; set; } = 2;

    /// <summary>Hidden width of each <c>OnlineCluster</c> head MLP. Default: 4096.</summary>
    [OptionBehaviourExempt("Pretraining only: the heads are not part of the predicting network.")]
    public int HeadHiddenChannels { get; set; } = 4096;

    /// <summary>Embedding width of each head before the prototypes. Default: 256.</summary>
    [OptionBehaviourExempt("Pretraining only: the heads are not part of the predicting network.")]
    public int HeadEmbedChannels { get; set; } = 256;

    /// <summary>Number of learnable prototypes per head. Default: 4096.</summary>
    [OptionBehaviourExempt("Pretraining only: the heads are not part of the predicting network.")]
    public int PrototypeCount { get; set; } = 4096;

    /// <summary>Global crops per scene; masked copies of them form the masked views. Default: 2.</summary>
    [OptionBehaviourExempt("Pretraining only: view sampling does not affect Predict or Train.")]
    public int GlobalViewCount { get; set; } = 2;

    /// <summary>Local crops per scene. Default: 4.</summary>
    [OptionBehaviourExempt("Pretraining only: view sampling does not affect Predict or Train.")]
    public int LocalViewCount { get; set; } = 4;

    /// <summary>Smallest global crop, as a fraction of the scene (capped at <see cref="MaxViewSize"/>). Default: 0.4.</summary>
    [OptionBehaviourExempt("Pretraining only: view sampling does not affect Predict or Train.")]
    public double GlobalViewScaleMin { get; set; } = 0.4;

    /// <summary>Largest global crop fraction. Default: 1.0.</summary>
    [OptionBehaviourExempt("Pretraining only: view sampling does not affect Predict or Train.")]
    public double GlobalViewScaleMax { get; set; } = 1.0;

    /// <summary>Smallest local crop fraction. Default: 0.1.</summary>
    [OptionBehaviourExempt("Pretraining only: view sampling does not affect Predict or Train.")]
    public double LocalViewScaleMin { get; set; } = 0.1;

    /// <summary>Largest local crop fraction. Default: 0.4.</summary>
    [OptionBehaviourExempt("Pretraining only: view sampling does not affect Predict or Train.")]
    public double LocalViewScaleMax { get; set; } = 0.4;

    /// <summary>Most points a crop may hold. Default: 65536.</summary>
    [OptionBehaviourExempt("Pretraining only: view sampling does not affect Predict or Train.")]
    public int MaxViewSize { get; set; } = 65536;

    /// <summary>Mask cube edge in metres at the start of warm-up. Default: 0.1.</summary>
    [OptionBehaviourExempt("Pretraining only: masking does not affect Predict or Train.")]
    public double MaskSizeStart { get; set; } = 0.1;

    /// <summary>Mask cube edge in metres after warm-up. Default: 0.4.</summary>
    [OptionBehaviourExempt("Pretraining only: masking does not affect Predict or Train.")]
    public double MaskSizeBase { get; set; } = 0.4;

    /// <summary>Fraction of the pretraining steps spent warming the mask size up. Default: 0.05.</summary>
    [OptionBehaviourExempt("Pretraining only: masking does not affect Predict or Train.")]
    public double MaskSizeWarmupRatio { get; set; } = 0.05;

    /// <summary>Fraction of cubes masked at the start of warm-up. Default: 0.3.</summary>
    [OptionBehaviourExempt("Pretraining only: masking does not affect Predict or Train.")]
    public double MaskRatioStart { get; set; } = 0.3;

    /// <summary>Fraction of cubes masked after warm-up. Default: 0.7.</summary>
    [OptionBehaviourExempt("Pretraining only: masking does not affect Predict or Train.")]
    public double MaskRatioBase { get; set; } = 0.7;

    /// <summary>Fraction of the pretraining steps spent warming the mask ratio up. Default: 0.05.</summary>
    [OptionBehaviourExempt("Pretraining only: masking does not affect Predict or Train.")]
    public double MaskRatioWarmupRatio { get; set; } = 0.05;

    /// <summary>Standard deviation in metres of the jitter added to masked coordinates (clipped above at twice it). Default: 0.01.</summary>
    [OptionBehaviourExempt("Pretraining only: masking does not affect Predict or Train.")]
    public double MaskJitter { get; set; } = 0.01;

    /// <summary>Teacher temperature at the start of warm-up. Default: 0.04.</summary>
    [OptionBehaviourExempt("Pretraining only: distillation temperatures do not affect Predict or Train.")]
    public double TeacherTemperatureStart { get; set; } = 0.04;

    /// <summary>Teacher temperature after warm-up. Default: 0.07.</summary>
    [OptionBehaviourExempt("Pretraining only: distillation temperatures do not affect Predict or Train.")]
    public double TeacherTemperatureBase { get; set; } = 0.07;

    /// <summary>Fraction of the pretraining steps spent warming the teacher temperature up. Default: 0.05.</summary>
    [OptionBehaviourExempt("Pretraining only: distillation temperatures do not affect Predict or Train.")]
    public double TeacherTemperatureWarmupRatio { get; set; } = 0.05;

    /// <summary>Student softmax temperature. Default: 0.1.</summary>
    [OptionBehaviourExempt("Pretraining only: distillation temperatures do not affect Predict or Train.")]
    public double StudentTemperature { get; set; } = 0.1;

    /// <summary>Weight of the masked-view loss against the same global view. Default: 2/8.</summary>
    [OptionBehaviourExempt("Pretraining only: loss weights do not affect Predict or Train.")]
    public double MaskLossWeight { get; set; } = 0.25;

    /// <summary>Weight of the masked-view loss against the other global view. Default: 2/8.</summary>
    [OptionBehaviourExempt("Pretraining only: loss weights do not affect Predict or Train.")]
    public double RollMaskLossWeight { get; set; } = 0.25;

    /// <summary>Weight of the local-view loss against the principal global view. Default: 4/8.</summary>
    [OptionBehaviourExempt("Pretraining only: loss weights do not affect Predict or Train.")]
    public double UnmaskLossWeight { get; set; } = 0.5;

    /// <summary>EMA momentum of the teacher at the first step (cosine schedule). Default: 0.994.</summary>
    [OptionBehaviourExempt("Pretraining only: the teacher does not affect Predict or Train.")]
    public double MomentumBase { get; set; } = 0.994;

    /// <summary>EMA momentum of the teacher at the last step. Default: 1.</summary>
    [OptionBehaviourExempt("Pretraining only: the teacher does not affect Predict or Train.")]
    public double MomentumFinal { get; set; } = 1.0;

    /// <summary>Largest distance in metres at which points of two views are matched. Default: 0.32.</summary>
    [OptionBehaviourExempt("Pretraining only: matching does not affect Predict or Train.")]
    public double MatchMaxRadius { get; set; } = 0.32;

    /// <summary>Sinkhorn-Knopp iterations of the default target centering. Default: 3.</summary>
    [OptionBehaviourExempt("Pretraining only: target centering does not affect Predict or Train.")]
    public int SinkhornIterations { get; set; } = 3;

    /// <summary>Weight of the feature regulariser (KoLeo by default). Default: 0, as in Sonata.</summary>
    [OptionBehaviourExempt("Pretraining only: the regulariser does not affect Predict or Train.")]
    public double KoLeoWeight { get; set; }

    /// <summary>Learning rate of the default pretraining optimizer (AdamW). Default: 0.004.</summary>
    [OptionBehaviourExempt("Pretraining only: the default pretraining optimizer is not used by Train.")]
    public double LearningRate { get; set; } = 0.004;

    /// <summary>Weight decay of the default pretraining optimizer. Default: 0.04.</summary>
    [OptionBehaviourExempt("Pretraining only: the default pretraining optimizer is not used by Train.")]
    public double WeightDecay { get; set; } = 0.04;

    /// <summary>Gradient-norm clip during pretraining. Default: 3.</summary>
    [OptionBehaviourExempt("Pretraining only: the clip applies to Pretrain steps.")]
    public double GradientClip { get; set; } = 3.0;

    /// <summary>Passes over the scenes per <see cref="Sonata{T}.Pretrain"/> call. Default: 1.</summary>
    [OptionBehaviourExempt("Pretraining only: the epoch count does not affect Predict or Train.")]
    public int PretrainingEpochs { get; set; } = 1;

    /// <summary>
    /// Throws when a value on this instance cannot produce a working model or pretraining run.
    /// </summary>
    /// <exception cref="ArgumentException">Thrown when any setting is invalid.</exception>
    public void Validate()
    {
        ValidateBackbone(encoderOnly: true);
        if (InputChannels < 3) throw new ArgumentException("Sonata feeds xyz as features, so InputChannels must be at least 3.");
        if (UpCastLevel < 0 || UpCastLevel >= EncoderDepths.Length)
            throw new ArgumentException("UpCastLevel must be between 0 and the number of poolings.");
        if (HeadHiddenChannels <= 0 || HeadEmbedChannels <= 0 || PrototypeCount <= 0)
            throw new ArgumentException("Head widths and the prototype count must be positive.");
        if (GlobalViewCount < 1 || GlobalViewCount > 2) throw new ArgumentException("Sonata's roll pairing supports one or two global views.");
        if (GlobalViewCount < 2 && RollMaskLossWeight > 0) throw new ArgumentException("The roll-mask loss needs two global views.");
        if (LocalViewCount < 0) throw new ArgumentException("LocalViewCount cannot be negative.");
        if (LocalViewCount == 0 && UnmaskLossWeight > 0) throw new ArgumentException("The unmask loss needs local views.");
        if (!(GlobalViewScaleMin > 0) || GlobalViewScaleMax < GlobalViewScaleMin || GlobalViewScaleMax > 1
            || !(LocalViewScaleMin > 0) || LocalViewScaleMax < LocalViewScaleMin || LocalViewScaleMax > 1)
            throw new ArgumentException("View scales must satisfy 0 < min <= max <= 1.");
        if (MaxViewSize < 1) throw new ArgumentException("MaxViewSize must be positive.");
        if (!(MaskSizeStart > 0) || !(MaskSizeBase > 0)) throw new ArgumentException("Mask sizes must be positive.");
        if (MaskRatioStart < 0 || MaskRatioStart > 1 || MaskRatioBase < 0 || MaskRatioBase > 1)
            throw new ArgumentException("Mask ratios must be in [0, 1].");
        if (MaskSizeWarmupRatio < 0 || MaskRatioWarmupRatio < 0 || TeacherTemperatureWarmupRatio < 0
            || MaskSizeWarmupRatio > 1 || MaskRatioWarmupRatio > 1 || TeacherTemperatureWarmupRatio > 1)
            throw new ArgumentException("Warm-up ratios must be in [0, 1].");
        if (MaskJitter < 0) throw new ArgumentException("MaskJitter cannot be negative.");
        if (!(TeacherTemperatureStart > 0) || !(TeacherTemperatureBase > 0) || !(StudentTemperature > 0))
            throw new ArgumentException("Temperatures must be positive.");
        if (MaskLossWeight < 0 || RollMaskLossWeight < 0 || UnmaskLossWeight < 0 || KoLeoWeight < 0)
            throw new ArgumentException("Loss weights cannot be negative.");
        if (!(MaskLossWeight + RollMaskLossWeight + UnmaskLossWeight > 0)) throw new ArgumentException("At least one loss must be enabled.");
        if (MomentumBase < 0 || MomentumBase > 1 || MomentumFinal < 0 || MomentumFinal > 1)
            throw new ArgumentException("EMA momenta must be in [0, 1].");
        if (!(MatchMaxRadius > 0)) throw new ArgumentException("MatchMaxRadius must be positive.");
        if (SinkhornIterations < 1) throw new ArgumentException("SinkhornIterations must be at least 1.");
        if (!(LearningRate > 0) || WeightDecay < 0 || GradientClip < 0) throw new ArgumentException("Optimizer settings are invalid.");
        if (PretrainingEpochs < 1) throw new ArgumentException("PretrainingEpochs must be at least 1.");
    }
}
