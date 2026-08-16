using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.PointCloud;

/// <summary>
/// Configuration options for Concerto, defaulting to the values published in
/// "Concerto: Joint 2D-3D Self-Supervised Learning Emerges Spatial Representations"
/// (arXiv:2510.23607).
/// </summary>
/// <remarks>
/// <para>Concerto is a SELF-SUPERVISED PRETRAINING method, not a supervised segmentation model.
/// It combines 3D intra-modal self-distillation — a Point Transformer V3 student matched against
/// a momentum-updated teacher — with 2D-3D cross-modal joint embedding against a frozen DINOv2
/// image encoder. Segmentation is a downstream probe on the learned representation, not the
/// training objective.</para>
/// <para><b>For Beginners:</b> The model learns about 3D scenes without any human labels, by
/// (a) checking that two different views of the same point cloud produce consistent features,
/// and (b) checking that the 3D features agree with what a strong 2D image model sees at the
/// same physical locations. Only afterwards is a small labelled head trained on top.</para>
/// </remarks>
public class ConcertoOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with the paper's published defaults.</summary>
    public ConcertoOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public ConcertoOptions(ConcertoOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;

        TeacherMomentum = other.TeacherMomentum;
        IntraModalLossWeight = other.IntraModalLossWeight;
        CrossModalLossWeight = other.CrossModalLossWeight;
        IntraModalUpcastLevel = other.IntraModalUpcastLevel;
        CrossModalUpcastLevel = other.CrossModalUpcastLevel;
        ImagesPerPointCloud = other.ImagesPerPointCloud;
        ImageEncoderResolution = other.ImageEncoderResolution;
        VisibilityDepthToleranceMeters = other.VisibilityDepthToleranceMeters;
        LearningRate = other.LearningRate;
        PretrainingEpochs = other.PretrainingEpochs;
    }

    /// <summary>
    /// Momentum for the exponential-moving-average update of the teacher encoder.
    /// </summary>
    /// <value>Defaults to 0.996.</value>
    /// <remarks>
    /// The student is optimized to match a momentum-updated teacher; the teacher's weights are
    /// never back-propagated into, only tracked as an EMA of the student's.
    /// </remarks>
    public double TeacherMomentum { get; set; } = 0.996;

    /// <summary>
    /// Weight on the 3D intra-modal self-distillation term (online clustering cross-entropy).
    /// </summary>
    /// <value>Defaults to 2.0.</value>
    /// <remarks>
    /// The paper reports best results at a cross:intra weight ratio of 2:2, so both terms carry
    /// equal weight by default. The ratio is what matters, not the absolute magnitudes.
    /// </remarks>
    public double IntraModalLossWeight { get; set; } = 2.0;

    /// <summary>
    /// Weight on the 2D-3D cross-modal joint-embedding term (cosine similarity).
    /// </summary>
    /// <value>Defaults to 2.0 — the paper's 2:2 ratio against the intra-modal term.</value>
    public double CrossModalLossWeight { get; set; } = 2.0;

    /// <summary>
    /// Decoder upcast level at which the intra-modal clustering loss is applied.
    /// </summary>
    /// <value>Defaults to 2, the paper's value.</value>
    public int IntraModalUpcastLevel { get; set; } = 2;

    /// <summary>
    /// Decoder upcast level at which the cross-modal cosine loss is applied.
    /// </summary>
    /// <value>Defaults to 3, the paper's value.</value>
    /// <remarks>
    /// The two objectives deliberately attach at DIFFERENT depths: clustering at level 2, image
    /// alignment at level 3.
    /// </remarks>
    public int CrossModalUpcastLevel { get; set; } = 3;

    /// <summary>
    /// Number of images paired with each point cloud during pretraining.
    /// </summary>
    /// <value>Defaults to 4, the paper's value.</value>
    public int ImagesPerPointCloud { get; set; } = 4;

    /// <summary>
    /// Square input resolution of the frozen 2D image encoder.
    /// </summary>
    /// <value>Defaults to 518, matching the paper's DINOv2-L configuration.</value>
    public int ImageEncoderResolution { get; set; } = 518;

    /// <summary>
    /// Depth agreement, in metres, required for a projected point to count as visible in an image.
    /// </summary>
    /// <value>Defaults to 0.01, the paper's threshold.</value>
    /// <remarks>
    /// <para>Correspondence is established by projecting 3D points into the image and then
    /// verifying visibility against the depth buffer: a point counts only when
    /// <c>|d_c - d_proj| &lt; tolerance</c>. Without that check, occluded points behind a surface
    /// would be paired with whatever is drawn in front of them.</para>
    /// <para>The paper notes that admitting FEWER visible points performed better than maximizing
    /// matches, so loosening this is not a free accuracy win.</para>
    /// </remarks>
    public double VisibilityDepthToleranceMeters { get; set; } = 0.01;

    /// <summary>Peak learning rate for the AdamW optimizer with cosine annealing.</summary>
    /// <value>Defaults to 0.004, the paper's upper value (scaled down by encoder depth).</value>
    public double LearningRate { get; set; } = 0.004;

    /// <summary>Number of self-supervised pretraining epochs.</summary>
    /// <value>Defaults to 100, the paper's value.</value>
    public int PretrainingEpochs { get; set; } = 100;
}
