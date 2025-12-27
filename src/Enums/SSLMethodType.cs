namespace AiDotNet.Enums;

/// <summary>
/// Specifies the type of self-supervised learning method to use for representation learning.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Self-supervised learning (SSL) methods learn useful representations
/// from unlabeled data by creating "pretext tasks" - artificial tasks that force the model to learn
/// meaningful features. Different methods use different strategies to achieve this.</para>
///
/// <para><b>Choosing a Method:</b>
/// <list type="bullet">
/// <item>Use <b>SimCLR</b> for simplicity and good performance (no memory bank needed)</item>
/// <item>Use <b>MoCo</b> variants for large batch sizes or limited GPU memory</item>
/// <item>Use <b>BYOL</b> or <b>SimSiam</b> to avoid negative sample mining</item>
/// <item>Use <b>BarlowTwins</b> for interpretable redundancy-reduction approach</item>
/// <item>Use <b>DINO</b> for Vision Transformers with self-distillation</item>
/// <item>Use <b>MAE</b> for generative masked autoencoding approach</item>
/// </list>
/// </para>
/// </remarks>
public enum SSLMethodType
{
    /// <summary>
    /// SimCLR: A Simple Framework for Contrastive Learning of Visual Representations (Chen et al., 2020).
    /// Uses large batch contrastive learning with strong augmentations.
    /// </summary>
    /// <remarks>
    /// <para><b>Best for:</b> Simple setup, strong performance, research baselines.</para>
    /// <para><b>Key Parameters:</b> Temperature (0.1-0.5), batch size (256-8192), projection dimension (128).</para>
    /// <para><b>Pros:</b> Simple architecture, strong performance, well-understood.</para>
    /// <para><b>Cons:</b> Requires large batch sizes for best performance.</para>
    /// </remarks>
    SimCLR = 0,

    /// <summary>
    /// MoCo: Momentum Contrast for Unsupervised Visual Representation Learning (He et al., 2020).
    /// Uses a momentum encoder and memory queue for efficient contrastive learning.
    /// </summary>
    /// <remarks>
    /// <para><b>Best for:</b> Limited GPU memory, consistent negative samples.</para>
    /// <para><b>Key Parameters:</b> Queue size (65536), momentum (0.999), temperature (0.07).</para>
    /// <para><b>Pros:</b> Memory efficient, consistent negative samples, good performance.</para>
    /// <para><b>Cons:</b> More complex than SimCLR, requires momentum encoder.</para>
    /// </remarks>
    MoCo = 1,

    /// <summary>
    /// MoCo v2: Improved Baselines with Momentum Contrastive Learning (Chen et al., 2020).
    /// Adds MLP projection head and stronger augmentations to MoCo.
    /// </summary>
    /// <remarks>
    /// <para><b>Best for:</b> Better performance than MoCo v1 with similar efficiency.</para>
    /// <para><b>Key Parameters:</b> Same as MoCo plus MLP projection head.</para>
    /// <para><b>Pros:</b> Combines MoCo efficiency with SimCLR improvements.</para>
    /// <para><b>Cons:</b> Slightly more complex than MoCo v1.</para>
    /// </remarks>
    MoCoV2 = 2,

    /// <summary>
    /// MoCo v3: An Empirical Study of Training Self-Supervised Vision Transformers (Chen et al., 2021).
    /// Adapted for Vision Transformers without memory queue.
    /// </summary>
    /// <remarks>
    /// <para><b>Best for:</b> Vision Transformers (ViT), modern architectures.</para>
    /// <para><b>Key Parameters:</b> Momentum (0.99-0.999), symmetric loss.</para>
    /// <para><b>Pros:</b> Optimized for ViT, simpler than MoCo v1/v2.</para>
    /// <para><b>Cons:</b> Best suited for transformer architectures.</para>
    /// </remarks>
    MoCoV3 = 3,

    /// <summary>
    /// BYOL: Bootstrap Your Own Latent (Grill et al., 2020).
    /// Non-contrastive method using momentum encoder without negative samples.
    /// </summary>
    /// <remarks>
    /// <para><b>Best for:</b> Avoiding negative sample mining, asymmetric networks.</para>
    /// <para><b>Key Parameters:</b> Momentum (0.99-0.999), predictor MLP.</para>
    /// <para><b>Pros:</b> No negative samples needed, robust to batch size.</para>
    /// <para><b>Cons:</b> Requires careful design to prevent collapse.</para>
    /// </remarks>
    BYOL = 4,

    /// <summary>
    /// SimSiam: Exploring Simple Siamese Representation Learning (Chen & He, 2021).
    /// Simple non-contrastive method with stop-gradient operation.
    /// </summary>
    /// <remarks>
    /// <para><b>Best for:</b> Simplicity, understanding non-contrastive learning.</para>
    /// <para><b>Key Parameters:</b> Predictor MLP, stop-gradient placement.</para>
    /// <para><b>Pros:</b> Simplest non-contrastive method, no momentum encoder.</para>
    /// <para><b>Cons:</b> Can be sensitive to hyperparameters.</para>
    /// </remarks>
    SimSiam = 5,

    /// <summary>
    /// Barlow Twins: Self-Supervised Learning via Redundancy Reduction (Zbontar et al., 2021).
    /// Reduces redundancy in embeddings by making cross-correlation close to identity.
    /// </summary>
    /// <remarks>
    /// <para><b>Best for:</b> Interpretable approach, avoiding collapse naturally.</para>
    /// <para><b>Key Parameters:</b> Lambda (redundancy reduction weight), projection dimension.</para>
    /// <para><b>Pros:</b> Interpretable loss, naturally avoids collapse, no negative samples.</para>
    /// <para><b>Cons:</b> Requires careful scaling of loss terms.</para>
    /// </remarks>
    BarlowTwins = 6,

    /// <summary>
    /// DINO: Emerging Properties in Self-Supervised Vision Transformers (Caron et al., 2021).
    /// Self-distillation with no labels using centering and sharpening.
    /// </summary>
    /// <remarks>
    /// <para><b>Best for:</b> Vision Transformers, emergent attention properties.</para>
    /// <para><b>Key Parameters:</b> Teacher temperature (0.04-0.07), centering momentum.</para>
    /// <para><b>Pros:</b> Emergent attention maps, strong ViT performance.</para>
    /// <para><b>Cons:</b> Primarily designed for Vision Transformers.</para>
    /// </remarks>
    DINO = 7,

    /// <summary>
    /// iBOT: Image BERT Pre-Training with Online Tokenizer (Zhou et al., 2022).
    /// Combines masked image modeling with self-distillation.
    /// </summary>
    /// <remarks>
    /// <para><b>Best for:</b> Combining generative and discriminative approaches.</para>
    /// <para><b>Key Parameters:</b> Mask ratio (0.4), patch tokenizer.</para>
    /// <para><b>Pros:</b> Best of both worlds (DINO + MAE-like objectives).</para>
    /// <para><b>Cons:</b> More complex than pure DINO or MAE.</para>
    /// </remarks>
    iBOT = 8,

    /// <summary>
    /// MAE: Masked Autoencoders Are Scalable Vision Learners (He et al., 2022).
    /// Generative approach that reconstructs masked image patches.
    /// </summary>
    /// <remarks>
    /// <para><b>Best for:</b> Efficient pretraining, generative understanding.</para>
    /// <para><b>Key Parameters:</b> Mask ratio (0.75), decoder depth.</para>
    /// <para><b>Pros:</b> Efficient (only encode visible patches), scalable.</para>
    /// <para><b>Cons:</b> May require fine-tuning for best downstream performance.</para>
    /// </remarks>
    MAE = 9
}
