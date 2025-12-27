namespace AiDotNet.Enums;

/// <summary>
/// Categorizes self-supervised learning methods by their learning paradigm.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> SSL methods can be grouped by how they learn representations:</para>
/// <list type="bullet">
/// <item><b>Contrastive:</b> Learn by pulling similar samples together and pushing different samples apart</item>
/// <item><b>NonContrastive:</b> Learn by predicting one view from another without explicit negatives</item>
/// <item><b>Generative:</b> Learn by reconstructing masked or corrupted inputs</item>
/// <item><b>SelfDistillation:</b> Learn by matching predictions between teacher and student networks</item>
/// </list>
/// </remarks>
public enum SSLMethodCategory
{
    /// <summary>
    /// Contrastive learning methods (SimCLR, MoCo).
    /// Learn by contrasting positive pairs against negative samples.
    /// </summary>
    /// <remarks>
    /// <para>Methods in this category maximize agreement between different augmented views
    /// of the same image while minimizing agreement with views from different images.</para>
    /// <para><b>Examples:</b> SimCLR, MoCo, MoCo v2, MoCo v3</para>
    /// </remarks>
    Contrastive = 0,

    /// <summary>
    /// Non-contrastive learning methods (BYOL, SimSiam, Barlow Twins).
    /// Learn without explicit negative samples.
    /// </summary>
    /// <remarks>
    /// <para>Methods in this category avoid the need for negative samples through
    /// asymmetric architectures, stop-gradients, or redundancy reduction.</para>
    /// <para><b>Examples:</b> BYOL, SimSiam, Barlow Twins</para>
    /// </remarks>
    NonContrastive = 1,

    /// <summary>
    /// Generative self-supervised methods (MAE).
    /// Learn by reconstructing masked or corrupted inputs.
    /// </summary>
    /// <remarks>
    /// <para>Methods in this category learn by predicting missing parts of the input,
    /// similar to language model pretraining (BERT, GPT).</para>
    /// <para><b>Examples:</b> MAE (Masked Autoencoder)</para>
    /// </remarks>
    Generative = 2,

    /// <summary>
    /// Self-distillation methods (DINO, iBOT).
    /// Learn by knowledge transfer from a teacher network to a student network.
    /// </summary>
    /// <remarks>
    /// <para>Methods in this category use a momentum-updated teacher network to provide
    /// soft targets for a student network, enabling self-supervised knowledge distillation.</para>
    /// <para><b>Examples:</b> DINO, iBOT</para>
    /// </remarks>
    SelfDistillation = 3
}
