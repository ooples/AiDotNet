namespace AiDotNet.Enums;

/// <summary>
/// Which parameter group a SeACo-Paraformer training step updates.
/// </summary>
/// <remarks>
/// <para>
/// <b>The three members are the paper's recipe plus the default that is deliberately not it.</b> Shi et
/// al. ("SeACo-Paraformer", arXiv:2308.03266, §3) train the ASR backbone first, then FREEZE it and train
/// only the bias parameters, "separate from the ASR training" -- which is <see cref="Backbone"/> followed
/// by <see cref="Bias"/>. <see cref="Joint"/> exists because one <c>Train</c> call updating every
/// parameter is what a caller expects from a single entry point, and it is the default for that reason
/// rather than a claim to reproduce the paper.
/// </para>
/// </remarks>
public enum SeACoTrainingStage
{
    /// <summary>
    /// Every parameter is updated in one step. The default: it is what a single <c>Train</c> entry point
    /// is expected to do, and it is NOT the paper's staged recipe.
    /// </summary>
    Joint,

    /// <summary>
    /// Only the ASR backbone is updated; the bias branch is frozen. The first of the paper's two stages.
    /// </summary>
    Backbone,

    /// <summary>
    /// Only the bias parameters are updated; the backbone is frozen. The second of the paper's two
    /// stages, and the one its hot-word-position-aware criterion is defined for.
    /// </summary>
    Bias
}
