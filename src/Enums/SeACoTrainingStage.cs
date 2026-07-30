namespace AiDotNet.Enums;

/// <summary>
/// Which parameter group SeACo-Paraformer trains, mirroring the paper's two-stage recipe.
/// </summary>
/// <remarks>
/// <para>
/// SeACo-Paraformer (Shi et al., arXiv 2308.03266, §3) trains in two separate stages:
/// "With a well-trained Paraformer model <b>freezing</b>, we enable contextualization of hotwords for
/// an ASR system by introducing bias out layer, bias decoder and bias encoder, and training them with
/// randomly sampled hotwords and their corresponding targets. Notably, the training of bias-related
/// parameters is <b>separate from the ASR training</b>."
/// </para>
/// <para>
/// <b>For Beginners:</b> the model has two halves. The first half does ordinary speech recognition;
/// the second half nudges it toward a caller-supplied "hot word" list (names, jargon). The paper
/// trains the recognizer first, then locks it and teaches only the nudging half — that way improving
/// hot-word accuracy cannot damage general recognition.
/// </para>
/// </remarks>
public enum SeACoTrainingStage
{
    /// <summary>
    /// Stage 1: train the Paraformer ASR backbone with Paraformer's own objective,
    /// gamma * L_CE + L_MAE (Gao et al., arXiv 2206.08317, Eq 6). Bias parameters are left alone.
    /// </summary>
    /// <remarks>
    /// L_MAE supervises the CIF predictor's predicted token count; Paraformer §2.2/2.4 note it "guides
    /// the predictor to convergence", so omitting it leaves that head unsupervised.
    /// </remarks>
    Backbone,

    /// <summary>
    /// Stage 2: freeze the ASR backbone and train ONLY the bias encoder, bias decoder and bias output
    /// layer, under the hotword-position-aware criterion where labels at non-hotword positions are
    /// replaced by the mask token (SeACo §3, L_bias).
    /// </summary>
    Bias,

    /// <summary>
    /// Train every parameter together under the combined objective. NOT a stage either paper
    /// describes; offered because it is the configuration most callers expect from a single
    /// <c>Train</c> call, and because it keeps every parameter receiving gradient.
    /// </summary>
    /// <remarks>
    /// Prefer <see cref="Backbone"/> followed by <see cref="Bias"/> to reproduce the published
    /// procedure. Joint training lets hot-word supervision alter the recognizer itself, which is
    /// exactly what the paper's freeze is designed to prevent.
    /// </remarks>
    Joint,
}
