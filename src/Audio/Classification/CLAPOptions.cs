using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Classification;

/// <summary>
/// Configuration options for the CLAP (Contrastive Language-Audio Pre-training) model.
/// </summary>
/// <remarks>
/// <para>
/// CLAP (Wu et al., ICASSP 2023) learns joint audio-text representations through contrastive
/// learning, similar to CLIP for images. It can classify audio using natural language descriptions
/// without task-specific training (zero-shot classification), achieving 26.7% zero-shot accuracy
/// on ESC-50 and 46.8% mAP on AudioSet with fine-tuning.
/// </para>
/// <para>
/// <b>For Beginners:</b> CLAP is special because it understands both audio and text. Instead
/// of having fixed labels like "dog bark" or "siren", you can describe any sound in plain
/// English and CLAP will find it in audio. For example, you can search for "the sound of
/// rain hitting a tin roof" without ever training on that specific label.
///
/// How it works:
/// <list type="number">
/// <item>CLAP has two encoders: one for audio and one for text</item>
/// <item>Both encoders map their inputs to the same shared space</item>
/// <item>Matching audio-text pairs are placed close together in this space</item>
/// <item>To classify audio, compare it with text descriptions and find the closest match</item>
/// </list>
/// </para>
/// <para>
/// <b>References:</b>
/// <list type="bullet">
/// <item>Paper: "Large-Scale Contrastive Language-Audio Pre-Training with Feature Fusion and Keyword-to-Caption Augmentation" (Wu et al., ICASSP 2023)</item>
/// <item>Repository: https://github.com/LAION-AI/CLAP</item>
/// </list>
/// </para>
/// </remarks>
public class CLAPOptions : ModelOptions
{
    #region Network

    /// <summary>
    /// Gets or sets the CLAP network: preprocessing, the audio and text towers and the joint space.
    /// </summary>
    /// <remarks>
    /// <para>The classifier runs the same network as <see cref="AiDotNet.Audio.Fingerprinting.CLAPModel{T}"/>,
    /// configured by the same options, with a linear head on the audio projection for fine-tuning.</para>
    /// </remarks>
    public CLAPModelOptions Network { get; set; } = new();

    #endregion

    #region Classification

    /// <summary>
    /// Gets or sets the confidence threshold.
    /// </summary>
    public double Threshold { get; set; } = 0.3;

    /// <summary>
    /// Gets or sets the window size in seconds.
    /// </summary>
    /// <remarks>
    /// <para>CLAP is trained on 10-second clips (Wu et al. 2023, §3.1).</para>
    /// </remarks>
    public double WindowSize { get; set; } = 10.0;

    /// <summary>
    /// Gets or sets the window overlap ratio.
    /// </summary>
    public double WindowOverlap { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the labels of the fine-tuning head (AudioSet's 527 when null).
    /// </summary>
    public string[]? CustomLabels { get; set; }

    #endregion

    #region Model Loading

    /// <summary>
    /// Gets or sets the path to a pre-trained ONNX audio encoder.
    /// </summary>
    public string? ModelPath { get; set; }

    /// <summary>
    /// Gets or sets the path to the ONNX text encoder, which zero-shot prompts need in ONNX mode.
    /// </summary>
    public string? TextEncoderModelPath { get; set; }

    /// <summary>
    /// Gets or sets ONNX runtime options.
    /// </summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>
    /// Gets or sets the initial learning rate (Wu et al. 2023, §4: 1e-4).
    /// </summary>
    public double LearningRate { get; set; } = 1e-4;

    #endregion
}
