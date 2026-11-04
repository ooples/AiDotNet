using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for <see cref="AiDotNet.Video.FrameInterpolation.ABME{T}"/>, asymmetric bilateral motion
/// estimation for video frame interpolation (Park et al., ICCV 2021).
/// </summary>
/// <remarks>
/// <para>
/// The widths and displacements default to the authors' release (JunHeum/ABME, model/SBMNet.py, model/ABMNet.py,
/// model/SynthesisNet.py); the training defaults to the paper (Sec. 4.1).
/// </para>
/// <para>
/// <b>For Beginners:</b> ABME first estimates one motion for both frames, mirrored around the middle; a second network
/// then lets the motion toward each frame differ, and a third network blends the two warped frames into the middle one.
/// </para>
/// </remarks>
public class ABMEOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public ABMEOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public ABMEOptions(ABMEOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        SymmetricEncoderWidths = (int[])other.SymmetricEncoderWidths.Clone();
        SymmetricDecoderWidths = (int[])other.SymmetricDecoderWidths.Clone();
        SymmetricDisplacements = (int[])other.SymmetricDisplacements.Clone();
        AsymmetricEncoderWidths = (int[])other.AsymmetricEncoderWidths.Clone();
        AsymmetricDecoderWidths = (int[])other.AsymmetricDecoderWidths.Clone();
        AsymmetricDisplacement = other.AsymmetricDisplacement;
        SynthesisWidths = (int[])other.SynthesisWidths.Clone();
        Stage = other.Stage;
        LearningRate = other.LearningRate;
        SymmetricMilestones = (int[])other.SymmetricMilestones.Clone();
        AsymmetricMilestones = (int[])other.AsymmetricMilestones.Clone();
        SynthesisMilestones = (int[])other.SynthesisMilestones.Clone();
        DecayRate = other.DecayRate;
        CharbonnierEpsilon = other.CharbonnierEpsilon;
        CensusWeight = other.CensusWeight;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
    }

    #region Architecture

    /// <summary>Gets or sets the symmetric estimator's six encoder widths (16, 32, 64, 96, 128, 196).</summary>
    public int[] SymmetricEncoderWidths { get; set; } = { 16, 32, 64, 96, 128, 196 };

    /// <summary>Gets or sets the five widths of each of the symmetric estimator's dense decoders (128, 128, 96, 64, 32).</summary>
    public int[] SymmetricDecoderWidths { get; set; } = { 128, 128, 96, 64, 32 };

    /// <summary>
    /// Gets or sets the bilateral correlation's maximum displacement at the symmetric estimator's levels 6 to 2
    /// (6, 4, 4, 2, 2).
    /// </summary>
    public int[] SymmetricDisplacements { get; set; } = { 6, 4, 4, 2, 2 };

    /// <summary>Gets or sets the asymmetric refiner's two encoder widths (16, 32).</summary>
    public int[] AsymmetricEncoderWidths { get; set; } = { 16, 32 };

    /// <summary>Gets or sets the five widths of each of the asymmetric refiner's dense decoders (64, 64, 48, 32, 16).</summary>
    public int[] AsymmetricDecoderWidths { get; set; } = { 64, 64, 48, 32, 16 };

    /// <summary>Gets or sets the asymmetric refiner's correlation displacement (2).</summary>
    public int AsymmetricDisplacement { get; set; } = 2;

    /// <summary>Gets or sets the synthesis network's feature-pyramid and GridNet row widths (32, 64, 96).</summary>
    public int[] SynthesisWidths { get; set; } = { 32, 64, 96 };

    #endregion

    #region Training

    /// <summary>Gets or sets the stage training runs (Synthesis, the last, which trains all three networks).</summary>
    public AbmeTrainingStage Stage { get; set; } = AbmeTrainingStage.Synthesis;

    /// <summary>Gets or sets Adam's initial learning rate, the same at every stage (1e-4).</summary>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>Gets or sets the symmetric stage's halving steps: 1e-4 to 0.1M, then every 0.04M of 0.2M.</summary>
    public int[] SymmetricMilestones { get; set; } = { 100000, 140000, 180000 };

    /// <summary>Gets or sets the asymmetric stage's halving steps: 1e-4 to 0.12M, then every 0.06M of 0.3M.</summary>
    public int[] AsymmetricMilestones { get; set; } = { 120000, 180000, 240000 };

    /// <summary>Gets or sets the synthesis stage's halving steps: 1e-4 to 0.35M, then every 0.15M of 0.8M.</summary>
    public int[] SynthesisMilestones { get; set; } = { 350000, 500000, 650000 };

    /// <summary>Gets or sets the factor the rate is multiplied by at each milestone (0.5).</summary>
    public double DecayRate { get; set; } = 0.5;

    /// <summary>Gets or sets the Charbonnier penalty's ε (1e-6; the exponent is ½).</summary>
    public double CharbonnierEpsilon { get; set; } = 1e-6;

    /// <summary>Gets or sets the census loss's weight against the Charbonnier term (1, the paper's unweighted sum).</summary>
    public double CensusWeight { get; set; } = 1.0;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion
}
