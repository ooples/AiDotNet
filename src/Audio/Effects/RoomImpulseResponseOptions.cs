using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Effects;

/// <summary>
/// Configuration options for the Room Impulse Response (RIR) estimation model.
/// </summary>
/// <remarks>
/// <para>
/// Neural Room Impulse Response estimation (2023-2024) uses deep learning to predict the
/// acoustic characteristics of a room from audio recordings. The model estimates the RIR
/// which encodes how sound propagates, reflects, and decays in a given space, enabling
/// applications like dereverberation, room simulation, and acoustic environment matching.
/// </para>
/// <para>
/// <b>For Beginners:</b> When you clap in a big room, you hear echoes. This model learns to
/// understand those echoes. Given a recording, it can figure out the room's acoustic "fingerprint"
/// and use it to remove room effects (dereverberation) or apply one room's sound to another recording.
/// </para>
/// </remarks>
public class RoomImpulseResponseOptions : ModelOptions
{
    #region Audio

    /// <summary>Gets or sets the audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 16000;

    #endregion

    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public string Variant { get; set; } = "base";

    /// <summary>Gets or sets the encoder dimension.</summary>
    public int EncoderDim { get; set; } = 256;

    /// <summary>Gets or sets the number of encoder layers.</summary>
    public int NumEncoderLayers { get; set; } = 6;

    /// <summary>Gets or sets the RIR length in samples.</summary>
    public int RIRLength { get; set; } = 16000;

    /// <summary>Gets or sets the number of frequency bins for spectral estimation.</summary>
    public int NumFrequencyBins { get; set; } = 257;

    /// <summary>Gets or sets the number of attention heads.</summary>
    public int NumHeads { get; set; } = 4;

    #endregion

    #region Enhancement

    /// <summary>Gets or sets the dereverberation strength (0-1).</summary>
    public double DereverberationStrength { get; set; } = 0.8;

    /// <summary>Gets or sets the RT60 estimation window in seconds.</summary>
    public double RT60WindowSeconds { get; set; } = 1.0;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the learning rate.</summary>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.1;

    #endregion
}
