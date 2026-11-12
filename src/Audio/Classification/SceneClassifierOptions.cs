using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Classification;

/// <summary>
/// Options for acoustic scene classification with the DCASE 2017 baseline system (Mesaros et al. 2017, section 2.2):
/// 40 log mel-band energies over 40 ms frames with 50% overlap, a 5-frame context, and a two-layer MLP of 50 units
/// with 20% dropout.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the SceneClassifier model. Default values follow the DCASE 2017
/// baseline system.</para>
/// </remarks>
public class SceneClassifierOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public SceneClassifierOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SceneClassifierOptions(SceneClassifierOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        SampleRate = other.SampleRate;
        FftSize = other.FftSize;
        HopLength = other.HopLength;
        NumMels = other.NumMels;
        NumMfccs = other.NumMfccs;
        TopK = other.TopK;
        CustomScenes = other.CustomScenes;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        ContextFrames = other.ContextFrames;
        HiddenUnits = other.HiddenUnits;
        NumHiddenLayers = other.NumHiddenLayers;
        DropoutRate = other.DropoutRate;
    }

    /// <summary>Audio sample rate. Default: 44100, the TUT Acoustic Scenes recordings' rate (mel bands cover 0 to
    /// 22050 Hz).</summary>
    public int SampleRate { get; set; } = 44100;

    /// <summary>Analysis frame length in samples. Default: 1764, 40 ms at 44.1 kHz.</summary>
    public int FftSize { get; set; } = 1764;

    /// <summary>Hop length in samples. Default: 882, 50% overlap.</summary>
    public int HopLength { get; set; } = 882;

    /// <summary>Number of mel bands. Default: 40.</summary>
    public int NumMels { get; set; } = 40;

    /// <summary>Frames concatenated into one feature vector, centred on the frame classified. Default: 5 (a 200-value
    /// vector with 40 bands).</summary>
    public int ContextFrames { get; set; } = 5;

    /// <summary>Units of each hidden dense layer. Default: 50.</summary>
    public int HiddenUnits { get; set; } = 50;

    /// <summary>Hidden dense layers. Default: 2.</summary>
    public int NumHiddenLayers { get; set; } = 2;

    /// <summary>Dropout after each hidden layer. Default: 0.2.</summary>
    public double DropoutRate { get; set; } = 0.2;

    /// <summary>Number of MFCCs in the descriptive <c>SceneFeatures</c> of a result (not a classifier input). Default:
    /// 20.</summary>
    public int NumMfccs { get; set; } = 20;

    /// <summary>Number of top predictions to return. Default: 3.</summary>
    public int TopK { get; set; } = 3;

    /// <summary>Custom scene labels (optional).</summary>
    public string[]? CustomScenes { get; set; }

    /// <summary>Path to ONNX model file (optional).</summary>
    public string? ModelPath { get; set; }

    /// <summary>ONNX model options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();
}
