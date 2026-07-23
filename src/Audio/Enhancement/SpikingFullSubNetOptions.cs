using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Enhancement;

/// <summary>
/// Configuration options for the Spiking-FullSubNet model.
/// </summary>
/// <remarks>
/// <para>
/// Spiking-FullSubNet (Yu et al., 2023) replaces traditional activation functions in the
/// FullSubNet architecture with spiking neural network (SNN) neurons, achieving comparable
/// speech enhancement quality with significantly lower computational cost (energy efficiency).
/// It combines full-band and sub-band processing with bio-inspired spiking activations.
/// </para>
/// <para>
/// <b>For Beginners:</b> This model works like the brain's neurons - they "fire" or "don't fire"
/// (binary spikes) instead of using continuous values. This makes it much more energy-efficient
/// while still cleaning up noisy audio effectively. Great for battery-powered devices.
/// </para>
/// </remarks>
public class SpikingFullSubNetOptions : ModelOptions
{
    #region Audio

    /// <summary>Gets or sets the audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>Gets or sets the FFT size.</summary>
    public int FftSize { get; set; } = 512;

    /// <summary>Gets or sets the hop length.</summary>
    public int HopLength { get; set; } = 256;

    /// <summary>Gets or sets the number of frequency bins.</summary>
    public int NumFreqBins { get; set; } = 257;

    #endregion

    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public string Variant { get; set; } = "base";

    /// <summary>Gets or sets the full-band hidden size.</summary>
    public int FullBandHiddenSize { get; set; } = 512;

    /// <summary>Gets or sets the sub-band hidden size.</summary>
    public int SubBandHiddenSize { get; set; } = 384;

    /// <summary>Gets or sets the number of full-band layers.</summary>
    public int NumFullBandLayers { get; set; } = 2;

    /// <summary>Gets or sets the number of sub-band layers.</summary>
    public int NumSubBandLayers { get; set; } = 2;

    /// <summary>Gets or sets the spiking neuron threshold.</summary>
    public double SpikingThreshold { get; set; } = 1.0;

    /// <summary>Gets or sets the spiking neuron time constant.</summary>
    public double TimeConstant { get; set; } = 0.25;

    /// <summary>Gets or sets the enhancement strength (0.0-1.0).</summary>
    public double EnhancementStrength { get; set; } = 1.0;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the learning rate.</summary>
    public double LearningRate { get; set; } = 1e-3;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.0;

    #endregion
}
