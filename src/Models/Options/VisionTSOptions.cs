using System;
using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for VisionTS (Visual Masked Autoencoders as Zero-Shot Time Series Forecasters).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// VisionTS repurposes Visual Masked Autoencoders (MAE) for time series forecasting,
/// demonstrating that vision foundation models can transfer effectively to the time
/// series domain through cross-modal transfer.
/// </para>
/// <para><b>For Beginners:</b> VisionTS brings image AI to time series:
///
/// <b>Cross-Modal Transfer:</b>
/// VisionTS converts time series into 2D image-like representations, then uses
/// a pretrained Visual MAE (originally trained on images) to process them. This
/// leverages the massive pretraining of vision models for time series tasks.
///
/// <b>How It Works:</b>
/// 1. Convert time series to 2D patch grid (like an image)
/// 2. Mask some patches (MAE-style)
/// 3. Use the pretrained ViT encoder to process visible patches
/// 4. Decode masked patches to reconstruct/forecast the series
/// </para>
/// <para>
/// <b>Reference:</b> "VisionTS: Visual Masked Autoencoders as Zero-Shot Time Series Forecasters",
/// ICML 2025.
/// </para>
/// </remarks>
public class VisionTSOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public VisionTSOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    public VisionTSOptions(VisionTSOptions<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        ContextLength = other.ContextLength;
        ForecastHorizon = other.ForecastHorizon;
        PatchLength = other.PatchLength;
        HiddenDimension = other.HiddenDimension;
        NumLayers = other.NumLayers;
        NumHeads = other.NumHeads;
        IntermediateSize = other.IntermediateSize;
        DropoutRate = other.DropoutRate;
        ModelSize = other.ModelSize;
        MaskRatio = other.MaskRatio;
        ImageHeight = other.ImageHeight;
        ImageWidth = other.ImageWidth;
    }

    /// <summary>
    /// Gets or sets the context length.
    /// </summary>
    /// <value>Defaults to 512.</value>
    public int ContextLength { get; set; } = 512;

    /// <summary>
    /// Gets or sets the forecast horizon.
    /// </summary>
    /// <value>Defaults to 96.</value>
    public int ForecastHorizon { get; set; } = 96;

    /// <summary>
    /// Gets or sets the patch length for 2D patch grid conversion.
    /// </summary>
    /// <value>Defaults to 16.</value>
    public int PatchLength { get; set; } = 16;

    /// <summary>
    /// Gets or sets the hidden dimension of the ViT encoder.
    /// </summary>
    /// <value>Defaults to 768.</value>
    public int HiddenDimension { get; set; } = 768;

    /// <summary>
    /// Gets or sets the number of transformer layers.
    /// </summary>
    /// <value>Defaults to 12.</value>
    public int NumLayers { get; set; } = 12;

    /// <summary>
    /// Gets or sets the number of attention heads.
    /// </summary>
    /// <value>Defaults to 12.</value>
    public int NumHeads { get; set; } = 12;

    /// <summary>
    /// Gets or sets the intermediate size.
    /// </summary>
    /// <value>Defaults to 3072.</value>
    public int IntermediateSize { get; set; } = 3072;

    /// <summary>
    /// Gets or sets the dropout rate.
    /// </summary>
    /// <value>Defaults to 0.1.</value>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the model size variant.
    /// </summary>
    /// <value>Defaults to <see cref="FoundationModelSize.Base"/>.</value>
    public FoundationModelSize ModelSize { get; set; } = FoundationModelSize.Base;

    /// <summary>
    /// Gets or sets the mask ratio for MAE pretraining.
    /// </summary>
    /// <value>Defaults to 0.75 (75% masking, standard for MAE).</value>
    public double MaskRatio { get; set; } = 0.75;

    /// <summary>
    /// Gets or sets the image height for 2D conversion.
    /// </summary>
    /// <value>Defaults to 224 (standard ViT input).</value>
    public int ImageHeight { get; set; } = 224;

    /// <summary>
    /// Gets or sets the image width for 2D conversion.
    /// </summary>
    /// <value>Defaults to 224 (standard ViT input).</value>
    public int ImageWidth { get; set; } = 224;
}
