using System;
using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for TOTEM (TOkenized Time Series EMbeddings).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// TOTEM learns discrete tokenized representations for time series via VQ-VAE,
/// enabling the use of discrete token-based methods (like LLMs) on continuous time series data.
/// </para>
/// <para>
/// <b>Reference:</b> Talukder et al., "TOTEM: TOkenized Time Series EMbeddings for General Time Series Analysis", 2024.
/// </para>
/// </remarks>
public class TOTEMOptions<T> : TimeSeriesRegressionOptions<T>
{
    public TOTEMOptions() { }

    public TOTEMOptions(TOTEMOptions<T> other)
    {
        if (other == null) throw new ArgumentNullException(nameof(other));
        ContextLength = other.ContextLength; ForecastHorizon = other.ForecastHorizon;
        HiddenDimension = other.HiddenDimension; NumLayers = other.NumLayers;
        NumHeads = other.NumHeads; CodebookSize = other.CodebookSize;
        CodebookDimension = other.CodebookDimension; NumCodebooks = other.NumCodebooks;
        DropoutRate = other.DropoutRate; CommitmentWeight = other.CommitmentWeight;
    }

    public int ContextLength { get; set; } = 512;
    public int ForecastHorizon { get; set; } = 96;
    public int HiddenDimension { get; set; } = 256;
    public int NumLayers { get; set; } = 6;
    public int NumHeads { get; set; } = 8;
    public int CodebookSize { get; set; } = 1024;
    public int CodebookDimension { get; set; } = 64;
    public int NumCodebooks { get; set; } = 4;
    public double DropoutRate { get; set; } = 0.1;
    public double CommitmentWeight { get; set; } = 0.25;
}
