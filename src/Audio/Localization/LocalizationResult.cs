namespace AiDotNet.Audio.Localization;

/// <summary>
/// Result of sound source localization.
/// </summary>
public class LocalizationResult
{
    /// <summary>Estimated azimuth angle in degrees (-180 to 180).</summary>
    public double AzimuthDegrees { get; init; }

    /// <summary>Estimated elevation angle in degrees (-90 to 90).</summary>
    public double ElevationDegrees { get; init; }

    /// <summary>Time difference of arrival in samples.</summary>
    public int TdoaSamples { get; init; }

    /// <summary>Time difference of arrival in seconds.</summary>
    public double TdoaSeconds { get; init; }

    /// <summary>Confidence of the estimate (0-1).</summary>
    public double Confidence { get; init; }

    /// <summary>Algorithm used for localization.</summary>
    public required string Algorithm { get; init; }

    /// <summary>
    /// Gets direction as unit vector (x, y, z).
    /// </summary>
    public (double X, double Y, double Z) GetDirectionVector()
    {
        double azimuthRad = AzimuthDegrees * Math.PI / 180;
        double elevationRad = ElevationDegrees * Math.PI / 180;

        double x = Math.Cos(elevationRad) * Math.Sin(azimuthRad);
        double y = Math.Cos(elevationRad) * Math.Cos(azimuthRad);
        double z = Math.Sin(elevationRad);

        return (x, y, z);
    }
}
