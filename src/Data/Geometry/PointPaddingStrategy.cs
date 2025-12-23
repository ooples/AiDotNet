namespace AiDotNet.Data.Geometry;

/// <summary>
/// Strategies for padding point clouds when fewer points than requested exist.
/// </summary>
public enum PointPaddingStrategy
{
    Repeat,
    Zero
}
