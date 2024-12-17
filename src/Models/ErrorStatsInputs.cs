namespace AiDotNet.Models;

internal class ErrorStatsInputs<T>
{
    public Vector<T> Actual { get; set; } = Vector<T>.Empty();
    public Vector<T> Predicted { get; set; } = Vector<T>.Empty();
    public int FeatureCount { get; set; }
}