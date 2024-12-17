namespace AiDotNet.Models;

internal class BasicStatsInputs<T>
{
    public Vector<T> Values { get; set; } = Vector<T>.Empty();
}