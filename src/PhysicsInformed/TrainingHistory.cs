using System.Collections.Generic;

namespace AiDotNet.PhysicsInformed;

/// <summary>
/// Stores training history for analysis.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class TrainingHistory<T>
{
    public List<T> Losses { get; } = new List<T>();

    public void AddEpoch(T loss)
    {
        Losses.Add(loss);
    }
}
