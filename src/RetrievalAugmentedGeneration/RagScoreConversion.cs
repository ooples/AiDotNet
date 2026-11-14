using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.RetrievalAugmentedGeneration;

/// <summary>
/// Converts relevance scores between <typeparamref name="T"/> and double in one engine pass, for the RAG
/// components whose scoring heuristics or external APIs work in double.
/// </summary>
/// <typeparam name="T">The numeric type of the scores.</typeparam>
internal static class RagScoreConversion<T>
{
    /// <summary>Converts <typeparamref name="T"/> scores to double (works for every T, including Half).</summary>
    public static double[] ToDouble(IEnumerable<T> values)
    {
        var array = values.ToArray();
        return array.Length == 0 ? new double[0] : new Tensor<T>(array, [array.Length]).Cast<double>().ToArray();
    }

    /// <summary>Converts double scores to <typeparamref name="T"/>.</summary>
    public static T[] FromDouble(IEnumerable<double> values)
    {
        var array = values.ToArray();
        return array.Length == 0 ? new T[0] : new Tensor<double>(array, [array.Length]).Cast<T>().ToArray();
    }
}
