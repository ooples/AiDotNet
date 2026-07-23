namespace AiDotNet.Preprocessing.Document;

/// <summary>
/// Internal helper methods for preprocessing utilities.
/// Provides compatibility with .NET Framework 4.7.1.
/// </summary>
internal static class PreprocessingHelpers
{
    /// <summary>
    /// Clamps a value between min and max.
    /// Compatible with .NET Framework 4.7.1 which doesn't have Math.Clamp.
    /// </summary>
    internal static int Clamp(int value, int min, int max)
    {
        if (value < min) return min;
        if (value > max) return max;
        return value;
    }

    /// <summary>
    /// Clamps a double value between min and max.
    /// </summary>
    internal static double Clamp(double value, double min, double max)
    {
        if (value < min) return min;
        if (value > max) return max;
        return value;
    }

    /// <summary>
    /// Fills an array with a value.
    /// Compatible with .NET Framework 4.7.1 which doesn't have Array.Fill.
    /// </summary>
    internal static void Fill<T>(T[] array, T value)
    {
        for (int i = 0; i < array.Length; i++)
        {
            array[i] = value;
        }
    }
}
