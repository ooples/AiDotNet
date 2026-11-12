namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>Checks embedding-table IDs supplied through the numeric tensor API.</summary>
internal static class TtsTokenIds
{
    internal static int[] Read<T>(Tensor<T> input, int vocabulary, int maximumLength, string component)
    {
        Validation.Guard.NotNull(input);
        if (input.Length == 0 || input.Length > maximumLength)
            throw new ArgumentException($"{component} needs between 1 and {maximumLength} token IDs.", nameof(input));
        var ops = MathHelper.GetNumericOperations<T>();
        var ids = new int[input.Length];
        for (int i = 0; i < ids.Length; i++)
        {
            double value = ops.ToDouble(input.GetFlat(i));
            if (double.IsNaN(value) || double.IsInfinity(value) || value != Math.Truncate(value) || value < 0 || value >= vocabulary)
                throw new ArgumentException($"{component} token {i} must be an integer ID in [0, {vocabulary}).", nameof(input));
            ids[i] = (int)value;
        }
        return ids;
    }
}
