namespace AiDotNet.Augmentation.Text;

/// <summary>
/// Randomly swaps the positions of words in text.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Random swap changes word order, like swapping
/// "the big dog" to "big the dog". This helps models become less sensitive
/// to exact word ordering.</para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Sentiment analysis (where word presence matters more than order)</item>
/// <item>Topic classification</item>
/// <item>Tasks where word order is somewhat flexible</item>
/// </list>
/// </para>
/// <para><b>When NOT to use:</b>
/// <list type="bullet">
/// <item>Tasks where word order is critical (machine translation, grammar checking)</item>
/// <item>Named entity recognition</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class RandomSwap<T> : TextAugmenterBase<T>
{
    /// <summary>
    /// Gets the number of word swaps to perform.
    /// </summary>
    /// <remarks>
    /// <para>Default: 2</para>
    /// </remarks>
    public int NumSwaps { get; }

    /// <summary>
    /// Creates a new random swap augmentation.
    /// </summary>
    /// <param name="numSwaps">Number of swap operations to perform (default: 2).</param>
    /// <param name="probability">Probability of applying this augmentation (default: 0.2).</param>
    /// <param name="languageCode">Language code for language-specific operations.</param>
    public RandomSwap(
        int numSwaps = 2,
        double probability = 0.2,
        string languageCode = "en") : base(probability, languageCode)
    {
        if (numSwaps < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numSwaps),
                "Number of swaps must be non-negative.");
        }

        NumSwaps = numSwaps;
    }

    /// <inheritdoc />
    protected override string[] ApplyAugmentation(string[] data, AugmentationContext<T> context)
    {
        var result = new string[data.Length];

        for (int i = 0; i < data.Length; i++)
        {
            result[i] = AugmentText(data[i], context);
        }

        return result;
    }

    private string AugmentText(string text, AugmentationContext<T> context)
    {
        var tokens = Tokenize(text);
        if (tokens.Length < 2) return text;

        var result = (string[])tokens.Clone();

        for (int s = 0; s < NumSwaps; s++)
        {
            // Select two different random positions
            int pos1 = context.Random.Next(result.Length);
            int pos2 = context.Random.Next(result.Length);

            // Ensure they're different positions
            int attempts = 0;
            while (pos1 == pos2 && attempts < 10)
            {
                pos2 = context.Random.Next(result.Length);
                attempts++;
            }

            if (pos1 != pos2)
            {
                // Swap the words
                (result[pos1], result[pos2]) = (result[pos2], result[pos1]);
            }
        }

        return Detokenize(result);
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["numSwaps"] = NumSwaps;
        return parameters;
    }
}
