namespace AiDotNet.Interfaces;

/// <summary>
/// Defines capability to shuffle data for randomized iteration.
/// </summary>
/// <remarks>
/// <para>
/// Data loaders that implement this interface can shuffle their data,
/// which is important for training to prevent the model from learning
/// the order of examples rather than the patterns in the data.
/// </para>
/// <para><b>For Beginners:</b> Shuffling is like shuffling a deck of cards before dealing.
/// When training, you don't want your model to learn "cat images always come first,
/// then dog images" - you want it to learn actual features. Shuffling ensures each
/// epoch sees data in a different order.
/// </para>
/// </remarks>
public interface IShuffleable
{
    /// <summary>
    /// Gets whether the data is currently shuffled.
    /// </summary>
    bool IsShuffled { get; }

    /// <summary>
    /// Shuffles the data order using the specified seed for reproducibility.
    /// </summary>
    /// <param name="seed">Optional seed for reproducible shuffling. Same seed produces same order.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> The seed is like a "recipe" for the randomness.
    /// Using the same seed gives the same "random" order every time, which is useful
    /// for reproducing experiments or debugging.
    /// </para>
    /// </remarks>
    void Shuffle(int? seed = null);

    /// <summary>
    /// Restores the original (unshuffled) data order.
    /// </summary>
    void Unshuffle();
}
