namespace AiDotNet.Enums;

/// <summary>
/// Defines the direction of sequential feature selection.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Sequential feature selection can work in two directions:
/// starting with no features and adding them, or starting with all features and removing them.
/// </para>
/// <para>
/// Think of it like packing for a trip:
/// - Forward Selection: Start with an empty suitcase and add items one by one, choosing the most
///   important item each time until you have enough.
/// - Backward Elimination: Start with a full suitcase and remove items one by one, removing the
///   least important item each time until you reach your desired size.
/// </para>
/// </remarks>
public enum SequentialFeatureSelectionDirection
{
    /// <summary>
    /// Forward selection starts with zero features and incrementally adds the feature that most improves performance.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Forward selection is like building a team by adding members one at a time.
    /// You start with no one, then add the person who contributes the most. Then you add another person
    /// who, combined with the first, provides the best improvement. You continue until you have your
    /// desired team size.
    /// </para>
    /// <para>
    /// Advantages:
    /// - Fast when you want to select only a small number of features
    /// - Good for exploring which features work well together
    /// </para>
    /// <para>
    /// Disadvantages:
    /// - Once a feature is added, it stays (no way to remove it later if it becomes redundant)
    /// - Can be slow if you want to select many features
    /// </para>
    /// </remarks>
    Forward,

    /// <summary>
    /// Backward elimination starts with all features and iteratively removes the feature whose removal least degrades performance.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Backward elimination is like pruning a tree. You start with all the branches
    /// and trim away the ones that contribute least to the tree's health. You keep trimming until you
    /// reach your desired size.
    /// </para>
    /// <para>
    /// Advantages:
    /// - Considers all features working together initially
    /// - Fast when you want to keep most features
    /// - Good at identifying truly redundant features
    /// </para>
    /// <para>
    /// Disadvantages:
    /// - Slow to start (must train with all features initially)
    /// - Computationally expensive with many features
    /// - Once a feature is removed, it can't be added back
    /// </para>
    /// </remarks>
    Backward
}
