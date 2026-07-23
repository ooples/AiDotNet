namespace AiDotNet.Enums;

/// <summary>
/// Specifies the direction of feature selection in stepwise regression and other statistical models.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Stepwise methods help AI decide which information is important to consider.
/// 
/// Imagine you're trying to predict house prices. There are many factors that could affect the price:
/// square footage, number of bedrooms, location, age of the house, etc. But using too many factors
/// can make your model complicated and less accurate.
/// 
/// Stepwise methods help you decide which factors (called "features" in AI) to include in your model.
/// They work by either starting with nothing and adding important features one by one, or starting
/// with everything and removing less important features one by one.
/// 
/// Think of it like packing for a trip: you can either start with an empty suitcase and add only what
/// you need (Forward), or start with everything you own and remove what you don't need (Backward).
/// </remarks>
public enum StepwiseMethod
{
    /// <summary>
    /// Starts with no features and adds them one at a time based on their statistical significance.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is like building a team by starting with no players and adding the best
    /// available player one at a time.
    /// 
    /// How it works:
    /// 1. Start with an empty model (no features/variables)
    /// 2. Try adding each available feature one by one
    /// 3. Add the feature that improves the model the most
    /// 4. Repeat steps 2-3 until no more features significantly improve the model
    /// 
    /// Advantages:
    /// - Tends to create simpler models with fewer features
    /// - Often faster when you have many potential features
    /// - Easier to understand which features are most important
    /// 
    /// This method is useful when you have many potential features and want to build a
    /// streamlined model with only the most important ones.
    /// </remarks>
    Forward,

    /// <summary>
    /// Starts with all features and removes them one at a time based on their lack of statistical significance.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is like starting with all players on a team and removing the least
    /// valuable players one by one.
    /// 
    /// How it works:
    /// 1. Start with a full model (all features/variables included)
    /// 2. Try removing each feature one by one
    /// 3. Remove the feature that hurts the model the least
    /// 4. Repeat steps 2-3 until removing any remaining feature would significantly harm the model
    /// 
    /// Advantages:
    /// - Less likely to miss important interactions between features
    /// - Can be better when features are correlated with each other
    /// - May find more complex relationships in the data
    /// 
    /// This method is useful when you suspect that combinations of features might be important,
    /// or when you're not sure which features to exclude.
    /// </remarks>
    Backward
}
