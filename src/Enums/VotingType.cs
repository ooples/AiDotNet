namespace AiDotNet.Enums
{
    /// <summary>
    /// Defines the types of voting strategies available for ensemble models.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When multiple models make predictions, we need a way to combine
    /// their outputs into a single final prediction. Voting strategies determine how this
    /// combination is performed.
    /// </para>
    /// </remarks>
    public enum VotingType
    {
        /// <summary>
        /// Each model gets one vote for its predicted class.
        /// </summary>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> Hard voting is like taking a majority vote. Each model
        /// predicts a class, and the class that gets the most votes wins. For example,
        /// if 3 models predict "cat" and 2 predict "dog", the final prediction is "cat".
        /// </para>
        /// <para>
        /// Best for: Classification tasks where models output discrete class labels.
        /// </para>
        /// </remarks>
        Hard,
        
        /// <summary>
        /// Models vote with their prediction probabilities.
        /// </summary>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> Soft voting uses the confidence levels (probabilities) from
        /// each model. Instead of just counting votes, it averages the probability estimates.
        /// For example, if one model is 90% confident it's a "cat" and another is 60% confident,
        /// soft voting considers these confidence levels.
        /// </para>
        /// <para>
        /// Best for: Classification tasks where models can output probability estimates,
        /// typically resulting in better performance than hard voting.
        /// </para>
        /// </remarks>
        Soft,
        
        /// <summary>
        /// Models vote with weights based on their performance.
        /// </summary>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> Weighted voting gives more importance to better-performing
        /// models. If one model has 95% accuracy and another has 80% accuracy, the first
        /// model's prediction counts more in the final decision.
        /// </para>
        /// <para>
        /// Best for: Situations where you have models with varying performance levels
        /// and want to leverage the strengths of better models while still benefiting
        /// from the diversity of all models.
        /// </para>
        /// </remarks>
        Weighted
    }
}