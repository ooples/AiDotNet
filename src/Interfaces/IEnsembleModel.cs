namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for ensemble models that combine predictions from multiple base models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix<double>, Tensor<double>, Vector<double>).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector<double>, Tensor<double>).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> An ensemble model is like a team of experts working together. Instead of 
/// relying on just one model's prediction, it combines predictions from multiple models to get a 
/// more accurate and reliable result. This is similar to getting multiple opinions before making 
/// an important decision.
/// </para>
/// <para>
/// Ensemble models can combine different types of models (neural networks, regression models, 
/// time series models, etc.) and use various strategies to merge their predictions. Common strategies 
/// include averaging, voting, stacking, and dynamic selection based on the input data.
/// </para>
/// </remarks>
public interface IEnsembleModel<T, TInput, TOutput> : IFullModel<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the collection of base models in the ensemble.
    /// </summary>
    /// <value>A read-only list of models that make up the ensemble.</value>
    /// <remarks>
    /// <b>For Beginners:</b> These are the individual "expert" models whose predictions 
    /// will be combined. Each model can be of a different type and trained differently.
    /// </remarks>
    IReadOnlyList<IFullModel<T, TInput, TOutput>> BaseModels { get; }
    
    /// <summary>
    /// Gets the combination strategy used to merge predictions.
    /// </summary>
    /// <value>The strategy that determines how predictions are combined.</value>
    /// <remarks>
    /// <b>For Beginners:</b> This is the method used to combine predictions from all the 
    /// models. For example, it might average them, pick the most common answer (voting), 
    /// or use a more sophisticated approach.
    /// </remarks>
    ICombinationStrategy<T, TInput, TOutput> CombinationStrategy { get; }
    
    /// <summary>
    /// Gets the weights assigned to each base model.
    /// </summary>
    /// <value>A vector where each element represents the importance of the corresponding model.</value>
    /// <remarks>
    /// <b>For Beginners:</b> Not all models are equally good. Weights let us give more 
    /// importance to better models when combining predictions. A model with weight 2.0 
    /// has twice the influence of a model with weight 1.0.
    /// </remarks>
    Vector<T> ModelWeights { get; }
    
    /// <summary>
    /// Adds a new model to the ensemble.
    /// </summary>
    /// <param name="model">The model to add to the ensemble.</param>
    /// <param name="weight">The initial weight for this model.</param>
    /// <remarks>
    /// <b>For Beginners:</b> Use this to add another "expert" to your team. The weight 
    /// determines how much this model's opinion matters compared to others.
    /// </remarks>
    void AddModel(IFullModel<T, TInput, TOutput> model, T weight);
    
    /// <summary>
    /// Removes a model from the ensemble.
    /// </summary>
    /// <param name="model">The model to remove from the ensemble.</param>
    /// <returns>True if the model was found and removed; false otherwise.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Sometimes you might want to remove a model that's not 
    /// performing well or is no longer needed.
    /// </remarks>
    bool RemoveModel(IFullModel<T, TInput, TOutput> model);
    
    /// <summary>
    /// Updates the weights of the base models.
    /// </summary>
    /// <param name="newWeights">The new weights to assign to the models.</param>
    /// <remarks>
    /// <b>For Beginners:</b> After seeing how well each model performs, you might want 
    /// to adjust their weights to give more importance to the better performers.
    /// </remarks>
    void UpdateWeights(Vector<T> newWeights);
    
    /// <summary>
    /// Gets individual predictions from each base model.
    /// </summary>
    /// <param name="input">The input data to make predictions for.</param>
    /// <returns>A list of predictions, one from each model in the ensemble.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This lets you see what each individual model predicts before 
    /// they're combined. Useful for understanding how the ensemble makes its decision.
    /// </remarks>
    List<TOutput> GetIndividualPredictions(TInput input);
}