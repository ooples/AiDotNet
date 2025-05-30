namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for selecting models in dynamic ensemble methods.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> In dynamic selection, instead of using all models for every 
/// prediction, we choose the best model(s) based on the specific input. It's like 
/// choosing different experts for different types of questions.
/// </para>
/// </remarks>
public interface IModelSelector<T>
{
    /// <summary>
    /// Selects the best models for a specific input.
    /// </summary>
    /// <param name="input">The input data to make predictions for.</param>
    /// <param name="models">The available models to choose from.</param>
    /// <returns>Indices of the selected models.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This looks at the input and decides which models are likely 
    /// to make the best predictions for this specific case.
    /// </remarks>
    List<int> SelectModelsForInput(Tensor<T> input, IReadOnlyList<IFullModel<T, Tensor<T>, Tensor<T>>> models);
    
    /// <summary>
    /// Updates the selector's internal state based on prediction performance.
    /// </summary>
    /// <param name="input">The input that was predicted.</param>
    /// <param name="predictions">The predictions made by each model.</param>
    /// <param name="actual">The actual correct output.</param>
    /// <param name="modelIndices">The indices of models that made predictions.</param>
    /// <remarks>
    /// <b>For Beginners:</b> This helps the selector learn which models work best for 
    /// different types of inputs by tracking their performance over time.
    /// </remarks>
    void UpdatePerformance(Tensor<T> input, List<Tensor<T>> predictions, Tensor<T> actual, List<int> modelIndices);
    
    /// <summary>
    /// Gets the selection method name.
    /// </summary>
    string SelectionMethod { get; }
}