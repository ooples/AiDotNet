namespace AiDotNet.Models.Inputs;

/// <summary>
/// Represents the input data for optimization processes, including training, validation, and test datasets.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <typeparam name="TInput">The type of the input data, typically Matrix&lt;T&gt; or Tensor&lt;T&gt;.</typeparam>
/// <typeparam name="TOutput">The type of the output data, typically Vector&lt;T&gt; or Tensor&lt;T&gt;.</typeparam>
public class OptimizationInputData<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the input features for the training dataset.
    /// </summary>
    public TInput XTrain { get; set; }

    /// <summary>
    /// Gets or sets the target values for the training dataset.
    /// </summary>
    public TOutput YTrain { get; set; }

    /// <summary>
    /// Gets or sets the input features for the validation dataset.
    /// </summary>
    public TInput XValidation { get; set; }

    /// <summary>
    /// Gets or sets the target values for the validation dataset.
    /// </summary>
    public TOutput YValidation { get; set; }

    /// <summary>
    /// Gets or sets the input features for the test dataset.
    /// </summary>
    public TInput XTest { get; set; }

    /// <summary>
    /// Gets or sets the target values for the test dataset.
    /// </summary>
    public TOutput YTest { get; set; }

    /// <summary>
    /// Gets or sets the initial model/solution before optimization.
    /// Used by distributed optimizers to save parameters before local optimization.
    /// </summary>
    public IFullModel<T, TInput, TOutput>? InitialSolution { get; set; }

    /// <summary>
    /// Initializes a new instance of the OptimizationInputData class with empty datasets.
    /// </summary>
    public OptimizationInputData()
    {
        (XTrain, YTrain, _) = ModelHelper<T, TInput, TOutput>.CreateDefaultModelData();
        (XValidation, YValidation, _) = ModelHelper<T, TInput, TOutput>.CreateDefaultModelData();
        (XTest, YTest, _) = ModelHelper<T, TInput, TOutput>.CreateDefaultModelData();
    }
}
