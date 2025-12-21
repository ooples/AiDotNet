namespace AiDotNet.Models
{
    /// <summary>
    /// Represents a comprehensive collection of evaluation data for a model across training, validation, and test datasets.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This class encapsulates all the evaluation data for a machine learning model, including detailed statistics for the 
    /// training, validation, and test datasets, as well as overall model statistics. It provides a complete picture of how 
    /// the model performs across different datasets, which is essential for assessing model quality, diagnosing issues like 
    /// overfitting or underfitting, and making informed decisions about model selection and improvement.
    /// </para>
    /// <para><b>For Beginners:</b> This class stores all the performance information about a model across different datasets.
    /// 
    /// When evaluating a machine learning model:
    /// - You typically split your data into training, validation, and test sets
    /// - You need to track how well the model performs on each set
    /// - You want to compare performance across these sets to detect issues like overfitting
    /// 
    /// This class organizes all that information in one place, including:
    /// - Detailed statistics for each dataset (training, validation, test)
    /// - Overall model statistics and metrics
    /// 
    /// Having this structured collection makes it easier to:
    /// - Evaluate model quality
    /// - Compare different models
    /// - Generate reports and visualizations
    /// - Make informed decisions about model selection and improvement
    /// </para>
    /// </remarks>
    /// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
    public class ModelEvaluationData<T, TInput, TOutput>
    {
        /// <summary>
        /// Gets or sets the statistics for the training dataset.
        /// </summary>
        /// <value>A DataSetStats&lt;T&gt; object containing detailed statistics for the training dataset.</value>
        /// <remarks>
        /// <para>
        /// This property contains detailed statistics for the model's performance on the training dataset. The training dataset 
        /// is the portion of the data used to train the model, and the model's performance on this dataset indicates how well 
        /// it has learned the patterns in the training data. However, good performance on the training dataset alone does not 
        /// guarantee that the model will generalize well to new, unseen data. Comparing the model's performance on the training 
        /// dataset with its performance on the validation and test datasets can help identify issues like overfitting.
        /// </para>
        /// <para><b>For Beginners:</b> This contains detailed information about how the model performs on the data it was trained with.
        /// 
        /// The training set statistics:
        /// - Show how well the model learned from the training data
        /// - Include error measurements, prediction quality metrics, and basic statistics
        /// - Store the actual input features, target values, and model predictions
        /// 
        /// This information is important because:
        /// - It helps you understand what the model learned
        /// - It serves as a baseline for comparison with validation and test performance
        /// - Very high performance here but poor performance on other sets can indicate overfitting
        /// 
        /// For example, if your model achieves 95% accuracy on the training data but only 70%
        /// on the validation data, it might be memorizing the training data rather than learning
        /// generalizable patterns.
        /// </para>
        /// </remarks>
        public DataSetStats<T, TInput, TOutput> TrainingSet { get; set; } = new();

        /// <summary>
        /// Gets or sets the statistics for the validation dataset.
        /// </summary>
        /// <value>A DataSetStats&lt;T&gt; object containing detailed statistics for the validation dataset.</value>
        /// <remarks>
        /// <para>
        /// This property contains detailed statistics for the model's performance on the validation dataset. The validation 
        /// dataset is a portion of the data that is not used for training but is used to tune hyperparameters and make decisions 
        /// about model selection. The model's performance on the validation dataset provides an estimate of how well it will 
        /// generalize to new, unseen data. Comparing the model's performance on the validation dataset with its performance on 
        /// the training dataset can help identify issues like overfitting, where the model performs well on the training data 
        /// but poorly on new data.
        /// </para>
        /// <para><b>For Beginners:</b> This contains detailed information about how the model performs on data it has seen but wasn't trained with.
        /// 
        /// The validation set statistics:
        /// - Show how well the model generalizes to data it wasn't trained on
        /// - Help detect overfitting (when the model performs much better on training than validation)
        /// - Guide decisions about model selection and hyperparameter tuning
        /// 
        /// This information is important because:
        /// - It provides a more realistic estimate of model performance than training data
        /// - It helps you make decisions about when to stop training or which model to choose
        /// - It can reveal problems before you test on your final test set
        /// 
        /// For example, if you're trying different model configurations, you would compare
        /// their performance on the validation set to decide which one to use.
        /// </para>
        /// </remarks>
        public DataSetStats<T, TInput, TOutput> ValidationSet { get; set; } = new();

        /// <summary>
        /// Gets or sets the statistics for the test dataset.
        /// </summary>
        /// <value>A DataSetStats&lt;T&gt; object containing detailed statistics for the test dataset.</value>
        /// <remarks>
        /// <para>
        /// This property contains detailed statistics for the model's performance on the test dataset. The test dataset is a 
        /// portion of the data that is set aside and not used during training or validation. It provides an unbiased evaluation 
        /// of the final model's performance and an estimate of how well the model will perform on new, unseen data in the real 
        /// world. The test dataset should only be used once, after all model selection and tuning is complete, to avoid 
        /// inadvertently fitting the model to the test data through multiple evaluations and adjustments.
        /// </para>
        /// <para><b>For Beginners:</b> This contains detailed information about how the model performs on completely new data.
        /// 
        /// The test set statistics:
        /// - Show how well the model will likely perform in the real world
        /// - Provide the final evaluation of model quality
        /// - Represent the most realistic assessment of model performance
        /// 
        /// This information is important because:
        /// - It gives you an unbiased estimate of how your model will perform on new data
        /// - It's the ultimate measure of whether your model is ready for deployment
        /// - It helps set realistic expectations for real-world performance
        /// 
        /// The test set should only be used once, after all model development is complete,
        /// to get a true measure of performance without any bias from the development process.
        /// </para>
        /// </remarks>
        public DataSetStats<T, TInput, TOutput> TestSet { get; set; } = new();

        /// <summary>
        /// Gets or sets the overall statistics for the model.
        /// </summary>
        /// <value>A ModelStats&lt;T&gt; object containing overall model statistics and metrics.</value>
        /// <remarks>
        /// <para>
        /// This property contains overall statistics and metrics for the model that are not specific to any particular dataset. 
        /// These might include information about the model's complexity, training time, memory usage, and other characteristics 
        /// that are important for understanding the model's behavior and performance. The model statistics complement the 
        /// dataset-specific statistics by providing a broader view of the model's qualities and limitations.
        /// </para>
        /// <para><b>For Beginners:</b> This contains general information about the model itself, not tied to a specific dataset.
        /// 
        /// The model statistics:
        /// - Describe characteristics of the model itself
        /// - May include information about model complexity, size, or training process
        /// - Provide metrics that apply to the model as a whole
        /// 
        /// This information is important because:
        /// - It helps you understand the model's overall characteristics
        /// - It can include metrics that aren't specific to a single dataset
        /// - It provides context for interpreting the dataset-specific results
        /// 
        /// For example, this might include information about:
        /// - The number of parameters in the model
        /// - How long the model took to train
        /// - Memory requirements for the model
        /// - Overall complexity measures
        /// </para>
        /// </remarks>
        public ModelStats<T, TInput, TOutput> ModelStats { get; set; } = ModelStats<T, TInput, TOutput>.Empty();
    }
}
