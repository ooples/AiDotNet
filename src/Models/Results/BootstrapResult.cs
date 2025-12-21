namespace AiDotNet.Models.Results
{
    /// <summary>
    /// Represents the results of bootstrap validation for a machine learning model, containing R² metrics
    /// for training, validation, and test datasets.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Bootstrap validation is a resampling technique used to evaluate machine learning models by repeatedly 
    /// sampling from the available data with replacement. This class stores the R² (coefficient of determination) 
    /// values for different data splits in the bootstrap process. R² measures the proportion of variance in the 
    /// dependent variable that is predictable from the independent variables, with values ranging from 0 to 1 
    /// where higher values indicate better model fit. The class uses generic type parameter T to support different 
    /// numeric types for the R² values, such as float, double, or decimal.
    /// </para>
    /// <para><b>For Beginners:</b> This class stores how well a model performs on different parts of your data.
    /// 
    /// When evaluating machine learning models:
    /// - It's important to know how well they perform on different datasets
    /// - Bootstrap validation creates multiple samples from your data to test model stability
    /// - R² (R-squared) is a common metric that measures how well your model explains the variation in the data
    /// 
    /// This class stores three R² values:
    /// - Training R²: How well the model fits the data it was trained on
    /// - Validation R²: How well the model performs on data used for tuning
    /// - Test R²: How well the model generalizes to completely new data
    /// 
    /// These values help you understand:
    /// - If your model is underfitting (low R² on all datasets)
    /// - If your model is overfitting (high training R², much lower test R²)
    /// - How stable your model's performance is across different data splits
    /// </para>
    /// </remarks>
    /// <typeparam name="T">The numeric type used for R² values, typically float or double.</typeparam>
    public class BootstrapResult<T>
    {
        /// <summary>
        /// Gets or sets the R² value for the training dataset.
        /// </summary>
        /// <value>The R² value for the training dataset, initialized to zero.</value>
        /// <remarks>
        /// <para>
        /// This property represents the coefficient of determination (R²) for the training dataset, which is the 
        /// portion of data used to fit the model parameters. The R² value measures how well the model explains the 
        /// variance in the dependent variable, with values ranging from 0 to 1. A value of 1 indicates that the model 
        /// perfectly explains all the variance, while a value of 0 indicates that the model explains none of the 
        /// variance. The training R² is typically higher than validation or test R² because the model is optimized 
        /// specifically for the training data. A very high training R² compared to validation and test R² may indicate 
        /// overfitting.
        /// </para>
        /// <para><b>For Beginners:</b> This value shows how well your model fits the data it was trained on.
        /// 
        /// The training R²:
        /// - Measures how well your model explains the variation in the training data
        /// - Ranges from 0 (poor fit) to 1 (perfect fit)
        /// - Is calculated using the same data used to build the model
        /// 
        /// This value is important because:
        /// - It shows whether your model can capture patterns in the data
        /// - It serves as a baseline to compare with validation and test performance
        /// - Very high values (close to 1) might indicate overfitting
        /// 
        /// For example, a training R² of 0.85 means your model explains 85% of the
        /// variation in the training data, which generally indicates a good fit.
        /// </para>
        /// </remarks>
        public T TrainingR2 { get; set; }

        /// <summary>
        /// Gets or sets the R² value for the validation dataset.
        /// </summary>
        /// <value>The R² value for the validation dataset, initialized to zero.</value>
        /// <remarks>
        /// <para>
        /// This property represents the coefficient of determination (R²) for the validation dataset, which is the 
        /// portion of data used to tune hyperparameters and evaluate the model during training. The validation R² 
        /// provides an estimate of how well the model will perform on unseen data. It is typically lower than the 
        /// training R² but should be reasonably close if the model is not overfitting. A significant drop from 
        /// training R² to validation R² may indicate that the model is overfitting to the training data and not 
        /// generalizing well. In bootstrap validation, the validation set is typically created by sampling with 
        /// replacement from the original dataset.
        /// </para>
        /// <para><b>For Beginners:</b> This value shows how well your model performs on data used for tuning.
        /// 
        /// The validation R²:
        /// - Measures how well your model generalizes to data not used in training
        /// - Helps detect overfitting (when a model performs well on training data but poorly on new data)
        /// - Is used to guide model selection and hyperparameter tuning
        /// 
        /// This value is important because:
        /// - It provides feedback during the model development process
        /// - It helps you choose between different model configurations
        /// - It gives an early indication of how well the model might perform in production
        /// 
        /// For example, if your training R² is 0.85 but validation R² is only 0.65,
        /// this suggests your model might be overfitting to the training data.
        /// </para>
        /// </remarks>
        public T ValidationR2 { get; set; }

        /// <summary>
        /// Gets or sets the R² value for the test dataset.
        /// </summary>
        /// <value>The R² value for the test dataset, initialized to zero.</value>
        /// <remarks>
        /// <para>
        /// This property represents the coefficient of determination (R²) for the test dataset, which is the portion 
        /// of data set aside and not used during model training or validation. The test R² provides the most realistic 
        /// estimate of how well the model will perform on completely new, unseen data. It is the final evaluation 
        /// metric used to assess the model's generalization ability. The test R² should be reasonably close to the 
        /// validation R² if the validation process was effective. A significant drop from validation R² to test R² 
        /// may indicate that the validation process was not representative of real-world data or that the model was 
        /// indirectly overfitted through repeated hyperparameter tuning based on the validation set.
        /// </para>
        /// <para><b>For Beginners:</b> This value shows how well your model performs on completely new data.
        /// 
        /// The test R²:
        /// - Measures how well your model generalizes to data it has never seen before
        /// - Provides the most realistic estimate of real-world performance
        /// - Is the final evaluation metric for your model
        /// 
        /// This value is important because:
        /// - It represents how your model will likely perform in production
        /// - It's the most honest assessment of your model's capabilities
        /// - It helps determine if your model is ready for deployment
        /// 
        /// For example, a test R² of 0.80 means your model can explain 80% of the
        /// variation in new data, which would generally be considered good performance.
        /// </para>
        /// </remarks>
        public T TestR2 { get; set; }

        /// <summary>
        /// Initializes a new instance of the BootstrapResult class with all R² values set to zero.
        /// </summary>
        /// <remarks>
        /// <para>
        /// This constructor creates a new BootstrapResult instance and initializes all R² values (TrainingR2, 
        /// ValidationR2, and TestR2) to zero using the numeric operations appropriate for the generic type T. 
        /// It uses the MathHelper.GetNumericOperations method to obtain the appropriate numeric operations for 
        /// the type T, which allows the class to work with different numeric types such as float, double, or 
        /// decimal. This initialization ensures that the R² values start at a well-defined state before being 
        /// updated with actual results from the bootstrap validation process.
        /// </para>
        /// <para><b>For Beginners:</b> This constructor creates a new result object with all values initialized to zero.
        /// 
        /// When a new BootstrapResult is created:
        /// - All three R² values are set to zero
        /// - The constructor uses MathHelper to handle different numeric types
        /// - This provides a clean starting point before actual results are calculated
        /// 
        /// This initialization is important because:
        /// - It ensures consistent behavior regardless of how the object is created
        /// - It prevents potential issues with uninitialized values
        /// - It makes the code more robust across different numeric types
        /// 
        /// You typically won't need to call this constructor directly, as it will be
        /// used internally by the bootstrap validation process.
        /// </para>
        /// </remarks>
        public BootstrapResult()
        {
            var numOps = MathHelper.GetNumericOperations<T>();
            TrainingR2 = numOps.Zero;
            ValidationR2 = numOps.Zero;
            TestR2 = numOps.Zero;
        }
    }
}
