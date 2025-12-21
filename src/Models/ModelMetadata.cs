namespace AiDotNet.Models;

/// <summary>
/// Represents metadata about a machine learning model, including its type, complexity, and additional descriptive information.
/// </summary>
/// <remarks>
/// <para>
/// This class encapsulates metadata about a machine learning model, providing information that describes the model's 
/// characteristics without containing the actual model implementation. It includes details such as the model type, 
/// feature count, complexity, and a textual description. Additionally, it provides an extensible dictionary for storing 
/// arbitrary additional information and can store serialized model data. This metadata is useful for model cataloging, 
/// selection, and management purposes.
/// </para>
/// <para><b>For Beginners:</b> This class stores descriptive information about a machine learning model.
/// 
/// When working with machine learning models:
/// - You often need to know what type of model it is (linear regression, neural network, etc.)
/// - You want to understand its complexity and what features it uses
/// - You may need to store additional information about how it was created or should be used
/// 
/// This class stores all that information, including:
/// - The type of model (classification, regression, etc.)
/// - How many features (input variables) it uses
/// - How complex the model is
/// - A human-readable description
/// - Any additional custom information you want to include
/// - The actual model data in serialized form
/// 
/// This metadata helps you understand what a model does and how it works
/// without having to examine the model itself.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class ModelMetadata<T>
{
    /// <summary>
    /// Gets or sets the name of the model.
    /// </summary>
    /// <value>A string representing the model's name.</value>
    /// <remarks>
    /// This property provides a human-readable name for the model, useful for identification and cataloging purposes.
    /// </remarks>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the version of the model.
    /// </summary>
    /// <value>A string representing the model's version.</value>
    /// <remarks>
    /// This property indicates the version of the model, which can be useful for tracking changes and updates over time.
    /// </remarks>
    public string Version { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the date and time (with timezone) when the model was trained.
    /// </summary>
    /// <value>A nullable DateTimeOffset representing when the model was trained, or null if unknown.</value>
    /// <remarks>
    /// This property stores the date and time (including timezone information) when the model was trained.
    /// It is nullable, allowing you to indicate when the training date is unknown or not set.
    /// Using DateTimeOffset ensures accurate tracking across different time zones.
    /// </remarks>
    public DateTimeOffset? TrainingDate { get; set; }

    /// <summary>
    /// Gets custom properties associated with the model.
    /// </summary>
    /// <value>A dictionary containing custom properties as key-value pairs.</value>
    /// <remarks>
    /// This property provides an extensible way to store custom properties and configuration settings specific to the model.
    /// It complements the AdditionalInfo property by providing a dedicated space for model-specific properties.
    /// Use <see cref="SetProperty"/> and <see cref="RemoveProperty"/> methods to modify the properties.
    /// </remarks>
    public Dictionary<string, object> Properties { get; private set; } = [];

    /// <summary>
    /// Adds or updates a custom property in the Properties dictionary.
    /// </summary>
    /// <param name="key">The property key.</param>
    /// <param name="value">The property value.</param>
    public void SetProperty(string key, object value)
    {
        Properties[key] = value;
    }

    /// <summary>
    /// Removes a custom property from the Properties dictionary.
    /// </summary>
    /// <param name="key">The property key to remove.</param>
    /// <returns>True if the property was removed; otherwise, false.</returns>
    public bool RemoveProperty(string key)
    {
        return Properties.Remove(key);
    }

    /// <summary>
    /// Gets or sets the type of the model.
    /// </summary>
    /// <value>A ModelType enumeration value indicating the model's type.</value>
    /// <remarks>
    /// <para>
    /// This property indicates the type of the model, such as regression, classification, clustering, or time series.
    /// The model type provides a high-level categorization of what the model does and what kind of problems it is designed
    /// to solve. This information is useful for understanding the model's purpose and for selecting appropriate models for
    /// specific tasks.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you what kind of problem the model is designed to solve.
    ///
    /// The model type:
    /// - Indicates whether the model is for regression, classification, clustering, etc.
    /// - Helps you understand what the model is designed to do
    /// - Guides how the model's outputs should be interpreted
    ///
    /// Common model types include:
    /// - Regression: Predicts continuous values (like prices, temperatures)
    /// - Classification: Predicts categories or classes (like spam/not spam)
    /// - Clustering: Groups similar items together
    /// - Time Series: Makes predictions based on time-ordered data
    ///
    /// Knowing the model type is essential for using the model correctly and
    /// understanding what kind of predictions it can make.
    /// </para>
    /// </remarks>
    public ModelType ModelType { get; set; }

    /// <summary>
    /// Gets or sets the number of features used by the model.
    /// </summary>
    /// <value>An integer representing the number of input features.</value>
    /// <remarks>
    /// <para>
    /// This property indicates how many input features or variables the model uses to make predictions. The feature count 
    /// is an important aspect of the model's structure and can affect its performance, interpretability, and computational 
    /// requirements. Models with more features may capture more complex patterns but might also be more prone to overfitting 
    /// and require more data to train effectively.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you how many input variables the model uses to make predictions.
    /// 
    /// The feature count:
    /// - Indicates how many different input variables the model considers
    /// - Affects the model's complexity and interpretability
    /// - Influences how much data is needed to train the model effectively
    /// 
    /// For example, a house price prediction model might use 10 features such as
    /// square footage, number of bedrooms, location, etc.
    /// 
    /// This information is important because:
    /// - It helps you prepare the right number of inputs when using the model
    /// - It gives you an idea of the model's complexity
    /// - It can indicate potential issues (too many features might cause overfitting)
    /// </para>
    /// </remarks>
    public int FeatureCount { get; set; }

    /// <summary>
    /// Gets or sets a measure of the model's complexity.
    /// </summary>
    /// <value>An integer representing the model's complexity.</value>
    /// <remarks>
    /// <para>
    /// This property provides a measure of the model's complexity, which can vary depending on the type of model. For example, 
    /// in a linear model, complexity might refer to the number of coefficients; in a decision tree, it might refer to the 
    /// depth or number of nodes; in a neural network, it might refer to the number of layers or parameters. Higher complexity 
    /// can allow a model to capture more intricate patterns but may also increase the risk of overfitting and the computational 
    /// resources required.
    /// </para>
    /// <para><b>For Beginners:</b> This indicates how complex or sophisticated the model is.
    /// 
    /// The complexity measure:
    /// - Gives you an idea of how intricate the model's structure is
    /// - Can mean different things for different model types
    /// - Higher values generally indicate more complex models
    /// 
    /// For different model types, complexity might represent:
    /// - Linear models: Number of coefficients or terms
    /// - Decision trees: Depth of the tree or number of nodes
    /// - Neural networks: Number of layers or parameters
    /// 
    /// This information is useful because:
    /// - More complex models can capture more intricate patterns
    /// - But they may also be more prone to overfitting
    /// - And they typically require more computational resources
    /// </para>
    /// </remarks>
    public int Complexity { get; set; }

    /// <summary>
    /// Gets or sets a human-readable description of the model.
    /// </summary>
    /// <value>A string containing a description of the model.</value>
    /// <remarks>
    /// <para>
    /// This property provides a human-readable description of the model, which can include information about its purpose, 
    /// how it was created, its strengths and limitations, or any other relevant details. The description is useful for 
    /// documentation purposes and for helping users understand what the model does and how it should be used.
    /// </para>
    /// <para><b>For Beginners:</b> This provides a human-readable explanation of what the model does and how it works.
    /// 
    /// The description:
    /// - Explains the model's purpose and functionality in plain language
    /// - May include information about how it was created or trained
    /// - Can describe the model's strengths, limitations, or intended use cases
    /// 
    /// For example, a description might be:
    /// "This is a gradient boosting model trained to predict house prices based on 10 features
    /// including square footage, location, and number of bedrooms. It was trained on data
    /// from 2010-2020 and performs best on properties in urban areas."
    /// 
    /// This information is valuable for:
    /// - Documentation purposes
    /// - Helping others understand the model
    /// - Remembering the model's purpose and limitations later
    /// </para>
    /// </remarks>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets additional information about the model as key-value pairs.
    /// </summary>
    /// <value>A dictionary containing additional model information.</value>
    /// <remarks>
    /// <para>
    /// This property provides an extensible way to store additional information about the model as key-value pairs. This can 
    /// include any information that doesn't fit into the other properties, such as the date the model was created, the author, 
    /// the dataset used for training, performance metrics, hyperparameter values, or any other relevant metadata. The flexible 
    /// dictionary structure allows for storing arbitrary information without modifying the class structure.
    /// </para>
    /// <para><b>For Beginners:</b> This allows you to store any additional custom information about the model.
    /// 
    /// The additional info dictionary:
    /// - Stores any extra information as key-value pairs
    /// - Is flexible and can contain any type of data
    /// - Allows you to extend the metadata without changing the class structure
    /// 
    /// Common types of information you might store:
    /// - Creation date: When the model was built
    /// - Author: Who created the model
    /// - Training dataset: What data was used to train it
    /// - Performance metrics: How well it performed in testing
    /// - Hyperparameters: Specific configuration settings used
    /// 
    /// This flexibility is useful because:
    /// - Different models may need different types of metadata
    /// - You can add custom information without modifying the class
    /// - It allows for future extensibility as needs change
    /// </para>
    /// </remarks>
    public Dictionary<string, object> AdditionalInfo { get; set; } = [];

    /// <summary>
    /// Gets or sets the serialized model data.
    /// </summary>
    /// <value>A byte array containing the serialized model data.</value>
    /// <remarks>
    /// <para>
    /// This property stores the serialized model data as a byte array. This can be used to store a compact representation of 
    /// the model itself, allowing the model to be reconstructed from the metadata. The serialization format depends on the 
    /// specific model implementation and serialization mechanism used. This property is particularly useful for model 
    /// persistence, sharing, and deployment scenarios.
    /// </para>
    /// <para><b>For Beginners:</b> This stores the actual model in a serialized (binary) format.
    /// 
    /// The model data:
    /// - Contains the serialized (converted to bytes) model itself
    /// - Allows the model to be stored, transmitted, or reconstructed later
    /// - Is in a format that depends on how the model was serialized
    /// 
    /// This is useful for:
    /// - Saving the model to a file or database
    /// - Transmitting the model over a network
    /// - Reconstructing the model without having to retrain it
    /// 
    /// For example, you might serialize a trained model to this property,
    /// then save the entire ModelMetadata object to a file. Later, you can
    /// load the file and deserialize the model data to recreate the working model.
    /// </para>
    /// </remarks>
    public byte[] ModelData { get; set; } = [];

    /// <summary>
    /// Gets or sets the importance of each feature in the model.
    /// </summary>
    /// <value>A dictionary mapping feature names to their importance values.</value>
    /// <remarks>
    /// <para>
    /// This property stores the importance of each feature used by the model. Feature importance indicates how much each input 
    /// variable contributes to the model's predictions. The exact meaning and scale of importance values can vary depending on 
    /// the model type and the method used to calculate importance. This information is crucial for understanding which features 
    /// have the most significant impact on the model's output and can be used for feature selection or model interpretation.
    /// </para>
    /// <para><b>For Beginners:</b> This shows how important each input variable is to the model's predictions.
    /// 
    /// The feature importance dictionary:
    /// - Maps each feature (input variable) name to a number representing its importance
    /// - Higher values typically indicate more important features
    /// - The exact meaning of the values depends on the type of model and how importance was calculated
    /// 
    /// This is useful for:
    /// - Understanding which inputs have the biggest impact on predictions
    /// - Identifying irrelevant or less important features
    /// - Simplifying the model by focusing on the most important features
    /// 
    /// For example, in a house price prediction model, you might see that 'location' has a high importance value, 
    /// while 'roof color' has a low value, indicating that location strongly influences the prediction but roof color doesn't.
    /// </para>
    /// </remarks>
    public Dictionary<string, T> FeatureImportance { get; set; } = [];
}
