global using Newtonsoft.Json;
global using Formatting = Newtonsoft.Json.Formatting;

namespace AiDotNet.Models.Results;

/// <summary>
/// Represents a complete predictive model with its optimization results, normalization information, and metadata.
/// This class implements the IPredictiveModel interface and provides serialization capabilities.
/// </summary>
/// <remarks>
/// <para>
/// This class encapsulates a trained predictive model along with all the information needed to use it for making 
/// predictions on new data. It includes the model itself, the results of the optimization process that created the 
/// model, normalization information for preprocessing input data and postprocessing predictions, and metadata about 
/// the model. The class also provides methods for serializing and deserializing the model, allowing it to be saved 
/// to and loaded from files.
/// </para>
/// <para><b>For Beginners:</b> This class represents a complete, ready-to-use predictive model.
/// 
/// When working with machine learning models:
/// - You need to store not just the model itself, but also how to prepare data for it
/// - You want to keep track of how the model was created and how well it performs
/// - You need to be able to save the model and load it later
/// 
/// This class handles all of that by storing:
/// - The actual model that makes predictions
/// - Information about how the model was optimized
/// - How to normalize/scale input data before making predictions
/// - Metadata about the model (like feature names, creation date, etc.)
/// 
/// It also provides methods to:
/// - Make predictions on new data
/// - Save the model to a file
/// - Load a model from a file
/// 
/// This makes it easy to train a model once and then use it many times in different applications.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[Serializable]
internal class PredictionModelResult<T, TInput, TOutput> : IPredictiveModel<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the underlying model used for making predictions.
    /// </summary>
    /// <value>An implementation of IFullModel&lt;T&gt; representing the trained model.</value>
    /// <remarks>
    /// <para>
    /// This property contains the actual model that is used to make predictions. The model implements the IFullModel&lt;T&gt; 
    /// interface, which provides methods for predicting outputs based on input features. The specific implementation could 
    /// be a linear regression model, a polynomial model, a neural network, or any other type of model that implements the 
    /// interface. This property is marked as nullable, but must be initialized before the model can be used for predictions.
    /// </para>
    /// <para><b>For Beginners:</b> This is the actual trained model that makes predictions.
    /// 
    /// The model:
    /// - Contains the mathematical formula or algorithm for making predictions
    /// - Is the core component that transforms input data into predictions
    /// - Could be a linear model, polynomial model, neural network, etc.
    /// 
    /// This property is marked as nullable (with the ? symbol) because:
    /// - When deserializing a model from storage, it might not be immediately available
    /// - The default constructor creates an empty object that will be populated later
    /// 
    /// However, the model must be initialized before you can use the Predict method,
    /// or you'll get an InvalidOperationException.
    /// </para>
    /// </remarks>
    public IFullModel<T, TInput, TOutput>? Model { get; private set; }
    
    /// <summary>
    /// Gets or sets the results of the optimization process that created the model.
    /// </summary>
    /// <value>An OptimizationResult&lt;T&gt; object containing detailed optimization information.</value>
    /// <remarks>
    /// <para>
    /// This property contains the results of the optimization process that was used to create the model. It includes 
    /// information such as the best fitness score achieved, the number of iterations performed, the history of fitness 
    /// scores during optimization, and detailed performance metrics on training, validation, and test datasets. This 
    /// information is useful for understanding how the model was created and how well it performs on different datasets.
    /// </para>
    /// <para><b>For Beginners:</b> This contains information about how the model was created and how well it performs.
    /// 
    /// The optimization result:
    /// - Records how the model was trained/optimized
    /// - Includes performance metrics on different datasets
    /// - Stores information about feature selection
    /// - Contains analysis of potential issues like overfitting
    /// 
    /// This information is valuable because:
    /// - It helps you understand the model's strengths and limitations
    /// - It provides context for interpreting predictions
    /// - It can guide decisions about when to retrain the model
    /// 
    /// For example, you might check the R-squared value on the test dataset
    /// to understand how well the model is likely to perform on new data.
    /// </para>
    /// </remarks>
    public OptimizationResult<T, TInput, TOutput> OptimizationResult { get; private set; } = new();
    
    /// <summary>
    /// Gets or sets the normalization information used to preprocess input data and postprocess predictions.
    /// </summary>
    /// <value>A NormalizationInfo&lt;T&gt; object containing normalization parameters and the normalizer.</value>
    /// <remarks>
    /// <para>
    /// This property contains information about how input data should be normalized before being fed into the model, and 
    /// how the model's outputs should be denormalized to obtain the final predictions. Normalization is a preprocessing 
    /// step that scales the input features to a standard range, which can improve the performance and stability of many 
    /// machine learning algorithms. The NormalizationInfo object includes the normalizer object that performs the actual 
    /// normalization and denormalization operations, as well as parameters that describe how the target variable (Y) was 
    /// normalized during training.
    /// </para>
    /// <para><b>For Beginners:</b> This contains information about how to scale data before and after prediction.
    /// 
    /// The normalization info:
    /// - Stores how input features should be scaled before making predictions
    /// - Stores how to convert predictions back to their original scale
    /// - Contains the actual normalizer object that performs these operations
    /// 
    /// Normalization is important because:
    /// - Many models perform better with normalized input data
    /// - The model was trained on normalized data, so new data must be normalized the same way
    /// - Predictions need to be converted back to the original scale to be meaningful
    /// 
    /// For example, if your input features were originally in different units (like dollars, years, and percentages),
    /// normalization might scale them all to a range of 0-1 for the model, and then the predictions
    /// need to be scaled back to the original units.
    /// </para>
    /// </remarks>
    public NormalizationInfo<T, TInput, TOutput> NormalizationInfo { get; private set; } = new();
    
    /// <summary>
    /// Gets or sets the metadata associated with the model.
    /// </summary>
    /// <value>A ModelMetadata&lt;T&gt; object containing descriptive information about the model.</value>
    /// <remarks>
    /// <para>
    /// This property contains metadata about the model, such as the names of the input features, the name of the target 
    /// variable, the date and time the model was created, the type of model, and any additional descriptive information. 
    /// This metadata is useful for understanding what the model does and how it should be used, without having to examine 
    /// the model itself. It can also be used for documentation, versioning, and tracking purposes.
    /// </para>
    /// <para><b>For Beginners:</b> This contains descriptive information about the model.
    /// 
    /// The model metadata:
    /// - Stores information like feature names and target variable name
    /// - Records when the model was created
    /// - Describes what type of model it is
    /// - May include additional descriptive information
    /// 
    /// This information is useful because:
    /// - It helps you understand what the model is predicting and what inputs it needs
    /// - It provides documentation for the model
    /// - It can help with versioning and tracking different models
    /// 
    /// For example, the metadata might tell you that this model predicts "house_price"
    /// based on features like "square_footage", "num_bedrooms", and "location_score".
    /// </para>
    /// </remarks>
    public ModelMetaData<T> ModelMetadata { get; private set; } = new();

    /// <summary>
    /// Initializes a new instance of the PredictionModelResult class with the specified model, optimization results, and normalization information.
    /// </summary>
    /// <param name="model">The underlying model used for making predictions.</param>
    /// <param name="optimizationResult">The results of the optimization process that created the model.</param>
    /// <param name="normalizationInfo">The normalization information used to preprocess input data and postprocess predictions.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new PredictionModelResult instance with the specified model, optimization results, and 
    /// normalization information. It also initializes the ModelMetadata property by calling the GetModelMetadata method on 
    /// the provided model. This constructor is typically used when a new model has been trained and needs to be packaged 
    /// with all the necessary information for making predictions and for later serialization.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates a new prediction model result with all the necessary components.
    /// 
    /// When creating a new PredictionModelResult:
    /// - You provide the trained model that will make predictions
    /// - You provide the optimization results that describe how the model was created
    /// - You provide the normalization information needed to process data
    /// - The constructor automatically extracts metadata from the model
    /// 
    /// This constructor is typically used when:
    /// - You've just finished training a model
    /// - You want to package it with all the information needed to use it
    /// - You plan to save it for later use or deploy it in an application
    /// 
    /// For example, after training a house price prediction model, you would use this constructor
    /// to create a complete package that can be saved and used for making predictions.
    /// </para>
    /// </remarks>
    public PredictionModelResult(IFullModel<T, TInput, TOutput>? model, OptimizationResult<T, TInput, TOutput> optimizationResult, 
        NormalizationInfo<T, TInput, TOutput> normalizationInfo)
    {
        Model = model;
        OptimizationResult = optimizationResult;
        NormalizationInfo = normalizationInfo;
        ModelMetadata = model?.GetModelMetaData() ?? new();
    }

    /// <summary>
    /// Initializes a new instance of the PredictionModelResult class with default values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This constructor creates a new PredictionModelResult instance with default values for all properties. It is primarily 
    /// used for deserialization, where the properties will be populated from the deserialized data. This constructor should 
    /// not be used directly for creating models that will be used for predictions, as the Model property will be null and 
    /// the Predict method will throw an exception.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates an empty prediction model result that will be filled in later.
    /// 
    /// This default constructor:
    /// - Creates an object with default/empty values
    /// - Is primarily used during deserialization (loading from a file)
    /// - Should not be used directly when you want to make predictions
    /// 
    /// When this constructor is used:
    /// - The Model property is null
    /// - The other properties are initialized with empty objects
    /// 
    /// If you try to call Predict on an object created with this constructor without
    /// first deserializing data into it, you'll get an error because the Model is null.
    /// </para>
    /// </remarks>
    public PredictionModelResult()
    {
    }

    /// <summary>
    /// Gets the metadata associated with the model.
    /// </summary>
    /// <returns>A ModelMetadata&lt;T&gt; object containing descriptive information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the metadata associated with the model, which is stored in the ModelMetadata property. It is 
    /// implemented to satisfy the IPredictiveModel interface, which requires a method to retrieve model metadata. The 
    /// metadata includes information such as the names of the input features, the name of the target variable, the date 
    /// and time the model was created, the type of model, and any additional descriptive information.
    /// </para>
    /// <para><b>For Beginners:</b> This method returns descriptive information about the model.
    /// 
    /// The GetModelMetadata method:
    /// - Returns the metadata stored in the ModelMetadata property
    /// - Is required by the IPredictiveModel interface
    /// - Provides access to information about what the model does and how it works
    /// 
    /// This method is useful when:
    /// - You want to display information about the model
    /// - You need to check what features the model expects
    /// - You're working with multiple models and need to identify them
    /// 
    /// For example, you might call this method to get the list of feature names
    /// so you can ensure your input data has the correct columns.
    /// </para>
    /// </remarks>
    public ModelMetaData<T> GetModelMetadata()
    {
        return ModelMetadata;
    }

    /// <summary>
    /// Makes predictions using the model on the provided input data.
    /// </summary>
    /// <param name="newData">A matrix of input features, where each row represents an observation and each column represents a feature.</param>
    /// <returns>A vector of predicted values, one for each observation in the input matrix.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the Model or Normalizer is not initialized.</exception>
    /// <remarks>
    /// <para>
    /// This method makes predictions using the model on the provided input data. It first normalizes the input data using 
    /// the normalizer from the NormalizationInfo property, then passes the normalized data to the model's Predict method, 
    /// and finally denormalizes the model's outputs to obtain the final predictions. This process ensures that the input 
    /// data is preprocessed in the same way as the training data was, and that the predictions are in the same scale as 
    /// the original target variable.
    /// </para>
    /// <para><b>For Beginners:</b> This method makes predictions on new data using the trained model.
    /// 
    /// The Predict method:
    /// - Takes a matrix of input features as its parameter
    /// - Normalizes the input data to match how the model was trained
    /// - Passes the normalized data to the model for prediction
    /// - Denormalizes the results to convert them back to the original scale
    /// - Returns a vector of predictions, one for each row in the input matrix
    /// 
    /// This method will throw an exception if:
    /// - The Model property is null (not initialized)
    /// - The Normalizer in NormalizationInfo is null (not initialized)
    /// 
    /// For example, if you have a matrix of house features (square footage, bedrooms, etc.),
    /// this method will return a vector of predicted house prices.
    /// </para>
    /// </remarks>
    public TOutput Predict(TInput newData)
    {
        if (Model == null)
        {
            throw new InvalidOperationException("Model is not initialized.");
        }

        if (NormalizationInfo.Normalizer == null)
        {
            throw new InvalidOperationException("Normalizer is not initialized.");
        }

        var (normalizedNewData, _) = NormalizationInfo.Normalizer.NormalizeInput(newData);
        var normalizedPredictions = Model.Predict(normalizedNewData);

        return NormalizationInfo.Normalizer.Denormalize(normalizedPredictions, NormalizationInfo.YParams);
    }

    /// <summary>
    /// Serializes the model to a byte array.
    /// </summary>
    /// <returns>A byte array containing the serialized model.</returns>
    /// <remarks>
    /// <para>
    /// This method serializes the entire PredictionModelResult object, including the model, optimization results, normalization 
    /// information, and metadata, to a JSON string and then converts it to a byte array. The serialization uses Newtonsoft.Json 
    /// with TypeNameHandling.All to ensure that all type information is preserved, which is necessary for correctly deserializing 
    /// the model later. This is particularly important for polymorphic types like the Model property, which could be any 
    /// implementation of IFullModel&lt;T&gt;.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts the model into a format that can be stored or transmitted.
    /// 
    /// The Serialize method:
    /// - Converts the entire model object to JSON format
    /// - Includes type information to ensure proper deserialization
    /// - Returns the result as a byte array that can be saved to a file or database
    /// 
    /// The serialization process:
    /// - Uses Newtonsoft.Json for the conversion
    /// - Preserves all type information with TypeNameHandling.All
    /// - Formats the JSON with indentation for readability
    /// 
    /// This method is useful when you need to:
    /// - Save a model for later use
    /// - Send a model to another application
    /// - Store a model in a database
    /// </para>
    /// </remarks>
    public byte[] Serialize()
    {
        var jsonString = JsonConvert.SerializeObject(this, Formatting.Indented, new JsonSerializerSettings
        {
            TypeNameHandling = TypeNameHandling.All
        });

        return Encoding.UTF8.GetBytes(jsonString);
    }

    /// <summary>
    /// Deserializes a model from a byte array.
    /// </summary>
    /// <param name="data">A byte array containing the serialized model.</param>
    /// <exception cref="InvalidOperationException">Thrown when deserialization fails.</exception>
    /// <remarks>
    /// <para>
    /// This method deserializes a PredictionModelResult object from a byte array. It first converts the byte array to a JSON 
    /// string, then uses Newtonsoft.Json to deserialize the string into a PredictionModelResult object. The deserialization 
    /// uses TypeNameHandling.All to correctly handle polymorphic types like the Model property. If deserialization is successful, 
    /// the method updates all properties of the current instance with the values from the deserialized object. If deserialization 
    /// fails, an InvalidOperationException is thrown.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a model from a previously serialized byte array.
    /// 
    /// The Deserialize method:
    /// - Takes a byte array containing a serialized model
    /// - Converts it back into a usable PredictionModelResult object
    /// - Updates the current instance with all the deserialized data
    /// 
    /// The deserialization process:
    /// - Converts the byte array to a JSON string
    /// - Uses Newtonsoft.Json to parse the JSON
    /// - Handles type information with TypeNameHandling.All
    /// - Copies all properties from the deserialized object to the current instance
    /// 
    /// This method will throw an exception if:
    /// - The deserialization process fails
    /// - The byte array doesn't contain a valid serialized model
    /// 
    /// This method is typically used when:
    /// - Loading a model from a file or database
    /// - Receiving a model from another application
    /// </para>
    /// </remarks>
    public void Deserialize(byte[] data)
    {
        var jsonString = Encoding.UTF8.GetString(data);
        var deserializedObject = JsonConvert.DeserializeObject<PredictionModelResult<T, TInput, TOutput>>(jsonString, new JsonSerializerSettings
        {
            TypeNameHandling = TypeNameHandling.All
        });

        if (deserializedObject != null)
        {
            Model = deserializedObject.Model;
            OptimizationResult = deserializedObject.OptimizationResult;
            NormalizationInfo = deserializedObject.NormalizationInfo;
            ModelMetadata = deserializedObject.ModelMetadata;
        }
        else
        {
            throw new InvalidOperationException("Failed to deserialize the model.");
        }
    }

    /// <summary>
    /// Saves the model to a file.
    /// </summary>
    /// <param name="filePath">The path where the model will be saved.</param>
    /// <remarks>
    /// <para>
    /// This method saves the serialized model to a file at the specified path. It first serializes the model to a byte array 
    /// using the Serialize method, then writes the byte array to the specified file. If the file already exists, it will be 
    /// overwritten. This method provides a convenient way to persist the model for later use.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves the model to a file on disk.
    /// 
    /// The SaveModel method:
    /// - Takes a file path where the model should be saved
    /// - Serializes the model to a byte array
    /// - Writes the byte array to the specified file
    /// 
    /// This method is useful when:
    /// - You want to save a trained model for later use
    /// - You need to share a model with others
    /// - You want to deploy a model to a production environment
    /// 
    /// For example, after training a model, you might save it with:
    /// `myModel.SaveModel("C:\\Models\\house_price_predictor.model");`
    /// 
    /// If the file already exists, it will be overwritten.
    /// </para>
    /// </remarks>
    public void SaveModel(string filePath)
    {
        File.WriteAllBytes(filePath, Serialize());
    }

    /// <summary>
    /// Loads a model from a file.
    /// </summary>
    /// <param name="filePath">The path of the file containing the serialized model.</param>
    /// <returns>A new PredictionModelResult&lt;T&gt; instance loaded from the file.</returns>
    /// <remarks>
    /// <para>
    /// This static method loads a serialized model from a file at the specified path. It first reads the file as a byte array, 
    /// then creates a new PredictionModelResult instance and deserializes the byte array into it using the Deserialize method. 
    /// This method provides a convenient way to load a previously saved model for use in making predictions.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a previously saved model from a file.
    /// 
    /// The LoadModel method:
    /// - Takes a file path where the model is stored
    /// - Reads the file into a byte array
    /// - Creates a new PredictionModelResult object
    /// - Deserializes the byte array into the object
    /// - Returns the fully loaded model
    /// 
    /// This method is useful when:
    /// - You want to use a previously trained model
    /// - You're deploying a model in a production environment
    /// - You're sharing models between different applications
    /// 
    /// For example, you might load a model with:
    /// ```csharp
    /// var model = new PredictionModelResult<double, Matrix<double>, Vector<double>>();
    /// model.LoadModel("C:\\Models\\house_price_predictor.model");
    /// ```
    /// </para>
    /// </remarks>
    public void LoadModel(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path must not be null or empty.", nameof(filePath));
        }

        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"The specified model file does not exist: {filePath}", filePath);
        }

        var data = File.ReadAllBytes(filePath);
        Deserialize(data);
    }

    public void Train(TInput input, TOutput expectedOutput)
    {
        if (Model == null)
        {
            throw new InvalidOperationException("Model is not initialized.");
        }
        Model.Train(input, expectedOutput);
    }

    public ModelMetaData<T> GetModelMetaData()
    {
        // Return the stored metadata (populated from the underlying model when available)
        return ModelMetadata;
    }

    public Vector<T> GetParameters()
    {
        if (Model == null)
        {
            throw new InvalidOperationException("Model is not initialized.");
        }
        return Model.GetParameters();
    }

    public IFullModel<T, TInput, TOutput> WithParameters(Vector<T> parameters)
    {
        if (Model == null)
        {
            throw new InvalidOperationException("Model is not initialized.");
        }
        return Model.WithParameters(parameters);
    }

    public IEnumerable<int> GetActiveFeatureIndices()
    {
        if (Model == null)
        {
            // No model initialized; no active features to report
            return Array.Empty<int>();
        }
        return Model.GetActiveFeatureIndices();
    }

    public bool IsFeatureUsed(int featureIndex)
    {
        if (Model == null)
        {
            return false;
        }
        return Model.IsFeatureUsed(featureIndex);
    }

    public IFullModel<T, TInput, TOutput> DeepCopy()
    {
        if (Model == null)
        {
            throw new InvalidOperationException("Model is not initialized.");
        }
        return Model.DeepCopy();
    }

    public IFullModel<T, TInput, TOutput> Clone()
    {
        if (Model == null)
        {
            throw new InvalidOperationException("Model is not initialized.");
        }
        return Model.Clone();
    }
}
