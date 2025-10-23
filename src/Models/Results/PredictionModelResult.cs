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
public class PredictionModelResult<T, TInput, TOutput> : IPredictiveModel<T, TInput, TOutput>
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
    public ModelMetadata<T> ModelMetadata { get; private set; } = new();

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
    public PredictionModelResult(OptimizationResult<T, TInput, TOutput> optimizationResult,
        NormalizationInfo<T, TInput, TOutput> normalizationInfo)
    {
        Model = optimizationResult.BestSolution;
        OptimizationResult = optimizationResult;
        NormalizationInfo = normalizationInfo;
        ModelMetadata = Model?.GetModelMetadata() ?? new();
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
    internal PredictionModelResult()
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
    public ModelMetadata<T> GetModelMetadata()
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
    /// This method serializes the entire PredictionModelResult object, including the model, optimization results, 
    /// normalization information, and metadata. The model is serialized using its own Serialize() method, 
    /// ensuring that model-specific serialization logic is properly applied. The other components are 
    /// serialized using JSON. This approach ensures that each component of the PredictionModelResult is 
    /// serialized in the most appropriate way.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts the model into a format that can be stored or transmitted.
    /// 
    /// The Serialize method:
    /// - Uses the model's own serialization method to properly handle model-specific details
    /// - Serializes other components (optimization results, normalization info, metadata) to JSON
    /// - Combines everything into a single byte array that can be saved to a file or database
    /// 
    /// This is important because:
    /// - Different model types may need to be serialized differently
    /// - It ensures all the model's internal details are properly preserved
    /// - It allows for more efficient and robust storage of the complete prediction model package
    /// </para>
    /// </remarks>
    public byte[] Serialize()
    {
        try
        {
            // Register all converters using our centralized registry
            JsonConverterRegistry.RegisterAllConverters();

            // Create JSON settings with custom converters for our types
            var settings = new JsonSerializerSettings
            {
                TypeNameHandling = TypeNameHandling.All,
                Formatting = Formatting.Indented
            };

            // Add all needed converters from the registry
            var allConverters = JsonConverterRegistry.GetAllConverters();

            // Add type-specific converters for T
            var typeSpecificConverters = JsonConverterRegistry.GetConvertersForType<T>();

            // Combine converters
            var converters = new List<JsonConverter>();
            converters.AddRange(allConverters);
            converters.AddRange(typeSpecificConverters);

            // Set the converters on our settings
            settings.Converters = converters;

            // Serialize the object
            var jsonString = JsonConvert.SerializeObject(this, settings);
            return Encoding.UTF8.GetBytes(jsonString);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to serialize the model: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Deserializes a model from a byte array.
    /// </summary>
    /// <param name="data">A byte array containing the serialized model.</param>
    /// <exception cref="InvalidOperationException">Thrown when deserialization fails.</exception>
    /// <remarks>
    /// <para>
    /// This method reconstructs a PredictionModelResult object from a serialized byte array. It reads 
    /// the serialized data of each component (model, optimization results, normalization information, 
    /// and metadata) and deserializes them using the appropriate methods. The model is deserialized 
    /// using its model-specific deserialization method, while the other components are deserialized 
    /// from JSON.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a model from a previously serialized byte array.
    /// 
    /// The Deserialize method:
    /// - Takes a byte array containing a serialized model
    /// - Extracts each component (model, optimization results, etc.)
    /// - Uses the appropriate deserialization method for each component
    /// - Reconstructs the complete PredictionModelResult object
    /// 
    /// This approach ensures:
    /// - Each model type is deserialized correctly using its own specific logic
    /// - All model parameters and settings are properly restored
    /// - The complete prediction pipeline (normalization, prediction, denormalization) is reconstructed
    /// 
    /// This method will throw an exception if the deserialization process fails for any component.
    /// </para>
    /// </remarks>
    public void Deserialize(byte[] data)
    {
        try
        {
            // Register all converters using our centralized registry
            JsonConverterRegistry.RegisterAllConverters();

            var jsonString = Encoding.UTF8.GetString(data);

            // Create JSON settings with custom converters for our types
            var settings = new JsonSerializerSettings
            {
                TypeNameHandling = TypeNameHandling.All
            };

            // Add all needed converters from the registry
            var allConverters = JsonConverterRegistry.GetAllConverters();

            // Add type-specific converters for T
            var typeSpecificConverters = JsonConverterRegistry.GetConvertersForType<T>();

            // Combine converters
            var converters = new List<JsonConverter>();
            converters.AddRange(allConverters);
            converters.AddRange(typeSpecificConverters);

            // Set the converters on our settings
            settings.Converters = converters;

            // Deserialize the object
            var deserializedObject = JsonConvert.DeserializeObject<PredictionModelResult<T, TInput, TOutput>>(jsonString, settings);

            if (deserializedObject != null)
            {
                Model = deserializedObject.Model;
                OptimizationResult = deserializedObject.OptimizationResult;
                NormalizationInfo = deserializedObject.NormalizationInfo;
                ModelMetadata = deserializedObject.ModelMetadata;
            }
            else
            {
                throw new InvalidOperationException("Deserialization resulted in a null object.");
            }
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to deserialize the model: {ex.Message}", ex);
        }
/// Represents the complete result of a prediction model building process, including the trained model,
/// optimization results, and normalization information.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The type of input data (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The type of output data (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> This class wraps a trained prediction model along with all the information
/// needed to use it effectively. It includes:
/// - The trained model itself
/// - Information about how the model was optimized
/// - Details about how the data was normalized
///
/// This wrapper allows the model to be used for predictions and provides access
/// to all the metadata about how it was created and how well it performs.
///
/// <b>IMPORTANT LIMITATION:</b> SaveModel and LoadModel methods are not yet fully implemented.
/// While SaveModel will serialize the model data to a file, LoadModel cannot currently deserialize
/// the data back into a working model due to incomplete type reconstruction logic. Use with caution
/// in production scenarios that require model persistence.
/// </remarks>
public class PredictionModelResult<T, TInput, TOutput> : IPredictiveModel<T, TInput, TOutput>, IFullModel<T, TInput, TOutput>
{
    private IFullModel<T, TInput, TOutput>? _innerModel;
    private OptimizationResult<T, TInput, TOutput>? _optimizationResult;
    private NormalizationInfo<T, TInput, TOutput>? _normalizationInfo;

    /// <summary>
    /// Gets the inner model used for predictions.
    /// </summary>
    public IFullModel<T, TInput, TOutput>? Model => _innerModel;

    /// <summary>
    /// Gets the optimization result containing detailed information about the training process.
    /// </summary>
    public OptimizationResult<T, TInput, TOutput>? OptimizationResult => _optimizationResult;

    /// <summary>
    /// Gets the normalization information used to preprocess the data.
    /// </summary>
    public NormalizationInfo<T, TInput, TOutput>? NormalizationInfo => _normalizationInfo;

    /// <summary>
    /// Initializes a new instance of the PredictionModelResult class.
    /// </summary>
    /// <param name="model">The trained model.</param>
    /// <param name="optimizationResult">The optimization results.</param>
    /// <param name="normalizationInfo">The normalization information.</param>
    public PredictionModelResult(
        IFullModel<T, TInput, TOutput>? model,
        OptimizationResult<T, TInput, TOutput> optimizationResult,
        NormalizationInfo<T, TInput, TOutput> normalizationInfo)
    {
        _innerModel = model;
        _optimizationResult = optimizationResult ?? throw new ArgumentNullException(nameof(optimizationResult));
        _normalizationInfo = normalizationInfo ?? throw new ArgumentNullException(nameof(normalizationInfo));
    }

    /// <summary>
    /// Initializes a new instance of the PredictionModelResult class for deserialization.
    /// </summary>
    public PredictionModelResult()
    {
        _innerModel = null;
        _optimizationResult = null;
        _normalizationInfo = null;
    }

    #region IPredictiveModel Implementation

    /// <summary>
    /// Makes predictions using the trained model on new input data.
    /// </summary>
    /// <param name="input">The input data to make predictions for.</param>
    /// <returns>The predicted output values.</returns>
    public TOutput Predict(TInput input)
    {
        if (_innerModel == null)
        {
            throw new InvalidOperationException("Model has not been initialized. Cannot make predictions.");
        }

        return _innerModel.Predict(input);
    }

    /// <summary>
    /// Retrieves metadata and performance information about the trained model.
    /// </summary>
    /// <returns>Metadata about the model's performance and configuration.</returns>
    public ModelMetaData<T> GetModelMetadata()
    {
        if (_innerModel == null)
        {
            throw new InvalidOperationException("Model has not been initialized. Cannot retrieve metadata.");
        }

        return _innerModel.GetModelMetaData();
    }

    #endregion

    #region IModel Implementation

    /// <summary>
    /// Trains the model using input features and their corresponding target values.
    /// </summary>
    /// <param name="input">The input training data.</param>
    /// <param name="expectedOutput">The expected output values.</param>
    public void Train(TInput input, TOutput expectedOutput)
    {
        if (_innerModel == null)
        {
            throw new InvalidOperationException("Model has not been initialized. Cannot train.");
        }

        _innerModel.Train(input, expectedOutput);
    }

    /// <summary>
    /// Retrieves metadata and performance metrics about the trained model.
    /// </summary>
    /// <returns>An object containing metadata and performance metrics about the trained model.</returns>
    public ModelMetaData<T> GetModelMetaData()
    {
        if (_innerModel == null)
        {
            throw new InvalidOperationException("Model has not been initialized. Cannot retrieve metadata.");
        }

        return _innerModel.GetModelMetaData();
    }

    #endregion

    #region IModelSerializer Implementation

    /// <summary>
    /// Data Transfer Object for serialization of PredictionModelResult.
    /// </summary>
    private class SerializationDto
    {
        public byte[] Model { get; set; } = Array.Empty<byte>();
        public OptimizationResult<T, TInput, TOutput>? OptimizationResult { get; set; }
        public NormalizationInfo<T, TInput, TOutput>? NormalizationInfo { get; set; }
    }

    /// <summary>
    /// Converts the current state of the model into a binary format.
    /// </summary>
    /// <returns>A byte array containing the serialized model data.</returns>
    public byte[] Serialize()
    {
        if (_innerModel == null)
        {
            throw new InvalidOperationException("Model has not been initialized. Cannot serialize.");
        }

        if (_optimizationResult == null)
        {
            throw new InvalidOperationException("OptimizationResult is missing. Cannot serialize.");
        }

        if (_normalizationInfo == null)
        {
            throw new InvalidOperationException("NormalizationInfo is missing. Cannot serialize.");
        }

        var dto = new SerializationDto
        {
            Model = _innerModel.Serialize(),
            OptimizationResult = _optimizationResult,
            NormalizationInfo = _normalizationInfo
        };

        return JsonSerializer.SerializeToUtf8Bytes(dto);
    }

    /// <summary>
    /// Loads a previously serialized model from binary data.
    /// </summary>
    /// <param name="data">The byte array containing the serialized model data.</param>
    public void Deserialize(byte[] data)
    {
        if (data == null)
        {
            throw new ArgumentNullException(nameof(data));
        }

        var dto = JsonSerializer.Deserialize<SerializationDto>(data);
        if (dto == null)
        {
            throw new InvalidOperationException("Failed to deserialize model data.");
        }

        // Note: This is a simplified implementation. A full implementation would need to:
        // 1. Deserialize the inner model type information
        // 2. Reconstruct the inner model using the appropriate type
        // 3. Properly deserialize OptimizationResult and NormalizationInfo
        throw new NotImplementedException("Full deserialization logic needs to be implemented based on specific model types.");
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
    /// <param name="modelFactory">A factory function that creates the appropriate model type based on metadata.</param>
    /// <returns>A new PredictionModelResult&lt;T&gt; instance loaded from the file.</returns>
    /// <remarks>
    /// <para>
    /// This static method loads a serialized model from a file at the specified path. It requires a model factory function
    /// that can create the appropriate model type based on metadata. This ensures that the correct model type is instantiated
    /// before deserialization.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a previously saved model from a file.
    /// 
    /// The LoadModel method:
    /// - Takes a file path where the model is stored
    /// - Uses the model factory to create the right type of model based on metadata
    /// - Reads the file and deserializes the data into a new PredictionModelResult object
    /// - Returns the fully loaded model ready for making predictions
    /// 
    /// The model factory is important because:
    /// - Different types of models (linear regression, neural networks, etc.) need different deserialization logic
    /// - The factory knows how to create the right type of model based on information in the saved file
    /// 
    /// For example, you might load a model with:
    /// `var model = PredictionModelResult<double, Matrix<double>, Vector<double>>.LoadModel(
    ///     "C:\\Models\\house_price_predictor.model", 
    ///     metadata => new LinearRegressionModel<double>());`
    /// </para>
    /// </remarks>
    public static PredictionModelResult<T, TInput, TOutput> LoadModel(
        string filePath,
        Func<ModelMetadata<T>, IFullModel<T, TInput, TOutput>> modelFactory)
    {
        // First, we need to read the file
        byte[] data = File.ReadAllBytes(filePath);

        // Extract metadata to determine model type
        var metadata = ExtractMetadataFromSerializedData(data);

        // Create a new model instance of the appropriate type
        var model = modelFactory(metadata);

        // Create a new PredictionModelResult with the model
        var result = new PredictionModelResult<T, TInput, TOutput>
        {
            Model = model
        };

        // Deserialize the data
        result.Deserialize(data);

        return result;
    }

    public void Train(TInput input, TOutput expectedOutput)
    {
        throw new NotImplementedException();
    }

    public ModelMetaData<T> GetModelMetaData()
    {
        throw new NotImplementedException();
    }

    public Vector<T> GetParameters()
    {
        throw new NotImplementedException();
    }

    public IFullModel<T, TInput, TOutput> WithParameters(Vector<T> parameters)
    {
        throw new NotImplementedException();
    }

    public IEnumerable<int> GetActiveFeatureIndices()
    {
        throw new NotImplementedException();
    }

    public bool IsFeatureUsed(int featureIndex)
    {
        throw new NotImplementedException();
    }

    public IFullModel<T, TInput, TOutput> DeepCopy()
    {
        throw new NotImplementedException();
    }

    public IFullModel<T, TInput, TOutput> Clone()
    {
        throw new NotImplementedException();
    }
}
    /// <param name="filePath">The path where the model should be saved.</param>
    public void SaveModel(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));
        }

        var data = Serialize();
        File.WriteAllBytes(filePath, data);
    }

    /// <summary>
    /// Loads the model from a file.
    /// </summary>
    /// <param name="filePath">The path to the file containing the saved model.</param>
    public void LoadModel(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));
        }

        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"Model file not found at path: {filePath}", filePath);
        }

        var data = File.ReadAllBytes(filePath);
        Deserialize(data);
    }

    #endregion

    #region IParameterizable Implementation

    /// <summary>
    /// Gets the parameters that can be optimized.
    /// </summary>
    /// <returns>A vector containing the model parameters.</returns>
    public Vector<T> GetParameters()
    {
        if (_innerModel == null)
        {
            throw new InvalidOperationException("Model has not been initialized. Cannot get parameters.");
        }

        return _innerModel.GetParameters();
    }

    /// <summary>
    /// Sets the model parameters.
    /// </summary>
    /// <param name="parameters">The parameter vector to set.</param>
    public void SetParameters(Vector<T> parameters)
    {
        if (_innerModel == null)
        {
            throw new InvalidOperationException("Model has not been initialized. Cannot set parameters.");
        }

        _innerModel.SetParameters(parameters);
    }

    /// <summary>
    /// Gets the number of parameters in the model.
    /// </summary>
    public int ParameterCount
    {
        get
        {
            if (_innerModel == null)
            {
                throw new InvalidOperationException("Model has not been initialized. Cannot get parameter count.");
            }

            return _innerModel.ParameterCount;
        }
    }

    /// <summary>
    /// Creates a new instance with the specified parameters.
    /// </summary>
    /// <param name="parameters">The parameters to use for the new instance.</param>
    /// <returns>A new model instance with the specified parameters.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the model or required metadata is not initialized.</exception>
    public IFullModel<T, TInput, TOutput> WithParameters(Vector<T> parameters)
    {
        if (_innerModel == null)
        {
            throw new InvalidOperationException("Model has not been initialized. Cannot create instance with parameters.");
        }

        if (_optimizationResult == null)
        {
            throw new InvalidOperationException("OptimizationResult is missing. Cannot create instance with parameters.");
        }

        if (_normalizationInfo == null)
        {
            throw new InvalidOperationException("NormalizationInfo is missing. Cannot create instance with parameters.");
        }

        var newInnerModel = _innerModel.WithParameters(parameters);
        var newOptimizationResult = _optimizationResult.WithParameters(parameters);
        var newNormalizationInfo = _normalizationInfo.WithParameters(parameters);
        return new PredictionModelResult<T, TInput, TOutput>(
            newInnerModel,
            newOptimizationResult,
            newNormalizationInfo);
    }

    #endregion

    #region IFeatureAware Implementation

    /// <summary>
    /// Gets the indices of features that are actively used by this model.
    /// </summary>
    /// <returns>An enumerable of feature indices.</returns>
    public IEnumerable<int> GetActiveFeatureIndices()
    {
        if (_innerModel == null)
        {
            throw new InvalidOperationException("Model has not been initialized. Cannot get active feature indices.");
        }

        return _innerModel.GetActiveFeatureIndices();
    }

    /// <summary>
    /// Sets the active feature indices for this model.
    /// </summary>
    /// <param name="featureIndices">The feature indices to set as active.</param>
    public void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
        if (_innerModel == null)
        {
            throw new InvalidOperationException("Model has not been initialized. Cannot set active feature indices.");
        }

        _innerModel.SetActiveFeatureIndices(featureIndices);
    }

    /// <summary>
    /// Checks if a specific feature is used by this model.
    /// </summary>
    /// <param name="featureIndex">The index of the feature to check.</param>
    /// <returns>True if the feature is used; otherwise, false.</returns>
    public bool IsFeatureUsed(int featureIndex)
    {
        if (_innerModel == null)
        {
            return false;
        }

        return _innerModel.IsFeatureUsed(featureIndex);
    }

    #endregion

    #region IFeatureImportance Implementation

    /// <summary>
    /// Gets the feature importance scores.
    /// </summary>
    /// <returns>A dictionary mapping feature names to importance scores.</returns>
    public Dictionary<string, T> GetFeatureImportance()
    {
        if (_innerModel == null)
        {
            throw new InvalidOperationException("Model has not been initialized. Cannot get feature importance.");
        }

        return _innerModel.GetFeatureImportance();
    }

    #endregion

    #region ICloneable Implementation

    /// <summary>
    /// Creates a deep copy of this object.
    /// </summary>
    /// <returns>A deep copy of the prediction model result.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the model or required metadata is not initialized.</exception>
    public IFullModel<T, TInput, TOutput> DeepCopy()
    {
        if (_innerModel == null)
        {
            throw new InvalidOperationException("Model has not been initialized. Cannot create deep copy.");
        }

        if (_optimizationResult == null)
        {
            throw new InvalidOperationException("OptimizationResult is missing. Cannot create deep copy.");
        }

        if (_normalizationInfo == null)
        {
            throw new InvalidOperationException("NormalizationInfo is missing. Cannot create deep copy.");
        }

        var copiedInnerModel = _innerModel.DeepCopy();
        var copiedOptimizationResult = _optimizationResult.DeepCopy();
        var copiedNormalizationInfo = _normalizationInfo.DeepCopy();
        return new PredictionModelResult<T, TInput, TOutput>(
            copiedInnerModel,
            copiedOptimizationResult,
            copiedNormalizationInfo);
    }

    /// <summary>
    /// Creates a shallow copy of this object.
    /// </summary>
    /// <returns>A shallow copy of the prediction model result.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the model or required metadata is not initialized.</exception>
    public IFullModel<T, TInput, TOutput> Clone()
    {
        if (_innerModel == null)
        {
            throw new InvalidOperationException("Model has not been initialized. Cannot create clone.");
        }

        if (_optimizationResult == null)
        {
            throw new InvalidOperationException("OptimizationResult is missing. Cannot create clone.");
        }

        if (_normalizationInfo == null)
        {
            throw new InvalidOperationException("NormalizationInfo is missing. Cannot create clone.");
        }

        var clonedInnerModel = _innerModel.Clone();
        return new PredictionModelResult<T, TInput, TOutput>(
            clonedInnerModel,
            _optimizationResult,
            _normalizationInfo);
    }

    #endregion
}
