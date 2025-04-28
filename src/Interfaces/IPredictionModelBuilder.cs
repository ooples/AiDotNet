namespace AiDotNet.Interfaces;

/// <summary>
/// Defines a builder pattern interface for creating and configuring predictive models.
/// </summary>
/// <remarks>
/// This interface provides a fluent API for setting up all components of a machine learning model.
/// 
/// <b>For Beginners:</b> Think of this as a step-by-step recipe builder for creating AI models.
/// Just like building a custom sandwich where you choose the bread, fillings, and condiments,
/// this builder lets you choose different components for your AI model.
/// 
/// The builder pattern makes it easy to:
/// - Configure your model piece by piece
/// - Change only the parts you want while keeping default settings for the rest
/// - Create different variations of models without writing repetitive code
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
public interface IPredictionModelBuilder<T, TInput, TOutput>
{
    /// <summary>
    /// Configures the feature selector component for the model.
    /// </summary>
    /// <remarks>
    /// A feature selector helps identify which input variables (features) are most important
    /// for making predictions.
    /// 
    /// <b>For Beginners:</b> Imagine you're trying to predict house prices. You have many possible 
    /// factors: size, location, age, number of rooms, etc. A feature selector helps figure out 
    /// which of these factors actually matter for making good predictions. This can improve 
    /// your model's accuracy and make it run faster by focusing only on what's important.
    /// </remarks>
    /// <param name="selector">The feature selector implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureFeatureSelector(IFeatureSelector<T, TInput> selector);

    /// <summary>
    /// Configures the data normalizer component for the model.
    /// </summary>
    /// <remarks>
    /// A normalizer transforms data to a standard scale, which helps many machine learning
    /// algorithms perform better.
    /// 
    /// <b>For Beginners:</b> Different features in your data might use different scales. For example, 
    /// a person's age (0-100) and income (thousands or millions) are on very different scales. 
    /// Normalization converts all features to a similar scale (like 0-1), which prevents features 
    /// with larger numbers from dominating the learning process just because they have bigger values.
    /// </remarks>
    /// <param name="normalizer">The normalizer implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureNormalizer(INormalizer<T, TInput, TOutput> normalizer);

    /// <summary>
    /// Configures the optimization algorithm for the model.
    /// </summary>
    /// <remarks>
    /// An optimizer determines how the model's parameters are updated during training.
    /// 
    /// <b>For Beginners:</b> The optimizer is like the "learning strategy" for your model. It decides:
    /// - How quickly the model should learn (learning rate)
    /// - How to adjust the model's parameters to improve predictions
    /// - When to stop trying to improve further
    /// 
    /// Common optimizers include Gradient Descent, Adam, and L-BFGS, each with different 
    /// strengths and weaknesses.
    /// </remarks>
    /// <param name="optimizationAlgorithm">The optimization algorithm implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureOptimizer(IOptimizer<T, TInput, TOutput> optimizationAlgorithm);

    /// <summary>
    /// Configures the data preprocessing component for the model.
    /// </summary>
    /// <remarks>
    /// A data preprocessor cleans and transforms raw data before it's used for training.
    /// 
    /// <b>For Beginners:</b> Data preprocessing is like preparing ingredients before cooking. 
    /// It involves:
    /// - Cleaning data (removing or fixing errors)
    /// - Transforming data (converting text to numbers, etc.)
    /// - Organizing data (putting it in the right format)
    /// 
    /// Good preprocessing can dramatically improve your model's performance by ensuring 
    /// it learns from high-quality data.
    /// </remarks>
    /// <param name="dataPreprocessor">The data preprocessor implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureDataPreprocessor(IDataPreprocessor<T, TInput, TOutput> dataPreprocessor);

    /// <summary>
    /// Configures the outlier removal component for the model.
    /// </summary>
    /// <remarks>
    /// An outlier removal component identifies and handles unusual data points that might
    /// negatively impact the model's performance.
    /// 
    /// <b>For Beginners:</b> Outliers are unusual data points that don't follow the general pattern. 
    /// For example, if you're analyzing house prices and most houses cost $100,000-$500,000, 
    /// a $10 million mansion would be an outlier. These unusual points can confuse your model 
    /// and make it perform worse. Outlier removal helps identify and handle these unusual cases.
    /// </remarks>
    /// <param name="outlierRemoval">The outlier removal implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    IPredictionModelBuilder<T, TInput, TOutput> ConfigureOutlierRemoval(IOutlierRemoval<T, TInput, TOutput> outlierRemoval);

    /// <summary>
    /// Builds a predictive model using the configured components and training data.
    /// </summary>
    /// <remarks>
    /// This method takes the input features and target values and creates a trained model
    /// ready to make predictions.
    /// 
    /// <b>For Beginners:</b> After configuring all the components of your model, this method actually 
    /// creates and trains the model using your data. It's like pressing "Start" after setting up 
    /// all your preferences. The model will learn patterns from your training data so it can make 
    /// predictions later.
    /// </remarks>
    /// <param name="x">The input features matrix, where each row is a data point and each column is a feature.</param>
    /// <param name="y">The target values vector that the model will learn to predict.</param>
    /// <returns>A trained predictive model ready to make predictions.</returns>
    IPredictiveModel<T, TInput, TOutput> Build(TInput x, TOutput y);

    /// <summary>
    /// Uses a trained model to make predictions on new data.
    /// </summary>
    /// <remarks>
    /// This method applies a previously trained model to new data to generate predictions.
    /// 
    /// <b>For Beginners:</b> Once your model is trained, you can use it to make predictions on new data 
    /// it hasn't seen before. For example, if you trained a model to predict house prices based on 
    /// features like size and location, you can now give it information about new houses and it will 
    /// estimate their prices.
    /// </remarks>
    /// <param name="newData">The new input data to make predictions for.</param>
    /// <param name="model">The trained model to use for making predictions.</param>
    /// <returns>A vector of predicted values.</returns>
    TOutput Predict(TInput newData, IPredictiveModel<T, TInput, TOutput> model);

    /// <summary>
    /// Saves a trained model to a file.
    /// </summary>
    /// <remarks>
    /// This method persists a model to disk so it can be reused later without retraining.
    /// 
    /// <b>For Beginners:</b> Training a model can take a lot of time and computing power. This method 
    /// lets you save your trained model to a file on your computer, so you can use it again later 
    /// without having to retrain it. It's like saving a document you've been working on.
    /// </remarks>
    /// <param name="model">The trained model to save.</param>
    /// <param name="filePath">The file path where the model should be saved.</param>
    void SaveModel(IPredictiveModel<T, TInput, TOutput> model, string filePath);

    /// <summary>
    /// Loads a previously saved model from a file.
    /// </summary>
    /// <remarks>
    /// This method retrieves a model that was previously saved to disk.
    /// 
    /// <b>For Beginners:</b> This method lets you load a previously saved model from a file. It's like 
    /// opening a document you worked on earlier. Once loaded, you can immediately use the model 
    /// to make predictions without having to train it again.
    /// </remarks>
    /// <param name="filePath">The file path where the model is stored.</param>
    /// <returns>The loaded predictive model.</returns>
    IPredictiveModel<T, TInput, TOutput> LoadModel(string filePath);

    /// <summary>
    /// Converts a trained model into a byte array for storage or transmission.
    /// </summary>
    /// <remarks>
    /// This method transforms a model into a compact binary format that can be stored in memory,
    /// databases, or transmitted over networks.
    /// 
    /// <b>For Beginners:</b> Serialization is like packing your model into a compact digital suitcase.
    /// Instead of saving to a file (like with SaveModel), this method converts your model into
    /// a series of bytes that can be:
    /// - Stored in a database
    /// - Sent over the internet
    /// - Kept in computer memory
    /// - Embedded in other applications
    /// 
    /// This is useful when you need to store models in places other than files or when you
    /// want to send models between different parts of your application.
    /// </remarks>
    /// <param name="model">The trained model to serialize.</param>
    /// <returns>A byte array containing the serialized model data.</returns>
    byte[] SerializeModel(IPredictiveModel<T, TInput, TOutput> model);

    /// <summary>
    /// Reconstructs a model from a previously serialized byte array.
    /// </summary>
    /// <remarks>
    /// This method converts a byte array back into a usable model object.
    /// 
    /// <b>For Beginners:</b> Deserialization is like unpacking your model from the digital suitcase
    /// created by SerializeModel. It takes the compact byte format and rebuilds your complete
    /// model so you can use it for making predictions again.
    /// 
    /// This is the counterpart to SerializeModel - first you serialize to create the byte array,
    /// then you deserialize to recreate the model when needed.
    /// 
    /// For example, if you stored your model in a database or received it over a network,
    /// you would use this method to convert it back into a working model.
    /// </remarks>
    /// <param name="modelData">The byte array containing the serialized model data.</param>
    /// <returns>The reconstructed predictive model.</returns>
    IPredictiveModel<T, TInput, TOutput> DeserializeModel(byte[] modelData);
}