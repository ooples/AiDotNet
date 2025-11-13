namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the core functionality for machine learning models that can be trained on data and make predictions.
/// </summary>
/// <remarks>
/// This interface represents the fundamental operations that all machine learning models should support:
/// training on data, making predictions with new data, and providing metadata about the model's performance.
/// 
/// <b>For Beginners:</b> A machine learning model is like a recipe that learns from examples.
/// 
/// Think of a model as a student learning to recognize patterns:
/// - First, you train the model by showing it examples (training data)
/// - Then, the model learns patterns from these examples
/// - Finally, when given new information, the model uses what it learned to make predictions
/// 
/// For example, if you want to predict house prices:
/// - You train the model with data about houses (size, location, etc.) and their prices
/// - The model learns the relationship between house features and prices
/// - When given information about a new house, it predicts what the price might be
/// 
/// This interface provides the essential methods needed for this learning and prediction process.
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
public interface IModel<TInput, TOutput, TMetadata>
{
    /// <summary>
    /// Trains the model using input features and their corresponding target values.
    /// </summary>
    /// <remarks>
    /// This method takes training data and adjusts the model's internal parameters to learn patterns in the data.
    /// 
    /// <b>For Beginners:</b> Training is like teaching the model by showing it examples.
    /// 
    /// Imagine teaching a child to identify fruits:
    /// - You show them many examples of apples, oranges, and bananas (input features x)
    /// - You tell them the correct name for each fruit (target values y)
    /// - Over time, they learn to recognize the patterns that distinguish each fruit
    /// 
    /// In machine learning:
    /// - The x parameter contains features (characteristics) of your data
    /// - The y parameter contains the correct answers you want the model to learn
    /// - During training, the model adjusts its internal calculations to get better at predicting y from x
    /// 
    /// For example, in a house price prediction model:
    /// - x would contain features like square footage, number of bedrooms, location
    /// - y would contain the actual sale prices of those houses
    /// </remarks>
    /// <param name="x">A matrix where each row represents a training example and each column represents a feature.</param>
    /// <param name="y">A vector containing the target values corresponding to each training example.</param>
    void Train(TInput input, TOutput expectedOutput);

    /// <summary>
    /// Uses the trained model to make predictions for new input data.
    /// </summary>
    /// <remarks>
    /// After training, this method applies the learned patterns to new data to predict outcomes.
    /// 
    /// <b>For Beginners:</b> Prediction is when the model uses what it learned to make educated guesses about new information.
    /// 
    /// Continuing the fruit identification example:
    /// - After learning from many examples, the child (model) can now identify new fruits they haven't seen before
    /// - They look at the color, shape, and size to make their best guess
    /// 
    /// In machine learning:
    /// - You give the model new data it hasn't seen during training
    /// - The model applies the patterns it learned to make predictions
    /// - The output is the model's best estimate based on its training
    /// 
    /// For example, in a house price prediction model:
    /// - You provide features of a new house (square footage, bedrooms, location)
    /// - The model predicts what price that house might sell for
    /// 
    /// This method is used after training is complete, when you want to apply your model to real-world data.
    /// </remarks>
    /// <param name="input">A matrix where each row represents a new example to predict and each column represents a feature.</param>
    /// <returns>A vector containing the predicted values for each input example.</returns>
    TOutput Predict(TInput input);

    /// <summary>
    /// Retrieves metadata and performance metrics about the trained model.
    /// </summary>
    /// <remarks>
    /// This method provides information about the model's structure, parameters, and performance metrics.
    /// 
    /// <b>For Beginners:</b> Model metadata is like a report card for your machine learning model.
    /// 
    /// Just as a report card shows how well a student is performing in different subjects,
    /// model metadata shows how well your model is performing and provides details about its structure.
    /// 
    /// This information typically includes:
    /// - Accuracy measures: How well does the model's predictions match actual values?
    /// - Error metrics: How far off are the model's predictions on average?
    /// - Model parameters: What patterns did the model learn from the data?
    /// - Training information: How long did training take? How many iterations were needed?
    /// 
    /// For example, in a house price prediction model, metadata might include:
    /// - Average prediction error (e.g., off by $15,000 on average)
    /// - How strongly each feature (bedrooms, location) influences the prediction
    /// - How well the model fits the training data
    /// 
    /// This information helps you understand your model's strengths and weaknesses,
    /// and decide if it's ready to use or needs more training.
    /// </remarks>
    /// <returns>An object containing metadata and performance metrics about the trained model.</returns>
    TMetadata GetModelMetadata();
}
