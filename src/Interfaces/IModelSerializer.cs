namespace AiDotNet.Interfaces;

/// <summary>
/// Defines methods for converting machine learning models to and from binary data for storage or transmission.
/// </summary>
/// <remarks>
/// This interface provides functionality to save trained models to binary format and load them back.
/// Serialization allows models to be stored to disk, transmitted over networks, or embedded in applications.
/// 
/// <b>For Beginners:</b> Think of serialization like taking a snapshot of your model.
/// 
/// Imagine you've spent hours training a machine learning model:
/// - Serialization is like taking a photo of your model's current state
/// - This "photo" (binary data) can be saved to your computer
/// - Later, you can use this "photo" to recreate the exact same model
/// - You don't need to train the model again - it's ready to use immediately
/// 
/// Real-world examples:
/// - Saving a trained model to use it in a mobile app
/// - Sharing your trained model with colleagues
/// - Backing up your model before experimenting with changes
/// - Deploying your model to a production environment
/// 
/// Without serialization, you would need to retrain your model from scratch every time you restart
/// your application, which could take hours or days depending on the complexity of your model and data.
/// </remarks>
public interface IModelSerializer
{
    /// <summary>
    /// Converts the current state of a machine learning model into a binary format.
    /// </summary>
    /// <remarks>
    /// This method captures all the essential information about a trained model and converts it
    /// into a sequence of bytes that can be stored or transmitted.
    /// 
    /// <b>For Beginners:</b> This is like exporting your work to a file.
    /// 
    /// When you call this method:
    /// - The model's current state (all its learned patterns and parameters) is captured
    /// - This information is converted into a compact binary format (bytes)
    /// - You can then save these bytes to a file, database, or send them over a network
    /// 
    /// For example:
    /// - After training a model to recognize cats vs. dogs in images
    /// - You can serialize the model to save all its learned knowledge
    /// - Later, you can use this saved data to recreate the model exactly as it was
    /// - The recreated model will make the same predictions as the original
    /// 
    /// Think of it like taking a snapshot of your model's brain at a specific moment in time.
    /// </remarks>
    /// <returns>A byte array containing the serialized model data.</returns>
    byte[] Serialize();

    /// <summary>
    /// Loads a previously serialized model from binary data.
    /// </summary>
    /// <remarks>
    /// This method takes binary data created by the Serialize method and uses it to
    /// restore a model to its previous state.
    /// 
    /// <b>For Beginners:</b> This is like opening a saved file to continue your work.
    /// 
    /// When you call this method:
    /// - You provide the binary data (bytes) that was previously created by Serialize
    /// - The model rebuilds itself using this data
    /// - After deserializing, the model is exactly as it was when serialized
    /// - It's ready to make predictions without needing to be trained again
    /// 
    /// For example:
    /// - You download a pre-trained model file for detecting spam emails
    /// - You deserialize this file into your application
    /// - Immediately, your application can detect spam without any training
    /// - The model has all the knowledge that was built into it by its original creator
    /// 
    /// This is particularly useful when:
    /// - You want to use a model that took days to train
    /// - You need to deploy the same model across multiple devices
    /// - You're creating an application that non-technical users will use
    /// 
    /// Think of it like installing the brain of a trained expert directly into your application.
    /// </remarks>
    /// <param name="data">The byte array containing the serialized model data.</param>
    void Deserialize(byte[] data);

    /// <summary>
    /// Saves the model to a file.
    /// </summary>
    /// <param name="filePath">The path where the model should be saved.</param>
    /// <remarks>
    /// This method provides a convenient way to save the model directly to disk.
    /// It combines serialization with file I/O operations.
    ///
    /// <b>For Beginners:</b> This is like clicking "Save As" in a document editor.
    /// Instead of manually calling Serialize() and then writing to a file, this method does both steps for you.
    /// </remarks>
    /// <exception cref="IOException">
    /// Thrown when an I/O error occurs while writing to the file.
    /// </exception>
    /// <exception cref="UnauthorizedAccessException">
    /// Thrown when the caller does not have the required permission to write to the specified file path.
    /// </exception>
    void SaveModel(string filePath);

    /// <summary>
    /// Loads the model from a file.
    /// </summary>
    /// <param name="filePath">The path to the file containing the saved model.</param>
    /// <remarks>
    /// This method provides a convenient way to load a model directly from disk.
    /// It combines file I/O operations with deserialization.
    ///
    /// <b>For Beginners:</b> This is like clicking "Open" in a document editor.
    /// Instead of manually reading from a file and then calling Deserialize(), this method does both steps for you.
    /// </remarks>
    /// <exception cref="FileNotFoundException">
    /// Thrown when the specified file does not exist.
    /// </exception>
    /// <exception cref="IOException">
    /// Thrown when an I/O error occurs while reading from the file or when the file contains corrupted or invalid model data.
    /// </exception>
    void LoadModel(string filePath);
}
