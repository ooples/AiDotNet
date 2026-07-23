namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for models that support saving and loading their internal state (checkpointing).
/// </summary>
/// <remarks>
/// <para>
/// This interface enables models to save their trained parameters and internal state to persistent storage
/// and restore them later, which is essential for model persistence, training interruption/resumption,
/// and distributed training scenarios.
/// </para>
/// <para>
/// <b>For Beginners:</b> This interface is like a "save game" feature for machine learning models.
///
/// Just as video games let you save your progress and load it later:
/// - SaveState: Writes the model's current state (all learned parameters) to a file
/// - LoadState: Reads a previously saved state back into the model
///
/// This is useful for:
/// - Saving the best model during training (so you can use it later)
/// - Resuming training if it gets interrupted
/// - Sharing trained models with others
/// - Deploying models to production systems
/// </para>
/// <para>
/// <b>Design Note:</b> This is a separate interface from IFullModel because not all models
/// support checkpointing (e.g., some stateless models or models that can't serialize their state).
/// By making it optional, we keep the type system honest and allow models to opt-in to checkpointing.
/// </para>
/// </remarks>
public interface ICheckpointableModel
{
    /// <summary>
    /// Saves the model's current state (parameters and configuration) to a stream.
    /// </summary>
    /// <param name="stream">The stream to write the model state to.</param>
    /// <exception cref="ArgumentNullException">Thrown when stream is null.</exception>
    /// <exception cref="ArgumentException">Thrown when stream is not writable.</exception>
    /// <exception cref="InvalidOperationException">Thrown when model state cannot be serialized (e.g., uninitialized model).</exception>
    /// <remarks>
    /// <para>
    /// This method serializes all the information needed to recreate the model's current state,
    /// including trained parameters, layer configurations, and any internal state variables.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is like creating a snapshot of your trained model.
    ///
    /// When you call SaveState:
    /// - All the learned parameters (weights and biases) are written to the stream
    /// - The model's architecture information is saved
    /// - Any other internal state (like normalization statistics) is preserved
    ///
    /// You can later use LoadState to restore the model to this exact state.
    /// </para>
    /// <para>
    /// <b>Stream Handling:</b>
    /// - The stream position will be advanced by the number of bytes written
    /// - The stream is flushed but not closed (caller must dispose)
    /// - For file-based persistence, wrap in File.Create/FileStream
    /// </para>
    /// <para>
    /// <b>Usage:</b>
    /// <code>
    /// // Save to file
    /// using var stream = File.Create("model.bin");
    /// model.SaveState(stream);
    /// </code>
    /// </para>
    /// </remarks>
    void SaveState(Stream stream);

    /// <summary>
    /// Loads the model's state (parameters and configuration) from a stream.
    /// </summary>
    /// <param name="stream">The stream to read the model state from.</param>
    /// <exception cref="ArgumentNullException">Thrown when stream is null.</exception>
    /// <exception cref="ArgumentException">Thrown when stream is not readable or contains invalid data.</exception>
    /// <exception cref="InvalidOperationException">Thrown when deserialization fails or data is incompatible with model architecture.</exception>
    /// <remarks>
    /// <para>
    /// This method deserializes model state that was previously saved with SaveState,
    /// restoring all parameters and configuration to recreate the saved model state.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is like loading a saved game.
    ///
    /// When you call LoadState:
    /// - All the parameters are read from the stream
    /// - The model is configured to match the saved architecture
    /// - The model becomes identical to when SaveState was called
    ///
    /// After loading, the model can make predictions using the restored parameters.
    /// </para>
    /// <para>
    /// <b>Stream Handling:</b>
    /// - The stream position will be advanced by the number of bytes read
    /// - The stream is not closed (caller must dispose)
    /// - Stream data must match the format written by SaveState
    /// </para>
    /// <para>
    /// <b>Versioning:</b>
    /// Implementations should consider:
    /// - Including format version number in serialized data
    /// - Validating compatibility before deserialization
    /// - Providing migration paths for old formats when possible
    /// </para>
    /// <para>
    /// <b>Usage:</b>
    /// <code>
    /// // Load from file
    /// using var stream = File.OpenRead("model.bin");
    /// model.LoadState(stream);
    /// </code>
    /// </para>
    /// <para>
    /// <b>Important:</b> The stream must contain state data saved by SaveState from a
    /// compatible model (same architecture and numeric type).
    /// </para>
    /// </remarks>
    void LoadState(Stream stream);
}
