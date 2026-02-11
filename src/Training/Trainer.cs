using AiDotNet.LinearAlgebra;
using AiDotNet.Training.Configuration;

namespace AiDotNet.Training;

/// <summary>
/// Default trainer that delegates to the model's built-in <c>Train()</c> method each epoch.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This is the standard trainer for models that know how to train
/// themselves (e.g., time series models like ARIMA and ExponentialSmoothing). Each epoch
/// it calls <c>model.Train(features, labels)</c>, then measures how well the model predicts
/// by computing the loss.
/// </para>
/// <para>
/// <b>Example usage from YAML:</b>
/// <code>
/// var trainer = new Trainer&lt;double&gt;("config/my-experiment.yaml");
/// var result = trainer.Run();
/// // result.TrainedModel is ready for predictions
/// // result.EpochLosses shows the training progress
/// </code>
/// </para>
/// <para>
/// <b>Example usage with in-memory data:</b>
/// <code>
/// var config = new TrainingRecipeConfig { ... };
/// var trainer = new Trainer&lt;double&gt;(config);
/// trainer.SetData(features, labels);
/// var result = trainer.Run();
/// </code>
/// </para>
/// <para>
/// <b>Custom logging:</b>
/// <code>
/// var trainer = new Trainer&lt;double&gt;(config);
/// trainer.LogAction = message => myLogger.Info(message);
/// </code>
/// </para>
/// </remarks>
internal class Trainer<T> : TrainerBase<T>
{
    /// <summary>
    /// Creates a trainer from a YAML configuration file.
    /// </summary>
    /// <param name="yamlFilePath">Path to the YAML training recipe file.</param>
    /// <exception cref="ArgumentException">Thrown when the file path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown when the YAML file does not exist.</exception>
    public Trainer(string yamlFilePath)
        : base(yamlFilePath)
    {
    }

    /// <summary>
    /// Creates a trainer from a <see cref="TrainingRecipeConfig"/> object.
    /// </summary>
    /// <param name="config">The training recipe configuration.</param>
    /// <exception cref="ArgumentNullException">Thrown when config is null.</exception>
    /// <exception cref="ArgumentException">Thrown when required config sections are missing.</exception>
    public Trainer(TrainingRecipeConfig config)
        : base(config)
    {
    }

    /// <summary>
    /// Trains the model for one epoch by calling its built-in Train method,
    /// then computes and returns the loss.
    /// </summary>
    /// <param name="features">The input feature matrix.</param>
    /// <param name="labels">The output label vector.</param>
    /// <param name="epoch">The zero-based epoch index.</param>
    /// <returns>The loss value after training this epoch.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Models like ARIMA and ExponentialSmoothing fit their parameters
    /// internally when you call <c>Train()</c>. This trainer simply delegates to that method,
    /// then measures how good the predictions are by computing the loss function.
    /// </para>
    /// </remarks>
    protected override T TrainEpoch(Matrix<T> features, Vector<T> labels, int epoch)
    {
        // Train the model on the data
        Model.Train(features, labels);

        // Compute loss
        var predictions = Model.Predict(features);
        return LossFunction.CalculateLoss(predictions, labels);
    }
}
