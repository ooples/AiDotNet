namespace AiDotNet.Interfaces;

/// <summary>
/// A model that its paper trains on typed samples carrying more supervision than an input/target pair, so that
/// <see cref="AiModelBuilder{T, TInput, TOutput}"/> can drive its training from those samples.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TSample">The training sample the model's objective is defined on, such as
/// <see cref="TextToSpeech.TtsTrainingSample{T}"/>.</typeparam>
/// <remarks>
/// <para>
/// A text-to-speech model trained on forced-alignment durations, codec tokens or a speaker reference, or a music
/// model trained on a clip and its description, cannot be trained from a tensor pair: the supervision is not in it.
/// Such a model exposes one training step and one side-effect-free evaluation of the same objective on its own
/// sample type, and the builder's <c>Build(samples)</c> / <c>BuildAsync(samples)</c> run the epochs, the
/// held-out evaluation, early stopping and the configured training callbacks around them.
/// </para>
/// <para><b>For Beginners:</b> You do not call these members yourself. Hand your samples to
/// <c>new AiModelBuilder&lt;...&gt;().ConfigureModel(model).Build(samples)</c> and the builder trains the model the
/// way its paper does.</para>
/// </remarks>
public interface ISampleObjectiveModel<T, in TSample>
{
    /// <summary>Runs one training step of the model's own objective on <paramref name="sample"/>.</summary>
    /// <param name="sample">One training sample.</param>
    /// <returns>The training loss of the step.</returns>
    T Train(TSample sample);

    /// <summary>
    /// Evaluates the model's training objective on <paramref name="sample"/> without updating the model.
    /// </summary>
    /// <param name="sample">One sample, typically held out from training.</param>
    /// <returns>The objective's value; lower is better.</returns>
    T EvaluateTrainingObjective(TSample sample);
}
