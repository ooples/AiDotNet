using AiDotNet.Agentic.Models.Local;
using AiDotNet.FineTuning;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;

namespace AiDotNet.Agentic.SelfImproving;

/// <summary>
/// Runs the online self-improvement loop's final step: fine-tune (e.g. LoRA-adapt) a model on a reward-filtered
/// <see cref="FineTuningDataset"/> by converting it to supervised data and invoking a fine-tuner. This closes
/// the loop capture → evaluate → reward-filter → <em>train</em>.
/// </summary>
/// <remarks>
/// <para>
/// The fine-tuner is any <see cref="FineTuningBase{T, TInput, TOutput}"/> over string in/out (notably
/// <c>SupervisedFineTuning</c>) and the model any <see cref="IFullModel{T, TInput, TOutput}"/> that consumes
/// text. The actual training (LoRA via the configured <see cref="FineTuningOptions{T}"/>) and its hardware cost
/// live in the fine-tuner; this is the thin, deterministic bridge from the agentic dataset to that call.
/// </para>
/// <para><b>For Beginners:</b> Hand it your "good runs" dataset, a fine-tuner, and a model, and it trains the
/// model to do more of what worked.
/// </para>
/// </remarks>
public static class LoRAFineTuner
{
    /// <summary>
    /// Fine-tunes a model on a reward-filtered dataset.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="fineTuner">The fine-tuner (e.g. SupervisedFineTuning) to run.</param>
    /// <param name="model">The model to fine-tune.</param>
    /// <param name="dataset">The reward-filtered dataset.</param>
    /// <param name="cancellationToken">Token used to cancel training.</param>
    /// <returns>The fine-tuned model.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="fineTuner"/>, <paramref name="model"/>, or <paramref name="dataset"/> is <c>null</c>.</exception>
    public static Task<IFullModel<T, string, string>> FineTuneFromDatasetAsync<T>(
        FineTuningBase<T, string, string> fineTuner,
        IFullModel<T, string, string> model,
        FineTuningDataset dataset,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNull(fineTuner);
        Guard.NotNull(model);
        Guard.NotNull(dataset);

        var trainingData = FineTuningDataConverter.ToSupervisedData<T>(dataset);
        return fineTuner.FineTuneAsync(model, trainingData, cancellationToken);
    }

    /// <summary>
    /// Trains a <em>tensor</em> model end-to-end on a reward-filtered dataset: the dataset text is tokenized
    /// into next-token tensor supervision (via <see cref="TextTensorDatasetConverter"/>) and the model is
    /// trained on it for the given number of epochs using the network's own supervised training path
    /// (<see cref="NeuralNetworkBase{T}.Train"/>). This is the runnable path for AiDotNet's own language models
    /// — the string overload suits text-native models, but a tensor model needs tokenization in front, which
    /// this provides, and it drives the verified per-example training loop rather than the generic
    /// gradient-vector path.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="model">The tensor language model to train (updated in place).</param>
    /// <param name="tokenizer">The tokenizer used to encode the dataset text.</param>
    /// <param name="vocabSize">The model's vocabulary size (one-hot width).</param>
    /// <param name="sequenceLength">The fixed token window length per example.</param>
    /// <param name="dataset">The reward-filtered dataset.</param>
    /// <param name="epochs">The number of passes over the dataset.</param>
    /// <param name="cancellationToken">Token used to cancel training.</param>
    /// <returns>The trained model (the same instance).</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="model"/>, <paramref name="tokenizer"/>, or <paramref name="dataset"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="epochs"/> is not positive.</exception>
    public static NeuralNetworkBase<T> TrainTensorModelOnDataset<T>(
        NeuralNetworkBase<T> model,
        IGenerationTokenizer tokenizer,
        int vocabSize,
        int sequenceLength,
        FineTuningDataset dataset,
        int epochs,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNull(model);
        Guard.NotNull(tokenizer);
        Guard.NotNull(dataset);
        Guard.Positive(epochs);

        var data = TextTensorDatasetConverter.ToTensorData<T>(dataset, tokenizer, vocabSize, sequenceLength);
        for (var epoch = 0; epoch < epochs; epoch++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            for (var i = 0; i < data.Count; i++)
            {
                model.Train(data.Inputs[i], data.Outputs[i]);
            }
        }

        return model;
    }
}
