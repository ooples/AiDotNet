using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet;

/// <content>
/// Training configuration on the streaming supervised path: options the streaming loop honours are pushed into the
/// optimizer here, and options only the in-memory supervised path can honour are refused instead of being dropped.
/// </content>
public partial class AiModelBuilder<T, TInput, TOutput>
{
    /// <summary>
    /// Refuses training options that only the in-memory supervised path consumes.
    /// </summary>
    /// <remarks>
    /// Each of these replaces or reshapes the in-memory training run (a search over trainings, a different trainer,
    /// a second training pass, or a rewrite of the materialised training set). The streaming loop trains batch by
    /// batch from the loader and has no place where they apply, so a build that carried on would return a model
    /// trained without them and give no sign of it.
    /// </remarks>
    /// <exception cref="NotSupportedException">One of the options is configured.</exception>
    private void RejectTrainingConfigurationUnsupportedOnStreaming()
    {
        string? dropped = null;
        if (_hyperparameterOptimizer is not null && _hyperparameterSearchSpace is not null)
        {
            dropped = nameof(ConfigureHyperparameterOptimizer);
        }
        else if (_federatedLearningOptions is not null)
        {
            dropped = nameof(ConfigureFederatedLearning);
        }
        else if (_knowledgeDistillationOptions is not null)
        {
            dropped = nameof(ConfigureKnowledgeDistillation);
        }
        else if (_configuredDistillationStrategy is not null)
        {
            dropped = nameof(ConfigureDistillationStrategy);
        }
        else if (_continualLearningEnabled)
        {
            dropped = nameof(ConfigureContinualLearning);
        }
        else if (_curriculumLearningOptions is { Dataset: not null })
        {
            dropped = nameof(ConfigureCurriculumLearning);
        }
        else if (_fineTuningConfiguration is { Enabled: true })
        {
            dropped = nameof(ConfigureFineTuning);
        }
        else if (_trainingPipelineConfiguration?.Stages is { Count: > 0 })
        {
            dropped = nameof(ConfigureTrainingPipeline);
        }
        else if (_selfSupervisedLearningPretrainAction is not null || _selfSupervisedLearningConfig?.Method is not null)
        {
            dropped = nameof(ConfigureSelfSupervisedLearning);
        }
        else if (_loraConfiguration is not null && _model is NeuralNetworks.NeuralNetworkBase<T>)
        {
            // The in-memory path swaps the adapters into the network before training; here the network would be
            // trained in full while the result still carried the LoRA configuration.
            dropped = nameof(ConfigureLoRA);
        }
        else if (_trainingGroups is not null)
        {
            dropped = nameof(ConfigureTrainingGroups);
        }
        else if (_dataPreparationPipeline is { Count: > 0 })
        {
            dropped = nameof(ConfigureDataPreparation);
        }
        else if (_crossValidator is not null)
        {
            dropped = nameof(ConfigureCrossValidation);
        }
        else if (_configuredDataSplitter is not null)
        {
            dropped = nameof(ConfigureDataSplitter);
        }

        if (dropped is not null)
        {
            throw new NotSupportedException(
                $"{dropped} is not supported on the streaming data-loader path: it is applied by the in-memory " +
                "supervised path, which needs the whole training set materialised, and the streaming loop would " +
                "train without it. Use an IInputOutputDataLoader (for example InMemoryDataLoader), or remove the " +
                $"{dropped} call.");
        }
    }

    /// <summary>
    /// Pushes <see cref="ConfigureRegularization"/> into the optimizer the streaming loop steps with, as the
    /// in-memory path does.
    /// </summary>
    /// <remarks>
    /// A neural network on the streaming path steps through the configured optimizer, whose regularization the tape
    /// step applies to the gradients. Any other model updates itself through
    /// <see cref="IGradientComputable{T, TInput, TOutput}"/> without an optimizer, so there is nothing that could
    /// apply the penalty and the configuration is refused.
    /// </remarks>
    /// <exception cref="NotSupportedException">The model does not train through an optimizer on this path.</exception>
    /// <exception cref="InvalidOperationException">No gradient-based optimizer is configured.</exception>
    private void ApplyStreamingRegularization()
    {
        if (_regularization is null)
        {
            return;
        }

        if (_model is not NeuralNetworks.NeuralNetworkBase<T>)
        {
            throw new NotSupportedException(
                $"{nameof(ConfigureRegularization)} is not supported on the streaming data-loader path for " +
                $"{_model?.GetType().Name ?? "a null model"}: only neural networks step through an optimizer there, " +
                "and the regularization is applied by the optimizer. Use an IInputOutputDataLoader, or remove the " +
                $"{nameof(ConfigureRegularization)} call.");
        }

        // The loop hands the network the configured optimizer only when it is a tensor optimizer; any other
        // optimizer would take the penalty and never step.
        if (_optimizer is Optimizers.GradientBasedOptimizerBase<T, TInput, TOutput> gradientOptimizer
            && _optimizer is IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>)
        {
            gradientOptimizer.SetRegularization(_regularization);
            return;
        }

        throw new InvalidOperationException(
            $"{nameof(ConfigureRegularization)} on the streaming data-loader path needs a gradient-based optimizer " +
            $"configured through {nameof(ConfigureOptimizer)} (AdamOptimizer, AdamWOptimizer and the like); the " +
            $"active optimizer is {_optimizer?.GetType().Name ?? "not configured"}, which has no regularization slot.");
    }
}
