using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet;

/// <content>
/// Training configuration on the streaming supervised path: options the streaming loop honours are wired in here,
/// and the options that cannot work on a stream are refused, each for its own reason, instead of being dropped.
/// </content>
public partial class AiModelBuilder<T, TInput, TOutput>
{
    /// <summary>
    /// Refuses the training options that need the whole training set at once, which a streaming loader never
    /// holds. Every configured one is named in the same error.
    /// </summary>
    /// <remarks>
    /// LoRA, knowledge distillation, fine-tuning, the training pipeline, curriculum learning and regularization
    /// are not in this list: the streaming path applies them.
    /// </remarks>
    /// <exception cref="NotSupportedException">One or more of the options is configured.</exception>
    private void RejectTrainingConfigurationUnsupportedOnStreaming()
    {
        var refused = new List<string>();
        if (_hyperparameterOptimizer is not null && _hyperparameterSearchSpace is not null)
        {
            refused.Add(
                $"{nameof(ConfigureHyperparameterOptimizer)}: every trial is a complete training run over the same " +
                "training and validation rows, and a stream is read once, in order, with no rows held back to score " +
                "a trial on.");
        }

        if (_federatedLearningOptions is not null)
        {
            refused.Add(
                $"{nameof(ConfigureFederatedLearning)}: the training rows are partitioned across the simulated " +
                "clients before the first round, which needs every row and its index up front.");
        }

        if (_continualLearningEnabled)
        {
            refused.Add(
                $"{nameof(ConfigureContinualLearning)}: the continual learner takes the task as an indexable " +
                "dataset that it subsets, shuffles and revisits (replay memory, importance estimates over the " +
                "task), and a stream has no index to revisit.");
        }

        if (_selfSupervisedLearningPretrainAction is not null || _selfSupervisedLearningConfig?.Method is not null)
        {
            refused.Add(
                $"{nameof(ConfigureSelfSupervisedLearning)}: pretraining makes its own passes over all the " +
                "unlabeled inputs before the supervised fit starts.");
        }

        if (_trainingGroups is not null)
        {
            refused.Add(
                $"{nameof(ConfigureTrainingGroups)}: the group ids are matched to training rows by position, and " +
                "a shuffled stream has no stable row positions.");
        }

        if (_dataPreparationPipeline is { Count: > 0 })
        {
            refused.Add(
                $"{nameof(ConfigureDataPreparation)}: its operations add and remove rows (resampling, outlier " +
                "removal) from statistics over the whole set. Per-feature transforms do work on a stream; " +
                "configure those through the preprocessing pipeline.");
        }

        if (_crossValidator is not null)
        {
            refused.Add(
                $"{nameof(ConfigureCrossValidation)}: every fold retrains on a different subset of the same rows.");
        }

        if (_configuredDataSplitter is not null)
        {
            refused.Add(
                $"{nameof(ConfigureDataSplitter)}: a splitter divides rows it holds. On a stream, pass the " +
                "held-out data as StreamingTrainingOptions.ValidationLoader.");
        }

        if (refused.Count == 0)
        {
            return;
        }

        throw new NotSupportedException(
            $"The streaming data-loader path cannot honour {refused.Count} configured option(s), and training " +
            "without them would give no sign of it. Remove the call(s), or load the data through an " +
            "IInputOutputDataLoader if it fits in memory:" + Environment.NewLine +
            string.Join(Environment.NewLine, refused.Select(reason => " - " + reason)));
    }

    /// <summary>
    /// Refuses, for a run with more than one rank, the configured trainers that update the model on each rank
    /// separately with nothing exchanged between the ranks.
    /// </summary>
    /// <exception cref="NotSupportedException">More than one rank and one or more such trainers.</exception>
    private void RejectUnsynchronizedTrainersOnMultipleRanks(DistributedTraining.ICommunicationBackend<T>? distBackend)
    {
        if (distBackend is not { WorldSize: > 1 })
        {
            return;
        }

        var refused = new List<string>();
        if (_knowledgeDistillationOptions is not null || _configuredDistillationStrategy is not null)
        {
            refused.Add(nameof(ConfigureKnowledgeDistillation));
        }

        if (_fineTuningConfiguration is { Enabled: true })
        {
            refused.Add(nameof(ConfigureFineTuning));
        }

        if (_trainingPipelineConfiguration?.Stages is { Count: > 0 })
        {
            refused.Add(nameof(ConfigureTrainingPipeline));
        }

        if (_curriculumLearningOptions is { Dataset: not null })
        {
            refused.Add(nameof(ConfigureCurriculumLearning));
        }

        if (refused.Count == 0)
        {
            return;
        }

        throw new NotSupportedException(
            $"{string.Join(", ", refused)} cannot run on {distBackend.WorldSize} ranks: each trains the model through " +
            "its own trainer, which exchanges no gradients, so every rank would finish with different weights. Run " +
            "with one rank, or run the stage on the trained model after the distributed build.");
    }

    /// <summary>
    /// Checks, before anything is changed, that the streaming loop has a place to apply
    /// <see cref="ConfigureRegularization"/> for the configured model and optimizer.
    /// </summary>
    /// <exception cref="NotSupportedException">Nothing on this path could apply the penalty.</exception>
    private void EnsureStreamingRegularizationCanBeApplied()
    {
        if (_regularization is null || _model is null)
        {
            return;
        }

        if (_model is NeuralNetworks.NeuralNetworkBase<T>)
        {
            bool tensorTraining = typeof(TInput) == typeof(Tensor<T>) && typeof(TOutput) == typeof(Tensor<T>);
            bool optimizerSteps = _optimizer is null
                ? tensorTraining
                : _optimizer is Optimizers.GradientBasedOptimizerBase<T, TInput, TOutput>
                    && _optimizer is IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>;
            if (optimizerSteps)
            {
                return;
            }

            throw new NotSupportedException(
                $"{nameof(ConfigureRegularization)} on the streaming data-loader path is applied by the optimizer " +
                "that steps the network, which must be a gradient-based optimizer over tensors (AdamOptimizer, " +
                "AdamWOptimizer, StochasticGradientDescentOptimizer and the like). " +
                (_optimizer is null
                    ? $"This build trains {typeof(TInput).Name} inputs against {typeof(TOutput).Name} targets, for " +
                      "which no such optimizer can be created."
                    : $"{_optimizer.GetType().Name} is not one, so the network would step without the penalty.") +
                $" Pass one to {nameof(ConfigureOptimizer)}.");
        }

        if (_model is IGradientComputable<T, TInput, TOutput> && _model is IParameterizable<T, TInput, TOutput>)
        {
            if (_regularization is Regularization.RegularizationBase<T, TInput, TOutput>
                || typeof(TOutput) == typeof(Vector<T>))
            {
                return;
            }

            throw new NotSupportedException(
                $"{nameof(ConfigureRegularization)} on the streaming data-loader path adds the penalty to the " +
                $"parameter gradient, a Vector<{typeof(T).Name}>. {_regularization.GetType().Name} can only regularize " +
                $"{typeof(TOutput).Name} values; derive it from RegularizationBase to give it the vector overload.");
        }

        throw new NotSupportedException(
            $"{nameof(ConfigureRegularization)} cannot be applied to {_model.GetType().Name} on the streaming " +
            "data-loader path: the model neither steps through an optimizer nor exposes its gradient and parameters " +
            "(IGradientComputable and IParameterizable), so there is no update to add the penalty to.");
    }

    /// <summary>
    /// Gives a neural network with no builder optimizer an <see cref="Optimizers.AdamOptimizer{T, TInput, TOutput}"/>
    /// when regularization is configured, as the in-memory path does, so there is an optimizer to carry the
    /// penalty.
    /// </summary>
    private void PromoteStreamingOptimizerForRegularization()
    {
        if (_regularization is not null && _optimizer is null && _model is NeuralNetworks.NeuralNetworkBase<T>)
        {
            _optimizer = new Optimizers.AdamOptimizer<T, TInput, TOutput>(_model);
        }
    }

    /// <summary>
    /// Pushes <see cref="ConfigureRegularization"/> into the optimizer that steps a neural network, as the
    /// in-memory path does. The tape step applies the optimizer's regularization to the gradients, once.
    /// </summary>
    /// <remarks>
    /// Models that update through <see cref="IGradientComputable{T, TInput, TOutput}"/> have no optimizer step;
    /// their penalty is added to the gradient by <see cref="RegularizeStreamingGradient"/>.
    /// </remarks>
    private void ApplyStreamingRegularization()
    {
        if (_regularization is not null
            && _model is NeuralNetworks.NeuralNetworkBase<T>
            && _optimizer is Optimizers.GradientBasedOptimizerBase<T, TInput, TOutput> gradientOptimizer)
        {
            gradientOptimizer.SetRegularization(_regularization);
        }
    }
}
