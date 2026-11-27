using AiDotNet.Models.Options;
using AiDotNet.Models.Results;

namespace AiDotNet;

public partial class AiModelBuilder<T, TInput, TOutput>
{
    private SampleObjectiveTrainingOptions? _sampleObjectiveTrainingOptions;

    // Set by BuildAsync(samples) for the duration of that build, so the shared BuildAsync preamble (determinism,
    // JIT, gradient checkpointing, weight streaming) runs before the sample-objective loop. It closes over the
    // sample type, which the builder itself is not generic in.
    private Func<CancellationToken, Task<AiModelResult<T, TInput, TOutput>>>? _sampleObjectiveBuild;

    /// <summary>
    /// Changes how <see cref="Build{TSample}(IReadOnlyList{TSample})"/> trains a model on its own sample-level
    /// objective. Optional: without it the builder uses the defaults of
    /// <see cref="SampleObjectiveTrainingOptions"/>.
    /// </summary>
    /// <param name="options">The settings to use. A validated copy is stored.</param>
    /// <returns>This builder, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="options"/> is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">A setting is out of range.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Skip this unless you want to change the number of epochs, the held-out
    /// fraction or early stopping; the defaults already shuffle, validate and stop early.</para>
    /// </remarks>
    /// <example>
    /// <code>
    /// var samples = new List&lt;TtsTrainingSample&lt;float&gt;&gt;();
    /// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
    /// var result = new AiModelBuilder&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;()
    ///     .ConfigureModel(new Tacotron2&lt;float&gt;(architecture, new Tacotron2Options()))
    ///     .ConfigureSampleObjectiveTraining(new SampleObjectiveTrainingOptions { Epochs = 500, EarlyStoppingPatience = 0 })
    ///     .Build(samples);
    /// </code>
    /// </example>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureSampleObjectiveTraining(SampleObjectiveTrainingOptions options)
    {
        if (options is null) throw new ArgumentNullException(nameof(options));
        var snapshot = new SampleObjectiveTrainingOptions(options);
        snapshot.Validate();
        _sampleObjectiveTrainingOptions = snapshot;
        return this;
    }

    /// <summary>
    /// Trains the configured model on typed samples with the objective its paper defines, blocking until the build
    /// completes.
    /// </summary>
    /// <typeparam name="TSample">The model's training sample, such as <see cref="TextToSpeech.TtsTrainingSample{T}"/>
    /// or <see cref="Audio.MusicGen.MusicGenTrainingSample{T}"/>.</typeparam>
    /// <param name="samples">The training samples.</param>
    /// <returns>The trained result. Its <c>GetTrainingMetricsHistory()</c> holds the mean training loss of every epoch
    /// under <c>"train_loss"</c> and, when samples were held out, the held-out objective under
    /// <c>"validation_loss"</c>.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="samples"/> is null.</exception>
    /// <exception cref="ArgumentException"><paramref name="samples"/> is empty or contains null.</exception>
    /// <exception cref="InvalidOperationException">No model is configured, the configured model does not train on
    /// <typeparamref name="TSample"/>, or a data loader is configured as well.</exception>
    /// <remarks>
    /// <para>
    /// Some models are trained on more than an input and a target: a text-to-speech model on forced-alignment
    /// durations, codec tokens, a speaker id or a reference recording; a music model on a clip and its description.
    /// <see cref="Build(TInput, TOutput)"/> has nowhere to put that supervision, so such a model
    /// (an <see cref="Interfaces.ISampleObjectiveModel{T, TSample}"/>) is trained through this overload instead.
    /// </para>
    /// <para>
    /// The builder runs the training loop with the defaults of <see cref="SampleObjectiveTrainingOptions"/>: it
    /// reshuffles the samples every epoch, holds 10% out and measures the model's objective on them after every
    /// epoch, stops early when that stops improving, reports every epoch to the configured training callbacks and
    /// monitor, and is reproducible unless <c>AllowNondeterminism()</c> was called. Change any of it with
    /// <see cref="ConfigureSampleObjectiveTraining"/>. The model keeps the optimizer its paper uses.
    /// </para>
    /// <para><b>For Beginners:</b> Fill one sample per recording with what your model needs (the error message names
    /// anything missing), hand the list to <c>Build</c>, and use the trained model.</para>
    /// </remarks>
    /// <example>
    /// <code>
    /// var tokens = new Tensor&lt;float&gt;(new[] { 5 }, new Vector&lt;float&gt;(new float[] { 8, 5, 12, 12, 15 }));
    /// var recording = Tensor&lt;float&gt;.CreateRandom(1, 22050);
    /// var samples = new List&lt;TtsTrainingSample&lt;float&gt;&gt; { new() { Tokens = tokens, Audio = recording } };
    /// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
    /// var tacotron = new Tacotron2&lt;float&gt;(architecture, new Tacotron2Options());
    /// var result = new AiModelBuilder&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;()
    ///     .ConfigureModel(tacotron)
    ///     .Build(samples);
    /// var trainLoss = result.GetTrainingMetricsHistory()?["train_loss"];
    /// </code>
    /// </example>
    public AiModelResult<T, TInput, TOutput> Build<TSample>(IReadOnlyList<TSample> samples) =>
        BuildAsync(samples, CancellationToken.None).GetAwaiter().GetResult();

    /// <summary>
    /// Trains the configured model on typed samples with the objective its paper defines.
    /// </summary>
    /// <typeparam name="TSample">The model's training sample, such as <see cref="TextToSpeech.TtsTrainingSample{T}"/>.</typeparam>
    /// <param name="samples">The training samples.</param>
    /// <returns>A task whose result is the trained result.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="samples"/> is null.</exception>
    /// <exception cref="ArgumentException"><paramref name="samples"/> is empty or contains null.</exception>
    /// <exception cref="InvalidOperationException">No model is configured, the configured model does not train on
    /// <typeparamref name="TSample"/>, or a data loader is configured as well.</exception>
    /// <remarks>
    /// The asynchronous form of <see cref="Build{TSample}(IReadOnlyList{TSample})"/>; see it for the defaults.
    /// </remarks>
    public Task<AiModelResult<T, TInput, TOutput>> BuildAsync<TSample>(IReadOnlyList<TSample> samples) =>
        BuildAsync(samples, CancellationToken.None);

    /// <summary>
    /// Trains the configured model on typed samples with the objective its paper defines, observing a cancellation
    /// token.
    /// </summary>
    /// <typeparam name="TSample">The model's training sample, such as <see cref="TextToSpeech.TtsTrainingSample{T}"/>.</typeparam>
    /// <param name="samples">The training samples.</param>
    /// <param name="cancellationToken">Stops training at the next epoch boundary; the model trained so far is returned.</param>
    /// <returns>A task whose result is the trained result.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="samples"/> is null.</exception>
    /// <exception cref="ArgumentException"><paramref name="samples"/> is empty or contains null.</exception>
    /// <exception cref="InvalidOperationException">No model is configured, the configured model does not train on
    /// <typeparamref name="TSample"/>, or a data loader is configured as well.</exception>
    /// <remarks>
    /// The asynchronous form of <see cref="Build{TSample}(IReadOnlyList{TSample})"/>; see it for the defaults.
    /// </remarks>
    public async Task<AiModelResult<T, TInput, TOutput>> BuildAsync<TSample>(
        IReadOnlyList<TSample> samples, CancellationToken cancellationToken)
    {
        if (samples is null) throw new ArgumentNullException(nameof(samples));
        if (samples.Count == 0) throw new ArgumentException("Supply at least one training sample.", nameof(samples));
        for (int i = 0; i < samples.Count; i++)
        {
            if (samples[i] is null) throw new ArgumentException($"Training sample {i} is null.", nameof(samples));
        }

        ThrowIfStreamingTrainingConfigured("a sample-objective Build(samples)");
        if (_dataLoader is not null || _imageDataLoader is not null)
        {
            throw new InvalidOperationException(
                "Build(samples) was called on a builder that already has a data loader configured. Supply the " +
                "training data one way only.");
        }

        if (_model is null)
        {
            throw new InvalidOperationException(
                $"Build(samples) needs a model. Call ConfigureModel with a model that trains on {typeof(TSample).Name} " +
                "before building.");
        }

        if (_model is not Interfaces.ISampleObjectiveModel<T, TSample> trainable)
        {
            throw new InvalidOperationException(
                $"The configured model, {_model.GetType().Name}, does not train on {typeof(TSample).Name}. " +
                "Build(samples) is for models whose paper trains them on typed samples (ISampleObjectiveModel); " +
                "train other models with Build(features, labels).");
        }

        var options = _sampleObjectiveTrainingOptions ?? new SampleObjectiveTrainingOptions();
        _sampleObjectiveBuild = token => Task.FromResult(TrainOnSampleObjective(trainable, samples, options, token));
        try
        {
            return await BuildAsync(cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            _sampleObjectiveBuild = null;
        }
    }

    [AiDotNet.Attributes.ScalarKernel("One scalar loss per training step and per held-out sample, read once each to average and report.")]
    private AiModelResult<T, TInput, TOutput> TrainOnSampleObjective<TSample>(
        Interfaces.ISampleObjectiveModel<T, TSample> trainable,
        IReadOnlyList<TSample> samples,
        SampleObjectiveTrainingOptions options,
        CancellationToken cancellationToken)
    {
        RejectBuildTimeAdversarialTraining(RobustnessBuildPath.SampleObjective);
        if (_preprocessingPipeline is not null || _targetPipeline is not null)
        {
            throw new InvalidOperationException(
                "Preprocessing pipelines transform input and target tensors, and a training sample is neither. " +
                "Remove ConfigurePreprocessing for a Build(samples) build.");
        }

        var profilerSession = CreateProfilerSession();
        using var profilerScope = profilerSession?.Scope(nameof(TrainOnSampleObjective));
        ApplyGpuConfiguration();
        ApplyMemoryConfiguration();

        var numOps = MathHelper.GetNumericOperations<T>();
        var random = options.Seed is int seed
            ? AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed)
            : _allowNondeterminism
                ? AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom()
                : AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(42);

        // Held-out split: a random subset, rounded down, never the whole set.
        var order = Enumerable.Range(0, samples.Count).OrderBy(_ => random.Next()).ToArray();
        int heldOut = Math.Min(samples.Count - 1, (int)Math.Floor(samples.Count * options.ValidationFraction));
        var validation = order.Take(heldOut).ToArray();
        var training = order.Skip(heldOut).ToArray();

        var trainLosses = new List<double>(options.Epochs);
        var validationLosses = new List<double>(options.Epochs);
        double bestValidation = double.PositiveInfinity;
        byte[]? bestState = null;
        bool lastEpochIsBest = true;
        int epochsWithoutImprovement = 0;
        bool earlyStop = false;
        string? stopReason = null;
        int steps = 0;

        EpochProgressBridge? epochBridge = null;
        if (_trainingCallbacks.Count > 0 || _trainingMonitor is not null)
        {
            epochBridge = new EpochProgressBridge(this, monitorSessionId: null, cancellationToken, options.Epochs, numOps.Zero);
            InvokeTrainingCallbacksBegin(options.Epochs);
        }

        try
        {
            for (int epoch = 0; epoch < options.Epochs; epoch++)
            {
                if (cancellationToken.IsCancellationRequested)
                {
                    earlyStop = true;
                    stopReason = "Training was cancelled.";
                    break;
                }

                if (options.ShuffleEachEpoch)
                {
                    for (int i = training.Length - 1; i > 0; i--)
                    {
                        int j = random.Next(i + 1);
                        (training[i], training[j]) = (training[j], training[i]);
                    }
                }

                double sum = 0;
                foreach (int index in training)
                {
                    sum += numOps.ToDouble(trainable.Train(samples[index]));
                    steps++;
                }
                double trainLoss = sum / training.Length;
                trainLosses.Add(trainLoss);

                double monitored = trainLoss;
                if (validation.Length > 0)
                {
                    double held = 0;
                    foreach (int index in validation)
                    {
                        held += numOps.ToDouble(trainable.EvaluateTrainingObjective(samples[index]));
                    }
                    monitored = held / validation.Length;
                    validationLosses.Add(monitored);

                    if (monitored < bestValidation)
                    {
                        bestValidation = monitored;
                        epochsWithoutImprovement = 0;
                        lastEpochIsBest = true;
                        // The whole serialized state, not GetParameters: running statistics and other buffers
                        // change in training too, and the input/target path keeps a DeepCopy of its best model.
                        if (options.RestoreBestModel) bestState = SnapshotModelState();
                    }
                    else
                    {
                        epochsWithoutImprovement++;
                        lastEpochIsBest = false;
                    }
                }

                if (epochBridge is not null && !epochBridge.OnEpoch(epoch + 1, options.Epochs, numOps.FromDouble(monitored)))
                {
                    earlyStop = true;
                    stopReason = epochBridge.StopReason ?? "A training callback requested a stop.";
                    break;
                }

                if (validation.Length > 0 && options.EarlyStoppingPatience > 0
                    && epochsWithoutImprovement >= options.EarlyStoppingPatience)
                {
                    earlyStop = true;
                    stopReason = $"The held-out objective did not improve for {options.EarlyStoppingPatience} epochs.";
                    break;
                }
            }
        }
        finally
        {
            if (epochBridge is not null)
            {
                InvokeTrainingCallbacksEnd(epochBridge.LastEpoch, options.Epochs, epochBridge.LastLoss, epochBridge.Elapsed);
            }
        }

        // As the input/target path returns its best step's model: the epoch with the lowest held-out objective.
        if (bestState is not null && !lastEpochIsBest) RestoreModelState(bestState);

        var history = new Dictionary<string, List<double>> { ["train_loss"] = trainLosses };
        if (validationLosses.Count > 0) history["validation_loss"] = validationLosses;

        var optimizationResult = new OptimizationResult<T, TInput, TOutput>
        {
            BestSolution = _model,
            Iterations = steps,
            FitnessHistory = new Vector<T>(trainLosses.Select(numOps.FromDouble)),
            BestFitnessScore = numOps.FromDouble(validationLosses.Count > 0
                ? validationLosses.Min()
                : trainLosses.Count > 0 ? trainLosses[trainLosses.Count - 1] : 0.0),
        };

        return CreateOwnObjectiveResult(
            optimizationResult, preprocessingInfo: null, quantizationInfo: null, profilerSession,
            history, earlyStop, stopReason);
    }

    // The model's full state, taken and restored in place inside an internal persistence scope: a snapshot for early
    // stopping is not a user save, so it does not need the ModelSave capability.
    private byte[] SnapshotModelState()
    {
        using (ModelPersistenceGuard.InternalOperation())
        {
            return _model!.Serialize();
        }
    }

    private void RestoreModelState(byte[] state)
    {
        // Deserialize may leave a network in inference mode; training ends in the mode it was in.
        bool? wasTraining = (_model as NeuralNetworks.NeuralNetworkBase<T>)?.IsTrainingMode;
        using (ModelPersistenceGuard.InternalOperation())
        {
            _model!.Deserialize(state);
        }
        if (wasTraining is bool mode && _model is NeuralNetworks.NeuralNetworkBase<T> network && network.IsTrainingMode != mode)
        {
            network.SetTrainingMode(mode);
        }
    }
}
