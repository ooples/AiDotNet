using AiDotNet.Augmentation;
using AiDotNet.AutoML.NAS;
using AiDotNet.RetrievalAugmentedGeneration.Graph.Communities;
using AiDotNet.RetrievalAugmentedGeneration.Graph.Embeddings;
using AiDotNet.AutoML.Policies;
using AiDotNet.AutoML.SearchSpace;
using AiDotNet.Postprocessing;
using AiDotNet.Preprocessing;
using AiDotNet.Preprocessing.DataPreparation;
using AiDotNet.Preprocessing.Imputers;
using AiDotNet.Preprocessing.Scalers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet;

/// <summary>
/// Internal helpers partial (model resolution, GPU config, quantization, etc.) of <see cref="AiModelBuilder{T, TInput, TOutput}"/>. Split out of the
/// 9.5k-LoC main file (audit-2026-05 finding #12) for reviewability; no behaviour change.
/// </summary>
public partial class AiModelBuilder<T, TInput, TOutput>
{

    /// <summary>
    /// Configures program synthesis to use <c>AiDotNet.Serving</c> for program execution and evaluation (optional).
    /// </summary>
    /// <param name="options">
    /// Serving client options. If null (and <paramref name="client"/> is null), a default configuration is used that targets
    /// <c>http://localhost:52432/</c>.
    /// </param>
    /// <param name="client">Optional custom client implementation. When provided, this takes precedence over <paramref name="options"/>.</param>
    /// <returns>This builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// Some program-synthesis workflows need to run or evaluate generated programs (for example, execute code against test cases).
    /// This method lets you route those operations through a Serving endpoint (or a custom client), which is useful for centralized
    /// execution, resource control, and isolation.
    /// </para>
    /// <para>
    /// Precedence rules:
    /// - If <paramref name="client"/> is provided, it is used.
    /// - Otherwise, if <paramref name="options"/> is provided it is used.
    /// - Otherwise, a default configuration is used that targets <c>http://localhost:52432/</c>.
    /// </para>
    /// <para><b>For Beginners:</b> If you only want the model to generate code, you can skip this.
    /// If you want to automatically execute or evaluate generated code, configure Serving. If you're running
    /// <c>AiDotNet.Serving</c> locally with default settings, calling this method with no parameters is enough.
    ///
    /// Example:
    /// <code>
    /// var result = await new AiModelBuilder&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;()
    ///     .ConfigureProgramSynthesis()
    ///     .ConfigureProgramSynthesisServing() // Defaults to http://localhost:52432/
    ///     .BuildAsync();
    /// </code>
    /// </para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureProgramSynthesisServing(
        ProgramSynthesisServingClientOptions? options = null,
        IProgramSynthesisServingClient? client = null)
    {
        // Defaults-first: calling this method opts into Serving using the standard local endpoint unless overridden.
        _programSynthesisServingClientOptions = options ?? (client is null
            ? new ProgramSynthesisServingClientOptions { BaseAddress = new Uri("http://localhost:52432/") }
            : null);

        _programSynthesisServingClient = client;
        return this;
    }

    /// <summary>
    /// Applies trial hyperparameters from HPO to the optimizer options.
    /// </summary>
    /// <param name="options">The optimizer options to modify.</param>
    /// <param name="trialHyperparameters">Dictionary of hyperparameter names to values.</param>
    /// <remarks>
    /// <para>
    /// This method applies hyperparameters discovered during HPO to the optimizer options.
    /// Common hyperparameters that can be tuned include:
    /// - learning_rate: The learning rate for gradient-based optimizers
    /// - max_iterations: Maximum number of training iterations/epochs
    /// - tolerance: Convergence tolerance
    /// - beta1, beta2: Adam optimizer momentum parameters
    /// - momentum: Momentum for SGD-style optimizers
    /// </para>
    /// <para><b>For Beginners:</b> Hyperparameter optimization (HPO) tries different combinations
    /// of settings to find what works best for your specific problem. This method takes those
    /// discovered settings and applies them to the optimizer before training.
    /// </para>
    /// </remarks>
    private static void ApplyTrialHyperparameters(object options, Dictionary<string, object> trialHyperparameters)
    {
        if (options is null || trialHyperparameters is null || trialHyperparameters.Count == 0)
        {
            return;
        }

        var optionsType = options.GetType();

        foreach (var kvp in trialHyperparameters)
        {
            var paramName = kvp.Key;
            var paramValue = kvp.Value;

            if (paramValue is null)
            {
                continue;
            }

            // Map common hyperparameter names to property names
            var propertyName = MapHyperparameterToProperty(paramName);
            if (string.IsNullOrEmpty(propertyName))
            {
                continue;
            }

            // Find the property on the options object
            var property = optionsType.GetProperty(propertyName,
                System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance);

            if (property is null || !property.CanWrite)
            {
                // Try base types for inherited properties
                var baseType = optionsType.BaseType;
                while (baseType is not null && property is null)
                {
                    property = baseType.GetProperty(propertyName,
                        System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance);
                    baseType = baseType.BaseType;
                }

                if (property is null || !property.CanWrite)
                {
                    continue;
                }
            }

            // Convert and set the value
            try
            {
                var convertedValue = ConvertHyperparameterValue(paramValue, property.PropertyType);
                if (convertedValue is not null)
                {
                    property.SetValue(options, convertedValue);
                }
            }
            catch (Exception)
            {
                // Skip hyperparameters that cannot be converted - this is non-fatal
                // as the optimizer will use its default value
            }
        }
    }

    /// <summary>
    /// Maps common hyperparameter names from search spaces to property names on optimizer options.
    /// </summary>
    private static string MapHyperparameterToProperty(string hyperparameterName)
    {
        // Normalize the name to lowercase for matching
        var normalizedName = hyperparameterName.ToLowerInvariant().Replace("_", "").Replace("-", "");

        return normalizedName switch
        {
            // Learning rate variations
            "learningrate" or "lr" or "initiallearningrate" => "InitialLearningRate",

            // Iteration/epoch settings
            "maxiterations" or "iterations" or "epochs" or "maxepochs" => "MaxIterations",

            // Convergence settings
            "tolerance" or "tol" or "convergencetolerance" => "Tolerance",

            // Early stopping
            "earlystoppingpatience" or "patience" => "EarlyStoppingPatience",

            // Adam-specific parameters (check optimizer-specific options first)
            "beta1" or "b1" => "Beta1",
            "beta2" or "b2" => "Beta2",
            "epsilon" or "eps" => "Epsilon",

            // Momentum
            "momentum" or "initialmomentum" => "InitialMomentum",

            // Learning rate scheduling
            "learningratedecay" or "lrdecay" or "decay" => "LearningRateDecay",
            "minlearningrate" or "minlr" => "MinLearningRate",
            "maxlearningrate" or "maxlr" => "MaxLearningRate",

            // Regularization strength
            "l2regularization" or "weightdecay" or "regularization" => "RegularizationStrength",

            // Batch size (for applicable optimizers)
            "batchsize" or "batch" => "BatchSize",

            // Gradient clipping
            "maxgradientnorm" or "clipnorm" or "gradientclipnorm" => "MaxGradientNorm",
            "maxgradientvalue" or "clipvalue" or "gradientclipvalue" => "MaxGradientValue",

            // Unknown hyperparameter - try using the name directly
            _ => ToPascalCase(hyperparameterName)
        };
    }

    /// <summary>
    /// Converts a hyperparameter name to PascalCase for property lookup.
    /// </summary>
    private static string ToPascalCase(string name)
    {
        if (string.IsNullOrEmpty(name))
        {
            return name;
        }

        var parts = name.Split(new[] { '_', '-' }, StringSplitOptions.RemoveEmptyEntries);
        var result = new System.Text.StringBuilder();

        foreach (var part in parts)
        {
            if (part.Length > 0)
            {
                result.Append(char.ToUpperInvariant(part[0]));
                if (part.Length > 1)
                {
                    result.Append(part.Substring(1).ToLowerInvariant());
                }
            }
        }

        return result.ToString();
    }

    /// <summary>
    /// Converts a hyperparameter value to the target property type.
    /// </summary>
    private static object? ConvertHyperparameterValue(object value, Type targetType)
    {
        if (value is null)
        {
            return null;
        }

        var valueType = value.GetType();

        // If already the correct type, return as-is
        if (targetType.IsAssignableFrom(valueType))
        {
            return value;
        }

        // Handle numeric conversions
        if (targetType == typeof(double))
        {
            return Convert.ToDouble(value);
        }
        if (targetType == typeof(float))
        {
            return Convert.ToSingle(value);
        }
        if (targetType == typeof(int))
        {
            return Convert.ToInt32(value);
        }
        if (targetType == typeof(long))
        {
            return Convert.ToInt64(value);
        }
        if (targetType == typeof(bool))
        {
            return Convert.ToBoolean(value);
        }

        // Handle nullable types
        var underlyingType = Nullable.GetUnderlyingType(targetType);
        if (underlyingType is not null)
        {
            return ConvertHyperparameterValue(value, underlyingType);
        }

        // Try using Convert.ChangeType as a last resort
        try
        {
            return Convert.ChangeType(value, targetType);
        }
        catch (Exception)
        {
            return null;
        }
    }


    /// <summary>
    /// Derives the open generic type definition from the actual model instance.
    /// </summary>
    private static Type? DeriveModelType(IFullModel<T, TInput, TOutput>? model)
    {
        if (model is null)
            return null;

        var runtimeType = model.GetType();
        return runtimeType.IsGenericType ? runtimeType.GetGenericTypeDefinition() : runtimeType;
    }

    /// <summary>
    /// Computes a robust hash of the training data for version control and lineage tracking.
    /// </summary>
    /// <param name="features">The feature matrix (X).</param>
    /// <param name="targets">The target vector (y).</param>
    /// <param name="numOps">Numeric operations for type conversion.</param>
    /// <returns>A 16-character hex hash representing the data version.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a hash that captures the essential characteristics of the training data:
    /// - Dataset dimensions (rows, columns)
    /// - Sample of feature values from first, middle, and last rows
    /// - Sample of target values from first, middle, and last positions
    /// - Statistical summary (sum of sampled values for collision resistance)
    /// </para>
    /// <para><b>For Beginners:</b> This hash is like a fingerprint for your training data.
    /// If the data changes, the hash will change too, allowing you to track exactly which
    /// version of the data was used to train a model. This is essential for reproducibility.
    /// </para>
    /// </remarks>
    private static string ComputeDataVersionHash(Matrix<T> features, Vector<T> targets, INumericOperations<T> numOps)
    {
        var hashBuilder = new StringBuilder();

        // Include dimensions for basic structure identification
        hashBuilder.Append($"X:{features.Rows}x{features.Columns};");
        hashBuilder.Append($"y:{targets.Length};");

        // Sample feature values from first, middle, and last rows for better coverage
        // This catches changes anywhere in the dataset, not just at boundaries
        int maxCols = Math.Min(features.Columns, 10);
        int[] sampleRows = GetSampleRowIndices(features.Rows);

        foreach (int row in sampleRows)
        {
            if (row >= 0 && row < features.Rows)
            {
                hashBuilder.Append($"r{row}:");
                for (int col = 0; col < maxCols; col++)
                {
                    hashBuilder.Append($"{numOps.ToDouble(features[row, col]):G6},");
                }
                hashBuilder.Append(';');
            }
        }

        // Include target values from sampled positions
        // This ensures the hash changes if targets are modified
        int[] sampleTargetIndices = GetSampleRowIndices(targets.Length);
        hashBuilder.Append("y:");
        foreach (int idx in sampleTargetIndices)
        {
            if (idx >= 0 && idx < targets.Length)
            {
                hashBuilder.Append($"{numOps.ToDouble(targets[idx]):G6},");
            }
        }
        hashBuilder.Append(';');

        // Add a statistical fingerprint for additional collision resistance
        // Sum of sampled values helps detect subtle changes
        double featureSum = 0.0;
        double targetSum = 0.0;

        foreach (int row in sampleRows)
        {
            if (row >= 0 && row < features.Rows)
            {
                for (int col = 0; col < maxCols; col++)
                {
                    featureSum += numOps.ToDouble(features[row, col]);
                }
            }
        }

        foreach (int idx in sampleTargetIndices)
        {
            if (idx >= 0 && idx < targets.Length)
            {
                targetSum += numOps.ToDouble(targets[idx]);
            }
        }

        hashBuilder.Append($"fsum:{featureSum:G6};tsum:{targetSum:G6};");

        // Compute SHA256 hash and return first 16 hex characters
        using var sha256 = System.Security.Cryptography.SHA256.Create();
        var hashBytes = sha256.ComputeHash(Encoding.UTF8.GetBytes(hashBuilder.ToString()));

        // Convert bytes to hex string (compatible with net471 which doesn't have Convert.ToHexString)
        var fullHex = BitConverter.ToString(hashBytes).Replace("-", "").ToLowerInvariant();
        return fullHex.Substring(0, Math.Min(16, fullHex.Length));
    }

    /// <summary>
    /// Gets sample row indices for data hashing (first, middle, last).
    /// </summary>
    private static int[] GetSampleRowIndices(int totalRows)
    {
        if (totalRows <= 0)
        {
            return Array.Empty<int>();
        }

        if (totalRows == 1)
        {
            return new[] { 0 };
        }

        if (totalRows == 2)
        {
            return new[] { 0, 1 };
        }

        // Sample first, middle, and last rows
        int middle = totalRows / 2;
        return new[] { 0, middle, totalRows - 1 };
    }

    /// <summary>
    /// Applies GPU acceleration configuration to the global AiDotNetEngine based on user settings.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method configures the AiDotNetEngine (internal GPU/CPU engine) according to the user's
    /// GPU acceleration preferences set via ConfigureGpuAcceleration(). This is an internal method
    /// called automatically during BuildAsync() and is not part of the public facade API.
    /// </para>
    /// <para>
    /// The facade pattern is maintained: users configure GPU via ConfigureGpuAcceleration() with
    /// nullable defaults (null = industry standard behavior), and this method translates those
    /// settings into internal engine configuration.
    /// </para>
    /// <para><b>GPU Usage Level Behaviors:</b>
    /// - <b>Null config (default)</b>: Auto-detect GPU with CPU fallback (industry standard)
    /// - <b>Default</b>: Balanced GPU usage, good for most desktop GPUs (recommended)
    /// - <b>Conservative</b>: Auto-detect GPU, use it only for very large operations, frequent CPU fallback
    /// - <b>Aggressive</b>: Force GPU, throw exception if not available, use GPU for smaller operations
    /// - <b>AlwaysGpu</b>: Force all operations to GPU (maximize GPU utilization)
    /// - <b>AlwaysCpu</b>: Force CPU-only execution, never use GPU
    /// </para>
    /// <para><b>GPU Device Type Behaviors:</b>
    /// - <b>Auto</b>: Use DirectGpu backend order (CUDA → OpenCL → HIP) with CPU fallback
    /// - <b>CUDA</b>: Force NVIDIA CUDA backend (throws if NVIDIA GPU not available)
    /// - <b>OpenCL</b>: Force OpenCL backend (works with NVIDIA, AMD, Intel, throws if no GPU)
    /// - <b>CPU</b>: Force CPU-only execution (equivalent to UsageLevel.AlwaysCpu)
    /// </para>
    /// </remarks>
    /// <summary>
    /// Wires instance-level preprocessing/postprocessing transformers onto DocumentNeuralNetworkBase models.
    /// Always passes the current pipeline values so prior state is cleared on reuse.
    /// </summary>
    private void ConfigureDocumentTransformers(IFullModel<T, TInput, TOutput>? model)
    {
        if (model is not Document.DocumentNeuralNetworkBase<T> documentModel)
        {
            return;
        }

        // Transformers are now configured through the pipeline directly, not on the model
    }

    /// <summary>
    /// Fits the configured <see cref="_postprocessingPipeline"/> on the model's
    /// training-set predictions BEFORE attaching it to an
    /// <see cref="AiModelResultOptions{T,TInput,TOutput}"/>. Shared across every
    /// Build* path so each routing variant (supervised / direct-training /
    /// meta-learning / RL / federated / distributed / inference-only) goes
    /// through the same fit-vs-fail decision (review #1368 C6WJG).
    /// </summary>
    /// <param name="bestSolution">
    /// The trained model used to produce training-set predictions. When non-null
    /// alongside <paramref name="trainingInput"/>, the helper fits the pipeline
    /// inline by calling <c>bestSolution.Predict(trainingInput)</c> and feeding
    /// the result to <c>PostprocessingPipeline.Fit</c>.
    /// </param>
    /// <param name="trainingInput">
    /// The training-set features. Required alongside <paramref name="bestSolution"/>
    /// for inline fit. When null, this path is treated as "no training data
    /// available" (inference-only / meta-learning / RL) and the pipeline
    /// must be pre-fitted by the caller.
    /// </param>
    /// <param name="buildPathName">
    /// Human-readable name of the Build* path emitting the call (used in the
    /// thrown diagnostic to point the user at the right configuration step).
    /// </param>
    /// <exception cref="InvalidOperationException">
    /// Thrown if the configured pipeline is non-empty and not yet fitted AND
    /// either the build path supplies no training data, or the inline fit
    /// throws (predict shape mismatch, pipeline transform failure, etc.).
    /// </exception>
    private void FitPostprocessingIfNeeded(
        IFullModel<T, TInput, TOutput>? bestSolution,
        TInput? trainingInput,
        string buildPathName)
    {
        if (_postprocessingPipeline is null
            || _postprocessingPipeline.Count == 0
            || _postprocessingPipeline.IsFitted)
        {
            return;
        }

        if (bestSolution is null || trainingInput is null)
        {
            throw new InvalidOperationException(
                $"ConfigurePostprocessing is configured but the {buildPathName} build path " +
                "does not have training data available to fit the pipeline (the path runs " +
                "without supervised features, or the trained model was not produced). " +
                "Pre-fit the pipeline via PostprocessingPipeline.Fit(...) on representative " +
                "model outputs before calling BuildAsync(), or remove ConfigurePostprocessing " +
                "on this build path (review #1368 C6WJG).");
        }

        try
        {
            // Optionally slice trainingInput to the configured max-rows cap
            // (review #1368 C7HAu) so the post-train fit doesn't double the
            // Build-time inference cost when the user's pipeline
            // transformers don't need the full training set.
            var fitInput = _postprocessingFitMaxRows is int cap
                ? TrySliceFirstNSamples(trainingInput, cap)
                : trainingInput;
            var trainPreds = bestSolution.Predict(fitInput);
            _postprocessingPipeline.Fit(trainPreds);
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (OutOfMemoryException)
        {
            throw;
        }
        catch (Exception ex)
        {
            // Narrowed (review #1368 C7mmQ): OCE/OOM rethrown above so they
            // surface with their original type; only genuine fit-time
            // failures (shape mismatch, pipeline transform misconfiguration)
            // get re-wrapped in InvalidOperationException with the
            // diagnostic-friendly message below.
            throw new InvalidOperationException(
                $"ConfigurePostprocessing on the {buildPathName} build path: failed to fit " +
                $"pipeline on training predictions: {ex.GetType().Name}: {ex.Message}. " +
                "Inspect the pipeline's transform definition and the training-prediction shape, " +
                "or omit ConfigurePostprocessing if the pipeline is meant to be fitted by the " +
                "caller after Build (review #1368 C6WJG).", ex);
        }
    }

    /// <summary>
    /// Returns a prefix slice of <paramref name="x"/> containing at most
    /// <paramref name="maxRows"/> leading samples. For <see cref="Tensor{T}"/>
    /// inputs uses the same row-major bulk Span CopyTo path as
    /// <see cref="TrySliceFirstSampleForLoRAWarmup"/>; for non-Tensor TInput
    /// (or inputs already smaller than the cap) returns x unchanged. Used
    /// by <see cref="FitPostprocessingIfNeeded"/> to cap the doubled
    /// Build-time inference cost (review #1368 C7HAu).
    /// </summary>
    private static TInput TrySliceFirstNSamples(TInput x, int maxRows)
    {
        if (x is Tensor<T> tensor && tensor.Shape.Length > 0 && tensor.Shape[0] > maxRows)
        {
            var sliceShape = new int[tensor.Shape.Length];
            sliceShape[0] = maxRows;
            for (int i = 1; i < tensor.Shape.Length; i++) sliceShape[i] = tensor.Shape[i];

            int perSample = 1;
            for (int i = 1; i < tensor.Shape.Length; i++) perSample *= tensor.Shape[i];
            long total = (long)maxRows * perSample;
            if (total > int.MaxValue)
            {
                // Cap that would overflow int slice — fall back to full input
                // rather than truncating to int.MaxValue and silently losing
                // the trailing rows.
                return x;
            }
            var slice = new Tensor<T>(sliceShape);
            tensor.Data.Span.Slice(0, (int)total).CopyTo(slice.Data.Span);
            if (slice is TInput typedSlice) return typedSlice;
        }
        return x;
    }

    /// <summary>
    /// Resolves a modality-specific built-in augmenter from
    /// <paramref name="config"/>'s settings blocks. Returns null if no
    /// modality settings are populated for a type matching
    /// <typeparamref name="TInput"/>, leaving augmentation disabled
    /// (review #1368 C6WKu).
    /// </summary>
    /// <remarks>
    /// The factory output types are modality-specific
    /// (<c>ImageTensor&lt;T&gt;</c>, <c>Tensor&lt;T&gt;</c>, <c>string[]</c>,
    /// <c>ImageTensor&lt;T&gt;[]</c>); this method picks the branch
    /// whose data type matches <typeparamref name="TInput"/>. When no
    /// modality matches the current TInput the method returns null and
    /// the caller treats augmentation as not configured rather than
    /// throwing — settings populated for a TInput-mismatched modality
    /// are simply inert (e.g. ImageSettings on a tabular AiModelBuilder).
    /// </remarks>
    private object? ResolveModalityAugmenter(Augmentation.AugmentationConfig config)
    {
        var globalProb = config.Probability;
        // typeof equality on the OPEN generic types — derived classes of
        // the AiDotNet shape primitives would NOT have a built-in
        // augmenter that knows their layout, and the factory's pipeline
        // would silently fail at the typed cast back to
        // IAugmentation<T,TInput> (review #1368 C8ehc). Exact-type match
        // is the safe contract: callers with custom subclasses must
        // supply their own CustomAugmenter.
        if (config.ImageSettings is { } img && typeof(TInput) == typeof(Augmentation.Image.ImageTensor<T>))
        {
            return Augmentation.ModalityAugmenterFactory.BuildImageAugmenter<T>(img, globalProb);
        }
        // Audio dispatches on Tensor<T> (waveform) and Tabular dispatches
        // on Matrix<T> (rows-by-features) — different TInput types, so
        // they cannot fire on the same builder (review #1368 C88Lb:
        // earlier comment mistakenly said both target Tensor<T>, which
        // would have made the audio-wins-over-tabular ordering
        // load-bearing — it isn't, because the two settings produce
        // pipelines for distinct TInput types and the typeof guards
        // already pick the unique match).
        if (config.AudioSettings is { } aud && typeof(TInput) == typeof(AiDotNet.Tensors.LinearAlgebra.Tensor<T>))
        {
            return Augmentation.ModalityAugmenterFactory.BuildAudioAugmenter<T>(aud, globalProb);
        }
        if (config.TabularSettings is { } tab && typeof(TInput) == typeof(AiDotNet.Tensors.LinearAlgebra.Matrix<T>))
        {
            return Augmentation.ModalityAugmenterFactory.BuildTabularAugmenter<T>(tab, globalProb);
        }
        if (config.TextSettings is { } txt && typeof(TInput) == typeof(string[]))
        {
            return Augmentation.ModalityAugmenterFactory.BuildTextAugmenter<T>(txt, globalProb);
        }
        if (config.VideoSettings is { } vid && typeof(TInput) == typeof(Augmentation.Image.ImageTensor<T>[]))
        {
            return Augmentation.ModalityAugmenterFactory.BuildVideoAugmenter<T>(vid, globalProb);
        }
        return null;
    }

    private void ApplyGpuConfiguration()
    {
        ApplyGpuConfigurationCore();
        ReportAccelerationIfRequested();
    }

    private void ReportAccelerationIfRequested()
    {
        if (!_reportAccelerationAtBuild)
        {
            return;
        }

        _accelerationSnapshot = AiDotNet.Diagnostics.AccelerationDiagnostics.GetSnapshot();
        var report = AiDotNet.Diagnostics.AccelerationDiagnostics.GetReport();
        (_accelerationLogger ?? Console.WriteLine).Invoke(report);
    }

    private AiModelResult<T, TInput, TOutput> AttachDiagnostics(AiModelResult<T, TInput, TOutput> result)
    {
        if (_accelerationSnapshot is not null)
        {
            result.AccelerationSnapshot = _accelerationSnapshot;
        }
        if (_tensorsOpProfilingEnabled)
        {
            _tensorsOperationProfile = AiDotNet.Diagnostics.TensorsOperationProfile.Capture();
            result.TensorsOperationProfile = _tensorsOperationProfile;
        }
        return result;
    }

    private void ApplyGpuConfigurationCore()
    {
        // Skip if no GPU configuration was provided (null = default = auto-detect with CPU fallback)
        if (_gpuAccelerationConfig == null)
        {
            // Honor the documented AIDOTNET_DISABLE_GPU opt-out on the auto-detect default path.
            // AiDotNet.Tensors' GpuAutoDetectModuleInit already respects this env var, but the
            // EXPLICIT AutoDetectAndConfigureGpu() call below did not — so a default BuildAsync()
            // would re-enable the GPU even when the consumer opted out. That gap makes every
            // BuildAsync run flip the process to the GPU engine; on a tiny model the per-op
            // host<->device copy cost makes training orders of magnitude SLOWER than CPU, and it
            // leaks the GPU engine to subsequent in-process work (the cause of the
            // BuildAsyncResidualModeCollapse / ByteLMV256 integration tests timing out / collapsing
            // on GPU-equipped dev boxes; CI has no GPU so it never reproduced there). Callers who
            // want GPU explicitly still get it via ConfigureGpuAcceleration(...) (the branch below),
            // which is unaffected by this opt-out.
            var disableGpu = Environment.GetEnvironmentVariable("AIDOTNET_DISABLE_GPU");
            if (!string.IsNullOrEmpty(disableGpu))
            {
                AiDotNetEngine.ResetToCpu();
                return;
            }

            // Industry standard default: Try to auto-detect GPU, use CPU fallback if not available
            // This is silent and non-intrusive - if GPU exists, use it; if not, use CPU
            try
            {
                AiDotNetEngine.AutoDetectAndConfigureGpu();
            }
            catch (Exception)
            {
                // Silently fall back to CPU if GPU detection fails
                // This ensures the library works out of the box on any hardware
            }
            return;
        }

        if (_gpuAccelerationConfig.UsageLevel == AiDotNet.Engines.GpuUsageLevel.AlwaysCpu)
        {
            AiDotNetEngine.ResetToCpu();
            return;
        }

        if (_gpuAccelerationConfig.DeviceType == AiDotNet.Engines.GpuDeviceType.CPU)
        {
            AiDotNetEngine.ResetToCpu();
            return;
        }

        if (_gpuAccelerationConfig.DeviceType == AiDotNet.Engines.GpuDeviceType.CUDA)
        {
            Environment.SetEnvironmentVariable("AIDOTNET_DIRECTGPU_BACKENDS", "cuda");
        }
        else if (_gpuAccelerationConfig.DeviceType == AiDotNet.Engines.GpuDeviceType.OpenCL)
        {
            Environment.SetEnvironmentVariable("AIDOTNET_DIRECTGPU_BACKENDS", "opencl");
        }

        // Apply configuration based on usage level
        switch (_gpuAccelerationConfig.UsageLevel)
        {
            case AiDotNet.Engines.GpuUsageLevel.AlwaysCpu:
                // Force CPU-only execution (useful for debugging, testing, or CPU-only servers)
                AiDotNetEngine.ResetToCpu();
                break;

            case AiDotNet.Engines.GpuUsageLevel.Default:
                // Balanced GPU usage - recommended mode for most users
                // Auto-detect GPU with intelligent fallback for typical desktop GPUs
                try
                {
                    bool gpuDetected = AiDotNetEngine.AutoDetectAndConfigureGpu();
                    if (!gpuDetected)
                    {
                        // No GPU detected - system already fell back to CPU
                        // No error needed, CPU fallback is expected behavior
                    }
                }
                catch (Exception ex)
                {
                    // GPU initialization failed - fall back to CPU
                    Console.WriteLine($"[AiDotNet] GPU initialization failed: {ex.Message}");
                    Console.WriteLine("[AiDotNet] Falling back to CPU execution");
                    AiDotNetEngine.ResetToCpu();
                }
                break;

            case AiDotNet.Engines.GpuUsageLevel.Conservative:
                // Use GPU conservatively - auto-detect but use higher thresholds and more frequent CPU fallback
                // This is for older/slower GPUs or systems where GPU reliability is a concern
                try
                {
                    bool gpuDetected = AiDotNetEngine.AutoDetectAndConfigureGpu();
                    if (gpuDetected)
                    {
                        Console.WriteLine($"[AiDotNet] Conservative GPU mode enabled: {AiDotNetEngine.Current.Name}");
                        Console.WriteLine("[AiDotNet] GPU will be used only for very large operations (100K+ elements)");
                    }
                    else
                    {
                        // No GPU detected - fall back to CPU (expected behavior in Conservative mode)
                        Console.WriteLine("[AiDotNet] No GPU detected - using CPU (Conservative mode)");
                    }
                }
                catch (Exception ex)
                {
                    // GPU initialization failed in Conservative mode - fall back to CPU silently
                    Console.WriteLine($"[AiDotNet] GPU initialization failed in Conservative mode: {ex.Message}");
                    Console.WriteLine("[AiDotNet] Falling back to CPU execution");
                    AiDotNetEngine.ResetToCpu();
                }
                break;

            case AiDotNet.Engines.GpuUsageLevel.Aggressive:
                // Force GPU with minimal fallback - throw exception if GPU is not available
                // This is for users with high-end GPUs who want maximum performance and need to know if GPU fails
                try
                {
                    bool gpuDetected = AiDotNetEngine.AutoDetectAndConfigureGpu();
                    if (!gpuDetected)
                    {
                        throw new InvalidOperationException(
                            "GPU acceleration is set to Aggressive mode but no compatible GPU was detected. " +
                            "Aggressive mode requires a GPU to be available. " +
                            "Options: (1) Install a compatible GPU (NVIDIA/AMD/Intel), " +
                            "(2) Install GPU drivers, " +
                            "(3) Use GpuUsageLevel.Default for automatic CPU fallback, " +
                            "(4) Use GpuUsageLevel.AlwaysCpu for CPU-only execution.");
                    }

                    // Verify GPU is actually being used
                    if (!AiDotNetEngine.Current.SupportsGpu)
                    {
                        throw new InvalidOperationException(
                            "GPU acceleration is set to Aggressive mode but the current engine does not support GPU. " +
                            "This may indicate a GPU initialization failure. Check GPU drivers and compatibility.");
                    }

                    Console.WriteLine($"[AiDotNet] Aggressive GPU mode enabled: {AiDotNetEngine.Current.Name}");
                }
                catch (InvalidOperationException)
                {
                    // Re-throw our explicit exceptions
                    throw;
                }
                catch (Exception ex)
                {
                    // GPU initialization failed in Aggressive mode - this is an error
                    throw new InvalidOperationException(
                        $"GPU acceleration is set to Aggressive mode but GPU initialization failed: {ex.Message}. " +
                        $"Aggressive mode requires a working GPU. " +
                        $"Options: (1) Fix GPU drivers/setup, " +
                        $"(2) Use GpuUsageLevel.Default for automatic CPU fallback, " +
                        $"(3) Use GpuUsageLevel.AlwaysCpu for CPU-only execution.",
                        ex);
                }
                break;

            case AiDotNet.Engines.GpuUsageLevel.AlwaysGpu:
                // Force all operations to GPU - maximize GPU utilization
                // Similar to Aggressive but even more strict
                try
                {
                    bool gpuDetected = AiDotNetEngine.AutoDetectAndConfigureGpu();
                    if (!gpuDetected)
                    {
                        throw new InvalidOperationException(
                            "GPU acceleration is set to AlwaysGpu mode but no compatible GPU was detected. " +
                            "AlwaysGpu mode requires a GPU to be available. " +
                            "Options: (1) Install a compatible GPU (NVIDIA/AMD/Intel), " +
                            "(2) Install GPU drivers, " +
                            "(3) Use GpuUsageLevel.Default for automatic CPU fallback, " +
                            "(4) Use GpuUsageLevel.AlwaysCpu for CPU-only execution.");
                    }

                    // Verify GPU is actually being used
                    if (!AiDotNetEngine.Current.SupportsGpu)
                    {
                        throw new InvalidOperationException(
                            "GPU acceleration is set to AlwaysGpu mode but the current engine does not support GPU. " +
                            "This may indicate a GPU initialization failure. Check GPU drivers and compatibility.");
                    }

                    Console.WriteLine($"[AiDotNet] AlwaysGpu mode enabled: {AiDotNetEngine.Current.Name}");
                    Console.WriteLine("[AiDotNet] All operations will run on GPU for maximum GPU utilization");
                }
                catch (InvalidOperationException)
                {
                    // Re-throw our explicit exceptions
                    throw;
                }
                catch (Exception ex)
                {
                    // GPU initialization failed in AlwaysGpu mode - this is an error
                    throw new InvalidOperationException(
                        $"GPU acceleration is set to AlwaysGpu mode but GPU initialization failed: {ex.Message}. " +
                        $"AlwaysGpu mode requires a working GPU. " +
                        $"Options: (1) Fix GPU drivers/setup, " +
                        $"(2) Use GpuUsageLevel.Default for automatic CPU fallback, " +
                        $"(3) Use GpuUsageLevel.AlwaysCpu for CPU-only execution.",
                        ex);
                }
                break;

            default:
                throw new ArgumentException($"Unknown GPU usage level: {_gpuAccelerationConfig.UsageLevel}");
        }

        if (_gpuAccelerationConfig.DeviceType != AiDotNet.Engines.GpuDeviceType.Auto)
        {
            Console.WriteLine($"[AiDotNet] DirectGpu backend order forced to {_gpuAccelerationConfig.DeviceType}.");
        }

        if (_gpuAccelerationConfig.DeviceIndex != 0)
        {
            Console.WriteLine("[AiDotNet] Warning: GPU DeviceIndex selection is not implemented for DirectGpu backends.");
        }

        // Apply advanced GPU execution options (Phase 2-3)
        ApplyAdvancedGpuExecutionOptions();
    }

    /// <summary>
    /// Applies advanced GPU execution options from the configuration.
    /// </summary>
    /// <remarks>
    /// <para><b>Phase 2-3: GPU Execution Optimization</b></para>
    /// <para>
    /// Configures advanced execution features:
    /// - Execution mode (eager, deferred, scoped-deferred)
    /// - Graph compilation and kernel fusion
    /// - Multi-stream compute/transfer overlap
    /// - Memory management and prefetching
    /// </para>
    /// </remarks>
    private void ApplyAdvancedGpuExecutionOptions()
    {
        if (_gpuAccelerationConfig == null)
        {
            return;
        }

        // Convert to internal execution options
        var executionOptions = _gpuAccelerationConfig.ToExecutionOptions();

        // Validate the options
        try
        {
            executionOptions.Validate();
        }
        catch (ArgumentException ex)
        {
            Console.WriteLine($"[AiDotNet] Warning: Invalid GPU execution option: {ex.Message}");
            return;
        }

        // Log advanced options if verbose logging is enabled
        if (_gpuAccelerationConfig.VerboseLogging)
        {
            Console.WriteLine($"[AiDotNet] Advanced GPU Execution Options:");
            Console.WriteLine($"  Execution Mode: {_gpuAccelerationConfig.ExecutionMode}");
            Console.WriteLine($"  Graph Compilation: {(_gpuAccelerationConfig.EnableGraphCompilation ? "Enabled" : "Disabled")}");
            Console.WriteLine($"  Auto Fusion: {(_gpuAccelerationConfig.EnableAutoFusion ? "Enabled" : "Disabled")}");
            Console.WriteLine($"  Compute/Transfer Overlap: {(_gpuAccelerationConfig.EnableComputeTransferOverlap ? "Enabled" : "Disabled")}");
            Console.WriteLine($"  Max Compute Streams: {_gpuAccelerationConfig.MaxComputeStreams}");
            Console.WriteLine($"  Min GPU Elements: {_gpuAccelerationConfig.MinGpuElements}");
            Console.WriteLine($"  Max GPU Memory: {_gpuAccelerationConfig.MaxGpuMemoryUsage:P0}");
            Console.WriteLine($"  Prefetch: {(_gpuAccelerationConfig.EnablePrefetch ? "Enabled" : "Disabled")}");
            Console.WriteLine($"  Graph Caching: {(_gpuAccelerationConfig.CacheCompiledGraphs ? "Enabled" : "Disabled")}");
            Console.WriteLine($"  Profiling: {(_gpuAccelerationConfig.EnableProfiling ? "Enabled" : "Disabled")}");
        }

        // Set environment variables for GPU execution options
        // These are read by GpuExecutionOptions.FromEnvironment() when creating execution contexts
        SetGpuExecutionEnvironmentVariables(executionOptions);
    }

    /// <summary>
    /// Sets environment variables for GPU execution options.
    /// </summary>
    /// <param name="options">The execution options to set.</param>
    private static void SetGpuExecutionEnvironmentVariables(AiDotNet.Tensors.Engines.Gpu.GpuExecutionOptions options)
    {
        Environment.SetEnvironmentVariable("AIDOTNET_GPU_MIN_ELEMENTS", options.MinGpuElements.ToString());
        Environment.SetEnvironmentVariable("AIDOTNET_GPU_STREAMS", options.MaxComputeStreams.ToString());
        Environment.SetEnvironmentVariable("AIDOTNET_GPU_FORCE_GPU", options.ForceGpu.ToString().ToLowerInvariant());
        Environment.SetEnvironmentVariable("AIDOTNET_GPU_FORCE_CPU", options.ForceCpu.ToString().ToLowerInvariant());
        Environment.SetEnvironmentVariable("AIDOTNET_GPU_ENABLE_GRAPH", options.EnableGraphCompilation.ToString().ToLowerInvariant());
        Environment.SetEnvironmentVariable("AIDOTNET_GPU_ENABLE_FUSION", options.EnableAutoFusion.ToString().ToLowerInvariant());
        Environment.SetEnvironmentVariable("AIDOTNET_GPU_MAX_MEMORY", options.MaxMemoryUsage.ToString("F2", System.Globalization.CultureInfo.InvariantCulture));
        Environment.SetEnvironmentVariable("AIDOTNET_GPU_ENABLE_PREFETCH", options.EnablePrefetch.ToString().ToLowerInvariant());
        Environment.SetEnvironmentVariable("AIDOTNET_GPU_ENABLE_OVERLAP", options.EnableComputeTransferOverlap.ToString().ToLowerInvariant());
        Environment.SetEnvironmentVariable("AIDOTNET_GPU_EXECUTION_MODE", options.ExecutionMode.ToString());
        Environment.SetEnvironmentVariable("AIDOTNET_GPU_RESIDENT", options.EnableGpuResidency.ToString().ToLowerInvariant());
        Environment.SetEnvironmentVariable("AIDOTNET_GPU_PROFILING", options.EnableProfiling.ToString().ToLowerInvariant());
        Environment.SetEnvironmentVariable("AIDOTNET_GPU_CACHE_GRAPHS", options.CacheCompiledGraphs.ToString().ToLowerInvariant());
    }

    /// <summary>
    /// Applies memory management configuration to models that support it.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Memory management helps train larger models by:
    /// - Gradient checkpointing: Trade compute for memory (recompute activations instead of storing all)
    /// - Activation pooling: Reuse tensor memory to reduce garbage collection
    /// </para>
    /// </remarks>
    private void ApplyMemoryConfiguration()
    {
        // Skip if no memory configuration was provided
        if (_memoryConfig is null)
            return;

        // Apply to models that support memory management
        if (_model is NeuralNetworks.NeuralNetworkBase<T> neuralNetwork)
        {
            neuralNetwork.EnableMemoryManagement(_memoryConfig);
        }
    }

    private static (TInput X, TOutput Y, List<(int ClientId, int StartRow, int SampleCount)> ClientRanges)
        BuildAggregatedDatasetFromClientData(IReadOnlyDictionary<int, FederatedClientDataset<TInput, TOutput>> clientData)
    {
        if (clientData is null)
        {
            throw new ArgumentNullException(nameof(clientData));
        }

        if (clientData.Count == 0)
        {
            throw new ArgumentException("Federated client data cannot be empty.", nameof(clientData));
        }

        var orderedClients = clientData.OrderBy(kvp => kvp.Key).ToList();

        int featureCount = -1;
        foreach (var dataset in orderedClients.Select(kvp => kvp.Value))
        {
            if (dataset is null)
            {
                throw new ArgumentException("Federated client data cannot contain null datasets.", nameof(clientData));
            }

            if (dataset.SampleCount <= 0)
            {
                continue;
            }

            var featuresMatrix = ConversionsHelper.ConvertToMatrix<T, TInput>(dataset.Features);
            featureCount = featuresMatrix.Columns;
            break;
        }

        if (featureCount < 0)
        {
            featureCount = 0;
        }

        int totalSamples = 0;
        foreach (var kvp in orderedClients)
        {
            var dataset = kvp.Value;
            if (dataset is null)
            {
                throw new ArgumentException("Federated client data cannot contain null datasets.", nameof(clientData));
            }

            if (dataset.SampleCount < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(clientData), "Federated client datasets must have non-negative SampleCount values.");
            }

            totalSamples += dataset.SampleCount;
        }

        if (totalSamples == 0)
        {
            throw new ArgumentException(
                "Federated client data contains no samples. At least one client must provide SampleCount > 0.",
                nameof(clientData));
        }

        var aggregatedMatrix = new Matrix<T>(totalSamples, featureCount);
        var aggregatedVector = new Vector<T>(totalSamples);
        var ranges = new List<(int ClientId, int StartRow, int SampleCount)>(orderedClients.Count);

        int row = 0;
        foreach (var kvp in orderedClients)
        {
            int clientId = kvp.Key;
            var dataset = kvp.Value;

            if (dataset is null)
            {
                throw new ArgumentException("Federated client data cannot contain null datasets.", nameof(clientData));
            }

            int sampleCount = dataset.SampleCount;
            if (sampleCount < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(clientData), "Federated client datasets must have non-negative SampleCount values.");
            }

            ranges.Add((clientId, row, sampleCount));

            if (sampleCount == 0)
            {
                continue;
            }

            var featuresMatrix = ConversionsHelper.ConvertToMatrix<T, TInput>(dataset.Features);
            var labelsVector = ConversionsHelper.ConvertToVector<T, TOutput>(dataset.Labels);

            if (featuresMatrix.Rows != sampleCount || labelsVector.Length != sampleCount)
            {
                throw new InvalidOperationException(
                    $"Federated client dataset for clientId={clientId} has inconsistent dimensions. " +
                    $"Expected SampleCount={sampleCount}, got X rows={featuresMatrix.Rows} and y length={labelsVector.Length}.");
            }

            if (featuresMatrix.Columns != featureCount)
            {
                throw new InvalidOperationException(
                    $"Federated client dataset for clientId={clientId} has inconsistent feature count. " +
                    $"Expected {featureCount} but found {featuresMatrix.Columns}.");
            }

            for (int i = 0; i < sampleCount; i++)
            {
                aggregatedMatrix.SetRow(row, featuresMatrix.GetRow(i));
                aggregatedVector[row] = labelsVector[i];
                row++;
            }
        }

        if (row != totalSamples)
        {
            throw new InvalidOperationException("Aggregated federated dataset construction produced an unexpected sample count.");
        }

        return (ConvertMatrixToInputType(aggregatedMatrix), ConvertVectorToOutputType(aggregatedVector), ranges);
    }

    private static Dictionary<int, FederatedClientDataset<TInput, TOutput>> CreateFederatedClientPartitionsFromClientRanges(
        TInput xAll,
        TOutput yAll,
        IReadOnlyList<(int ClientId, int StartRow, int SampleCount)> clientRanges)
    {
        if (clientRanges is null)
        {
            throw new ArgumentNullException(nameof(clientRanges));
        }

        if (clientRanges.Count == 0)
        {
            throw new ArgumentException("Federated client ranges cannot be empty.", nameof(clientRanges));
        }

        var xMatrix = ConversionsHelper.ConvertToMatrix<T, TInput>(xAll);
        var yVector = ConversionsHelper.ConvertToVector<T, TOutput>(yAll);

        if (xMatrix.Rows != yVector.Length)
        {
            throw new ArgumentException("Federated client range partitioning requires X row count to match y length.");
        }

        int expectedTotal = 0;
        foreach (var range in clientRanges)
        {
            expectedTotal += range.SampleCount;
        }

        if (expectedTotal != xMatrix.Rows)
        {
            throw new InvalidOperationException(
                "Federated client range partitioning requires preprocessing to preserve the total number of samples. " +
                $"Expected {expectedTotal} samples from client partitions but preprocessing produced {xMatrix.Rows}.");
        }

        var partitions = new Dictionary<int, FederatedClientDataset<TInput, TOutput>>(clientRanges.Count);

        foreach (var (clientId, startRow, sampleCount) in clientRanges)
        {
            if (sampleCount < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(clientRanges), "SampleCount must be non-negative.");
            }

            if (startRow < 0 || startRow + sampleCount > xMatrix.Rows)
            {
                throw new ArgumentOutOfRangeException(nameof(clientRanges), "Client range is outside the dataset bounds.");
            }

            var xClientMatrix = new Matrix<T>(sampleCount, xMatrix.Columns);
            var yClientVector = new Vector<T>(sampleCount);

            for (int i = 0; i < sampleCount; i++)
            {
                int sourceRow = startRow + i;
                xClientMatrix.SetRow(i, xMatrix.GetRow(sourceRow));
                yClientVector[i] = yVector[sourceRow];
            }

            var xClient = ConvertMatrixToInputType(xClientMatrix);
            var yClient = ConvertVectorToOutputType(yClientVector);

            partitions[clientId] = new FederatedClientDataset<TInput, TOutput>(xClient, yClient, sampleCount);
        }

        return partitions;
    }

    private static Dictionary<int, FederatedClientDataset<TInput, TOutput>> CreateFederatedClientPartitions(
        TInput xTrain,
        TOutput yTrain,
        int numberOfClients,
        int? randomSeed)
    {
        if (numberOfClients <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numberOfClients), "Number of clients must be positive.");
        }

        var xMatrix = ConversionsHelper.ConvertToMatrix<T, TInput>(xTrain);
        var yVector = ConversionsHelper.ConvertToVector<T, TOutput>(yTrain);

        if (xMatrix.Rows != yVector.Length)
        {
            throw new ArgumentException("Federated partitioning requires X row count to match y length.");
        }

        if (numberOfClients > xMatrix.Rows)
        {
            throw new ArgumentOutOfRangeException(
                nameof(numberOfClients),
                "NumberOfClients must not exceed the number of training samples when creating federated partitions.");
        }

        var indices = Enumerable.Range(0, xMatrix.Rows).ToList();
        var rng = randomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(randomSeed.Value)
            : RandomHelper.CreateSecureRandom();
        ShuffleInPlace(indices, rng);

        var clientIndices = new List<int>[numberOfClients];
        for (int i = 0; i < numberOfClients; i++)
        {
            clientIndices[i] = new List<int>();
        }

        for (int i = 0; i < indices.Count; i++)
        {
            int clientId = i % numberOfClients;
            clientIndices[clientId].Add(indices[i]);
        }

        var partitions = new Dictionary<int, FederatedClientDataset<TInput, TOutput>>(numberOfClients);
        for (int clientId = 0; clientId < numberOfClients; clientId++)
        {
            var rows = clientIndices[clientId];
            rows.Sort();

            var xClientMatrix = xMatrix.GetRows(rows);
            var yClientVector = new Vector<T>(rows.Count);
            for (int i = 0; i < rows.Count; i++)
            {
                yClientVector[i] = yVector[rows[i]];
            }

            var xClient = ConvertMatrixToInputType(xClientMatrix);
            var yClient = ConvertVectorToOutputType(yClientVector);

            partitions[clientId] = new FederatedClientDataset<TInput, TOutput>(xClient, yClient, rows.Count);
        }

        return partitions;
    }

    private static void ShuffleInPlace<TItem>(IList<TItem> list, Random random)
    {
        for (int i = list.Count - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (list[i], list[j]) = (list[j], list[i]);
        }
    }

    private static TInput ConvertMatrixToInputType(Matrix<T> matrix)
    {
        if (matrix is TInput typedMatrix)
        {
            return typedMatrix;
        }

        if (typeof(TInput) == typeof(Tensor<T>))
        {
            var tensor = Tensor<T>.FromRowMatrix(matrix);
            if (tensor is TInput typedTensor)
            {
                return typedTensor;
            }
        }

        throw new InvalidOperationException($"Federated learning currently supports TInput of Matrix<T> or Tensor<T>. Got {typeof(TInput).Name}.");
    }

    private static TOutput ConvertVectorToOutputType(Vector<T> vector)
    {
        if (vector is TOutput typedVector)
        {
            return typedVector;
        }

        if (typeof(TOutput) == typeof(Tensor<T>))
        {
            var tensor = Tensor<T>.FromVector(vector);
            if (tensor is TOutput typedTensor)
            {
                return typedTensor;
            }
        }

        throw new InvalidOperationException($"Federated learning currently supports TOutput of Vector<T> or Tensor<T>. Got {typeof(TOutput).Name}.");
    }

    private static IAggregationStrategy<IFullModel<T, TInput, TOutput>> CreateDefaultFederatedAggregationStrategy(FederatedLearningOptions options)
    {
        switch (options.AggregationStrategy)
        {
            case FederatedAggregationStrategy.FedAvg:
                return new AiDotNet.FederatedLearning.Aggregators.FedAvgFullModelAggregationStrategy<T, TInput, TOutput>();

            case FederatedAggregationStrategy.FedProx:
                return new AiDotNet.FederatedLearning.Aggregators.FedProxFullModelAggregationStrategy<T, TInput, TOutput>(options.ProximalMu);

            case FederatedAggregationStrategy.FedBN:
                return new AiDotNet.FederatedLearning.Aggregators.FedBNFullModelAggregationStrategy<T, TInput, TOutput>();

            case FederatedAggregationStrategy.Median:
                return new AiDotNet.FederatedLearning.Aggregators.MedianFullModelAggregationStrategy<T, TInput, TOutput>();

            case FederatedAggregationStrategy.TrimmedMean:
            {
                var robust = options.RobustAggregation ?? new RobustAggregationOptions();
                return new AiDotNet.FederatedLearning.Aggregators.TrimmedMeanFullModelAggregationStrategy<T, TInput, TOutput>(robust.TrimFraction);
            }

            case FederatedAggregationStrategy.WinsorizedMean:
            {
                var robust = options.RobustAggregation ?? new RobustAggregationOptions();
                return new AiDotNet.FederatedLearning.Aggregators.WinsorizedMeanFullModelAggregationStrategy<T, TInput, TOutput>(robust.TrimFraction);
            }

            case FederatedAggregationStrategy.Rfa:
            {
                var robust = options.RobustAggregation ?? new RobustAggregationOptions();
                return AiDotNet.FederatedLearning.Aggregators.RfaFullModelAggregationStrategy<T, TInput, TOutput>.FromOptions(robust);
            }

            case FederatedAggregationStrategy.Krum:
            {
                var robust = options.RobustAggregation ?? new RobustAggregationOptions();
                return new AiDotNet.FederatedLearning.Aggregators.KrumFullModelAggregationStrategy<T, TInput, TOutput>(robust.ByzantineClientCount);
            }

            case FederatedAggregationStrategy.MultiKrum:
            {
                var robust = options.RobustAggregation ?? new RobustAggregationOptions();
                return new AiDotNet.FederatedLearning.Aggregators.MultiKrumFullModelAggregationStrategy<T, TInput, TOutput>(
                    robust.ByzantineClientCount,
                    robust.MultiKrumSelectionCount,
                    robust.UseClientWeightsWhenAveragingSelectedUpdates);
            }

            case FederatedAggregationStrategy.Bulyan:
            {
                var robust = options.RobustAggregation ?? new RobustAggregationOptions();
                return new AiDotNet.FederatedLearning.Aggregators.BulyanFullModelAggregationStrategy<T, TInput, TOutput>(
                    robust.ByzantineClientCount,
                    robust.UseClientWeightsWhenAveragingSelectedUpdates);
            }

            default:
                throw new InvalidOperationException($"Unsupported federated aggregation strategy '{options.AggregationStrategy}'.");
        }
    }

    private static (IFullModel<T, TInput, TOutput> Model, IOptimizer<T, TInput, TOutput> Optimizer) CreateDistributedPair(
        IFullModel<T, TInput, TOutput> distributedModel,
        IOptimizer<T, TInput, TOutput> distributedOptimizer)
    {
        return (distributedModel, distributedOptimizer);
    }

    /// <summary>
    /// Computes deep ensemble members and attaches them to the result when deep ensemble uncertainty quantification is enabled.
    /// </summary>
    /// <param name="result">The prediction model result to update.</param>
    /// <param name="deepEnsembleTemplate">A template model used to create additional ensemble members.</param>
    /// <param name="optimizationInputData">Optimization/training data for the ensemble members.</param>
    /// <param name="templateOptimizer">The optimizer used for the main model, used as a template for ensemble member optimizers.</param>
    /// <param name="options">Uncertainty quantification configuration.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> A deep ensemble trains multiple similar models and combines their predictions. If the models disagree,
    /// it usually means the prediction is less certain.</para>
    /// <para>
    /// This method reuses the best solution from the primary optimization run (if available) and then trains additional members using:
    /// - Optional bootstrapping (sampling training rows with replacement)
    /// - Optional parameter perturbation (small Gaussian noise)
    /// </para>
    /// </remarks>
    private static void TryComputeAndAttachDeepEnsembleModels(
        AiModelResult<T, TInput, TOutput> result,
        IFullModel<T, TInput, TOutput>? deepEnsembleTemplate,
        OptimizationInputData<T, TInput, TOutput> optimizationInputData,
        IOptimizer<T, TInput, TOutput> templateOptimizer,
        UncertaintyQuantificationOptions? options)
    {
        if (options is not { Enabled: true, Method: UncertaintyQuantificationMethod.DeepEnsemble })
        {
            return;
        }

        if (deepEnsembleTemplate == null)
        {
            return;
        }

        var ensembleSize = Math.Max(2, options.DeepEnsembleSize);
        var members = new List<IFullModel<T, TInput, TOutput>>(capacity: ensembleSize);

        if (result.OptimizationResult.BestSolution != null)
        {
            members.Add(result.OptimizationResult.BestSolution);
        }

        var baseSeed = options.RandomSeed ?? Environment.TickCount;

        for (int memberIndex = members.Count; memberIndex < ensembleSize; memberIndex++)
        {
            var memberModel = deepEnsembleTemplate.DeepCopy();

            PerturbInitialParametersIfSupported(memberModel, baseSeed, memberIndex, options.DeepEnsembleInitialNoiseStdDev);

            var memberOptimizer = CreateOptimizerForEnsembleMember(memberModel, templateOptimizer);
            memberOptimizer.Reset();

            // Ensure the optimizer has a model set before calling Optimize/InitializeRandomSolution
            memberOptimizer.SetModel(memberModel);

            var memberInputData = CreateDeepEnsembleMemberOptimizationInputData(optimizationInputData, baseSeed, memberIndex);
            var memberResult = memberOptimizer.Optimize(memberInputData);
            if (memberResult.BestSolution != null)
            {
                members.Add(memberResult.BestSolution);
            }
        }

        if (members.Count > 0)
        {
            result.SetDeepEnsembleModels(members);
        }
    }

    /// <summary>
    /// Creates per-member optimization input data for deep ensembles.
    /// </summary>
    /// <param name="baseInputData">The baseline input data.</param>
    /// <param name="baseSeed">Seed used to deterministically vary members.</param>
    /// <param name="memberIndex">The ensemble member index.</param>
    /// <returns>Optimization input data for the ensemble member.</returns>
    /// <remarks>
    /// <para>
    /// This optionally bootstraps training data to encourage diversity across members while keeping validation/test stable.
    /// </para>
    /// </remarks>
    private static OptimizationInputData<T, TInput, TOutput> CreateDeepEnsembleMemberOptimizationInputData(
        OptimizationInputData<T, TInput, TOutput> baseInputData,
        int baseSeed,
        int memberIndex)
    {
        var rng = RandomHelper.CreateSeededRandom(unchecked(baseSeed + (memberIndex + 1) * 10007));

        if (TryBootstrapTrainingData(baseInputData.XTrain, baseInputData.YTrain, rng, out var bootstrappedXTrain, out var bootstrappedYTrain))
        {
            return new OptimizationInputData<T, TInput, TOutput>
            {
                XTrain = bootstrappedXTrain,
                YTrain = bootstrappedYTrain,
                XValidation = baseInputData.XValidation,
                YValidation = baseInputData.YValidation,
                XTest = baseInputData.XTest,
                YTest = baseInputData.YTest
            };
        }

        return new OptimizationInputData<T, TInput, TOutput>
        {
            XTrain = baseInputData.XTrain,
            YTrain = baseInputData.YTrain,
            XValidation = baseInputData.XValidation,
            YValidation = baseInputData.YValidation,
            XTest = baseInputData.XTest,
            YTest = baseInputData.YTest
        };
    }

    /// <summary>
    /// Attempts to bootstrap the training data (sample with replacement) for deep-ensemble diversity.
    /// </summary>
    /// <param name="xTrain">Training inputs.</param>
    /// <param name="yTrain">Training targets.</param>
    /// <param name="rng">Random number generator.</param>
    /// <param name="bootstrappedXTrain">Bootstrapped inputs if supported.</param>
    /// <param name="bootstrappedYTrain">Bootstrapped targets if supported.</param>
    /// <returns>True if bootstrapping was applied; otherwise false.</returns>
    /// <remarks>
    /// <para>
    /// This method is best-effort and only supports common vector/matrix/tensor training data representations.
    /// </para>
    /// </remarks>
    private static bool TryBootstrapTrainingData(
        TInput xTrain,
        TOutput yTrain,
        Random rng,
        out TInput bootstrappedXTrain,
        out TOutput bootstrappedYTrain)
    {
        if (xTrain is Matrix<T> xTrainMatrix)
        {
            var sampleCount = xTrainMatrix.Rows;
            if (sampleCount <= 0)
            {
                bootstrappedXTrain = xTrain;
                bootstrappedYTrain = yTrain;
                return false;
            }

            var indices = new int[sampleCount];
            for (int i = 0; i < sampleCount; i++)
            {
                indices[i] = rng.Next(sampleCount);
            }

            var xBoot = xTrainMatrix.GetRows(indices);

            if (yTrain is Vector<T> yTrainVector && yTrainVector.Length == sampleCount)
            {
                var yBoot = yTrainVector.GetElements(indices);
                bootstrappedXTrain = (TInput)(object)xBoot;
                bootstrappedYTrain = (TOutput)(object)yBoot;
                return true;
            }

            if (yTrain is Matrix<T> yTrainMatrix && yTrainMatrix.Rows == sampleCount)
            {
                var yBoot = yTrainMatrix.GetRows(indices);
                bootstrappedXTrain = (TInput)(object)xBoot;
                bootstrappedYTrain = (TOutput)(object)yBoot;
                return true;
            }
        }

        if (xTrain is Tensor<T> xTrainTensor && xTrainTensor.Rank >= 2)
        {
            var sampleCount = xTrainTensor.Shape[0];
            if (sampleCount <= 0)
            {
                bootstrappedXTrain = xTrain;
                bootstrappedYTrain = yTrain;
                return false;
            }

            var indices = new int[sampleCount];
            for (int i = 0; i < sampleCount; i++)
            {
                indices[i] = rng.Next(sampleCount);
            }

            var xSlices = indices.Select(i => xTrainTensor.GetSlice(i)).ToArray();
            var xBoot = Tensor<T>.Stack(xSlices, axis: 0);

            if (yTrain is Tensor<T> yTrainTensor &&
                yTrainTensor.Rank >= 2 &&
                yTrainTensor.Shape[0] == sampleCount)
            {
                var ySlices = indices.Select(i => yTrainTensor.GetSlice(i)).ToArray();
                var yBoot = Tensor<T>.Stack(ySlices, axis: 0);
                bootstrappedXTrain = (TInput)(object)xBoot;
                bootstrappedYTrain = (TOutput)(object)yBoot;
                return true;
            }
        }

        bootstrappedXTrain = xTrain;
        bootstrappedYTrain = yTrain;
        return false;
    }

    /// <summary>
    /// Perturbs initial model parameters to avoid ensemble member collapse to identical solutions.
    /// </summary>
    /// <param name="model">The model to perturb.</param>
    /// <param name="baseSeed">Base seed for deterministic perturbation.</param>
    /// <param name="memberIndex">Ensemble member index.</param>
    /// <param name="noiseStdDev">Standard deviation of Gaussian noise to add.</param>
    /// <remarks>
    /// <para>
    /// The perturbation is only applied when the model supports parameter get/set via <see cref="IParameterizable{T,TInput,TOutput}"/>.
    /// </para>
    /// </remarks>
    private static void PerturbInitialParametersIfSupported(
        IFullModel<T, TInput, TOutput> model,
        int baseSeed,
        int memberIndex,
        double noiseStdDev)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        if (model is not IParameterizable<T, TInput, TOutput> parameterizable)
        {
            return;
        }

        if (noiseStdDev <= 0)
        {
            return;
        }

        var parameters = parameterizable.GetParameters();
        if (parameters.Length == 0)
        {
            return;
        }

        var rng = RandomHelper.CreateSeededRandom(unchecked(baseSeed + (memberIndex + 1) * 10007));
        var perturbed = new Vector<T>(parameters.Length);
        for (int i = 0; i < parameters.Length; i++)
        {
            var noise = NextGaussian(rng, mean: 0.0, stdDev: noiseStdDev);
            perturbed[i] = numOps.Add(parameters[i], numOps.FromDouble(noise));
        }

        parameterizable.SetParameters(perturbed);
    }

    /// <summary>
    /// Generates a Gaussian random value using the Box-Muller transform.
    /// </summary>
    /// <param name="rng">Random source.</param>
    /// <param name="mean">Gaussian mean.</param>
    /// <param name="stdDev">Gaussian standard deviation.</param>
    /// <returns>A normally distributed random value.</returns>
    private static double NextGaussian(Random rng, double mean, double stdDev)
    {
        return rng.NextGaussian(mean, stdDev);
    }

    /// <summary>
    /// Creates an optimizer instance for an ensemble member based on the template optimizer type and options.
    /// </summary>
    /// <param name="model">The ensemble member model.</param>
    /// <param name="templateOptimizer">The optimizer used as a template.</param>
    /// <returns>An optimizer instance that targets <paramref name="model"/>.</returns>
    /// <remarks>
    /// <para>
    /// This is best-effort and supports common optimizer constructors (model + options, or model only).
    /// </para>
    /// </remarks>
    private static IOptimizer<T, TInput, TOutput> CreateOptimizerForEnsembleMember(
        IFullModel<T, TInput, TOutput> model,
        IOptimizer<T, TInput, TOutput> templateOptimizer)
    {
        var optimizerType = templateOptimizer.GetType();
        var options = templateOptimizer.GetOptions();

        foreach (var ctor in optimizerType.GetConstructors())
        {
            var parameters = ctor.GetParameters();
            if (parameters.Length == 2 &&
                parameters[0].ParameterType.IsInstanceOfType(model) &&
                parameters[1].ParameterType.IsInstanceOfType(options))
            {
                return (IOptimizer<T, TInput, TOutput>)ctor.Invoke([model, options]);
            }

            if (parameters.Length == 1 &&
                parameters[0].ParameterType.IsInstanceOfType(model))
            {
                return (IOptimizer<T, TInput, TOutput>)ctor.Invoke([model]);
            }
        }

        if (templateOptimizer is NormalOptimizer<T, TInput, TOutput>)
        {
            return new NormalOptimizer<T, TInput, TOutput>(model);
        }

        throw new InvalidOperationException(
            $"Unable to construct a deep ensemble optimizer of type '{optimizerType.FullName}'. " +
            $"Expected a constructor with signature ({typeof(IFullModel<T, TInput, TOutput>).Name}, {options?.GetType().Name ?? "null"}) or ({typeof(IFullModel<T, TInput, TOutput>).Name}).");
    }

    private static void ApplyUncertaintyQuantificationIfConfigured(
        IFullModel<T, TInput, TOutput>? model,
        UncertaintyQuantificationOptions? options)
    {
        if (model == null)
        {
            return;
        }

        if (options is not { Enabled: true })
        {
            return;
        }

        var method = options.Method == UncertaintyQuantificationMethod.Auto
            ? UncertaintyQuantificationMethod.MonteCarloDropout
            : options.Method;

        if (method != UncertaintyQuantificationMethod.MonteCarloDropout)
        {
            return;
        }

        if (model is not NeuralNetworks.NeuralNetworkBase<T> neuralNetworkModel)
        {
            throw new InvalidOperationException(
                "Uncertainty quantification is currently supported for neural network models only. " +
                "Use a NeuralNetworkBase<T> subclass to enable Monte Carlo Dropout uncertainty estimation.");
        }

        var injectedCount = TryInjectMonteCarloDropoutLayers(neuralNetworkModel, options);
        if (injectedCount == 0)
        {
            System.Diagnostics.Debug.WriteLine(
                "Warning: Monte Carlo Dropout was enabled but no dropout layers were injected automatically. " +
                "This can happen if the network has no suitable activation layers or uses a non-standard architecture. " +
                "Consider inserting MCDropoutLayer explicitly in your network definition.");
        }
    }

    private static int TryInjectMonteCarloDropoutLayers(
        NeuralNetworks.NeuralNetworkBase<T> neuralNetworkModel,
        UncertaintyQuantificationOptions options)
    {
        var layers = neuralNetworkModel.LayersReadOnly;
        if (layers.OfType<MCDropoutLayer<T>>().Any())
        {
            return -1;
        }

        if (options.MonteCarloDropoutRate <= 0 || options.MonteCarloDropoutRate >= 1)
        {
            throw new ArgumentException("MonteCarloDropoutRate must be between 0 and 1.", nameof(options));
        }

        var injectedCount = 0;
        for (int i = 0; i < layers.Count - 1; i++)
        {
            if (layers[i] is not ActivationLayer<T>)
            {
                continue;
            }

            if (i >= layers.Count - 2)
            {
                continue;
            }

            if (layers[i + 1] is DropoutLayer<T> || layers[i + 1] is MCDropoutLayer<T>)
            {
                continue;
            }

            int? seed = options.RandomSeed.HasValue ? options.RandomSeed.Value + i : (int?)null;
            neuralNetworkModel.InsertLayerIntoCollection(i + 1, new MCDropoutLayer<T>(options.MonteCarloDropoutRate, mcMode: false, randomSeed: seed));
            injectedCount++;
            i++;
        }

        return injectedCount;
    }


    /// <summary>
    /// Applies quantization to the model if configured.
    /// </summary>
    /// <param name="model">The model to quantize.</param>
    /// <param name="config">The quantization configuration.</param>
    /// <param name="calibrationData">Optional calibration data for advanced quantization strategies.</param>
    /// <returns>A tuple containing the quantized model and QuantizationInfo, or (null, null) if quantization was not configured.</returns>
    private (IFullModel<T, TInput, TOutput>? QuantizedModel, QuantizationInfo? Info) ApplyQuantizationIfConfigured(
        IFullModel<T, TInput, TOutput> model,
        QuantizationConfig? config,
        IEnumerable<TInput>? calibrationData = null)
    {
        // Check if quantization is enabled (Mode != None indicates enabled)
        if (config == null || config.Mode == QuantizationMode.None)
        {
            return (null, null);
        }

        // Convert user-facing config to internal configuration
        var internalConfig = config.ToQuantizationConfiguration();

        // Create the appropriate quantizer based on strategy
        Deployment.Optimization.Quantization.IQuantizer<T, TInput, TOutput> quantizer = internalConfig.Strategy switch
        {
            QuantizationStrategy.GPTQ => new Deployment.Optimization.Quantization.Strategies.GPTQQuantizer<T, TInput, TOutput>(internalConfig),
            QuantizationStrategy.AWQ => new Deployment.Optimization.Quantization.Strategies.AWQQuantizer<T, TInput, TOutput>(internalConfig),
            QuantizationStrategy.SmoothQuant => new Deployment.Optimization.Quantization.Strategies.SmoothQuantQuantizer<T, TInput, TOutput>(internalConfig),
            QuantizationStrategy.SpinQuant => new Deployment.Optimization.Quantization.Strategies.SpinQuantQuantizer<T, TInput, TOutput>(internalConfig),
            QuantizationStrategy.QuIPSharp => new Deployment.Optimization.Quantization.Strategies.QuIPSharpQuantizer<T, TInput, TOutput>(internalConfig),
            QuantizationStrategy.MinMax or QuantizationStrategy.Dynamic => internalConfig.Mode switch
            {
                QuantizationMode.Int8 => new Deployment.Optimization.Quantization.Int8Quantizer<T, TInput, TOutput>(),
                QuantizationMode.Float16 => new Deployment.Optimization.Quantization.Float16Quantizer<T, TInput, TOutput>(),
                QuantizationMode.Float32 => throw new NotSupportedException("Float32 mode represents no quantization. Use QuantizationMode.None instead."),
                QuantizationMode.Mixed => throw new NotSupportedException("Mixed precision requires a specific strategy like GPTQ or AWQ."),
                _ => throw new NotSupportedException($"Quantization mode {internalConfig.Mode} is not supported with {internalConfig.Strategy} strategy. " +
                    "Use a specific strategy like GPTQ or AWQ for advanced quantization, or use Int8/Float16 mode.")
            },
            _ => new Deployment.Optimization.Quantization.Int8Quantizer<T, TInput, TOutput>()
        };

        // Get model parameters for size calculation
        var parameters = InterfaceGuard.Parameterizable(model).GetParameters();
        // Estimate parameter size based on type: 8 bytes for double, 4 for float/int, 2 for half
        int bytesPerParameter = typeof(T) == typeof(float) ? 4
            : typeof(T) == typeof(double) ? 8
            : typeof(T) == typeof(Half) ? 2
            : 8; // Default to 8 bytes for unknown types
        // Cast to long first to prevent int overflow for large models
        long originalSizeBytes = (long)parameters.Length * bytesPerParameter;

        // Calibrate if calibration data is provided and strategy requires it
        if (calibrationData != null && internalConfig.CalibrationMethod != CalibrationMethod.None)
        {
            try
            {
                quantizer.Calibrate(model, calibrationData);
            }
            catch (Exception ex)
            {
                // Log detailed error info to help diagnose calibration issues
                Console.WriteLine($"Warning: Calibration failed ({ex.GetType().Name}): {ex.Message}");
                if (ex.InnerException != null)
                {
                    Console.WriteLine($"  Inner exception: {ex.InnerException.Message}");
                }
                Console.WriteLine("Proceeding with default quantization behavior.");
            }
        }

        // QAT SIMULATION (Post-Training): If QAT is enabled, apply fake quantization once
        // to condition parameters. NOTE: This is a simplified post-training simulation, not true QAT.
        // Real QAT integrates fake quantization into the training loop across multiple epochs,
        // allowing the model to learn under quantization constraints. This simulation provides
        // some benefit by conditioning parameters but won't achieve the full accuracy of true QAT.
        // For full QAT, use the training API with quantization hooks enabled during training.
        if (internalConfig.UseQuantizationAwareTraining)
        {
            try
            {
                var qatHook = new Deployment.Optimization.Quantization.Training.QATTrainingHook<T>(internalConfig);

                // Simulate warmup completion by setting epoch to warmup value
                qatHook.OnEpochStart(internalConfig.QATWarmupEpochs);

                // Apply fake quantization to model parameters to condition them for quantization
                // This simulates what would happen during QAT training iterations
                var currentParams = InterfaceGuard.Parameterizable(model).GetParameters();
                var qatConditionedParams = qatHook.ApplyFakeQuantization(currentParams, "model_weights");

                // Create a new model with QAT-conditioned parameters
                model = InterfaceGuard.Parameterizable(model).WithParameters(qatConditionedParams);

                // Recalibrate after QAT conditioning — calibration stats from pre-QAT weights
                // would skew scale selection for the quantized model
                if (calibrationData is not null)
                {
                    quantizer.Calibrate(model, calibrationData);
                }

                Console.WriteLine($"QAT simulation applied using {internalConfig.QATMethod} method");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Warning: QAT simulation failed: {ex.Message}. Proceeding with standard PTQ.");
            }
        }

        // Apply quantization - this returns a NEW model with quantized parameters
        var quantizedModel = quantizer.Quantize(model, internalConfig);
        var quantizedParameters = InterfaceGuard.Parameterizable(quantizedModel).GetParameters();

        // Calculate actual quantized size based on bit width
        // For sub-byte quantization (4-bit), we need to account for packing
        long quantizedSizeBytes = ((long)quantizedParameters.Length * internalConfig.EffectiveBitWidth + 7) / 8;

        // Build QuantizationInfo
        var info = new QuantizationInfo
        {
            IsQuantized = true,
            Mode = internalConfig.Mode,
            Strategy = internalConfig.Strategy,
            Granularity = internalConfig.Granularity,
            BitWidth = internalConfig.EffectiveBitWidth,
            OriginalSizeBytes = originalSizeBytes,
            QuantizedSizeBytes = quantizedSizeBytes,
            TotalParameters = parameters.Length,
            QuantizedParameters = quantizedParameters.Length,
            CalibrationMethod = internalConfig.CalibrationMethod,
            GroupSize = internalConfig.Granularity == QuantizationGranularity.PerGroup ? internalConfig.GroupSize : 128,
            ActivationsQuantized = internalConfig.QuantizeActivations,
            UsedQAT = internalConfig.UseQuantizationAwareTraining,
            QATMethod = internalConfig.UseQuantizationAwareTraining ? internalConfig.QATMethod : null
        };

        return (quantizedModel, info);
    }
}
