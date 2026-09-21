using AiDotNet.Models.Options;
using AiDotNet.Preprocessing;
using AiDotNet.Models.Results;

namespace AiDotNet;

public partial class AiModelBuilder<T, TInput, TOutput>
{
    /// <summary>
    /// Wraps an already-trained model in a result, without training it.
    /// </summary>
    /// <returns>
    /// A result over the configured model, supporting the same prediction surface a built one does.
    /// </returns>
    /// <remarks>
    /// <para>
    /// <see cref="Build(TInput, TOutput)"/> trains. That is the right default, but it leaves nowhere to
    /// go for a model that arrives ready to use: weights loaded from disk, a model another algorithm
    /// returned, or one constructed from parameters you already have. Because
    /// <see cref="AiModelResult{T, TInput, TOutput}"/>'s constructors are internal, such a model could
    /// not reach the facade at all except by being retrained — which for an adapted or loaded model
    /// destroys the thing that made it worth having.
    /// </para>
    /// <para>
    /// This is a terminal like <c>Build</c>, so everything configured for inference still applies: the
    /// preprocessing pipeline (which must already be fitted, since there is no training pass to fit it),
    /// the target and postprocessing pipelines, the tokenizer and text vectorizer, the embedding model,
    /// and the inference and JIT configuration. Training-time configuration is not applied and cannot
    /// be: there is no training. If you configured a data loader, an optimizer or cross-validation and
    /// meant to train, call <c>Build</c> instead.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Use this when the model has already learned what it needs to. Training it
    /// again on whatever data you happen to have would overwrite that.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">
    /// No model has been configured, or a preprocessing pipeline was configured but never fitted.
    /// </exception>
    /// <example>
    /// <code>
    /// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
    ///     InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 8, outputSize: 1);
    /// var pretrained = new NeuralNetwork&lt;float&gt;(architecture);
    /// // ... load weights into `pretrained` ...
    ///
    /// var result = new AiModelBuilder&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;()
    ///     .ConfigureModel(pretrained)
    ///     .BuildForInference();
    ///
    /// var prediction = result.Predict(Tensor&lt;float&gt;.CreateRandom(1, 8));
    /// </code>
    /// </example>
    public AiModelResult<T, TInput, TOutput> BuildForInference()
    {
        if (_model is null)
        {
            throw new InvalidOperationException(
                "BuildForInference needs a model. Call ConfigureModel with the already-trained model " +
                "you want to predict with.");
        }

        // A pipeline that has not been fitted cannot transform anything, and there is no training pass
        // here to fit it. Saying so now beats a confusing failure on the first Predict.
        if (_preprocessingPipeline is not null && !_preprocessingPipeline.IsFitted)
        {
            throw new InvalidOperationException(
                "BuildForInference was given a preprocessing pipeline that has not been fitted. There is " +
                "no training pass here to fit it, so either fit it against the data the model was " +
                "trained on before calling this, or drop the ConfigureDataPreprocessor call.");
        }

        var options = new AiModelResultOptions<T, TInput, TOutput>
        {
            // The result reads its model from the optimization result rather than from a Model
            // property; there is no optimization here, so this carries the model and nothing else.
            OptimizationResult = new OptimizationResult<T, TInput, TOutput> { BestSolution = _model },

            // Everything below is inference-time configuration, which applies just as much to a model
            // that arrived trained as to one this builder trained. Anything training-time is absent
            // deliberately, and the remarks say so.
            TextVectorizer = _configuredTextVectorizer,
            EmbeddingModel = _configuredEmbeddingModel,
            PreprocessingInfo = _preprocessingPipeline is not null || _targetPipeline is not null
                ? new PreprocessingInfo<T, TInput, TOutput>
                {
                    Pipeline = _preprocessingPipeline,
                    TargetPipeline = _targetPipeline,
                }
                : null,
            PostprocessingPipeline = _postprocessingPipeline,
            Tokenizer = _tokenizer,
            TokenizationConfig = _tokenizationConfig,
            InferenceOptimizationConfig = _inferenceOptimizationConfig,
            JitCompilationConfig = _jitCompilationConfig,
            JitCompiledFunction = BuildCompiledPredictFunction(_model),
            AllowNondeterminism = _allowNondeterminism,
        };

        return new AiModelResult<T, TInput, TOutput>(options);
    }
}
