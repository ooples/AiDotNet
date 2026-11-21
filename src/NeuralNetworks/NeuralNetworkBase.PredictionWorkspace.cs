using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks;

public abstract partial class NeuralNetworkBase<T>
{
    // The engine version that introduced workspace handoff returns a value-type scope. Bind its
    // concrete type once so each prediction enters and disposes it without boxing, even when nested.
    private static class PredictionWorkspace
    {
        internal delegate Tensor<T> Prediction(NeuralNetworkBase<T> model, Tensor<T> input, TensorScope? resultScope, IEngine? engine);
        internal static readonly Prediction Run = Bind();

        private static Prediction Bind()
        {
            var enter = typeof(IEngine).Assembly.GetType("AiDotNet.Tensors.Engines.InferenceWorkspaceDemandScope")?
                .GetMethod("Enter", BindingFlags.NonPublic | BindingFlags.Static, null, new[] { typeof(IEngine) }, null);
            if (enter is null)
                return static (model, input, resultScope, _) => model.CompletePrediction(input, resultScope);

            var bind = typeof(PredictionWorkspace).GetMethod(nameof(BindScope), BindingFlags.NonPublic | BindingFlags.Static)
                ?? throw new InvalidOperationException("The prediction workspace binder is missing.");
            return bind.MakeGenericMethod(enter.ReturnType).Invoke(null, new object[] { enter }) as Prediction
                ?? throw new InvalidOperationException("The prediction workspace binder returned no delegate.");
        }

        private static Prediction BindScope<TScope>(MethodInfo enterMethod) where TScope : struct, IDisposable
        {
            var enter = (Func<IEngine?, TScope>)enterMethod.CreateDelegate(typeof(Func<IEngine, TScope>));
            return (model, input, resultScope, engine) =>
            {
                using var workspace = enter(engine);
                return model.CompletePrediction(input, resultScope);
            };
        }
    }

    private Tensor<T> CompletePrediction(Tensor<T> input, TensorScope? resultScope)
    {
        var output = PredictInInferenceMode(input);
        RequireArchitectureAgreesWithOutput(output);
        // The default PredictCore records the first forward and retries weight-streaming detection; an override that
        // answers from its own path (FeedForwardNeuralNetwork's fused dense predict) never did, so detection never
        // finalized and every Predict ran it again: both SetTrainingMode transitions built and SHA-256 hashed the
        // parameter layout, about a sixth of a GPU MLP's batch-32 inference. Recorded here for every model.
        // Retried until detection finalizes (a count that threw is read again next call), with lazy-shape resolution
        // deferred as in the mode transitions: a branch this forward did not visit must not be resolved from the
        // architecture's shape instead of its eventual input.
        _firstForwardCompleted = true;
        if (!_streamingAutoDetectFinalized)
        {
            bool priorRetryDefer = _deferLazyShapeResolutionToForward;
            _deferLazyShapeResolutionToForward = true;
            try
            {
                TryAutoEnableWeightStreaming();
            }
            finally
            {
                _deferLazyShapeResolutionToForward = priorRetryDefer;
            }
        }
        return resultScope is null ? output : resultScope.Keep(output);
    }
}
