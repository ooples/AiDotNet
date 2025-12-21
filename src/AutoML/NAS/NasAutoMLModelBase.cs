using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.AutoML.SearchSpace;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AutoML.NAS
{
    /// <summary>
    /// Base class for NAS-based AutoML models.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public abstract class NasAutoMLModelBase<T> : AutoMLModelBase<T, Tensor<T>, Tensor<T>>
    {
        /// <summary>
        /// Gets the numeric operations provider for <typeparamref name="T"/>.
        /// </summary>
        protected abstract INumericOperations<T> NumOps { get; }

        /// <summary>
        /// Gets the NAS search space.
        /// </summary>
        protected abstract SearchSpaceBase<T> NasSearchSpace { get; }

        /// <summary>
        /// Gets the number of nodes to search over.
        /// </summary>
        protected abstract int NasNumNodes { get; }

        /// <summary>
        /// Gets the best architecture found by the NAS search.
        /// </summary>
        public Architecture<T>? BestArchitecture { get; protected set; }

        /// <summary>
        /// Runs the NAS search and returns the best model found.
        /// </summary>
        public override async Task<IFullModel<T, Tensor<T>, Tensor<T>>> SearchAsync(
            Tensor<T> inputs,
            Tensor<T> targets,
            Tensor<T> validationInputs,
            Tensor<T> validationTargets,
            TimeSpan timeLimit,
            CancellationToken cancellationToken = default)
        {
            Status = AutoMLStatus.Running;

            try
            {
                cancellationToken.ThrowIfCancellationRequested();

                var searchStopwatch = System.Diagnostics.Stopwatch.StartNew();
                var architecture = await Task.Run(
                    () => SearchArchitecture(inputs, targets, validationInputs, validationTargets, timeLimit, cancellationToken),
                    cancellationToken);
                searchStopwatch.Stop();

                BestArchitecture = architecture;

                var model = new SuperNet<T>(NasSearchSpace, numNodes: NasNumNodes);
                ApplyArchitectureToModel(model, architecture);

                // Initialize model weights deterministically for downstream training/evaluation.
                _ = model.Predict(inputs);

                BestModel = model;

                var score = await EvaluateModelAsync(model, validationInputs, validationTargets);
                await ReportTrialResultAsync(
                    new Dictionary<string, object> { ["Architecture"] = architecture.GetDescription() },
                    score,
                    searchStopwatch.Elapsed);

                Status = AutoMLStatus.Completed;
                return model;
            }
            catch (OperationCanceledException)
            {
                Status = AutoMLStatus.Cancelled;
                throw;
            }
            catch (Exception)
            {
                Status = AutoMLStatus.Failed;
                throw;
            }
        }

        /// <summary>
        /// Suggests the next trial parameters.
        /// </summary>
        public override Task<Dictionary<string, object>> SuggestNextTrialAsync()
        {
            return Task.FromResult(new Dictionary<string, object>());
        }

        protected override Task<IFullModel<T, Tensor<T>, Tensor<T>>> CreateModelAsync(ModelType modelType, Dictionary<string, object> parameters)
        {
            return Task.FromResult((IFullModel<T, Tensor<T>, Tensor<T>>)new SuperNet<T>(NasSearchSpace, numNodes: NasNumNodes));
        }

        protected override Dictionary<string, ParameterRange> GetDefaultSearchSpace(ModelType modelType)
        {
            return new Dictionary<string, ParameterRange>();
        }

        /// <summary>
        /// Performs algorithm-specific architecture search.
        /// </summary>
        protected abstract Architecture<T> SearchArchitecture(
            Tensor<T> inputs,
            Tensor<T> targets,
            Tensor<T> validationInputs,
            Tensor<T> validationTargets,
            TimeSpan timeLimit,
            CancellationToken cancellationToken);

        protected virtual void ApplyArchitectureToModel(SuperNet<T> model, Architecture<T> architecture)
        {
            if (NasSearchSpace.Operations == null || NasSearchSpace.Operations.Count == 0)
            {
                return;
            }

            var opIndex = new Dictionary<string, int>(StringComparer.Ordinal);
            for (int i = 0; i < NasSearchSpace.Operations.Count; i++)
            {
                opIndex[NasSearchSpace.Operations[i]] = i;
            }

            var alphas = model.GetArchitectureParameters();
            T low = NumOps.FromDouble(-10.0);
            T high = NumOps.FromDouble(10.0);
            int defaultOpIdx = opIndex.TryGetValue("identity", out int identityIdx) ? identityIdx : 0;

            for (int nodeIdx = 0; nodeIdx < alphas.Count; nodeIdx++)
            {
                var alpha = alphas[nodeIdx];
                for (int row = 0; row < alpha.Rows; row++)
                {
                    for (int col = 0; col < alpha.Columns; col++)
                    {
                        alpha[row, col] = low;
                    }

                    if (defaultOpIdx >= 0 && defaultOpIdx < alpha.Columns)
                    {
                        alpha[row, defaultOpIdx] = high;
                    }
                }
            }

            foreach (var (toNode, fromNode, operation) in architecture.Operations)
            {
                int nodeIdx = toNode - 1;
                if (nodeIdx < 0 || nodeIdx >= alphas.Count)
                {
                    continue;
                }

                if (!opIndex.TryGetValue(operation, out int opIdx))
                {
                    continue;
                }

                var alpha = alphas[nodeIdx];
                if (fromNode < 0 || fromNode >= alpha.Rows)
                {
                    continue;
                }

                for (int col = 0; col < alpha.Columns; col++)
                {
                    alpha[fromNode, col] = low;
                }

                alpha[fromNode, opIdx] = high;
            }
        }
    }
}
