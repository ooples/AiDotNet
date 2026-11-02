using System;
using System.Linq;
using AiDotNet.Data.Structures;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Models;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.MetaLearning;

/// <summary>Vector-output tasks over 3 features with 2 classes, as the meta-learner family builds them.</summary>
internal static class AdaptedModelTasks
{
    internal const int FeatureCount = 3;
    internal const int NumWays = 2;
    internal const int EmbeddingDimension = 4;

    internal static MetaLearningTask<double, Matrix<double>, TOutput> Create<TOutput>(
        int seed, Func<double[], TOutput> labels)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var centres = Enumerable.Range(0, NumWays)
            .Select(_ => Enumerable.Range(0, FeatureCount).Select(_ => (rng.NextDouble() * 2.0) - 1.0).ToArray())
            .ToArray();

        Matrix<double> Features(int rows)
        {
            var x = new Matrix<double>(rows, FeatureCount);
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < FeatureCount; j++) x[i, j] = centres[i % NumWays][j] + (0.1 * (rng.NextDouble() - 0.5));
            return x;
        }

        double[] Labels(int rows) => Enumerable.Range(0, rows).Select(i => (double)(i % NumWays)).ToArray();

        return new MetaLearningTask<double, Matrix<double>, TOutput>
        {
            SupportSetX = Features(4),
            SupportSetY = labels(Labels(4)),
            QuerySetX = Features(4),
            QuerySetY = labels(Labels(4)),
            NumWays = NumWays,
            NumShots = 2,
            NumQueryPerClass = 2,
            Name = $"adapted-task-{seed}",
        };
    }
}

/// <summary>Adapted-model invariants for <see cref="AdaptedMetaModel{T, TInput, TOutput}"/>, produced by ACL.</summary>
public class AdaptedMetaModelTests : AdaptedModelTestBase<Vector<double>>
{
    protected override Type ExpectedType => typeof(AdaptedMetaModel<double, Matrix<double>, Vector<double>>);

    protected override IMetaLearner<double, Matrix<double>, Vector<double>> CreateLearner()
        => new ACLAlgorithm<double, Matrix<double>, Vector<double>>(
            new ACLOptions<double, Matrix<double>, Vector<double>>(new LinearVectorModel(AdaptedModelTasks.FeatureCount))
            {
                RandomSeed = 1337,
            });

    protected override MetaLearningTask<double, Matrix<double>, Vector<double>> CreateTask(int seed)
        => AdaptedModelTasks.Create(seed, labels => new Vector<double>(labels));

    protected override double[] Values(Vector<double> output) => output.ToArray();
}

/// <summary>Adapted-model invariants for <see cref="NeuralProcessModel{T, TInput, TOutput}"/>, produced by CNP.</summary>
public class NeuralProcessModelTests : AdaptedModelTestBase<Vector<double>>
{
    protected override Type ExpectedType => typeof(NeuralProcessModel<double, Matrix<double>, Vector<double>>);

    protected override IMetaLearner<double, Matrix<double>, Vector<double>> CreateLearner()
        => new CNPAlgorithm<double, Matrix<double>, Vector<double>>(
            new CNPOptions<double, Matrix<double>, Vector<double>>(new LinearVectorModel(AdaptedModelTasks.FeatureCount))
            {
                RandomSeed = 1337,
                LossFunction = new MeanSquaredErrorLoss<double>(),
            });

    protected override MetaLearningTask<double, Matrix<double>, Vector<double>> CreateTask(int seed)
        => AdaptedModelTasks.Create(seed, labels => new Vector<double>(labels));

    protected override double[] Values(Vector<double> output) => output.ToArray();
}

/// <summary>Adapted-model invariants for <see cref="MbPAAdaptedModel{T, TInput, TOutput}"/>, produced by MbPA.</summary>
public class MbPAAdaptedModelTests : AdaptedModelTestBase<Tensor<double>>
{
    protected override Type ExpectedType => typeof(MbPAAdaptedModel<double, Matrix<double>, Tensor<double>>);

    protected override IMetaLearner<double, Matrix<double>, Tensor<double>> CreateLearner()
        => new MbPAAlgorithm<double, Matrix<double>, Tensor<double>>(
            new MbPAOptions<double, Matrix<double>, Tensor<double>>(
                new LinearEmbeddingModel(AdaptedModelTasks.FeatureCount, AdaptedModelTasks.EmbeddingDimension))
            {
                RandomSeed = 1337,
                MemorySize = 8,
                FeatureDimension = AdaptedModelTasks.EmbeddingDimension,
                OutputDimension = AdaptedModelTasks.NumWays,
            });

    protected override MetaLearningTask<double, Matrix<double>, Tensor<double>> CreateTask(int seed)
        => AdaptedModelTasks.Create(seed, labels => new Tensor<double>(labels, new[] { labels.Length }));

    protected override double[] Values(Tensor<double> output) => output.ToArray();
}