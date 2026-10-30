using System;
using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;

namespace AiDotNet.NeuralNetworks.SyntheticData;

/// <summary>
/// Shared caller-facing shape contract for neural synthetic-tabular generators.
/// </summary>
/// <remarks>
/// These models consume one feature vector or a batch of feature vectors and emit the configured
/// table width. Their internal generators differ, but that public geometry is common and is derived
/// from the architecture rather than copied into every implementation.
/// </remarks>
[TensorLayout(TensorAxis.Batch, TensorAxis.Features,
    Direction = TensorLayoutDirection.Input, BatchOptional = true)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output, BatchOptional = true)]
public abstract class NeuralSyntheticTabularGeneratorBase<T> : NeuralNetworkBase<T>, IShapeContract
{
    /// <summary>Initializes the shared neural generator base.</summary>
    protected NeuralSyntheticTabularGeneratorBase(
        NeuralNetworkArchitecture<T> architecture,
        ILossFunction<T> lossFunction,
        double maxGradNorm)
        : base(architecture, lossFunction, maxGradNorm)
    {
    }

    /// <summary>
    /// One training call is one step of the model's own objective -- a GAN's critic and generator steps, a VAE's ELBO
    /// step, a diffusion model's denoising step -- on the batch of real data in <paramref name="target"/>, the way
    /// <c>Fit</c> takes it on every batch. The supervised step (the generator's output against the target) is no
    /// generator's objective: under it a GAN trained toward the target row for whatever noise came in, or crashed where
    /// its forward is not the plain layer walk.
    /// </summary>
    /// <param name="input">The generator input (noise or conditioning); the step draws its own latents, as Fit does.</param>
    /// <param name="target">Real rows in the generator's output space: [features] or [batch, features].</param>
    protected sealed override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> target)
    {
        if (target is null || target.Length == 0)
            throw new ArgumentException("A generator trains on a batch of real data rows; the target holds none.", nameof(target));
        TrainOnBatch(target);
        return true;
    }

    /// <summary>
    /// Takes one step of this generator's training objective on <paramref name="realData"/>, building its networks for
    /// the data's width first when they are not built for it.
    /// </summary>
    protected abstract void TrainOnBatch(Tensor<T> realData);

    /// <summary>The rows of <paramref name="data"/> ([features] is one row; leading axes are flattened).</summary>
    protected static Matrix<T> RowsOf(Tensor<T> data)
    {
        int width = data.Shape[data.Shape.Length - 1];
        if (width <= 0)
            throw new ArgumentException("A data row needs at least one feature.", nameof(data));
        int rows = data.Length / width;
        var values = data.ToArray();
        var matrix = new Matrix<T>(rows, width);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < width; c++)
                matrix[r, c] = values[r * width + c];
        return matrix;
    }

    /// <summary>
    /// Column metadata for data that arrives without any (a training call's rows): every column continuous, the
    /// statistics left for the generator's own preparation to compute.
    /// </summary>
    protected static IReadOnlyList<ColumnMetadata> ContinuousColumns(int width)
    {
        var columns = new List<ColumnMetadata>(width);
        for (int c = 0; c < width; c++)
            columns.Add(new ColumnMetadata($"column{c}", ColumnDataType.Continuous, columnIndex: c));
        return columns;
    }

    /// <inheritdoc />
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
        => inputRank switch
        {
            1 =>
            [
                new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(Architecture.OutputSize)),
            ],
            2 =>
            [
                new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
                new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(Architecture.OutputSize)),
            ],
            _ => null,
        };
}
