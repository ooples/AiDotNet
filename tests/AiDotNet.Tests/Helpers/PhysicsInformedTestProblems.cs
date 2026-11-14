using System;
using System.Collections.Generic;
using AiDotNet.Interfaces;
using AiDotNet.PhysicsInformed.Interfaces;
using AiDotNet.PhysicsInformed.PDEs;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tests.Helpers;

/// <summary>
/// A two-scale Poisson problem for generated MultiScalePINN fixtures.
/// </summary>
/// <remarks>
/// Nothing in the library implements <see cref="IMultiScalePDE{T}"/>, so MultiScalePINN could not be
/// constructed by a generated fixture. Both scales solve the same 2-D Poisson equation and the solution is
/// their sum, so the scales need no coupling residual. The characteristic lengths (1 and 0.1) mark a coarse
/// and a fine scale; each contributes one output with unit loss weight.
/// </remarks>
public sealed class TwoScalePoissonProblem<T> : IMultiScalePDE<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly PoissonEquation<T> _poisson = new PoissonEquation<T>();

    /// <inheritdoc/>
    public int NumberOfScales => 2;

    /// <inheritdoc/>
    public T[] ScaleCharacteristicLengths => new[] { NumOps.One, NumOps.FromDouble(0.1) };

    /// <inheritdoc/>
    public int InputDimension => _poisson.InputDimension;

    /// <inheritdoc/>
    public int OutputDimension => _poisson.OutputDimension;

    /// <inheritdoc/>
    public string Name => "Two-scale Poisson (test problem)";

    /// <inheritdoc/>
    public T ComputeResidual(Vector<T> inputs, Vector<T> outputs, PDEDerivatives<T> derivatives)
        => _poisson.ComputeResidual(inputs, outputs, derivatives);

    /// <inheritdoc/>
    public Tensor<T> ComputeScaleResidual(int scaleIndex, Tensor<T> inputs, Tensor<T> outputs, PDEDerivatives<T> derivatives)
        => _poisson.ComputeTapeResidual(inputs, outputs, derivatives);

    /// <inheritdoc/>
    public Tensor<T> ComputeScaleCoupling(
        int coarseIndex,
        int fineIndex,
        Tensor<T> inputs,
        Tensor<T> coarseOutputs,
        Tensor<T> fineOutputs,
        PDEDerivatives<T> coarseDerivatives,
        PDEDerivatives<T> fineDerivatives)
        => new Tensor<T>(new[] { inputs.Shape[0] });

    /// <inheritdoc/>
    public T GetScaleLossWeight(int scaleIndex) => NumOps.One;

    /// <inheritdoc/>
    public int GetScaleOutputDimension(int scaleIndex) => 1;
}

/// <summary>
/// Poisson's equation with one unknown source strength, for generated InverseProblemPINN fixtures.
/// </summary>
/// <remarks>
/// Nothing in the library implements <see cref="IInverseProblem{T}"/>, so InverseProblemPINN could not be
/// constructed by a generated fixture. The unknown is a in laplacian(u) = a. The observations come from the
/// exact solution u(x, y) = (x^2 + y^2) / 4, whose Laplacian is 1, so the true value is a = 1.
/// </remarks>
public sealed class PoissonSourceInverseProblem<T> : IInverseProblem<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <inheritdoc/>
    public string[] ParameterNames => new[] { "source_strength" };

    /// <inheritdoc/>
    public int NumberOfParameters => 1;

    /// <inheritdoc/>
    public T[] InitialParameterGuesses => new[] { NumOps.FromDouble(0.5) };

    /// <inheritdoc/>
    public T[]? ParameterLowerBounds => null;

    /// <inheritdoc/>
    public T[]? ParameterUpperBounds => null;

    /// <inheritdoc/>
    public IReadOnlyList<(T[] location, T[] value)> Observations { get; } = CreateObservations();

    /// <inheritdoc/>
    public bool HasMeasurementNoiseLevel => false;

    /// <inheritdoc/>
    public T MeasurementNoiseLevel => NumOps.Zero;

    /// <inheritdoc/>
    public bool ValidateParameters(T[] parameters)
    {
        if (parameters is null || parameters.Length != NumberOfParameters)
            return false;

        double value = NumOps.ToDouble(parameters[0]);
        return !double.IsNaN(value) && !double.IsInfinity(value);
    }

    /// <inheritdoc/>
    public IPDESpecification<T> CreateParameterizedPDE(T[] parameters)
    {
        T sourceStrength = parameters[0];
        return new PoissonEquation<T>(points => Tensor<T>.CreateDefault(new[] { points.Shape[0] }, sourceStrength));
    }

    private static IReadOnlyList<(T[] location, T[] value)> CreateObservations()
    {
        var points = new[] { (0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75) };
        var observations = new List<(T[] location, T[] value)>(points.Length);
        foreach (var (x, y) in points)
        {
            observations.Add((
                new[] { NumOps.FromDouble(x), NumOps.FromDouble(y) },
                new[] { NumOps.FromDouble((x * x + y * y) / 4.0) }));
        }

        return observations;
    }
}

/// <summary>
/// Builds PDE derivative tensors from the row-major multi-dimensional arrays the tests write by hand.
/// </summary>
public static class PdeTestTensors
{
    /// <summary>Copies <paramref name="values"/> into a tensor of the same shape.</summary>
    public static Tensor<double> From(Array values)
    {
        var shape = new int[values.Rank];
        for (int d = 0; d < shape.Length; d++)
        {
            shape[d] = values.GetLength(d);
        }

        var data = new double[values.Length];
        int i = 0;
        foreach (double v in values)
        {
            data[i++] = v;
        }

        return new Tensor<double>(data, shape);
    }
}
