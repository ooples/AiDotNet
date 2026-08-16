using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Finance.Base;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.NeuralNetworks.Graph;

/// <summary>
/// Separates a forecasting model's output-rank CONVENTION from an output-rank DEFECT.
/// </summary>
/// <remarks>
/// <para>
/// The conformance sweep found the forecasting family answering the same question at four different
/// ranks, but it only ever predicts at batch 1 - and at batch 1 a model returning <c>[24]</c> is
/// indistinguishable from one returning <c>[1,24]</c> with the unit axes squeezed. Those two are not
/// the same finding:
/// </para>
/// <list type="bullet">
/// <item><description>Squeezing a UNIT FEATURE axis when NumFeatures is 1 is a convention. GluonTS,
/// Darts and pytorch-forecasting all return a univariate forecast without a trailing 1.</description></item>
/// <item><description>Dropping the BATCH axis is a defect. A model whose output does not grow with
/// the batch cannot be batched at all, and the caller has no way to tell which row is which.</description></item>
/// </list>
/// <para>
/// So this predicts the SAME model at batch 1 and batch 2 and asks one question: does the output
/// track the batch? Reporting-only - it prints a classification rather than failing, because the
/// remedy for a real defect is a model change that has not been decided yet.
/// </para>
/// </remarks>
public class ForecastOutputRankDiagnosticTests
{
    private readonly ITestOutputHelper _out;
    public ForecastOutputRankDiagnosticTests(ITestOutputHelper output) => _out = output;

    private const int SeqLen = 16;
    private const int Features = 4;

    [Fact]
    public void AForecastOutputEitherTracksTheBatchAxisOrCannotBeBatched()
    {
        var models = typeof(NeuralNetworkBase<>).Assembly.GetTypes()
            .Where(t => t.IsClass && !t.IsAbstract && t.IsGenericTypeDefinition
                        && t.GetGenericArguments().Length == 1
                        && DerivesFromForecastingBase(t))
            .OrderBy(t => t.Name, StringComparer.Ordinal)
            .ToList();

        _out.WriteLine($"forecasting models: {models.Count}");

        var tracksBatch = new List<string>();
        var ignoresBatch = new List<string>();
        var skipped = new List<string>();

        foreach (var open in models)
        {
            Type closed;
            try { closed = open.MakeGenericType(typeof(double)); }
            catch { continue; }

            object? model = null;
            try
            {
                model = Construct(closed);
                if (model is null) { skipped.Add($"{open.Name}: no architecture constructor"); continue; }

                int[]? one = TryPredictShape(model, 1);
                int[]? two = TryPredictShape(model, 2);
                if (one is null || two is null)
                {
                    skipped.Add($"{open.Name}: Predict failed at batch 1 or 2");
                    continue;
                }

                // The question is only whether the OUTPUT GREW. A model may legitimately squeeze a
                // unit feature axis; it may not legitimately return the same tensor for two rows.
                long sizeOne = one.Aggregate(1L, (a, b) => a * b);
                long sizeTwo = two.Aggregate(1L, (a, b) => a * b);

                string line = $"{open.Name,-28} batch1 [{string.Join(",", one)}]  batch2 [{string.Join(",", two)}]";
                if (sizeTwo == sizeOne * 2) tracksBatch.Add(line);
                else ignoresBatch.Add(line);
            }
            catch (Exception ex) { skipped.Add($"{open.Name}: {Unwrap(ex).GetType().Name}"); }
            finally { (model as IDisposable)?.Dispose(); }
        }

        _out.WriteLine("");
        _out.WriteLine($"output doubles with the batch (batchable)      : {tracksBatch.Count}");
        _out.WriteLine($"output IGNORES the batch (cannot be batched)   : {ignoresBatch.Count}");
        _out.WriteLine($"skipped                                        : {skipped.Count}");
        _out.WriteLine("");
        _out.WriteLine("--- tracks the batch ---");
        foreach (var s in tracksBatch) _out.WriteLine($"  {s}");
        _out.WriteLine("");
        _out.WriteLine("--- IGNORES the batch ---");
        foreach (var s in ignoresBatch) _out.WriteLine($"  {s}");
        _out.WriteLine("");
        foreach (var s in skipped.Take(20)) _out.WriteLine($"  skipped: {s}");

        Assert.True(tracksBatch.Count + ignoresBatch.Count > 0,
            "no forecasting model was exercised - the harness, not the finding, is broken");
    }

    private static bool DerivesFromForecastingBase(Type type)
    {
        for (var t = type.BaseType; t is not null; t = t.BaseType)
        {
            if (t.IsGenericType && t.GetGenericTypeDefinition() == typeof(ForecastingModelBase<>)) return true;
        }
        return false;
    }

    private static int[]? TryPredictShape(object model, int batch)
    {
        try
        {
            var input = new Tensor<double>(new[] { batch, SeqLen, Features });
            return ((NeuralNetworkBase<double>)model).Predict(input).Shape.ToArray();
        }
        catch { return null; }
    }

    private static object? Construct(Type closed)
    {
        var ctor = closed.GetConstructors().FirstOrDefault(c =>
        {
            var ps = c.GetParameters();
            return ps.Length > 0
                && ps[0].ParameterType == typeof(NeuralNetworkArchitecture<double>)
                && ps.Skip(1).All(p => p.HasDefaultValue);
        });
        if (ctor is null) return null;

        var architecture = new NeuralNetworkArchitecture<double>(
            InputType.TwoDimensional, NeuralNetworkTaskType.Regression,
            inputHeight: SeqLen, inputWidth: Features, outputSize: 1);

        var pars = ctor.GetParameters();
        var args = new object?[pars.Length];
        args[0] = architecture;
        for (int i = 1; i < pars.Length; i++) args[i] = pars[i].DefaultValue;

        try { return ctor.Invoke(args); }
        catch { return null; }
    }

    private static Exception Unwrap(Exception ex)
        => ex is System.Reflection.TargetInvocationException { InnerException: not null } tie
            ? Unwrap(tie.InnerException) : ex;
}
