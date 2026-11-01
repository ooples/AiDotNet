using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// A Gaussian mixture classifier: one diagonal-covariance Gaussian mixture per class, emitting each class's
/// log-likelihood <c>log p(x | class)</c>. Under equal class priors a softmax over the output is the Bayes posterior.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>This is the statistical pattern recognition classifier of Tzanetakis and Cook 2002 (section IV): a mixture of
/// <c>components</c> Gaussians with diagonal covariance per class, initialized with K-means and fitted with EM
/// (<see cref="Fit"/>). One component is their "simple Gaussian" classifier. The parameters are also differentiable, so
/// the layer can be trained discriminatively.</para>
/// <para><b>For Beginners:</b> Each class gets a few "clouds" of points describing where its examples usually lie. A new
/// example is scored by how well it fits each class's clouds, and the best-fitting class wins.</para>
/// </remarks>
[LayerCategory(LayerCategory.Dense)]
[LayerTask(LayerTask.Projection)]
[LayerProperty(IsTrainable = true, TestInputShape = "2, 4", TestConstructorArgs = "3, 2, 4")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, BatchOptional = true, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, BatchOptional = true, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class GaussianMixtureClassifierLayer<T> : LayerBase<T>, IShapeContract
{
    private const double VarianceFloor = 1e-6;
    private readonly int _classes, _components, _features;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _means;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _logVariances;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _logWeights;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>Creates the layer with unit-variance components at the origin and equal weights.</summary>
    /// <param name="classes">The classes.</param>
    /// <param name="components">Gaussians per class.</param>
    /// <param name="features">The feature dimension.</param>
    public GaussianMixtureClassifierLayer([LayerState] int classes, [LayerState] int components, [LayerState] int features)
        : base(new[] { features }, new[] { classes })
    {
        if (classes <= 0) throw new ArgumentOutOfRangeException(nameof(classes));
        if (components <= 0) throw new ArgumentOutOfRangeException(nameof(components));
        if (features <= 0) throw new ArgumentOutOfRangeException(nameof(features));
        _classes = classes;
        _components = components;
        _features = features;
        _means = new Tensor<T>(new[] { classes, components, features });
        _logVariances = new Tensor<T>(new[] { classes, components, features });
        _logWeights = new Tensor<T>(new[] { classes, components });
        RegisterTrainableParameter(_means, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_logVariances, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_logWeights, PersistentTensorRole.Weights);
    }

    /// <summary>Gaussians per class.</summary>
    internal int Components => _components;

    /// <inheritdoc />
    /// <remarks>The trailing feature axis becomes one log-likelihood per class; a leading batch axis is carried.</remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        var classes = new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_classes));
        return inputRank switch
        {
            1 => new[] { classes },
            2 => new[] { new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)), classes },
            _ => null,
        };
    }

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        bool single = input.Rank == 1;
        if (input.Rank > 2 || input.Shape[input.Rank - 1] != _features)
            throw new ArgumentException($"Expected {_features} features, got [{string.Join(", ", input.Shape)}].", nameof(input));
        int batch = input.Length / _features;
        var x = Engine.TensorTile(Engine.Reshape(input, new[] { batch, 1, 1, _features }), new[] { 1, _classes, _components, 1 });
        var means = Engine.TensorTile(Engine.Reshape(_means, new[] { 1, _classes, _components, _features }), new[] { batch, 1, 1, 1 });
        var logVariances = Engine.TensorTile(Engine.Reshape(_logVariances, new[] { 1, _classes, _components, _features }), new[] { batch, 1, 1, 1 });
        var diff = Engine.TensorSubtract(x, means);
        // sum_d (x_d - mu_d)^2 / var_d + log var_d, per component.
        var mahalanobis = Engine.TensorAdd(Engine.TensorMultiply(Engine.TensorMultiply(diff, diff), Engine.TensorExp(Engine.TensorNegate(logVariances))), logVariances);
        var quadratic = Engine.ReduceSum(mahalanobis, new[] { 3 }, false);
        var logMixture = Engine.TensorTile(Engine.Reshape(Engine.TensorLogSoftmax(_logWeights, 1), new[] { 1, _classes, _components }), new[] { batch, 1, 1 });
        var perComponent = Engine.TensorAddScalar(Engine.TensorSubtract(logMixture, Engine.TensorMultiplyScalar(quadratic, NumOps.FromDouble(0.5))),
            NumOps.FromDouble(-0.5 * _features * Math.Log(2 * Math.PI)));
        var logLikelihood = Engine.TensorLogSumExp(perComponent, 2, false);
        return single ? Engine.Reshape(logLikelihood, new[] { _classes }) : logLikelihood;
    }

    /// <summary>
    /// Fits every class's mixture to its samples by maximum likelihood: K-means initialization, then EM for diagonal
    /// covariance (Tzanetakis and Cook 2002, section IV). A class without samples keeps its parameters.
    /// </summary>
    /// <param name="samples">The feature vectors.</param>
    /// <param name="labels">Each sample's class.</param>
    /// <param name="iterations">EM iterations (and K-means iterations).</param>
    /// <param name="seed">Seed of the K-means initialization.</param>
    internal void Fit(IReadOnlyList<double[]> samples, IReadOnlyList<int> labels, int iterations, int seed)
    {
        var random = RandomHelper.CreateSeededRandom(seed);
        for (int c = 0; c < _classes; c++)
        {
            var points = new List<double[]>();
            for (int i = 0; i < samples.Count; i++) if (labels[i] == c) points.Add(samples[i]);
            if (points.Count == 0) continue;
            FitClass(c, points, iterations, random);
        }
        Engine.InvalidatePersistentTensor(_means);
        Engine.InvalidatePersistentTensor(_logVariances);
        Engine.InvalidatePersistentTensor(_logWeights);
    }

    private void FitClass(int c, List<double[]> points, int iterations, Random random)
    {
        int n = points.Count, k = _components, d = _features;
        // The class's per-feature variance sets the floor, so features of very different scales are each regularized.
        var globalMean = new double[d];
        var floor = new double[d];
        foreach (var p in points) for (int j = 0; j < d; j++) globalMean[j] += p[j] / n;
        foreach (var p in points) for (int j = 0; j < d; j++) floor[j] += (p[j] - globalMean[j]) * (p[j] - globalMean[j]) / n;
        for (int j = 0; j < d; j++) floor[j] = Math.Max(floor[j] * 1e-3, VarianceFloor);

        // K-means: centres from distinct samples, then Lloyd iterations.
        var centres = points.OrderBy(_ => random.Next()).Take(k).Select(p => (double[])p.Clone()).ToList();
        while (centres.Count < k) centres.Add((double[])points[centres.Count % n].Clone());
        var assignment = new int[n];
        for (int it = 0; it < iterations; it++)
        {
            for (int i = 0; i < n; i++) assignment[i] = Nearest(points[i], centres, floor);
            for (int m = 0; m < k; m++)
            {
                var members = Enumerable.Range(0, n).Where(i => assignment[i] == m).ToList();
                if (members.Count == 0) continue;
                for (int j = 0; j < d; j++) centres[m][j] = members.Average(i => points[i][j]);
            }
        }

        var weights = new double[k];
        var means = centres;
        var variances = new double[k][];
        for (int m = 0; m < k; m++)
        {
            variances[m] = new double[d];
            var members = Enumerable.Range(0, n).Where(i => assignment[i] == m).ToList();
            weights[m] = Math.Max(members.Count, 1) / (double)(n + k);
            for (int j = 0; j < d; j++)
                variances[m][j] = members.Count > 1 ? Math.Max(members.Average(i => (points[i][j] - means[m][j]) * (points[i][j] - means[m][j])), floor[j]) : Math.Max(floor[j] * 1e3, VarianceFloor);
        }

        var responsibility = new double[n, k];
        for (int it = 0; it < iterations; it++)
        {
            // E step.
            for (int i = 0; i < n; i++)
            {
                double max = double.NegativeInfinity;
                var log = new double[k];
                for (int m = 0; m < k; m++)
                {
                    double s = Math.Log(weights[m]);
                    for (int j = 0; j < d; j++)
                    {
                        double diff = points[i][j] - means[m][j];
                        s -= 0.5 * (diff * diff / variances[m][j] + Math.Log(2 * Math.PI * variances[m][j]));
                    }
                    log[m] = s;
                    max = Math.Max(max, s);
                }
                double total = 0;
                for (int m = 0; m < k; m++) total += Math.Exp(log[m] - max);
                for (int m = 0; m < k; m++) responsibility[i, m] = Math.Exp(log[m] - max) / total;
            }
            // M step.
            for (int m = 0; m < k; m++)
            {
                double mass = 0;
                for (int i = 0; i < n; i++) mass += responsibility[i, m];
                if (mass < 1e-10) continue;
                weights[m] = mass / n;
                for (int j = 0; j < d; j++)
                {
                    double mean = 0;
                    for (int i = 0; i < n; i++) mean += responsibility[i, m] * points[i][j];
                    mean /= mass;
                    double variance = 0;
                    for (int i = 0; i < n; i++) variance += responsibility[i, m] * (points[i][j] - mean) * (points[i][j] - mean);
                    means[m][j] = mean;
                    variances[m][j] = Math.Max(variance / mass, floor[j]);
                }
            }
        }

        for (int m = 0; m < k; m++)
        {
            _logWeights[c, m] = NumOps.FromDouble(Math.Log(Math.Max(weights[m], 1e-12)));
            for (int j = 0; j < d; j++)
            {
                _means[c, m, j] = NumOps.FromDouble(means[m][j]);
                _logVariances[c, m, j] = NumOps.FromDouble(Math.Log(variances[m][j]));
            }
        }
    }

    // K-means distance scaled per feature, so a large-valued feature does not decide the clustering alone.
    private static int Nearest(double[] point, List<double[]> centres, double[] scale)
    {
        int best = 0;
        double bestDistance = double.PositiveInfinity;
        for (int m = 0; m < centres.Count; m++)
        {
            double s = 0;
            for (int j = 0; j < point.Length; j++) s += (point[j] - centres[m][j]) * (point[j] - centres[m][j]) / scale[j];
            if (s < bestDistance) { bestDistance = s; best = m; }
        }
        return best;
    }

    /// <inheritdoc />
    public override void ResetState()
    {
    }

    /// <summary>Persists the constructor arguments.</summary>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["Classes"] = _classes.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["Components"] = _components.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["Features"] = _features.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }
}
