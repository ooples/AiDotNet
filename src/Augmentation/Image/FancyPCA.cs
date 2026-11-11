namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Fancy PCA color augmentation as described in AlexNet (Krizhevsky et al. 2012).
/// Applies perturbation along principal components of color channels.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class FancyPCA<T> : ImageAugmenterBase<T>
{
    public double AlphaStd { get; }

    public FancyPCA(double alphaStd = 0.1, double probability = 0.5) : base(probability)
    {
        AlphaStd = alphaStd;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        if (data.Channels < 3) return data.Clone();

        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        int numPixels = data.Height * data.Width;

        // Compute channel means and the 3x3 covariance matrix of the RGB values
        var engine = Engine;
        var image = data.ToPlanarTensor();
        var rgb = engine.Reshape(engine.TensorSlice(image, [0, 0, 0], [3, data.Height, data.Width]), [3, numPixels]);
        var centered = engine.TensorSubtract(rgb, engine.ReduceMean(rgb, [1], true));
        var covTensor = engine.TensorDivideScalar(
            engine.TensorMatMul(centered, engine.TensorTranspose(centered)), NumOps.FromDouble(numPixels));
        // Power iteration with deflation on the 3x3 covariance; once an update's norm drops below
        // 1e-10 that eigenvector stops changing (the scalar loop's break)
        T eps = NumOps.FromDouble(1e-10);
        T zero = NumOps.Zero;
        var identity = ImageTensor<T>.KernelTensor(new double[,] { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } });
        var tempCov = covTensor;
        var eigenvalues = new Tensor<T>[3];
        var eigenvectors = new Tensor<T>[3];

        for (int ev = 0; ev < 3; ev++)
        {
            var v = engine.TensorSlice(identity, [0, ev], [3, 1]);
            var stopped = new Tensor<T>([1, 1]);

            for (int iter = 0; iter < 50; iter++)
            {
                var newV = engine.TensorMatMul(tempCov, v);
                var norm = engine.TensorSqrt(engine.ReduceSum(engine.TensorMultiply(newV, newV), [0], true));
                stopped = engine.TensorMax(stopped, engine.TensorLessThan(norm, eps));
                v = engine.TensorWhere(engine.TensorBroadcastTo(stopped, [3, 1]), v,
                    engine.TensorDivide(newV, engine.TensorBroadcastTo(norm, [3, 1])));
            }

            var eigenvalue = engine.ReduceSum(engine.TensorMultiply(v, engine.TensorMatMul(tempCov, v)), [0], true);
            eigenvalues[ev] = engine.TensorMax(eigenvalue, zero);
            eigenvectors[ev] = v;

            // Deflate
            tempCov = engine.TensorSubtract(tempCov,
                engine.TensorMultiply(engine.TensorBroadcastTo(eigenvalue, [3, 3]), engine.TensorMatMul(v, engine.TensorTranspose(v))));
        }

        // Generate random alpha values (Krizhevsky: perturbation = sum_i(alpha_i * lambda_i * p_i))
        var alpha = engine.TensorMultiply(
            AugmentationRandom<T>.Gaussian(context, [3, 1], 0, AlphaStd), engine.TensorConcatenate(eigenvalues, 0));

        // Compute perturbation per channel: [p1, p2, p3] * [alpha1*lambda1, alpha2*lambda2, alpha3*lambda3]
        var perturbation = engine.Reshape(engine.TensorMatMul(engine.TensorConcatenate(eigenvectors, 1), alpha), [3, 1, 1]);

        // Apply perturbation
        var perturbed = engine.TensorClamp(
            engine.TensorAdd(engine.Reshape(rgb, [3, data.Height, data.Width]), perturbation),
            NumOps.Zero, NumOps.FromDouble(maxVal));
        result.SetFromPlanarTensor(ImageTensor<T>.ReplaceLeadingChannels(image,
            engine.TensorSliceAxis(perturbed, 0, 0), engine.TensorSliceAxis(perturbed, 0, 1), engine.TensorSliceAxis(perturbed, 0, 2)));

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["alpha_std"] = AlphaStd;
        return p;
    }
}
