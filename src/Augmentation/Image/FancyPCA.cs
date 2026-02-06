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

        // Compute channel means
        var mean = new double[3];
        for (int y = 0; y < data.Height; y++)
            for (int x = 0; x < data.Width; x++)
                for (int c = 0; c < 3; c++)
                    mean[c] += NumOps.ToDouble(data.GetPixel(y, x, c));
        for (int c = 0; c < 3; c++) mean[c] /= numPixels;

        // Compute covariance matrix (3x3)
        var cov = new double[3, 3];
        for (int y = 0; y < data.Height; y++)
            for (int x = 0; x < data.Width; x++)
            {
                var diff = new double[3];
                for (int c = 0; c < 3; c++)
                    diff[c] = NumOps.ToDouble(data.GetPixel(y, x, c)) - mean[c];
                for (int i = 0; i < 3; i++)
                    for (int j = i; j < 3; j++)
                    {
                        cov[i, j] += diff[i] * diff[j];
                        if (i != j) cov[j, i] = cov[i, j];
                    }
            }
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                cov[i, j] /= numPixels;

        // Simple eigendecomposition for 3x3 symmetric matrix using power iteration
        var eigenvalues = new double[3];
        var eigenvectors = new double[3, 3];
        var tempCov = new double[3, 3];
        Array.Copy(cov, tempCov, 9);

        for (int ev = 0; ev < 3; ev++)
        {
            var v = new double[3];
            v[ev] = 1.0;

            for (int iter = 0; iter < 50; iter++)
            {
                var newV = new double[3];
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++)
                        newV[i] += tempCov[i, j] * v[j];

                double norm = Math.Sqrt(newV[0] * newV[0] + newV[1] * newV[1] + newV[2] * newV[2]);
                if (norm < 1e-10) break;
                for (int i = 0; i < 3; i++) v[i] = newV[i] / norm;
            }

            double eigenvalue = 0;
            var av = new double[3];
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    av[i] += tempCov[i, j] * v[j];
            for (int i = 0; i < 3; i++) eigenvalue += v[i] * av[i];

            eigenvalues[ev] = Math.Max(0, eigenvalue);
            for (int i = 0; i < 3; i++) eigenvectors[i, ev] = v[i];

            // Deflate
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    tempCov[i, j] -= eigenvalue * v[i] * v[j];
        }

        // Generate random alpha values (Krizhevsky: perturbation = sum_i(alpha_i * lambda_i * p_i))
        var alpha = new double[3];
        for (int i = 0; i < 3; i++)
            alpha[i] = context.SampleGaussian(0, AlphaStd) * eigenvalues[i];

        // Compute perturbation per channel: [p1, p2, p3] * [alpha1*lambda1, alpha2*lambda2, alpha3*lambda3]
        var perturbation = new double[3];
        for (int c = 0; c < 3; c++)
            for (int ev = 0; ev < 3; ev++)
                perturbation[c] += eigenvectors[c, ev] * alpha[ev];

        // Apply perturbation
        for (int y = 0; y < data.Height; y++)
            for (int x = 0; x < data.Width; x++)
                for (int c = 0; c < 3; c++)
                {
                    double val = NumOps.ToDouble(data.GetPixel(y, x, c)) + perturbation[c];
                    result.SetPixel(y, x, c, NumOps.FromDouble(Math.Max(0, Math.Min(maxVal, val))));
                }

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["alpha_std"] = AlphaStd;
        return p;
    }
}
