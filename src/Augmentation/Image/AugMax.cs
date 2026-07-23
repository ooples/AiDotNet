namespace AiDotNet.Augmentation.Image;

/// <summary>
/// AugMax (Wang et al., 2021) - adversarial composition of random augmentations.
/// Selects the augmented version that maximizes loss diversity.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class AugMax<T> : ImageAugmenterBase<T>
{
    public int K { get; }
    public int N { get; }
    public double M { get; }

    public AugMax(int k = 3, int n = 2, double m = 9,
        double probability = 1.0) : base(probability)
    {
        K = k; N = n; M = m;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        // Generate K augmented versions using RandAugment, return a random one
        // (In a training loop, the one maximizing loss would be selected)
        var candidates = new List<ImageTensor<T>>();

        for (int i = 0; i < K; i++)
        {
            var randAug = new RandAugment<T>(N, M, probability: 1.0);
            candidates.Add(randAug.Apply(data, context));
        }

        // Without loss info, randomly select (in practice, training loop picks hardest)
        int selected = context.GetRandomInt(0, K);
        return candidates[selected];
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["k"] = K; p["n"] = N; p["m"] = M;
        return p;
    }
}
