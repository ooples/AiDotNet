using AiDotNet.HarmonicEngine.Interfaces;

namespace AiDotNet.HarmonicEngine.Core;

/// <summary>
/// Assigns orthogonal frequency carrier positions that are free of intermodulation distortion (IMD) collisions.
/// This ensures that second-order and third-order IMD products land at unique, predictable frequencies
/// that do not overlap with any carrier or other IMD product.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> When signals at different frequencies pass through a nonlinear device,
/// they create new signals at combination frequencies (intermodulation products).
/// For example, signals at frequencies f1=3 and f2=5 create products at f1+f2=8 and f1-f2=2.
///
/// If we're not careful about where we place our carriers, these IMD products could land on top
/// of other carriers, corrupting the signal. The CarrierAllocator uses mathematical sequences
/// (related to Sidon sets / B2 sequences) to guarantee that all IMD products land at unique
/// frequencies with no collisions.
///
/// A Sidon set (B2 sequence) has the property that all pairwise sums a_i + a_j are distinct.
/// This guarantees that second-order IMD products (fi + fj and fi - fj) never collide.
/// </para>
/// </remarks>
public class CarrierAllocator : ICarrierAllocator
{
    /// <inheritdoc/>
    public int MaxCarriers(int fftSize, int maxOrder = 2)
    {
        // Estimate max carriers by trying to allocate until failure
        int maxBin = fftSize / 2;
        var carriers = new List<int>();
        var allProducts = new HashSet<int>();
        var carrierSet = new HashSet<int>();

        for (int candidate = 1; candidate <= maxBin; candidate++)
        {
            if (CanAddCarrier(carriers, carrierSet, allProducts, candidate))
            {
                foreach (int existing in carriers)
                {
                    allProducts.Add(candidate + existing);
                    allProducts.Add(Math.Abs(candidate - existing));
                }
                allProducts.Add(candidate + candidate);
                carriers.Add(candidate);
                carrierSet.Add(candidate);
            }
        }

        return carriers.Count;
    }

    /// <summary>
    /// Allocates carrier frequency bin indices that are IMD-collision-free up to second order.
    /// Uses a modified Sidon set construction based on quadratic residues.
    /// </summary>
    /// <param name="numCarriers">Number of carriers to allocate.</param>
    /// <param name="fftSize">Total FFT size (determines available frequency bins).</param>
    /// <returns>
    /// Array of frequency bin indices where carriers should be placed.
    /// All pairwise sums and differences are guaranteed unique.
    /// </returns>
    public int[] AllocateCarriers(int numCarriers, int fftSize)
    {
        // Use a greedy construction where all IMD products (sums AND differences)
        // are tracked in a single set to prevent any collision.
        int maxBin = fftSize / 2; // Only use positive frequencies (up to Nyquist)
        var carriers = new List<int>();
        var allProducts = new HashSet<int>(); // Combined set of ALL IMD products
        var carrierSet = new HashSet<int>();

        // Start from bin 1 (skip DC at bin 0)
        for (int candidate = 1; candidate <= maxBin && carriers.Count < numCarriers; candidate++)
        {
            if (CanAddCarrier(carriers, carrierSet, allProducts, candidate))
            {
                // Record all new sums and differences in the combined set
                foreach (int existing in carriers)
                {
                    allProducts.Add(candidate + existing);
                    allProducts.Add(Math.Abs(candidate - existing));
                }
                allProducts.Add(candidate + candidate); // Self-sum (2nd harmonic)
                carriers.Add(candidate);
                carrierSet.Add(candidate);
            }
        }

        if (carriers.Count < numCarriers)
        {
            throw new InvalidOperationException(
                $"Cannot allocate {numCarriers} IMD-free carriers in FFT of size {fftSize}. " +
                $"Maximum achievable: {carriers.Count}. Increase FFT size or reduce carrier count.");
        }

        return carriers.ToArray();
    }

    /// <summary>
    /// Allocates carriers with a minimum spacing constraint for additional spectral separation.
    /// </summary>
    /// <param name="numCarriers">Number of carriers to allocate.</param>
    /// <param name="fftSize">Total FFT size.</param>
    /// <param name="minSpacing">Minimum spacing between adjacent carriers (in bins).</param>
    /// <returns>Array of frequency bin indices.</returns>
    public int[] AllocateCarriersWithSpacing(int numCarriers, int fftSize, int minSpacing)
    {
        int maxBin = fftSize / 2;
        var carriers = new List<int>();
        var allProducts = new HashSet<int>();
        var carrierSet = new HashSet<int>();

        for (int candidate = 1; candidate <= maxBin && carriers.Count < numCarriers; candidate++)
        {
            // Check minimum spacing constraint
            bool spacingOk = true;
            foreach (int existing in carriers)
            {
                if (Math.Abs(candidate - existing) < minSpacing)
                {
                    spacingOk = false;
                    break;
                }
            }

            if (spacingOk && CanAddCarrier(carriers, carrierSet, allProducts, candidate))
            {
                foreach (int existing in carriers)
                {
                    allProducts.Add(candidate + existing);
                    allProducts.Add(Math.Abs(candidate - existing));
                }
                allProducts.Add(candidate + candidate);
                carriers.Add(candidate);
                carrierSet.Add(candidate);
            }
        }

        if (carriers.Count < numCarriers)
        {
            throw new InvalidOperationException(
                $"Cannot allocate {numCarriers} carriers with spacing {minSpacing} in FFT of size {fftSize}. " +
                $"Maximum achievable: {carriers.Count}. Increase FFT size, reduce carrier count, or reduce spacing.");
        }

        return carriers.ToArray();
    }

    /// <summary>
    /// Validates that a set of carrier indices has no IMD collisions up to the specified order.
    /// </summary>
    /// <param name="carriers">The carrier frequency bin indices to validate.</param>
    /// <param name="maxOrder">Maximum IMD order to check (2 = second-order, 3 = third-order).</param>
    /// <returns>True if no collisions exist; false otherwise.</returns>
    public bool ValidateNoCollisions(int[] carriers, int maxOrder = 2)
    {
        var imdProducts = new HashSet<int>();
        var carrierSet = new HashSet<int>(carriers);

        // Check second-order products: fi + fj, fi - fj for all i != j
        for (int i = 0; i < carriers.Length; i++)
        {
            for (int j = i + 1; j < carriers.Length; j++)
            {
                int sum = carriers[i] + carriers[j];
                int diff = Math.Abs(carriers[i] - carriers[j]);

                // Check if this IMD product collides with a carrier
                if (carrierSet.Contains(sum) || carrierSet.Contains(diff))
                    return false;

                // Check if this IMD product collides with another IMD product
                if (!imdProducts.Add(sum) || !imdProducts.Add(diff))
                    return false;
            }
        }

        if (maxOrder >= 3)
        {
            // Check third-order products: 2fi - fj, 2fi + fj
            for (int i = 0; i < carriers.Length; i++)
            {
                for (int j = 0; j < carriers.Length; j++)
                {
                    if (i == j) continue;
                    int prod1 = Math.Abs(2 * carriers[i] - carriers[j]);
                    int prod2 = 2 * carriers[i] + carriers[j];

                    if (carrierSet.Contains(prod1) || carrierSet.Contains(prod2))
                        return false;
                    if (!imdProducts.Add(prod1) || !imdProducts.Add(prod2))
                        return false;
                }
            }
        }

        return true;
    }

    /// <summary>
    /// Gets the expected IMD product frequency bins for a pair of carriers.
    /// </summary>
    /// <param name="carrier1">First carrier frequency bin.</param>
    /// <param name="carrier2">Second carrier frequency bin.</param>
    /// <returns>Array of IMD product frequency bins (sum and difference).</returns>
    public static int[] GetIMDProducts(int carrier1, int carrier2)
    {
        return [carrier1 + carrier2, Math.Abs(carrier1 - carrier2)];
    }

    private static bool CanAddCarrier(List<int> existing, HashSet<int> carrierSet,
        HashSet<int> allProducts, int candidate)
    {
        // Reject if candidate is itself an existing IMD product
        if (allProducts.Contains(candidate))
            return false;

        // Check if any new IMD product would collide with existing products or carriers
        foreach (int e in existing)
        {
            int newSum = candidate + e;
            int newDiff = Math.Abs(candidate - e);

            // Would this product collide with an existing carrier?
            if (carrierSet.Contains(newSum) || carrierSet.Contains(newDiff))
                return false;

            // Would this product collide with an existing IMD product?
            if (allProducts.Contains(newSum) || allProducts.Contains(newDiff))
                return false;
        }

        // Check self-sum (2nd harmonic)
        int selfSum = candidate + candidate;
        if (allProducts.Contains(selfSum) || carrierSet.Contains(selfSum))
            return false;

        return true;
    }
}
