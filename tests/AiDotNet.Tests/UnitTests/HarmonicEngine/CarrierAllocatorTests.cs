using AiDotNet.HarmonicEngine.Core;
using Xunit;

namespace AiDotNetTests.UnitTests.HarmonicEngine;

/// <summary>
/// Tests that the carrier allocator produces IMD-collision-free frequency assignments.
/// </summary>
public class CarrierAllocatorTests
{
    [Theory]
    [InlineData(4, 256)]
    [InlineData(8, 1024)]
    [InlineData(16, 4096)]
    public void AllocateCarriers_ProducesCorrectCount(int numCarriers, int fftSize)
    {
        var allocator = new CarrierAllocator();
        var carriers = allocator.AllocateCarriers(numCarriers, fftSize);

        Assert.Equal(numCarriers, carriers.Count);
    }

    [Theory]
    [InlineData(4, 256)]
    [InlineData(8, 1024)]
    [InlineData(16, 4096)]
    public void AllocateCarriers_NoSecondOrderCollisions(int numCarriers, int fftSize)
    {
        var allocator = new CarrierAllocator();
        var carriers = allocator.AllocateCarriers(numCarriers, fftSize);

        Assert.True(allocator.ValidateNoCollisions(carriers, maxOrder: 2),
            $"Carrier allocation for N={numCarriers}, FFT={fftSize} has second-order IMD collisions");
    }

    [Fact]
    public void ValidateNoCollisions_ThirdOrder_RejectsKnownCollision()
    {
        // Hand-picked carriers with a known third-order collision:
        // carriers {1, 2, 3} → 2*1 - 3 = -1 (abs=1) which equals carrier 1
        var allocator = new CarrierAllocator();

        // These carriers are collision-free at second order but have a third-order collision
        int[] colliding = [1, 5, 9]; // 2*5 - 9 = 1, which is carrier[0]
        Assert.False(allocator.ValidateNoCollisions(colliding, maxOrder: 3),
            "Should detect third-order IMD collision: 2*5 - 9 = 1 hits carrier[0]");
    }

    [Fact]
    public void ValidateNoCollisions_SecondOrder_AcceptsAllocatedCarriers()
    {
        // Carriers from the allocator should always be second-order collision-free
        var allocator = new CarrierAllocator();
        var carriers = allocator.AllocateCarriers(8, 4096);

        Assert.True(allocator.ValidateNoCollisions(carriers, maxOrder: 2),
            "Allocated carriers should be second-order collision-free");
    }

    [Fact]
    public void AllocateCarriers_AllPositiveAndWithinBounds()
    {
        var allocator = new CarrierAllocator();
        int fftSize = 1024;
        var carriers = allocator.AllocateCarriers(16, fftSize);

        foreach (int c in carriers)
        {
            Assert.True(c > 0, "Carrier should be positive (skip DC)");
            Assert.True(c < fftSize / 2, "Carrier should be below Nyquist");
        }
    }

    [Fact]
    public void AllocateCarriers_AllUnique()
    {
        var allocator = new CarrierAllocator();
        var carriers = allocator.AllocateCarriers(16, 1024);

        Assert.Equal(carriers.Count, carriers.Distinct().Count());
    }

    [Fact]
    public void AllocateCarriersWithSpacing_RespectsMinimumSpacing()
    {
        var allocator = new CarrierAllocator();
        int minSpacing = 5;
        var carriers = allocator.AllocateCarriersWithSpacing(8, 1024, minSpacing);

        for (int i = 0; i < carriers.Count; i++)
        {
            for (int j = i + 1; j < carriers.Count; j++)
            {
                Assert.True(Math.Abs(carriers[i] - carriers[j]) >= minSpacing,
                    $"Carriers {carriers[i]} and {carriers[j]} are closer than minimum spacing {minSpacing}");
            }
        }
    }

    [Fact]
    public void GetIMDProducts_ReturnsCorrectSumAndDifference()
    {
        var products = CarrierAllocator.GetIMDProducts(10, 17);

        Assert.Contains(27, products); // 10 + 17
        Assert.Contains(7, products);  // |10 - 17|
    }
}
