using AiDotNet.Interfaces;
using AiDotNet.PromptEngineering.FewShot;
using Xunit;

namespace AiDotNet.Tests.PromptEngineering;

public class RandomExampleSelectorTests
{
    [Fact]
    public void Constructor_WithSeed_CreatesReproducibleSelector()
    {
        var selector1 = new RandomExampleSelector<double>(seed: 42);
        var selector2 = new RandomExampleSelector<double>(seed: 42);

        selector1.AddExample(new FewShotExample { Input = "A", Output = "1" });
        selector1.AddExample(new FewShotExample { Input = "B", Output = "2" });
        selector1.AddExample(new FewShotExample { Input = "C", Output = "3" });
        selector1.AddExample(new FewShotExample { Input = "D", Output = "4" });

        selector2.AddExample(new FewShotExample { Input = "A", Output = "1" });
        selector2.AddExample(new FewShotExample { Input = "B", Output = "2" });
        selector2.AddExample(new FewShotExample { Input = "C", Output = "3" });
        selector2.AddExample(new FewShotExample { Input = "D", Output = "4" });

        var examples1 = selector1.SelectExamples("test", 2);
        var examples2 = selector2.SelectExamples("test", 2);

        Assert.Equal(examples1[0].Input, examples2[0].Input);
        Assert.Equal(examples1[1].Input, examples2[1].Input);
    }

    [Fact]
    public void SelectExamples_ReturnsRequestedCount()
    {
        var selector = new RandomExampleSelector<double>();

        selector.AddExample(new FewShotExample { Input = "A", Output = "1" });
        selector.AddExample(new FewShotExample { Input = "B", Output = "2" });
        selector.AddExample(new FewShotExample { Input = "C", Output = "3" });

        var examples = selector.SelectExamples("test", 2);

        Assert.Equal(2, examples.Count);
    }

    [Fact]
    public void SelectExamples_WithCountGreaterThanAvailable_ReturnsAllExamples()
    {
        var selector = new RandomExampleSelector<double>();

        selector.AddExample(new FewShotExample { Input = "A", Output = "1" });
        selector.AddExample(new FewShotExample { Input = "B", Output = "2" });

        var examples = selector.SelectExamples("test", 5);

        Assert.Equal(2, examples.Count);
    }

    [Fact]
    public void AddExample_IncreasesExampleCount()
    {
        var selector = new RandomExampleSelector<double>();

        selector.AddExample(new FewShotExample { Input = "A", Output = "1" });

        Assert.Equal(1, selector.ExampleCount);
    }

    [Fact]
    public void AddExample_WithNullExample_ThrowsArgumentNullException()
    {
        var selector = new RandomExampleSelector<double>();

        Assert.Throws<ArgumentNullException>(() => selector.AddExample(null!));
    }

    [Fact]
    public void RemoveExample_DecreasesExampleCount()
    {
        var selector = new RandomExampleSelector<double>();
        var example = new FewShotExample { Input = "A", Output = "1" };

        selector.AddExample(example);
        var removed = selector.RemoveExample(example);

        Assert.True(removed);
        Assert.Equal(0, selector.ExampleCount);
    }

    [Fact]
    public void GetAllExamples_ReturnsAllAddedExamples()
    {
        var selector = new RandomExampleSelector<double>();

        selector.AddExample(new FewShotExample { Input = "A", Output = "1" });
        selector.AddExample(new FewShotExample { Input = "B", Output = "2" });

        var all = selector.GetAllExamples();

        Assert.Equal(2, all.Count);
    }
}
