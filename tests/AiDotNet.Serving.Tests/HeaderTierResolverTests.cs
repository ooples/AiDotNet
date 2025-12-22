using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Security;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Options;
using Xunit;

namespace AiDotNet.Serving.Tests;

public class HeaderTierResolverTests
{
    [Fact]
    public void Constructor_Throws_WhenOptionsNull()
    {
        Assert.Throws<ArgumentNullException>(() => new HeaderTierResolver(options: null!));
    }

    [Fact]
    public void ResolveTier_ReturnsEnterprise_WhenEnforcementDisabled()
    {
        var resolver = new HeaderTierResolver(Options.Create(new TierEnforcementOptions { Enabled = false }));

        var tier = resolver.ResolveTier(httpContext: null!);

        Assert.Equal(SubscriptionTier.Enterprise, tier);
    }

    [Fact]
    public void ResolveTier_ReturnsDefault_WhenContextNull()
    {
        var resolver = new HeaderTierResolver(Options.Create(new TierEnforcementOptions
        {
            Enabled = true,
            DefaultTier = SubscriptionTier.Free
        }));

        var tier = resolver.ResolveTier(httpContext: null!);

        Assert.Equal(SubscriptionTier.Free, tier);
    }

    [Fact]
    public void ResolveTier_ReturnsDefault_WhenHeaderMissing()
    {
        var options = new TierEnforcementOptions
        {
            Enabled = true,
            TierHeaderName = "X-Tier",
            DefaultTier = SubscriptionTier.Free
        };
        var resolver = new HeaderTierResolver(Options.Create(options));
        var context = new DefaultHttpContext();

        var tier = resolver.ResolveTier(context);

        Assert.Equal(SubscriptionTier.Free, tier);
    }

    [Fact]
    public void ResolveTier_ReturnsParsedTier_WhenHeaderValid()
    {
        var options = new TierEnforcementOptions
        {
            Enabled = true,
            TierHeaderName = "X-Tier",
            DefaultTier = SubscriptionTier.Free
        };
        var resolver = new HeaderTierResolver(Options.Create(options));
        var context = new DefaultHttpContext();
        context.Request.Headers[options.TierHeaderName] = "Pro";

        var tier = resolver.ResolveTier(context);

        Assert.Equal(SubscriptionTier.Pro, tier);
    }

    [Fact]
    public void ResolveTier_ReturnsDefault_WhenHeaderInvalid()
    {
        var options = new TierEnforcementOptions
        {
            Enabled = true,
            TierHeaderName = "X-Tier",
            DefaultTier = SubscriptionTier.Free
        };
        var resolver = new HeaderTierResolver(Options.Create(options));
        var context = new DefaultHttpContext();
        context.Request.Headers[options.TierHeaderName] = "not-a-tier";

        var tier = resolver.ResolveTier(context);

        Assert.Equal(SubscriptionTier.Free, tier);
    }
}
