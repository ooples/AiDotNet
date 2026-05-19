using Xunit;

namespace AiDotNet.Serving.Tests;

/// <summary>
/// Collection definition for serving integration tests to ensure proper test isolation.
/// This ensures all tests in this collection run sequentially and clean up the singleton repository.
/// </summary>
/// <remarks>
/// Uses <see cref="ServingTestWebApplicationFactory"/> (a derivative of
/// <c>WebApplicationFactory&lt;Program&gt;</c>) so the test auth handler
/// is wired in. Necessary after PR #1384 added a default-auth
/// FallbackPolicy to the serving host — without the test handler every
/// existing integration-test request would return 401.
/// </remarks>
[CollectionDefinition("ServingIntegrationTests")]
public class ServingIntegrationTestCollection : ICollectionFixture<ServingTestWebApplicationFactory>
{
}
