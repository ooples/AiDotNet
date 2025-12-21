using Microsoft.AspNetCore.Mvc.Testing;
using Xunit;

namespace AiDotNet.Serving.Tests;

/// <summary>
/// Collection definition for serving integration tests to ensure proper test isolation.
/// This ensures all tests in this collection run sequentially and clean up the singleton repository.
/// </summary>
[CollectionDefinition("ServingIntegrationTests")]
public class ServingIntegrationTestCollection : ICollectionFixture<WebApplicationFactory<Program>>
{
}
