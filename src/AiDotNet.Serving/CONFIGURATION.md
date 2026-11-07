# AiDotNet.Serving Configuration Guide

## Azure App Service Environment Variables

### Database Connection

| Variable | Required | Description |
|----------|----------|-------------|
| `ServingPersistenceOptions__Provider` | No | Database provider: `Sqlite` (default), `PostgreSql`, `SqlServer`, `MySql` |
| `ServingPersistenceOptions__ConnectionString` | No | EF Core connection string. Empty with SQLite uses `data/aidn-serving.db` beside the app |
| `ServingPersistenceOptions__MySqlServerVersion` | No | MySQL server version, e.g. `8.0.21` (default `8.0.0`) |
| `ServingPersistenceOptions__MigrateOnStartup` | No | Apply EF Core migrations before the server accepts requests (default `true`) |

Example (PostgreSQL):
```
ServingPersistenceOptions__Provider=PostgreSql
ServingPersistenceOptions__ConnectionString=Host=myserver;Database=aidotnet;Username=app;Password=secret
```

### Stripe Payment Integration

| Variable | Required | Description |
|----------|----------|-------------|
| `Stripe__SecretKey` | Yes | Stripe secret API key (`sk_live_...` or `sk_test_...`) |
| `Stripe__WebhookSigningSecret` | Yes | Stripe webhook endpoint signing secret (`whsec_...`) |
| `Stripe__ProPriceId` | Yes | Stripe Price ID for Pro tier monthly (`price_...`) |
| `Stripe__ProAnnualPriceId` | Yes | Stripe Price ID for Pro tier annual (`price_...`) |
| `Stripe__EnterprisePriceId` | Yes | Stripe Price ID for Enterprise tier (`price_...`) |
| `Stripe__SuccessUrl` | Yes | Redirect URL after successful checkout |
| `Stripe__CancelUrl` | Yes | Redirect URL after cancelled checkout |

### Authentication

Every endpoint except `/health`, Stripe checkout/webhook and license validation needs an API key or, when enabled, a
JWT. The `ServingSecurity` section is validated at startup: an invalid value stops the server with an error naming it.

**API keys.** A key has the form `aidn.<keyId>.<secret>` and is sent in the API key header or as
`Authorization: Bearer <key>`. Keys created through `POST /api/admin/api-keys` live in the database. Config-backed
(static) keys hold only a PBKDF2-SHA256 hash; mint one, with its configuration entry, with
`StaticApiKeys.Generate(...)`. A static entry decides its own key id: wrong secret, `Revoked` or a past `ExpiresAt` refuse it.

| Variable | Required | Description |
|----------|----------|-------------|
| `ServingSecurity__ApiKeys__HeaderName` | No | API key header (default `X-AiDotNet-ApiKey`). The legacy `ApiKeyAuthenticationOptions__HeaderName` still overrides it, with a startup warning |
| `ServingSecurity__ApiKeys__Pbkdf2Iterations` | No | PBKDF2 iterations for newly created keys (default 600,000; existing keys keep theirs) |
| `ServingSecurity__ApiKeys__HmacSecrets__0` | No | HMAC secrets, newest first. When set, every static key must carry `HmacIdBase64` = HMAC-SHA256(secret, keyId) under one of them |
| `ServingSecurity__ApiKeys__StaticApiKeys__0__Prefix` | No | Static key id, then `__HmacIdBase64`, `__Pbkdf2SaltBase64`, `__Pbkdf2HashBase64`, `__Pbkdf2Iterations`, `__Tier` (`Free`, `Premium` = Pro, `Enterprise`), `__ExpiresAt`, `__Revoked` |

**JWT.** With `Enabled`, a bearer token that is not an `aidn.` key is validated as a JWT, against an OpenID Connect
authority or a symmetric signing key. Claim names are kept as issued.

| Variable | Required | Description |
|----------|----------|-------------|
| `ServingSecurity__JwtAuthentication__Enabled` | No | Accept JWT bearer callers (default `false`) |
| `ServingSecurity__JwtAuthentication__Authority` | One of these two | OAuth2/OIDC authority URL (signing keys and issuer from its metadata) |
| `ServingSecurity__JwtAuthentication__SigningKey` | One of these two | Symmetric HMAC-SHA256 key, base64 or UTF-8, at least 32 bytes |
| `ServingSecurity__JwtAuthentication__Issuer` | With `SigningKey` | Expected issuer (overrides the authority's when both are set) |
| `ServingSecurity__JwtAuthentication__Audience` | Yes, when enabled | Expected audience |
| `ServingSecurity__JwtAuthentication__RequireHttpsMetadata` | No | Fetch authority metadata over HTTPS only (default `true`) |
| `ServingSecurity__Jwt__TierClaimType` | No | Claim holding the caller's tier, `Free`/`Pro`/`Enterprise` (default `tier`) |
| `ServingSecurity__Admin__ClaimType` / `__ClaimValue` | No | Claim that grants a JWT caller the admin endpoints (default `role` = `admin`). API keys need the `Admin` scope |

### Model Serving

| Variable | Required | Description |
|----------|----------|-------------|
| `ServingOptions__Port` | No | Server port (default: 52432; the `PORT` variable takes precedence) |
| `ServingOptions__ModelDirectory` | No | Root directory for model files (default: `models`) |
| `ServingOptions__MaxBatchSize` | No | Maximum inference batch size (default: 100) |

## GitHub Repository Secrets

All secrets required for CI/CD pipelines:

| Secret | Used By | Description |
|--------|---------|-------------|
| `AIDOTNET_BUILD_KEY` | `automated-release.yml` | 32-byte hex-encoded signing key for Layer 1 build signing and Layer 3 integrity hash |
| `NUGET_API_KEY` | `automated-release.yml` | NuGet.org API key for package publishing |
| `AUTOFIX_PAT` | `autofix.yml` | GitHub PAT for automated fix commits |
| `AUTOFIX_GPG_PRIVATE_KEY` | `autofix.yml` | GPG private key for signed autofix commits |
| `AUTOFIX_GPG_PASSPHRASE` | `autofix.yml` | Passphrase for the GPG private key |
| `AZURE_WEBAPP_PUBLISH_PROFILE` | `deploy.yml` | Azure App Service publish profile for the Serving API |
| `AZURE_FUNCTIONAPP_PUBLISH_PROFILE` | `deploy.yml` | Azure Functions publish profile |
| `CODACY_PROJECT_TOKEN` | `quality.yml` | Codacy project analysis token |
| `CODECOV_TOKEN` | `quality.yml` | Codecov upload token |
| `SONAR_TOKEN` | `quality.yml` | SonarCloud analysis token |
| `OPENAI_API_KEY` | Tests | OpenAI API key for integration tests |
| `VERCEL_TOKEN` | Website deploy | Vercel deployment token |
| `VERCEL_ORG_ID` | Website deploy | Vercel organization ID |
| `VERCEL_PROJECT_ID` | Website deploy | Vercel project ID |

## Local Development Setup

### 1. Database (SQLite - no setup needed)

SQLite is the default provider. The database file is created automatically.

### 2. Stripe (test mode)

1. Create a [Stripe test account](https://dashboard.stripe.com/test/dashboard)
2. Copy the test secret key from Developers > API keys
3. Create test products and prices in the Stripe dashboard
4. Set up a webhook endpoint pointing to `https://localhost:{port}/api/webhooks/stripe`
5. Copy the webhook signing secret

### 3. User Secrets

```bash
cd src/AiDotNet.Serving

dotnet user-secrets set "Stripe:SecretKey" "sk_test_..."
dotnet user-secrets set "Stripe:WebhookSigningSecret" "whsec_..."
dotnet user-secrets set "Stripe:ProPriceId" "price_..."
dotnet user-secrets set "Stripe:ProAnnualPriceId" "price_..."
dotnet user-secrets set "Stripe:EnterprisePriceId" "price_..."
dotnet user-secrets set "Stripe:SuccessUrl" "https://localhost:5001/success"
dotnet user-secrets set "Stripe:CancelUrl" "https://localhost:5001/cancel"
```

### 4. Build Key (optional for development)

Development builds work without a build key (Layer 1 signing and Layer 3 integrity checks are skipped). To test with a build key:

```bash
mkdir -p src/BuildKey
openssl rand 32 > src/BuildKey/BuildKey.bin

# Generate integrity hash
openssl dgst -sha256 -mac HMAC -macopt hexkey:$(xxd -p -c 256 src/BuildKey/BuildKey.bin | tr -d '\n') -binary < src/BuildKey/BuildKey.bin > src/IntegrityHash/IntegrityHash.bin
```

### 5. Run

```bash
dotnet run --project src/AiDotNet.Serving
```

## Three-Layer Encryption Architecture

| Layer | Component | Key Source | CI Requirement |
|-------|-----------|-----------|----------------|
| Layer 1 | `BuildKeyProvider` | `AIDOTNET_BUILD_KEY` secret embedded as `BuildKey.bin` | Secret must be configured |
| Layer 2 | `LicenseService` escrow | Server-side `EscrowSecret` + client `LicenseKey` | Database tables must exist |
| Layer 3 | `AssemblyIntegrityChecker` | HMAC of build key embedded as `IntegrityHash.bin` | CI step must generate hash |
