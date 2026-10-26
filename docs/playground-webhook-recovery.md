# Playground webhook recovery

## Incident evidence (2026-10-26)

The running `aidotnet-playground` Function App in `aidotnet-playground_group`
returns 404 for POST `/api/stripe-webhook`. Azure's function list and deployed
`functions.metadata` contain only `execute`. The active package pointer is
`20260124021456.zip`; retained package metadata dates to January 2026.
Deployment run `35116514175` built successfully but failed with Kudu 401 using
a publish profile. SCM basic publishing authentication is disabled.

The inspected course-session records show read-only Azure inventory, not a
deployment to this app. No evidence was found that those actions removed the
webhook. Retained deployment/activity records are incomplete, so this is not
a claim about all historical changes.

Two enabled Stripe sandbox endpoints target the same URL:
`we_1T7LHJBCSxBmC8eojsJel9NC` (account default API version) and
`we_1T7LG7BCSxBmC8eoi7ScoXwU` (explicit `2026-02-25.clover`). The account default
and recent events also use Clover. Only one signing secret is configured in
Azure; endpoint listing does not reveal which registration owns it.

## Recovery contents

Both original HTTP POST routes and the `/api` prefix are preserved. Package
validation rejects a release missing either function or its runtime assembly.
The webhook rejects absent signatures before calling Stripe.net. Signed Clover
events and the SDK's current event version are accepted; other versions are
rejected. Stripe signature verification and its default timestamp tolerance
remain enabled. Tests cover Clover checkout processing, subscription fields
consumed by the existing handler, invalid/expired signatures, missing settings,
unsupported versions, and execution of `System.Console.WriteLine(6 * 7);`.

The deployment workflow uses Azure federation instead of the disabled password
mechanism. It fails early until federation is configured. Do not enable SCM
basic authentication as a workaround.

## Federated deployment setup (requires approval)

1. Use a dedicated Azure deployment identity in the CrowdTrainer tenant. Create
   a GitHub federated credential with issuer `https://token.actions.githubusercontent.com`,
   audience `api://AzureADTokenExchange`, and subject
   `repo:ooples/AiDotNet:ref:refs/heads/master`. If deployment from `main` is
   actually needed, authorize that branch separately; do not use a wildcard.
2. Grant the identity Website Contributor scoped only to the existing Function
   App resource, not subscription-wide Contributor. Verify any additional
   permissions requested by the deployment action before broadening access.
3. Set repository variables `AZURE_PLAYGROUND_CLIENT_ID`, `AZURE_TENANT_ID`,
   and `AZURE_SUBSCRIPTION_ID`. These are identifiers, not credentials.
4. Run the workflow from the authorized branch after the recovery is approved.
   It tests, publishes, checks the package, deploys, and checks both routes.

An approved incident deployment can instead use the operator's existing Azure
CLI identity and the locally verified ZIP, without waiting for federation or
merging this draft. Preserve all existing application settings and routes.

## Deployment and rollback procedure (requires approval)

1. Record the target subscription, app, current package pointer and application
   configuration securely. Download the current package for rollback using
   Azure bearer authentication to Kudu; never log credentials/settings values.
2. Verify the release ZIP hash against the reviewed artifact. ZIP entries must
   start at `host.json`/`functions.metadata`, without a containing directory or
   `local.settings.json`. Deploy only to `aidotnet-playground`.
3. Confirm Azure lists both functions, then run
   `python scripts/check-playground-routes.py https://aidotnet-playground-emfxa0abhzdqfyh3.westeurope-01.azurewebsites.net`.
   Both empty unsigned POST requests must return 400. These checks do not
   prove successful Stripe delivery or database updates.
4. In Stripe, identify the registration matching the configured signing secret.
   With explicit approval, reconcile duplicate registrations and resend a
   suitable failed test event. Verify its delivery response and intended profile
   change. Do not rotate secrets or disable either registration speculatively.
5. If the execute route regresses, redeploy the captured previous ZIP and verify
   execute again. That package lacks the webhook: rollback restores the old
   execute service but reintroduces webhook 404, and is not incident resolution.

## Local verification

Run the Functions test project in Release with `GeneratePackageOnBuild=false`.
Publish the tested output with `--no-build --no-restore`. Run
`python scripts/test_playground_package.py` and
`python scripts/verify-playground-package.py <publish-directory>`.
Start Azure Functions Core Tools in the published directory with
`FUNCTIONS_WORKER_RUNTIME=dotnet-isolated`, `AzureWebJobsSecretStorageType=files`,
and a dummy `STRIPE_WEBHOOK_SECRET`. Do not use production Stripe/Supabase keys.

Recovery verification passed: 16 .NET tests and five package-check tests.
Core Tools 4.6.0 registered both routes. Actual local HTTP requests returned
400 for unsigned, invalidly signed and expired webhook requests; 200 for a
valid synthetic Clover unhandled event; 400 for empty execute input; and 200
with output `42` for the execution check. This does not establish that both
remote Stripe registrations have the right signing secret, nor validate a live
subscription-to-database transaction. Those remain deployment checks.
