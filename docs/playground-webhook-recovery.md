# Playground webhook recovery

## Durable event processing

Apply `website/supabase/migrations/20261027000000_playground_stripe_event_state.sql`
and the corrective `20261027010000_playground_stripe_ordering_and_tiers.sql`
before deploying this handler. Refresh PostgREST's schema cache after a manual
migration (`notify pgrst, 'reload schema';`). The migration adds two service-only
tables and three restricted RPCs; it does not rewrite existing profile data.

Each profile has a two-minute database lease. Competing workers return retryable
500 responses; expired leases can be reclaimed, and a token prevents an old
worker from committing or releasing a replacement lease. Successful completion
updates the profile, advances its event timestamp and records the Stripe event
ID in one transaction. Failed writes roll back all three. Replays from either
endpoint are acknowledged without another profile update. Event timestamps are
retained for diagnostics, not used to reject claims: a replacement checkout can
precede the old subscription's cancellation.

Stripe event timestamps have second precision, so event IDs are not used to
break ties. While holding the lease, the handler retrieves the current Stripe
subscription for checkout, update and deletion events. This also prevents a
delayed checkout from restoring a canceled subscription. Subscription creation
timestamps prevent a delayed cancellation of an older subscription from revoking
its replacement. Only `active` and `trialing` grant `pro`; all other Stripe
statuses grant `free`, with `trialing` stored as `active`. The existing single-subscription-per-profile model is retained.

The original handler passed 37 Functions tests. The PostgreSQL regression suite
also covers delayed replacement checkout and all eight subscription statuses. Run
`python scripts/test_playground_event_state.py` with Docker available; it creates
and removes an isolated PostgreSQL container and tests concurrent claims,
duplicates, stale events, same-second reconciliation, expired-worker fencing,
transaction rollback, replacement subscriptions and role restrictions. This check
also runs in the deployment workflow.

The migration was applied and recorded in Supabase migration history on
2026-10-27. Deployment `81b55279c7da4a429326e69a51b614e3` contains the durable
handler (ZIP SHA-256
`023949f6fed82d228b08bc3e092b63a61e13efa51070514d08cf65589649e7e6`).
Real Stripe subscription update/delete events updated the exact test profile.
Signed replays through both secrets left its timestamp unchanged with one receipt
per event. Before the corrective migration, an older active event was recorded as stale
without changing the canceled profile. The correction instead reconciles current
Stripe state even for older event timestamps. Same-second active snapshots, delayed checkout and concurrent
duplicate/retry controls all preserved `free/canceled`; the concurrent event had
exactly one receipt. All temporary subscription/customer/user/profile fixtures
were cleaned up. Signature rejection and execute-output-42 checks still passed.

Rollback: the preceding Functions package can still run with these additive
tables present, but it does not enforce ordering or deduplication. Keep the
tables and receipts when rolling back; do not erase the processing history.


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

## Recovery progress (2026-10-27)

The four review findings are addressed: customer-lookup configuration/API failures
now return retryable 500 responses; checkout tests verify the exact profile and
subscription values; package validation requires the isolated-worker artifacts;
and route checks resolve the deployed app's hostname. Cold-start read timeouts
are retried by the route checker.

Verification passed: 29 Functions tests, seven package-validator tests, and two
route-checker tests. The original failure behavior was also observed against the
old Functions binary: all four new customer-lookup failure cases returned 200
instead of the required 500. All four pass with the updated binary.

The `aidotnet-playground-github` Azure application is configured for GitHub OIDC
from `repo:ooples/AiDotNet:ref:refs/heads/master`, with Website Contributor scoped
only to this Function App. The three repository variables are set. No publishing
password or client secret was created. The master-branch GitHub deployment has
not yet run with this identity; the recovery deployment used the operator's
existing Azure CLI identity.

The previous ZIP (`20260124021456.zip`, 435,735,453 bytes) was downloaded before
deployment. Its SHA-256 is
`db8954ea1053eb8df5747e6f89a29b815e0701c09339c17a3818148589297ef3`.
Its cached copy occupied most of the Windows Consumption worker's 500 MB
temporary disk. Portable and initial Windows-specific uploads failed for lack
of temporary space. Only the incomplete uploads from these attempts were removed.

Publishing for `win-x64`, framework-dependent, without reference XML documentation
reduced the ZIP to 85,196,805 bytes while retaining binaries and PDBs. The app is
configured for 64-bit workers and .NET 10. The resulting package deployed
successfully (`f2772c964d4c485eb5e5282edcc135b1`), with SHA-256
`45a69ea8e8cf94387d96a48e06385bb4a5f8e1e6eb6ba9e6d3efbb0fddeaa33d`.
Azure now lists both functions. Live checks returned 400 for empty requests,
200 for a valid signed unhandled Clover event, 400 for invalid/expired signatures,
and 200 with output `42` for actual code execution.

Live verification exposed a Supabase Auth failure: listing users returned 500.
Auth logs identified `confirmation_token` NULL-to-string conversion failures.
An aggregate SQL check found two malformed rows with NULL confirmation, recovery,
email-change token and email-change fields. A transaction guarded by the exact
two-row count normalized only those NULL fields to empty strings. No user was
deleted or granted access. The Auth admin list endpoint then returned 200.
Signed subscription controls verified the exact temporary profile/customer and
`pro/past_due` then `free/canceled` values; all temporary resources were removed.

Dashboard comparison matched the original Azure signing secret to
`we_1T7LHJBCSxBmC8eojsJel9NC`. The owner requested preserving both registrations
and approved optional `STRIPE_WEBHOOK_SECRET_SECONDARY` support for
`we_1T7LG7BCSxBmC8eoi7ScoXwU`. The original primary setting remains required;
the secondary secret is tried only after primary verification fails, with the
same signature, timestamp and API-version checks. Tests verify exact profile
updates signed by either secret and reject unknown secrets, expired signatures
and unsupported versions. Neither endpoint's secret was rotated.

The final package with secondary-secret support deployed successfully as
`82c041874400457b95e9b8895920fb6e` (84,567,738 bytes; SHA-256
`efb331398e3e9763696fde495420744127242b97a2ae0c6613b651a4debbcf99`).
Live checks accepted signatures from either configured secret, rejected an
unknown secret and expired signatures from both, and executed code with output
`42`. Actual Stripe sandbox events `evt_1UVDsxBCSxBmC8eod49akgBk` (subscription
updated) and `evt_1UVDt3BCSxBmC8eowOIK2oNP` (subscription deleted) changed the
exact temporary profile/customer to `pro/active` then `free/canceled`. Both
events reported zero pending webhook deliveries; each existing endpoint's
dashboard showed HTTP 200 for both events. The trial subscription was
canceled and the temporary customer, Auth user and profile were removed.

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
Publish the Functions host with `-r win-x64 --self-contained false`,
`-p:BuildProjectReferences=false -p:GeneratePackageOnBuild=false`, and
`-p:AllowedReferenceRelatedFileExtensions=.pdb` to reuse the tested library and
exclude reference XML documentation from the runtime package. Run
`python scripts/test_playground_package.py` and
`python scripts/test_playground_routes.py`, then
`python scripts/verify-playground-package.py <publish-directory>`.
Start Azure Functions Core Tools in the published directory with
`FUNCTIONS_WORKER_RUNTIME=dotnet-isolated`, `AzureWebJobsSecretStorageType=files`,
and a dummy `STRIPE_WEBHOOK_SECRET`. Do not use production Stripe/Supabase keys.

Initial draft verification passed: 16 .NET tests and five package-check tests.
Core Tools 4.6.0 registered both routes. Actual local HTTP requests returned
400 for unsigned, invalidly signed and expired webhook requests; 200 for a
valid synthetic Clover unhandled event; 400 for empty execute input; and 200
with output `42` for the execution check. This does not establish that both
remote Stripe registrations have the right signing secret, nor validate a live
subscription-to-database transaction. The recovery progress above records those
subsequent live checks separately.
