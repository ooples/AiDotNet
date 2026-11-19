-- A subscription keeps one durable entitlement across plan/status changes.
-- Fail on legacy duplicates rather than silently choosing or deleting a key.
create unique index if not exists idx_license_keys_subscription_product
  on public.license_keys (stripe_subscription_id, product)
  where stripe_subscription_id is not null;
