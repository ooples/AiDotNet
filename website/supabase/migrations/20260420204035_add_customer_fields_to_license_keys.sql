-- Adds customer_email + customer_full_name to license_keys so an admin
-- can attribute a license to a real person *before* that person has
-- signed up on the website.
--
-- Context: 45 training-cohort licenses were issued directly via SQL
-- (issued_at 2026-04-06) while the signup flow was still broken. They
-- were all attributed to the synthetic ci@aidotnet.dev user_id since no
-- real auth.users rows existed at the time. The real attendees' names
-- ended up in the `notes` column as free text and their emails stayed
-- only in the AiTraining Azure DevOps org.
--
-- Constraints the user pinned for this migration:
--   - The `license_key` string column is immutable — customers have
--     those strings baked into env vars / ~/.aidotnet/license.key files
--     and anything that invalidates them breaks their workstations.
--   - `user_id` stays ci@aidotnet.dev. If we created 46 fake auth.users
--     rows to move the FK, any of those students signing up with their
--     real email later would end up with two accounts (the one we
--     pre-provisioned + the one Supabase just issued) and Supabase
--     doesn't merge on email.
--
-- So: add two nullable columns that the admin UI can prefer for display,
-- but that carry no FK weight. When the student does sign up, a separate
-- reconciliation pass (future migration) can reassign user_id on the
-- license row; the license key string and the customer_email column
-- don't need to change when that happens.

ALTER TABLE public.license_keys
  ADD COLUMN IF NOT EXISTS customer_email     TEXT,
  ADD COLUMN IF NOT EXISTS customer_full_name TEXT;

-- Index the email column so the (future) reconciliation trigger can look
-- up "which licenses belong to this brand-new auth.users row" in O(log n)
-- when a new profile is created. No unique constraint — a single customer
-- can legitimately hold multiple licenses, and the uniqueness invariant
-- is license_key itself, not the email.
CREATE INDEX IF NOT EXISTS idx_license_keys_customer_email
  ON public.license_keys (customer_email)
  WHERE customer_email IS NOT NULL;

-- Lightweight sanity constraint: if customer_email is set, it must look
-- like an email. This is deliberately *not* the full RFC 5322 regex —
-- that's an O(1) sanity check, not an authenticator. The auth stack
-- (Supabase) is the source of truth for address validity.
ALTER TABLE public.license_keys
  DROP CONSTRAINT IF EXISTS license_keys_customer_email_format;
ALTER TABLE public.license_keys
  ADD CONSTRAINT license_keys_customer_email_format
  CHECK (customer_email IS NULL OR customer_email ~* '^[^@\s]+@[^@\s]+\.[^@\s]+$');

COMMENT ON COLUMN public.license_keys.customer_email IS
  'Contact email for the person this license was issued to. Set when a license is issued before the customer has signed up (e.g., training cohort). Separate from user_id/profile so admin UI can attribute rows without forging auth.users rows.';

COMMENT ON COLUMN public.license_keys.customer_full_name IS
  'Display name of the license owner when user_id points to a synthetic/bootstrap account (e.g., ci@aidotnet.dev for bulk-issued training licenses).';
