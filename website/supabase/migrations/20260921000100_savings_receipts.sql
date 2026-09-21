-- Migration: savings receipts, with the counterfactual on the row
--
-- WHY NOT `api_usage`. That table is described as serving "analytics and
-- billing" and carries user_id, endpoint, model_id, status_code, latency_ms
-- and created_at. There is no token count, no cost, and no project, so a
-- savings figure cannot be derived from it at any granularity a buyer would
-- accept. Adding four columns to it would also conflate two different things:
-- api_usage records that a request happened, whereas a receipt records what a
-- request WOULD have cost had the optimiser been absent.
--
-- WHAT MAKES THIS A RECEIPT RATHER THAN AN ESTIMATE. Every competitor in this
-- space reports savings as a counterfactual computed from its own assumptions:
-- "we removed N tokens, N tokens times the list price is your saving". That is
-- unfalsifiable. Token Optimizer withholds a fraction of traffic as an
-- unshaped control arm, so `arm` on this row says whether the request was
-- optimised or held out, and a saving is the DIFFERENCE between the two arms
-- on comparable work rather than a multiplication. The column is not
-- nullable: a row that cannot say which arm it belongs to cannot support a
-- claim, and letting it in would quietly reintroduce the estimate.
--
-- TOKENS ARE SPLIT BY CLASS because they are not fungible. A provider bills a
-- cache read at roughly a tenth of uncached input and a cache write at rather
-- more than it, so a single "tokens" number can move opposite to the invoice.
-- This package has measured exactly that inversion, which is why the class
-- split is part of the schema and not a reporting detail.
--
-- COST IS STORED, NOT COMPUTED ON READ. Prices change, and a receipt issued in
-- March must still say what March cost. `effective_cost_usd` is what the
-- request actually billed at the rates in force when it ran.

create table if not exists public.savings_receipts (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null references auth.users(id) on delete cascade,

    -- Which arm this request belonged to. The whole point of the table.
    arm text not null check (arm in ('optimized', 'holdout')),

    -- Attribution a buyer can act on: which project, and which team owns it.
    -- Hashed rather than named -- a repository path is frequently a customer's
    -- or an employee's name, and a receipt does not need to know it to be
    -- attributable. The customer holds the mapping.
    project_hash text not null check (char_length(project_hash) between 1 and 128),
    team text check (char_length(team) <= 200),

    model_id text,

    -- Token classes, kept separate because they bill at different rates.
    input_tokens bigint not null default 0 check (input_tokens >= 0),
    output_tokens bigint not null default 0 check (output_tokens >= 0),
    cache_read_tokens bigint not null default 0 check (cache_read_tokens >= 0),
    cache_write_tokens bigint not null default 0 check (cache_write_tokens >= 0),

    -- What this request actually cost at the rates in force when it ran.
    -- Nullable on purpose: a subscription or an enterprise agreement can make
    -- the marginal cost genuinely unknown, and a null is honest where a zero
    -- would be a lie that averages into every report built on top of it.
    effective_cost_usd numeric(12, 6),

    created_at timestamptz not null default now()
);

comment on table public.savings_receipts is
    'Per-request savings evidence. `arm` carries the withheld-control label, so a '
    'saving is a measured difference between arms rather than a counterfactual '
    'estimate. Token classes are split because cache reads and writes bill at '
    'different rates from uncached input.';

-- The three questions a report asks: by customer over time, by project, and
-- by arm. Without the last one, every arm comparison is a full scan.
create index if not exists idx_savings_receipts_user_created
    on public.savings_receipts(user_id, created_at);
create index if not exists idx_savings_receipts_project
    on public.savings_receipts(user_id, project_hash);
create index if not exists idx_savings_receipts_arm
    on public.savings_receipts(user_id, arm, created_at);

-- RLS, following the pattern the telemetry and usage tables already use: a
-- customer reads only their own rows, and nothing is readable anonymously.
alter table public.savings_receipts enable row level security;

create policy "savings_receipts_select_own"
    on public.savings_receipts
    for select
    to authenticated
    using (auth.uid() = user_id);

create policy "savings_receipts_insert_own"
    on public.savings_receipts
    for insert
    to authenticated
    with check (auth.uid() = user_id);

-- Deliberately no update or delete policy. A receipt is evidence; if it can be
-- edited after the fact it is not evidence any more. A correction is a new row.
