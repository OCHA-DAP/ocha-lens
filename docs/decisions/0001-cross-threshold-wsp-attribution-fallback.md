---
status: "accepted"
date: 2026-08-31
decision-makers: Tristan Downing
consulted: Zack Arno (PR #50 review), Claude (pairing session)
informed: ocha-lens contributors, ds-storms-pipeline consumers
---

# Attribute orphan WSP bands via lower-threshold containment, only when unambiguous

## Context and Problem Statement

`match_wsp_to_tracks` attributes NHC wind-speed-probability polygons to storms
by (1) track-line intersection, then (2) containment inside an already-matched
polygon at the same wind threshold. Both passes can fail together: a weakening
storm's forecast track line departs its trailing higher-threshold band, and if
*no* polygon at that threshold matched, there is no same-threshold container
either — the entire threshold comes back `atcf_id=NULL`. Downstream consumers
(the exposure tables, and the alerts pipeline that filters by `atcf_id`)
silently drop that exposure: with Dolly/AL042026 at the 2026-08-28 06:00
issuance, every 50 kt polygon over the Leeward Islands went unattributed and
~679k `pop_exposed` vanished from the alert products.

Physically a storm's higher-threshold probability area nests inside its own
lower-threshold area, so the matched lower-kt footprint is a natural
attribution donor. But at matching time the part's owner is exactly what is
unknown: an orphan band of storm B sitting inside storm A's footprint would be
*misattributed* to A — and unlike an unmatched part, a misattribution is
invisible downstream and compounds (the wrongly-owned band becomes a pass-2
container for later inner lobes). How much recall do we buy, at what
misattribution risk?

## Considered Options

1. No fallback (status quo): orphan thresholds stay `NULL`; exposure silently
   drops.
2. Cross-threshold containment, smallest container wins: maximal recall, but
   a part inside two storms' footprints is confidently assigned to one of
   them by a geometric tiebreak with no physical basis.
3. Cross-threshold containment, **only when every containing lower-kt donor
   names the same storm**; otherwise leave unmatched.

## Decision Outcome

Chosen option 3. The single-storm nesting rationale only identifies an owner
when it is the *only* candidate; when footprints of two storms both contain
the part, the honest answer is "unknown", and `NULL` is the visible,
recoverable form of unknown (option 2's wrong `atcf_id` is neither). This
resolves the observed failure class — real orphan bands overwhelmingly occur
inside exactly one storm's footprint (storms close enough to overlap
footprints have track lines close enough for pass 1) — while keeping the
matcher's errors conservative in the same direction as before.

### Consequences

- Good: the Dolly-class silent exposure drop is fixed; misattribution risk is
  not traded in for it.
- Accepted: a genuinely ambiguous orphan (two overlapping storms, both
  plausible owners) stays `NULL` and its exposure is still dropped — rare,
  and detectable by monitoring `nhc_wsp_polygon_matched` for NULL-`atcf_id`
  rows carrying nonzero downstream exposure.
- The ascending-`wind_threshold_kt` processing order is now load-bearing
  (lower-kt donors must be matched first), so the sort is applied
  unconditionally, with or without the optional `percentage` column.
