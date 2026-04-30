# AI Coding Session – Production ML Serving (Deep Dive on Design Tradeoffs)

## Session Log (Condensed but High-Signal)

User:
The current implementation is growing in a single file. This won’t scale. Why is this a problem and how should we modularize?

---

AI:
A single-file design leads to:

* tight coupling between API, business logic, and infra concerns
* poor testability
* harder parallel development

Suggested modular structure:

* api/ → request handling
* services/ → inference logic
* infra/ → cache, DB, rate limiting
* workers/ → batching logic
* core/ → config, shared utilities

---

User:
Be more specific. What logic goes where and why?

---

AI:

* api/routes.py → FastAPI endpoints (pure I/O layer)
* services/inference.py → model execution (stateless, testable)
* workers/batching.py → async queue + batching window logic
* infra/cache.py → caching abstraction (swap in Redis later)
* infra/rate_limiter.py → token bucket logic
* infra/db.py → usage + quota persistence
* core/config.py → environment + settings

This separation ensures:

* clear ownership boundaries
* infra can evolve independently (e.g., Redis swap)
* easier unit testing per layer

---

User:
Why do we even need a database here? Can’t we stay in-memory?

---

AI:
In-memory works only for a single instance. A DB is needed for:

1. Multi-instance consistency (Kubernetes scaling)
2. Persistent quota tracking (monthly limits)
3. Auditability (usage logs for billing/compliance)
4. Fault tolerance (no data loss on restart)

---

User:
What would you store in DB vs cache?

---

AI:
DB (persistent, source of truth):

* user quotas (monthly usage)
* API keys / identity
* audit logs (optional)

Cache (ephemeral, performance layer):

* recent inference results
* rate limit counters (short-lived)
* hot feature transformations

---

User:
What DB would you choose and why?

---

AI:

* PostgreSQL → strong consistency, good for quotas + transactions
* Redis → fast access for rate limiting + caching

Typical pattern:
Postgres (truth) + Redis (performance layer)

---

User:
Explain caching strategy in detail. Don’t stay high level.

---

AI:

### Cache Key

* hash(input features)
* include model version to avoid stale predictions

### TTL Strategy

* short TTL (e.g., 5–15 min) for dynamic data
* longer TTL for static/tabular inference

### Invalidation

* on model update → version bump (preferred over delete)
* LRU eviction for memory control

### Risks

* cache stampede → mitigate with request coalescing
* stale data → TTL tuning

---

User:
What is cache stampede and how would you fix it here?

---

AI:
Cache stampede occurs when many requests hit a missing key simultaneously.

Mitigations:

* request coalescing (only one request computes result)
* async locking per key
* stale-while-revalidate strategy

---

User:
You mentioned TTL—how do you actually choose it?

---

AI:
Depends on:

* data volatility (how often input distribution changes)
* model update frequency
* latency vs freshness tradeoff

Example:

* real-time pricing → TTL 1–2 min
* static tabular inference → TTL 10–30 min

---

User:
Now talk about rate limiting vs quotas. People confuse these.

---

AI:
Rate limiting:

* short-term control (requests/sec or per minute)
* protects system from bursts

Quotas:

* long-term control (daily/monthly usage)
* enforces business constraints

Both are required:
rate limiting → system stability
quota → cost + fairness

---

User:
Where should rate limiting live in architecture?

---

AI:

* ideally at API gateway (NGINX / Envoy) for coarse filtering
* application-level (Redis-backed) for per-user logic

Hybrid approach is best.

---

User:
Let’s talk batching. What are the tradeoffs?

---

AI:
Pros:

* higher throughput (vectorized model execution)
* better hardware utilization

Cons:

* increased latency (waiting window)
* fairness issues under uneven load

---

User:
How do you balance latency vs throughput?

---

AI:

* dynamic batching:

  * max batch size (e.g., 32)
  * max wait time (e.g., 50ms)
* adaptive tuning based on traffic

---

User:
What happens under very low traffic?

---

AI:

* batches never fill → latency penalty
  Fix:
* enforce max wait timeout → process partial batches

---

User:
And under very high traffic?

---

AI:

* queue buildup → increased latency
  Fix:
* backpressure (reject or shed load)
* autoscaling workers
* priority queues if needed

---

User:
How would this design change for large tabular foundation models?

---

AI:

* move to GPU-backed batch inference
* columnar memory formats (optimize feature access)
* async data ingestion (large tables via object storage)
* streaming inference for large datasets
* tighter memory management (avoid duplication)

---

User:
Final: what are the biggest production risks in this system?

---

AI:

1. Queue overload → cascading latency
2. Cache miss storms → compute spikes
3. Inconsistent rate limiting in distributed setup
4. Model latency variance (tail latency issues)
5. Lack of observability → hard to debug

---

## Key Takeaways

* Broke down monolithic design into modular, production-ready architecture
* Distinguished clearly between DB vs cache responsibilities
* Designed robust caching strategy with TTL and stampede protection
* Analyzed batching tradeoffs under different traffic patterns
* Considered distributed scaling, fairness, and failure modes

---

## Reflection

This session shows how I use AI not just to generate code, but to stress-test system design decisions—especially around scalability, state management, and real-world failure scenarios.
