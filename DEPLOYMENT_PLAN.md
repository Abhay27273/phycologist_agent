# Deployment Plan — Azure vs AWS

## 1. What we're actually deploying

Psych-platform-core is not a stateless CRUD API — three characteristics shape every decision below:

1. **In-process ML models.** `BAAI/bge-base-en-v1.5` (embeddings) and `cross-encoder/ms-marco-MiniLM-L-6-v2` (reranker) load into each worker process's memory (~1-1.5GB RSS observed locally, ~10-20s cold-load time). This means: size containers generously (2 vCPU / 4GB RAM minimum), avoid scale-to-zero on the main API service, and run **one worker per container** — scale via replica count, not `--workers N`, so memory usage per instance stays predictable. (The app is already horizontally-scaling-ready: Postgres-backed LangGraph checkpointer, Qdrant server mode, Redis-backed rate limiting and RAG cache were all built in Phase 1 specifically so N replicas can share state.)
2. **WebSocket support required.** The `/api/v1/ws/chat/{session_id}` endpoint needs a load balancer that supports long-lived WebSocket upgrades. This rules out AWS API Gateway (REST/HTTP APIs) and AWS App Runner for the primary entry point, and rules out any Azure tier that doesn't have WebSockets explicitly enabled.
3. **External LLM dependency.** Groq (primary) and Gemini (fallback) are both outside your infrastructure. Latency benchmarking (`tests/test_latency.py`) confirmed that bursty traffic can exceed Groq's free-tier per-minute quota and trigger SDK retry backoff (+4-5s tail latency) — this is a production risk independent of which cloud you pick. Budget for a paid Groq tier before launch, and the existing hard-coded CrisisNode template (no LLM call) already protects the one path where latency is life-safety-critical.

## 2. Decision framework

| Factor | Azure | AWS |
|---|---|---|
| Container platform w/ WebSocket + easy scale-to-N | **Azure Container Apps** (KEDA-based, simplest) | ECS Fargate + ALB (more setup, more control) |
| Managed Postgres | Azure Database for PostgreSQL – Flexible Server | RDS for PostgreSQL / Aurora PostgreSQL |
| Managed Redis | Azure Cache for Redis | ElastiCache for Redis |
| Vector DB | Self-host Qdrant (Container Apps/AKS) or swap to **Pinecone** (already supported via `VECTOR_DB_BACKEND=pinecone`, zero ops either cloud) | Same choice, same recommendation |
| Secrets | Azure Key Vault | AWS Secrets Manager |
| If team already has cloud experience | Pick that one — the architecture maps cleanly to both | Pick that one |

**Recommendation if you have no existing preference: Azure Container Apps.** It's the least operational overhead for this specific workload — native WebSocket support, scale-to-N (not to zero, per point 1 above) out of the box, and a single `az containerapp` deployment replaces what would otherwise be an ECS cluster + ALB + target group + security groups on AWS. The AWS blueprint below is equally valid if your team already runs AWS.

**Recommendation for the vector DB regardless of cloud: switch `VECTOR_DB_BACKEND=pinecone` for production.** Self-hosting Qdrant means one more stateful container to patch, back up, and monitor. Pinecone is already a first-class backend in `rag_service.py` and `scripts/ingest.py` — changing one env var removes an entire ops surface. Keep Qdrant for local dev only (already the default).

---

## 3. Azure blueprint

```
                          ┌─────────────────────────┐
  Users ── HTTPS/WSS ──▶  │  Azure Container Apps    │──▶ Azure Database for
                          │  (psych-api, 2-10        │    PostgreSQL Flexible
                          │   replicas, WS enabled)  │    Server
                          └───────────┬─────────────┘
                                      │
                     ┌────────────────┼────────────────┐
                     ▼                ▼                ▼
            Azure Cache for      Pinecone (or       Key Vault
              Redis              self-hosted            (secrets)
           (rate limit +          Qdrant on
            RAG cache)          Container Apps)
```

### Steps

1. **Container Registry** — `az acr create` (Basic tier is enough at this scale). Push the image built from the repo's `Dockerfile`:
   ```
   az acr build --registry <acr-name> --image psych-platform-core:latest .
   ```
2. **Managed Postgres** — `az postgres flexible-server create` (Burstable B1ms to start, ~$15/mo). Note the connection string uses `postgresql+asyncpg://` for `DATABASE_URL` — the app's `init_graph()` already branches on this scheme to use `AsyncPostgresSaver`.
3. **Redis** — `az redis create` (Basic C0, ~$16/mo) or Azure Cache for Redis Enterprise if you need higher throughput later. Set `REDIS_URL=rediss://<host>:6380,password=<key>,ssl=True` (Azure Redis requires TLS on 6380).
4. **Vector DB** — either:
   - Pinecone: set `VECTOR_DB_BACKEND=pinecone`, `PINECONE_API_KEY`, `PINECONE_INDEX_NAME` (no Azure resource needed), **or**
   - Self-hosted Qdrant: a second Container Apps app (1 replica, no autoscale, persistent volume via Azure Files) running `qdrant/qdrant:1.18.2`.
5. **Secrets** — `az keyvault create`, store `GOOGLE_API_KEY`, `GROQ_API_KEY`, `JWT_SECRET_KEY`, `DATABASE_URL`, `PINECONE_API_KEY`. Container Apps reads Key Vault secrets natively via managed identity — no secrets in the container spec itself.
6. **Deploy the app**:
   ```
   az containerapp create \
     --name psych-api --resource-group <rg> \
     --image <acr-name>.azurecr.io/psych-platform-core:latest \
     --target-port 8000 --ingress external \
     --min-replicas 2 --max-replicas 10 \
     --cpu 2.0 --memory 4.0Gi \
     --secrets google-api-key=keyvaultref:... groq-api-key=keyvaultref:... \
     --env-vars DATABASE_URL=secretref:db-url REDIS_URL=secretref:redis-url ...
   ```
   `--min-replicas 2` avoids scale-to-zero (model cold-load tax) and gives you one spare replica for rolling deploys/failover. Container Apps' built-in ingress supports WebSockets by default — no extra config needed for `/ws/chat`.
7. **Custom domain + TLS** — Container Apps issues a free managed certificate for custom domains automatically.
8. **CI/CD** — GitHub Actions: `az acr build` on push to `main`, then `az containerapp update --image ...`. ~15 lines of YAML; happy to write the actual workflow file if you want it.
9. **Observability** — enable Application Insights (`az monitor app-insights component create`) and wire `structlog` (Phase 4 item already scoped in `IMPLEMENTATION_PLAN.md`) to it for request tracing across the sentiment → RAG → generation pipeline.

### Estimated monthly cost (small production load, ~50-200 daily active users)
| Resource | Tier | ~Cost/mo |
|---|---|---|
| Container Apps (2 replicas × 2vCPU/4GB, ~50% duty) | Consumption plan | $70-120 |
| PostgreSQL Flexible Server | Burstable B1ms | $15 |
| Azure Cache for Redis | Basic C0 | $16 |
| Pinecone | Starter (free) or Standard | $0-70 |
| Key Vault, ACR, bandwidth | — | ~$10 |
| **Total** | | **~$110-230/mo** |

---

## 4. AWS blueprint

```
                          ┌──────────────────────────┐
  Users ── HTTPS/WSS ──▶  │  Application Load         │
                          │  Balancer (WS-enabled)    │
                          └───────────┬───────────────┘
                                      ▼
                          ┌──────────────────────────┐
                          │  ECS Fargate service      │──▶ RDS for PostgreSQL
                          │  (psych-api, 2-10 tasks)  │    (or Aurora)
                          └───────────┬───────────────┘
                     ┌────────────────┼────────────────┐
                     ▼                ▼                ▼
              ElastiCache          Pinecone (or      Secrets
               for Redis          self-hosted          Manager
                                  Qdrant on ECS)
```

**Why ALB + ECS Fargate, not App Runner or API Gateway:** App Runner's HTTP-only ingress doesn't support WebSocket upgrade. API Gateway WebSocket APIs use a fundamentally different connection model (Lambda-backed, `$connect`/`$disconnect`/`$default` routes) that doesn't map onto a persistent FastAPI WebSocket handler without a rewrite. ALB passes WebSocket upgrades straight through to Fargate tasks — no rewrite needed.

### Steps

1. **ECR** — `aws ecr create-repository`, push the built image (same `Dockerfile`).
2. **RDS PostgreSQL** — `db.t4g.micro` to start (~$13/mo), or Aurora Serverless v2 if traffic is spiky (scales down between bursts, still Postgres-wire-compatible with the existing `AsyncPostgresSaver`).
3. **ElastiCache for Redis** — `cache.t4g.micro` (~$12/mo). Set `REDIS_URL=redis://<primary-endpoint>:6379/0` (enable in-transit encryption for `rediss://` if handling PHI-adjacent data — recommended given this is a mental-health app).
4. **Vector DB** — Pinecone (recommended, zero AWS resources) or self-hosted Qdrant as a second Fargate service with an EFS-backed volume for `/qdrant/storage`.
5. **Secrets Manager** — store the same secret set as the Azure plan; reference via `secrets` block in the ECS task definition (injected as env vars at container start, never baked into the image).
6. **Task definition** — 2 vCPU / 4GB per task (matches the model-memory sizing note in §1), health check hitting `/health` on port 8000.
7. **ECS Service + ALB**:
   - Target group with `Stickiness` **not required** (Postgres checkpointer + Redis cache make the app stateless across replicas — any replica can serve any session).
   - ALB idle timeout: bump from the 60s default to **300s** — the default will silently drop WebSocket connections that sit idle between voice-agent turns.
   - Enable `Deregistration delay` tuning (~30s) so in-flight WebSocket connections aren't cut mid-turn during deploys.
8. **Auto Scaling** — target-tracking on CPU (scale out above ~60%) with `min=2, max=10` tasks — same rationale as Azure's `min-replicas 2` (avoid cold-start on the ML models).
9. **CI/CD** — GitHub Actions → build/push to ECR → `aws ecs update-service --force-new-deployment`. CodeDeploy blue/green if you want zero-downtime cutover with automatic rollback on health-check failure.
10. **Observability** — CloudWatch Container Insights for the ECS service; ship `structlog` JSON logs to CloudWatch Logs; consider X-Ray for tracing the sentiment→RAG→generation call chain.

### Estimated monthly cost (same load assumption)
| Resource | Tier | ~Cost/mo |
|---|---|---|
| ECS Fargate (2 tasks × 2vCPU/4GB, ~50% duty) | Fargate | $90-140 |
| ALB | — | $20 |
| RDS PostgreSQL | db.t4g.micro | $13 |
| ElastiCache Redis | cache.t4g.micro | $12 |
| Pinecone | Starter (free) or Standard | $0-70 |
| Secrets Manager, ECR, bandwidth | — | ~$10 |
| **Total** | | **~$145-265/mo** |

---

## 5. Cross-cutting requirements (apply to either cloud)

### Must fix before any production deploy
- **Rotate the leaked Postgres password.** `docker-compose.yml` had `Apple55@@7727` committed in plaintext (fixed in this change — now reads from `.env` via `${POSTGRES_PASSWORD:?...}`). If that password was ever used for a real database, **rotate it now** — it's in git history regardless of the current file content.
- **Set a real `JWT_SECRET_KEY`.** The default (`change-me-in-production`) must never reach a prod environment — inject via Key Vault / Secrets Manager, not `.env`.
- **Switch `DATABASE_URL` to Postgres.** SQLite's `AsyncSqliteSaver` is explicitly single-process (per `workflow.py` comments) — it will silently corrupt or lock up under >1 replica.

### Within the first month
- **Pin a paid Groq tier** (or add a second provider key + round-robin) — the latency benchmarks in `tests/test_latency.py` reproducibly show ~4-5s tail latency once the free-tier per-minute quota is exceeded. At >1 concurrent user this will happen regularly.
- **CORS_ORIGINS** — set to your actual frontend domain(s); the current default (`localhost:3000`) must not ship to prod.
- **Backups** — enable automated Postgres backups (both clouds default to 7-day retention; bump to 30 for a mental-health app given data sensitivity).
- **TLS everywhere** — both blueprints terminate TLS at the load balancer/ingress; confirm `CORS_ORIGINS` and any frontend fetch/WebSocket URLs use `https://`/`wss://`, never plain `http://`/`ws://`, outside local dev.

### Data sensitivity note
This app handles mental-health conversations (mood, risk scores, crisis flags). Before launch, confirm: encryption at rest for Postgres (both clouds enable by default) and Redis (enable explicitly), a data-retention policy for `chat_messages`, and whether your jurisdiction's regulations (HIPAA, GDPR, etc.) require a signed BAA/DPA with the cloud provider and with Groq/Google for the LLM calls themselves — this is a legal/compliance decision, not an infrastructure one, and it needs an answer before real user data flows through Groq or Gemini.

## 6. Migration checklist (in order)

- [ ] Rotate any real secret matching the committed Postgres password
- [ ] Provision managed Postgres, run `alembic upgrade head`
- [ ] Provision managed Redis, set `REDIS_URL`
- [ ] Decide Qdrant-self-hosted vs Pinecone; run `scripts/ingest.py` (+ `ingest_datasets.py`) against the target vector DB
- [ ] Store all secrets in Key Vault / Secrets Manager; set `JWT_SECRET_KEY` to a real random value
- [ ] Build + push the image from the fixed `Dockerfile`
- [ ] Deploy with `min-replicas ≥ 2`, WebSocket ingress confirmed working end-to-end (test `/ws/chat` through the real load balancer, not just locally)
- [ ] Point `CORS_ORIGINS` at the real frontend domain
- [ ] Set up CI/CD (build → push → deploy on merge to `main`)
- [ ] Enable structured logging + tracing (Application Insights / CloudWatch)
- [ ] Load-test with `tests/test_latency.py` against the deployed URL (`TEST_BASE_URL=https://your-domain`) before opening to real users
- [ ] Confirm backup + retention policy meets your data-sensitivity requirements

---

## 7. AWS free-tier / minimal-cost deployment (single EC2 instance)

This is a different deployment shape from §4 — not "the ECS blueprint, cheaper," but a genuinely minimal single-VM setup for a demo, pilot, or portfolio deployment. Be upfront with yourself about what it's for: **no redundancy, no autoscaling, one instance is a single point of failure.** Given this app handles mental-health conversations, don't point real users at this without accepting that risk — it's for validating the product, not serving them long-term.

### The one honest caveat: RAM

AWS's actual free-tier compute is a **t2.micro or t3.micro — 1 vCPU, 1GB RAM** (750 hrs/month, 12 months from account creation). This app's in-process ML models (BGE embeddings + cross-encoder reranker, both PyTorch-backed) alone consume ~800MB-1.2GB RSS once loaded (§1). That leaves very little headroom for the OS, Python runtime, FastAPI, and SQLite on a 1GB box — it can work, but it's tight enough to risk an OOM kill under any concurrent load, and possibly even at startup.

Two honest options, pick one:
- **True $0**: t2.micro/t3.micro + a swap file (see step 2 below) as an OOM safety net. Converts crashes into slow-but-alive degraded performance. Fine for a solo demo with no concurrent users.
- **~$15/mo, much safer**: step up to **t3.small** (2 vCPU, 2GB RAM) — not free-tier eligible, but cheap enough that "minimal-cost" is still the right description, and it removes the OOM risk entirely.

### Architecture — deliberately simplified, not a smaller version of §4

Skip Postgres, Redis, and Docker entirely. A single instance doesn't need any of the multi-replica plumbing built in Phase 1 — that infrastructure exists to let N replicas share state, and here N=1:

| Phase 1 component | Single-VM equivalent |
|---|---|
| Postgres (`AsyncPostgresSaver`) | **SQLite** (`DATABASE_URL=sqlite+aiosqlite:///./psych_db.sqlite`) — this is exactly how the app already runs in local dev |
| Redis (rate limit + RAG cache) | `REDIS_URL=memory://` — in-process, already the documented no-Redis fallback |
| Qdrant server mode | `QDRANT_MODE=local` (file-based) — or Pinecone Starter (free tier) if you'd rather not manage a local index file |
| Docker containers | Run the app directly via a Python venv + systemd — Docker/containerd's own RAM overhead matters when the whole box has 1GB |
| ALB (WebSocket-capable ingress) | **nginx**, reverse-proxying to `127.0.0.1:8000` with `Upgrade`/`Connection: upgrade` headers passed through — nginx handles WebSocket proxying natively |
| ACM certificate | **Let's Encrypt via certbot** — free, no load balancer needed |

### Steps

1. **Launch the instance** — Ubuntu 24.04 LTS AMI, t3.micro (or t3.small), free-tier-eligible EBS (30GB gp3). Security group: allow 22 (SSH, restrict to your IP), 80, 443.
2. **Add a 2GB swap file** (skip if you went with t3.small):
   ```bash
   sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
   sudo mkswap /swapfile && sudo swapon /swapfile
   echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
   ```
3. **Install Python 3.11 + deps**:
   ```bash
   sudo apt update && sudo apt install -y python3.11 python3.11-venv nginx certbot python3-certbot-nginx
   git clone <your-repo> && cd psych-platform-core
   python3.11 -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   ```
4. **Configure `.env`** — SQLite + in-process fallbacks, no managed services:
   ```
   DATABASE_URL=sqlite+aiosqlite:///./psych_db.sqlite
   REDIS_URL=memory://
   VECTOR_DB_BACKEND=qdrant
   QDRANT_MODE=local
   QDRANT_PATH=./qdrant_data
   JWT_SECRET_KEY=<real random value>
   GROQ_API_KEY=<your key>
   GOOGLE_API_KEY=<your key>
   CORS_ORIGINS=https://your-domain.example
   ```
5. **Run migrations + ingest**: `alembic upgrade head`, `python scripts/ingest.py`.
6. **systemd unit** (`/etc/systemd/system/psych-api.service`) so it survives reboots and auto-restarts on crash — important given the OOM risk on t3.micro:
   ```ini
   [Unit]
   After=network.target

   [Service]
   WorkingDirectory=/home/ubuntu/psych-platform-core
   ExecStart=/home/ubuntu/psych-platform-core/venv/bin/python start_server.py
   Restart=always
   RestartSec=5
   Environment=PORT=8000

   [Install]
   WantedBy=multi-user.target
   ```
   `sudo systemctl enable --now psych-api`
7. **nginx reverse proxy** (`/etc/nginx/sites-available/psych-api`) — the `Upgrade`/`Connection` headers are what make `/api/v1/ws/chat` work through nginx:
   ```nginx
   server {
       listen 80;
       server_name your-domain.example;
       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
           proxy_set_header Host $host;
           proxy_read_timeout 300s;   # keep long-lived WS connections alive
       }
   }
   ```
8. **Free TLS**: `sudo certbot --nginx -d your-domain.example` — handles cert issuance and nginx config automatically, auto-renews via a systemd timer certbot installs.
9. **Domain**: a real domain is the one item that isn't free (~$10-15/yr). For literal $0, use a free dynamic-DNS subdomain (e.g. DuckDNS) pointed at the instance's **Elastic IP** (free while attached to a running instance — AWS only bills unattached Elastic IPs).

### Cost: $0/month for 12 months (new AWS account), then ~$7.50/mo (t3.micro on-demand) after the free-tier window closes

No RDS, no ElastiCache, no ALB, no Fargate — all genuinely $0 line items. The only recurring cost is the domain if you choose a real one over a free DuckDNS-style subdomain.

### When to graduate off this

The moment you have real concurrent users (not just yourself testing), move to §4's ECS Fargate blueprint — this single-VM shape has no redundancy (one instance restart mid-conversation drops every in-flight WebSocket connection) and the 1-2GB RAM ceiling will not absorb concurrent RAG + reranking calls gracefully. Treat this section as "how to get a working public URL for $0 to validate the product," not as a production posture.

---

## 8. Azure free-tier / minimal-cost deployment

Checked against current Microsoft Learn documentation (numbers below are verified, not estimated) before writing this — Azure has three candidate free options for this app, and they are **not equally viable**.

### Option A — Azure App Service Free (F1): ruled out, not just constrained

F1 looks tempting (permanently free, not a 12-month window) but has a **hard blocker**: the Free and Shared App Service tiers force **32-bit application architecture** (confirmed via Azure's own subscription-limits docs). Modern PyTorch — a dependency of `sentence-transformers`, which this app needs for the BGE embeddings and cross-encoder reranker — doesn't ship 32-bit wheels. This isn't "it'll be slow," it's "`pip install -r requirements.txt` or the runtime import will very likely fail outright." Beyond that, even if it did run: 60 CPU-minutes/day (app gets HTTP 403'd until midnight UTC reset once exceeded), 165MB/day bandwidth, and only 5 WebSocket connections per instance. **Skip F1 entirely for this app.**

### Option B — Azure Container Apps: real free grant, but only with scale-to-zero

Verified free grant, forever (not time-limited like the 12-month VM/Postgres offers): **180,000 vCPU-seconds + 360,000 GiB-seconds + 2,000,000 HTTP requests, per subscription, per month.** Idle replicas (min-replicas kept >0 but not actively serving traffic) do bill at a reduced idle rate rather than the full active rate — but for a replica sized at the 2vCPU/4GB this app needs, running 24/7 will still very likely exceed the free grant (180k vCPU-seconds ≈ 50 vCPU-hours; a month has ~730 hours) and produce a modest bill, not a $0 one.

To actually stay inside the free grant, you'd set `min-replicas=0` (true scale-to-zero) — which reintroduces exactly the problem §1 of this doc told you to avoid: every request after an idle period pays the full ~15-30s ML-model cold-load tax before your first response. Workable if you're comfortable being the only one testing it and tolerating that wait after any gap in usage; not something to demo live to someone else.

### Option C — Azure VM free tier: the real equivalent of §7's AWS plan

This is the one that actually parallels the AWS single-VM approach, and it has a small, confirmed edge over AWS: Azure's free-tier VM options are **B1S, B2pts_v2 (ARM), or B2ats_v2 (AMD) — 750 combined hours/month for 12 months**, and the B2 variants give you **2 vCPUs at the same ~1GB RAM** as AWS's t2/t3.micro (vs AWS's 1 vCPU at 1GB). Same RAM tightness caveat as §7 applies — still worth the swap-file safety net — but the extra vCPU genuinely helps with the CPU-bound RAG reranking step.

**Everything else is identical to §7** — same architecture table (SQLite instead of Postgres, `REDIS_URL=memory://` instead of Redis, `QDRANT_MODE=local` instead of server mode, nginx + certbot instead of a load balancer + ACM), same systemd unit, same nginx WebSocket-upgrade config, same "no redundancy, graduate off this once you have real concurrent users" caveat. Only the provisioning step differs:

1. Create a VM: `az vm create --name psych-api --image Ubuntu2404 --size Standard_B2pts_v2 --admin-username azureuser --generate-ssh-keys` (use `Standard_B1s` or `Standard_B2ats_v2` if ARM isn't convenient for your tooling).
2. Open an NSG rule for 80/443 (and 22 for your own IP): `az vm open-port --port 80 --name psych-api` (repeat for 443).
3. SSH in and follow §7 steps 2-8 verbatim (swap file → Python/nginx/certbot install → `.env` config → systemd unit → nginx WebSocket proxy config → `certbot --nginx`).
4. For step 9's domain: Azure DNS zones aren't free (~$0.50/mo) — same recommendation as AWS, use a free DuckDNS-style subdomain pointed at the VM's public IP for literal $0, or accept the small DNS zone cost for a real domain.

### Cost: $0/month for 12 months (new Azure subscription), then pay-as-you-go B-series pricing after (comparable to AWS's post-free-tier t3.micro cost)

### Bottom line

If someone asks "can we host this free on Azure," the accurate answer is: **not on App Service (architecturally can't run it), maybe on Container Apps (only with a cold-start tax you'll feel on every idle-then-resume request), yes on a free-tier VM (same shape and same honest caveats as the AWS plan in §7, with a minor CPU edge).** Use Option C.

### Field update — brand-new subscriptions may not actually get B-series capacity

In practice, deploying this on a fresh Azure Free Trial account hit a real, reproducible blocker: `az vm create` failed with `SkuNotAvailable` for `Standard_B1s`, `Standard_B2s`, and `Standard_B2ats_v2` across **every region tried** (10+, spanning US, Europe, and Asia), despite `az vm list-usage` confirming a healthy quota (`Standard BS Family vCPUs: Limit 4`, not 0). Quota and eligibility are different things — this is consistent with Microsoft applying an anti-abuse restriction on cheap burstable SKUs specifically for very new subscriptions (B-series is the most commonly abused free-tier resource), separate from the quota shown in the portal/CLI.

Two unrelated issues compounded this and are worth calling out for anyone retracing these steps:
1. **Resource providers** (`Microsoft.Compute`, `Microsoft.Network`, `Microsoft.Storage`) may show `NotRegistered` on a subscription that hasn't created a resource via the Portal UI yet — every VM create fails with a misleading preflight error until you run `az provider register --namespace <ns>` for each and wait for `Registered`. This is a one-time, permanent fix at the subscription level.
2. **Cloud Shell without a mounted storage account is fully ephemeral** — shell variables and the SSH keypair silently vanish on any reconnect/idle-timeout, which produces confusing `argument --resource-group/-g: expected one argument` errors that look like a scripting bug but are actually session loss. Mount a storage account (fractions of a cent/month) when setting up Cloud Shell for any multi-step deployment session.

**Practical fallback**: if B-series is unavailable everywhere, use a general-purpose size like `Standard_D2s_v3` instead. It isn't free-tier, but it isn't subject to the same new-account throttle either, and at ~$0.10/hr the $200/30-day signup credit absorbs it many times over — effectively free for the duration of initial setup and testing. Revisit true $0 B-series pricing later once the account has aged past whatever window triggers this restriction (undocumented by Microsoft; anecdotally hours to a few days).
