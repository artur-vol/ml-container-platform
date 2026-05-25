# ml-container-platform

[![CI](https://github.com/artur-vol/ml-container-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/artur-vol/ml-container-platform/actions/workflows/ci.yml)

Containerised ML platform demonstrating the full inference lifecycle:
**train → register → deploy → monitor → A/B canary**.

Model: MobileNetV2 fine-tuned on CIFAR-10. Infrastructure: Docker, Helm, k3s (homelab) and GKE Autopilot (cloud), same chart for both.

```
MacBook (dev)
    │  docker buildx / helm lint
    ▼
GitHub Actions ──► ghcr.io (images)
    │                   │
    ▼                   ▼
k3s homelab        GKE Autopilot
(4 replicas)       (1 replica, Tailscale proxy → homelab MLflow)
    │
Prometheus + Grafana + Alertmanager
```

---

## Prerequisites

| Tool | Version |
|------|---------|
| Docker (buildx) | 24+ |
| Helm | 3.14+ |
| kubectl | 1.29+ |
| Python | 3.11+ |
| Terraform | 1.7+ (GKE only) |

---

## 1. Local development (MacBook)

```bash
# Train a quick model and start MLflow
./ops.sh train

# Promote latest run to Production
./ops.sh promote

# Build serve image and start on :8000
./ops.sh build
./ops.sh run

# Smoke-test
./ops.sh smoke-test
```

Cross-arch build for homelab/GKE (linux/amd64):

```bash
docker buildx build --platform linux/amd64 \
  -f docker/serve.Dockerfile -t ghcr.io/artur-vol/ml-container-platform-serve:latest .
```

---

## 2. k3s homelab

**Prerequisites:** k3s running, `KUBECONFIG=~/k3s.yaml`, MLflow running on host at port 5000.

```bash
# Secrets
kubectl create namespace ml-platform

kubectl create secret docker-registry ghcr-secret \
  --docker-server=ghcr.io \
  --docker-username=artur-vol \
  --docker-password=<GITHUB_PAT> \
  --namespace ml-platform

# Deploy
helm upgrade --install ml-platform helm/ml-platform \
  -f helm/ml-platform/values.yaml \
  -f helm/ml-platform/values-homelab.yaml \
  --namespace ml-platform --create-namespace --wait

# Verify
curl http://192.168.0.100:30800/health
```

**Canary deployment (80/20 split):**

```bash
helm upgrade --install ml-platform-canary helm/ml-platform \
  -f helm/ml-platform/values.yaml \
  -f helm/ml-platform/values-homelab.yaml \
  --set track=canary \
  --set replicaCount=1 \
  --set sharedService.enabled=false \
  --set image.tag=<CANARY_TAG> \
  --namespace ml-platform

# Validate and promote (or rollback)
PROMETHEUS_URL=http://<PROMETHEUS_IP>:9090 bash scripts/canary_promote.sh
```

**Monitoring:**

```bash
helm dependency build helm/monitoring
helm upgrade --install monitoring helm/monitoring \
  -f helm/monitoring/values.yaml \
  --namespace monitoring --create-namespace
# Grafana: http://192.168.0.100:30300  admin/admin
```

---

## 3. GKE Autopilot

**Prerequisites:** `gcloud` authenticated, Terraform installed.

```bash
# 1. Create cluster
cd terraform/gke
terraform init
terraform apply -var="project_id=<GCP_PROJECT_ID>"
gcloud container clusters get-credentials ml-platform-thesis \
  --region europe-west1 --project <GCP_PROJECT_ID>

# 2. Namespace + secrets
kubectl create namespace ml-platform

kubectl create secret docker-registry ghcr-secret \
  --docker-server=ghcr.io \
  --docker-username=artur-vol \
  --docker-password=<GITHUB_PAT> \
  --namespace ml-platform

# Ephemeral Tailscale key from tailscale.com/admin/settings/keys
kubectl create secret generic tailscale-auth \
  --from-literal=authkey=<TS_AUTHKEY> \
  --namespace ml-platform

# 3. Tailscale proxy (bridges GKE → homelab MLflow via Tailscale VPN)
kubectl apply -f k8s/tailscale-proxy.yaml

# 4. Ingress controller
helm upgrade --install ingress-nginx ingress-nginx \
  --repo https://kubernetes.github.io/ingress-nginx \
  --namespace ingress-nginx --create-namespace --wait

EXTERNAL_IP=$(kubectl get svc ingress-nginx-controller \
  -n ingress-nginx -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

# 5. Update ingress host
sed -i "s/EXTERNAL_IP/$EXTERNAL_IP/" helm/ml-platform/values-gke.yaml

# 6. Deploy
helm upgrade --install ml-platform helm/ml-platform \
  -f helm/ml-platform/values.yaml \
  -f helm/ml-platform/values-gke.yaml \
  --namespace ml-platform --wait

# 7. Verify
curl http://ml-platform.$EXTERNAL_IP.nip.io/health

# 8. Cleanup (stops billing)
helm uninstall ml-platform -n ml-platform
helm uninstall ingress-nginx -n ingress-nginx
cd terraform/gke && terraform destroy
```

---

## Repository structure

```
src/              # train.py, serve.py
docker/           # train.Dockerfile, serve.Dockerfile
helm/
  ml-platform/    # Helm chart (values.yaml, values-homelab.yaml, values-gke.yaml)
  monitoring/     # kube-prometheus-stack wrapper
k8s/              # tailscale-proxy.yaml (GKE ↔ homelab bridge)
scripts/          # promote.py, canary_check.py, canary_promote.sh
terraform/gke/    # GKE Autopilot cluster (Terraform)
tests/            # smoke_test.py, locustfile.py
.github/workflows/ci.yml
```

## CI/CD

Pipeline runs on a self-hosted runner (homelab). Trigger: `workflow_dispatch`.

Steps: lint (ruff) → build images → quick train → smoke-test → push to ghcr.io → deploy to k3s.
