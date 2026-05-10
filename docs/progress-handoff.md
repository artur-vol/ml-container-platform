# Progress Handoff — ml-container-platform

**Адресат:** десктопний агент (Claude Code на іншій машині), який підхопить роботу.
**Автор знімку:** Claude Code (M1 MacBook), сесія від 2026-05-08.
**Мета:** дати повний контекст — що зроблено, як зроблено, коли, чим відрізняється від плану зі `docs/thesis/thesis-scope-v1.0.md`, і що робити далі.

> Перш ніж будь-що змінювати — прочитай `CLAUDE.md` (інструкції для агента) і `AGENTS.md` (правила репозиторію). Цей документ — фактичний знімок стану, він не замінює ті два.

---

## 1. Тема проєкту в одному абзаці

Бакалаврська (НУ ЛП, ПП-43, керівник — Бокла Н.І., квітень 2026): **«Проєктування та розгортання контейнерної інфраструктури для систем машинного навчання»**. Деліверабл — контейнерна платформа з повним циклом ML (train → register → deploy → monitor → A/B). Сама модель (CIFAR-10 на MobileNetV2) — *демонстратор*; робота оцінюється за **інфраструктурою**, тож при компромісах пріоритет завжди на боці інфри. Робочі мови: код/коміти/ідентифікатори — англ.; тези в `docs/thesis/` — укр., цитати у форматі `[Author, Year]` / `[файл.pdf, p. N]` зберігати дослівно.

---

## 2. Високорівневий стан станом на 2026-05-08

| Шар | Стан | Коментар |
|---|---|---|
| Документація з тези (scope/draft/blueprint) | ✅ імпортована | `docs/thesis/`, не редаговано — це baseline |
| ML-пайплайн (`src/train.py`, `scripts/promote.py`, `src/serve.py`) | ✅ працює локально end-to-end | MLflow registry, реєстрація моделі, FastAPI з `/predict /health /metrics` |
| Docker-образи (train + serve, multi-stage, non-root) | ✅ збираються локально | `python:3.11-slim`, окремий serve-CPU-only варіант |
| Локальний dev driver `ops.sh` | ✅ повний: train/build/run/smoke-test/clean | MLflow підіймається з SQLite + `mlflow-artifacts:/` proxy |
| Helm chart `helm/ml-platform/` | 🟡 написаний, **end-to-end не перевірений у кластері** | values для k3d/k3s/GKE окремими файлами |
| CI/CD (`.github/workflows/`) | ❌ не створено | DoD #4 — відкритий пункт |
| Prometheus + Grafana stack | ❌ не створено | експортер у `serve.py` уже є (`/metrics`), але стек ще не задеплоєний |
| Canary / A/B (DoD #6) | ❌ не створено | потрібен другий deployment + traffic split |
| GPU на homelab (DoD #7) | ❌ не торкався | `gpu.enabled: false` у всіх values, очікує `nvidia-device-plugin` |
| Кросплатформні білди для amd64 | 🟡 інструкція в `docs/local-quickstart.md`, образи ще не пушилися в `ghcr.io` |
| Деплой на GKE (DoD #8) | ❌ stub-values тільки | `values-gke.yaml` — заглушка з TODO на тиждень 7 |
| Smoke-тест (`tests/smoke_test.py`) | ❌ ще не існує | у `tests/` лише `gen_test_image.py` + згенерований `test_image.jpg` |
| Git-історія | ❌ нуль комітів | `git status` показує всі файли як untracked |

Сумарно: **готова перша третина** робіт за планом scope §6 (тижні 1–2 практично закриті, тиждень 3 частково — MLflow інтегровано, GPU ще ні).

---

## 3. Розкладка по компонентах — *що* і *як* зроблено

### 3.1. Документи з тези (`docs/thesis/`)
- `thesis-scope-v1.0.md` — обсяг, DoD (8 пунктів), 8-тижневий план, обмеження по залізу. **Це нормативний документ — будь-яке розширення обсягу прапорувати, не виконувати тихо.**
- `thesis-draft-v2.0.md` — структура самої БКР (розділи).
- `thesis-blueprint.md` — текст-заготовка для розділу 1 (літогляд) з академічними цитуваннями.
- Файли імпортовані 2026-04-28 о 00:14, з тих пір не редаговані. Версія в імені (`v1.0`, `v2.0`) — при суттєвих правках бампити, а не перезаписувати.

### 3.2. ML-пайплайн

#### `src/train.py` — тренування + реєстрація
- Click CLI: `--epochs`, `--subset`, `--batch-size`, `--lr`, `--num-workers`, `--tracking-uri`, `--experiment`, `--register/--no-register`.
- Модель: `mobilenet_v2(weights=IMAGENET1K_V2)`, голова замінюється на `nn.Linear(in_features, 10)`. Backbone **не** заморожується — повний fine-tune швидший за feature-extraction на CIFAR-10.
- Дані: `torchvision.datasets.CIFAR10`, кеш у `./data` (керується `CIFAR_DATA_DIR`). Resize до 96×96 (нижче — depthwise convs MobileNet «згортають» просторовість), нормалізація ImageNet means/stds. Subset режим: `train = first N`, `test = max(N//5, 200)`.
- Логування в MLflow: per-epoch `train_loss/acc`, `val_loss/acc`, `val_latency_ms_per_batch`. Окремо — `inference_latency_ms` (single-image, 3 warmup + 20 timed) — це число для DoD #7 (CPU vs GPU).
- Реєстрація: `mlflow.pytorch.log_model(... registered_model_name="cifar10-mobilenet")`. Модель `.cpu()` перед лог-моделлю — артефакт портабельний.
- Параметри також логуються через `log_params` + tag `classes=...`.
- Реальні запуски проводилися 2026-04-29 (див. `mlruns/artifacts/1/<run_id>/`); є дві registered versions на момент знімку.

#### `scripts/promote.py` — promote latest → Production
- `MlflowClient.search_model_versions("name='cifar10-mobilenet'")`, бере `max(version)`, викликає `transition_model_version_stage(stage="Production", archive_existing_versions=True)`.
- Свідомо використовує **stages** (не aliases), і саме тому `mlflow==2.17.2` запіновано — 3.x stages депрекує. Stage-based promotion потрібен для DoD #6 (canary між двома версіями).
- ENV: `MODEL_NAME`, `MLFLOW_TRACKING_URI`.

#### `src/serve.py` — FastAPI inference
- Lifespan-хук завантажує `models:/cifar10-mobilenet/Production`, при відсутності Production — fallback на `models:/.../latest`. Логує warn у разі fallback.
- Ендпоінти:
  - `GET /health` → `{"status":"ok"}`.
  - `POST /predict` (`{"image_b64": "..."}`) → `{class_name, confidence, latency_ms}`.
  - `GET /metrics` → Prometheus-сумісний експорт.
- Безпека входу: `MAX_IMAGE_BYTES = 5 MB`, `Image.MAX_IMAGE_PIXELS = 25M` (захист від decompression bomb), валідація base64 і помилок PIL → 400; перевищення розміру → 413.
- Метрики: `ml_predict_requests_total{status}` (Counter, статуси `ok|bad_request|error`), `ml_predict_latency_seconds` (Histogram).
- ENV: `MODEL_NAME`, `MODEL_STAGE`, `MLFLOW_TRACKING_URI`, `LOG_LEVEL`.

### 3.3. Docker-образи (`docker/`)

Обидва образи — **multi-stage**, фінальний шар `python:3.11-slim`, **non-root** (`appuser`, UID/GID 1000), системні бібліотеки `gcc/g++` лише на builder-стадії.

- `train.Dockerfile` — ставить `requirements-train.txt`, ENTRYPOINT = `python /app/src/train.py`. Дефолти: `CIFAR_DATA_DIR=/app/data`, `MLFLOW_TRACKING_URI=http://host.docker.internal:5000`.
- `serve.Dockerfile` — ставить `requirements-serve-cpu.txt` (це навмисно: на runtime в кластері/homelab без CUDA-wheel-ів `pip install torch` із CUDA був би +2GB). ENTRYPOINT = `uvicorn src.serve:app --host 0.0.0.0 --port 8000`. EXPOSE 8000.
- `.dockerignore` виключає `.venv/`, `data/`, `mlruns/`, `*.pt|*.pth`, `docs/`, `tests/test_image.jpg`, `CLAUDE.md`, `ops.sh`. Виключення `docs/` зменшує контекст білду — образи цілеспрямовано не несуть тезу.

### 3.4. Requirements

- `requirements-train.txt`: `torch==2.4.1`, `torchvision==0.19.1`, `mlflow==2.17.2`, `click==8.1.7`.
- `requirements-serve.txt`: full stack з `torch`/`torchvision` 2.4.1 (CUDA-варіант — для локального dev).
- `requirements-serve-cpu.txt`: те саме, але з `--index-url https://download.pytorch.org/whl/cpu` — це той файл, який тягне serve.Dockerfile.
- Пін `mlflow==2.17.2` — остання 2.x з повноцінними stages.
- Пін `torch==2.4.1` + `torchvision==0.19.1` — синхронна пара з ABI-сумісними wheels для osx-arm64 і linux-amd64.

### 3.5. `ops.sh` — локальний dev driver

Команди (всі довідуються залежностей через `ensure_*` ledger-функції):

| Команда | Що робить |
|---|---|
| `train` | піднімає MLflow → `python src/train.py --epochs 1 --subset 1000` |
| `train-full` | піднімає MLflow → повний прогон (`--epochs 10`, full CIFAR-10) |
| `promote` | викликає `scripts/promote.py` |
| `build` | `docker build` обох образів (локальна арх.) |
| `run` | стартує контейнер `ml-platform-serve` на :8000, з `--add-host=host.docker.internal:host-gateway`, монтує `mlruns/` 1:1 і `MLFLOW_TRACKING_URI=http://host.docker.internal:5000` |
| `smoke-test` | `curl /health` + POST `/predict` із `tests/test_image.jpg` |
| `clean` | вбиває контейнер і MLflow (через PIDFILE + `lsof`-fallback) |

Ключове рішення в `start_mlflow`: сервер запускається з `--default-artifact-root mlflow-artifacts:/` + `--serve-artifacts` + `--artifacts-destination $REPO_ROOT/mlruns/artifacts`. Це дає **контейнер-портабельні артефакт-URI** замість абсолютних шляхів хоста — критично, бо інакше serve-контейнер не зможе знайти артефакт.

PIDFILE: `mlruns/mlflow.pid`, лог: `mlruns/mlflow.log`. На macOS додатково kill-аємо все, що тримає порт 5000 (через `lsof`), бо `pkill -f` тут ненадійний. Останнє shutdown-повідомлення в логу — 2026-05-06 14:25.

### 3.6. Helm chart (`helm/ml-platform/`)

- `Chart.yaml`: name=`ml-platform`, version `0.1.0`, appVersion `1.0.0`.
- `values.yaml` — дефолти для **k3d local** (M1, arm64):
  - `image.repository: ml-platform/serve`, `pullPolicy: Never` (передбачає `k3d image import`).
  - `mlflow.trackingUri: http://host.k3d.internal:5000`.
  - `service.type: ClusterIP`, `port: 80`. Ingress `nginx`, host `ml-platform.local`.
  - resources: requests 100m/256Mi, limits 500m/512Mi.
  - probes: liveness initialDelay **90s** (модель довго вантажиться з MLflow при старті — закладаємось), readiness 30s.
  - Коментарі у файлі описують як підняти k3d-кластер з пристроєним Traefik або з ingress-nginx — у нагоді коли почнеш помилки траблшутити.
- `values-homelab.yaml` — overlay для k3s на Asus:
  - `image.repository: ghcr.io/artur-vol/ml-container-platform-serve`, `pullPolicy: Always` (отже, потрібен push з CI; зараз ні CI, ні push нема).
  - `mlflow.trackingUri: http://HOMELAB_IP:5000` — `HOMELAB_IP` потрібно перебити через `--set` під час деплою.
  - `gpu.enabled: false` із коментарем «flip to true once nvidia-device-plugin is ready (week 3)».
- `values-gke.yaml` — заглушка для GKE Autopilot, TODO позначено явно (`week 7`). MLflow передбачається in-cluster (`mlflow.ml-platform.svc.cluster.local`) або managed; ingress-nginx треба ставити окремо.
- Темплейти:
  - `deployment.yaml` — securityContext non-root (1000:1000), envFrom ConfigMap, liveness/readiness через `/health`, resources з values, опційний `nvidia.com/gpu: 1` + tolerations при `gpu.enabled`.
  - `service.yaml` — ClusterIP, target 8000.
  - `ingress.yaml` — конфігурується через values, опційний.
  - `configmap.yaml` — `MLFLOW_TRACKING_URI`, `MODEL_NAME`, `MODEL_STAGE`, `LOG_LEVEL`.
  - `_helpers.tpl` — стандартні `name`, `fullname`, `labels`, `selectorLabels`.

⚠️ **Жодного `helm template` / `helm install` ще не запускалось** — синтаксис на око валідний, але не перевірений на реальному кластері.

### 3.7. Тести і допоміжне

- `tests/gen_test_image.py` — argparse, тягне CIFAR-10 testset, зберігає sample 0 як JPEG у `tests/test_image.jpg` (qualtity=95). Викликається автоматично з `ops.sh smoke-test`, якщо файлу нема.
- `tests/test_image.jpg` — згенерований 2026-04-28 о 10:29, не комітиться (`.gitignore`).
- **`tests/smoke_test.py` — обіцяний у `CLAUDE.md`, але ще НЕ написаний.** Це борг.

### 3.8. Документація проєкту

- `CLAUDE.md` — інструкції для агента: контекст тези, репо-стан, команди, стек, обмеження, DoD. Версія від 2026-04-29 14:27 — синхронізувати, якщо змінюється структура.
- `AGENTS.md` — Repository Guidelines (стиль, тести, коміти).
- `docs/local-quickstart.md` — мінімальна інструкція для людини: prereqs, повний run через `ops.sh`, кросс-арч-білди, порти/ендпоінти, troubleshooting (зокрема — порт 5000 і AirPlay на macOS).
- `docs/code-walkthrough.md` — **обов'язкове до прочитання** перед першою правкою коду. Розгорнутий day-1-розбір кожного файлу: чому такі піни, чому resize до 96×96, чому stages а не aliases, чому `host.docker.internal` через `--add-host`, тощо. 679 рядків — економить багато часу при онбордингу.

---

## 4. Хронологія (по mtime файлів)

| Дата (2026) | Що зроблено |
|---|---|
| 04-28 00:14 | Імпорт `docs/thesis/{scope, draft, blueprint}` |
| 04-28 00:25–28 | `requirements-train.txt`, `requirements-serve.txt`, `train.Dockerfile`, `gen_test_image.py`, `.gitignore`, `.dockerignore` |
| 04-28 10:29 | Згенеровано `tests/test_image.jpg` |
| 04-28 11:00 | Записано `docs/code-walkthrough.md` (день-1 розбір) |
| 04-29 14:27 | Записано `CLAUDE.md` |
| 04-29 14:40 | `docs/local-quickstart.md` |
| 04-29 15:04–09 | `serve.Dockerfile`, `ops.sh`, `src/train.py`, `src/serve.py`, `scripts/promote.py`, `requirements-serve-cpu.txt` (CPU-only PyTorch wheel) |
| 04-29 16:12–18 | Helm chart: `Chart.yaml`, `values.yaml`, `values-homelab.yaml`, `values-gke.yaml`, всі темплейти |
| 04-29 22:00–01 | Реальні MLflow runs (зареєстровано версії моделі — `mlruns/artifacts/1/<run_id>`) |
| 05-02 14:16 | `AGENTS.md` |
| 05-06 14:25 | Останні signal/term у `mlruns/mlflow.log` (зупинка локального serve-стеку) |
| 05-08 | **Сьогодні. Цей документ.** |

Зауваж: `git log` порожній — **жодного коміту**. Це треба буде виправити перед налагодженням CI/CD (DoD #4 без історії не запуститься).

---

## 5. Як з нуля підняти все локально

```bash
# 1) venv + train deps
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-train.txt -r requirements-serve.txt

# 2) повний цикл
./ops.sh train          # MLflow on :5000 + 1-епохний прогон на subset 1000
./ops.sh promote        # latest → Production
./ops.sh build          # docker build train + serve (локальна архітектура)
./ops.sh run            # serve container на :8000
./ops.sh smoke-test     # GET /health + POST /predict із test_image.jpg
./ops.sh clean          # стоп контейнера + MLflow
```

Кросс-арч (для homelab/GKE — обидва amd64):
```bash
docker buildx build --platform linux/amd64 -f docker/serve.Dockerfile -t ml-platform/serve:latest .
docker buildx build --platform linux/amd64 -f docker/train.Dockerfile  -t ml-platform/train:latest  .
```

---

## 6. Стан DoD-критеріїв (scope §5) — як проєкт буде оцінюватися

| # | Критерій | Стан | Що бракує |
|---|---|---|---|
| 1 | Тренування у контейнері, метрики у MLflow | 🟡 | Тренування в **host venv** працює і пише в MLflow. Контейнер `train` зібраний, але зразкового запуску `docker run train …` під MLflow з `host.docker.internal` ще не задокументовано/не перевірено. |
| 2 | `/predict` відповідає у K8s | 🟡 | Працює локально через `docker run`. У кластері не перевірено. |
| 3 | `helm install` піднімає все | 🟡 | Чарт написаний, не запускався (`helm install` / `helm template` не виконувалися). |
| 4 | CI/CD: push у main → build + deploy | ❌ | `.github/workflows/` не існує. Репозиторій навіть не закомічений. |
| 5 | Grafana показує latency/throughput/errors | ❌ | `serve.py` уже експортує Prometheus-метрики, але стек Prom+Grafana не задеплоєний, дашборду нема. |
| 6 | Canary між двома версіями | ❌ | Потрібен другий Deployment + Service/Ingress traffic split (scope §2.1: «не blue/green, не shadow»). |
| 7 | GPU-інференс на homelab + CPU vs GPU | ❌ | `nvidia-device-plugin` ще не ставився, `gpu.enabled: false`. Метрика `inference_latency_ms` уже логується — буде про що порівнювати, коли GPU оживе. |
| 8 | Той самий чарт у k3s і GKE | 🟡 | values-overlay'ї підготовлені, але реального `helm install` ні там, ні там не було. |

Тижні з scope §6: формально закриті 1–2, частково 3 (MLflow готовий, GPU нема). Поточно ми трохи пізніше графіку (зараз тиждень ~5 за календарем — 8 квітня + 4 тижні), але рушійні шматки на місці.

---

## 7. Відомі пастки і нюанси (gotchas)

1. **MLflow артефакт-URI повинен бути проксі (`mlflow-artifacts:/`)**, інакше serve-контейнер падатиме при `mlflow.pytorch.load_model` бо побачить хост-абсолютний шлях. `ops.sh start_mlflow` уже робить правильно — не «оптимізуй» цю команду.
2. **macOS і порт 5000** — AirPlay receiver. `MLFLOW_PORT=5001 ./ops.sh train` як обхід (див. quickstart).
3. **`host.docker.internal` на Linux** — не існує без `--add-host=host.docker.internal:host-gateway`, який `ops.sh run` уже додає.
4. **MLflow 2.x stages свідомі** — не «модернізуй» `promote.py` на aliases; це зламає сюжет DoD #6.
5. **`gpu.enabled` у chart’і** додає `nvidia.com/gpu: 1` в requests і limits *і* tolerations. Без `nvidia-device-plugin` це призведе до `Unschedulable` подів — не вмикай раніше тижня роботи з GPU.
6. **`pullPolicy: Never` у дефолтному values.yaml** — навмисний для k3d-сценарію (`k3d image import`); у homelab/gke values він перевизначений на `Always`.
7. **Health-probe initialDelay 90s** — тому що завантаження моделі з MLflow + `torch.load` може довго розхитуватись на 8GB-хості. Не вріж — буде flapping.
8. **`requirements-serve.txt` vs `requirements-serve-cpu.txt`** — обидва існують навмисно. Локально (M1) ставиться повний, у serve-image — CPU-only (через `--index-url`). Не зливай.
9. **Тренування не закомічено в git** — і `data/`, і `mlruns/`, і `*.pt`/`*.pth` у `.gitignore`. При зміні машини доведеться перетренувати або принести `mlruns/` руками.
10. **`MODEL_NAME=cifar10-mobilenet`** хардкоджений у трьох місцях (train, promote, serve, configmap) — змінювати лише все одразу.

---

## 8. Що робити далі — пріоритезований бэклог

1. **Перший git commit + push на GitHub.** Без репо в GitHub нема ні CI, ні `ghcr.io` push, ні DoD #4. Запропонуй це користувачу до старту інших робіт.
2. **`tests/smoke_test.py`** — обіцяний у `CLAUDE.md`, потрібен як цільовий артефакт для `Smoke-тести після деплою` (scope §2.1) і для CI кроку «після build/deploy curl-нути сервіс».
3. **`helm template` локально** — запустити над усіма трьома values-файлами, переконатися що валідно. Потім `helm install` у `k3d` (швидке арм-кластер на M1) — закриває DoD #2 і #3 у спрощеному вигляді.
4. **`.github/workflows/ci-cd.yml`** — lint (black/ruff?) → build обох образів через `docker buildx --platform linux/amd64` → push до `ghcr.io/artur-vol/ml-container-platform-{train,serve}` → smoke. Закриває DoD #4. Перед цим треба коміт + push (див. п.1).
5. **Prometheus + Grafana stack** — додати `kube-prometheus-stack` як subchart або окремим релізом, ServiceMonitor для нашого Service, Grafana dashboard JSON (latency p50/p95/p99 з Histogram, RPS з Counter, error-rate). Закриває DoD #5.
6. **Canary** — другий Deployment (`ml-platform-canary`) із власним `MODEL_STAGE=Staging` (наприклад) + Ingress canary annotations (nginx: `nginx.ingress.kubernetes.io/canary` + `canary-weight`). Закриває DoD #6.
7. **GPU на homelab** — `kubectl apply` `nvidia-device-plugin v0.14.3` (URL у коментарі `values-homelab.yaml`); flip `gpu.enabled: true`; перетренувати/проміжно прогнати `inference_latency_ms` з CUDA — це те саме поле в MLflow, готова таблиця CPU vs GPU. Закриває DoD #7.
8. **GKE deploy** (фінальний тиждень) — оновити `values-gke.yaml` (real ingress IP, nip.io хост, MLflow strategy: in-cluster чи Cloud SQL+GCS), `gcloud container clusters get-credentials`, `helm upgrade --install`. Бюджет ≤ 50 € — тримати кластер 3–5 днів. Закриває DoD #8.
9. **Текст БКР** — заповнення `thesis-draft-v2.0.md` за матеріалом інфри, що зараз будується. Скріншоти Grafana, MLflow UI, `kubectl get pods`, latency-таблиці.

---

## 9. Що НЕ робити (з scope §2.2)

Якщо користувач попросить — прапорувати, не робити:
- розподілене / multi-node тренування,
- повноцінний Kubeflow,
- автоматичний retraining (data drift detection),
- продакшн-grade RBAC / NetworkPolicy / secrets,
- multi-GPU,
- Terraform / IaC,
- тренування великих моделей з нуля.

Це обмеження з нормативного scope-документа.

---

## 10. Швидкий чек, чи все ще працює

```bash
# у репо, на тій машині, де щойно почав
ls -la mlruns/artifacts/   # повинні бути зареєстровані версії моделі
./ops.sh train             # MLflow up + новий run за ~30s на M1
./ops.sh promote           # latest → Production
./ops.sh build && ./ops.sh run && ./ops.sh smoke-test
# очікуваний останній рядок:
# {"class_name":"...","confidence":0.xx,"latency_ms":...}
```

Якщо `smoke-test` зеленіє — все ядро (DoD #1 у локальному вигляді) працює, можна братися за п.1–9 з §8.

---

*Якщо щось у цьому документі застаріло — оновлюй, не роби паралельний.*
