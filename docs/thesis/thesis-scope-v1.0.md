#context/university🎓/thesis 

# Scope Document v1.0

## Бакалаврська кваліфікаційна робота

**Тема:** Проєктування та розгортання контейнерної інфраструктури для систем машинного навчання

**Студент:** Волинець Артур Васильович, група ПП-43

**Кафедра:** Системи автоматизованого проєктування, НУ «Львівська політехніка»

**Керівник:** к.т.н, доц. каф. САП Бокла Наталія Іванівна

**Дата:** квітень 2026

---

## 1. Мета роботи

Спроєктувати та реалізувати контейнерне середовище для повного життєвого циклу ML-моделей (тренування → реєстрація → розгортання → моніторинг → оновлення), яке забезпечує відтворюваність експериментів, автоматизоване розгортання та масштабованість у Kubernetes-кластері.

---

## 2. Scope (обсяг роботи)

### 2.1. In Scope — що входить у роботу

#### Контейнеризація ML-сервісів
- Multi-stage Docker builds для тренування та інференсу
- Окремі образи: train, serve, monitoring
- Оптимізація розміру образів (slim base images)
- Non-root користувач у контейнерах

#### Оркестрація (Kubernetes)
- Розгортання на локальному кластері (k3s на Asus homelab)
- Розгортання на хмарному кластері (GKE Autopilot)
- Helm-чарти для всіх компонентів системи
- Порівняння розгортання: k3s (локально) vs GKE (хмара)
- Namespace-ізоляція (dev / prod)

#### CI/CD пайплайн
- GitHub Actions: lint → test → build image → push to registry → deploy
- Автоматичний тригер при пуші у main
- Smoke-тести після деплою

#### Моніторинг та алертинг
- Prometheus — збір метрик (latency, throughput, error rate, resource usage)
- Grafana — дашборди візуалізації
- Алерти при деградації latency або зростанні error rate

#### A/B тестування моделей
- Canary deployment: поступове переключення трафіку між версіями моделі
- Два Kubernetes Deployments з розподілом трафіку через Service/Ingress
- Метрики порівняння: accuracy, latency, error rate

#### GPU-підтримка
- nvidia-device-plugin для Kubernetes на Asus homelab (GeForce 940M)
- Порівняння CPU vs GPU інференс (latency benchmarks)
- Конфігурація resource requests/limits для GPU

#### ML-компонент (демонстраційний)
- Задача: класифікація зображень (CIFAR-10 або аналогічний датасет)
- Модель: MobileNetV2 або ResNet-18 (pretrained + fine-tuning)
- Фреймворк: PyTorch
- MLflow для трекінгу експериментів та реєстрації моделей
- REST API інференсу (FastAPI)

---

### 2.2. Out of Scope — що НЕ входить у роботу

| Аспект | Причина виключення |
|---|---|
| Розподілене тренування (multi-node) | Потребує значних ресурсів, окрема тема |
| Kubeflow повноцінний | Надто складний для 2 місяців з базовим K8s |
| Автоматичний retraining pipeline | Потребує data drift detection, MLOps-зрілість |
| Продакшн-ready security (RBAC, network policies, secrets management) | Виходить за рамки бакалаврської |
| Підтримка кількох GPU / multi-GPU інференс | Апаратні обмеження (одна 940M) |
| Terraform / IaC для хмарної інфраструктури | Фокус на контейнеризації, не на IaC |
| Навчання складних моделей з нуля | ML-модель є демонстраційною, фокус — інфраструктура |

---

## 3. Інфраструктура

### 3.1. Середовища розгортання

| Середовище | Призначення | Конфігурація |
|---|---|---|
| MacBook Pro M1 | Розробка, написання коду та Helm-чартів | macOS, Docker Desktop |
| Asus homelab (Ubuntu 24.04) | Локальний k3s-кластер, GPU-інференс | i5-6200U, 8 GB RAM, GeForce 940M (2 GB), 931 GB HDD |
| GKE Autopilot | Хмарний деплой, фінальна демонстрація | Google Cloud Free Trial або бюджет ≤50€ |

### 3.2. Розподіл робіт по середовищах

| Етап | Середовище |
|---|---|
| Розробка коду, Dockerfile, Helm-чартів | MacBook M1 |
| Тестування k3s, GPU-інференс, моніторинг | Asus homelab |
| Фінальне розгортання, скріншоти для БКР | GKE Autopilot |

### 3.3. Ризики інфраструктури

| Ризик | Вплив | Мітигація |
|---|---|---|
| 8 GB RAM на homelab — мало для k3s + Prometheus + Grafana + ML | Кластер може OOM | Використовувати k3s (легший за Minikube), обмежити resource requests, вимикати непотрібні компоненти |
| 940M (2 GB VRAM) — обмежена для моделей | Не вмістить великі моделі | Використовувати MobileNetV2 (малий footprint), batch size = 1 |
| GKE бюджет ≤50€ | Може не вистачити на тривале тестування | Розробка локально, GKE лише для фінальної демонстрації (3–5 днів) |
| Free Trial Google Cloud може бути вже використаний | Немає безкоштовних кредитів | Перевірити завчасно, альтернатива — Civo K3s Cloud ($50 free credits) |

---

## 4. Технологічний стек

| Категорія | Технологія |
|---|---|
| Мова програмування | Python 3.11+ |
| ML-фреймворк | PyTorch |
| Дані | CIFAR-10 (torchvision) |
| Інференс API | FastAPI + Uvicorn |
| Трекінг експериментів | MLflow |
| Контейнеризація | Docker (multi-stage builds) |
| Реєстр образів | GitHub Container Registry (ghcr.io) |
| Оркестрація (лок.) | k3s |
| Оркестрація (хмара) | GKE Autopilot |
| Package manager K8s | Helm 3 |
| CI/CD | GitHub Actions |
| Моніторинг | Prometheus + Grafana |
| GPU plugin | NVIDIA device plugin for Kubernetes |
| Версіювання коду | Git + GitHub |

---

## 5. Критерії успіху (Definition of Done)

| # | Критерій | Як перевірити |
|---|---|---|
| 1 | Модель тренується у контейнері, метрики логуються в MLflow | Запуск train job → перевірка MLflow UI |
| 2 | Інференс-сервіс відповідає на /predict у K8s кластері | curl до ендпоінту → коректна JSON-відповідь |
| 3 | Helm chart розгортає всю систему однією командою | `helm install` → всі поди Running |
| 4 | CI/CD: push у main автоматично збирає та деплоїть | Перевірка GitHub Actions logs |
| 5 | Grafana дашборд показує latency, throughput, error rate | Скріншот дашборду під навантаженням |
| 6 | Canary deployment перемикає трафік між двома версіями моделі | Два деплойменти, розподіл трафіку видно у метриках |
| 7 | GPU-інференс працює на homelab | nvidia-smi показує утилізацію, порівняння latency CPU vs GPU |
| 8 | Система працює як на k3s, так і на GKE без змін у чартах | Деплой на обох кластерах з тим самим Helm chart |

---

## 6. Орієнтовний план (8 тижнів)

| Тиждень | Фокус | Результат |
|---|---|---|
| 1 | Dockerfiles (train + serve), локальна перевірка | Працюючі образи, інференс через docker run |
| 2 | k3s на homelab, Helm chart v1, базовий деплой | Модель працює у K8s кластері |
| 3 | MLflow інтеграція, GPU plugin, CPU vs GPU бенчмарки | Трекінг у MLflow, GPU-інференс працює |
| 4 | CI/CD (GitHub Actions), автоматична збірка та деплой | Push → build → deploy автоматично |
| 5 | Prometheus + Grafana, дашборди, алерти | Моніторинг працює, дашборд готовий |
| 6 | Canary deployment, A/B тестування | Два деплойменти, розподіл трафіку |
| 7 | GKE деплой, порівняння k3s vs GKE, фінальні тести | Система працює у хмарі |
| 8 | Написання тексту БКР, оформлення, підготовка до захисту | Готовий текст роботи |

---

## 7. Структура репозиторію (орієнтовна)

```
ml-container-platform/
├── docker/
│   ├── train.Dockerfile
│   └── serve.Dockerfile
├── src/
│   ├── train.py
│   ├── serve.py
│   └── model/
├── helm/
│   └── ml-platform/
│       ├── Chart.yaml
│       ├── values.yaml
│       ├── values-gpu.yaml
│       └── templates/
│           ├── train-job.yaml
│           ├── serve-deployment.yaml
│           ├── serve-canary.yaml
│           ├── service.yaml
│           ├── ingress.yaml
│           ├── prometheus/
│           └── grafana/
├── .github/
│   └── workflows/
│       └── ci-cd.yaml
├── tests/
│   └── smoke_test.py
├── docs/
└── README.md
```

---

## 8. Примітки

- Цей документ є живим і може оновлюватися протягом роботи
- Пріоритет: ядро системи (Docker + K8s + інференс) → моніторинг → CI/CD → A/B → GPU
- Якщо часу не вистачає, GPU та A/B можуть бути спрощені (описані архітектурно + мінімальна реалізація)
- ML-модель є засобом демонстрації інфраструктури, а не основним фокусом роботи
