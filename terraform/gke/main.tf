terraform {
  required_providers {
    google = { source = "hashicorp/google", version = "~> 6.0" }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_project_service" "container" {
  service            = "container.googleapis.com"
  disable_on_destroy = false
}

resource "google_container_cluster" "ml_platform" {
  name       = "ml-platform-thesis"
  location   = var.region
  depends_on = [google_project_service.container]

  enable_autopilot = true

  # deletion_protection=false allows `terraform destroy` without a manual step.
  # Acceptable for a short-lived thesis demo; never disable in production.
  deletion_protection = false

  # Required for Autopilot — enables VPC-native networking (alias IPs).
  ip_allocation_policy {}
}

output "get_credentials_cmd" {
  value = "gcloud container clusters get-credentials ${google_container_cluster.ml_platform.name} --region ${var.region} --project ${var.project_id}"
}
