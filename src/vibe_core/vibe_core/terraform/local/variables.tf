# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

variable "acr_registry" {
  description = "ACR Registry"
}

variable "namespace" {
  default = "default"
}

variable "run_as_user_id" {
}

variable "run_as_group_id" {
}

variable "host_assets_dir" {
}

variable "kubernetes_config_path" {
  default = "~/.kube/config"
}

variable "kubernetes_config_context" {
}

variable "image_tag" {
}

variable "node_pool_name" {
  default = ""
}

variable "host_storage_path" {
}

variable "worker_replicas" {
  default = 1
}

variable "worker_memory_request" {
  description = "K8s memory request per worker pod (e.g. 100Mi, 2Gi)"
  default     = "100Mi"
}

variable "worker_memory_limit" {
  description = "K8s memory limit per worker pod. Empty string = no limit."
  default     = ""
}

variable "worker_cpu_request" {
  description = "K8s CPU request per worker pod (e.g. 500m, 0.5). Empty string = no request."
  default     = ""
}

variable "worker_cpu_limit" {
  description = "K8s CPU limit per worker pod (e.g. 2, 4000m). Empty string = no limit."
  default     = ""
}

variable "image_prefix" {
  default     = ""
  description = "Prefix for the image name"
}

variable "redis_image_tag" {
}

variable "redis_image_repository" {
  default = "bitnamilegacy/redis"
}

variable "rabbitmq_image_tag" {
}

variable "rabbitmq_image_repository" {
  default = "bitnamilegacy/rabbitmq"
}

variable "enable_telemetry" {
  description = "Use telemetry"
  type        = bool
}

variable "farmvibes_log_level" {
  default     = "INFO"
  description = "Log level to use with FarmVibes.AI services"
}

variable "max_log_file_bytes" {
 description = "Maximum size of a worker log file in bytes"
}

variable "log_backup_count" {
 description = "Number of log files to keep for each service instance"
}

variable "environment" {
  description = "Unused"
  default = ""
}
