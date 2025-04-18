output "bucket_name" {
  description = "The name of the S3 bucket"
  value       = aws_s3_bucket.project_data.id
}

output "bucket_arn" {
  description = "The ARN of the S3 bucket"
  value       = aws_s3_bucket.project_data.arn
}

output "ecr_repository_url" {
  description = "The URI of the ECR repository for Docker images"
  value       = aws_ecr_repository.umlspipeline.repository_url
} 