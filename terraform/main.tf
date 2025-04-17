# Create S3 bucket for project data
resource "aws_s3_bucket" "project_data" {
  bucket = "hwangsy-umlspipeline"
  
  tags = {
    Name  = "Sy Hwang Project Data Bucket"
    Owner = element(split("/", data.aws_caller_identity.current.arn), 1)
  }
}

# Enable versioning for the bucket
resource "aws_s3_bucket_versioning" "project_data_versioning" {
  bucket = aws_s3_bucket.project_data.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

# Configure server-side encryption for the bucket
resource "aws_s3_bucket_server_side_encryption_configuration" "project_data_encryption" {
  bucket = aws_s3_bucket.project_data.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Block public access to the bucket
resource "aws_s3_bucket_public_access_block" "project_data_public_access_block" {
  bucket = aws_s3_bucket.project_data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_ecr_repository" "umlspipeline" {
  name                 = "hwangsy_umlspipeline"
  image_scanning_configuration {
    scan_on_push = true
  }
  tags = {
    Name  = "Sy Hwang UMLSPipeline ECR"
    Owner = element(split("/", data.aws_caller_identity.current.arn), 1)
  }
} 