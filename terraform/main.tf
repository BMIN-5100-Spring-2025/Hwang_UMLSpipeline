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
  force_delete = true
  image_scanning_configuration {
    scan_on_push = true
  }
  tags = {
    Name  = "Sy Hwang UMLSPipeline ECR"
    Owner = element(split("/", data.aws_caller_identity.current.arn), 1)
  }
}

# CloudWatch Log Group for ECS Task
resource "aws_cloudwatch_log_group" "ecs_task" {
  name              = "/ecs/${var.project_name}-task"
  retention_in_days = 7
}

# IAM Role for ECS Task Execution
resource "aws_iam_role" "ecs_task_execution" {
  name = "${var.project_name}-ecs-task-execution-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_task_execution_policy" {
  role       = aws_iam_role.ecs_task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}


# IAM Role for ECS Task (app permissions)
resource "aws_iam_role" "ecs_task" {
  name = "${var.project_name}-ecs-task-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_policy" "ecs_task_s3_policy" {
  name        = "${var.project_name}-ecs-task-s3-policy"
  description = "Allow ECS task to access S3 bucket"
  policy      = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ]
      Resource = [
        aws_s3_bucket.project_data.arn,
        "${aws_s3_bucket.project_data.arn}/*"
      ]
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_task_s3_policy_attachment" {
  role       = aws_iam_role.ecs_task.name
  policy_arn = aws_iam_policy.ecs_task_s3_policy.arn
}

# ECS Task Definition
resource "aws_ecs_task_definition" "umlspipeline" {
  family                   = "${var.project_name}-task"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = "2048"
  memory                   = "12288"
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn
  ephemeral_storage {
    size_in_gib = 50
  }
  container_definitions = jsonencode([
    {
      name      = "umlspipeline"
      image     = aws_ecr_repository.umlspipeline.repository_url
      essential = true
      environment = [
        { name = "MODE", value = "fargate" },
        { name = "S3_BUCKET", value = aws_s3_bucket.project_data.id },
        { name = "INPUT_DIR", value = "/tmp/input" },
        { name = "OUTPUT_DIR", value = "/tmp/output" },
        { name = "TRANSFORMERS_CACHE", value = "/tmp/hf_cache" }
      ]
      command = [
        "python", "main.py",
        "--input", "/tmp/input/placeholder.csv",
        "--output", "/tmp/output/output.jsonl",
        "--tcol", "transcription",
        "--idcol", "note_id",
        "--umls", "/tmp/input/2020AB-full/2020AB-quickumls-install",
        "--embeddings", "/tmp/input/data/embeddings/cui2vec_pretrained.txt",
        "--mrrel", "/tmp/input/2020AB-full/MRREL.RRF",
        "--sbert-model", "/tmp/models/miniLM",
        "--fusion", "concat",
        "--fallback", "text2vec",
        "--vectors-out", "/tmp/output/doc_vectors",
        "--visualize", "/tmp/output/visualization.html"
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.ecs_task.name
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "ecs"
        }
      }
    }
  ])
}

resource "aws_s3_bucket_cors_configuration" "project_data_cors_configuration" {
  bucket = aws_s3_bucket.project_data.id

  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["GET", "POST", "PUT", "HEAD"]
    allowed_origins = ["http://localhost:5173", "http://localhost:3000", "https://hwangsy-umlspipeline.bmin-5100.com"]
    expose_headers  = ["ETag"]
    max_age_seconds = 3000
  }
}

module "invoke_fargate_lambda" {
  source = "git@github.com:BMIN-5100-Spring-2025/infrastructure.git//invoke_fargate_lambda/terraform"

  project_name = var.project_name
  ecs_task_definition_arn = aws_ecs_task_definition.umlspipeline.arn
  ecs_task_execution_role_arn = aws_iam_role.ecs_task_execution.arn
  ecs_task_task_role_arn = aws_iam_role.ecs_task.arn
  ecs_task_definition_container_name = "umlspipeline"

  ecs_cluster_arn = data.terraform_remote_state.infrastructure.outputs.ecs_cluster_arn
  ecs_security_group_id = data.terraform_remote_state.infrastructure.outputs.ecs_security_group_id
  private_subnet_id = data.terraform_remote_state.infrastructure.outputs.private_subnet_id
  api_gateway_authorizer_id = data.terraform_remote_state.infrastructure.outputs.api_gateway_authorizer_id
  api_gateway_execution_arn = data.terraform_remote_state.infrastructure.outputs.api_gateway_execution_arn
  api_gateway_id = data.terraform_remote_state.infrastructure.outputs.api_gateway_id
  environment_variables = {}
} 