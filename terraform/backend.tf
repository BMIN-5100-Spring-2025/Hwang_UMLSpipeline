terraform {
  backend "s3" {
    bucket = "bmin5100-terraform-state"
    region = "us-east-1"
    key    = "Sy.Hwang@Pennmedicine.upenn.edu-UMLSPipeline/terraform.tfstate"
    encrypt = true
  }
} 