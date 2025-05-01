import logging
from pathlib import Path
import boto3

def download_from_s3(bucket, prefix, local_dir):
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    local_dir = Path(local_dir)
    logging.info(f"[s3_utils] Listing objects in bucket '{bucket}' with prefix '{prefix}'")
    found_objects = False
    downloaded_files = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            found_objects = True
            key = obj['Key']
            logging.info(f"[s3_utils] Found object key: {key}")
            if key == prefix:
                logging.info(f"[s3_utils] Skipping key identical to prefix: {key}")
                continue
            try:
                rel_path = Path(key).relative_to(prefix)
            except ValueError:
                logging.warning(f"[s3_utils] Object key '{key}' not relative to prefix '{prefix}', using full key as rel_path.")
                rel_path = Path(key)
            local_path = local_dir / rel_path
            logging.info(f"[s3_utils] Calculated rel_path: '{rel_path}', local_path: '{local_path}'")
            local_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                logging.info(f"[s3_utils] Attempting download: s3://{bucket}/{key} -> {local_path}")
                s3.download_file(bucket, key, str(local_path))
                logging.info(f"[s3_utils] Download successful for: {key}")
                downloaded_files.append(str(local_path))
            except Exception as e:
                logging.error(f"[s3_utils] FAILED to download s3://{bucket}/{key} to {local_path}: {e}")
    if not found_objects:
        logging.warning(f"[s3_utils] No objects found in bucket '{bucket}' with prefix '{prefix}'")
    return downloaded_files

def download_single_file(bucket, s3_key, local_path):
    """Downloads a single specified file from S3."""
    s3 = boto3.client('s3')
    local_path = Path(local_path)
    
    # Ensure parent directory exists
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"[s3_utils] Attempting single file download: s3://{bucket}/{s3_key} -> {local_path}")
    try:
        s3.download_file(bucket, s3_key, str(local_path))
        logging.info(f"[s3_utils] Download successful for: {s3_key}")
        return True # Indicate success
    except Exception as e:
        logging.error(f"[s3_utils] FAILED to download s3://{bucket}/{s3_key} to {local_path}: {e}")
        # Depending on boto3 version/error type, check for specific errors like Not Found
        if 'Not Found' in str(e) or 'NoSuchKey' in str(e):
            logging.error(f"[s3_utils] Specific error: Key '{s3_key}' not found in bucket '{bucket}'.")
        return False # Indicate failure

def upload_to_s3(bucket, prefix, local_dir):
    s3 = boto3.client('s3')
    local_dir = Path(local_dir)
    for file_path in local_dir.rglob('*'):
        if file_path.is_file():
            rel_path = file_path.relative_to(local_dir)
            s3_key = str(Path(prefix) / rel_path)
            logging.info(f"Uploading {file_path} to s3://{bucket}/{s3_key}")
            s3.upload_file(str(file_path), bucket, s3_key) 