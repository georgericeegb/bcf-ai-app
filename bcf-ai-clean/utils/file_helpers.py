"""
File helper utilities for GCS operations and file management.
"""

import os
import logging
from typing import Optional, Union
from google.cloud import storage
from google.cloud.exceptions import NotFound, GoogleCloudError
import tempfile

logger = logging.getLogger(__name__)


def download_from_gcs(gcs_path):
    """Download file from GCS with proper error handling"""
    try:
        if not gcs_path.startswith("gs://"):
            logger.error(f"Invalid GCS path: {gcs_path}")
            return None

        parts = gcs_path[5:].split("/", 1)
        if len(parts) < 2:
            logger.error(f"Invalid GCS path format: {gcs_path}")
            return None

        bucket_name = parts[0]
        blob_path = parts[1]

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)

        if not blob.exists():
            logger.error(f"File not found in GCS: {gcs_path}")
            return None

        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.gpkg') as tmp:
            blob.download_to_filename(tmp.name)
            logger.info(f"Downloaded {gcs_path} to {tmp.name}")
            return tmp.name

    except Exception as e:
        logger.error(f"Error downloading from GCS: {e}")
        return None


def upload_to_gcs(bucket_name: str, local_file_path: str, gcs_file_path: str) -> str:
    """
    Upload a file to Google Cloud Storage.
    
    Args:
        bucket_name: GCS bucket name
        local_file_path: Local file path to upload
        gcs_file_path: Destination path in GCS bucket
        
    Returns:
        GCS file path
        
    Raises:
        FileNotFoundError: If local file doesn't exist
        GoogleCloudError: If GCS operation fails
    """
    try:
        if not os.path.exists(local_file_path):
            raise FileNotFoundError(f"Local file {local_file_path} not found")
        
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_file_path)
        
        blob.upload_from_filename(local_file_path)
        logger.info(f"Uploaded {local_file_path} to GCS as {gcs_file_path}")
        
        return gcs_file_path
        
    except Exception as e:
        logger.error(f"Error uploading {local_file_path} to GCS: {str(e)}")
        raise GoogleCloudError(f"Failed to upload file to GCS: {str(e)}")


def format_file_size(size_bytes: Union[int, float]) -> str:
    """
    Format file size in bytes to human readable format.
    
    Args:
        size_bytes: File size in bytes
        
    Returns:
        Formatted file size string (e.g., "1.2 MB", "345 KB")
    """
    if size_bytes == 0:
        return "0 B"
    
    size_bytes = float(size_bytes)
    
    # Define size units
    units = ["B", "KB", "MB", "GB", "TB"]
    
    # Find appropriate unit
    unit_index = 0
    while size_bytes >= 1024 and unit_index < len(units) - 1:
        size_bytes /= 1024
        unit_index += 1
    
    # Format with appropriate decimal places
    if unit_index == 0:  # Bytes
        return f"{int(size_bytes)} {units[unit_index]}"
    else:
        return f"{size_bytes:.1f} {units[unit_index]}"


def get_file_info(bucket_name: str, file_path: str) -> dict:
    """
    Get file information from GCS.
    
    Args:
        bucket_name: GCS bucket name
        file_path: Path to file in GCS bucket
        
    Returns:
        Dictionary with file information (size, created, updated, etc.)
        
    Raises:
        FileNotFoundError: If file doesn't exist in GCS
        GoogleCloudError: If GCS operation fails
    """
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(file_path)
        
        if not blob.exists():
            raise FileNotFoundError(f"File {file_path} not found in bucket {bucket_name}")
        
        # Reload to get latest metadata
        blob.reload()
        
        return {
            "name": blob.name,
            "size": blob.size,
            "size_formatted": format_file_size(blob.size),
            "created": blob.time_created,
            "updated": blob.updated,
            "content_type": blob.content_type,
            "md5_hash": blob.md5_hash,
            "etag": blob.etag
        }
        
    except NotFound:
        raise FileNotFoundError(f"File {file_path} not found in bucket {bucket_name}")
    except Exception as e:
        logger.error(f"Error getting file info for {file_path}: {str(e)}")
        raise GoogleCloudError(f"Failed to get file info: {str(e)}")


def list_gcs_files(bucket_name: str, prefix: str = "", limit: int = 100) -> list:
    """
    List files in GCS bucket with optional prefix filter.
    
    Args:
        bucket_name: GCS bucket name
        prefix: Optional prefix to filter files
        limit: Maximum number of files to return
        
    Returns:
        List of file information dictionaries
        
    Raises:
        GoogleCloudError: If GCS operation fails
    """
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        blobs = bucket.list_blobs(prefix=prefix, max_results=limit)
        
        files = []
        for blob in blobs:
            files.append({
                "name": blob.name,
                "size": blob.size,
                "size_formatted": format_file_size(blob.size),
                "created": blob.time_created,
                "updated": blob.updated,
                "content_type": blob.content_type
            })
        
        return files
        
    except Exception as e:
        logger.error(f"Error listing files in bucket {bucket_name}: {str(e)}")
        raise GoogleCloudError(f"Failed to list files: {str(e)}")


def delete_from_gcs(bucket_name: str, file_path: str) -> bool:
    """
    Delete a file from Google Cloud Storage.
    
    Args:
        bucket_name: GCS bucket name
        file_path: Path to file in GCS bucket
        
    Returns:
        True if file was deleted successfully
        
    Raises:
        FileNotFoundError: If file doesn't exist in GCS
        GoogleCloudError: If GCS operation fails
    """
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(file_path)
        
        if not blob.exists():
            raise FileNotFoundError(f"File {file_path} not found in bucket {bucket_name}")
        
        blob.delete()
        logger.info(f"Deleted {file_path} from GCS bucket {bucket_name}")
        
        return True
        
    except NotFound:
        raise FileNotFoundError(f"File {file_path} not found in bucket {bucket_name}")
    except Exception as e:
        logger.error(f"Error deleting {file_path} from GCS: {str(e)}")
        raise GoogleCloudError(f"Failed to delete file: {str(e)}")


def cleanup_temp_file(file_path: str) -> None:
    """
    Clean up temporary file if it exists.
    
    Args:
        file_path: Path to temporary file to delete
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup temp file {file_path}: {str(e)}")