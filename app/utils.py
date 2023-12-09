import os
import tarfile

from minio import Minio, S3Error


class DatasetMover:
    def _compress_folder(self, folder_path: str, output_filename: str) -> None:
        with tarfile.open(output_filename, "w:gz") as tar:
            tar.add(folder_path, arcname=os.path.basename(folder_path))

    def _upload_to_s3(self, file_name: str, bucket_name: str, object_name: str) -> None:
        try:
            # Initialize MinIO client
            minio_client = Minio(
                os.environ["S3_ENDPOINT"],
                access_key=os.environ["S3_ACCESS_KEY_ID"],
                secret_key=os.environ["S3_SECRET_ACCESS_KEY"],
                region=os.environ["S3_REGION"],
                secure=os.environ.get("S3_SECURE", "True").lower() == "true",
            )  # Use secure=False if not using https
            minio_client.fput_object(bucket_name, object_name, file_name)
            print(f"'{file_name}' is successfully uploaded as '{object_name}' to bucket '{bucket_name}'.")
        except S3Error as e:
            print("Error occurred: ", e)

    def upload(self, folder_path: str, output_filename: str, bucket_name: str) -> None:
        self._compress_folder(folder_path, output_filename)
        self._upload_to_s3(output_filename, bucket_name, output_filename)

    def _download_from_s3(self, bucket_name: str, object_name: str, file_name: str) -> None:
        try:
            # Initialize MinIO client
            minio_client = Minio(
                os.environ["S3_ENDPOINT"],
                access_key=os.environ["S3_ACCESS_KEY_ID"],
                secret_key=os.environ["S3_SECRET_ACCESS_KEY"],
                region=os.environ["S3_REGION"],
                secure=os.environ.get("S3_SECURE", "True").lower() == "true",
            )
            minio_client.fget_object(bucket_name, object_name, file_name)
            print(f"'{object_name}' from bucket '{bucket_name}' is successfully downloaded as '{file_name}'.")
        except S3Error as e:
            print("Error occurred: ", e)

    def _decompress_folder(self, input_filename: str, output_folder_path: str) -> None:
        try:
            with tarfile.open(input_filename, "r:gz") as tar:
                tar.extractall(path=output_folder_path)
            print(f"'{input_filename}' is successfully decompressed to '{output_folder_path}'.")
        except Exception as e:
            print("Error occurred: ", e)

    def download(self, bucket_name: str, object_name: str, output_folder_path: str) -> None:
        temp_filename = "temp.tar.gz"
        self._download_from_s3(bucket_name, object_name, temp_filename)
        self._decompress_folder(temp_filename, output_folder_path)
        os.remove(temp_filename)  # Clean up the temporary compressed file
