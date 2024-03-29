#__________________________________________________________________________
#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# Author:        Matthew O.
# Date:          March 2024
#
# _________________________________________________________________________
#  This module sets up the boto3 client for AWS S3 operations.

from datetime import datetime, timezone
import boto3
from botocore.exceptions import *
import os, sys, threading
import dgbpy.keystr as dbk
import dgbpy.hdf5 as dgbhdf5
import tempfile
from pathlib import Path

import odpy.hdf5 as odhdf5
import odpy.common as odcommon

def handleS3FileSaving(savefunc, hdf5nm, bucket_name):
    """ Handles function for saving model to S3 bucket. 
    Parameters:
    savefunc: function to save model to a file(s)
    hdf5nm: Output HDF5 filename or path
    bucket_name: S3 bucket name
    """
    if dgbhdf5.isS3Uri(hdf5nm): hdf5nm = Path(hdf5nm).name
    filename = os.path.basename(hdf5nm)
    with tempfile.TemporaryDirectory(prefix='s3_model_') as tmpdirname:
        newhdf5nm = os.path.join(tmpdirname, filename) 
        if savefunc: savefunc(newhdf5nm) # Save model to tempdir
        modelfiles = getFilenamesFromPath(tmpdirname)
        s3_dest_folder = os.path.splitext(filename)[0] # Set s3 upload folder to model name
        s3_paths = createS3PathList(modelfiles, s3_dest_folder)
        upload_multiple_to_s3(modelfiles, s3_paths, bucket_name)
        odcommon.log_msg('\nModel uploaded to S3 Bucket.')


def handleS3FileLoading(loadfunc, s3Uri):
    """ Handles function for loading model from S3 bucket. 
    Parameters:
    loadfunc: function to load model from a file
    hdf5nm: Model HDF5 filepath
    bucket_name: S3 bucket name
    """
    bucket_name, s3folder = parseS3Uri(s3Uri)
    s3paths = getFilesInS3Folder(bucket_name, s3folder)
    local_paths = [os.path.basename(s3path) for s3path in s3paths]
    localhdf5path = getHdf5File(local_paths)
    s3hdf5path = getHdf5File(s3paths)
    if os.path.exists(localhdf5path) and checkLocalS3FileValidity(localhdf5path, s3hdf5path, bucket_name):
        return loadfunc(localhdf5path)
    
    download_multiple_from_s3(local_paths, bucket_name, s3paths)
    AddS3InfoToHDF5(localhdf5path, s3hdf5path, bucket_name)
    return loadfunc(localhdf5path)


def getFilesInS3Folder(bucket_name, s3_folder):
    """ Get all filepaths in an s3 folder
    Parameters:
    bucket_name: S3 bucket name
    s3_folder: S3 folder path

    Returns:
    List of files in the S3 folder
    """
    try:
        s3 = boto3.client('s3')
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=s3_folder)
        files = [obj['Key'] for obj in response['Contents']]
        return files
    except Exception as e:
        odcommon.log_msg(f'Unable to access s3 metadata: {str(e)}')
        raise e


def parseS3Uri(s3Uri):
    """ Parse s3 uri to get bucket name and path
    Parameters:
    s3Uri: S3 URI

    Returns:
    Tuple of bucket name and path
    """
    if not dgbhdf5.isS3Uri(s3Uri):
        raise ValueError('Invalid S3 URI')
    s3Uri = s3Uri.replace('s3://', '')
    bucket_name, s3_path = s3Uri.split('/', 1)
    return bucket_name, s3_path


def checkLocalS3FileValidity(localhdf5, s3hdf5path, bucket_name):
    """ Check if an s3 file should be re-downloaded
    Parameters:
    localhdf5: Local HDF5 file path
    s3hdf5path: S3 HDF5 file path
    bucket_name: S3 bucket name

    Returns: boolean
    """

    h5file = odhdf5.openFile(localhdf5, 'r')
    if not odhdf5.hasAttr(h5file, 'S3_LastModified') or not odhdf5.hasAttr(h5file, 'DateCreated'):
        return False
    
    last_modified = odhdf5.getAttr(h5file, 'S3_LastModified')
    date_created_str = odhdf5.getAttr(h5file, 'DateCreated')
    h5file.close()

    if date_created_str:
        date_created = datetime.strptime(date_created_str, '%Y-%m-%dT%H:%M:%S%z')
        now = datetime.now(timezone.utc)
        if (now - date_created).total_seconds() > 600:
            return True
    
    new_s3_last_modified = getS3ObjectLastModifiedDateTime(bucket_name, s3hdf5path)
    if last_modified:
        last_modified = datetime.strptime(last_modified, '%Y-%m-%dT%H:%M:%S%z')
    if new_s3_last_modified and new_s3_last_modified >= last_modified:
        return True
    
    return False

    
def getS3ObjectLastModifiedDateTime(bucket_name, s3_path):
    """ Get the last modified datetime of an object in S3
    Parameters:
    bucket_name: S3 bucket name
    s3_path: S3 object path

    Returns:
    Last modified datetime of the object
    """
    s3 = boto3.client('s3')
    response = s3.head_object(Bucket=bucket_name, Key=s3_path)
    return response['LastModified']


def AddS3InfoToHDF5(localhdfpath, s3hdfpath, bucket_name):
    """ Add validity datetime and model path to the HDF5 file
    Parameters:
    localhdfpath: Local HDF5 file path
    s3hdfpath: S3 HDF5 file path
    bucket_name: S3 bucket names
    """
    last_modified = getS3ObjectLastModifiedDateTime(bucket_name, s3hdfpath).isoformat()
    date_created = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    h5file = odhdf5.openFile(localhdfpath, 'a')
    odhdf5.setAttr(h5file, 'S3_LastModified', last_modified )
    odhdf5.setAttr(h5file, 'DateCreated', date_created )

    if 'model' in h5file:
        modelgrp = h5file['model']
        if odhdf5.hasAttr(modelgrp, 'path'):
            modelfnm = os.path.basename(odhdf5.getText(modelgrp, 'path'))
            modelpth = os.path.join(os.path.dirname(localhdfpath), modelfnm)
            odhdf5.setAttr(modelgrp, 'path', modelpth)
    h5file.close()


def getHdf5File(paths: list) -> str:
    """ Get the HDF5 file in a path
    Parameters:
    path: Path to file

    Returns:
    HDF5 file
    """
    for path in paths:
        if path.endswith('.h5'):
            return path
    return None


def getFilenamesFromPath(path):
    """Get all the files in a path 
    Parameters:
    path: Path to directory

    Returns:
    List of files in the directory represented by path
    """
    return [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def createS3PathList(files, destinationfolder):
    """Return a list of s3 paths for files during upload
    Parameters:
    files: List of files
    destinationfolder: Destination folder in S3

    Returns:
    List of path in s3 to upload files.
    """
    return [os.path.join(destinationfolder, os.path.basename(f)).replace('\\', '/') for f in files]


def get_s3_object_size(s3, bucket_name, s3_path):
    try:
        size = s3.head_object(Bucket=bucket_name, Key=s3_path)['ContentLength']
        return size
    except Exception as e:
        print('Error downloading file from S3')
        raise e


def upload_to_s3(local_path, s3_path, bucket_name):
    try:
        s3 = boto3.client('s3')
        total_size = os.path.getsize(local_path)
        progress = S3Progress(local_path, total_size=total_size, upload=True)
        progress.set_current_file()
        s3.upload_file(local_path, bucket_name, s3_path, Callback=progress)
    except Exception as e:
        print('Error uploading file to S3')
        raise e


def upload_multiple_to_s3(local_paths, s3_paths, bucket_name):
    try:
        s3 = boto3.client('s3')
        total_size = sum([os.path.getsize(local_path) for local_path in local_paths])
        progress = S3Progress(local_paths, total_size=total_size, upload=True)
        for local_path,s3_path in zip(local_paths, s3_paths):
            progress.set_current_file()
            s3.upload_file(local_path,  bucket_name, s3_path, Callback=progress)
    except Exception as e:
        print('Error uploading file to S3')
        raise e


def download_from_s3(local_path, bucket_name, s3_path):
    try:
        s3 = boto3.client('s3')
        total_size = get_s3_object_size(s3, bucket_name, s3_path)
        progress = S3Progress(local_path, total_size=total_size, download=True)
        progress.set_current_file()
        s3.download_file(bucket_name, s3_path, local_path, Callback=progress)
    except Exception as e:
        print('Error downloading file from S3')
        raise e


def download_multiple_from_s3(local_paths, bucket_name, s3_paths):
    try:
        s3 = boto3.client('s3')
        total_size = sum([get_s3_object_size(s3, bucket_name, s3_path) for s3_path in s3_paths])
        progress = S3Progress(local_paths, total_size=total_size, download=True)
        for local_path, s3_path in zip(local_paths, s3_paths):
            progress.set_current_file()
            s3.download_file(bucket_name, s3_path, local_path, Callback=progress)
    except Exception as e:
        print('Error downloading file from S3')
        raise e


class S3Progress:
    """
    A class to monitor and display the progress of file upload/download to/from AWS S3.
    """

    def __init__(self, filenames, total_size=0, download=False, upload=False):
        """
        Initializes the S3Progress object.

        :param filenames: A list of filenames or a single filename to be uploaded/downloaded
        :param total_size: The total size of the file(s) to be uploaded/downloaded
        :param download: A boolean flag indicating whether the operation is a download
        :param upload: A boolean flag indicating whether the operation is an upload
        """
        if not total_size:
            raise ValueError("File/Object size must be greater than zero.")

        self.filenames = filenames if isinstance(filenames, list) else [filenames]
        self.total_size = total_size
        self.seen_so_far = 0
        self.download = download
        self.upload = upload
        self.current_file_index = 0
        self.lock = threading.Lock()
        self.num_files = len(self.filenames)
        self.process_repr = ['Downloading', 'from'] if download else ['Uploading', 'to']

    def __call__(self, bytes_amount):
        """
        Update the progress based on the amount of bytes transferred.

        :param bytes_amount: The number of bytes transferred
        """
        with self.lock:
            self.seen_so_far += bytes_amount
            percentage = (self.seen_so_far / self.total_size) * 100
            self.show(percentage)

    def set_current_file(self):
        """
        Increment the counter for the number of files processed.
        """
        with self.lock:
            self.current_file_index += 1

    def show(self, percentage):
        """
        Print the current progress to stdout.

        :param percentage: The current progress percentage
        """
        sys.stdout.write(f'\r{self.process_repr[0]} model {self.process_repr[1]} S3: {percentage:.2f}% ({self.current_file_index}/{self.num_files})')
        sys.stdout.flush()
        
