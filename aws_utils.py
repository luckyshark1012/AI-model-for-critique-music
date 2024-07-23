import boto3
import os


# song_name = "1 hope ur ok.mp3"
# bucket_name='paramusicgroup'
# bucket_name='paradbbackup'
# bucket_region = 'us-east-1'

# song_name = "10 Believer.mp3"
# bucket_name='top100songstreams'
# bucket_region = 'us-west-2'

aws_access_key_id = "xxxxx"
aws_secret_access_key = "xxxxxx"


def download_song_from_awss3(song_name, bucket_name, bucket_region):
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=bucket_region
    )
    
    s3 = session.client('s3')
    
    obj = s3.get_object(Bucket=bucket_name, Key=song_name)
    song_content = obj['Body'].read()
    
    temp_dir = 'temp/'
    
    if not os.path.isdir(temp_dir):
        os.makedirs(temp_dir)
        
    local_file_path = f'{temp_dir}song.mp3'
    with open(local_file_path, 'wb') as file:
        file.write(song_content)
    
    return local_file_path


def remove_temp_file(local_file_path):
    os.remove(local_file_path)
    return

