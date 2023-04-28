from bark import SAMPLE_RATE, generate_audio, preload_models
import torchaudio
import torch
import click
import os
import re
import tempfile
import boto3
from tqdm import tqdm
import argparse

BUCKET = "test-bark-sagemaker"
s3_client = boto3.client('s3')

PUNCS = ['.', '?', '!']
PUNCS_COUNT = [20, 20, 60]

def list_objects_in_bucket(bucket_name):
    objects = []

    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name):
        objects.extend(page.get('Contents', []))

    return {obj['Key'] for obj in objects}

def update_object_key_cache():
    global object_key_cache
    object_key_cache = list_objects_in_bucket(BUCKET)

def file_exists_in_cache(object_key):
    return object_key in object_key_cache

def upload_file_to_s3(file_path, object_key):
    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(file_path, BUCKET, object_key)
        print(f"File {file_path} uploaded to {BUCKET}/{object_key}.")
    except Exception as e:
        print(f"Error uploading file to S3: {e}")

def save_audio_to_wav(audio_array, file_name, sample_rate):
    audio_tensor = torch.from_numpy(audio_array)
    
    if len(audio_tensor.shape) == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    
    # Create a temporary file to save the audio
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_file_path = temp_file.name
    
    # Save the audio to the temporary file
    torchaudio.save(temp_file_path, audio_tensor, sample_rate)

    # Upload the temporary file to S3
    upload_file_to_s3(temp_file_path, file_name)

    # Delete the temporary file
    os.remove(temp_file_path)

# Populate the object_key_cache initially
object_key_cache = list_objects_in_bucket(BUCKET)

def split_csv_to_parts(file_name, n_parts):
    def clean_string(s):
        # Remove spaces
        s = s.strip()
        
        # Ensure the last character is a valid alphabet or numerical
        s = re.sub(r'[^a-zA-Z0-9]+$', '', s)
        
        return s
    # Read the CSV file line by line
    with open(file_name, 'r') as f:
        items = [(idx, line.strip()) for idx, line in enumerate(f)]

    tasks = []
    finished = 0
    for row_idx, prompt in items:
        prompt = clean_string(prompt)
        for punc_idx, punc in enumerate(PUNCS):
            prompt_with_punc = prompt + punc
            for generation_count in range(PUNCS_COUNT[punc_idx]):
                save_name = f"{row_idx}-{punc_idx}-{generation_count}.wav"
                if not file_exists_in_cache(save_name):
                    tasks.append((save_name, prompt_with_punc))
                else:
                    finished += 1
    print("Total Generations Left:", len(tasks))
    print("Total Generations Finished:", finished)
    # Calculate the number of rows for each part
    rows_per_part = len(tasks) // n_parts

    # Split the list of lines into parts
    parts = []
    for i in range(n_parts):
        start = i * rows_per_part
        if i == n_parts - 1:
            end = len(tasks)  # Include the remaining rows in the last part
        else:
            end = start + rows_per_part
        parts.append(tasks[start:end])

    return parts

def main(part, n_parts):
    sentences_file = '/opt/ml/sentences_clean.csv'
    print('cuda is available: ', torch.cuda.is_available())
    print('current device: ', torch.cuda.current_device())
    print(f"Running part {part} of {n_parts} parts")
    parts = split_csv_to_parts(sentences_file, n_parts)
    curr_part = parts[part]
    print("Current Part Generations Left:", len(curr_part))
    print("single example: ", curr_part[0])
    preload_models()
    for item in tqdm(curr_part, desc="Processing items"):
        save_name, prompt = item
        if not os.path.exists(save_name):
            audio_array = generate_audio(prompt, silent=True)
            save_audio_to_wav(audio_array, save_name, SAMPLE_RATE)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_parts', type=int, default=1)
    parser.add_argument('--part', type=int, default=0)
    args, unknown = parser.parse_known_args()
    main(args.part, args.n_parts)