import os
import pandas as pd
from apify_client import ApifyClient
from tqdm import tqdm
from multiprocessing import Pool
import glob

# Set your Apify API token here
APIFY_TOKEN = ""
BATCH_SIZE = 20  # Number of videos per Apify call
NUM_WORKERS = 20

client = ApifyClient(APIFY_TOKEN)


def fetch_transcripts_batch(video_ids):
    """Fetch transcripts for a batch of video IDs using local mapping."""
    video_urls = [f"https://www.youtube.com/watch?v={vid}" for vid in video_ids]
    url_to_vid = dict(zip(video_urls, video_ids))  # Map back from URL

    run_input = {"startUrls": video_urls}
    results = {}

    try:
        run = client.actor("topaz_sharingan/youtube-transcript-scraper").call(
            run_input=run_input,
            memory_mbytes=2048,
            timeout_secs=900,
            build="latest"
        )

        # Loop over Apify results
        for item in client.dataset(run["defaultDatasetId"]).iterate_items():
            url = item.get("url")
            transcript = item.get("transcript", "No transcription")

            if url and url in url_to_vid:
                vid = url_to_vid[url]
                results[vid] = transcript

    except Exception as e:
        print(f"Batch error ({video_ids}): {e}")

    # Check for mismatch (informational only)
    if len(results) != len(video_ids):
        print(f" Mismatch: Got {len(results)} transcripts for {len(video_ids)} videos")

    return results



def chunkify(lst, size):
    """Split list into chunks of given size."""
    return [lst[i:i + size] for i in range(0, len(lst), size)]


def main(multi_file=True, input_base="video_chunks/video_chunk"):
    if multi_file:
        input_files = sorted(glob.glob(f"input_data/{input_base}_*.csv"))
    else:
        input_files = [f"input_data/{input_base}.csv"]

    os.makedirs("collected_data", exist_ok=True)

    for input_path in input_files:
        print(f"Processing: {input_path}")
        df = pd.read_csv(input_path)
        unique_video_ids = df['video_id'].dropna().unique().tolist()

        video_batches = chunkify(unique_video_ids, BATCH_SIZE)
        print(f"Total videos: {len(unique_video_ids)} | Total batches: {len(video_batches)}")

        all_results = []
        with Pool(processes=NUM_WORKERS) as pool:
            for batch_result in tqdm(pool.imap(fetch_transcripts_batch, video_batches), total=len(video_batches)):
                all_results.append(batch_result)

        transcript_map = {}
        for batch in all_results:
            transcript_map.update(batch)

        df['transcription'] = df['video_id'].map(transcript_map)

        # Save to separate output file
        filename = os.path.splitext(os.path.basename(input_path))[0]
        output_path = f"collected_data/{filename}_transcriptions.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved with transcriptions: {output_path}")



if __name__ == '__main__':
    main()
