import os
import sys
import argparse
from PIL import Image
import imagehash
from multiprocessing import cpu_count, Pool
from tqdm import tqdm
import math
import pickle

# Supported image extensions
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'}
# Hash size used by imagehash.phash (usually 64 bits)
# If using a different hash function, adjust this.
DEFAULT_HASH_SIZE = 64
# Default name for the cache file
DEFAULT_CACHE_FILE = '.image_hashes.pkl'
# Default search folder
DEFAULT_SEARCH_FOLDER = '/Volumes/CRUCIALSSD'

def calculate_hash(image_path, hash_func=imagehash.phash):
    """
    Calculates the perceptual hash for a given image file.
    Returns the hash object or None if the file cannot be processed.
    """
    try:
        # Basic extension check first
        if not os.path.splitext(image_path)[1].lower() in IMAGE_EXTENSIONS:
            return None

        img = Image.open(image_path)
        img_hash = hash_func(img)
        return img_hash
    except IOError:
        # Suppress warnings for files that Pillow can't open by default
        # print(f"Warning: Could not read file (may not be image): {image_path}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Warning: Error processing file {image_path}: {e}", file=sys.stderr)
        return None

def find_image_files(folder_path):
    """Recursively finds all potential image files in the given folder."""
    image_files = []
    print(f"Scanning folder: {folder_path}")
    for root, _, files in os.walk(folder_path):
        for filename in files:
            # Check extension before adding to list
            if os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS:
                image_files.append(os.path.join(root, filename))
    print(f"Found {len(image_files)} potential image files to check.")
    return image_files

def convert_percent_to_distance(percent_threshold, hash_size=DEFAULT_HASH_SIZE):
    """Converts similarity percentage (100=identical) to Hamming distance threshold."""
    if not (0 <= percent_threshold <= 100):
        raise ValueError("Percentage threshold must be between 0 and 100.")
    # Formula: distance = hash_size * (1 - percent / 100)
    # We want images with distance <= calculated max distance
    max_distance = hash_size * (1.0 - percent_threshold / 100.0)
    return math.floor(max_distance) # Use floor to be inclusive

def calculate_similarity_percent(distance, hash_size=DEFAULT_HASH_SIZE):
    """Calculates similarity percentage from Hamming distance."""
    if distance < 0:
        return 0.0 # Should not happen with Hamming distance
    if distance > hash_size:
        distance = hash_size # Cap distance at hash_size

    similarity = (hash_size - distance) / hash_size * 100.0
    return similarity

def load_hashes_from_cache(cache_file):
    """Loads image hashes from a cache file."""
    if os.path.exists(cache_file):
        print(f"Loading image hashes from cache: {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load cache file {cache_file}: {e}", file=sys.stderr)
    return {}

def save_hashes_to_cache(cache_file, image_hashes):
    """Saves image hashes to a cache file."""
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(image_hashes, f)
        print(f"Saved image hashes to cache: {cache_file}")
    except Exception as e:
        print(f"Warning: Could not save cache file {cache_file}: {e}", file=sys.stderr)

def calculate_hashes_parallel(image_paths, existing_hashes=None):
    """Calculates image hashes in parallel for the given image paths."""
    num_processes = cpu_count()
    print(f"Using {num_processes} processes for hash calculation.")

    hashes = {}
    if existing_hashes:
        hashes.update(existing_hashes)

    paths_to_process = [path for path in image_paths if path not in hashes]
    total_to_process = len(paths_to_process)

    if not paths_to_process:
        print("All image hashes found in cache.")
        return hashes

    print(f"Calculating hashes for {total_to_process} new images...")
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap_unordered(calculate_hash_wrapper, paths_to_process), total=total_to_process, desc="Hashing Images"))

    for path, hash_value in results:
        if hash_value:
            hashes[path] = hash_value

    return hashes

def calculate_hash_wrapper(image_path):
    """Wrapper function for calculate_hash to return path along with hash."""
    return image_path, calculate_hash(image_path)

def parse_arguments():
    """Parses command-line arguments and prompts user for missing values."""
    parser = argparse.ArgumentParser(
        description="Find the first image in a folder similar to the input image."
    )
    parser.add_argument("input_image", nargs='?', default=None, help="Path to the input image file.")
    parser.add_argument("search_folder", nargs='?', default=None,
                        help="Path to the folder to search for similar images (includes subfolders).")
    parser.add_argument("-t", "--threshold", type=float,
                        help="Similarity threshold percentage (0-100). 100 means identical hash. Default: 90.0.")
    parser.add_argument("-c", "--concurrency", type=int,
                        help=f"Number of concurrent processes to use for hashing. Defaults to CPU count ({cpu_count()}). Use 1 for sequential processing.")
    parser.add_argument("--cache_file", type=str,
                        help=f"Path to the cache file for storing image hashes. Defaults to '{DEFAULT_CACHE_FILE}' in the search folder.")

    args = parser.parse_args()

    # If positional arguments were not provided via CLI, prompt the user
    if args.input_image is None:
        args.input_image = input("Enter the path to the input image file: ").strip()
        if not args.input_image:
            print("Error: Input image path cannot be empty.", file=sys.stderr)
            sys.exit(1)

    if args.search_folder is None:
        args.search_folder = input(f"Enter the path to the folder to search for similar images[Default: '{DEFAULT_SEARCH_FOLDER}']: ").strip()
        if args.search_folder:
            args.search_folder = os.path.abspath(args.search_folder)
        else:
            args.search_folder = DEFAULT_SEARCH_FOLDER

    # Prompt for optional arguments if not provided
    if args.threshold is None:
        threshold_input = input(f"Enter similarity threshold percentage (0-100) [Default: 90.0]: ").strip()
        if threshold_input:
            try:
                args.threshold = float(threshold_input)
            except ValueError:
                print("Warning: Invalid threshold input. Using default 90.0.", file=sys.stderr)
                args.threshold = 90.0
        else:
            args.threshold = 90.0 # Use default if input is empty

    if args.concurrency is None:
        concurrency_input = input(f"Enter number of concurrent processes [Default: {cpu_count() - 1}]: ").strip()
        if concurrency_input:
            try:
                args.concurrency = int(concurrency_input)
                if args.concurrency < 1:
                     print("Warning: Concurrency must be at least 1. Using default.", file=sys.stderr)
                     args.concurrency = cpu_count() - 1
            except ValueError:
                print(f"Warning: Invalid concurrency input. Using default {cpu_count() - 1}.", file=sys.stderr)
                args.concurrency = cpu_count() - 1
        else:
            args.concurrency = cpu_count() - 1 # Use default if input is empty

    if args.cache_file is None:
        default_cache_path = os.path.join(args.search_folder, DEFAULT_CACHE_FILE)
        cache_file_input = input(f"Enter path to cache file [Default: '{default_cache_path}' in search folder]: ").strip()
        if cache_file_input:
            args.cache_file = cache_file_input
        else:
            args.cache_file = default_cache_path

    # --- Input Validation ---
    if not os.path.isfile(args.input_image):
        print(f"Error: Input image not found: {args.input_image}")
        sys.exit(1)

    if not os.path.isdir(args.search_folder):
        print(f"Error: Search folder not found: {args.search_folder}")
        sys.exit(1)
        
    return args

def find_similar_image(args):
    """Finds the first image in a folder similar to the input image based on parsed arguments."""

    try:
        distance_threshold = convert_percent_to_distance(args.threshold)
        # Determine hash size from a sample hash if needed, but pHash is typically 64
        # For simplicity, we assume DEFAULT_HASH_SIZE is correct for pHash.
        hash_size = DEFAULT_HASH_SIZE
        print(f"Similarity threshold: {args.threshold}% translates to max Hamming distance: {distance_threshold} (for hash size {hash_size})")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # --- Determine Cache File Path ---
    cache_file_path = args.cache_file
    if cache_file_path is None:
        cache_file_path = os.path.join(args.search_folder, DEFAULT_CACHE_FILE)

    # --- Load Hashes from Cache ---
    image_hashes = load_hashes_from_cache(cache_file_path)

    # --- Find Candidate Images ---
    candidate_paths = find_image_files(args.search_folder)
    if not candidate_paths:
        print("No potential image files found in the search folder.")
        sys.exit(0)

    # --- Calculate Hashes for Candidate Images (with Concurrency) ---
    if args.concurrency > 1:
        image_hashes.update(calculate_hashes_parallel(candidate_paths, image_hashes))
    else:
        print("Calculating image hashes sequentially...")
        for path in tqdm(candidate_paths, desc="Hashing Images"):
            if path not in image_hashes:
                hash_value = calculate_hash(path)
                if hash_value:
                    image_hashes[path] = hash_value

    # --- Save Hashes to Cache ---
    save_hashes_to_cache(cache_file_path, image_hashes)

    # --- Calculate Hash for Input Image ---
    print(f"\nCalculating hash for input image: {args.input_image}")
    input_hash = calculate_hash(args.input_image)
    if input_hash is None:
        print(f"Error: Could not process input image: {args.input_image}")
        sys.exit(1)
    print(f"Input image hash: {input_hash}")

    # Resolve to absolute paths to prevent matching the same file via different relative paths
    abs_input_image_path = os.path.abspath(args.input_image)

    # --- Search for First Similar Image ---
    print(f"\nSearching for first similar image (Threshold >= {args.threshold}%)...")
    found_match = False
    processed_count = 0

    with tqdm(total=len(candidate_paths), desc="Scanning Candidates") as pbar:
        for candidate_path in candidate_paths:
            pbar.update(1)
            processed_count += 1

            # Avoid comparing the input file to itself
            abs_candidate_path = os.path.abspath(candidate_path)
            if abs_candidate_path == abs_input_image_path:
                continue

            # Get hash for the candidate image from the calculated hashes
            candidate_hash = image_hashes.get(candidate_path)

            if candidate_hash:
                # Compare hashes
                distance = input_hash - candidate_hash # Hamming distance

                # Check if similarity threshold is met
                if distance <= distance_threshold:
                    similarity_percent = calculate_similarity_percent(distance, hash_size)
                    # Ensure floating point inaccuracies don't show slightly below threshold
                    if similarity_percent >= args.threshold:
                        print("\n--- Match Found! ---")
                        print(f"Input Image:    '{args.input_image}'")
                        print(f"Similar Image:  '{candidate_path}'")
                        print(f"Hamming Distance: {distance} (Threshold <= {distance_threshold})")
                        print(f"Similarity:       {similarity_percent:.2f}% (Threshold >= {args.threshold}%)")
                        print(f"Processed {processed_count} out of {len(candidate_paths)} candidates before stopping.")
                        found_match = True
                        sys.exit(0) # Stop processing

    # --- No Match Found ---
    if not found_match:
        print(f"\nNo similar image found matching the threshold (>= {args.threshold}%)")
        print(f"Scanned {processed_count} candidate files.")
        sys.exit(0) # Exit normally, indicating completion without a match

def main():
    args = parse_arguments()
    find_similar_image(args)

if __name__ == "__main__":
    main()
