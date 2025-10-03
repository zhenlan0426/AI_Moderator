#!/usr/bin/env python3
"""
OpenAI Moderation API High-Throughput Optimized Script
Maximizes throughput for rate limits: 1,000 RPM (16.67 req/sec) and 150,000 TPM (2,500 tokens/sec)
Uses aggressive batching and minimal delays for maximum speed.
"""

import pandas as pd
import os
import argparse
from openai import OpenAI
import time
from typing import Dict, Any, List
import json
from tqdm import tqdm
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

def process_moderation_response(response) -> Dict[str, float]:
    """Extract category scores from OpenAI moderation response"""
    if not response.results:
        return {}
    
    result = response.results[0]
    category_scores = result.category_scores
    
    scores = {
        'harassment_score': category_scores.harassment,
        'harassment_threatening_score': category_scores.harassment_threatening,
        'hate_score': category_scores.hate,
        'hate_threatening_score': category_scores.hate_threatening,
        'illicit_score': category_scores.illicit,
        'illicit_violent_score': category_scores.illicit_violent,
        'self_harm_score': category_scores.self_harm,
        'self_harm_instructions_score': category_scores.self_harm_instructions,
        'self_harm_intent_score': category_scores.self_harm_intent,
        'sexual_score': category_scores.sexual,
        'sexual_minors_score': category_scores.sexual_minors,
        'violence_score': category_scores.violence,
        'violence_graphic_score': category_scores.violence_graphic,
        'flagged': result.flagged,
    }
    
    categories = result.categories
    scores.update({
        'harassment_flag': categories.harassment,
        'harassment_threatening_flag': categories.harassment_threatening,
        'hate_flag': categories.hate,
        'hate_threatening_flag': categories.hate_threatening,
        'illicit_flag': categories.illicit,
        'illicit_violent_flag': categories.illicit_violent,
        'self_harm_flag': categories.self_harm,
        'self_harm_instructions_flag': categories.self_harm_instructions,
        'self_harm_intent_flag': categories.self_harm_intent,
        'sexual_flag': categories.sexual,
        'sexual_minors_flag': categories.sexual_minors,
        'violence_flag': categories.violence,
        'violence_graphic_flag': categories.violence_graphic,
    })
    
    return scores

class RateLimiter:
    """Rate limiter to ensure we don't exceed API limits"""
    def __init__(self, max_requests_per_minute: int = 950):  # 50 request buffer
        self.max_requests_per_minute = max_requests_per_minute
        self.requests = []
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Wait if we're approaching rate limit"""
        with self.lock:
            now = time.time()
            # Remove requests older than 1 minute
            self.requests = [req_time for req_time in self.requests if now - req_time < 60]
            
            if len(self.requests) >= self.max_requests_per_minute:
                # Wait until the oldest request is more than 1 minute old
                wait_time = 60 - (now - self.requests[0]) + 0.1  # Small buffer
                if wait_time > 0:
                    time.sleep(wait_time)
                    # Clean up again after waiting
                    now = time.time()
                    self.requests = [req_time for req_time in self.requests if now - req_time < 60]
            
            self.requests.append(now)

def process_single_text(client: OpenAI, text: str, rate_limiter: RateLimiter) -> Dict:
    """Process a single text with rate limiting"""
    try:
        if pd.isna(text) or text == "":
            return {}
        
        # Truncate very long texts
        if len(text) > 32000:
            text = text[:32000]
        
        # Wait for rate limit if needed
        rate_limiter.wait_if_needed()
        
        response = client.moderations.create(
            model="text-moderation-latest",
            input=text,
        )
        return process_moderation_response(response)
        
    except Exception as e:
        print(f"Error processing text: {e}")
        return {}

def moderate_texts_optimized(client: OpenAI, texts: List[str], max_workers: int = 10) -> List[Dict]:
    """
    Process texts with high throughput using threading
    Max workers set to 10 to balance speed vs rate limits
    """
    rate_limiter = RateLimiter(max_requests_per_minute=950)  # 50 request buffer under 1000 limit
    results = [None] * len(texts)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(texts), desc="Processing texts") as pbar:
            # Submit all tasks
            futures = []
            for i, text in enumerate(texts):
                future = executor.submit(process_single_text, client, text, rate_limiter)
                futures.append((i, future))
            
            # Collect results as they complete
            for i, future in futures:
                try:
                    results[i] = future.result()
                except Exception as e:
                    print(f"Error in future {i}: {e}")
                    results[i] = {}
                pbar.update(1)
    
    return results

def save_checkpoint(df: pd.DataFrame, output_file: str):
    """Save checkpoint"""
    checkpoint_file = output_file.replace('.csv', '_checkpoint.csv')
    df.to_csv(checkpoint_file)
    print(f"Checkpoint saved to {checkpoint_file}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='High-throughput OpenAI moderation processing')
    parser.add_argument('--input', '-i', help='Input CSV file path (required unless using --resume)')
    parser.add_argument('--output', '-o', help='Output CSV file path')
    parser.add_argument('--max-rows', '-m', type=int, help='Maximum number of rows to process')
    parser.add_argument('--workers', '-w', type=int, default=8, help='Number of worker threads (default: 8)')
    parser.add_argument('--chunk-size', '-c', type=int, default=1000, help='Process in chunks of this size (default: 1000)')
    parser.add_argument('--resume', '-r', help='Resume from checkpoint file')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.input and not args.resume:
        parser.error("Either --input or --resume must be provided")
    
    # Set up file paths
    if args.resume:
        input_file = args.resume
        if args.output:
            output_file = args.output
        else:
            output_file = input_file  # Overwrite the checkpoint file
    else:
        input_file = args.input
        if args.output:
            output_file = args.output
        else:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_optimized_moderation.csv"
    
    # Initialize OpenAI client
    try:
        client = OpenAI()
        print("OpenAI client initialized successfully")
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        return
    
    print(f"Processing: {input_file}")
    print(f"Output: {output_file}")
    print(f"Workers: {args.workers}")
    print(f"Chunk size: {args.chunk_size}")
    
    # Load dataset
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        df = pd.read_csv(input_file, index_col=0)
    else:
        print("Loading dataset...")
        df = pd.read_csv(input_file, index_col=0)
    
    print(f"Loaded {len(df)} rows")
    
    if args.max_rows:
        df = df.head(args.max_rows)
        print(f"Limited to {len(df)} rows for testing")
    
    # Check for existing moderation columns
    moderation_cols = [col for col in df.columns if col.endswith('_score') or col.endswith('_flag') or col == 'flagged']
    if moderation_cols:
        unprocessed_mask = df['flagged'].isna()
        unprocessed_count = unprocessed_mask.sum()
        if unprocessed_count == 0:
            print("All rows already processed!")
            return
        print(f"Found {unprocessed_count} unprocessed rows")
        process_df = df[unprocessed_mask].copy()
    else:
        process_df = df.copy()
        unprocessed_count = len(df)
    
    print(f"Starting high-throughput processing of {unprocessed_count} texts...")
    
    start_time = time.time()
    
    # Process in chunks to manage memory and allow checkpoints
    chunk_size = args.chunk_size
    total_processed = 0
    
    for chunk_start in range(0, len(process_df), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(process_df))
        chunk_df = process_df.iloc[chunk_start:chunk_end].copy()
        
        print(f"\nProcessing chunk {chunk_start//chunk_size + 1}/{(len(process_df) + chunk_size - 1)//chunk_size}")
        print(f"Rows {chunk_start + 1} to {chunk_end}")
        
        # Extract texts for this chunk
        texts = chunk_df['text'].tolist()
        
        # Process chunk
        chunk_results = moderate_texts_optimized(client, texts, max_workers=args.workers)
        
        # Add results back to dataframe
        for i, result in enumerate(chunk_results):
            idx = chunk_df.index[i]
            for key, value in result.items():
                df.loc[idx, key] = value
        
        total_processed += len(chunk_results)
        
        # Save checkpoint
        save_checkpoint(df, output_file)
        
        # Print progress
        elapsed = time.time() - start_time
        rate = total_processed / elapsed
        remaining = unprocessed_count - total_processed
        eta = remaining / rate if rate > 0 else 0
        
        print(f"Progress: {total_processed}/{unprocessed_count} ({total_processed/unprocessed_count*100:.1f}%)")
        print(f"Rate: {rate:.2f} texts/second")
        print(f"ETA: {eta/60:.1f} minutes")
    
    # Save final results
    print(f"\nSaving final results to {output_file}")
    df.to_csv(output_file)
    
    # Final statistics
    elapsed = time.time() - start_time
    print(f"\n🎉 Processing complete!")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Average rate: {unprocessed_count/elapsed:.2f} texts/second")
    print(f"Total requests: {unprocessed_count}")
    
    # Analyze results
    if unprocessed_count > 0:
        flagged_rows = df['flagged'] == True
        flagged_count = flagged_rows.sum()
        print(f"\nResults Summary:")
        print(f"Texts flagged: {flagged_count} ({flagged_count/len(df)*100:.1f}%)")
        
        # Category breakdown
        category_counts = {}
        for col in df.columns:
            if col.endswith('_flag'):
                category = col.replace('_flag', '')
                count = df[col].sum()
                if count > 0:
                    category_counts[category] = count
        
        if category_counts:
            print("Category breakdown:")
            for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {category}: {count} ({count/len(df)*100:.1f}%)")

if __name__ == "__main__":
    main()
