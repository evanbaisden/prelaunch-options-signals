"""
Batch Options Data Collector
Collects historical options data in batches to work around API rate limits.
Stores results incrementally and combines them for complete analysis.
"""
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional
import logging
from pathlib import Path
import json
import requests
import os
from dotenv import load_dotenv
import time
import pickle

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BatchOptionsCollector:
    """Collects options data in batches to avoid API rate limits."""
    
    def __init__(self, batch_size: int = 3):
        self.api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if not self.api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY environment variable required")
        
        self.base_url = "https://www.alphavantage.co/query"
        self.batch_size = batch_size
        self.storage_dir = Path("data/raw/options")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Progress tracking
        self.progress_file = self.storage_dir / "collection_progress.json"
        self.progress = self.load_progress()
    
    def load_progress(self) -> Dict:
        """Load collection progress from disk."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            'completed_events': [],
            'failed_events': [],
            'last_batch': 0,
            'total_events': 0
        }
    
    def save_progress(self):
        """Save collection progress to disk."""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2, default=str)
    
    def get_options_data(self, ticker: str, event_date: date, event_name: str) -> Optional[pd.DataFrame]:
        """Get options data for a specific event."""
        # Check if already collected
        event_id = f"{ticker}_{event_date}_{event_name.replace(' ', '_')}"
        if event_id in self.progress['completed_events']:
            logger.info(f"Event {event_name} already collected, skipping...")
            return self.load_cached_data(event_id)
        
        logger.info(f"Fetching options data for {event_name} ({ticker}) on {event_date}")
        
        try:
            url = f"{self.base_url}?function=HISTORICAL_OPTIONS&symbol={ticker}&date={event_date}&apikey={self.api_key}"
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API error messages
            if 'Error Message' in data:
                logger.error(f"API Error for {event_name}: {data['Error Message']}")
                self.progress['failed_events'].append(event_id)
                return None
            
            if 'Information' in data:
                logger.warning(f"API Info for {event_name}: {data['Information']}")
                return None
            
            if 'data' not in data or not data['data']:
                logger.warning(f"No options data found for {event_name}")
                self.progress['failed_events'].append(event_id)
                return None
            
            # Convert to DataFrame
            options_list = []
            for contract in data['data']:
                options_list.append({
                    'event_id': event_id,
                    'event_name': event_name,
                    'ticker': ticker,
                    'event_date': event_date,
                    'contract_id': contract.get('contractID', ''),
                    'strike': float(contract.get('strike', 0)),
                    'expiration': contract.get('expiration', ''),
                    'option_type': contract.get('type', ''),
                    'last_price': float(contract.get('lastTradePrice', 0)),
                    'bid': float(contract.get('bid', 0)),
                    'ask': float(contract.get('ask', 0)),
                    'volume': int(contract.get('volume', 0)),
                    'open_interest': int(contract.get('openInterest', 0)),
                    'implied_volatility': float(contract.get('impliedVolatility', 0)),
                    'delta': float(contract.get('delta', 0)),
                    'gamma': float(contract.get('gamma', 0)),
                    'theta': float(contract.get('theta', 0)),
                    'vega': float(contract.get('vega', 0)),
                    'collection_timestamp': datetime.now().isoformat()
                })
            
            df = pd.DataFrame(options_list)
            
            # Cache the data
            self.cache_data(event_id, df)
            
            # Update progress
            self.progress['completed_events'].append(event_id)
            self.save_progress()
            
            logger.info(f"Successfully collected {len(df)} contracts for {event_name}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching options data for {event_name}: {e}")
            self.progress['failed_events'].append(event_id)
            self.save_progress()
            return None
    
    def cache_data(self, event_id: str, df: pd.DataFrame):
        """Cache options data to disk."""
        cache_file = self.storage_dir / f"{event_id}.csv"
        df.to_csv(cache_file, index=False)
        logger.debug(f"Cached {len(df)} contracts to {cache_file}")
    
    def load_cached_data(self, event_id: str) -> Optional[pd.DataFrame]:
        """Load cached options data from disk."""
        cache_file = self.storage_dir / f"{event_id}.csv"
        if cache_file.exists():
            return pd.read_csv(cache_file)
        return None
    
    def collect_batch(self, events_df: pd.DataFrame, batch_num: int) -> int:
        """Collect a batch of events."""
        start_idx = batch_num * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(events_df))
        
        if start_idx >= len(events_df):
            logger.info("All events processed!")
            return 0
        
        batch_events = events_df.iloc[start_idx:end_idx]
        
        logger.info(f"Processing batch {batch_num + 1}: events {start_idx + 1}-{end_idx} of {len(events_df)}")
        
        collected_count = 0
        
        for _, event in batch_events.iterrows():
            result = self.get_options_data(event['ticker'], event['announcement'], event['name'])
            if result is not None:
                collected_count += 1
            
            # Rate limiting - Alpha Vantage free tier
            logger.info("Waiting 13 seconds for API rate limit...")
            time.sleep(13)
        
        self.progress['last_batch'] = batch_num
        self.save_progress()
        
        return collected_count
    
    def run_collection_session(self, max_batches: Optional[int] = None) -> Dict:
        """Run a collection session for a specified number of batches."""
        # Load events
        events_file = Path("data/processed/events_master.csv")
        if not events_file.exists():
            raise FileNotFoundError("Events master file not found")
        
        events_df = pd.read_csv(events_file)
        events_df['announcement'] = pd.to_datetime(events_df['announcement']).dt.date
        
        # Sort by date (newest first) for better data availability
        events_df = events_df.sort_values('announcement', ascending=False)
        
        self.progress['total_events'] = len(events_df)
        
        logger.info(f"Starting collection session...")
        logger.info(f"Total events to process: {len(events_df)}")
        logger.info(f"Previously completed: {len(self.progress['completed_events'])}")
        logger.info(f"Previously failed: {len(self.progress['failed_events'])}")
        logger.info(f"Starting from batch: {self.progress['last_batch'] + 1}")
        
        session_start = datetime.now()
        total_collected = 0
        batches_processed = 0
        
        # Start from where we left off
        current_batch = self.progress['last_batch']
        
        while True:
            if max_batches and batches_processed >= max_batches:
                logger.info(f"Reached maximum batches limit ({max_batches})")
                break
            
            collected = self.collect_batch(events_df, current_batch)
            if collected == 0:
                break
            
            total_collected += collected
            batches_processed += 1
            current_batch += 1
            
            logger.info(f"Batch {current_batch} complete. Collected {collected} events this batch.")
            logger.info(f"Session total: {total_collected} events, {batches_processed} batches")
        
        session_end = datetime.now()
        session_duration = session_end - session_start
        
        return {
            'session_duration': str(session_duration),
            'batches_processed': batches_processed,
            'events_collected_this_session': total_collected,
            'total_completed_events': len(self.progress['completed_events']),
            'total_failed_events': len(self.progress['failed_events']),
            'remaining_events': len(events_df) - len(self.progress['completed_events']) - len(self.progress['failed_events'])
        }
    
    def get_collection_status(self) -> Dict:
        """Get current collection status."""
        events_file = Path("data/processed/events_master.csv")
        if events_file.exists():
            events_df = pd.read_csv(events_file)
            total_events = len(events_df)
        else:
            total_events = self.progress.get('total_events', 0)
        
        completed = len(self.progress['completed_events'])
        failed = len(self.progress['failed_events'])
        remaining = total_events - completed - failed
        
        return {
            'total_events': total_events,
            'completed_events': completed,
            'failed_events': failed,
            'remaining_events': remaining,
            'completion_percentage': (completed / total_events * 100) if total_events > 0 else 0,
            'success_rate': (completed / (completed + failed) * 100) if (completed + failed) > 0 else 0,
            'last_batch': self.progress['last_batch']
        }
    
    def combine_all_data(self) -> pd.DataFrame:
        """Combine all collected options data into a single DataFrame."""
        logger.info("Combining all collected options data...")
        
        all_data = []
        
        for event_id in self.progress['completed_events']:
            data = self.load_cached_data(event_id)
            if data is not None:
                all_data.append(data)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"Combined {len(combined_df)} total options contracts from {len(all_data)} events")
            
            # Save combined dataset
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            combined_file = Path("results") / f"combined_options_data_{timestamp}.csv"
            combined_df.to_csv(combined_file, index=False)
            logger.info(f"Saved combined dataset to {combined_file}")
            
            return combined_df
        else:
            logger.warning("No options data found to combine")
            return pd.DataFrame()
    
    def analyze_combined_data(self) -> Dict:
        """Run analysis on all collected options data."""
        combined_df = self.combine_all_data()
        
        if combined_df.empty:
            return {'error': 'No options data available for analysis'}
        
        # Import the analysis functions
        from options_integration_analysis import SimpleOptionsAnalyzer
        
        # Create event-level summary
        event_summaries = []
        
        for event_id in combined_df['event_id'].unique():
            event_data = combined_df[combined_df['event_id'] == event_id]
            
            # Get event info
            event_info = {
                'name': event_data['event_name'].iloc[0],
                'ticker': event_data['ticker'].iloc[0],
                'announcement': event_data['event_date'].iloc[0],
                'company': event_data['ticker'].iloc[0]  # Simplified
            }
            
            # Create temporary analyzer for analysis functions
            analyzer = SimpleOptionsAnalyzer()
            flow_analysis = analyzer.analyze_options_flow(event_data, event_info)
            event_summaries.append(flow_analysis)
        
        # Generate aggregate statistics
        if event_summaries:
            results_df = pd.DataFrame(event_summaries)
            analyzer = SimpleOptionsAnalyzer()
            aggregate_stats = analyzer.generate_aggregate_stats(results_df)
            
            # Save results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = Path("results") / f"complete_options_analysis_{timestamp}.json"
            
            analysis_results = {
                'event_results': results_df.to_dict('records'),
                'raw_data_contracts': len(combined_df),
                'events_analyzed': len(event_summaries),
                'aggregate_statistics': aggregate_stats,
                'collection_status': self.get_collection_status()
            }
            
            with open(results_file, 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)
            
            logger.info(f"Complete analysis saved to {results_file}")
            return analysis_results
        else:
            return {'error': 'No events could be analyzed'}


def main():
    """Main collection interface."""
    print("="*80)
    print("BATCH OPTIONS DATA COLLECTOR")
    print("Historical Data Collection with Rate Limit Management")
    print("="*80)
    
    collector = BatchOptionsCollector(batch_size=3)
    
    # Show current status
    status = collector.get_collection_status()
    print(f"\n[STATUS] Collection Progress:")
    print(f"  Total Events: {status['total_events']}")
    print(f"  Completed: {status['completed_events']} ({status['completion_percentage']:.1f}%)")
    print(f"  Failed: {status['failed_events']}")
    print(f"  Remaining: {status['remaining_events']}")
    print(f"  Success Rate: {status['success_rate']:.1f}%")
    
    if status['remaining_events'] == 0:
        print("\n[COMPLETE] All events have been processed!")
        print("[ANALYSIS] Running complete analysis...")
        
        results = collector.analyze_combined_data()
        if 'error' not in results:
            print(f"  Events Analyzed: {results['events_analyzed']}")
            print(f"  Total Contracts: {results['raw_data_contracts']:,}")
            
            if 'aggregate_statistics' in results:
                vol_stats = results['aggregate_statistics']['volume_statistics']
                print(f"  Average Volume per Event: {vol_stats['avg_total_volume']:,.0f}")
                print(f"  Average P/C Ratio: {vol_stats['avg_pcr_volume']:.3f}")
        
        return
    
    # Ask user how many batches to run
    print(f"\n[COLLECTION] Ready to collect options data")
    print(f"  Batch Size: {collector.batch_size} events per batch")
    print(f"  Time per Batch: ~{collector.batch_size * 13 / 60:.1f} minutes")
    
    try:
        # For demonstration, auto-run 2 batches
        max_batches = 2
        estimated_time = max_batches * collector.batch_size * 13 / 60
        print(f"\nAuto-running {max_batches} batches for demonstration (~{estimated_time:.1f} minutes)")
        print("(To run more/all batches, modify the script or run interactively)")
        
        # Run collection session
        print("\n[STARTING] Collection session...")
        session_results = collector.run_collection_session(max_batches)
        
        print("\n[SESSION COMPLETE]")
        print(f"  Duration: {session_results['session_duration']}")
        print(f"  Batches Processed: {session_results['batches_processed']}")
        print(f"  Events Collected: {session_results['events_collected_this_session']}")
        print(f"  Total Completed: {session_results['total_completed_events']}")
        print(f"  Remaining: {session_results['remaining_events']}")
        
        if session_results['remaining_events'] == 0:
            print("\n[ANALYZING] All data collected! Running complete analysis...")
            results = collector.analyze_combined_data()
            
            if 'error' not in results:
                print(f"  Final Results: {results['events_analyzed']} events, {results['raw_data_contracts']:,} contracts")
            
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Collection stopped by user")
        print("Progress has been saved. Run again to continue from where you left off.")
    except Exception as e:
        print(f"\n[ERROR] Collection failed: {e}")
        logger.error(f"Collection session failed: {e}")


if __name__ == "__main__":
    main()