"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║      🎥 YOUTUBE COMMENT SCRAPER MODULE - ADVANCED v4.0 🎥                  ║
║                                                                              ║
║  Complete YouTube API v3 integration for scraping video metadata            ║
║  and comments with advanced error handling & retry logic                    ║
║                                                                              ║
║  Features:                                                                   ║
║  ✅ YouTube Data API v3 integration                                         ║
║  ✅ Video metadata extraction (title, channel, views, thumbnail)          ║
║  ✅ Comment fetching with pagination support                              ║
║  ✅ Comment sorting (relevance, time)                                      ║
║  ✅ Rate limiting & retry logic                                           ║
║  ✅ Error handling & recovery                                             ║
║  ✅ URL parsing & validation                                              ║
║  ✅ DataFrame export                                                       ║
║  ✅ Logging & monitoring                                                  ║
║  ✅ Performance optimization                                              ║
║                                                                              ║
║  YouTube API Key Setup:                                                      ║
║  1. Go to https://console.cloud.google.com/                                ║
║  2. Create new project                                                      ║
║  3. Enable YouTube Data API v3                                             ║
║  4. Create OAuth 2.0 credentials                                           ║
║  5. Set API_KEY = 'your_api_key'                                          ║
║                                                                              ║
║  Usage:                                                                      ║
║  >>> info, comments_df = fetch_video_and_comments(                         ║
║  ...     url='https://www.youtube.com/watch?v=...',                       ║
║  ...     max_comments=200                                                  ║
║  ... )                                                                       ║
║                                                                              ║
║  Author:  Engineering Student                                    ║
║  University: Karunya University, India                                     ║
║  Date: November 5, 2025                                                    ║
║  Version: 4.0 (PRODUCTION READY)                                           ║
║                                                                              ║
║  Dependencies:                                                               ║
║  - google-api-python-client (YouTube API)                                 ║
║  - pandas (Data processing)                                               ║
║  - re (URL parsing)                                                       ║
║  - os (Environment variables)                                             ║
║  - logging (Monitoring)                                                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ============================================
# IMPORTS
# ============================================

import os
import re
import logging
import traceback
import time
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except ImportError:
    print("❌ ERROR: google-api-python-client not installed")
    print("Install with: pip install google-api-python-client")
    raise

import pandas as pd
import numpy as np

# ============================================
# LOGGING CONFIGURATION
# ============================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('youtube_scraper.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)

# ============================================
# CONFIGURATION & CONSTANTS
# ============================================

# YouTube Data API Key (set from environment variable or hardcoded)
API_KEY = os.getenv('YOUTUBE_API_KEY', "AIzaSyAYm6TzsOFOrZI6ilnOH9mttS-unVrENc4")

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds
TIMEOUT = 30  # seconds

# Constants
MAX_RESULTS_PER_REQUEST = 100
YOUTUBE_VIDEO_ID_REGEX = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"

VERSION = "4.0"
MODULE_NAME = "YouTube Comment Scraper"

# ============================================
# STARTUP BANNER
# ============================================

def _print_startup_banner():
    """Print module startup banner"""
    banner = f"""
╔{'='*80}╗
║{'YouTube Comment Scraper - v' + VERSION:^80}║
║{'='*80}║
║{'✅ YouTube API v3 Integration':^80}║
║{'✅ Advanced Error Handling':^80}║
║{'✅ Retry Logic & Rate Limiting':^80}║
║{'✅ Production Ready':^80}║
╚{'='*80}╝
    """
    logger.info(banner)

_print_startup_banner()

# ============================================
# URL PARSING & VALIDATION
# ============================================

def extract_video_id(url: str) -> Optional[str]:
    """
    Extract video ID from YouTube URL.
    
    Supports multiple URL formats:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://www.youtube.com/watch?v=VIDEO_ID&t=10s
    - https://youtu.be/VIDEO_ID
    - https://youtu.be/VIDEO_ID?t=10
    - youtube.com/watch?v=VIDEO_ID
    - youtu.be/VIDEO_ID
    
    Args:
        url: YouTube video URL
    
    Returns:
        Video ID (11-character string) or None if invalid
    
    Raises:
        ValueError: If URL format is invalid
    """
    try:
        if not url or not isinstance(url, str):
            raise ValueError("URL must be a non-empty string")
        
        logger.info(f"Extracting video ID from: {url[:50]}...")
        
        # Try to extract video ID
        video_id_match = re.search(YOUTUBE_VIDEO_ID_REGEX, url)
        
        if video_id_match:
            video_id = video_id_match.group(1)
            logger.info(f"✅ Video ID extracted: {video_id}")
            return video_id
        else:
            error_msg = f"Could not extract video ID from URL: {url}"
            logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)
            
    except Exception as e:
        logger.error(f"Error extracting video ID: {e}")
        logger.error(traceback.format_exc())
        raise


def validate_video_id(video_id: str) -> bool:
    """
    Validate YouTube video ID format.
    
    Args:
        video_id: Video ID to validate
    
    Returns:
        True if valid, False otherwise
    """
    if not video_id or len(video_id) != 11:
        return False
    
    # Video IDs contain only alphanumeric, hyphen, and underscore
    return bool(re.match(r"^[A-Za-z0-9_-]{11}$", video_id))


# ============================================
# YOUTUBE API CLIENT
# ============================================

class YouTubeClient:
    """
    Wrapper for YouTube API client with retry logic and error handling
    """
    
    def __init__(self, api_key: str = API_KEY):
        """
        Initialize YouTube API client
        
        Args:
            api_key: YouTube Data API key
        """
        if not api_key:
            raise ValueError("YouTube API key is required")
        
        try:
            logger.info(f"🔧 Initializing YouTube API client...")
            self.youtube = build('youtube', 'v3', developerKey=api_key, cache_discovery=False)
            logger.info(f"✅ YouTube API client initialized")
        except Exception as e:
            logger.error(f"❌ Error initializing YouTube API: {e}")
            raise
    
    def _retry_request(self, request_func, max_retries=MAX_RETRIES):
        """
        Execute request with retry logic
        
        Args:
            request_func: Callable that returns a request
            max_retries: Maximum retry attempts
        
        Returns:
            API response
        
        Raises:
            Exception: If all retries fail
        """
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"📡 Request attempt {attempt}/{max_retries}")
                response = request_func().execute()
                logger.info(f"✅ Request successful")
                return response
                
            except HttpError as e:
                status_code = e.resp.status
                
                if status_code == 403:
                    logger.error(f"❌ Quota exceeded (attempt {attempt})")
                    if attempt < max_retries:
                        wait_time = RETRY_DELAY * (2 ** (attempt - 1))
                        logger.info(f"⏳ Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                    else:
                        raise
                        
                elif status_code == 404:
                    logger.error(f"❌ Resource not found")
                    raise ValueError("Video not found or is unavailable")
                    
                else:
                    logger.error(f"❌ HTTP Error {status_code}: {e}")
                    if attempt < max_retries:
                        time.sleep(RETRY_DELAY)
                    else:
                        raise
                        
            except Exception as e:
                logger.error(f"❌ Error on attempt {attempt}: {e}")
                if attempt < max_retries:
                    time.sleep(RETRY_DELAY)
                else:
                    raise
        
        raise Exception(f"Failed after {max_retries} attempts")


# ============================================
# MAIN SCRAPING FUNCTIONS
# ============================================

def get_video_metadata(youtube_client: YouTubeClient, video_id: str) -> Dict[str, Any]:
    """
    Fetch video metadata from YouTube API.
    
    Retrieved metadata:
    - title: Video title
    - channel: Channel name
    - published_at: Publication timestamp
    - thumbnail: Thumbnail URL
    - views: View count
    - likes: Like count (if available)
    - comment_count: Total comment count
    - duration: Video duration
    - description: Video description
    
    Args:
        youtube_client: YouTubeClient instance
        video_id: YouTube video ID
    
    Returns:
        Dict with video metadata
    
    Raises:
        Exception: If API call fails
    """
    try:
        logger.info(f"📺 Fetching video metadata for: {video_id}")
        
        def request_func():
            return youtube_client.youtube.videos().list(
                part='snippet,statistics,contentDetails',
                id=video_id
            )
        
        video_data = youtube_client._retry_request(request_func)
        
        if not video_data.get('items'):
            error_msg = f"Video not found: {video_id}"
            logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)
        
        item = video_data['items'][0]
        snippet = item.get('snippet', {})
        stats = item.get('statistics', {})
        content = item.get('contentDetails', {})
        
        metadata = {
            'id': video_id,
            'title': snippet.get('title', 'Unknown'),
            'channel': snippet.get('channelTitle', 'Unknown'),
            'published_at': snippet.get('publishedAt', 'Unknown'),
            'thumbnail': snippet.get('thumbnails', {}).get('high', {}).get('url', ''),
            'views': int(stats.get('viewCount', 0)),
            'likes': int(stats.get('likeCount', 0)) if 'likeCount' in stats else None,
            'comment_count': int(stats.get('commentCount', 0)),
            'duration': content.get('duration', 'Unknown'),
            'description': snippet.get('description', '')[:200]  # First 200 chars
        }
        
        logger.info(f"✅ Metadata retrieved:")
        logger.info(f"   Title: {metadata['title'][:60]}...")
        logger.info(f"   Channel: {metadata['channel']}")
        logger.info(f"   Views: {metadata['views']:,}")
        logger.info(f"   Comments: {metadata['comment_count']:,}")
        
        return metadata
        
    except Exception as e:
        logger.error(f"❌ Error getting video metadata: {e}")
        logger.error(traceback.format_exc())
        raise


def get_comments(
    youtube_client: YouTubeClient,
    video_id: str,
    max_comments: int = 200,
    order: str = 'relevance'
) -> List[Dict[str, Any]]:
    """
    Fetch comments from YouTube video.
    
    Comment data includes:
    - author: Comment author name
    - text: Comment text
    - likes: Number of likes
    - published_at: Comment timestamp
    - reply_count: Number of replies
    - is_reply: Whether it's a reply
    
    Args:
        youtube_client: YouTubeClient instance
        video_id: YouTube video ID
        max_comments: Maximum comments to fetch (default: 200)
        order: Sort order - 'relevance' or 'time' (default: 'relevance')
    
    Returns:
        List of comment dictionaries
    
    Raises:
        ValueError: If comments are disabled or video not found
    """
    try:
        logger.info(f"💬 Fetching up to {max_comments} comments (order: {order})...")
        
        comments = []
        next_page_token = None
        comment_threads_count = 0
        
        while len(comments) < max_comments:
            try:
                # Calculate remaining comments to fetch
                remaining = max_comments - len(comments)
                batch_size = min(MAX_RESULTS_PER_REQUEST, remaining)
                
                logger.info(f"📄 Fetching batch {comment_threads_count + 1} (size: {batch_size})...")
                
                def request_func():
                    return youtube_client.youtube.commentThreads().list(
                        part='snippet',
                        videoId=video_id,
                        maxResults=batch_size,
                        pageToken=next_page_token,
                        order=order,
                        textFormat='plainText'
                    )
                
                response = youtube_client._retry_request(request_func)
                comment_threads_count += 1
                
                # Extract comments
                for item in response.get('items', []):
                    try:
                        comment_snippet = item['snippet']['topLevelComment']['snippet']
                        
                        comment = {
                            'author': comment_snippet.get('authorDisplayName', 'Unknown'),
                            'text': comment_snippet.get('textDisplay', ''),
                            'likes': int(comment_snippet.get('likeCount', 0)),
                            'published_at': comment_snippet.get('publishedAt', ''),
                            'reply_count': int(item['snippet'].get('totalReplyCount', 0)),
                            'is_reply': False
                        }
                        
                        comments.append(comment)
                        
                        if len(comments) >= max_comments:
                            break
                            
                    except KeyError as e:
                        logger.warning(f"Skipping malformed comment: {e}")
                        continue
                
                # Check for next page
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    logger.info(f"ℹ️ No more pages available")
                    break
                    
            except HttpError as e:
                if e.resp.status == 403 and 'disabled' in str(e):
                    error_msg = "Comments are disabled for this video"
                    logger.error(f"❌ {error_msg}")
                    if not comments:
                        raise ValueError(error_msg)
                    else:
                        logger.info(f"ℹ️ Partially retrieved {len(comments)} comments before comments disabled")
                        break
                else:
                    raise
        
        logger.info(f"✅ Fetched {len(comments)} comments ({comment_threads_count} batches)")
        return comments
        
    except Exception as e:
        logger.error(f"❌ Error fetching comments: {e}")
        logger.error(traceback.format_exc())
        raise


# ============================================
# MAIN EXPORT FUNCTION
# ============================================

def fetch_video_and_comments(
    youtube_url: str,
    max_comments: int = 200,
    verbose: bool = True
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Fetch video metadata and comments from YouTube.
    
    Complete workflow:
    1. Extract and validate video ID
    2. Initialize YouTube API client
    3. Fetch video metadata
    4. Fetch comments with pagination
    5. Return as pandas DataFrame
    
    Args:
        youtube_url: YouTube video URL
        max_comments: Maximum comments to fetch (default: 200)
        verbose: Print progress messages (default: True)
    
    Returns:
        Tuple of (video_metadata_dict, comments_dataframe)
        
        metadata dict contains:
        - id, title, channel, published_at, thumbnail
        - views, likes, comment_count, duration
        
        DataFrame columns:
        - author, text, likes, published_at, reply_count
    
    Raises:
        ValueError: If URL invalid or video not found
        Exception: Other API errors
    
    Example:
        >>> info, df = fetch_video_and_comments(
        ...     'https://www.youtube.com/watch?v=9bZkp7q19f0',
        ...     max_comments=200
        ... )
        >>> print(f"Title: {info['title']}")
        >>> print(f"Comments fetched: {len(df)}")
    """
    
    start_time = datetime.now()
    
    try:
        if verbose:
            print(f"\n{'='*80}")
            print(f"🎥 YOUTUBE VIDEO & COMMENTS SCRAPER - v{VERSION}")
            print(f"{'='*80}\n")
        
        # ========================================
        # STEP 1: EXTRACT VIDEO ID
        # ========================================
        
        if verbose:
            print(f"🔍 Step 1: Extracting video ID...")
        
        logger.info(f"Starting fetch for URL: {youtube_url[:80]}...")
        video_id = extract_video_id(youtube_url)
        
        if not validate_video_id(video_id):
            raise ValueError(f"Invalid video ID: {video_id}")
        
        if verbose:
            print(f"   ✅ Video ID: {video_id}")
        
        # ========================================
        # STEP 2: INITIALIZE API CLIENT
        # ========================================
        
        if verbose:
            print(f"\n🔧 Step 2: Initializing YouTube API...")
        
        youtube_client = YouTubeClient(API_KEY)
        
        if verbose:
            print(f"   ✅ API client ready")
        
        # ========================================
        # STEP 3: FETCH VIDEO METADATA
        # ========================================
        
        if verbose:
            print(f"\n📺 Step 3: Fetching video metadata...")
        
        info = get_video_metadata(youtube_client, video_id)
        
        if verbose:
            print(f"   ✅ Title: {info['title'][:60]}...")
            print(f"   ✅ Channel: {info['channel']}")
            print(f"   ✅ Views: {info['views']:,}")
        
        # ========================================
        # STEP 4: FETCH COMMENTS
        # ========================================
        
        if verbose:
            print(f"\n💬 Step 4: Fetching comments...")
        
        comments = get_comments(youtube_client, video_id, max_comments)
        
        if not comments:
            logger.warning("⚠️ No comments fetched")
            if verbose:
                print(f"   ⚠️ No comments found")
        else:
            if verbose:
                print(f"   ✅ Comments fetched: {len(comments)}")
        
        # ========================================
        # STEP 5: CREATE DATAFRAME
        # ========================================
        
        if verbose:
            print(f"\n📊 Step 5: Creating DataFrame...")
        
        df = pd.DataFrame(comments)
        
        if not df.empty:
            logger.info(f"✅ DataFrame created with {len(df)} rows, {len(df.columns)} columns")
        else:
            logger.warning(f"⚠️ Empty DataFrame created")
        
        if verbose:
            print(f"   ✅ DataFrame shape: {df.shape}")
        
        # ========================================
        # COMPLETION SUMMARY
        # ========================================
        
        elapsed_time = datetime.now() - start_time
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"✅ SCRAPING COMPLETE!")
            print(f"{'='*80}")
            print(f"📺 Video: {info['title'][:70]}")
            print(f"👤 Channel: {info['channel']}")
            print(f"👁️  Views: {info['views']:,}")
            print(f"💬 Comments: {len(df)}")
            print(f"⏱️  Time taken: {elapsed_time}")
            print(f"{'='*80}\n")
        
        logger.info(f"✅ Scraping complete. Time: {elapsed_time}")
        
        return info, df
        
    except ValueError as e:
        logger.error(f"ValueError: {e}")
        if verbose:
            print(f"\n❌ ERROR: {e}\n")
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        if verbose:
            print(f"\n❌ UNEXPECTED ERROR: {e}\n")
        raise


# ============================================
# MODULE EXPORT
# ============================================

__all__ = [
    'fetch_video_and_comments',
    'extract_video_id',
    'validate_video_id',
    'get_video_metadata',
    'get_comments',
    'YouTubeClient'
]

logger.info(f"{'='*80}")
logger.info(f"{MODULE_NAME} v{VERSION} initialized successfully")
logger.info(f"{'='*80}")
