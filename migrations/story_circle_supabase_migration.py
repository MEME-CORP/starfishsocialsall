import os
import sys
import json
import logging
from datetime import datetime

# Get the absolute path to the project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# Add the project root to the Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.database.supabase_client import DatabaseService
from src.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('migration')

# Phase order mapping
PHASE_ORDER = {
    'You': 1,
    'Need': 2,
    'Go': 3,
    'Search': 4,
    'Find': 5,
    'Take': 6,
    'Return': 7,
    'Change': 8
}

def clean_existing_data(db):
    """Clean existing data from all tables"""
    logger.info("Cleaning existing data...")
    
    # Delete all records in reverse dependency order
    tables = ['circle_memories', 'events_dialogues', 'story_phases', 'story_circle']
    
    for table in tables:
        try:
            # Delete all records with a WHERE clause that matches everything
            db.client.table(table).delete().neq('id', -1).execute()
            logger.info(f"Cleaned {table}")
        except Exception as e:
            logger.error(f"Error cleaning {table}: {e}")
    
    logger.info("Cleaned existing data")

def migrate_story_circle():
    try:
        logger.info("Running database migration...")
        db = DatabaseService()
        
        # Clean existing data
        clean_existing_data(db)
        
        # Create story circle
        logger.info("Creating story circle...")
        story = db.client.table('story_circle').insert({
            'is_current': True
        }).execute()
        
        story_circle_id = story.data[0]['id']
        logger.info(f"Created story circle with ID: {story_circle_id}")
        
        # Insert story phases
        logger.info("Inserting story phases...")
        phase_order = ["You", "Need", "Go", "Search", "Find", "Take", "Return", "Change"]
        for i, phase_name in enumerate(phase_order, 1):
            db.client.table('story_phases').insert({
                'story_circle_id': story_circle_id,
                'phase_name': phase_name,
                'phase_number': i,
                'phase_description': ''
            }).execute()
            logger.info(f"Inserted phase: {phase_name}")
        
        # Insert initial events and dialogues for first phase
        logger.info("Inserting events and dialogues...")
        initial_events = [
            {
                'story_circle_id': story_circle_id,
                'phase_number': 1,
                'event': 'Fwog wakes up to the sound of birds chirping and stretches its tiny limbs in the morning sunlight.',
                'inner_dialogue': 'Ahh, what a perfect start to the day! So peaceful and warm.'
            },
            {
                'story_circle_id': story_circle_id,
                'phase_number': 1,
                'event': 'Fwog splashes in the pond, chasing colorful ripples and giggling to itself.',
                'inner_dialogue': 'Heehee, these ripples are so fun to chase. I wonder if they\'re running away from me!'
            },
            {
                'story_circle_id': story_circle_id,
                'phase_number': 1,
                'event': 'Fwog sits on a lily pad, talking to dragonflies about the beauty of the day.',
                'inner_dialogue': 'Dragonflies are so elegant. Do they think the pond is as pretty as I do?'
            }
        ]
        
        for event_data in initial_events:
            db.client.table('events_dialogues').insert(event_data).execute()
            logger.info("Inserted event-dialogue pair")
        
        # Insert initial circle memory
        logger.info("Inserting circle memories...")
        db.client.table('circle_memories').insert({
            'story_circle_id': story_circle_id,
            'memory': 'Fwog began their journey in a peaceful pond, finding joy in the simple pleasures of nature.'
        }).execute()
        logger.info("Inserted circle memory")
        
        logger.info("Migration completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        logger.error(f"Full error details: {repr(e)}")
        return False

if __name__ == "__main__":
    migrate_story_circle() 