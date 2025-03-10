import streamlit as st
import logging
import os
from components.dashboard import setup_dashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """
    Main entry point for the Stock Sentiment Analysis Dashboard.
    Sets up the dashboard and handles any global exceptions.
    """
    try:
        # Ensure required directories exist
        os.makedirs(".streamlit", exist_ok=True)
        
        # Set up the dashboard
        setup_dashboard()
        
    except Exception as e:
        logging.error(f"Application error: {e}", exc_info=True)
        st.error(f"An unexpected error occurred: {str(e)}")
        st.error("Please refresh the page and try again.")

if __name__ == "__main__":
    main()
