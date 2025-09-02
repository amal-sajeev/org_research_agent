import uvicorn
import logging
import sys
import os
from datetime import datetime
from google.adk.cli.fast_api import get_fast_api_app

# Setup logging configuration
def setup_logging():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(script_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d")
    log_filename = os.path.join(logs_dir, f"adk_console_{timestamp}.log")
    
    # Configure logging format with timestamps
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Setup file logging
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)  # Also output to console
        ]
    )
    
    # Capture uvicorn logs
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.setLevel(logging.INFO)
    
    # Capture ADK logs (adjust logger name if needed)
    adk_logger = logging.getLogger("google.adk")
    adk_logger.setLevel(logging.INFO)
    
    print(f"Logging initialized. Console output will be saved to: {log_filename}")
    return log_filename

# Custom stream class to capture all stdout/stderr
class LoggingStream:
    def __init__(self, original_stream, logger, level):
        self.original_stream = original_stream
        self.logger = logger
        self.level = level
        
    def write(self, data):
        # Write to original stream (console)
        self.original_stream.write(data)
        self.original_stream.flush()
        
        # Also log to file if it's not just whitespace
        if data.strip():
            self.logger.log(self.level, data.strip())
    
    def flush(self):
        self.original_stream.flush()
    
    def isatty(self):
        # Return the original stream's isatty value for compatibility
        return getattr(self.original_stream, 'isatty', lambda: False)()
    
    def __getattr__(self, name):
        # Delegate any other attribute access to the original stream
        return getattr(self.original_stream, name)

def setup_stream_capture():
    # Setup logging
    logger = logging.getLogger("console_capture")
    
    # Store original streams
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Capture stdout and stderr
    sys.stdout = LoggingStream(original_stdout, logger, logging.INFO)
    sys.stderr = LoggingStream(original_stderr, logger, logging.ERROR)

# Initialize logging
log_file = setup_logging()
setup_stream_capture()

# Log startup information
logging.info("=== ADK Agent Starting ===")
logging.info(f"Script location: {os.path.abspath(__file__)}")
logging.info(f"Working directory: {os.getcwd()}")
logging.info(f"Log file: {log_file}")

try:
    # Create the ADK FastAPI application
    logging.info("Creating ADK FastAPI application...")
    app = get_fast_api_app(agents_dir=".", web=True)
    logging.info("ADK FastAPI application created successfully")
    
    if __name__ == "__main__":
        logging.info("Starting uvicorn server...")
        logging.info("Server configuration:")
        logging.info("  Host: 0.0.0.0")
        logging.info("  Port: 8501")
        logging.info("  SSL Key: SSLCerts/key.pem")
        logging.info("  SSL Cert: SSLCerts/cert.pem")
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8501,
            ssl_keyfile="SSLCerts/key.pem",
            ssl_certfile="SSLCerts/cert.pem",            log_level="info",  # Ensure uvicorn logs at info level
            access_log=True    # Enable access logging
        )

except Exception as e:
    logging.error(f"Error occurred: {str(e)}")
    logging.exception("Full exception details:")
    raise

finally:
    logging.info("=== ADK Agent Shutdown ===")