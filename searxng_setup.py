import subprocess
import sys

def start_searxng():
    try:
        # Check if Docker is running
        subprocess.run(["docker", "info"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Docker is running...")

        # Start SearxNG container
        subprocess.run([
            "docker", "run", "-d", "--name", "searxng", "-p", "8080:8080", 
            "searxng/searxng"
        ], check=True)

        print("SearxNG is now running at http://localhost:8080")
        
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_searxng()
