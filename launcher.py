import os
import subprocess
import sys
import webbrowser
import time

def clear_screen():
    """Clear the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print the application header."""
    clear_screen()
    print("=" * 70)
    print("                      RAG System Launcher                         ")
    print("=" * 70)
    print("\n")

def launch_app(app_name, script_name):
    """Launch a Streamlit app and open it in the browser."""
    print(f"Starting {app_name}...")
    
    # Start the Streamlit process
    process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", script_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait a moment for the app to start
    time.sleep(3)
    
    # Open the browser
    webbrowser.open('http://localhost:8501')
    
    print(f"{app_name} is running at http://localhost:8501")
    print("Press Ctrl+C to stop the application and return to the launcher.")
    
    try:
        # Keep the app running until user interrupts
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # Terminate the process when user presses Ctrl+C
        process.terminate()
        print(f"\n{app_name} has been stopped.")

def main():
    """Main function to display menu and handle user input."""
    while True:
        print_header()
        print("Select an application to launch:")
        print("1. Document QA System (withstreamlit.py)")
        print("2. Website RAG System (rag_app.py)")
        print("3. Enhanced Website RAG System (enhanced_rag.py)")
        print("4. Exit")
        print("\n")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            launch_app("Document QA System", "withstreamlit.py")
        elif choice == '2':
            launch_app("Website RAG System", "rag_app.py")
        elif choice == '3':
            launch_app("Enhanced Website RAG System", "enhanced_rag.py")
        elif choice == '4':
            print("Exiting launcher. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")
            time.sleep(2)

if __name__ == "__main__":
    main()
