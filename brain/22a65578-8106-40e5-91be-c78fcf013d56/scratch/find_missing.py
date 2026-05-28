# find_missing.py
import sys

def main():
    try:
        with open('index.html.orig', 'r', encoding='utf-16') as f:
            orig = f.read()
    except Exception as e:
        print("Failed to read index.html.orig:", e)
        return

    # Find the starting point of truncation
    start_target = 'class="report-details-metrics"'
    start_idx = orig.find(start_target)
    if start_idx == -1:
        print("Could not find start target in orig")
        return

    # Find the ending point (beginning of Settings card 1)
    end_target = '<!-- Card 1: General Preferences -->'
    end_idx = orig.find(end_target)
    if end_idx == -1:
        print("Could not find end target in orig")
        return

    segment = orig[start_idx:end_idx]
    
    with open(r'C:\Users\VEDAN\Silent-Outbreak-Predictor\brain\22a65578-8106-40e5-91be-c78fcf013d56\scratch\missing_output.txt', 'w', encoding='utf-8') as f_out:
        f_out.write(segment)
        
    print(f"Successfully wrote segment of size {len(segment)} to missing_output.txt")

if __name__ == '__main__':
    main()
