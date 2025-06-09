import pandas as pd
import json

def clean_record(record):
    """Remove empty/null fields and clean data types"""
    return {
        k: v for k, v in record.items() 
        if pd.notna(v) and v not in ["", " ", None]
    }

def csv_to_json(input_csv, output_json):
    # Read CSV with proper null handling
    df = pd.read_csv(
        input_csv,
        keep_default_na=False,  # Don't auto-convert to pandas NA
        na_values=["", " ", "NA", "N/A", "null", "None"]  # Custom NA values
    )
    
    # Convert to cleaned records
    records = [
        clean_record(record) 
        for record in df.to_dict(orient='records')
    ]
    
    # Write to JSON
    with open(output_json, "w") as f:
        json.dump(records, f, indent=2)
    
    print(f"✅ Converted {len(records)} clean records from {input_csv} to {output_json}")

if __name__ == "__main__":
    input_csv = "/Users/jaiharishsatheshkumar/synthetic_data_generator/.venv/bin/enriched_employee_dataset_50000.csv"
    output_json = "enriched_employee_dataset_50000.json"
    csv_to_json(input_csv, output_json)