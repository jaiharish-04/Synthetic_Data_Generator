import pandas as pd
import json

def clean_record(record):
    """Ensure all fields have values — replace null/empty with 'NA'"""
    return {
        k: (str(v).strip() if pd.notna(v) and str(v).strip() else "NA")
        for k, v in record.items()
    }

def csv_to_json(input_csv, output_json):
    # Read CSV without converting empty strings automatically
    df = pd.read_csv(
        input_csv,
        keep_default_na=False,  # Prevents pandas from auto-setting NaN
    )
    
    # Apply 'NA' replacement logic to each record
    records = [clean_record(rec) for rec in df.to_dict(orient='records')]
    
    # Write the output
    with open(output_json, "w") as f:
        json.dump(records, f, indent=2)
    
    print(f"✅ Converted {len(records)} records to {output_json} with missing values as 'NA'.")

if __name__ == "__main__":
    input_csv = "/Users/jaiharishsatheshkumar/synthetic_data_generator/enriched_employee_dataset_50000.csv"
    output_json = "enriched_employee_dataset_50000.json"
    csv_to_json(input_csv, output_json)
