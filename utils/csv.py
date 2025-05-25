import csv

def write_stats_to_csv(stats, filename):
    """
    Writes stats to a CSV file, omitting the 'misclassified' field.
    Always includes: model, variant, accuracy, precision, recall, f1, training_time, roc_auc.
    Appends rows if file exists, using model/variant as primary key.
    """
    import os
    keys = ['model', 'variant', 'accuracy', 'precision', 'recall', 'f1', 'training_time', 'roc_auc']
    
    # Format numeric values with appropriate significant figures
    row = {}
    for k in keys:
        value = stats.get(k, '')
        if k == 'model' or k == 'variant':
            row[k] = value
        elif k == 'training_time':
            # Format training time to 4 decimal places
            row[k] = f"{value:.4f}" if isinstance(value, (int, float)) else value
        elif isinstance(value, (int, float)):
            # Format metrics to 4 decimal places for consistency
            row[k] = f"{value:.4f}" if value else value
        else:
            row[k] = value
    
    # Check if file exists and load existing data
    existing_data = {}
    if os.path.exists(filename):
        with open(filename, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for existing_row in reader:
                # Use model/variant as primary key
                key = (existing_row['model'], existing_row['variant'])
                existing_data[key] = existing_row
    
    # Update or add new row
    key = (row['model'], row['variant'])
    existing_data[key] = row
    
    # Write all data back to file
    with open(filename, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        for row_data in existing_data.values():
            writer.writerow(row_data)