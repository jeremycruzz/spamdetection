import pandas as pd

def create_latex_table(filepath="./data/stats.csv"):
    # Load CSV into a DataFrame
    df = pd.read_csv(filepath)

    # Start the LaTeX table
    latex_table = "\\begin{table}[H]\n"
    latex_table += "    \\centering\n"
    latex_table += "    \\begin{tabular}{llrrrrrr}\n"
    latex_table += "    \\hline\n"
    
    # Add header row
    headers = df.columns.tolist()
    latex_table += f"     {headers[0]}    & {headers[1]}    &   {headers[2]} &   {headers[3]} &   {headers[4]} &     {headers[5]} &   {headers[6]} &   {headers[7]} \\\\\n"
    
    # Add hline
    latex_table += "    \\hline\n"
    
    # Add data rows
    for _, row in df.iterrows():
        latex_table += f"     {row['model']} & {row['variant']}        &     {row['accuracy']:.4f} &      {row['precision']:.4f} &   {row['recall']:.4f} & {row['f1']:.4f} &          {row['training_time']:.4f} &    {row['roc_auc']:.4f} \\\\\n"
    
    # Close the table
    latex_table += "    \\hline\n"
    latex_table += "    \\end{tabular}\n"
    latex_table += "    \\caption{Experiment Results Summary}\n"
    latex_table += "    \\label{tab:results}\n"
    latex_table += "\\end{table}\n"
    
    return latex_table

# Example usage:
print(create_latex_table())
