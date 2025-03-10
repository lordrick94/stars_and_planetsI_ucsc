import pandas as pd

def csv_to_latex(csv_file, output_file, sample_rows=None):
    # Read CSV file
    df = pd.read_csv(csv_file)

    if sample_rows: 
        df = df.sample(sample_rows)

    # Convert DataFrame to LaTeX table
    latex_table = df.to_latex(index=False, escape=False)

    # Write to a .tex file
    with open(output_file, "w") as f:
        f.write(latex_table)

    print(f"LaTeX file '{output_file}' generated successfully!")

if __name__ == "__main__":
    input_file = 'top_candidate.csv'
    output_file = 'test_table.tex'

    csv_to_latex(input_file, output_file)
