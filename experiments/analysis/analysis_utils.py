def extract_core_dataset_name(dataset_name):
    parts = dataset_name.split('__')
    return parts[0]

def dataframe_to_markdown(df):
    # Get the column headers
    headers = df.columns.tolist()
    
    # Create the header row
    header_row = '| | ' + ' | '.join(headers) + ' |'
    
    # Create the separator row
    separator_row = '| --- | ' + ' | '.join(['---'] * len(headers)) + ' |'
    
    # Create the data rows
    data_rows = []
    for index, row in df.iterrows():
        data_row = '| ' + str(index) + ' | ' + ' | '.join(str(cell).strip() for cell in row) + ' |'
        data_rows.append(data_row)
    
    # Combine all rows into a single markdown table string
    markdown_table = '\n'.join([header_row, separator_row] + data_rows)
    
    return markdown_table

# Extract noise ratio from dataset name
def extract_noise_ratio(dataset_name):
    parts = dataset_name.split('__')
    for part in parts:
        if 'noise' in part:
            return float(part.split('=')[1])
    return 0.01

def flatten_noise_level(df):
    # df.columns = pd.MultiIndex.from_tuples([(col[0],"low" if col[1]==0.01 else "high") for col in df.columns])
    columns = df.columns.values
    df.columns = [col[0] for col in columns]
    second_level_values = [col[1] for col in columns]
    df.loc['Noise Level'] = second_level_values
    df = df.reindex(['Noise Level'] + [col for col in df.index if col != 'Noise Level'])
    return df