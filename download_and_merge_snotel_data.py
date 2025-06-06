def download_and_merge_snotel_data(id_list, state_list, start_date, end_date, output_dir):
    merged_csv_path = os.path.join(output_dir, f"merged_snotel_{end_date}.csv")
    
    # Skip if already downloaded
    if os.path.exists(merged_csv_path):
        print("Sensors already downloaded.")
        return pd.read_csv(merged_csv_path)
    
    print("Downloading SNOTEL data...")

    for ids, state in zip(id_list, state_list):
        if ids == 0:
            continue

        url = (
            f"https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/customMultiTimeSeriesGroupByStationReport/daily/"
            f"start_of_period/{ids}:{state}:SNTL%257Cid=%2522%2522%257Cname/"
            f"{start_date},{end_date}/stationId,name,WTEQ::value?fitToScreen=false"
        )

        try:
            response = requests.get(url)
            response.raise_for_status()
            
            # Filter lines
            lines = response.text.splitlines()
            filtered_lines = [line for line in lines if not line.lstrip().startswith('#')]
            filtered_csv = "\n".join(filtered_lines)
            
            df = pd.read_csv(io.StringIO(filtered_csv))

            # Find SWE column
            matching_cols = [col for col in df.columns if "Snow Water Equivalent" in col]
            if matching_cols:
                col = matching_cols[0]
                match = re.search(r'\(([^)]+)\)', col)
                new_col_name = match.group(1) if match else col
                df = df.rename(columns={col: new_col_name})

            # Save to temp CSV
            temp_csv_path = os.path.join(output_dir, f"snotel_{ids}_{state}_{end_date}.csv")
            df.to_csv(temp_csv_path, index=False)

        except Exception as ex:
            print(f"Error downloading {ids}, {state}: {ex}")
            continue

    # Merge downloaded CSVs
    csv_files = [f for f in os.listdir(output_dir) if f.startswith("snotel") and f.endswith(".csv")]
    merged_df = None

    for file in csv_files:
        file_path = os.path.join(output_dir, file)
        try:
            df = pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            print(f"Skipping empty file: {file}")
            continue

        df.columns = [col.strip() for col in df.columns]
        if 'Date' not in df.columns:
            print(f"Skipping file without 'Date' column: {file}")
            continue

        sensor_name = os.path.splitext(file)[0].split("_")[1]
        data_cols = [col for col in df.columns if col != 'Date']
        if not data_cols:
            print(f"No SWE column in file: {file}")
            continue

        df = df[['Date', data_cols[0]]].rename(columns={data_cols[0]: sensor_name})

        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on='Date', how='outer')

    # Save and clean up
    if merged_df is not None:
        merged_df.to_csv(merged_csv_path, index=False)
        print(f"Merged CSV saved to: {merged_csv_path}")
    else:
        print("No dataframes merged.")
        return pd.DataFrame()  # return empty df

    # Delete intermediate files
    for file in os.listdir(output_dir):
        if file.startswith("snotel") and not file.startswith("merged"):
            os.remove(os.path.join(output_dir, file))

    return merged_df
