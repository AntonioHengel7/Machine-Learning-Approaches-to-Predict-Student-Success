from pandas import read_csv

def download_data(fileLocation):
    """
    Downloads the data for this script into a pandas DataFrame. Uses column indices provided.
    """
    frame = read_csv(
        fileLocation,
        sep=';',  # Specify the semicolon delimiter
        encoding='latin-1',  # Adjust encoding if needed
        skipinitialspace=True
    )
    return frame
