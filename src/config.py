csv_files = ["ALP", "BS", "DR", "EC", "HD", "JF", "JR", "SS2"]
file_suffix = ".csv"

BASE_DIR = "~/Research/wheelchair/data/raw/Max"
OUTPUT_DIR = "~/Research/wheelchair/data/processed/"


# configurations for glove types
HYB = "HYB"
PLA = "PLA"

"""
Get the base file

material: either 'HYB' or 'PLA'
"""
def format_file(material, initials):
    return f"{BASE_DIR}/{material}/{initials}25{material}.csv"