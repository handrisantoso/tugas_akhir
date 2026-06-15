import struct

def get_gz_uncompressed_size(gz_filepath):
    """
    Get the uncompressed size of a gzip file without decompressing it.
    
    Args:
        gz_filepath (str): Path to the gzip file
        
    Returns:
        int: Uncompressed size in bytes
    """
    with open(gz_filepath, 'rb') as f:
        # Seek to the last 4 bytes to read the ISIZE field
        f.seek(-4, 2)  # Seek 4 bytes from the end
        isize_bytes = f.read(4)
        
        # Unpack the 4 bytes as little-endian unsigned integer
        uncompressed_size = struct.unpack('<I', isize_bytes)[0]
        
        return uncompressed_size

# Example usage:
if __name__ == "__main__":
    # Replace 'your_file.gz' with the path to your gzip file
    gz_file = "your_file.gz"
    
    try:
        size = get_gz_uncompressed_size(gz_file)
        print(f"Uncompressed size: {size:,} bytes")
    except FileNotFoundError:
        print(f"File not found: {gz_file}")
    except Exception as e:
        print(f"Error: {e}") 